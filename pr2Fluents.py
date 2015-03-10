import util
from util import nearAngle
import numpy as np
from pr2Util import PoseD, defaultPoseD, NextColor, shadowName, Violations, drawPath
from dist import DeltaDist
from planGlobals import debugMsg, debugDraw, debug, pause
import planGlobals as glob
from miscUtil import isGround, isVar, prettyString, applyBindings
import fbch
from fbch import Fluent, getMatchingFluents, Operator
from belief import B, Bd
from pr2Visible import visible

################################################################
## Fluent definitions
################################################################

class Graspable(Fluent):
    predicate = 'Graspable'
    immutable = True
    def test(self, details):
        (objName,) = self.args
        return objName[0:3] == 'obj'

# Assumption is that the robot is holding at most one object, but it
# could be in either hand.

class In(Fluent):
    predicate = 'In'
    implicit = True
    def bTest(self, bState, v, p):
        (obj, region) = self.args
        assert v == True
        return inTest(bState, obj, region, p)

def confWithin(c1, c2, delta, robot):
    def withinDelta(a, b):
        if isinstance(a, list):
            d = delta[0]                # !! hack
            return all([abs(a[i] - b[i]) <= d \
                        for i in range(min(len(a), len(b)))])
        else:
            return a.withinDelta(b, delta)

    if not all([d >= 0 for d in delta]):
        return False

    # We only care whether | conf - targetConf | <= delta
    # Check that the moving frames are all within specified delta
    c1CartConf = robot.forwardKin(c1)
    c2CartConf = robot.forwardKin(c2)

    # Also be sure two head angles are the same
    (c1h1, c1h2) = c1['pr2Head']
    (c2h1, c2h2) = c2['pr2Head']

    return all([withinDelta(c1CartConf[x],c2CartConf[x]) \
                for x in robot.moveChainNames]) and \
                nearAngle(c1h1, c2h1, delta[-1]) and \
                nearAngle(c1h2, c2h2, delta[-1])

class Conf(Fluent):
    # This is will not be wrapped in a belief fluent.
    # Args: conf, delta
    predicate = 'Conf'
    def test(self, bState):
        (targetConf, delta) = self.args
        robot = bState.pbs.beliefContext.world.robot
        return confWithin(bState.pbs.conf, targetConf, delta, robot)

    def getGrounding(self, details):
        assert self.value == True
        (targetConf, delta) = self.args
        assert not isGround(targetConf)
        return {targetConf : details.pbs.conf}

    def fglb(self, other, details = None):
        if other.predicate != 'Conf':
            return {self, other}, {}

        (sval, sdelta) = self.args
        (oval, odelta) = other.args

        b = {}
        if isVar(sval):
            b[sval] = oval
        elif isVar(oval):
            b[oval] = sval

        if isVar(sdelta):
            b[sdelta] = odelta
        elif isVar(odelta):
            b[odelta] = sdelta

        (bsval, boval, bsdelta, bodelta) = \
           applyBindings((sval, oval, sdelta, odelta), b)

        if isGround((bsval, boval, bsdelta, bodelta)):
            robot = details.pbs.beliefContext.world.robot
            bigDelta = tuple([od + sd for (od, sd) in zip(odelta, sdelta)])
            smallDelta = (1e-4,)*4

            # Kind of hard to average the confs.  Just be cheap for now.

            # If intervals don't overlap, then definitely false.
            if not confWithin(bsval, boval, bigDelta, robot):
                return False, {}

            # One contains the other
            if confWithin(bsval, boval, smallDelta, robot):
                if all([s < o  for (s, o) in zip(bsdelta,bodelta)]):
                    return self, b
                else:
                    return other, b

            print 'Should really compute the intersection of these intervals.  Too lazy.  Returning False.'
            return False, {}
        else:
            return self, b

# Check reachability to "home"
class CanReachHome(Fluent):
    predicate = 'CanReachHome'
    implicit = True
    conditional = True

    def getViols(self, bState, v, p, strict = True):
        assert v == True
        (conf, hand,
               lobj, lface, lgraspMu, lgraspVar, lgraspDelta,
               robj, rface, rgraspMu, rgraspVar, rgraspDelta, cond) = \
                            self.args   # Args
        # Note that all object poses are permanent, no collisions can be ignored
        newPBS = bState.pbs.copy()
        if strict:
            newPBS.updateFromAllPoses(cond, updateHeld=False)
        else:
            newPBS.updateFromGoalPoses(cond, updateHeld=False)

        if hand == 'right':
            # Swap arguments!
            (lobj, lface, lgraspMu, lgraspVar, lgraspDelta, \
             robj, rface, rgraspMu, rgraspVar, rgraspDelta) =  \
            (robj, rface, rgraspMu, rgraspVar, rgraspDelta, \
             lobj, lface, lgraspMu, lgraspVar, lgraspDelta)
        
        lgraspD = PoseD(util.Pose(*lgraspMu), lgraspVar) \
                          if lobj != 'none' else None
        rgraspD = PoseD(util.Pose(*rgraspMu), rgraspVar) \
                          if robj != 'none' else None
        
        newPBS.updateHeld(lobj, lface, lgraspD, 'left', lgraspDelta)
        newPBS.updateHeld(robj, rface, rgraspD, 'right', rgraspDelta)
        
        path, violations = canReachHome(newPBS, conf, p, Violations(),
                                        draw=False)
        debugMsg('CanReachHome',
                 ('conf', conf),
                 ('lobjGrasp', lobj, lface, lgraspD),
                 ('robjGrasp', robj, rface, rgraspD),
                 ('->', violations))
        return path, violations

    def bTest(self, bState, v, p):
        path, violations = self.getViols(bState, v, p)
        return bool(path and violations.empty())

    def heuristicVal(self, bState, v, p):
        # Return cost estimate and a set of dummy operations
        obstCost = 40
        shadowCost = 20
        path, violations = self.getViols(bState.details, v, p, strict = False)
        if path == None:
            #!! should this happen?
            print 'hv infinite'
            raw_input('go?')
            return float('inf'), {}
        obstacles = violations.obstacles
        shadows = violations.shadows
        obstOps = [Operator('RemoveObst', [o.name()],{},[]) for o in obstacles]
        for o in obstOps: o.instanceCost = obstCost
        shadowOps = [Operator('RemoveShadow', [o.name()],{},[]) \
                     for o in shadows]
        for o in shadowOps: o.instanceCost = shadowCost
        ops = set(obstOps + shadowOps)
        return (obstCost * len(obstacles) + shadowCost * len(shadows), ops)

    def prettyString(self, eq = True, includeValue = True):
        (conf, hand, lobj, lface, lgraspMu, lgraspVar, lgraspDelta,
               robj, rface, rgraspMu, rgraspVar, rgraspDelta, cond) = \
                            self.args   # Args
        if hand == 'right':
            (lobj, lface, lgraspMu, lgraspVar, lgraspDelta, \
             robj, rface, rgraspMu, rgraspVar, rgraspDelta) =  \
            (robj, rface, rgraspMu, rgraspVar, rgraspDelta, \
             lobj, lface, lgraspMu, lgraspVar, lgraspDelta)

        argStr = prettyString(self.args) if eq else \
                  prettyString([conf, lobj, robj, 'Conds'], eq)
        valueStr = ' = ' + prettyString(self.value) if includeValue else ''
        return self.predicate + ' ' + argStr + valueStr

zeroPose = zeroVar = (0.0,)*4
tinyDelta = (1e-8,)*4
awayPose = (100.0, 100.0, 0.0, 0.0)

# Check all three reachability conditions together.  For now, try to
# piggy-back on existing code.  Can probably optimize later.
class CanPickPlace(Fluent):
    predicate = 'CanPickPlace'
    implicit = True
    conditional = True

    def getConds(self, details = None):
        # Will recompute if new bindings are applied because the result
        # won't have this attribute
        (preConf, ppConf, hand, obj, pose, poseVar, poseDelta, poseFace,
          graspFace, graspMu, graspVar, graspDelta,
         oObj, oFace, oGraspMu, oGraspVar, oGraspDelta, inconds) = self.args
        if not hasattr(self, 'conds'):
            objInPlace = B([Pose([obj, poseFace]), pose, poseVar, poseDelta,
                            1.0], True)
            objInPlaceZeroVar = B([Pose([obj, poseFace]), pose, zeroVar,
                                   tinyDelta,1.0], True)
            self.conds = \
          [# 1.  Home to approach, holding nothing, obj in place
              CanReachHome([preConf, hand,
                        'none', 0, zeroPose, zeroVar, tinyDelta,
                        oObj, oFace, oGraspMu, oGraspVar, oGraspDelta,
                      [objInPlace]]),
              # 2.  Home to approach with object in hand
              CanReachHome([preConf, hand, obj,
                                graspFace, graspMu, graspVar, graspDelta,
                                oObj, oFace, oGraspMu, oGraspVar, oGraspDelta,
                                []]),
              # 3.  Home to pick with hand empty, obj in place with zero var
              CanReachHome([ppConf, hand, 
                               'none', 0, zeroPose, zeroVar, tinyDelta,
                                oObj, oFace, oGraspMu, oGraspVar, oGraspDelta,
                       [objInPlaceZeroVar]]),
             # 4. Home to pick with the object in hand with zero var and delta
              CanReachHome([ppConf, hand,
                                obj, graspFace, graspMu, zeroVar, tinyDelta,
                                oObj, oFace, oGraspMu, oGraspVar, oGraspDelta,
                        []])]
        for c in self.conds: c.addConditions(inconds, details)
        return self.conds

    def getViols(self, bState, v, p, strict = True):
        def violCombo(v1, v2):
            return Violations(v1.obstacles.union(v2.obstacles),
                              v1.shadows.union(v2.shadows))

        condViols = [c.getViols(bState, v, p, strict) \
                     for c in self.getConds(bState)]
        pathNone = any([p == None for (p, v) in condViols])
        if pathNone:
            return (None, None)
        allViols = [v for (p, v) in condViols]
        violations = reduce(violCombo, allViols)
        return True, violations

    def bTest(self, bState, v, p):
        path, violations = self.getViols(bState, v, p, strict = True)
        return bool(path and violations.empty())

    def heuristicVal(self, bState, v, p):
        # Return cost estimate and a set of dummy operations
        obstCost = 40
        shadowCost = 20
        path, violations = self.getViols(bState.details, v, p, strict = False)
        if path == None:
            #!! should this happen?
            print 'hv infinite'
            glob.debugOn.append('confViolations')
            bState.details.pbs.getRoadMap().confReachCache = {}
            self.getViols(bState.details, v, p, strict = False)
            glob.debugOn.pop(-1)
            raw_input('should this happen?')
            return float('inf'), {}
        obstacles = violations.obstacles
        shadows = violations.shadows
        obstOps = [Operator('RemoveObst', [o.name()],{},[]) for o in obstacles]
        for o in obstOps: o.instanceCost = obstCost
        shadowOps = [Operator('RemoveShadow', [o.name()],{},[]) \
                     for o in shadows]
        for o in shadowOps: o.instanceCost = shadowCost
        ops = set(obstOps + shadowOps)
        return (obstCost * len(obstacles) + shadowCost * len(shadows), ops)

    def prettyString(self, eq = True, includeValue = True):
        (preConf, ppConf, hand, lobj, pose, poseVar, poseDelta, poseFace,
          lface, lgraspMu, lgraspVar, lgraspDelta,
          robj, rface, rgraspMu, rgraspVar, rgraspDelta, conds) = self.args

        if hand == 'right':
            (lobj, lface, lgraspMu, lgraspVar, lgraspDelta, \
             robj, rface, rgraspMu, rgraspVar, rgraspDelta) =  \
            (robj, rface, rgraspMu, rgraspVar, rgraspDelta, \
             lobj, lface, lgraspMu, lgraspVar, lgraspDelta)

        argStr = prettyString(self.args) if eq else \
                  prettyString([lobj, hand, pose, 'Conds'], eq)
        valueStr = ' = ' + prettyString(self.value) if includeValue else ''
        return self.predicate + ' ' + argStr + valueStr

        
class Holding(Fluent):
    predicate = 'Holding'
    def dist(self, bState):
        (hand,) = self.args          # Args
        return bState.pbs.getHeld(hand) # a DDist instance (over object names)

    # This is only called if the regular matching stuff doesn't make
    # the answer clear.
    def fglb(self, other, bState = None):
        (hand,) = self.args          # Args
        # Inconsistent: This object is placed somewhere
        if other.predicate == 'Pose' and  other.args[0] == self.value:
            return False, {}
        # Inconsistent:  Can't have same object in both hands!
        if other.predicate == 'Holding' and other.args[0] != hand and \
          self.value == other.value and self.value != 'none':
            return False, {}
        # Holding = none entails all other holding-related fluents
        # for none and the same hand
        if other.predicate in ('Grasp', 'GraspFace') and \
           self.value == 'none' and \
           other.args[0] == 'none' and other.args[1] == hand:
            return self, {}
        # Inconsistent: Holding X is inconsistent with all the other
        # grasp-related fluents for same hand, different objects
        if other.predicate in ('Grasp', 'GraspFace') and \
          other.args[1] == hand and other.args[0] != self.value:
          return False, {}

        # Nothing special
        return {self, other}, {}

class GraspFace(Fluent):
    predicate = 'GraspFace'
    def dist(self, bState):
        (obj, hand) = self.args                  # Args
        if obj == 'none':
            return DeltaDist(0)
        else:
            return bState.pbs.getGraspB(obj, hand).grasp # a DDist over integers

    def fglb(self, other, bState = None):
        (obj, hand) = self.args                  # Args
        if obj == 'none':
            if other.predicate == 'Holding' and \
               other.value == 'none' and other.args[0] == hand:
                return other, {}
        if other.predicate == 'Holding' and \
              other.args[0] == hand and other.value != obj:
                return False, {}
        if other.predicate in ('SupportFace', 'Pose') and \
              other.args[0] == obj:
                return False, {}
        return {self, other}, {}


# Grasp of an object;  conditional distribution
class Grasp(Fluent):
    predicate = 'Grasp'
    def dist(self, bState):
        (obj, hand, face) = self.args
        return bState.graspModeDist(obj, hand, face)

    def fglb(self, other, bState = None):
        (obj, hand, face) = self.args
        if obj == 'none':
            if other.predicate == 'Holding' and \
               other.args[0] == hand:
                return other, {}
        if other.predicate == 'Holding' and \
              other.args[0] == hand and other.value != obj:
                return False, {}
        if other.predicate in ('SupportFace', 'Pose') and \
              other.args[0] == obj:
                return False, {}
        return {self, other}, {}

class Pose(Fluent):
    predicate = 'Pose'
    def dist(self, bState):
        (obj, face) = self.args              
        result = bState.poseModeDist(obj, face)
        return result

    def fglb(self, other, bState = None):
        if (other.predicate == 'Holding' and \
            self.args[0] == other.value) or \
           (other.predicate in ('Grasp', 'GraspFace') and \
                 self.args[0] == other.args[0]):
           return False, {}
        else:
           return {self, other}, {}

class SupportFace(Fluent):
    predicate = 'SupportFace'
    def dist(self, bState):
        (obj,) = self.args               # args
        # Mode should be 'none' if the object is in the hand
        return bState.pbs.getPlaceB(obj).support # a DDist (over integers)

    def fglb(self, other, bState = None):
        if (other.predicate == 'Holding' and \
                self.args[0] == other.value) or \
                (other.predicate in ('Grasp', 'GraspFace') and \
                 self.args[0] == other.args[0]):
            return False, {}
        else:
            return {self, other}, {}

class CanSeeFrom(Fluent):
    # Args (Obj, Pose, Conf, Conditions)
    predicate = 'CanSeeFrom'
    implicit = True
    conditional = True
    def bTest(self, details, v, p):
        assert v == True
        (obj, pose, poseFace, conf, cond) = self.args
        # Note that all object poses are permanent, no collisions can be ignored
        newPBS = details.pbs.copy()
        newPBS.updateFromAllPoses(cond)
        placeB = newPBS.getPlaceB(obj)
        if placeB.support.mode() != poseFace:
            placeB.support = DeltaDist(poseFace)
        if placeB.poseD.mode() != pose:
            newPBS.updatePermObjPose(placeB.modifyPoseD(mu=pose))
        shWorld = newPBS.getShadowWorld(p)
        shName = shadowName(obj)
        sh = shWorld.objectShapes[shName]
        obstacles = [s for s in shWorld.getNonShadowShapes() if s.name() != obj ]
        ans, _ = visible(shWorld, conf, sh, obstacles, p)
        return ans

    def prettyString(self, eq = True, includeValue = True):
        (obj, pose, poseFace, conf, cond) = self.args
        argStr = prettyString(self.args) if eq else \
                  prettyString([conf, obj, 'Conds'], eq)
        valueStr = ' = ' + prettyString(self.value) if includeValue else ''
        return self.predicate + ' ' + argStr + valueStr

    

# Given a set of fluents, partition them into groups that are achieved
# together, for the heuristic.  Should be able to do this by automatic
# analysis, but by hadn for now.

# Groups:
# SupportFace, Pose, In  (same obj)
# Holding, GraspFace, Grasp (same hand)
# CanReachHome (all separate)
# Conf

# Can make this nicer by specifying sets of predicates that have to go
# together, somehow.

def partition(fluents):
    groups = []
    fluents = set(fluents)
    while len(fluents) > 0:
        f = fluents.pop()
        newSet = set([f])
        if f.predicate in ('B', 'Bd'):
            rf = f.args[0]
            if rf.predicate == 'SupportFace':
                (obj,) = rf.args
                face = f.args[1]
                pf = getMatchingFluents(fluents,
                             B([Pose([obj, face]), 'M','V','D','P'], True)) + \
                      getMatchingFluents(fluents,
                                 Bd([In([obj, 'R']), True,'P'], True))
                for (ff, b) in pf:
                    newSet.add(ff)
                    fluents.remove(ff)
                
            elif rf.predicate == 'Pose':
                (obj, face) = rf.args
                pf = getMatchingFluents(fluents,
                              Bd([SupportFace([obj]), face,'P'], True)) + \
                      getMatchingFluents(fluents,
                                 Bd([In([obj, 'R']), True,'P'], True))
                for (ff, b) in pf:
                    newSet.add(ff)
                    fluents.remove(ff)

            elif rf.predicate == 'In':
                (obj, region) = rf.args
                pf = getMatchingFluents(fluents,
                              Bd([SupportFace([obj]), 'F','P'], True)) + \
                     getMatchingFluents(fluents,
                             B([Pose([obj, 'F']), 'M','V','D','P'], True))
                for (ff, b) in pf:
                    newSet.add(ff)
                    fluents.remove(ff)
                
            elif rf.predicate == 'Holding':
                # Find GraspFace and Grasp
                (hand,) = rf.args
                obj = f.args[1]
                pf = getMatchingFluents(fluents,
                              Bd([GraspFace([obj, hand]), 'F','P'], True)) +\
                     getMatchingFluents(fluents,
                           B([Grasp([obj, hand, 'F']),'M','V','D','P'], True))
                for (ff, b) in pf:
                    newSet.add(ff)
                    fluents.remove(ff)
                
            elif rf.predicate == 'GraspFace':
                # Find Holding and Grasp
                (obj, hand) = rf.args
                face = f.value
                pf = getMatchingFluents(fluents,
                              Bd([Holding([hand]), obj,'P'], True)) + \
                     getMatchingFluents(fluents,
                           B([Grasp([obj, hand, 'F']),'M','V','D','P'], True))
                for (ff, b) in pf:
                    newSet.add(ff)
                    fluents.remove(ff)

            elif rf.predicate == 'Grasp':
                # Find holding and Graspface
                (obj, hand, face) = rf.args
                pf = getMatchingFluents(fluents,
                              Bd([Holding([hand]), obj,'P'], True)) + \
                     getMatchingFluents(fluents,
                           Bd([GraspFace([obj, hand]), face, 'P'], True))
                for (ff, b) in pf:
                    newSet.add(ff)
                    fluents.remove(ff)
                    
        groups.append(frozenset(newSet))
    return groups

###
# Tests
###

def canReachHome(bState, conf, prob, initViol,
                 avoidShadow = [], startConf = None, draw=True):
    rm = bState.getRoadMap()
    robot = bState.getRobot()
    initConf = startConf or rm.homeConf
    # Reverse start and target
    viol, cost, pathRev = rm.confReachViol(initConf, bState, prob,
                                           initViol,
                                           avoidShadow=avoidShadow,
                                           startConf=conf)
    path = pathRev[::-1] if pathRev else pathRev

    if debug('checkCRH') and fbch.inHeuristic:
        bState.draw(prob, 'W')
        fbch.inHeuristic = False
        bState.shadowWorld = None
        bState.draw(prob, 'W', clear=False) # overlay
        conf.draw('W', 'blue')
        initConf.draw('W', 'pink')
        viol2, cost2, pathRev2 = rm.confReachViol(initConf, bState, prob,
                                                  initViol,
                                                  avoidShadow=avoidShadow,
                                                  startConf=conf)
        fbch.inHeuristic = True
        # Check for heuristic (viol) being worse than actual (viol2)
        if viol != viol2 and viol2 != None \
               and ((viol == None and viol2 != None) \
                    or (viol.weight() > viol2.weight())):
            print 'viol with full model', viol2
            print 'viol with min  model', viol
            if viol2:
                [o.draw('W', 'red') for o in viol2.obstacles]
                [o.draw('W', 'orange') for o in viol2.shadows]
            if viol:
                [o.draw('W', 'purple') for o in viol.obstacles]
                [o.draw('W', 'pink') for o in viol.shadows]
            raw_input('Continue?')

    if debug('traceCRH'):
        print '    canReachHome h=', fbch.inHeuristic, 'viol=:', viol.weight() if viol else None
    if not path:
        if fbch.inHeuristic and debug('extraTests'):
            bState.draw(prob, 'W')
            print 'canReachHome failed with inHeuristic=True'
            fbch.inHeuristic = False
            viol, cost, path = rm.confReachViol(conf, bState, prob,
                                                initViol,
                                                avoidShadow=avoidShadow,
                                                startConf=startConf)
            if path:
                raw_input('Inconsistency')
            else:
                print 'Consistent result with inHeuristic=False'
            fbch.inHeuristic = True

    if (not fbch.inHeuristic) or debug('drawInHeuristic'):
        if debug('canReachHome'):
            bState.draw(prob, 'W')
            if path:
                drawPath(path, viol=viol,
                         attached=bState.getShadowWorld(prob).attached)
            else:
                print 'viol, cost, path', viol, cost, path
        debugMsg('canReachHome', ('viol', viol))

    return path, viol

# !! Should this specify the support surface??
def inTest(bState, obj, regName, prob, pB=None):
    # compute a shadow for this object
    placeB = pB or bState.pbs.getPlaceB(obj)
    faceFrame = placeB.faceFrames[placeB.support.mode()]

    # !! Clean this up
    
    sh = bState.pbs.objShadow(obj, True, prob, placeB, faceFrame)
    shadow = sh.applyLoc(placeB.objFrame()) # !! is this right?
    shWorld = bState.pbs.getShadowWorld(prob)
    region = shWorld.regionShapes[regName]
    
    ans = np.all(region.containsPts(shadow.vertices().T))

    if debug('testVerbose'):
        shadow.draw('W', 'brown')
        region.draw('W', 'purple')
        print 'shadow in brown, region in purple'
        print 'inTest', obj, '->', ans
        raw_input('Ok?')

    return ans

                
'''                    




class In(Fluent):
    predicate = 'In'
    implicit = True
    def bTest(self, bState, v, p):
        (obj, region) = self.args         # Args
        assert v == True
        result = bool(inTest(bState, obj, region, p))
        debugMsg('In', ('obj', obj, 'region', region), ('->', result))
        return result

# Checks visibility, but not reachability
class CanSeeFrom(Fluent):
    predicate = 'CanSeeFrom'
    implicit = True
    conditional = True
    def bTest(self, bState, v, p):
        assert v == True
        # !! Check args
        # !! sup = support face (is this a DDist or a value)?
        (conf, objPlace, cond) = self.args   # Args
        newBS = bState.copy()
        newBS.updateFromAllPoses(cond)
        debugMsg('CanSeeFrom',
                 ('objPlace', objPlace),
                 ('conf', conf),
                 ('->', violations))
        return bool(canSeeFromConf(newBS, conf, obj, sup, poseD)) # !! DEFINE


# CanPick and CanPlace test legality, but not the motions
# Need to be able to reach from preConf to pickConf with a single Pick
# !! Should also test collision with permanent obstacles
# !! graspD describes target grasp and allowed variation?? 
class CanPick(Fluent):
    predicate = 'CanPick'
    immutable = True
    def bTest(self, bState, v, p):
        assert v == True
        # !! Check args
        (preConf, pickConf, hand, objGrasp, objPlace, delPose) = self.args
        # Check that pickConf can grasp obj
        result1 = legalGrasp(bState, pickConf, hand, objGrasp, objPlace, delPose)
        # Check pre-image for grasping
        result0 = inPickApproach(bState, preConf, pickConf, hand, objGrasp, objPlace) \
          if result1 else False
        debugMsg('CanPick',
                 ('objGrasp', objGrasp), 
                 ('objPlace', objPlace),
                 ('preConf', preConf, 'pickConf', pickConf),
                 ('hand', hand, 'delPose', delPose),
                 ('->', result1, result0))
        return result1 and result0

class CanPlace(Fluent):
    predicate = 'CanPlace'
    immutable = True
    def bTest(self, bState, v, p):
        assert v == True
        # !! Check args
        (placeConf, postConf, hand, objGrasp, objPlace, delPose) = self.args
        # Check that placeConf can grasp obj
        result1 = legalGrasp(bState, placeConf, hand, objGrasp, objPlace, delPose)
        result0 = inPlaceDeproach(bState, placeConf, postConf, hand, objGrasp, objPlace) \
          if result1 else False
        debugMsg('CanPlace',
                 ('objGrasp', objGrasp),
                 ('objPlace', objPlace), 
                 ('placeConf', placeConf, 'postConf', postConf),
                 ('hand', hand, 'delPose', delPose),
                 ('->', result1, result0))
        return result1 and result0

'''        
