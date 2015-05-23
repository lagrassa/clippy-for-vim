import util
from util import nearAngle
import numpy as np
from pr2Util import PoseD, defaultPoseD, NextColor, shadowName, Violations, drawPath, objectName
from dist import DeltaDist, probModeMoved
from planGlobals import debugMsg, debugDraw, debug, pause
import planGlobals as glob
from miscUtil import isGround, isVar, prettyString, applyBindings
import fbch
from fbch import Fluent, getMatchingFluents, Operator
from belief import B, Bd
from pr2Visible import visible
from pr2BeliefState import lostDist

tiny = 1.0e-6

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
        lObj =  bState.pbs.getHeld('left').mode()
        rObj =  bState.pbs.getHeld('right').mode()
        if obj in (lObj, rObj):
            return False
        else: 
            return inTest(bState, obj, region, p)

    
def baseConfWithin(bc1, bc2, delta):
    (x1, y1, t1) = bc1
    (x2, y2, t2) = bc2
    bp1 = util.Pose(x1, y1, 0, t1)
    bp2 = util.Pose(x2, y2, 0, t2)
    return bp1.near(bp2, delta[0], delta[-1])
    
def confWithin(c1, c2, delta):
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
    c1CartConf = c1.cartConf()
    c2CartConf = c2.cartConf()
    robot = c1.robot

    # Also be sure two head angles are the same
    (c1h1, c1h2) = c1['pr2Head']
    (c2h1, c2h2) = c2['pr2Head']

    return all([withinDelta(c1CartConf[x],c2CartConf[x]) \
                for x in robot.moveChainNames]) and \
                nearAngle(c1h1, c2h1, delta[-1]) and \
                nearAngle(c1h2, c2h2, delta[-1])

# Do we know where the object is?
class BLoc(Fluent):
    predicate = 'BLoc'
    def test(self, state):
        (obj, var, prob) = self.args
        return B([Pose([obj, '*']), '*', var, '*', prob], True).test(state) \
          or B([Grasp([obj, '*', '*']), '*', var, '*', prob], True).test(state)

    def fglb(self, other, details = None):
        (so, sv, sp) = self.args
        if other.predicate == 'BLoc':
            (oo, ov, op) = other.args

            svGeq = all([a >= b for (a, b) in zip(sv, ov)])
            ovGeq = all([a >= b for (a, b) in zip(ov, sv)])
            spGeq = sp >= op
            opGeq = op >= sp

            if ovGeq and spGeq and so == oo:
                return (self, {})
            if svGeq and opGeq and so == oo:
                return (other, {})
            else:
                return ({self, other}, {})

        if other.predicate == 'B' and other.args[0].predicate == 'Pose' and \
                other.args[0].args[0] == so:
            (of, om, ov, od, op) = other.args
            # Pose can entail BLoc but not other way
            svGeq = all([a >= b for (a, b) in zip(sv, ov)])
            opGeq = op >= sp
            if svGeq and opGeq:
                return (other, {})
            else:
                return ({self, other}, {})

        if other.predicate == 'B' and other.args[0].predicate == 'Grasp' and \
                other.args[0].args[0] == so:
            (of, om, ov, od, op) = other.args
            # Grasp can entail BLoc but not other way
            svGeq = all([a >= b for (a, b) in zip(sv, ov)])
            opGeq = op >= sp
            if svGeq and opGeq:
                return (other, {})
            else:
                return ({self, other}, {})
            
        return ({self, other}, {})

    def argString(self, eq = True):
        (obj, var, prob) = self.args
        stdev = tuple([np.sqrt(v) for v in var]) \
                         if (not var == None and not isVar(var)) else var
        return '['+obj + ', '+prettyString(stdev)+', '+prettyString(prob)+']'

class Conf(Fluent):
    # This is will not be wrapped in a belief fluent.
    # Args: conf, delta
    predicate = 'Conf'
    def test(self, bState):
        (targetConf, delta) = self.args
        return confWithin(bState.pbs.conf, targetConf, delta)

    def getGrounding(self, details):
        assert self.value == True
        (targetConf, delta) = self.args
        assert not isGround(targetConf)
        return {targetConf : details.pbs.conf}

    def couldClobber(self, other, details = None):
        return other.predicate in ('Conf', 'BaseConf')

    def fglb(self, other, details = None):
        if other.predicate == 'BaseConf':
            return other.fglb(self, details)

        if other.predicate != 'Conf':
            return {self, other}, {}

        (oval, odelta) = other.args
        (sval, sdelta) = self.args

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
            bigDelta = tuple([od + sd for (od, sd) in zip(odelta, sdelta)])
            smallDelta = (1e-4,)*4

            # Kind of hard to average the confs.  Just be cheap for now.
            # If intervals don't overlap, then definitely false.
            if not confWithin(bsval, boval, bigDelta):
                return False, {}

            # One contains the other
            if confWithin(bsval, boval, smallDelta):
                if all([s < o  for (s, o) in zip(bsdelta,bodelta)]):
                    return self.applyBindings(b), b
                else:
                    return other.applyBindings(b), b

            print 'Should really compute the intersection of these intervals.'
            print 'Too lazy.  GLB returning False.'
            return False, {}
        else:
            return self.applyBindings(b), b

class BaseConf(Fluent):
    # This is will not be wrapped in a belief fluent.
    # Args: (x, y, theta), (dd, da)
    predicate = 'BaseConf'
    def test(self, bState):
        (targetConf, delta) = self.args
        return baseConfWithin(bState.pbs.conf['pr2Base'], targetConf, delta)

    def heuristicVal(self, details):
        # Assume we will need to do a look and a move
        # Could be smarter.
        dummyOp = Operator('LookMove', ['dummy'],{},[])
        dummyOp.instanceCost = 3
        return (dummyOp.instanceCost, {dummyOp})

    def fglb(self, other, details = None):
        (sval, sdelta) = self.args
        if other.predicate == 'Conf':
            (oval, odelta) = other.args
            if isVar(oval) or isVar(odelta):
                return {self, other}, {}
            obase = oval['pr2Base']
            if isVar(sval):
                return other, {sval : obase}
            if obase == sval:
                return other, {}
            else:
                # Contradiction
                return False, {}

        if other.predicate != 'BaseConf':
            return {self, other}, {}

        # Other predicate is BaseConf
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
            bigDelta = tuple([od + sd for (od, sd) in zip(odelta, sdelta)])
            smallDelta = (1e-4,)*4

            # Kind of hard to average the confs.  Just be cheap for now.
            # If intervals don't overlap, then definitely false.
            if not baseConfWithin(bsval, boval, bigDelta):
                return False, {}

            # One contains the other
            if baseConfWithin(bsval, boval, smallDelta):
                if all([s < o  for (s, o) in zip(bsdelta,bodelta)]):
                    return self, b
                else:
                    return other, b

            print 'Fix this!!  It is easy for base conf'
            print 'Should really compute the intersection of these intervals.'
            print 'Too lazy.  GLB returning False.'
            return False, {}
        else:
            return self, b

def innerPred(thing):
    return thing.args[0].predicate if thing.predicate in ('B', 'Bd')\
                          else thing.predicate
# Check reachability to "home"
class CanReachHome(Fluent):
    predicate = 'CanReachHome'
    implicit = True
    conditional = True

    def conditionOn(self, f):
        return f.predicate in ('Pose', 'SupportFace') and not ('*' in f.args)

    def getViols(self, bState, v, p, strict = True):
        assert v == True
        (conf, hand,
               lobj, lface, lgraspMu, lgraspVar, lgraspDelta,
               robj, rface, rgraspMu, rgraspVar, rgraspDelta,
               firstCondPerm, cond) = \
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

        if firstCondPerm:
            fc = cond[0]
            assert fc.args[0].predicate == 'Pose'
            avoidShadow = [fc.args[0].args[0]]
        else:
            avoidShadow = []
        newPBS.updateAvoidShadow(avoidShadow)

        path, violations = canReachHome(newPBS, conf, p, Violations(), draw=False)
        debugMsg('CanReachHome',
                 ('conf', conf),
                 ('lobjGrasp', lobj, lface, lgraspD),
                 ('robjGrasp', robj, rface, rgraspD),
                 ('->', violations))
        return path, violations

    def bTest(self, bState, v, p):
        path, violations = self.getViols(bState, v, p)

        result = bool(path and violations.empty())
        return result

    def heuristicVal(self, details, v, p):
        # Return cost estimate and a set of dummy operations
        (conf, hand,
               lobj, lface, lgraspMu, lgraspVar, lgraspDelta,
               robj, rface, rgraspMu, rgraspVar, rgraspDelta,
               firstCondPerm, cond) = \
                            self.args   # Args
        
        obstCost = 10  # move pick move place

        if not self.isGround():
            # assume an obstacle, if we're asking.  May need to decrease this
            dummyOp = Operator('RemoveObst', ['dummy'],{},[])
            dummyOp.instanceCost = obstCost
            return (obstCost, {dummyOp})
        
        path, violations = self.getViols(details, v, p, strict = False)
        if path == None:
            #!! should this happen?
            print '&&&&&&', self, v, p
            print 'hv infinite'
            raw_input('go?')
            return float('inf'), {}
        obstacles = violations.obstacles
        shadows = violations.shadows
        obstOps = set([Operator('RemoveObst', [o.name()],{},[]) \
                       for o in obstacles])
        for o in obstOps: o.instanceCost = obstCost
        shadowOps = set([Operator('RemoveShadow', [o.name()],{},[]) \
                     for o in shadows])
        d = details.domainProbs.minDelta
        ep = details.domainProbs.obsTypeErrProb
        vo = details.domainProbs.obsVarTuple
        # compute shadow costs individually
        shadowSum = 0
        for o in shadowOps:
            # Use variance in start state
            obj = objectName(o.args[0])
            vb = details.pbs.getPlaceB('table1').poseD.variance()
            deltaViolProb = probModeMoved(d[0], vb[0], vo[0])        
            c = 1.0 / ((1 - deltaViolProb) * (1 - ep) * 0.9 * 0.95)
            o.instanceCost = c
            shadowSum += c
        ops = obstOps.union(shadowOps)
        if debug('hAddBack'):
            print 'Heuristic val', self.predicate
            print 'ops', ops, 'cost',\
             prettyString(obstCost * len(obstOps) + shadowSum)
            raw_input('foo?')
        return (obstCost * len(obstacles) + shadowSum, ops)

    def prettyString(self, eq = True, includeValue = True):
        (conf, hand, lobj, lface, lgraspMu, lgraspVar, lgraspDelta,
               robj, rface, rgraspMu, rgraspVar, rgraspDelta,
               firstObjPerm, cond) = self.args   # Args
        if hand == 'right':
            (lobj, lface, lgraspMu, lgraspVar, lgraspDelta, \
             robj, rface, rgraspMu, rgraspVar, rgraspDelta) =  \
            (robj, rface, rgraspMu, rgraspVar, rgraspDelta, \
             lobj, lface, lgraspMu, lgraspVar, lgraspDelta)

        condStr = self.args[-1] if isVar(self.args[-1]) else \
          str([innerPred(c) for c in self.args[-1]]) 

        argStr = prettyString(self.args) if eq else \
                  prettyString([conf, lobj, robj, condStr], eq)
        valueStr = ' = ' + prettyString(self.value) if includeValue else ''
        return self.predicate + ' ' + argStr + valueStr

# LPK: Better if we could share code with CanReachHome
class CanReachNB(Fluent):
    predicate = 'CanReachNB'
    implicit = True
    conditional = True

    def conditionOn(self, f):
        return f.predicate in ('Pose', 'SupportFace') and not ('*' in f.args)

    def getViols(self, bState, v, p, strict = True):
        assert v == True
        (startConf, endConf, hand,
               lobj, lface, lgraspMu, lgraspVar, lgraspDelta,
               robj, rface, rgraspMu, rgraspVar, rgraspDelta,
               firstCondPerm, cond) = self.args   # Args

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

        # Fix this!!!
        # if firstCondPerm:
        #     fc = cond[0]
        #     assert fc.args[0].predicate == 'Pose'
        #     avoidShadow = [fc.args[0].args[0]]
        # else:
        #     avoidShadow = []

        path, violations = canReachNB(newPBS, startConf, endConf, p,
                                      Violations(), draw=False)
        debugMsg('CanReachNB',
                 ('confs', startConf, endConf),
                 ('lobjGrasp', lobj, lface, lgraspD),
                 ('robjGrasp', robj, rface, rgraspD),
                 ('->', violations))
        return path, violations

    def bTest(self, bState, v, p):
        ## Real version
        (startConf, endConf) = (self.args[0], self.args[1])

        if isVar(endConf):
            assert 'need to have end conf bound to test'
        elif isVar(startConf):
            print self
            print 'BTest canReachNB returning True'
            # Assume we can make it work out
            return True
        elif startConf['pr2Base'] != endConf['pr2Base']:
            # Bases have to be equal!
            debugMsg('canReachNB', 'Base not belong to us', startConf, endConf)
            return False
        
        path, violations = self.getViols(bState, v, p)

        if not bool(path and violations.empty()) and debug('canReachNB'):
            startConf.draw('W', 'black')
            endConf.draw('W', 'blue')
            print 'Conditions'
            for c in self.args[-1]: print '    ', c
            print 'Violations', violations
            debugMsg('canReachNB', 'CanReachNB is false!!')

        return bool(path and violations.empty())

    def getGrounding(self, details):
        assert self.value == True
        (startConf, targetConf) = (self.args[0], self.args[1])
        assert not isGround(targetConf)
        # Allow startConf to remain unbound
        return {}

    def fglb(self, other, details):
        # If start and target are the same, then everybody entails us!
        # Really should have a more general mechanism for simplifying conditions
        (startConf, conf, hand,
               lobj, lface, lgraspMu, lgraspVar, lgraspDelta,
               robj, rface, rgraspMu, rgraspVar, rgraspDelta,
               firstCondPerm, cond) = \
                            self.args   # Args
        if startConf == conf:
            return other, {}
        else:
            return {self, other}, {}
        
    # Exactly the same as for CanReachHome
    def heuristicVal(self, details, v, p):
        # Return cost estimate and a set of dummy operations
        (startConf, conf, hand,
               lobj, lface, lgraspMu, lgraspVar, lgraspDelta,
               robj, rface, rgraspMu, rgraspVar, rgraspDelta,
               firstCondPerm, cond) = \
                            self.args   # Args

        obstCost = 10  # move pick move place
        unboundCost = 1  # we don't know whether this will be hard or not

        if not self.isGround():
            dummyOp = Operator('UnboundStart', ['dummy'],{},[])
            dummyOp.instanceCost = unboundCost
            return (obstCost, {dummyOp})
            
        path, violations = self.getViols(details, v, p, strict = False)
        if path == None:
            #!! should this happen?
            print '&&&&&&', self, v, p
            print 'hv infinite'
            raw_input('go?')
            return float('inf'), {}
        obstacles = violations.obstacles
        shadows = violations.shadows
        obstOps = set([Operator('RemoveObst', [o.name()],{},[]) \
                       for o in obstacles])
        for o in obstOps: o.instanceCost = obstCost
        shadowOps = set([Operator('RemoveShadow', [o.name()],{},[]) \
                     for o in shadows])
        d = details.domainProbs.minDelta
        ep = details.domainProbs.obsTypeErrProb
        vo = details.domainProbs.obsVarTuple
        # compute shadow costs individually
        shadowSum = 0
        for o in shadowOps:
            # Use variance in start state
            obj = objectName(o.args[0])
            vb = details.pbs.getPlaceB('table1').poseD.variance()
            deltaViolProb = probModeMoved(d[0], vb[0], vo[0])        
            c = 1.0 / ((1 - deltaViolProb) * (1 - ep) * 0.9 * 0.95)
            o.instanceCost = c
            shadowSum += c
        ops = obstOps.union(shadowOps)
        if debug('hAddBack'):
            print 'Heuristic val', self.predicate
            print 'ops', ops, 'cost',\
             prettyString(obstCost * len(obstOps) + shadowSum)
            raw_input('foo?')
        return (obstCost * len(obstacles) + shadowSum, ops)

    def prettyString(self, eq = True, includeValue = True):
        (startConf, endConf, hand,
               lobj, lface, lgraspMu, lgraspVar, lgraspDelta,
               robj, rface, rgraspMu, rgraspVar, rgraspDelta,
               firstObjPerm, cond) = self.args   # Args
        if hand == 'right':
            (lobj, lface, lgraspMu, lgraspVar, lgraspDelta, \
             robj, rface, rgraspMu, rgraspVar, rgraspDelta) =  \
            (robj, rface, rgraspMu, rgraspVar, rgraspDelta, \
             lobj, lface, lgraspMu, lgraspVar, lgraspDelta)
             
        condStr = self.args[-1] if isVar(self.args[-1]) else \
          str([innerPred(c) for c in self.args[-1]]) 

        argStr = prettyString(self.args) if eq else \
                  prettyString([startConf, endConf, lobj, robj, condStr], eq)
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

    def conditionOn(self, f):
        return f.predicate in ('Pose', 'SupportFace') and not ('*' in f.args)

    # Add a glb method that will at least return False, {} if the two are
    # in contradiction.  How to test, exactly?


    # Override the default version of this so that the component conds
    # will be recalculated
    def addConditions(self, newConds, details = None):
        self.conds = None
        Fluent.addConditions(self, newConds, details)

    def getConds(self, details = None):
        # Will recompute if new bindings are applied because the result
        # won't have this attribute
        (preConf, ppConf, hand, obj, pose, poseVar, poseDelta, poseFace,
          graspFace, graspMu, graspVar, graspDelta,
         oObj, oFace, oGraspMu, oGraspVar, oGraspDelta,
        opType, inconds) = self.args

        if details:
            pbs = details.pbs
            pbs.getRoadMap().approachConfs[ppConf] = preConf
    
        assert obj != 'none'

        if not hasattr(self, 'conds') or self.conds == None:
            objInPlace = B([Pose([obj, poseFace]), pose, poseVar, poseDelta,
                            1.0], True)
            objInPlaceZeroVar = B([Pose([obj, poseFace]), pose, zeroVar,
                                   tinyDelta,1.0], True)
            self.conds = \
          [# 1.  Home to approach, holding nothing, obj in place
           # If it's a place operation, the shadow is irreducible
              CanReachHome([preConf, hand,
                        'none', 0, zeroPose, zeroVar, tinyDelta,
                        oObj, oFace, oGraspMu, oGraspVar, oGraspDelta,
                        opType == 'place', [objInPlace]]),
              # 2.  Home to approach with object in hand
              CanReachHome([preConf, hand, obj,
                                graspFace, graspMu, graspVar, graspDelta,
                                oObj, oFace, oGraspMu, oGraspVar, oGraspDelta,
                                False, []]),
              # 3.  Home to pick with hand empty, obj in place with zero var
              CanReachHome([ppConf, hand, 
                               'none', 0, zeroPose, zeroVar, tinyDelta,
                                oObj, oFace, oGraspMu, oGraspVar, oGraspDelta,
                                False, [objInPlaceZeroVar]]),
             # 4. Home to pick with the object in hand with zero var and delta
              CanReachHome([ppConf, hand,
                                obj, graspFace, graspMu, zeroVar, tinyDelta,
                                oObj, oFace, oGraspMu, oGraspVar, oGraspDelta,
                                False, []])]
            for c in self.conds: c.addConditions(inconds, details)
        return self.conds

    def getViols(self, bState, v, p, strict = True):
        def violCombo(v1, v2):
            return v1.update(v2)
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

    def heuristicVal(self, details, v, p):
        # Return cost estimate and a set of dummy operations
        obstCost = 5  # move, pick, move, place, maybe a look at hand

        if not self.isGround():
            # assume an obstacle, if we're asking.  May need to decrease this
            dummyOp = Operator('RemoveObst', ['dummy'],{},[])
            dummyOp.instanceCost = obstCost
            return (obstCost, {dummyOp})

        shadowCost = 3  # move look, if we're lucky
        path, violations = self.getViols(details, v, p, strict = False)
        if path == None:
            #!! should this happen?
            print 'hv infinite'
            print self
            raw_input('go?')
            return float('inf'), {}
        obstacles = violations.obstacles
        shadows = violations.shadows
        obstOps = set([Operator('RemoveObst', [o.name()],{},[]) \
                       for o in obstacles])
        for o in obstOps: o.instanceCost = obstCost
        shadowOps = set([Operator('RemoveShadow', [o.name()],{},[]) \
                     for o in shadows])

        d = details.domainProbs.minDelta
        ep = details.domainProbs.obsTypeErrProb
        vo = details.domainProbs.obsVarTuple
        # compute shadow costs individually
        shadowSum = 0
        for o in shadowOps:
            # Use variance in start state
            obj = objectName(o.args[0])
            vb = details.pbs.getPlaceB('table1').poseD.variance()
            deltaViolProb = probModeMoved(d[0], vb[0], vo[0])        
            c = 1.0 / ((1 - deltaViolProb) * (1 - ep) * 0.9 * 0.95)
            o.instanceCost = c
            shadowSum += c
        ops = obstOps.union(shadowOps)
        if debug('hAddBack'):
            print 'Heuristic val', self.predicate
            print 'ops', ops, 'cost',\
             prettyString(obstCost * len(obstOps) + shadowSum)
            raw_input('foo?')

        return (obstCost * len(obstacles) + shadowSum, ops)

    def prettyString(self, eq = True, includeValue = True):
        (preConf, ppConf, hand, obj, pose, poseVar, poseDelta, poseFace,
          face, graspMu, graspVar, graspDelta,
          oobj, oface, ograspMu, ograspVar, ograspDelta, op, conds) = self.args
        assert obj != 'none'

        condStr = self.args[-1] if isVar(self.args[-1]) else \
          str([innerPred(c) for c in self.args[-1]]) 

        argStr = prettyString(self.args) if eq else \
                  prettyString([obj, hand, pose, condStr], eq)
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

        if hand == '*':
            hl = bState.pbs.getHeld('left').mode()
            hr = bState.pbs.getHeld('right').mode()
            if obj == hl:
                hand = 'left'
            elif obj == hr:
                hand == 'right'
            else:
                # We don't think it's in the hand, so dist is huge
                return lostDist
        if face == '*':
            face = bState.pbs.getGraspB(obj, hand).grasp.mode()

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
        if face == '*':
            hl = bState.pbs.getHeld('left').mode()
            hr = bState.pbs.getHeld('right').mode()
            if hl == obj or hr == obj:
                # We think it's in the hand;  so pose dist is huge
                result = lostDist
            else:
                face = bState.pbs.getPlaceB(obj).support.mode()
                result = bState.poseModeDist(obj, face)
        elif face in ('left', 'right'):
            # Actually, the grasp dist, if the face is a hand!
            hand = face
            graspFace = bState.pbs.getGraspB(obj, hand).grasp.mode() 
            result = bState.graspModeDist(obj, hand, face)
        else:
            # normal case
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

# Not currently in use       
class RelPose(Fluent):
    predicate = 'Pose'
    def dist(self, bState):
        (obj1, face1, obj2, face2) = self.args
        d1 = bState.poseModeDist(obj1, face1)
        p1 = bState.poseModeProbs[obj1]
        if obj2 == 'robot':
            r = bState.pbs.conf['basePose']
            mu = d1.mode().compose(r.inverse())
            return GMU([(MVG(mu.xyztTuple(), d1.variance()), p1)])
        else:
            d2 = bState.poseModeDist(obj2, face2)
            p2 = bState.poseModeProbs[obj2]
            mu = d1.mode().compose(d2.mode().inverse())
            variance = [a+b for (a, b) in zip(d1.varianceTuple(),
                                              d2.varianceTuple())]
            return GMU([(MVG(mu.xyztTuple(), diagToSq(variance)),
                            p1 * p2)])

    def fglb(self, other, bState = None):
        assert False, 'Not implemented'
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
        # Mode should be 'left' or 'right' if the object is in the hand
        hl = bState.pbs.getHeld('left').mode()
        hr = bState.pbs.getHeld('right').mode()
        if hl == obj:
            return DeltaDist('left')
        elif hr == obj:
            return DeltaDist('right')
        else:
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

    def conditionOn(self, f):
        return f.predicate in \
          ('Pose', 'SupportFace', 'Holding', 'Grasp', 'GraspFace') \
          and not ('*' in f.args) 

    def bTest(self, details, v, p):
        assert v == True
        (obj, pose, poseFace, conf, cond) = self.args

        # Note that all object poses are permanent, no collisions can be ignored
        newPBS = details.pbs.copy()

        if pose == '*' and \
          (newPBS.getHeld('left').mode() == obj or \
           newPBS.getHeld('right').mode() == obj):
           # Can't see it (in the usual way) if it's in the hand and a pose
           # isn't specified
           return False
         
        newPBS.updateFromAllPoses(cond)
        placeB = newPBS.getPlaceB(obj)

        # LPK! Forcing the variance to be very small.  Currently it's
        # using variance from the initial state, and then overriding
        # it based on conditions.  This is incoherent.  Could change
        # it to put variance explicitly in the fluent.
        placeB = placeB.modifyPoseD(var = (0.0001, 0.0001, 0.0001, 0.0005))

        if placeB.support.mode() != poseFace and poseFace != '*':
            placeB.support = DeltaDist(poseFace)
        if placeB.poseD.mode() != pose and pose != '*':
            placeB = placeB.modifyPoseD(mu = pose)
        newPBS.updatePermObjPose(placeB)

        # LPK! Force recompute
        newPBS.shadowWorld = None

        shWorld = newPBS.getShadowWorld(p)
        shName = shadowName(obj)
        sh = shWorld.objectShapes[shName]
        obstacles = [s for s in shWorld.getNonShadowShapes() if s.name() != obj ] + \
                    [conf.placement(shWorld.attached)]
        ans, _ = visible(shWorld, conf, sh, obstacles, p, moveHead=False)

        return ans

    def getViols(self, bState, v, p, strict = True):
        assert v == True
        (obj, pose, poseFace, conf, cond) = self.args
         
        # Note that all object poses are permanent, no collisions can be ignored
        newPBS = bState.pbs.copy()
        if strict:
            newPBS.updateFromAllPoses(cond, updateHeld=False)
        else:
            newPBS.updateFromGoalPoses(cond, updateHeld=False)

        placeB = newPBS.getPlaceB(obj)
        if placeB.support.mode() != poseFace and poseFace != '*':
            placeB.support = DeltaDist(poseFace)
        if placeB.poseD.mode() != pose and pose != '*':
            newPBS.updatePermObjPose(placeB.modifyPoseD(mu=pose))
        shWorld = newPBS.getShadowWorld(p)
        shName = shadowName(obj)
        sh = shWorld.objectShapes[shName]
        obstacles = [s for s in shWorld.getNonShadowShapes() if \
                     s.name() != obj ] + \
                     [conf.placement(shWorld.attached)]
        ans, occluders = visible(shWorld, conf, sh, obstacles, p, moveHead=False)

        debugMsg('CanSeeFrom',
                ('obj', obj, pose), ('conf', conf),
                 ('->', occluders))
        return ans, occluders
    
    '''
    def fglb(self, other, details = None):
        if other.predicate != 'CanSeeFrom' or
            self.args[:-1] != other.args[:-1]:
            return {self, other}
        cSelf = self.args[-1]
        cOther = other.args[-1]

        sMinusO = cSelf.setMinus(cOther)
        oMinusS = self.setMinus(cOther)

        # If the only difference is holding = none, that is entailed
        # by a fluent with no conditions.
    ''' 
        

    def heuristicVal(self, details, v, p):
        # Return cost estimate and a set of dummy operations
        (obj, pose, poseFace, conf, cond) = self.args
        
        obstCost = 10  # move pick move place

        if not self.isGround():
            # assume an obstacle, if we're asking.  May need to decrease this
            dummyOp = Operator('RemoveObst', ['dummy'],{},[])
            dummyOp.instanceCost = obstCost
            return (obstCost, {dummyOp})
        
        path, occluders = self.getViols(details, v, p, strict = False)
        if path == None:
            #!! should this happen?
            print '&&&&&&', self, v, p
            print 'hv infinite'
            raw_input('go?')
            return float('inf'), {}
        obstacles = occluders
        shadows = [] # I think these are never shadows?
        obstOps = set([Operator('RemoveObst', [oName],{},[]) \
                       for oName in obstacles])
        for o in obstOps: o.instanceCost = obstCost
        shadowOps = set([Operator('RemoveShadow', [oName],{},[]) \
                     for oName in shadows])
        d = details.domainProbs.minDelta
        ep = details.domainProbs.obsTypeErrProb
        vo = details.domainProbs.obsVarTuple
        # compute shadow costs individually
        shadowSum = 0
        for o in shadowOps:
            # Use variance in start state
            obj = objectName(o.args[0])
            vb = details.pbs.getPlaceB('table1').poseD.variance()
            deltaViolProb = probModeMoved(d[0], vb[0], vo[0])        
            c = 1.0 / ((1 - deltaViolProb) * (1 - ep) * 0.9 * 0.95)
            o.instanceCost = c
            shadowSum += c
        ops = obstOps.union(shadowOps)
        if debug('hAddBack'):
            print 'Heuristic val', self.predicate
            print 'ops', ops, 'cost',\
             prettyString(obstCost * len(obstOps) + shadowSum)
            raw_input('foo?')
        return (obstCost * len(obstacles) + shadowSum, ops)

    def prettyString(self, eq = True, includeValue = True):
        (obj, pose, poseFace, conf, cond) = self.args

        # if not isVar(cond) and len(cond) > 0:
        #     print 'CanSeeFrom'
        #     print self.args
        #     raw_input('conditions okay?')

        condStr = self.args[-1] if isVar(self.args[-1]) else \
          str([innerPred(c) for c in self.args[-1]]) 

        argStr = prettyString(self.args) if eq else \
                  prettyString([conf, obj, condStr], eq)
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
# BaseConf

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
        # else:
        #     # Not a B fluent
        #     pf = getMatchingFluents(fluents, Conf(['C', 'D'], True)) + \
        #          getMatchingFluents(fluents, BaseConf(['C', 'D'], True))
        #     for (ff, b) in pf:
        #         newSet.add(ff)
        #         fluents.remove(ff)
                    
        groups.append(frozenset(newSet))
    return groups

###
# Tests
###

# LPK: canReachNB which is like canReachHome, but without moving
# the base.  args are pbs, startConf, endConf, prob, initViol, ...
# (rest the same as canReachHome).

# !!!! needs fixing

def canReachNB(pbs, startConf, conf, prob, initViol,
               optimize = False, draw=True):
    rm = pbs.getRoadMap()
    robot = pbs.getRobot()
    initConf = startConf or rm.homeConf
    # Reverse start and target
    viol, cost, pathRev = rm.confReachViol(conf, pbs, prob,
                                           initViol,
                                           startConf=initConf,
                                           optimize=optimize,
                                           moveBase = False)
    path = pathRev[::-1] if pathRev else pathRev

    if debug('traceCRH'):
        print '    canReachNB h=', fbch.inHeuristic, 'viol=:', viol.weight() if viol else None

    if (not fbch.inHeuristic) or debug('drawInHeuristic'):
        if debug('canReachNB'):
            pbs.draw(prob, 'W')
            if path:
                drawPath(path, viol=viol,
                         attached=pbs.getShadowWorld(prob).attached)
            else:
                print 'viol, cost, path', viol, cost, path
        debugMsg('canReachNB', ('viol', viol))

    return path, viol

# 
def canReachHome(pbs, conf, prob, initViol,
                 avoidShadow = [], startConf = None, reversePath = False,
                 optimize = False, draw=True):
    rm = pbs.getRoadMap()
    robot = pbs.getRobot()
    # Reverse start and target
    viol, cost, path = rm.confReachViol(conf, pbs, prob, initViol,
                                        startConf=startConf,
                                        reversePath = reversePath,
                                        optimize = optimize)
    if debug('checkCRH') and fbch.inHeuristic:
        pbs.draw(prob, 'W')
        fbch.inHeuristic = False
        pbs.shadowWorld = None
        pbs.draw(prob, 'W', clear=False) # overlay
        conf.draw('W', 'blue')
        rm.homConf.draw('W', 'pink')
        viol2, cost2, pathRev2 = rm.confReachViol(rm.homeConf, pbs, prob, initViol,
                                                  startConf=conf,
                                                  optimize = optimize)
        if pathRev2: path2 = pathRev2[::-1]
        else: path2 = pathRev2
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

        # LPK took this out;  it's not necessarily a bad thing
        # if fbch.inHeuristic:
        #     pbs.draw(prob, 'W')
        #     conf.draw('W', attached=pbs.getShadowWorld(prob).attached)
        #     raw_input('canReachHome failed with inHeuristic=True')

        if fbch.inHeuristic and debug('extraTests'):
            pbs.draw(prob, 'W')
            print 'canReachHome failed with inHeuristic=True'
            fbch.inHeuristic = False
            viol, cost, path = rm.confReachViol(conf, pbs, prob, initViol,
                                                startConf=startConf)
            if path:
                raw_input('Inconsistency')
            else:
                print 'Consistent result with inHeuristic=False'
            fbch.inHeuristic = True

    if (not fbch.inHeuristic) or debug('drawInHeuristic'):
        if debug('canReachHome'):
            pbs.draw(prob, 'W')
            if path:
                drawPath(path, viol=viol,
                         attached=pbs.getShadowWorld(prob).attached)
            else:
                print 'viol, cost, path', viol, cost, path
        debugMsg('canReachHome', ('viol', viol))

    return path, viol

def findRegionParent(bState, region):
    regs = bState.pbs.getWorld().regions
    for (obj, stuff) in regs.items():
        for (regName, regShape, regTr) in stuff:
            if regName == region:
                return obj
    raw_input('No parent object for region '+str(region))
    return None

# probability is: pObj * pParent * pObjFitsGivenRelativeVar = prob
def inTest(bState, obj, regName, prob, pB=None):
    regs = bState.pbs.getWorld().regions
    parent = findRegionParent(bState, regName)
    pObj = bState.poseModeProbs[obj]
    pParent = bState.poseModeProbs[parent]
    pFits = prob / (pObj * pParent)
    if pFits > 1: return False

    # compute a shadow for this object
    placeB = pB or bState.pbs.getPlaceB(obj)
    faceFrame = placeB.faceFrames[placeB.support.mode()]

    # !! Clean this up
    sh = bState.pbs.objShadow(obj, shadowName(obj), pFits, placeB, faceFrame)
    shadow = sh.applyLoc(placeB.objFrame()) # !! is this right?
    shWorld = bState.pbs.getShadowWorld(prob)
    region = shWorld.regionShapes[regName]
    
    ans = np.all(np.all(np.dot(region.planes(), shadow.vertices()) <= tiny, axis=1))

    if debug('testVerbose') or debug('inTest'):
        shadow.draw('W', 'brown')
        region.draw('W', 'purple')

        print 'shadow', shadow.bbox()
        print 'region', region.bbox()

        print 'shadow in brown, region in purple'
        print 'inTest', obj, '->', ans
        raw_input('Ok?')

    return ans

                
