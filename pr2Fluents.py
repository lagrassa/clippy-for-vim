import pdb
import math
import time
import hu
import numpy as np
from planUtil import Violations, ObjPlaceB, ObjGraspB
from pr2Util import shadowName, drawPath, objectName, PoseD, \
     supportFaceIndex, GDesc, inside, otherHand, graspable, pushable, permanent
from dist import DeltaDist, probModeMoved
from traceFile import debugMsg, debug, pause
import planGlobals as glob
from miscUtil import isGround, isVar, prettyString, applyBindings
from fbch import Fluent, getMatchingFluents, Operator
from belief import B, Bd, ActSet
from pr2Visible import visible
from pr2BeliefState import lostDist
from pr2RoadMap import validEdgeTest
from pr2Robot import gripperFaceFrame, pr2BaseLink
from traceFile import tr, trAlways
import mathematica
import windowManager3D as wm
from transformations import rotation_matrix
from shapes import drawFrame

tiny = 1.0e-6

#### Ugly!!!!!
graspObstCost = 20  # Heuristic cost of moving an object
pushObstCost = 75  # Heuristic cost of moving an object

################################################################
## Fluent definitions
################################################################

class Graspable(Fluent):
    predicate = 'Graspable'
    immutable = True
    # noinspection PyUnusedLocal
    def test(self, details):
        (objName,) = self.args
        return graspable(objName)

class Pushable(Fluent):
    predicate = 'Pushable'
    immutable = True
    # noinspection PyUnusedLocal
    def test(self, details):
        (objName,) = self.args
        return pushable(objName)
    
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
    bp1 = hu.Pose(x1, y1, 0, t1)
    bp2 = hu.Pose(x2, y2, 0, t2)
    return bp1.near(bp2, delta[0], delta[-1])
    
def confWithin(c1, c2, delta):
    def withinDelta(a, b):
        if isinstance(a, list):
            dd = delta[0]                # !! hack
            return all([abs(a[i] - b[i]) <= dd \
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

    # Also look at gripper

    return all([withinDelta(c1CartConf[x],c2CartConf[x]) \
                for x in robot.moveChainNames]) and \
                hu.nearAngle(c1h1, c2h1, delta[-1]) and \
                hu.nearAngle(c1h2, c2h2, delta[-1])

# Do we know where the object is?
class BLoc(Fluent):
    predicate = 'BLoc'
    def test(self, state):
        (obj, var, prob) = self.args
        return B([Pose([obj, '*']), '*', var, '*', prob], True).test(state) \
          or B([Grasp([obj, '*', '*']), '*', var, '*', prob], True).test(state)

    # noinspection PyUnusedLocal
    def fglb(self, other, details = None):
        (so, sv, sp) = self.args
        if other.predicate == 'BLoc':
            (oo, ov, op) = other.args

            svGeq = all([a >= b for (a, b) in zip(sv, ov)])
            ovGeq = all([a >= b for (a, b) in zip(ov, sv)])
            spGeq = sp >= op
            opGeq = op >= sp

            if ovGeq and spGeq and so == oo:
                return self, {}
            if svGeq and opGeq and so == oo:
                return other, {}
            else:
                return {self, other}, {}

        if other.predicate == 'B' and other.args[0].predicate == 'Pose' and \
                other.args[0].args[0] == so:
            (of, om, ov, od, op) = other.args
            # Pose can entail BLoc but not other way
            svGeq = all([a >= b for (a, b) in zip(sv, ov)])
            opGeq = op >= sp
            if svGeq and opGeq:
                return other, {}
            else:
                return {self, other}, {}

        if other.predicate == 'B' and other.args[0].predicate == 'Grasp' and \
                other.args[0].args[0] == so:
            (of, om, ov, od, op) = other.args
            # Grasp can entail BLoc but not other way
            svGeq = all([a >= b for (a, b) in zip(sv, ov)])
            opGeq = op >= sp
            if svGeq and opGeq:
                return other, {}
            else:
                return {self, other}, {}
            
        return {self, other}, {}

    def argString(self, eq = True):
        (obj, var, prob) = self.args
        stdev = tuple([np.sqrt(v) for v in var]) \
                         if (not var is None and not isVar(var)) else var
        return '['+obj + ', '+prettyString(stdev)+', '+prettyString(prob)+']'

class Conf(Fluent):
    # This is will not be wrapped in a belief fluent.
    # Args: conf, delta
    predicate = 'Conf'
    def test(self, bState):
        assert self.isGround()
        (targetConf, delta) = self.args
        return confWithin(bState.pbs.conf, targetConf, delta)

    def getGrounding(self, bstate):
        assert self.value == True
        (targetConf, delta) = self.args
        assert not isGround(targetConf)
        return {targetConf : bstate.details.pbs.conf}

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

            trAlways('Should really compute the intersection of these intervals.',
                     'Too lazy.  GLB returning False.', pause = False)
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
        # Needs heuristic val because we don't have an operator that
        # explicitly produces this.  Assume we will need to do a look
        # and a move Could be smarter.
        if self.test(details):
            return 0
        
        dummyOp = Operator('LookMove', ['dummy'],{},[])
        dummyOp.instanceCost = 7.0
        return (dummyOp.instanceCost, ActSet([dummyOp]))

    def fglb(self, other, details = None):
        (sval, sdelta) = self.args
        if other.predicate == 'Conf':
            (oval, odelta) = other.args
            if isVar(oval) or isVar(odelta):
                return {self, other}, {}
            obase = oval['pr2Base']
            if isVar(sval):
                return other, {sval : obase}
            if baseConfWithin(obase, sval, sdelta):
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

            trAlways('Fix this!!  It is easy for base conf',
                     'Should really compute the intersection of these intervals.',
                     'Too lazy.  GLB returning False.', pause = False)
            return False, {}
        else:
            return self, b

def innerPred(thing):
    pred = thing.args[0].predicate if thing.predicate in ('B', 'Bd')\
                          else thing.predicate
    arg0 = thing.args[0].args[0] if thing.predicate in ('B', 'Bd')\
                          else thing.args[0]
    return pred+'('+arg0+',...)'

# Check reachability to "home"
class CanReachHome(Fluent):
    predicate = 'CanReachHome'
    implicit = True
    conditional = True

    # Args are: conf, cond

    def conditionOn(self, f):
        preds = ('Pose', 'SupportFace', 'Holding', 'Grasp', 'GraspFace')
        return (f.predicate in preds and not ('*' in f.args)) or \
          (f.predicate in ('B', 'Bd') and self.conditionOn(f.args[0]))

    def update(self):
        super(CanReachHome, self).update()
        self.viols = {}
        self.hviols = {}

    # TODO : LPK: Try to share this code across CanX fluents
    def getViols(self, bState, v, p):
        assert v == True
        (conf, cond) = self.args

        key = (hash(bState.pbs), p)
        if not hasattr(self, 'viols'): self.viols = {}
        if not hasattr(self, 'hviols'): self.hviols = {}
        if key in self.viols: return self.viols[key]
        if glob.inHeuristic and key in self.hviols: return self.hviols[key]

        newPBS = bState.pbs.conditioned([], cond)
        path, violations = canReachHome(newPBS, conf, p, Violations())
        debugMsg('CanReachHome',
                 ('conf', conf),
                 ('->', violations))
        if glob.inHeuristic:
            self.hviols[key] = path, violations
        else:
            self.viols[key] = path, violations
            
        return path, violations

    def bTest(self, bState, v, p):
        path, violations = self.getViols(bState, v, p)

        return bool(path and violations.empty())

    def feasible(self, bState, v, p):
        path, violations = self.getViols(bState, v, p)
        return violations != None

    def heuristicVal(self, details, v, p):
        # Return cost estimate and a set of dummy operations
        (conf, cond) = self.args

        if not self.isGround():
            # assume an obstacle, if we're asking.  May need to decrease this
            dummyOp = Operator('RemoveObst', ['dummy'],{},[])
            dummyOp.instanceCost = graspObstCost
            return (graspObstCost, ActSet([dummyOp]))
        
        path, violations = self.getViols(details, v, p)

        totalCost, ops = hCost(violations, details)
        tr('hv', ('Heuristic val', self.predicate),
           ('ops', ops), 'cost', totalCost)
        return totalCost, ops

    def prettyString(self, eq = True, includeValue = True):
        (conf, cond) = self.args
        condStr = self.args[-1] if isVar(self.args[-1]) else \
          str([innerPred(c) for c in self.args[-1]]) 

        argStr = prettyString(self.args) if eq else \
                  prettyString([conf, condStr], eq)
        valueStr = ' = ' + prettyString(self.value) if includeValue else ''
        return self.predicate + ' ' + argStr + valueStr

# Returns cost, ops
def hCost(violations, details):
    if violations is None:
        tr('hAddBackInf', 'computed hcost is infinite')
        return float('inf'), {}
    obstacles = violations.obstacles
    shadows = violations.shadows
    obstOps = set([Operator('RemoveObst', [o.name()],{},[]) \
                   for o in obstacles])
    for o in obstOps: o.instanceCost = graspObstCost \
                    if graspable(o.args[0]) else pushObstCost
    shadowOps = set([Operator('RemoveShadow', [o.name()],{},[]) \
                 for o in shadows])
    d = details.domainProbs.minDelta
    ep = details.domainProbs.obsTypeErrProb
    vo = details.domainProbs.obsVarTuple

    for o in shadowOps:
        # Use variance in start state
        # Should try to figure out a number of looks needed
        # For now, assume 2
        obj = objectName(o.args[0])
        vb = details.pbs.getPlaceB(obj).poseD.variance()
        deltaViolProb = probModeMoved(d[0], vb[0], vo[0])        
        c = 1.0 / ((1 - deltaViolProb) * (1 - ep) * 0.9 * 0.95)
        o.instanceCost = c * 2    # two looks
    ops = obstOps.union(shadowOps)

    # look at hand or drop an object
    (heldObstL, heldObstR) = violations.heldObstacles # left set, right set
    (heldShadL, heldShadR) = violations.heldShadows # left set, right set

    leftDrop = len(heldObstL) > 0
    rightDrop = len(heldObstR) > 0
    leftLook = len(heldShadL) > 0
    rightLook = len(heldShadR) > 0

    if leftDrop:
        op = Operator('DropLeft', [], {}, [])
        op.instanceCost = graspObstCost / 2.0
        ops.add(op)
    if rightDrop:
        op = Operator('DropRight', [], {}, [])
        op.instanceCost = graspObstCost / 2.0
        ops.add(op)
    if leftLook:
        op = Operator('Lookleft', [], {}, [])
        # Should calculate grasp variance and delta, etc.  Skipping for now
        op.instanceCost = 1.0
        ops.add(op)
    if rightLook:
        op = Operator('LookRight', [], {}, [])
        # Should calculate grasp variance and delta, etc.  Skipping for now
        op.instanceCost = 1.0
        ops.add(op)

    totalCost = sum([o.instanceCost for o in ops])

    return totalCost, ActSet(ops)

def hCostSee(vis, occluders, details):
    # vis is whether enough is visible;   we're 0 cost if that's true
    if vis:
        return 0, ActSet()

    obstOps = set([Operator('RemoveObst', [o],{},[]) \
                   for o in occluders])
    for o in obstOps: o.instanceCost = graspObstCost
    totalCost = sum([o.instanceCost for o in obstOps])
    return totalCost, ActSet(obstOps)

class CanReachNB(Fluent):
    predicate = 'CanReachNB'
    implicit = True
    conditional = True

    def conditionOn(self, f):
        preds = ('Pose', 'SupportFace', 'Holding', 'Grasp', 'GraspFace')
        return (f.predicate in preds and not ('*' in f.args)) or \
          (f.predicate in ('B', 'Bd') and self.conditionOn(f.args[0]))

    def feasible(self, details, v, p):
        if not self.isGround():
            (startConf, endConf, cond) = self.args
            assert isGround(endConf) and isGround(cond)
            path, violations = CanReachNB([endConf, endConf, cond], True).\
                                    getViols(details, v, p)
        else:
            path, violations = self.getViols(details, v, p)
        return violations is not  None

    def update(self):
        super(CanReachNB, self).update()
        self.viols = {}
        self.hviols = {}

    def getViols(self, bState, v, p):
        assert v == True
        (startConf, endConf, cond) = self.args
        key = (hash(bState.pbs), p)
        if not hasattr(self, 'viols'): self.viols = {}
        if not hasattr(self, 'hviols'): self.hviols = {}
        if key in self.viols: return self.viols[key]
        if glob.inHeuristic and key in self.hviols: return self.hviols[key]

        newPBS = bState.pbs.copy()
        newPBS.updateFromGoalPoses(cond, permShadows=True)

        path, violations = canReachNB(newPBS, startConf, endConf, p,
                                      Violations())
        debugMsg('CanReachNB',
                 ('confs', startConf, endConf),
                 ('->', violations))
        if glob.inHeuristic:
            self.hviols[key] = path, violations
        else:
            self.viols[key] = path, violations
        return path, violations

    def bTest(self, bState, v, p):
        ## Real version
        (startConf, endConf, cond) = self.args

        if isVar(endConf):
            assert 'need to have end conf bound to test'
        elif isVar(startConf):
            tr('canReachNB',
               'BTest canReachNB returning False because startconf unbound',
                     self)
            return False
        elif max(abs(a-b) for (a,b) in \
                 zip(startConf['pr2Base'], endConf['pr2Base'])) > 1.0e-4:
            # Bases have to be equal!
            debugMsg('canReachNB', 'Base not belong to us', startConf, endConf)
            return False
        
        path, violations = self.getViols(bState, v, p)

        if violations is None:
            tr('canReachNB', 'impossible',
               ('conditions', self.args[-1]),
               ('violations', violations),
               draw = [(startConf, 'W', 'black'),(endConf, 'W', 'blue')],
               snap = ['W'])

        if not path:
            return False
        elif violations.empty():
            return True
        elif not (violations.obstacles or violations.heldShadows or \
                  violations.heldObstacles):
            assert violations.shadows
            (startConf, endConf, cond) = self.args
            return onlyBaseCollides(startConf, violations.shadows) and \
                       onlyBaseCollides(endConf, violations.shadows)
        else:
            return False

    def getGrounding(self, bstate):
        assert self.value == True
        (startConf, targetConf, cond) = self.args
        assert not isGround(targetConf)
        # Allow startConf to remain unbound
        return {}

    # def fglb(self, other, details):
    #     # If start and target are the same, then everybody entails us!
    #     # Really should have a more general mechanism for simplifying conditions
    #     (startConf, conf, cond) = self.args
    #     if startConf == conf:
    #         return other, {}
    #     else:
    #         return {self, other}, {}
        
    # Exactly the same as for CanReachHome
    def heuristicVal(self, details, v, p):
        # Return cost estimate and a set of dummy operations

        unboundCost = 1  # we don't know whether this will be hard or not

        if not self.isGround():
            dummyOp = Operator('UnboundStart', ['dummy'],{},[])
            dummyOp.instanceCost = unboundCost
            return unboundCost, ActSet([dummyOp])
            
        path, violations = self.getViols(details, v, p)

        totalCost, ops = hCost(violations, details)
        tr('hv', ('Heuristic val', self.predicate),
           ('ops', ops), 'cost', totalCost)
        return totalCost, ops

    def prettyString(self, eq = True, state = None, heuristic = None,
                     includeValue = True):
        (startConf, endConf, cond) = self.args
        condStr = self.args[-1] if isVar(self.args[-1]) else \
          str([innerPred(c) for c in self.args[-1]]) 

        argStr = prettyString(self.args) if eq else \
                  prettyString([startConf, endConf, condStr], eq)
        valueStr = ' = ' + prettyString(self.value) if includeValue else ''
        return self.predicate + ' ' + argStr + valueStr

def onlyBaseCollides(conf, shadows):
    parts = dict([(part.name(), part) for part in conf.placement().parts()])
    collide = any(any(parts[p].collides(sh) for sh in shadows) for p in parts if p != 'pr2Base')
    print parts, collide
    raw_input('onlyBaseCollides')
    if not collide:
        assert any(parts['pr2Base'].collides(sh) for sh in shadows)
        return True
    return False

zeroPose = zeroVar = (0.0,)*4
tinyDelta = (1e-8,)*4
zeroDelta = (0.0,)*4
awayPose = (100.0, 100.0, 0.0, 0.0)

# Check all four reachability conditions together.  For now, try to
# piggy-back on existing code.  Can probably optimize later.
class CanPickPlace(Fluent):
    predicate = 'CanPickPlace'
    implicit = True
    conditional = True

    def conditionOn(self, f):
        preds = ('Pose', 'SupportFace', 'Holding', 'Grasp', 'GraspFace')
        return (f.predicate in preds and not ('*' in f.args)) or \
          (f.predicate in ('B', 'Bd') and self.conditionOn(f.args[0]))

    def feasible(self, bState, v, p):
        path, violations = self.getViols(bState, v, p)
        return violations is not None

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
          opType, inconds) = self.args

        if details:
            pbs = details.pbs
            pbs.getRoadMap().approachConfs[ppConf] = preConf
    
        assert obj != 'none'

        tinyDelta = zeroDelta

        if not hasattr(self, 'conds') or self.conds is None:
            placeVar = details.domainProbs.placeVar if details else poseVar
            objInPlacePlaceVar = [B([Pose([obj, poseFace]), pose, placeVar, poseDelta,
                                     1.0], True),
                                  Bd([SupportFace([obj]), poseFace, 1.0], True)]
            objInPlaceZeroVar = [B([Pose([obj, poseFace]), pose, zeroVar,
                                   tinyDelta,1.0], True),
                          Bd([SupportFace([obj]), poseFace, 1.0], True)]
            objInPlace = objInPlacePlaceVar if opType == 'place' else objInPlaceZeroVar
            holdingNothing = [Bd([Holding([hand]), 'none', 1.0], True)]
            objInHand = [B([Grasp([obj, hand, graspFace]),
                           graspMu, graspVar, graspDelta, 1.0], True),
                         Bd([GraspFace([obj, hand]), graspFace, 1.0], True),
                         Bd([Holding([hand]), obj, 1.0], True)]               
            objInHandZeroVar = [B([Grasp([obj, hand, graspFace]),
                                   graspMu, zeroVar, tinyDelta, 1.0], True),
                         Bd([GraspFace([obj, hand]), graspFace, 1.0], True),
                         Bd([Holding([hand]), obj, 1.0], True)]
                                   
            self.conds = \
             [# 1.  Home to approach, holding nothing, obj in place
              CanReachHome([preConf,
                            objInPlace + holdingNothing]),
              # 2.  Home to approach with object in hand
              CanReachHome([preConf, objInHand]),
              # 3.  Home to pick with hand empty, obj in place with zero var
              CanReachHome([ppConf,
                            holdingNothing + objInPlaceZeroVar]),
              # 4. Home to pick with the object in hand with zero var and delta
              CanReachHome([ppConf, objInHandZeroVar])]
            for c in self.conds: c.addConditions(inconds, details)
        return self.conds


    def update(self):
        super(CanPickPlace, self).update()
        self.viols = {}
        self.hviols = {}

    def getViols(self, bState, v, p):
        def violCombo(v1, v2):
            return v1.update(v2)

        key = (hash(bState.pbs), p)
        if not hasattr(self, 'viols'): self.viols = {}
        if not hasattr(self, 'hviols'): self.hviols = {}
        if key in self.viols: return self.viols[key]
        if glob.inHeuristic and key in self.hviols: return self.hviols[key]
            
        condViols = [c.getViols(bState, v, p) for c in self.getConds(bState)]

        if debug('CanPickPlace'):
            print 'canPickPlace getViols'
            for (cond, (p, viol)) in zip(self.getConds(bState), condViols):
                print '    cond', cond
                print '    viol', viol

        pathNone = any([p is None for (p, v) in condViols])
        if pathNone:
            return (None, None)
        allViols = [v for (p, v) in condViols]
        violations = reduce(violCombo, allViols)
        if glob.inHeuristic:
            self.hviols[key] = True, violations
        else:
            self.viols[key] = True, violations
        return True, violations

    def bTest(self, bState, v, p):
        path, violations = self.getViols(bState, v, p)
        success = bool(violations and violations.empty())

        # Test the other way to be sure we are consistent
        # (preConf, ppConf, hand, obj, pose, poseVar, poseDelta, poseFace,
        #   graspFace, graspMu, graspVar, graspDelta,
        #   opType, inconds) = self.args

        # newBS = bState.pbs.copy().updateFromGoalPoses(inconds, permShadows=True)
        # world = newBS.getWorld()
        # graspB = ObjGraspB(obj, world.getGraspDesc(obj), graspFace, poseFace,
        #                    PoseD(graspMu, graspVar), delta= graspDelta)
        # placeB = ObjPlaceB(obj, world.getFaceFrames(obj), poseFace,
        #                    PoseD(pose, poseVar), delta=poseDelta)
        # violPPTest = canPickPlaceTest(newBS, preConf, ppConf, hand,
        #                               graspB, placeB, p, op=opType)


        # LPK!!! Fix the following if we put it back in
        # testEq = (violations == violPPTest or \
        #   (violations and violPPTest and \
        #    set([x.name() for x in violations.obstacles]) == \
        #      set([x.name() for x in violPPTest.obstacles]) and \
        #    set([x.name() for x in violations.shadows]) == \
        #      set([x.name() for x in violPPTest.shadows])))

        # if not testEq:
        #     print 'Drawing newBS'
        #     newBS.draw(p, 'W')
            
        #     print 'Mismatch in canPickPlaceTest!!'
        #     print 'From the fluent', violations
        #     print 'From the test', violPPTest
        #     raw_input('okay?')
        
        
        # If fluent is false but would have been true without
        # conditioning, raise a flag
        # if not success:
        #     # this fluent, but with no conditions
        #     sc = self.copy()
        #     sc.args[-1] = tuple()
        #     sc.update()
        #     p2, v2 = sc.getViols(bState, v, p)
        #     if bool(p2 and v2.empty()):
        #         print 'CanPickPlace fluent made false by conditions'
        #         print self
        #         raw_input('go?')
        #     elif len(self.args[-1]) > 0:
        #         print 'CanPickPlace fluent false'
        #         print self
        #         #raw_input('go?')
        return success

    def heuristicVal(self, details, v, p):
        # Return cost estimate and a set of dummy operations
        if not self.isGround():
            # assume an obstacle, if we're asking.  May need to decrease this
            dummyOp = Operator('RemoveObst', ['dummy'],{},[])
            dummyOp.instanceCost = graspObstCost
            return graspObstCost, ActSet([dummyOp])

        path, violations = self.getViols(details, v, p)
        totalCost, ops = hCost(violations, details)

        tr('hv', ('Heuristic val', self.predicate),
           ('ops', ops), 'cost', totalCost)
        return totalCost, ops


    def prettyString(self, eq = True, state = None, heuristic = None,
                     includeValue = True):
        (preConf, ppConf, hand, obj, pose, poseVar, poseDelta, poseFace,
          face, graspMu, graspVar, graspDelta,
          op, conds) = self.args
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


class CanPush(Fluent):
    predicate = 'CanPush'
    implicit = True
    conditional = True

    def conditionOn(self, f):
        preds = ('Pose', 'SupportFace', 'Holding', 'Grasp', 'GraspFace')
        return (f.predicate in preds and not ('*' in f.args)) or \
          (f.predicate in ('B', 'Bd') and self.conditionOn(f.args[0]))

    def feasible(self, bState, v, p):
        path, violations = self.getViols(bState, v, p)
        return violations is not None

    def update(self):
        super(CanPush, self).update()
        self.viols = {}
        self.hviols = {}

    def getViols(self, bState, v, p):
        assert v == True
        assert self.isGround()
        (obj, hand, poseFace, prePose, pose, preConf, pushConf,
         postConf, poseVar, prePoseVar,  poseDelta, cond) = self.args

        key = (hash(bState.pbs), p)
        if not hasattr(self, 'viols'): self.viols = {}
        if not hasattr(self, 'hviols'): self.hviols = {}
        if key in self.viols: return self.viols[key]
        if glob.inHeuristic and key in self.hviols: return self.hviols[key]

        newPBS = bState.pbs.conditioned([], cond)
        path, violations = canPush(newPBS, obj, hand, poseFace, prePose, pose,
                                   preConf, pushConf, postConf, poseVar,
                                   prePoseVar, poseDelta, p, Violations())
        debugMsg('CanPush', ('pose', pose), ('key', key), ('->', violations))
        if glob.inHeuristic:
            self.hviols[key] = path, violations
        else:
            self.viols[key] = path, violations
        return path, violations

    def bTest(self, bState, v, p):
        path, violations = self.getViols(bState, v, p)
        return bool(violations and violations.empty())

    def heuristicVal(self, details, v, p):
        # Return cost estimate and a set of dummy operations
        if not self.isGround():
            # assume an obstacle, if we're asking.  May need to decrease this
            dummyOp = Operator('RemoveObst', ['dummy'],{},[])
            dummyOp.instanceCost = graspObstCost
            return graspObstCost, ActSet([dummyOp])

        path, violations = self.getViols(details, v, p)
        totalCost, ops = hCost(violations, details)

        tr('hv', ('Heuristic val', self.predicate),
           ('ops', ops), ('cost', totalCost))
        return totalCost, ops


    # TODO : LPK Can this be shared among CanX fluents?
    def prettyString(self, eq = True, state = None, heuristic = None,
                     includeValue = True):
        (obj, hand, poseFace, prePose, pose,
         preConf, pushConf, postConf, poseVar,
         prePoseVar, poseDelta, cond) = self.args
        assert obj != 'none'

        condStr = self.args[-1] if isVar(self.args[-1]) else \
          str([innerPred(c) for c in self.args[-1]]) 

        argStr = prettyString(self.args) if eq else \
                  prettyString([obj, hand, pose, condStr], eq)
        valueStr = ' = ' + prettyString(self.value) if includeValue else ''
        return self.predicate + ' ' + argStr + valueStr

        
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
                hand = 'right'
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

        return bState.poseModeDist(obj, face)


    def fglb(self, other, bState = None):
        if (other.predicate == 'Holding' and self.args[0] == other.value) or \
           (other.predicate in ('Grasp', 'GraspFace') and
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
        if (other.predicate == 'Holding' and self.args[0] == other.value) or \
                (other.predicate in ('Grasp', 'GraspFace') and
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
        preds = ('Pose', 'SupportFace', 'Holding', 'Grasp', 'GraspFace')
        return (f.predicate in preds and not ('*' in f.args)) or \
          (f.predicate in ('B', 'Bd') and self.conditionOn(f.args[0]))

    def feasible(self, bState, v, p):
        path, violations = self.getViols(bState, v, p)
        return violations is not None

    def bTest(self, details, v, p):
        assert v == True
        ans, occluders = self.getViols(details, v, p, strict=True)
        return ans

    def update(self):
        super(CanSeeFrom, self).update()
        self.viols = {}
        self.hviols = {}

    def getViols(self, bState, v, p, strict=True):
        assert v == True
        (obj, pose, poseFace, conf, cond) = self.args
        key = (hash(bState.pbs), p)
        if not hasattr(self, 'viols'): self.viols = {}
        if not hasattr(self, 'hviols'): self.hviols = {}
        if key in self.viols: return self.viols[key]
        if glob.inHeuristic and key in self.hviols: return self.hviols[key]

        pbs = bState.pbs
        if pose == '*' and \
          (pbs.getHeld('left').mode() == obj or \
           pbs.getHeld('right').mode() == obj):
            # Can't see it (in the usual way) if it's in the hand and a pose
            # isn't specified
            (ans, occluders) = False, None
        else:
            # All object poses are permanent, no collisions can be ignored
            newPBS = bState.pbs.copy()
            newPBS.updateFromGoalPoses(cond, permShadows=True)
            placeB = newPBS.getPlaceB(obj)
            # LPK! Forcing the variance to be very small.  Currently it's
            # using variance from the initial state, and then overriding
            # it based on conditions.  This is incoherent.  Could change
            # it to put variance explicitly in the fluent.
            placeB = placeB.modifyPoseD(var = (0.0001, 0.0001, 0.0001, 0.0005))
            if placeB.support.mode() != poseFace and poseFace != '*':
                placeB.support = DeltaDist(poseFace)
            if placeB.poseD.mode() != pose and pose != '*':
                placeB = placeB.modifyPoseD(mu=pose)
            newPBS.updatePermObjPose(placeB)
            newPBS.reset()   # recompute shadow world
            shWorld = newPBS.getShadowWorld(p)
            shName = shadowName(obj)
            sh = shWorld.objectShapes[shName]
            fixed = []
            obstacles = [s for s in shWorld.getNonShadowShapes() if \
                        s.name() != obj ] + \
                        [conf.placement(shWorld.attached)]
            if strict:
                fixed = obstacles
                obstacles = []
            ans, occluders = visible(shWorld, conf, sh, obstacles,
                                     p, moveHead=False, fixed=fixed)
            debugMsg('CanSeeFrom',
                    ('obj', obj, pose), ('conf', conf),
                    ('->', ans, 'occluders', occluders))
        if glob.inHeuristic:
            self.hviols[key] = ans, occluders
        else:
            self.viols[key] = ans, occluders
        debugMsg('CanSee', 'ans', ans, 'occluders', occluders)
        return ans, occluders
    
    def heuristicVal(self, details, v, p):
        # Return cost estimate and a set of dummy operations
        (obj, pose, poseFace, conf, cond) = self.args
        
        if not self.isGround():
            # assume an obstacle, if we're asking.  May need to decrease this
            dummyOp = Operator('RemoveObst', ['dummy'],{},[])
            dummyOp.instanceCost = graspObstCost
            return graspOCost, ActSet([dummyOp])
        
        vis, occluders = self.getViols(details, v, p)
        totalCost, ops = hCostSee(vis, occluders, details)
        tr('hv', ('Heuristic val', self.predicate),
           ('ops', ops), 'cost', totalCost)
        return totalCost, ops

    def prettyString(self, eq = True, includeValue = True):
        (obj, pose, poseFace, conf, cond) = self.args

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
# the base.  
def canReachNB(pbs, startConf, conf, prob, initViol,
               optimize = False):
    # canReachHome goes towards its homeConf arg, that is it's destination.
    return canReachHome(pbs, startConf, prob, initViol,
                        homeConf=conf, moveBase=False,
                        optimize=optimize) 

# returns a path (conf -> homeConf (usually home))
def canReachHome(pbs, conf, prob, initViol, homeConf = None, reversePath = False,
                 optimize = False, moveBase = True):
    rm = pbs.getRoadMap()
    if not homeConf: homeConf = rm.homeConf
    robot = pbs.getRobot()
    tag = 'canReachHome' if moveBase else 'canReachNB'
    viol, cost, path = rm.confReachViol(conf, pbs, prob, initViol,
                                        startConf=homeConf,
                                        reversePath = reversePath,
                                        moveBase = moveBase,
                                        optimize = optimize)

    if path:
        assert path[0] == conf, 'Start of path'
        assert path[-1] == homeConf, 'End of path'

    if viol is None or viol.weight() > 0:
        # Don't log the "trivial" ones...
        tr('CRH', '%s h=%s'%(tag, glob.inHeuristic) + \
           ' viol=%s'%(viol.weight() if viol else None))

    if path and debug('backwards'):
        backSteps = []
        # unless reversePath is True, the direction of motion is
        # "backwards", that is, from conf to home.
        for i, c in enumerate(path[::-1] if reversePath else path ):
            if i == 0: continue
            # that is, moving from i to i-1 should be forwards (valid)
            if not validEdgeTest(c['pr2Base'], path[i-1]['pr2Base']):
                backSteps.append((c['pr2Base'], path[i-1]['pr2Base']))
        if backSteps:
            for (pre, post) in backSteps:
                trAlways('Backward step:', pre, '->', post, ol = True,
                         pause = False)
            # raw_input('CRH - Backwards steps')

        if debug('canReachHome'):
            pbs.draw(prob, 'W')
            if path:
                drawPath(path, viol=viol,
                         attached=pbs.getShadowWorld(prob).attached)
        tr('canReachHome', ('viol', viol), ('cost', cost), ('path', path),
               snap = ['W'])

    if not viol and debug('canReachHome'):
        pbs.draw(prob, 'W')
        conf.draw('W', 'blue', attached=pbs.getShadowWorld(prob).attached)
        raw_input('CRH Failed')

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

    ans = any([np.all(np.all(np.dot(r.planes(), shadow.prim().vertices()) <= tiny, axis=1)) \
               for r in region.parts()])

    tr('testVerbose', 'In test, shadow in orange, region in purple',
       (shadow, region, ans), draw = [(bState.pbs, prob, 'W'),
                                      (shadow, 'W', 'orange'),
                                      (region, 'W', 'purple')], snap=['W'])
    return ans

# Pushing                

# returns path, violations
pushStepSize = 0.01
def canPush(pbs, obj, hand, poseFace, prePose, pose,
            preConf, pushConf, postConf, poseVar, prePoseVar,
            poseDelta, prob, initViol, prim=False):
    tag = 'canPush'
    held = pbs.held[hand].mode()
    newBS = pbs.copy()
    if held != 'none':
        tr(tag, 'Hand=%s is holding %s in pbs'%(hand, held))
        newBS.updateHeld('none', None, None, hand, None)
    if obj in [h.mode() for h in newBS.held.values()]:
        tr(tag, '=> obj is in the other hand')
        # LPK!! Changed hand below to otherHand(hand)
        assert pbs.held[otherHand(hand)].mode() == obj
        newBS.updateHeld('none', None, None, otherHand(hand), None)
    post = hu.Pose(*pose)
    placeB = ObjPlaceB(obj, pbs.getWorld().getFaceFrames(obj), poseFace,
                       PoseD(post, poseVar), poseDelta)
    # graspB - from hand and objFrame
    graspB = pushGraspB(newBS, pushConf, hand, placeB)
    pathViols, reason = pushPath(newBS, prob, graspB, placeB, pushConf,
                                 prePose, preConf, None, hand)
    if not pathViols or reason != 'done':
        tr(tag, 'pushPath failed')
        return None, None
    viol = pathViols[0][1]
    path = []
    for (c, v, _) in pathViols:
        viol = viol.update(v)
        if viol is None:
            return None, None
        path.append(c)
    if held != 'none':
        # if we had something in the hand indicate a collision
        shape = placeB.shape(pbs.getWorld())
        heldColl = ([shape],[]) if hand=='left' else ([],[shape])
        viol.update(Violations([],[],heldColl,([],[])))
    tr(tag, 'path=%s, viol=%s'%(path, viol))
    return path, viol

def pushGraspB(pbs, pushConf, hand, placeB):
    obj = placeB.obj
    pushWrist = robotGraspFrame(pbs, pushConf, hand)
    objFrame = placeB.objFrame()
    support = placeB.support.mode()
    # TODO: what should these values be?
    graspVar = 4*(0.01**2,)
    graspDelta = 4*(0.0,)
    graspFrame = objFrame.inverse().compose(pushWrist.compose(gripperFaceFrame[hand]))
    graspDescList = [GDesc(obj, graspFrame, 0.0, 0.0, 0.0)]
    graspDescFrame = objFrame.compose(graspDescList[-1].frame)
    graspB =  ObjGraspB(obj, graspDescList, -1, support,
                        PoseD(hu.Pose(0.,0.,0.,0), graspVar), delta=graspDelta)
    return graspB

# TODO: Straighten out the mess with the imports
# Duplicated from pr2GenAux - we can't import it since that imports this.
def robotGraspFrame(pbs, conf, hand):
    robot = pbs.getRobot()
    _, frames = robot.placement(conf, getShapes=[])
    wristFrame = frames[robot.wristFrameNames[hand]]
    if debug('robotGraspFrame'):
        print 'robot wristFrame\n', wristFrame
    return wristFrame

pushPathCacheStats = [0, 0]
pushPathCache = {}

if glob.useHandTiltForPush:
    handTiltOffset = 0.0375                 # 0.18*math.sin(math.pi/15)
    # handTiltOffset = 0.0560                 # 0.18*math.sin(math.pi/10)
else:
    handTiltOffset = 0.0

# return None (if no useful cache hit) or the cached ans
def checkPushPathCache(key, names,  pbs, prob, gB, conf, newBS):
    tag = 'pushPath'
    pushPathCacheStats[0] += 1
    val = pushPathCache.get(key, None)
    if val is not None:
        for v in val:
            (bs, p, gB1, ans) = v
            if bs == pbs and p >= prob and gB == gB1:
                if debug(tag): print tag, gB.obj, 'cached ->', ans[-1]
                pushPathCacheStats[1] += 1
                return ans
        replay = checkReplay(newBS, prob, val)
        if replay:
            if debug(tag): print tag, 'cached replay ->', replay[-1]
            pushPathCacheStats[1] += 1
            return replay
    else:
        tr(tag, 'pushPath cache did not hit')
        if debug(tag):
            print '-----------'
            conf.prettyPrint()
            for n, x in zip(names, key): print n, x
            print '-----------'
        pushPathCache[key] = []

# preConf = approach conf, before contact
# initPose = object pose before push
# initConf = conf at initPose
# pushConf = conf at the end of push
# pushPose = pose at end of push
# returns (appDir, appDist, pushDir, pushDist)
def pushDirections(preConf, initConf, initPose, pushConf, pushPose, hand):
    # Approach dir and dist
    handFrameName = preConf.robot.armChainNames[hand]
    preWrist = preConf.cartConf()[handFrameName]
    initWrist = initConf.cartConf()[handFrameName]
    appDir = (initWrist.point().matrix.reshape(4) - preWrist.point().matrix.reshape(4))[:3]
    appDir[2] = 0.0
    appDist = (appDir[0]**2 + appDir[1]**2)**0.5 # xy app distance
    if appDist != 0:
        appDir /= appDist
    appDist -= handTiltOffset     # the tilt reduces the approach dist
    # Push dir and dist
    pushDir = (pushPose.point().matrix.reshape(4) - initPose.point().matrix.reshape(4))[:3]
    pushDir[2] = 0.0
    pushDist = (pushDir[0]**2 + pushDir[1]**2)**0.5 # xy push distance
    if pushDist != 0:
        pushDir /= pushDist
    pushDist -= handTiltOffset     # the tilt reduces the push dist
    # Return
    return (appDir, appDist, pushDir, pushDist)

# The conf in the input is the robot conf in contact with the object
# at the destination pose.

def pushPath(pbs, prob, gB, pB, conf, initPose, preConf, regShape, hand,
             reachObsts=[]):
    tag = 'pushPath'
    def finish(reason, gloss, safePathViols=[], cache=True):
        if debug(tag):
            for (ig, obst) in reachObsts: obst.draw('W', 'orange')
            print '->', reason, gloss, 'path len=', len(safePathViols)
            if pause(tag): raw_input(reason)
        ans = (safePathViols, reason)
        if cache: pushPathCache[key].append((pbs, prob, gB, ans))
        return ans
    #######################
    # Preliminaries
    #######################
    initPose = hu.Pose(*initPose) if isinstance(initPose, (tuple, list)) else initPose
    postPose = pB.poseD.mode()
    # Create pbs in which the object is grasped
    newBS = pbs.copy().updateHeldBel(gB, hand)
    # Check cache and return it if appropriate
    baseSig = "%.6f, %.6f, %.6f"%tuple(conf['pr2Base'])
    key = (postPose, baseSig, initPose, hand, frozenset(reachObsts),
           glob.pushBuffer, glob.inHeuristic)
    names =('postPose', 'base', 'initPose', 'hand', 'reachObsts', 'pushBuffer', 'inHeuristic')
    cached = checkPushPathCache(key, names, pbs, prob, gB, conf, newBS)
    if cached is not None:
        safePathViols, reason = cached
        return finish(reason, 'Cached answer', safePathViols, cache=False)
    if debug(tag): newBS.draw(prob, 'W'); raw_input('pushPath: Go?')
    # Check there is no permanent collision at the goal
    viol = newBS.confViolations(conf, prob)
    if not viol:
        return finish('collide', 'Final conf collides in pushPath')
    #######################
    # Set up scan parameters, directions, steps, etc.
    #######################
    # initConf is for initial contact at initPose
    initConf = displaceHandRot(conf, hand, initPose.compose(postPose.inverse()))
    if not initConf:
        return finish('invkin', 'No invkin at initial contact')
    # the approach and push directions (and distances)
    appDir, appDist, pushDir, pushDist = \
            pushDirections(preConf, initConf, initPose, conf, postPose, hand)
    if pushDist == 0:
        return finish('dist=0', 'Push distance = 0')
    # Find tilt, if any, for hand given the direction
    tiltRot = handTilt(preConf, hand, appDir)
    if tiltRot is None:
        return finish('bad dir', 'Illegal hand orientation')
    # rotation angle (if any) - can only do small ones (if we're lucky)
    angleDiff = hu.angleDiff(postPose.theta, initPose.theta)
    if debug(tag): print 'angleDiff', angleDiff
    if abs(angleDiff) > math.pi/6 or \
           (abs(angleDiff) > 0.1 and pushDist < 0.02):
        return finish('tilt', 'Angle too large for pushing')
    # The minimal shadow (or just shape if we don't have it (why not?)
    shape = pbs.getPlaceB(pB.obj).shadow(pbs.getShadowWorld(0.0)) or \
            pbs.getPlaceB(pB.obj).shape(pbs.getWorld())
    assert shape
    if debug(tag):
        offPose = postPose.inverse().compose(initPose)
        shape.draw('W', 'pink'); shape.applyTrans(offPose).draw('W', 'blue')
        conf.draw('W', 'pink'); preConf.draw('W', 'blue'); raw_input('Go?')
    #######################
    # Set up state for the combined scans
    #######################
    # We will return (conf, viol, pose) for steps along the path --
    # starting at initPose.  Before contact, pose in None.
    pathViols = []
    safePathViols = []
    reason = 'done'                     # done indicates success
    #######################
    # Set up state for the approach scan
    #######################
    # Number of steps for approach displacement
    nsteps = int(appDist / pushStepSize)
    delta = appDist / nsteps
    stepVals = [0, nsteps-1] if glob.inHeuristic else xrange(nsteps)
    #######################
    # Do the approach scan
    #######################
    for step_i in stepVals:
        step = (step_i * delta)
        hOffsetPose = hu.Pose(*((step*appDir).tolist()+[0.0]))
        nconf = displaceHandRot(preConf, hand, hOffsetPose, tiltRot = tiltRot)
        if not nconf:
            reason = 'invkin'; break
        viol = newBS.confViolations(nconf, prob)
        if viol is None:
            reason = 'collide'; break
        if armCollides(nconf, shape, hand):
            reason = 'selfCollide'; break
        if debug('pushPath'):
            print 'approach step=', step, viol
            drawState(newBS, prob, nconf, shape, reachObsts)
        pathViols.append((nconf, viol, None))
    if reason != 'done':
        return finish(reason, 'During approach', [])
    #######################
    # Set up state for the push scan
    #######################
    # Number of steps for approach displacement
    nsteps = max(int(pushDist / pushStepSize), 1)
    delta = pushDist / nsteps
    if angleDiff == 0 or pushDist < pushStepSize:
        deltaAngle = 0.0
    else:
        deltaAngle = angleDiff / nsteps
    if nsteps > 1:
        stepVals = [0, nsteps-1] if glob.inHeuristic else xrange(nsteps)
    else:
        stepVals = [0]
    if debug(tag): 
        print 'nsteps=', nsteps, 'delta=', delta, 'deltaAngle', deltaAngle
    handFrameName = conf.robot.armChainNames[hand]
    #######################
    # Do the push scan
    #######################
    for step_i in stepVals:
        step = (step_i * delta)
        hOffsetPose = hu.Pose(*((step*pushDir).tolist()+[0.0]))
        nconf = displaceHandRot(initConf, hand, hOffsetPose,
                                tiltRot = tiltRot, angle=step*deltaAngle)
        if not nconf:
            reason = 'invkin'; break
        if step_i == nsteps-1:
            nconf = conf
        viol = newBS.confViolations(nconf, prob)
        if viol is None:
            reason = 'collide'; break
        offsetPose = hu.Pose(*(step*pushDir).tolist()+[0.0])
        offsetRot = hu.Pose(0.,0.,0.,step*deltaAngle)
        newPose = offsetPose.compose(initPose).compose(offsetRot).pose()
        if debug('pushPath'):
            print step_i, 'newPose:', newPose
            print step_i, 'nconf point', nconf.cartConf()[handFrameName].point()
        offsetPB = pB.modifyPoseD(newPose, var=4*(0.01**2,))
        offsetPB.delta=4*(0.001,)
        nshape = offsetPB.makeShadow(pbs, prob)
        if regShape and not inside(nshape, regShape):
            reason = 'outside'; break
        if armCollides(nconf, nshape, hand):
            reason = 'selfCollide'; break
        if debug('pushPath'):
            print 'push step=', step, viol
            drawState(newBS, prob, nconf, nshape, reachObsts)
        pathViols.append((nconf, viol, offsetPB.poseD.mode()))
        if all(not nshape.collides(obst) for (ig, obst) in reachObsts \
               if (pB.obj not in ig)):
            safePathViols = list(pathViols)
    #######################
    # Prepare ans
    #######################
    if not safePathViols:
        reason = 'reachObst collision'
    return finish(reason, 'Final pushPath', safePathViols)

def armCollides(conf, objShape, hand):
    armShape = conf.placement()
    parts = dict([(part.name(), part) for part in armShape.parts()])
    gripperName = conf.robot.gripperChainNames[hand]
    return any(objShape.collides(parts[name]) for name in parts if name != gripperName)

def drawState(pbs, prob, conf, shape=None, reachObsts=[]):
    shWorld = pbs.getShadowWorld(prob)
    attached = shWorld.attached
    if glob.useMathematica:
        wm.getWindow('W').startCapture()
    pbs.draw(prob, 'W')
    conf.draw('W', 'green', attached)
    if shape: shape.draw('W', 'blue')
    wm.getWindow('W').update()
    if glob.useMathematica:
        mathematica.mathFile(wm.getWindow('W').stopCapture(),
                             view = "ViewPoint -> {2, 0, 2}",
                             filenameOut='./pushPath.m')
        
def displaceHandRot(conf, hand, offsetPose, nearTo=None, tiltRot=None, angle=0.0):
    cart = conf.cartConf()
    handFrameName = conf.robot.armChainNames[hand]
    trans = cart[handFrameName]         # initial hand position
    # wrist x points down, so we negate angle to get rotation around z.
    xrot = hu.Transform(rotation_matrix(-angle, (1,0,0)))
    nTrans = offsetPose.compose(trans).compose(xrot) # final hand position
    if tiltRot and trans.matrix[2,0] < -0.9:     # rot and vertical (wrist x along -z)
        nTrans = nTrans.compose(tiltRot)
    nCart = cart.set(handFrameName, nTrans)
    nConf = conf.robot.inverseKin(nCart, conf=(nearTo or conf)) # use conf to resolve
    if all(nConf.values()):
        assert all(conf[g] == nConf[g] for g in ('pr2LeftGripper', 'pr2RightGripper'))
        return nConf
    
def handTilt(conf, hand, direction):
    cart = conf.cartConf()
    handFrameName = conf.robot.armChainNames[hand]
    trans = cart[handFrameName]
    # Horizontal hand orientation
    if trans.matrix[2,0] >= -0.9:       
        # x axis of wrist points along hand, we don't need tilt
        return hu.Pose(0,0,0,0)
    # Rest is for vertical hand orientation
    transInv = trans.inverse()
    transInvMat = transInv.matrix
    handDir = np.dot(transInvMat, np.hstack([direction, np.array([0.0])]).reshape(4,1))
    if abs(handDir[2,0]) > 0.7:
        sign = -1.0 if handDir[2,0] < 0 else 1.0
        # Because of the wrist orientation, the sign is negative
        if glob.useHandTiltForPush:
            # Tilting the hand causes a discontinuity at the end of the push
            return hu.Transform(rotation_matrix(-sign*math.pi/15., (0,1,0)))
        else:
            return hu.Pose(0,0,0,0)
    else:
        if debug('pushPath'):
            print 'Bad direction relative to hand'
        return None

def checkReplay(pbs, prob, cachedValues):
    doneVals = [val for (bs, p, gB, val) in cachedValues if val[-1] == 'done']
    for (pathViols, reason) in doneVals:
        viol = [pbs.confViolations(conf, prob) for (conf, _, _) in pathViols]
        if all(viol):
            return ([(c, v2, p) for ((c, v1, p), v2) in zip(pathViols, viol)], 'done')
