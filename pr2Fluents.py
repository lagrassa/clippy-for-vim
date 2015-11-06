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
from traceFile import tr, trAlways
import mathematica
import windowManager3D as wm
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
        lObj =  bState.pbs.getHeld('left')
        rObj =  bState.pbs.getHeld('right')
        if obj in (lObj, rObj):
            return False
        else: 
            return inTest(bState.pbs, obj, region, p)

    
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
        return confWithin(bState.pbs.getConf(), targetConf, delta)

    def getGrounding(self, bstate):
        assert self.value == True
        (targetConf, delta) = self.args
        assert not isGround(targetConf)
        return {targetConf : bstate.details.pbs.getConf()}

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
        return baseConfWithin(bState.pbs.getConf()['pr2Base'], targetConf, delta)

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
    def getViols(self, pbs, v, p):
        assert v == True
        (conf, cond) = self.args

        key = (hash(pbs), p)
        if not hasattr(self, 'viols'): self.viols = {}
        if not hasattr(self, 'hviols'): self.hviols = {}
        if key in self.viols: return self.viols[key]
        if glob.inHeuristic and key in self.hviols: return self.hviols[key]

        newPBS = pbs.conditioned([], cond)
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
        path, violations = self.getViols(bState.pbs, v, p)

        return bool(path and violations.empty())

    def feasible(self, bState, v, p):
        return self.feasiblePBS(bState.pbs, v, p)

    def feasiblePBS(self, pbs, v, p):
        path, violations = self.getViols(pbs, v, p)
        return violations != None

    def heuristicVal(self, details, v, p):
        # Return cost estimate and a set of dummy operations
        (conf, cond) = self.args

        if not self.isGround():
            # assume an obstacle, if we're asking.  May need to decrease this
            dummyOp = Operator('RemoveObst', ['dummy'],{},[])
            dummyOp.instanceCost = graspObstCost
            return (graspObstCost, ActSet([dummyOp]))
        
        path, violations = self.getViols(details.pbs, v, p)

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

    def feasible(self, bState, v, p):
        return self.feasiblePBS(bState.pbs, v, p)

    def feasiblePBS(self, bState, v, p):
        if not self.isGround():
            (startConf, endConf, cond) = self.args
            assert isGround(endConf) and isGround(cond)
            path, violations = CanReachNB([endConf, endConf, cond], True).\
                                    getViols(pbs, v, p)
        else:
            path, violations = self.getViols(pbs, v, p)
        return violations is not  None

    def update(self):
        super(CanReachNB, self).update()
        self.viols = {}
        self.hviols = {}

    def getViols(self, pbs, v, p):
        assert v == True
        (startConf, endConf, cond) = self.args
        key = (hash(pbs), p)
        if not hasattr(self, 'viols'): self.viols = {}
        if not hasattr(self, 'hviols'): self.hviols = {}
        if key in self.viols: return self.viols[key]
        if glob.inHeuristic and key in self.hviols: return self.hviols[key]

        newPBS = pbs.copy()
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
        
        path, violations = self.getViols(bState.pbs, v, p)

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
            
        path, violations = self.getViols(details.pbs, v, p)

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
        return self.feasiblePBS(bState.pbs, v, p)
    
    def feasiblePBS(self, pbs, v, p):
        path, violations = self.getViols(pbs, v, p)
        return violations is not None

    # Add a glb method that will at least return False, {} if the two are
    # in contradiction.  How to test, exactly?

    # Override the default version of this so that the component conds
    # will be recalculated
    def addConditions(self, newConds, details = None):
        self.conds = None
        Fluent.addConditions(self, newConds, details)

    def getConds(self, pbs):
        # Will recompute if new bindings are applied because the result
        # won't have this attribute
        (preConf, ppConf, hand, obj, pose, poseVar, poseDelta, poseFace,
          graspFace, graspMu, graspVar, graspDelta,
          opType, inconds) = self.args

        pbs.getRoadMap().approachConfs[ppConf] = preConf
        assert obj != 'none'
        tinyDelta = zeroDelta

        if not hasattr(self, 'conds') or self.conds is None:
            placeVar = pbs.domainProbs.placeVar
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
            for c in self.conds: c.addConditions(inconds)
        return self.conds


    def update(self):
        super(CanPickPlace, self).update()
        self.viols = {}
        self.hviols = {}

    def getViols(self, pbs, v, p):
        def violCombo(v1, v2):
            return v1.update(v2)

        key = (hash(pbs), p)
        if not hasattr(self, 'viols'): self.viols = {}
        if not hasattr(self, 'hviols'): self.hviols = {}
        if key in self.viols: return self.viols[key]
        if glob.inHeuristic and key in self.hviols: return self.hviols[key]
            
        condViols = [c.getViols(pbs, v, p) for c in self.getConds(pbs)]

        if debug('CanPickPlace'):
            print 'canPickPlace getViols'
            for (cond, (p, viol)) in zip(self.getConds(pbs), condViols):
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
        path, violations = self.getViols(bState.pbs, v, p)
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
        #     p2, v2 = sc.getViols(bState.pbs, v, p)
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

        path, violations = self.getViols(details.pbs, v, p)
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
        return DeltaDist(bState.pbs.getHeld(hand))

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
        return self.feasiblePBS(bState.pbs, v, p)

    def feasiblePBS(self, pbs, v, p):
        path, violations = self.getViols(pbs, v, p)
        return violations is not None

    def update(self):
        super(CanPush, self).update()
        self.viols = {}
        self.hviols = {}

    def getViols(self, pbs, v, p):
        assert v == True
        assert self.isGround()
        (obj, hand, poseFace, prePose, pose, preConf, pushConf,
         postConf, poseVar, prePoseVar,  poseDelta, cond) = self.args

        key = (hash(pbs), p)
        if not hasattr(self, 'viols'): self.viols = {}
        if not hasattr(self, 'hviols'): self.hviols = {}
        if key in self.viols: return self.viols[key]
        if glob.inHeuristic and key in self.hviols: return self.hviols[key]

        newPBS = pbs.conditioned([], cond)
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
        path, violations = self.getViols(bState.pbs, v, p)
        return bool(violations and violations.empty())

    def heuristicVal(self, details, v, p):
        # Return cost estimate and a set of dummy operations
        if not self.isGround():
            # assume an obstacle, if we're asking.  May need to decrease this
            dummyOp = Operator('RemoveObst', ['dummy'],{},[])
            dummyOp.instanceCost = graspObstCost
            return graspObstCost, ActSet([dummyOp])

        path, violations = self.getViols(details.pbs, v, p)
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
            return bState.pbs.getGraspBForObj(obj, hand).grasp # a DDist over integers

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
            hl = bState.pbs.getHeld('left')
            hr = bState.pbs.getHeld('right')
            if obj == hl:
                hand = 'left'
            elif obj == hr:
                hand = 'right'
            else:
                # We don't think it's in the hand, so dist is huge
                return lostDist
        if face == '*':
            face = bState.pbs.getGraspBForObj(obj, hand).grasp.mode()

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
            hl = bState.pbs.getHeld('left')
            hr = bState.pbs.getHeld('right')
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
        hl = bState.pbs.getHeld('left')
        hr = bState.pbs.getHeld('right')
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
        return self.feasiblePBS(bState.pbs, v, p)
    
    def feasiblePBS(self, pbs, v, p):
        path, violations = self.getViols(pbs, v, p)
        return violations is not None

    def bTest(self, details, v, p):
        assert v == True
        ans, occluders = self.getViols(details.pbs, v, p, strict=True)
        return ans

    def update(self):
        super(CanSeeFrom, self).update()
        self.viols = {}
        self.hviols = {}

    def getViols(self, pbs, v, p, strict=True):
        assert v == True
        (obj, pose, poseFace, conf, cond) = self.args
        key = (hash(pbs), p)
        if not hasattr(self, 'viols'): self.viols = {}
        if not hasattr(self, 'hviols'): self.hviols = {}
        if key in self.viols: return self.viols[key]
        if glob.inHeuristic and key in self.hviols: return self.hviols[key]

        if pose == '*' and \
          (pbs.getHeld('left') == obj or \
           pbs.getHeld('right') == obj):
            # Can't see it (in the usual way) if it's in the hand and a pose
            # isn't specified
            (ans, occluders) = False, None
        else:
            # All object poses are permanent, no collisions can be ignored
            newPBS = pbs.copy()
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
            newPBS.updatePermObjBel(placeB)
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
        
        vis, occluders = self.getViols(details.pbs, v, p)
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
# analysis, but by hand for now.

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
