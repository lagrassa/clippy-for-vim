import numpy as np
import math
import random
import util
import copy
import time
import windowManager3D as wm
import planGlobals as glob
from planGlobals import debugMsg, debugMsgSkip, debugDraw, debug, pause, torsoZ
from traceFile import tr
from miscUtil import isVar, argmax, isGround, tuplify, roundrobin
from dist import DeltaDist, UniformDist
from pr2Robot import CartConf
from planUtil import PoseD, ObjGraspB, ObjPlaceB, Violations
from pr2Util import shadowName, objectName, NextColor, supportFaceIndex, Memoizer, shadowWidths
import fbch
from belief import Bd
from pr2Fluents import CanReachHome, canReachHome, inTest
from pr2Visible import visible, lookAtConf
from pr2PlanBel import getConf, getGoalPoseBels

from shapes import Box

Ident = util.Transform(np.eye(4))            # identity transform

import pr2GenAux
from pr2GenAux import *

#  How many candidates to generate at a time...  Larger numbers will
#  generally lead to better solutions.
pickPlaceBatchSize = 3

easyGraspGenCacheStats = [0,0]

def easyGraspGen(args, goalConds, bState, outBindings):
    tag = 'easyGraspGen'
    graspVar = 4*(0.1,)                # make precondition even weaker
    graspDelta = 4*(0.001,)            # put back to prev value
    
    pbs = bState.pbs.copy()
    (obj, hand) = args
    assert obj != None and obj != 'none'
    tr(tag, 0, '(%s,%s) h=%s'%(obj,hand,glob.inHeuristic))
    if obj == 'none' or (goalConds and getConf(goalConds, None)):
        tr(tag, 1, '=> obj is none or conf in goal conds, failing')
        return
    prob = 0.75
    # Set up pbs
    newBS = pbs.copy()
    # Just placements specified in goal
    newBS = newBS.updateFromGoalPoses(goalConds)
    shWorld = newBS.getShadowWorld(prob)
    if obj == newBS.held[hand].mode():
        gB = newBS.graspB[hand]
        ans = (gB.grasp.mode(), gB.poseD.mode().xyztTuple(), graspVar, graspDelta)
        tr(tag, 1, '(%s,%s)'%(obj, hand)+'=> inHand (g=%s)'%gB.grasp.mode())
        tr(tag, 2, ans)
        yield ans
        return
    if obj == newBS.held[otherHand(hand)].mode():
        tr(tag, 1, 'no easy grasp with this hand, failing')
        return
    rm = newBS.getRoadMap()
    placeB = newBS.getPlaceB(obj)
    graspB = ObjGraspB(obj, pbs.getWorld().getGraspDesc(obj), None,
                       PoseD(None, graspVar), delta=graspDelta)
    cache = pbs.beliefContext.genCaches[tag]
    key = (newBS, placeB, graspB, hand, prob)
    easyGraspGenCacheStats[0] += 1
    val = cache.get(key, None)
    if val != None:
        easyGraspGenCacheStats[1] += 1
        cached = 'Cached'
        memo = val.copy()
    else:
        memo = Memoizer(tag,
                        easyGraspGenAux(newBS, placeB, graspB, hand, prob))
        cache[key] = memo
        cached = ''
    for ans in memo:
        tr(tag, 1, '%s (%s,%s)'%(cached, obj, hand)+ \
           '=> (p=%s,g=%s)'%(placeB.support.mode(), ans[0]))
        tr(tag, 2, 'ans=%s'%(ans,))
        yield ans
    tr(tag, 1, '(%s,%s)='%(obj, hand)+'=> out of values')
    return

def easyGraspGenAux(newBS, placeB, graspB, hand, prob):
    tag = 'easyGraspGen'
    graspVar = 4*(0.001,)
    graspDelta = 4*(0.001,)   # put back to prev value
    
    def graspApproachConfGen(firstConf):
        if firstConf:
            yield firstConf
        for c, ca, _ in graspConfGen:
            approached[ca] = c
            yield ca

    def pickable(ca, c, pB, gB):
        viol, reason = canPickPlaceTest(newBS, ca, c, hand, gB, pB, prob, op='pick')
        return viol

    obj = placeB.obj
    approached = {}
    for gB in graspGen(newBS, obj, graspB):
        tr(tag, 3, 'considering grasp=%s'%gB)
        graspConfGen = potentialGraspConfGen(newBS, placeB, gB, None, hand, None, prob)
        firstConf = next(graspApproachConfGen(None), None)
        if not firstConf:
            tr(tag, 3, 'no confs for grasp = %s'%gB)
            continue
        for ca in graspApproachConfGen(firstConf):
            tr(tag, 3, 'considering conf=%s'%ca.conf)
            if pickable(ca, approached[ca], placeB, gB):
                tr(tag, 3, 'pickable')
                ans = (gB.grasp.mode(), gB.poseD.mode().xyztTuple(),
                       graspVar, graspDelta)
                yield ans
                break
            else:
                tr(tag, 3, 'not pickable')

# R1: Pick bindings that make pre-conditions not inconsistent with goalConds
# R2: Pick bindings so that results do not make conditional fluents in the goalConds infeasible

# Preconditions (for R1):

# 1. Pose(obj) - since there is a Holding fluent in the goal (therefore
# we pick), there cannot be a conflicting Pose fluent

# 2. CanPickPlace(...) - has to be feasible given (permanent)
# placement of objects in the goalConds, but it's ok to violate
# shadows.

# 3. Conf() - if there is Conf in goalConds, then fail.  If there's a
# baseConf in goalConds, then we have to use that base.

# 4. Holding(hand)=none - should not be a different Holding(hand)
# value in goalConds, but there can't be.

# Results (for R2):

# Holding(hand)

def pickGen(args, goalConds, bState, outBindings, onlyCurrent = False):
    (obj, graspFace, graspPose,
     objV, graspV, objDelta, confDelta, graspDelta, hand, prob) = args

    base = sameBase(goalConds)          # base is (x, y, th)
    tr('pickGen', 0, 'obj=%s, base=%s'%(obj, base))
    # tr('pickGen', 2, ('args', args))

    pbs = bState.pbs.copy()
    world = pbs.getWorld()
    graspB = ObjGraspB(obj, world.getGraspDesc(obj), graspFace,
                       PoseD(util.Pose(*graspPose), graspV), delta=graspDelta)
    placeB = ObjPlaceB(obj, world.getFaceFrames(obj), None,
                       PoseD(None,  objV), delta=objDelta)
    for ans, viol in pickGenTop((obj, graspB, placeB, hand, base, prob,),
                             goalConds, pbs, outBindings, onlyCurrent):
        (pB, c, ca) = ans
        yield (pB.poseD.mode().xyztTuple(), pB.support.mode(), c, ca)

def pickGenTop(args, goalConds, pbs, outBindings,
               onlyCurrent = False):
    (obj, graspB, placeB, hand, base, prob) = args
    tag = 'pickGen'
    graspDelta = pbs.domainProbs.pickStdev
    tr(tag, 0, '(%s,%s,%d) b=%s h=%s'%(obj,hand,graspB.grasp.mode(),base,glob.inHeuristic))
    tr(tag, 2, 
       zip(('obj', 'graspB', 'placeB', 'hand', 'prob'), args),
       ('goalConds', goalConds),
       ('moveObjBs', pbs.moveObjBs),
       ('fixObjBs', pbs.fixObjBs),
       ('held', (pbs.held['left'].mode(),
                 pbs.held['right'].mode(),
                 pbs.graspB['left'],
                 pbs.graspB['right'])))

    if obj == 'none':                   # can't pick up 'none'
        tr(tag, 1, '=> cannot pick up none, failing')
        return
    if goalConds:
        if getConf(goalConds, None):
            tr(tag, 1, '=> conf is already specified')
            return
    if obj == pbs.held[hand].mode():
        attachedShape = pbs.getRobot().attachedObj(pbs.getShadowWorld(prob),
                                                   hand)
        shape = pbs.getWorld().getObjectShapeAtOrigin(obj).\
                                             applyLoc(attachedShape.origin())
        sup = supportFaceIndex(shape)
        pose = None
        conf = None
        confAppr = None
        tr(tag, 2, 'Object already in hand, support=%s'%sup)
    elif obj == pbs.held[otherHand(hand)].mode():
        attachedShape = pbs.getRobot().attachedObj(pbs.getShadowWorld(prob),
                                                   otherHand(hand))
        shape = pbs.getWorld().getObjectShapeAtOrigin(obj).\
                                       applyLoc(attachedShape.origin())
        sup = supportFaceIndex(shape)
        pose = None
        conf = None
        confAppr = None
        tr(tag, 2, 'Object already in other hand, support=%s'%sup)
    else:
        # Use placeB from the current state
        pose = pbs.getPlaceB(obj).poseD.mode()
        sup =  pbs.getPlaceB(obj).support.mode()
        conf = None
        confAppr = None
        tr(tag, 2, 'Using current state, support=%s, pose=%s'%(sup, pose.xyztTuple()))
    placeB.poseD = PoseD(pose, placeB.poseD.var) # record the pose
    placeB.support = DeltaDist(sup)              # and supportFace
    tr(tag, 2, 'target placeB=%s'%placeB)
    # Set up pbs
    newBS = pbs.copy()
    # Just placements specified in goal
    newBS = newBS.updateFromGoalPoses(goalConds)
    shWorld = newBS.getShadowWorld(prob)
    tr('pickGen', 2, 'Goal conditions', draw=[(newBS, prob, 'W')], snap=['W'])
    gen = pickGenAux(newBS, obj, confAppr, conf, placeB, graspB, hand, base, prob,
                     goalConds, onlyCurrent=onlyCurrent)
    for x,v in gen:
        (pB, cf, ca) = x
        pose = pB.poseD.mode().xyztTuple() if pB else None
        grasp = graspB.grasp.mode() if graspB else None
        pg = (placeB.support.mode(), grasp)
        w = v.weight() if v else None
        tr(tag, 1, '(%s) viol=%s'%(obj, w)+'=> (h=%s,pg=%s,pose=%s)'%(hand, pg, pose),
           draw=[(newBS, prob, 'W'),
                 (cf, 'W', 'orange', shWorld.attached)],
           snap=['W'])
        yield x,v

def pickGenAux(pbs, obj, confAppr, conf, placeB, graspB, hand, base, prob,
               goalConds, onlyCurrent = False):
    def pickable(ca, c, pB, gB):
        return canPickPlaceTest(pbs, ca, c, hand, gB, pB, prob, op='pick')

    def checkInfeasible(conf):
        newBS = pbs.copy()
        newBS.updateConf(conf)
        newBS.updateHeldBel(graspB, hand)
        viol = rm.confViolations(conf, newBS, prob)
        if not viol:                # was valid when not holding, so...
            tr(tag, 0, 'Held collision', draw=[(newBS, prob, 'W')], snap=['W'])
            return True            # punt.

    def graspApproachConfGen(firstConf):
        if firstConf:
            yield firstConf
        for c, ca, _ in graspConfGen:
            approached[ca] = c
            yield ca

    def currentGraspFeasible():
        wrist = objectGraspFrame(pbs, graspB, placeB, hand)

    tag = 'pickGen'
    shw = shadowWidths(placeB.poseD.var, placeB.delta, prob)
    if any(w > t for (w, t) in zip(shw, pbs.domainProbs.pickTolerance)):
        print 'pickGen shadow widths', shw
        print 'poseVar', placeB.poseD.var
        print 'delta', placeB.delta
        print 'prob', prob
        tr(tag, 0, '=> Shadow widths exceed tolerance in pickGen')
        return
    shWorld = pbs.getShadowWorld(prob)
    approached = {}
    rm = pbs.getRoadMap()
    failureReasons = []
    if placeB.poseD.mode() != None: # otherwise go to regrasp
        if not base:
            # Try current conf
            (x,y,th) = pbs.conf['pr2Base']
            currBasePose = util.Pose(x, y, 0.0, th)
            ans = graspConfForBase(pbs, placeB, graspB, hand, currBasePose, prob)
            if ans:
                (c, ca, viol) = ans
                tr(tag, 1, ('=>', ans))
                yield (placeB, c, ca), viol        
        graspConfGen = potentialGraspConfGen(pbs, placeB, graspB, conf, hand, base, prob)
        firstConf = next(graspApproachConfGen(None), None)
        # This used to have an or clause
        # (firstConf and checkInfeasible(firstConf))
        # but infeasibility of one of the grasp confs due to held
        # object does not guarantee there are no solutions.
        if (not firstConf):
            tr(tag, 1, 'No potential grasp confs, will need to regrasp',
               draw=[(pbs, prob, 'W')], snap=['W'])
            raw_input('need to regrasp')
        else:
            targetConfs = graspApproachConfGen(firstConf)
            batchSize = 1 if glob.inHeuristic else pickPlaceBatchSize
            batch = 0
            while True:
                # Collect the next batch of trialConfs
                batch += 1
                trialConfs = []
                count = 0
                minCost = 1e6
                for ca in targetConfs:       # targetConfs is a generator
                    viol, reason = pickable(ca, approached[ca], placeB, graspB)
                    if viol:
                        trialConfs.append((viol.weight(), viol, ca))
                        minCost = min(viol.weight(), minCost)
                    else:
                        failureReasons.append(reason)
                        tr(tag, 2, 'target conf failed: ' + reason)
                        continue
                    count += 1
                    if count == batchSize or minCost == 0: break
                if count == 0: break
                trialConfs.sort()
                for _, viol, ca in trialConfs:
                    c = approached[ca]
                    ans = (placeB, c, ca)
                    tr(tag, 2,
                       ('currently graspable ->', ans), ('viol', viol),
                       draw=[(pbs, prob, 'W'),
                             (placeB.shape(pbs.getShadowWorld(prob)), 'W', 'navy'),
                             (c, 'W', 'navy', shWorld.attached)],
                       snap=['W'])
                    yield ans, viol
        if onlyCurrent:
            tr(tag, 0, 'onlyCurrent: out of values')
            return
        
    # Try a regrasp... that is place the object somewhere else where it can be grasped.
    if glob.inHeuristic:
        return
    if failureReasons and all(['visibility' in reason for reason in failureReasons]):
        tr(tag, 1, 'There were valid targets that failed due to visibility')
        return
    
    tr(tag, 0, 'Calling for regrasping... h=%s'%glob.inHeuristic)
    raw_input('Regrasp?')
    # !! Needs to look for plausible regions...
    regShapes = regShapes = [shWorld.regionShapes[region] for region in pbs.awayRegions()]
    plGen = placeInGenTop((obj, regShapes, graspB, placeB, None, prob),
                          goalConds, pbs, [],
                          regrasp = True,
                          )
    for pl, viol in plGen:
        (pB, gB, cf, ca) = pl
        v, reason = pickable(ca, cf, pB, gB)
        tr(tag, 2, 'Regrasp pickable=%s'%v,
           draw=[(pbs, prob, 'W'), (cf, 'W', 'blue', shWorld.attached)],
           snap=['W'])
        if v:
            yield (pB, cf, ca), v
    tr(tag, 1, '=> out of values')

# Preconditions (for R1):

# 1. CanPickPlace(...) - has to be feasible given (permanent)
# placement of objects in the goalConds, but it's ok to violate
# shadows.

# 2. Holding(hand) - should not suggest h if goalConds already has
# Holding(h)

# 3. Conf() - if there is Conf in goalConds, then fail.  If there's a
# baseConf in goalConds, then we have to use that base.

# Results (for R2):

# Pose(obj)
# Holding(hand) = none

# Returns (hand, graspMu, graspFace, graspConf,  preConf)

def placeGen(args, goalConds, bState, outBindings):
    gen = placeGenGen(args, goalConds, bState, outBindings)
    for ans, viol, hand in gen:
        tr('placeGen', 1, 'hand='+hand)
        (gB, pB, c, ca) = ans
         # LPK: added return of pose mode and face, in the case they
         # weren't boudn coming in.
        yield (hand, gB.poseD.mode().xyztTuple(), gB.grasp.mode(), c, ca,
               pB.poseD.mode().xyztTuple(), pB.support.mode())

# Either hand or poses will be specified, but generally not both.  They will never both be unspecified.

def placeGenGen(args, goalConds, bState, outBindings):
    (obj, hand, poses, support, objV, graspV, objDelta, graspDelta, confDelta,
     prob) = args
    tag = 'placeGen'
    base = sameBase(goalConds)
    tr(tag, 0, 'obj=%s, base=%s'%(obj, base))
    # tr(tag, 2, ('args', args))

    if goalConds:
        if getConf(goalConds, None):
            tr(tag, 1, '=> conf is already specified, failing')
            return

    pbs = bState.pbs.copy()
    world = pbs.getWorld()

    if poses == '*' or isVar(poses) or support == '*' or isVar(support):
        tr(tag, 2, 'Unspecified pose')
        if base:
            # Don't try to keep the same base, if we're trying to place the object away.
            tr(tag, 1, '=> unspecified pose with same base constraint, failing')
            return
        assert not isVar(hand)
        
        # Just placements specified in goal (and excluding obj)
        # placeInGenAway does not do this when calling placeGen
        newBS = pbs.copy()
        newBS = newBS.updateFromGoalPoses(goalConds, updateConf=False)
        newBS = newBS.excludeObjs([obj])
        # v is viol
        for ans,v in placeInGenAway((obj, objDelta, prob),
                                    goalConds, newBS, outBindings):
            (pB, gB, c, ca) = ans
            yield (gB, pB, c, ca), v, hand
        return

    if not isinstance(poses[0], (list, tuple, frozenset)):
        poses = frozenset([poses])

    graspB = ObjGraspB(obj, world.getGraspDesc(obj), None,
                       PoseD(None, graspV), delta=graspDelta)
    def placeBGen():
        for pose in poses:
            yield ObjPlaceB(obj, world.getFaceFrames(obj), support,
                            PoseD(pose, objV), delta=objDelta)
    placeBs = Memoizer('placeBGen_placeGen', placeBGen())

    # Figure out whether one hand or the other is required;  if not, do round robin
    leftGen = placeGenTop((obj, graspB, placeBs, 'left', base, prob),
                                 goalConds, pbs, outBindings)
    rightGen = placeGenTop((obj, graspB, placeBs, 'right', base, prob),
                                 goalConds, pbs, outBindings)
    
    for ans in chooseHandGen(pbs, goalConds, obj, hand, leftGen, rightGen):
        yield ans

def chooseHandGen(pbs, goalConds, obj, hand, leftGen, rightGen):
    tag = 'chooseHandGen'
    assert not (pbs.useRight == False and hand == 'right')
    mustUseLeft = (hand == 'left' or not pbs.useRight)
    mustUseRight = (hand == 'right')
    holding = dict(getHolding(goalConds))   # values might be 'none'
    # What are we required to be holding
    leftHeldInGoal = 'left' in holding
    rightHeldInGoal = 'right' in holding
    # What are we currently holding (heuristic value)
    leftHeldNow = pbs.held['left'].mode() != 'none'
    rightHeldNow = pbs.held['right'].mode() != 'none'
    # Are we already holding the desired object
    leftHeldTargetObjNow = pbs.held['left'].mode() == obj
    rightHeldTargetObjNow = pbs.held['right'].mode() == obj

    if mustUseLeft or rightHeldInGoal:
        if leftHeldInGoal:
            tr(tag, 0, '=> Left held already in goal, fail')
            return
        else:
            gen = leftGen
    elif mustUseRight or leftHeldInGoal:
        if rightHeldInGoal:
            tr(tag, 0, '=> Right held already in goal, fail')
            return
        else:
            gen = rightGen
    elif rightHeldTargetObjNow or (leftHeldNow and not leftHeldTargetObjNow):
        # Try right hand first if we're holding something in the left
        gen = roundrobin(rightGen, leftGen)
    else:
        gen = roundrobin(leftGen, rightGen)
    return gen

# returns values for (?graspPose, ?graspFace, ?conf, ?confAppr)
def placeGenTop(args, goalConds, pbs, outBindings, regrasp=False, away=False, update=True):
    (obj, graspB, placeBs, hand, base, prob) = args

    key = ((obj, graspB, placeBs, hand, tuple(base) if base else None, prob),
           frozenset(goalConds), pbs, regrasp, away, update)
    startTime = time.clock()
    tag = 'placeGen'
    tr(tag, 0, '(%s,%s) h=%s'%(obj,hand, glob.inHeuristic))
    tr(tag, 2, 
       zip(('obj', 'graspB', 'placeBs', 'hand', 'prob'), args),
       ('goalConds', goalConds),
       ('moveObjBs', pbs.moveObjBs),
       ('fixObjBs', pbs.fixObjBs),
       ('held', (pbs.held['left'].mode(),
                 pbs.held['right'].mode(),
                 pbs.graspB['left'],
                 pbs.graspB['right'])))
    if obj == 'none' or not placeBs:
        tr(tag, 1, '=> obj is none or no placeB, failing')
        return
    if goalConds:
        if getConf(goalConds, None) and not away:
            tr(tag, 1, '=> goal conf specified and not away, failing')
            return
        for (h, o) in getHolding(goalConds):
            if h == hand:
                tr(tag, 1, '=> Hand=%s is already Holding, failing'%hand)
                return
    conf = None
    confAppr = None
    # Set up pbs
    newBS = pbs.copy()
    # Just placements specified in goal (and excluding obj)
    if update:                          # could be done by caller
        newBS = newBS.updateFromGoalPoses(goalConds, updateConf=not away)
        newBS = newBS.excludeObjs([obj])
    tr(tag, 2, 'Goal conditions', draw=[(newBS, prob, 'W')], snap=['W'])
    gen = placeGenAux(newBS, obj, confAppr, conf, placeBs.copy(),
                      graspB, hand, base, prob,
                      regrasp=regrasp, pbsOrig = pbs)

    # !! double check reachObst collision?
    for x,v in gen:
        (gB, pB, c, ca) = x
        pose = pB.poseD.mode().xyztTuple() if pB else None
        grasp = gB.grasp.mode() if gB else None
        pg = (pB.support.mode(), grasp)
        w = v.weight() if v else None
        tr(tag, 1, '(%s,%s) viol=%s'%(obj,hand, w)+' (p,g)=%s'%(pg,)\
           +' pose=%s'%(pose,) +' (t=%s)'%(time.clock()-startTime))
        yield x,v, hand

def placeGenAux(pbs, obj, confAppr, conf, placeBs, graspB, hand, base, prob,
                regrasp=False, pbsOrig=None):
    def placeable(ca, c, quick=False):
        (pB, gB) = context[ca]
        return canPickPlaceTest(pbs, ca, c, hand, gB, pB, prob,
                                op='place', quick=quick)

    def checkRegraspable(pB):
        if pB in regraspablePB:
            return regraspablePB[pB]
        other =  [next(potentialGraspConfGen(pbs, pB, gBO, conf, hand, base, prob, nMax=1),
                       (None, None, None))[0] \
                  for gBO in gBOther]
        if any(other):
            tr(tag, 2,
               ('Regraspable', pB.poseD.mode(), [gBO.grasp.mode() for gBO in gBOther]),
               draw=[(c, 'W', 'green') for (c, ca, v) in \
                     [o for o in other if o != None]], snap=['W'])
            regraspablePB[pB] = True
            return True
        else:
            regraspablePB[pB] = False
            tr(tag, 2, ('Not regraspable', pB.poseD.mode()))
            return False

    def checkOrigGrasp(gB):
        # 0 if currently true
        # 1 if could be used on object's current position
        # 2 otherwise
        
        # Prefer current grasp
        if obj == pbsOrig.held[hand].mode():
            currGraspB = pbsOrig.graspB[hand]
            match = (gB.grasp.mode() == currGraspB.grasp.mode()) and \
                      gB.poseD.mode().near(currGraspB.poseD.mode(), .01, .01)
            if match:
                tr(tag, 2, 'current grasp is a match',
                   ('curr', currGraspB), ('desired', gB))
                return 0

        pB = pbsOrig.getPlaceB(obj, default=False) # check we know where obj is.
        if pbsOrig and pbsOrig.held[hand].mode() != obj and pB:
            nextGr = next(potentialGraspConfGen(pbsOrig, pB, gB, conf, hand, base,
                                          prob, nMax=1),
                              (None, None, None))
            # !!! LPK changed this because next was returning None
            if nextGr and nextGr[0]:
                return 1
            else:
                return 2
        else:
            return 1

    def placeApproachConfGen(grasps):
        placeBsCopy = placeBs.copy()
        for pB in placeBsCopy:          # re-generate
            for gB in grasps:
                tr(tag, 2,
                   ('considering grasps for ', pB),
                   ('for grasp class', gB.grasp),
                   ('placeBsCopy.values', len(placeBsCopy.values)))
                if regrasp:
                    checkRegraspable(pB)
                graspConfGen = potentialGraspConfGen(pbs, pB, gB, conf, hand, base, prob)
                count = 0
                for c,ca,_ in graspConfGen:
                    tr(tag, 2, 'Yielding grasp approach conf',
                       draw=[(pbs, prob, 'W'), (c, 'W', 'orange', shWorld.attached)],
                       snap=['W'])
                    approached[ca] = c
                    count += 1
                    context[ca] = (pB, gB)
                    yield ca
                    # if count > 2: break # !! ??
        tr(tag, 2, 'found %d confs'%count)

    def regraspCost(ca):
        if not regrasp:
            # if debug('placeGen'): print 'not in regrasp mode, cost = 0'
            return 0
        (pB, gB) = context[ca]
        if pB in regraspablePB:
            if regraspablePB[pB]:
                # if debug('placeGen'): print 'regrasp cost = 0'
                return 0
            else:
                # if debug('placeGen'): print 'regrasp cost = 5'
                return 5
        else:
            # if debug('placeGen'): print 'unknown pB, cost = 0'
            return 0

    tag = 'placeGen'
    approached = {}
    context = {}
    regraspablePB = {}
    rm = pbs.getRoadMap()
    shWorld = pbs.getShadowWorld(prob)
    if regrasp:
         graspBOther = graspB.copy()
         otherGrasps = range(len(graspBOther.graspDesc))
         otherGrasps.remove(graspB.grasp.mode())
         if otherGrasps:
             graspBOther.grasp = UniformDist(otherGrasps)
             gBOther = list(graspGen(pbs, obj, graspBOther))
         else:
             gBOther = []

    allGrasps = [(checkOrigGrasp(gB), gB) for gB in graspGen(pbs, obj, graspB)]
    gClasses, gCosts = groupByCost(allGrasps)

    for grasps, gCost in zip(gClasses, gCosts):
        targetConfs = placeApproachConfGen(grasps)
        batchSize = 1 if glob.inHeuristic else pickPlaceBatchSize
        batch = 0
        while True:
            # Collect the next batach of trialConfs
            batch += 1
            trialConfs = []
            count = 0
            minCost = 1e6
            for ca in targetConfs:   # targetConfs is a generator
                viol, reason = placeable(ca, approached[ca])
                if viol:
                    cost = viol.weight() + gCost + regraspCost(ca)
                    minCost = min(cost, minCost)
                    trialConfs.append((cost, viol, ca))
                else:
                    tr(tag, 2, 'Failure of placeable: '+reason)
                    continue
                count += 1
                if count == batchSize or minCost == 0: break
            if count == 0: break
            pbs.getShadowWorld(prob)
            trialConfs.sort()
            for _, viol, ca in trialConfs:
                (pB, gB) = context[ca]
                c = approached[ca]
                ans = (gB, pB, c, ca)
                tr(tag, 2,
                   ('->', ans), ('viol', viol),
                   draw=[(pbs, prob, 'W'),
                         (pB.shape(shWorld), 'W', 'magenta'),
                         (c, 'W', 'magenta', shWorld.attached)],
                   snap=['W'])
                yield ans, viol
    tr(tag, 0, 'out of values')

# Preconditions (for R1):

# 1. Pose(obj) - pick pose that does not conflict with what goalConds
# say about this obj.  So, if Pose in goalConds with smaller variance
# works, then fine, but otherwise a Pose in goalConds should cause
# failure.

# Results (for R2):

# In(obj, Region)

# Return objPose, poseFace.
def placeInRegionGen(args, goalConds, bState, outBindings, away = False, update=True):
    gen = placeInRegionGenGen(args, goalConds, bState, outBindings, away = False)
    for ans, viol in gen:
        (pB, gB, cf, ca) = ans
        yield (pB.poseD.mode().xyztTuple(), pB.support.mode())

def placeInRegionGenGen(args, goalConds, bState, outBindings, away = False, update=True):
    (obj, region, var, delta, prob) = args
    tag = 'placeInGen'
    pbs = bState.pbs.copy()
    world = pbs.getWorld()

    # Get the regions
    if not isinstance(region, (list, tuple, frozenset)):
        regions = frozenset([region])
    elif len(region) == 0:
        raise Exception, 'need a region to place into'
    else:
        regions = frozenset(region)
    shWorld = pbs.getShadowWorld(prob)
    regShapes = [shWorld.regionShapes[region] for region in regions]
    tr(tag, 1, 'Target region in purple',
       draw=[(pbs, prob, 'W')] + [(rs, 'W', 'purple') for rs in regShapes], snap=['W'])
    # Set pose and support from current state
    pose = None
    if pbs.getPlaceB(obj, default=False):
        # If it is currently placed, use that support
        support = pbs.getPlaceB(obj).support.mode()
        pose = pbs.getPlaceB(obj).poseD.mode()
    elif obj == pbs.held['left'].mode():
        attachedShape = pbs.getRobot().attachedObj(pbs.getShadowWorld(prob),
                                                   'left')
        shape = pbs.getWorld().getObjectShapeAtOrigin(obj).\
                                        applyLoc(attachedShape.origin())
        support = supportFaceIndex(shape)
    elif obj == pbs.held['right'].mode():
        attachedShape = pbs.getRobot().attachedObj(pbs.getShadowWorld(prob),
                                                   'right')
        shape = pbs.getWorld().getObjectShapeAtOrigin(obj).\
                                        applyLoc(attachedShape.origin())
        support = supportFaceIndex(shape)
    else:
        assert None, 'Cannot determine support'

    graspV = bState.domainProbs.maxGraspVar
    graspDelta = bState.domainProbs.graspDelta
    graspB = ObjGraspB(obj, world.getGraspDesc(obj), None,
                       PoseD(None, graspV), delta=graspDelta)

    # Check if object pose is specified in goalConds
    poseBels = getGoalPoseBels(goalConds, world.getFaceFrames)
    if obj in poseBels:
        pB = poseBels[obj]
        shw = shadowWidths(pB.poseD.var, pB.delta, prob)
        shwMin = shadowWidths(graspV, graspDelta, prob)
        if any(w > mw for (w, mw) in zip(shw, shwMin)):
            args = (obj, None, pB.poseD.mode().xyztTuple(),
                    support, var, graspV,
                    delta, graspDelta, None, prob)
            gen = placeGenGen(args, goalConds, bState, outBindings)
            for ans, v, hand in gen:
                (gB, pB, c, ca) = ans
                pose = pB.poseD.mode() if pB else None
                grasp = gB.grasp.mode() if gB else None
                sup = pB.support.mode() if pB else None
                pg = (sup, grasp)
                regions = [x.name() for x in regShapes]
                tr(tag, 1, '(%s,%s) h=%s'%(obj,regions,glob.inHeuristic) + \
                   ' v=%s'%(v.weight() if v else None) + \
                   ' (p,g)=%s, pose=%s', (pg, pose),
                   draw=[(pbs, prob, 'W')] + [(rs, 'W', 'purple') for rs in regShapes],
                   snap=['W'])
                yield (pB, gB, c, ca), v
            return
        else:
            # If pose is specified and variance is small, return
            return

    # The normal case

    # Use the input var and delta to select candidate poses in the
    # region.  We will use smaller values (in general) for actually
    # placing.
    placeB = ObjPlaceB(obj, world.getFaceFrames(obj), support,
                       PoseD(pose, var), delta=delta)

    gen = placeInGenTop((obj, regShapes, graspB, placeB, None, prob),
                          goalConds, pbs, outBindings, away = away, update=update)
    for ans in gen:
        yield ans

placeVarIncreaseFactor = 3 # was 2
lookVarIncreaseFactor = 2


def placeInGenAway(args, goalConds, pbs, outBindings):
    # !! Should search over regions and hands
    (obj, delta, prob) = args
    if not pbs.awayRegions():
        raw_input('Need some awayRegions')
        return
    tr('placeInGenAway', 2, zip(('obj', 'delta', 'prob'), args),
       draw=[(pbs, prob, 'W')], snap=['W'])
    targetPlaceVar = tuple([placeVarIncreaseFactor * x \
                            for x in pbs.domainProbs.obsVarTuple])
    for ans,v in placeInRegionGenGen((obj, pbs.awayRegions(),
                                      targetPlaceVar, delta, prob),
                                     # preserve goalConds to get reachObsts
                                     goalConds, pbs, [], away=True, update=False):
        yield ans,v

placeInGenMaxPoses  = 50
placeInGenMaxPosesH = 10

def placeInGenTop(args, goalConds, pbs, outBindings,
                  regrasp=False, away = False, update=True):
    (obj, regShapes, graspB, placeB, base, prob) = args
    tag = 'placeInGen'
    regions = [x.name() for x in regShapes]
    tr(tag, 0, '(%s,%s) h=%s'%(obj,regions, glob.inHeuristic))
    tr(tag, 2, 
       zip(('obj', 'regShapes', 'graspB', 'placeB', 'prob'), args))
    if obj == 'none' or not regShapes:
        # Nothing to do
        tr(tag, 1, '=> object is none or no regions, failing')
        return
    if goalConds and getConf(goalConds, None) and not away:
        # if conf is specified, just fail
        tr(tag, 1, '=> conf is specified, failing')
        return

    conf = None
    confAppr = None
    # Obstacles for all Reachable fluents
    reachObsts = getReachObsts(goalConds, pbs)
    if reachObsts == None:
        tr(tag, 1, '=> No path for reachObst, failing')
        return
    tr(tag, 2, '%d reachObsts - in brown'%len(reachObsts),
       draw=[(pbs, prob, 'W')] + [(obst, 'W', 'brown') for _,obst in reachObsts],
       snap=['W'])
    newBS = pbs.copy()           #  not necessary
    pB = placeB
    shWorld = newBS.getShadowWorld(prob)
    nPoses = placeInGenMaxPosesH if glob.inHeuristic else placeInGenMaxPoses
    poseGenLeft = Memoizer('regionPosesLeft',
                           potentialRegionPoseGen(newBS, obj, pB, graspB, prob, regShapes,
                                                  reachObsts, 'left', base,
                                                  maxPoses=nPoses))
    poseGenRight = Memoizer('regionPosesRight',
                            potentialRegionPoseGen(newBS, obj, pB, graspB, prob, regShapes,
                                                   reachObsts, 'right', base,
                                                   maxPoses=nPoses))
    # note the use of PB...
    leftGen = placeInGenAux(newBS, poseGenLeft, goalConds, confAppr,
                            conf, pB, graspB, 'left', base, prob,
                            regrasp=regrasp, away=away, update=update)
    rightGen = placeInGenAux(newBS, poseGenRight, goalConds, confAppr,
                             conf, pB, graspB, 'right', base, prob,
                             regrasp=regrasp, away=away, update=update)
    # Figure out whether one hand or the other is required;  if not, do round robin
    mainGen = chooseHandGen(newBS, goalConds, obj, None, leftGen, rightGen)

    # Picks among possible target poses and then try to place it in region
    for ans,v in mainGen:
        (pB, gB, c, ca) = ans
        pose = pB.poseD.mode() if pB else None
        grasp = gB.grasp.mode() if gB else None
        sup = pB.support.mode() if pB else None
        pg = (sup, grasp)
        tr(tag, 1, '(%s,%s) h='%(obj,regions) + \
           ' v=%s'%(v.weight() if v else None) + \
           ' (p,g)=%s pose=%s'%(pg, pose),
           draw=[(c, 'W', 'green', shWorld.attached)] + \
           [(rs, 'W', 'purple') for rs in regShapes],
           snap=['W'])
        yield ans,v

# Don't try to place all objects at once
def placeInGenAux(pbs, poseGen, goalConds, confAppr, conf, placeB, graspB,
                  hand, base, prob, regrasp=False, away=False, update=True):

    def placeBGen():
        for pose in poseGen.copy():
            yield placeB.modifyPoseD(mu=pose)
    tries = 0
    shWorld = pbs.getShadowWorld(prob)
    gen = Memoizer('placeBGen_placeInGenAux1', placeBGen())
    for ans, viol, hand in placeGenTop((graspB.obj, graspB, gen, hand, base, prob),
                               goalConds, pbs, [], regrasp=regrasp, away=away, update=update):
        (gB, pB, cf, ca) = ans
        tr('placeInGen', 1, ('=> blue', ans),
           draw=[(pbs, prob, 'W'),
                 (pB.shape(shWorld), 'W', 'blue'),
                 (cf, 'W', 'blue', shWorld.attached)],
           snap=['W'])
        yield (pB, gB, cf, ca), viol

maxLookDist = 1.5

# Preconditions (for R1):

# 1. CanSeeFrom() - make a world from the goalConds and CanSeeFrom (visible) should be true.

# 2. Conf() - if there is Conf in goalConds, then fail.  If there's a
# baseConf in goalConds, then we have to use that base.

# Results (for R2):

# Condition to avoid violating future canReach
# If we're in shadow in starting state, ok.   Otherwise, don't walk into a shadow.

# Returns lookConf
# The lookDelta is a slop factor.  Ideally if the robot is within that
# factor, visibility should still hold.
def lookGen(args, goalConds, bState, outBindings):
    (obj, pose, support, objV, objDelta, lookDelta, prob) = args
    pbs = bState.pbs.copy()
    world = pbs.getWorld()

    base = sameBase(goalConds)

    if pose == '*':
        # This could produce a mode of None
        pB = pbs.getPlaceB(obj, default=False)
        if pB == None:
            print 'Trying to reduce variance on object pose but obj is in hand'
            return
        poseD = pB.poseD if pB else PoseD(None, 4*(0.,))
    else: 
        poseD = PoseD(pose, objV)
    if isVar(support) or support == '*':
        support = pbs.getPlaceB(obj).support.mode()
    if objDelta == '*':
        objDelta = lookDelta
        
    placeB = ObjPlaceB(obj, world.getFaceFrames(obj), support, poseD,
                       delta = objDelta)
    # Pretend that the object has bigger delta
    # delta=tuple([o+l for (o,l) in zip(objDelta, lookDelta)]))

    # Be careful here!  There is a shadow for the purposes of looking
    # and a shadow for the purposes of moving, and I guess they might
    # be different.

    # Don't try to look at the whole shadow
    # We'll do this later.
    # placeB = placeB.modifyPoseD(var = (0.0001, 0.0001, 0.0001, 0.0005))


    for ans, viol in lookGenTop((obj, placeB, lookDelta, base, prob),
                                goalConds, pbs, outBindings):
        yield ans

def lookGenTop(args, goalConds, pbs, outBindings):

    def testFn(c):
        print 'Trying base conf', c['pr2Base']
        obst_rob = obst + [c.placement(shWorld.attached)]
        return visible(shWorld, c, sh, obst_rob, prob, moveHead=True)[0]

    (obj, placeB, lookDelta, base, prob) = args
    tag = 'lookGen'
    tr(tag, 0, '(%s) h=%s'%(obj, glob.inHeuristic))
    newBS = pbs.copy()
    newBS = newBS.updateFromGoalPoses(goalConds)
    newBS.addAvoidShadow([obj])
    if placeB.poseD.mode() == None:
        tr(tag, 1, '=> object is in the hand, failing')
        return
    newBS.updatePermObjPose(placeB)
    rm = newBS.getRoadMap()
    shWorld = newBS.getShadowWorld(prob)
    if any(shWorld.attached.values()):
        tr(tag, 1, 'attached=%s'%shWorld.attached)
    shName = shadowName(obj)
    # Uses original placeB
    sh = shWorld.objectShapes[shName]
    obst = [s for s in shWorld.getNonShadowShapes() if s.name() != obj ]

    goalConf = getConf(goalConds, None)
    if goalConds and goalConf:
        # if conf is specified, just fail
        tr(tag, 1, '=> Conf is specified, failing: ' + str(goalConf))
        return
    if obj in [newBS.held[hand].mode() for hand in ['left', 'right']]:
        tr(tag, 1, '=> object is in the hand, failing')
        return
    if base:
        conf = targetConf(goalConds)
        if conf == None:
            print 'No conf found for lookConf with specified base'
            raw_input('This might be an error in regression')
            return
        path, viol = canReachHome(newBS, conf, prob, Violations(), moveBase=False)
        tr(tag, 1, '(%s) specified base viol=%s'%(obj, viol.weight() if viol else None))
        if not path:
            tr(tag, 2, 'Failed to find a path to look conf (cyan) with specified base.',
               draw=[(newBS, prob, 'W'), (conf, 'W', 'cyan', shWorld.attached)], snap=['W'])
            return
        if testFn(conf):
            lookConf = lookAtConfCanView(newBS, prob, conf, sh)
            if lookConf:
                tr(tag, 1, '=> Found a path to look conf with specified base.',
                   ('-> cyan', lookConf.conf),
                   draw=[(newBS, prob, 'W'), (lookConf, 'W', 'cyan', shWorld.attached)],
                   snap=['W'])
                yield (lookConf,), viol
        return

    # A shape for the purposes of viewing.  Make the shadow very small
    placeB = placeB.modifyPoseD(var = (0.0001, 0.0001, 0.0001, 0.0005))
    # be smarter about this?  LPK took this out
    # tempPlaceB.delta = (.01, .01, .01, .01)
    shape = placeB.shadow(newBS.getShadowWorld(prob))

    # Check current conf
    curr = newBS.conf
    lookConf = lookAtConfCanView(newBS, prob, curr, shape)
    if lookConf and testFn(lookConf):
        tr(tag, 1, '=> Using current conf.',
           draw=[(pbs, prob, 'W'),
                 (lookConf, 'W', 'cyan', shWorld.attached)])
        yield (lookConf,), rm.confViolations(curr, newBS, prob)
        
    world = newBS.getWorld()
    if obj in world.graspDesc and not glob.inHeuristic:
        graspVar = 4*(0.001,)
        graspDelta = 4*(0.001,)   # put back to prev value
        graspB = ObjGraspB(obj, world.getGraspDesc(obj), None,
                           PoseD(None, graspVar), delta=graspDelta)
        # Use pbs to generate candidate confs, since they will need to
        # collide with shadow of obj.
        for gB in graspGen(pbs, obj, graspB):
            for hand in ['left', 'right']:
                # Changed smallPlaceB to placeB
                for ans, viol in pickGenTop((obj, gB, placeB, hand, base,
                                             prob),
                                            goalConds, pbs, outBindings,
                                            onlyCurrent=True):
                    (pB, c, ca) = ans   # pB should be placeB
                    lookConf = lookAtConfCanView(pbs, prob, ca, shape)
                    if not lookConf:
                        tr(tag, 2, 'canView failed')
                        continue
                    viol = rm.confViolations(lookConf, pbs, prob)
                    if testFn(lookConf):
                        vw = viol.weight() if viol else None
                        tr(tag, 2, '(%s) canView cleared viol=%s'%(obj, vw))
                        yield (lookConf,), viol
    lookConfGen = potentialLookConfGen(newBS, prob, sh, maxLookDist) # look unconstrained by base
    for ans in rm.confReachViolGen(lookConfGen, newBS, prob,
                                   testFn = testFn):
        viol, cost, path = ans
        tr(tag, 2, '(%s) viol=%s'%(obj, viol.weight() if viol else None))
        if not path:
            tr(tag, 2, 'Failed to find a path to look conf.')
            raw_input('Failed to find a path to look conf.')
            continue
        conf = path[-1]
        lookConf = lookAtConfCanView(newBS, prob, conf, sh)
        if lookConf:
            tr(tag, 2, '(%s) general conf viol=%s'%(obj, viol.weight() if viol else None),
               ('-> cyan', lookConf.conf),
               draw=[(pbs, prob, 'W'),
                     (lookConf, 'W', 'cyan', shWorld.attached)],
               snap=['W'])
            yield (lookConf,), viol

def lookAtConfCanView(pbs, prob, conf, shape, hands=['left', 'right']):
    lookConf = lookAtConf(conf, shape)
    if not glob.inHeuristic:
        for hand in hands:
            if not lookConf:
                raw_input('lookAtConfCanView failed conf')
                return None
            path = canView(pbs, prob, lookConf, hand, shape)
            if not path:
                tr('lookAtConfCanView', 1, 'lookAtConfCanView failed path')
                return None
            lookConf = path[-1]
    return lookConf

## lookHandGen
## obj, hand, graspFace, grasp, graspVar, graspDelta and gives a conf

# !! NEEDS UPDATING

# Preconditions (for R1):

# 1. CanSeeFrom() - make a world from the goalConds and CanSeeFrom
# should be true.

# 2. Conf() - if there is Conf in goalConds, then fail.  If there's a
# baseConf in goalConds, then we have to use that base.

# Returns lookConf
def lookHandGen(args, goalConds, bState, outBindings):
    (obj, hand, graspFace, grasp, graspV, graspDelta, prob) = args
    pbs = bState.pbs.copy()
    world = pbs.getWorld()
    if obj == 'none':
        graspB = None
    else:
        graspB = ObjGraspB(obj, world.getGraspDesc(obj), graspFace,
                           PoseD(grasp, graspV), delta=graspDelta)
    for ans, viol in lookHandGenTop((obj, hand, graspB, prob),
                                    goalConds, pbs, outBindings):
        yield ans

def lookHandGenTop(args, goalConds, pbs, outBindings):
    def objInHand(conf, hand):
        if (conf, hand) not in handObj:
            attached = shWorld.attached
            if not attached[hand]:
                attached = attached.copy()
                tool = conf.robot.toolOffsetX[hand]
                attached[hand] = Box(0.1,0.05,0.1, None, name='virtualObject').applyLoc(tool)
            _, attachedParts = conf.placementAux(attached, getShapes=[])
            handObj[(conf, hand)] = attachedParts[hand]
        return handObj[(conf, hand)]

    def testFn(c):
        if c not in placements:
            placements[c] = c.placement()
        ans = visible(shWorld, c, objInHand(c, hand),
                       [placements[c]]+obst, prob, moveHead=True)[0]
        return ans
    
    (obj, hand, graspB, prob) = args
    tag = 'lookHandGen'
    placements = {}
    handObj = {}
    tr(tag, 0, '(%s) h=%s'%(obj, glob.inHeuristic))
    newBS = pbs.copy()
    newBS = newBS.updateFromGoalPoses(goalConds)
    newBS.updateHeldBel(graspB, hand)
    shWorld = newBS.getShadowWorld(prob)
    if glob.inHeuristic:
        lookConf = lookAtConf(newBS.conf, objInHand(newBS.conf, hand))
        if lookConf:
            tr(tag, 1, ('->', lookConf))
            yield (lookConf,), Violations()
        return
    if goalConds and getConf(goalConds, None):
        tr(tag, 1, '=> conf is specified, failing')
        return
    rm = newBS.getRoadMap()
    obst = [s for s in shWorld.getNonShadowShapes() if s.name() != obj ]
    lookConfGen = potentialLookHandConfGen(newBS, prob, hand)
    for ans in rm.confReachViolGen(lookConfGen, newBS, prob,
                                   startConf = newBS.conf,
                                   testFn = testFn):
        viol, cost, path = ans
        tr(tag, 2, '(%s) viol=%s'%(obj, viol.weight() if viol else None))
        if not path:
            tr(tag, 2, 'Failed to find a path to look conf.')
            continue
        tr(tag, 1, ('-> cyan', lookConf.conf),
           draw=[(pbs, prob, 'W'),
                 (lookConf, 'W', 'cyan', shWorld.attached)],
           snap=['W'])
        yield (lookConf,), viol

def moveOut(pbs, prob, obst, delta, goalConds):
    tr('moveOut', 0, 'obst=%s'%obst)
    domainPlaceVar = tuple([placeVarIncreaseFactor * x \
                            for x in pbs.domainProbs.obsVarTuple])
    if not isinstance(obst, str):
        obst = obst.name()
    for ans, v in placeInGenAway((obst, delta, prob), goalConds, pbs, None):
        (pB, gB, cf, ca) = ans
        yield (obst, pB.poseD.mode().xyztTuple(), pB.support.mode(), domainPlaceVar, delta)

# Preconditions (for R1):

# 1. CanReach(...) - new Pose fluent should not make the canReach
# infeasible (use fluent as taboo).
# new Pose fluent should not already be in conditions (any Pose for this obj).

# 2. Pose(obj) - new Pose has to be consistent with the goal (ok to
# reduce variance wrt goal but not cond). if Pose(obj) in goalConds,
# can only reduce variance.

# returns
# ['Occ', 'Pose', 'PoseFace', 'PoseVar', 'PoseDelta']
# obj, pose, face, var, delta
def canReachGen(args, goalConds, bState, outBindings):
    (conf, fcp, prob, cond) = args
    pbs = bState.pbs.copy()
    # Don't make this infeasible
    goalFluent = Bd([CanReachHome([conf, fcp, cond]), True, prob], True)
    goalConds = goalConds + [goalFluent]
    # Set up PBS
    newBS = pbs.copy()
    newBS = newBS.updateFromGoalPoses(goalConds)
    newBS = newBS.updateFromGoalPoses(cond, permShadows=True)
    shWorld = newBS.getShadowWorld(prob)
    tr('canReachGen', 0, 
       draw=[(newBS, prob, 'W'),
             (conf, 'W', 'pink', shWorld.attached)], snap=['W'])
    tr('canReachGen', 2, zip(('conf', 'fcp', 'prob', 'cond'),args))
    # Call
    def violFn(pbs):
        path, viol = canReachHome(pbs, conf, prob, Violations())
        return viol
    lookVar = tuple([lookVarIncreaseFactor * x \
                            for x in pbs.domainProbs.obsVarTuple])
    for ans in canXGenTop(violFn, (cond, prob, lookVar),
                          goalConds, newBS, outBindings, 'canReachGen'):
        tr('canReachGen', 1, ('->', ans))
        yield ans
    tr('canReachGen', 1, 'exhausted')

# Preconditions (for R1):

# 1. CanPickPlace(...) - new Pose fluent should not make the
# canPickPlace infeasible.  new Pose fluent should not already be in
# conditions.

# 2. Pose(obj) - new Pose has to be consistent with the goal (ok to
# reduce variance wrt goal but not cond)

# LPK!! More efficient if we notice right away that we cannot ask to
# change the pose of an object that is in the hand in goalConds
def canPickPlaceGen(args, goalConds, bState, outBindings):
    (preconf, ppconf, hand, obj, pose, realPoseVar, poseDelta, poseFace,
     graspFace, graspMu, graspVar, graspDelta, prob, cond, op) = args
    pbs = bState.pbs.copy()
    # Don't make this infeasible
    cppFluent = Bd([CanPickPlace([preconf, ppconf, hand, obj, pose,
                                  realPoseVar, poseDelta, poseFace,
                                  graspFace, graspMu, graspVar, graspDelta,
                                  op, cond]), True, prob], True)
    poseFluent = B([Pose([obj, poseFace]), pose, realPoseVar, poseDelta, prob],
                    True)
    goalConds = goalConds + [cppFluent, poseFluent]
    world = pbs.getWorld()
    lookVar = tuple([lookVarIncreaseFactor * x \
                            for x in pbs.domainProbs.obsVarTuple])
    graspB = ObjGraspB(obj, world.getGraspDesc(obj), graspFace,
                       PoseD(graspMu, graspVar), delta= graspDelta)
    placeB = ObjPlaceB(obj, world.getFaceFrames(obj), poseFace,
                       PoseD(pose, realPoseVar), delta=poseDelta)
    # Set up PBS
    newBS = pbs.copy()
    newBS = newBS.updateFromGoalPoses(goalConds)
    newBS = newBS.updateFromGoalPoses(cond, permShadows=True)
    # Debug
    shWorld = newBS.getShadowWorld(prob)
    tr('canPickPlaceGen', 0, 
       draw=[(newBS, prob, 'W'),
             (preconf, 'W', 'blue', shWorld.attached),
             (ppconf, 'W', 'pink', shWorld.attached),
             (placeB.shape(shWorld), 'W', 'pink')],
       snap=['W'])
    tr('canPickPlaceGen', 2,
       zip(('preconf', 'ppconf', 'hand', 'obj', 'pose', 'realPoseVar', 'poseDelta', 'poseFace',
            'graspFace', 'graspMu', 'graspVar', 'graspDelta', 'prob', 'cond', 'op'),
           args))
    # Initial test
    def violFn(pbs):
        v, r = canPickPlaceTest(pbs, preconf, ppconf, hand,
                                graspB, placeB, prob, op=op)
        return v
    for ans in canXGenTop(violFn, (cond, prob, lookVar),
                          goalConds, newBS, outBindings, 'canPickPlaceGen'):
        tr('canPickPlaceGen', 1, ('->', ans))
        yield ans
    tr('canPickPlaceGen', 1, 'exhausted')

def canXGenTop(violFn, args, goalConds, newBS, outBindings, tag):
    (cond, prob, lookVar) = args
    tr(tag, 0, 'h=%s'%glob.inHeuristic)
    # Initial test
    viol = violFn(newBS)
    tr(tag, 1, ('viol', viol),
       draw=[(newBS, prob, 'W')], snap=['W'])
    if not viol:                  # hopeless
        tr(tag, 1, 'Impossible dream')
        if tag == 'canPickPlaceGen':
            glob.debugOn.append('canPickPlaceTest')
            violFn(newBS)
            glob.debugOn.remove('canPickPlaceTest')
        raw_input('=> Impossible dream')
        return
    if viol.empty():
        tr(tag, 1, '=> No obstacles or shadows; returning')
        return

    #objBMinVarGrasp = tuple([x**2/2*x for x in newBS.domainProbs.obsVarTuple])
    
    # LPK Make this a little bigger?
    objBMinVarGrasp = tuple([x/2 for x in newBS.domainProbs.obsVarTuple])
    objBMinVarStatic = tuple([x**2 for x in newBS.domainProbs.odoError])
    objBMinProb = 0.95
    # The irreducible shadow
    objBMinDelta = newBS.domainProbs.shadowDelta
    
    lookDelta = objBMinDelta
    moveDelta = newBS.domainProbs.placeDelta
    shWorld = newBS.getShadowWorld(prob)
    fixed = shWorld.fixedObjects
    # Try to fix one of the violations if any...
    obstacles = [o.name() for o in viol.allObstacles() \
                 if o.name() not in fixed]
    shadows = [sh.name() for sh in viol.allShadows() \
               if not sh.name() in fixed]
    if not (obstacles or shadows):
        tr(tag, 1, '=> No movable obstacles or shadows to fix')
        return       # nothing available
    if obstacles:
        obst = obstacles[0]
        for ans in moveOut(newBS, prob, obst, moveDelta, goalConds):
            yield ans
        return
    if shadows:
        shadowName = shadows[0]
        obst = objectName(shadowName)
        graspable = obst in newBS.getWorld().graspDesc    
        objBMinVar = objBMinVarGrasp if graspable else objBMinVarStatic
        placeB = newBS.getPlaceB(obst)
        tr(tag, 1, '=> reduce shadow %s (in red):'%obst,
           draw=[(newBS, prob, 'W'),
                 (placeB.shadow(newBS.getShadowWorld(prob)), 'W', 'red')],
           snap=['W'])
        yield (obst, placeB.poseD.mode().xyztTuple(),
                placeB.support.mode(), objBMinVar, lookDelta)
        # Either reducing the shadow is not enough or we failed and
        # need to move the object (if it's movable).
        if obst not in fixed:
            for ans in moveOut(newBS, prob, obst, moveDelta, goalConds):
                yield ans
    tr(tag, 1, '=> Out of remedies')

# Preconditions (for R1):

# 1. CanSeeFrom(...) - new Pose fluent should not make the CanSeeFrom
# infeasible.  new Pose fluent should not already be in conditions.

# 2. Pose(obj) - new Pose has to be consistent with the goal (ok to
# reduce variance wrt goal but not cond)

# returns
# ['Occ', 'PoseFace', 'Pose', 'PoseVar', 'PoseDelta']
def canSeeGen(args, goalConds, bState, outBindings):
    (obj, pose, support, objV, objDelta, lookConf, lookDelta, prob) = args
    pbs = bState.pbs.copy()
    world = pbs.getWorld()

    if pose == '*':
        poseD = pbs.getPlaceB(obj).poseD
    else: 
        poseD = PoseD(pose, objV)
    if isVar(support) or support == '*':
        support = pbs.getPlaceB(obj).support.mode()
    if objDelta == '*':
        objDelta = lookDelta
    
    placeB = ObjPlaceB(obj, world.getFaceFrames(obj), support,
                       poseD,
                       # Pretend that the object has bigger delta
                       delta=tuple([o+l for (o,l) in zip(objDelta, lookDelta)]))

    for ans in canSeeGenTop((lookConf, placeB, [], prob),
                            goalConds, pbs, outBindings):
        yield ans

def canSeeGenTop(args, goalConds, pbs, outBindings):
    (conf, placeB, cond, prob) = args
    obj = placeB.obj
    tr('canSeeGen', 0, '(%s) h=%s'%(obj, glob.inHeuristic))
    tr('canSeeGen', 2, zip(('conf', 'placeB', 'cond', 'prob'), args))

    newBS = pbs.copy()
    newBS = newBS.updateFromGoalPoses(goalConds)
    newBS = newBS.updateFromGoalPoses(cond, permShadows=True)
    newBS = newBS.updatePermObjPose(placeB)

    shWorld = newBS.getShadowWorld(prob)
    shape = shWorld.objectShapes[placeB.obj]
    obst = [s for s in shWorld.getNonShadowShapes() \
            if s.name() != placeB.obj ]
    p, occluders = visible(shWorld, conf, shape, obst, prob, moveHead=True)
    occluders = [oc for oc in occluders if oc not in newBS.fixObjBs]
    if not occluders:
        tr('canSeeGen', 1, '=> no occluders')
        return
    obst = occluders[0] # !! just pick one
    moveDelta = pbs.domainProbs.placeDelta
    for ans in moveOut(newBS, prob, obst, moveDelta, goalConds):
        yield ans 

def groupByCost(entries):
    classes = []
    values = []
    sentries = sorted(entries)
    for (c, e) in sentries:
        if not(values) or values[-1] != c:
            classes.append([e])
            values.append(c)
        else:
            classes[-1].append(e)
    return classes, values

