import numpy as np
import math
import random
import util
import copy
import time
import windowManager3D as wm
import planGlobals as glob
from planGlobals import debugMsg, debugMsgSkip, debugDraw, debug, pause, torsoZ
from miscUtil import isVar, argmax, isGround, tuplify, roundrobin
from dist import DeltaDist, UniformDist
from pr2Robot import CartConf, gripperTip, gripperFaceFrame
from pr2Util import PoseD, ObjGraspB, ObjPlaceB, Violations, shadowName, objectName, \
     NextColor, supportFaceIndex, Memoizer, shadowWidths
import fbch
from belief import Bd
from pr2Fluents import CanReachHome, canReachHome, inTest
from pr2Visible import visible, lookAtConf
from pr2PlanBel import getConf, getGoalPoseBels, getPoseObjs

from shapes import Box

Ident = util.Transform(np.eye(4))            # identity transform

import pr2GenAux
from pr2GenAux import *

#  How many candidates to generate at a time...  Larger numbers will
#  generally lead to better solutions.
pickPlaceBatchSize = 5

easyGraspGenCacheStats = [0,0]

def trace(*msg):
    if debug('traceGen'):
        for m in msg: print m,
        print ' '

def tracep(pause, *msg):
    if debug('traceGen'):
        print pause+':',
        for m in msg: print m,
        print ' '
    if pause:
        debugMsg(pause)

def easyGraspGen(args, goalConds, bState, outBindings):
    assert fbch.inHeuristic
    graspVar = 4*(0.001,)
    graspDelta = 4*(0.001,)   # put back to prev value
    
    pbs = bState.pbs.copy()
    (obj, hand) = args
    trace('easyGraspGen(%s,%s) h='%(obj,hand), fbch.inHeuristic)
    if obj == 'none' or (goalConds and getConf(goalConds, None)):
        trace('easyGraspGen', 'obj is none or conf in goal conds')
        return
    prob = 0.75
    # Set up pbs
    newBS = pbs.copy()
    # Just placements specified in goal
    newBS = newBS.updateFromGoalPoses(goalConds)
    shWorld = newBS.getShadowWorld(prob)
    if obj == newBS.held[hand].mode():
        gB = newBS.graspB[hand]
        trace('    easyGraspGen(%s,%s)='%(obj, hand), '(p,g)=', (None, gB.grasp.mode()))
        yield (gB.grasp.mode(), gB.poseD.mode().xyztTuple(),
               graspVar, graspDelta)
        return
    if obj == newBS.held[otherHand(hand)].mode():
        tracep('easyGraspGen', 'no easy grasp with this hand')
        return
    rm = newBS.getRoadMap()
    placeB = newBS.getPlaceB(obj)
    graspB = ObjGraspB(obj, pbs.getWorld().getGraspDesc(obj), None,
                       PoseD(None, graspVar), delta=graspDelta)
    cache = pbs.beliefContext.genCaches['easyGraspGen']
    key = (newBS, placeB, graspB, hand, prob)
    easyGraspGenCacheStats[0] += 1
    val = cache.get(key, None)
    if val != None:
        easyGraspGenCacheStats[1] += 1
        cached = 'C'
        memo = val.copy()
    else:
        memo = Memoizer('easyGraspGen',
                        easyGraspGenAux(newBS, placeB, graspB, hand, prob))
        cache[key] = memo
        cached = ''
    for ans in memo:
        trace('    %s easyGraspGen(%s,%s)='%(cached, obj, hand), '(p,g)=', (placeB.support.mode(), ans[0]), ans)
        yield ans
    trace('    easyGraspGen(%s,%s)='%(obj, hand), None)
    debugMsg('easyGraspGen', 'out of values')
    return

def easyGraspGenAux(newBS, placeB, graspB, hand, prob):
    graspVar = 4*(0.001,)
    graspDelta = 4*(0.001,)   # put back to prev value
    
    def graspApproachConfGen(firstConf):
        if firstConf:
            yield firstConf
        for c, ca, _ in graspConfGen:
            approached[ca] = c
            yield ca

    def pickable(ca, c, pB, gB):
        return canPickPlaceTest(newBS, ca, c, hand, gB, pB, prob, op='pick')

    obj = placeB.obj
    approached = {}
    for gB in graspGen(newBS, obj, graspB):
        graspConfGen = potentialGraspConfGen(newBS, placeB, gB, None, hand, None, prob)
        firstConf = next(graspApproachConfGen(None), None)
        if not firstConf: continue
        for ca in graspApproachConfGen(firstConf):
            if pickable(ca, approached[ca], placeB, gB):
                ans = (gB.grasp.mode(), gB.poseD.mode().xyztTuple(),
                       graspVar, graspDelta)
                yield ans
                break

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
    if base:
        print('Same base constraint in pickGen')

    debugMsg('pickGen', 'args', args, ('base', base))

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
    trace('pickGen(%s,%s,%d) b=%s h='%(obj,hand,graspB.grasp.mode(),str(base)), fbch.inHeuristic)
    skip = (fbch.inHeuristic and not debug('inHeuristic'))

    graspDelta = pbs.domainProbs.pickStdev

    debugMsgSkip('pickGen', skip,
                 zip(('obj', 'graspB', 'placeB', 'hand', 'prob'), args),
                 ('goalConds', goalConds),
                 ('moveObjBs', pbs.moveObjBs),
                 ('fixObjBs', pbs.fixObjBs),
                 ('held', (pbs.held['left'].mode(),
                           pbs.held['right'].mode(),
                           pbs.graspB['left'],
                           pbs.graspB['right'])))

    if obj == 'none':                   # can't pick up 'none'
        tracep('pickGen', '    cannot pick up none')
        return
    if goalConds:
        if  getConf(goalConds, None):
            tracep('pickGen', '    conf is already specified')
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
        debugMsg('pickGen',
                 ('Trying to pick object already in hand',
                  ' -- support surface is', sup))
    elif obj == pbs.held[otherHand(hand)].mode():
        attachedShape = pbs.getRobot().attachedObj(pbs.getShadowWorld(prob),
                                                   otherHand(hand))
        shape = pbs.getWorld().getObjectShapeAtOrigin(obj).\
                                       applyLoc(attachedShape.origin())
        sup = supportFaceIndex(shape)
        pose = None
        conf = None
        confAppr = None
        debugMsg('pickGen',
                 ('Trying to pick object already in hand other hand',
                  '-- support surface is', sup))
    else:
        # Use placeB from the current state
        pose = pbs.getPlaceB(obj).poseD.mode()
        sup =  pbs.getPlaceB(obj).support.mode()
        conf = None
        confAppr = None
    placeB.poseD = PoseD(pose, placeB.poseD.var) # record the pose
    placeB.support = DeltaDist(sup)                             # and supportFace
    debugMsgSkip('pickGen', skip, ('target placeB', placeB))
    # Set up pbs
    newBS = pbs.copy()
    # Just placements specified in goal
    newBS = newBS.updateFromGoalPoses(goalConds)
    if debug('pickGen', skip=skip):
        newBS.draw(prob, 'W')
        debugMsg('pickGen', 'Goal conditions')
    gen = pickGenAux(newBS, obj, confAppr, conf, placeB, graspB, hand, base, prob,
                     goalConds, onlyCurrent=onlyCurrent)
    for x,v in gen:
        if debug('traceGen'):
            (pB, cf, ca) = x
            pose = pB.poseD.mode() if pB else None
            grasp = graspB.grasp.mode() if graspB else None
            pg = (placeB.support.mode(), grasp)
            w = v.weight() if v else None
            trace('    pickGen(%s) viol='%obj, w, '(h,p,g)=', hand, pg, pose)
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
            trace('    pickGen: Held collision')
            if debug('pickGen'):
                newBS.draw(prob, 'W')
                debugMsg('pickGen', 'Held collision.')
            return True            # punt.

    def graspApproachConfGen(firstConf):
        if firstConf:
            yield firstConf
        for c, ca, _ in graspConfGen:
            approached[ca] = c
            yield ca

    def currentGraspFeasible():
        wrist = objectGraspFrame(pbs, graspB, placeB)

    shw = shadowWidths(placeB.poseD.var, placeB.delta, prob)
    if any(w > t for (w, t) in zip(shw, pbs.domainProbs.pickTolerance)):
        print 'pickGen shadow widths', shw
        print 'poseVar', placeB.poseD.var
        print 'delta', placeB.delta
        print 'prob', prob
        debugMsg('pickGen', 'Shadow widths exceed tolerance in pickGen')
        return

    skip = (fbch.inHeuristic and not debug('inHeuristic'))
    shWorld = pbs.getShadowWorld(prob)
    approached = {}
    rm = pbs.getRoadMap()
    if placeB.poseD.mode() != None: # otherwise go to regrasp
        if not base:
            # Try current conf
            (x,y,th) = pbs.conf['pr2Base']
            currBasePose = util.Pose(x, y, 0.0, th)
            ans = graspConfForBase(pbs, placeB, graspB, hand, currBasePose, prob)
            if ans:
                (c, ca, viol) = ans
                yield (placeB, c, ca), viol        
        graspConfGen = potentialGraspConfGen(pbs, placeB, graspB, conf, hand, base, prob)
        firstConf = next(graspApproachConfGen(None), None)
        if (not firstConf) or (firstConf and checkInfeasible(firstConf)):
            debugMsg('pickGen', 'No potential grasp confs, will need to regrasp')
        else:
            targetConfs = graspApproachConfGen(firstConf)
            batchSize = pickPlaceBatchSize
            batch = 0
            while True:
                # Collect the next batach of trialConfs
                batch += 1
                trialConfs = []
                count = 0
                minCost = 1e6
                for ca in targetConfs:       # targetConfs is a generator
                    viol = pickable(ca, approached[ca], placeB, graspB)
                    if viol:
                        trialConfs.append((viol.weight(), viol, ca))
                        minCost = min(viol.weight(), minCost)
                    else:
                        continue
                    count += 1
                    if count == batchSize or minCost == 0: break
                if count == 0: break
                trialConfs.sort()
                for _, viol, ca in trialConfs:
                    c = approached[ca]
                    ans = (placeB, c, ca)
                    if debug('pickGen', skip=skip):
                        drawPoseConf(pbs, placeB, c, ca, prob, 'W', color = 'navy')
                        debugMsg('pickGen', ('-> currently graspable', ans), ('viol', viol))
                        wm.getWindow('W').clear()
                    yield ans, viol
        if onlyCurrent:
            tracep('pickGen', 'onlyCurrent: out of values')
            return
        
    # Try a regrasp... that is place the object somewhere else where it can be grasped.
    if fbch.inHeuristic:
        return
    print 'Calling for regrasping... h=', fbch.inHeuristic
    raw_input('Regrasp?')
    tracep('pickGen', 'Regrasp?')
    shWorld = pbs.getShadowWorld(prob)
    # !! Needs to look for plausible regions...
    regShapes = regShapes = [shWorld.regionShapes[region] for region in pbs.awayRegions()]
    plGen = placeInGenTop((obj, regShapes, graspB, placeB, None, prob),
                          goalConds, pbs, [],
                          regrasp = True,
                          )
    for pl, viol in plGen:
        (pB, gB, cf, ca) = pl
        v = pickable(ca, cf, pB, gB)
        if debug('pickGen'):
            pbs.draw(prob, 'W')
            ca.draw('W', attached=shWorld.attached)
        if not v:
            if debug('pickGen'):
                print 'pickable viol=', v
                debugMsg('pickGen', 'Regrasp is NOT pickable')
            continue
        else:
            if debug('pickGen'):
                print 'pickable viol=', v
                debugMsg('pickGen', 'Regrasp is pickable')
        yield (pB, cf, ca), viol
    tracep('pickGen', 'out of values')

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
    if debug('placeGen'):
        print 'goalConds in placeGen'
        for x in goalConds: print x
    for ans, viol, hand in gen:
        print 'placeGen ->', hand
        (gB, pB, c, ca) = ans
         # LPK: added return of pose mode and face, in the case they
         # weren't boudn coming in.
        yield (hand, gB.poseD.mode().xyztTuple(), gB.grasp.mode(), c, ca,
               pB.poseD.mode().xyztTuple(), pB.support.mode())

def placeGenGen(args, goalConds, bState, outBindings):
    (obj, hand, poses, support, objV, graspV, objDelta, graspDelta, confDelta,
     prob) = args

    base = sameBase(goalConds)
    if base:
        print('Same base constraint in placeGen')
    if goalConds:
        if  getConf(goalConds, None):
            tracep('placeGen', '    conf is already specified')
            return

    pbs = bState.pbs.copy()
    world = pbs.getWorld()

    if poses == '*' or isVar(poses) or support == '*' or isVar(support):
        print '    placeGen with unspecified pose'
        hand = 'left'
        if goalConds:
            if base:
                tracep('placeGen', '    same base constraint, failing')
                return
            if getConf(goalConds, None):
                tracep('placeGen', '    goal conf specified, failing')
                return
            holding = dict(getHolding(goalConds))
            if 'left' in holding and 'right' in holding:
                tracep('placeGen', '    both hands are already Holding')
                return
            elif 'left' in holding:
                hand = 'right'
            else:
                hand = 'left'
        # Just placements specified in goal (and excluding obj)

        # placeInGenAway does not do this when calling placeGen
        newBS = pbs.copy()
        newBS = newBS.updateFromGoalPoses(goalConds, updateConf=False)
        newBS = newBS.excludeObjs([obj])
        for ans,v in placeInGenAway((obj, objDelta, prob),
                                    goalConds, newBS, outBindings, hand):
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

    assert not (pbs.useRight == False and hand == 'right')

    mustUseLeft = (hand == 'left' or not pbs.useRight)
    mustUseRight = (hand == 'right')
    leftGen = placeGenTop((obj, graspB, placeBs, 'left', base, prob),
                                 goalConds, pbs, outBindings)
    rightGen = placeGenTop((obj, graspB, placeBs, 'right', base, prob),
                                 goalConds, pbs, outBindings)

    if mustUseLeft:
        gen = leftGen
    elif mustUseRight:
        gen = rightGen
    else:
        gen = roundrobin(leftGen, rightGen)

    for ans in gen:
        yield ans

# returns values for (?graspPose, ?graspFace, ?conf, ?confAppr)
def placeGenTop(args, goalConds, pbs, outBindings, regrasp=False, away=False, update=True):
    (obj, graspB, placeBs, hand, base, prob) = args
    trace('placeGen(%s,%s) h='%(obj,hand), fbch.inHeuristic)

    startTime = time.clock()
    skip = (fbch.inHeuristic and not debug('inHeuristic'))
    debugMsgSkip('placeGen', skip,
                 zip(('obj', 'graspB', 'placeBs', 'hand', 'prob'), args),
                 ('goalConds', goalConds),
                 ('moveObjBs', pbs.moveObjBs),
                 ('fixObjBs', pbs.fixObjBs),
                 ('held', (pbs.held['left'].mode(),
                           pbs.held['right'].mode(),
                           pbs.graspB['left'],
                           pbs.graspB['right'])))
    if obj == 'none' or not placeBs:
        tracep('placeGen', '    objs is none or no placeB')
        return
    if goalConds:
        if getConf(goalConds, None) and not away:
            tracep('placeGen', '    goal conf specified and not away')
            return
        for (h, o) in getHolding(goalConds):
            if h == hand:
                tracep('placeGen', '    this hand is already Holding')
                return

    conf = None
    confAppr = None
    # Set up pbs
    newBS = pbs.copy()
    # Just placements specified in goal (and excluding obj)
    if update:                          # could be done by caller
        newBS = newBS.updateFromGoalPoses(goalConds, updateConf=not away)
        newBS = newBS.excludeObjs([obj])
    if debug('placeGen', skip=skip):
        for gc in goalConds: print gc
        newBS.draw(prob, 'W')
        debugMsg('placeGen', 'Goal conditions')
    gen = placeGenAux(newBS, obj, confAppr, conf, placeBs.copy(),
                      graspB, hand, base, prob,
                      regrasp=regrasp, pbsOrig = pbs)

    # !! double check reachObst collision?
    for x,v in gen:
        if debug('traceGen'):
            (gB, pB, c, ca) = x
            pose = pB.poseD.mode() if pB else None
            grasp = gB.grasp.mode() if gB else None
            pg = (pB.support.mode(), grasp)
            w = v.weight() if v else None
            trace('    placeGen(%s,%s) viol='%(obj,hand), w, '(p,g)=', pg,
                  pose, '(t=', time.clock()-startTime, ')')
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
                       (None, None))[0] \
                  for gBO in gBOther]
        if any(other):
            if debug('placeGen', skip=skip):
                print 'Regraspable', pB.poseD.mode(), [gBO.grasp.mode() for gBO in gBOther]
                for (c, ca, v) in other:
                    c.draw('W', 'green')
                    debugMsg('placeGen', 'other')
            regraspablePB[pB] = True
            return True
        else:
            regraspablePB[pB] = False
            if debug('placeGen', skip=skip): print 'Not regraspable'
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
                debugMsg('placeGen', 'current grasp is a match',
                          ('curr', currGraspB), ('desired', gB))
                return 0

        pB = pbsOrig.getPlaceB(obj, default=False) # check we know where obj is.
        if pbsOrig and pbsOrig.held[hand].mode() != obj and pB:
            if next(potentialGraspConfGen(pbsOrig, pB, gB, conf, hand, base,
                                          prob, nMax=1),
                    (None, None))[0]:
                return 1
            else:
                return 2
        else:
            return 1

    def placeApproachConfGen(grasps):
        placeBsCopy = placeBs.copy()
        for pB in placeBsCopy:          # re-generate
            for gB in grasps:
                if debug('placeGen', skip=skip):
                    print 'placeGen: considering grasps for ', pB
                    print 'placeGen: for grasp class', gB.grasp
                    print 'placeBsCopy.values', len(placeBsCopy.values)
                if regrasp:
                    checkRegraspable(pB)
                graspConfGen = potentialGraspConfGen(pbs, pB, gB, conf, hand, base, prob)
                count = 0
                for c,ca,_ in graspConfGen:
                    if debug('placeGen', skip=skip):
                        c.draw('W', 'orange')
                    approached[ca] = c
                    count += 1
                    context[ca] = (pB, gB)
                    debugMsg('placeGen', 'Yielding conf')
                    yield ca
                    # if count > 2: break # !! ??
        if debug('placeGen', skip=skip):
            print '    placeGen: found', count, 'confs'

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

    skip = (fbch.inHeuristic and not debug('inHeuristic'))
    approached = {}
    context = {}
    regraspablePB = {}
    rm = pbs.getRoadMap()
    if regrasp:
         graspBOther = copy.copy(graspB)
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
        batchSize = pickPlaceBatchSize
        batch = 0
        while True:
            # Collect the next batach of trialConfs
            batch += 1
            trialConfs = []
            count = 0
            minCost = 1e6
            for ca in targetConfs:   # targetConfs is a generator
                viol = placeable(ca, approached[ca])
                if viol:
                    cost = viol.weight() + gCost + regraspCost(ca)
                    minCost = min(cost, minCost)
                    trialConfs.append((cost, viol, ca))
                else:
                    if debug('placeable'):
                        print 'Failure of placeable'
                        save = glob.debugOn[:]
                        glob.debugOn.extend(['successors', 'confReachViol', 'confReachViolGen',
                                             'minViolPath', 'canPickPlaceTest', 'addToCluster'])
                        placeable(ca, approached[ca])
                        glob.debugOn = save
                        raw_input('Continue?')
                    continue
                count += 1
                if count == batchSize or minCost == 0: break
            if count == 0: break
            trialConfs.sort()
            for _, viol, ca in trialConfs:
                (pB, gB) = context[ca]
                c = approached[ca]
                ans = (gB, pB, c, ca)
                if debug('placeGen', skip=skip):
                    drawPoseConf(pbs, pB, c, ca, prob, 'W', color='magenta')
                    debugMsg('placeGen', ('->', ans), ('viol', viol))
                    wm.getWindow('W').clear()
                print 'Trying grasp', gB
                yield ans, viol
    tracep('placeGen', 'out of values')

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
    if not isinstance(region, (list, tuple, frozenset)):
        regions = frozenset([region])
    elif len(region) == 0:
        raise Exception, 'need a region to place into'
    else:
        regions = frozenset(region)

    skip = (fbch.inHeuristic and not debug('inHeuristic'))
    pbs = bState.pbs.copy()
    world = pbs.getWorld()

    # !! Should derive this from the clearance in the region
    domainPlaceVar = bState.domainProbs.obsVarTuple 

    # Reasonable?
    graspV = domainPlaceVar
    graspDelta = bState.domainProbs.pickStdev
    pose = None
    if pbs.getPlaceB(obj, default=False):
        # If it is currently placed, use that support
        support = pbs.getPlaceB(obj).support
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

    graspB = ObjGraspB(obj, world.getGraspDesc(obj), None,
                       PoseD(None, graspV), delta=graspDelta)

    # Put all variance on the object being placed
    placeB = ObjPlaceB(obj, world.getFaceFrames(obj), support,
                       PoseD(pose, var), delta=delta)

    shWorld = pbs.getShadowWorld(prob)
    regShapes = [shWorld.regionShapes[region] for region in regions]
    if debug('placeInGen'):
        shWorld.draw('W')
        for rs in regShapes: rs.draw('W', 'purple')
        debugMsgSkip('placeInGen', skip, 'Target region in purple')

    poseBels = getGoalPoseBels(goalConds, world.getFaceFrames)
    if obj in poseBels:
        # Object pose is specified in goalConds
        pB = poseBels[obj]
        shw = shadowWidths(pB.poseD.var, pB.delta, prob)
        shwMin = shadowWidths(graspV, graspDelta, prob)
        if any(w > mw for (w, mw) in zip(shw, shwMin)):
            args = (obj, None, pB.poseD.mode().xyztTuple(),
                    support, placeB.poseD.var, graspV,
                    delta, graspDelta, None, prob)
            gen = placeGenGen(args, goalConds, bState, outBindings)
            for ans, v, hand in gen:
                (gB, pB, c, ca) = ans
                if debug('traceGen'):
                    pose = pB.poseD.mode() if pB else None
                    grasp = gB.grasp.mode() if gB else None
                    sup = pB.support.mode() if pB else None
                    pg = (sup, grasp)
                    print '    placeInGen(%s,%s) h='%(obj,[x.name() \
                                                           for x in regShapes]), \
                          v.weight() if v else None, '(p,g)=', pg, pose
                yield (pB, gB, c, ca)
            return
        else:
            # If pose is specified and variance is small, return
            return

    gen = placeInGenTop((obj, regShapes, graspB, placeB, None, prob),
                          goalConds, pbs, outBindings, away = away, update=update)
    for ans in gen:
        yield ans

def placeInGenAway(args, goalConds, pbs, outBindings, hand='left'):
    # !! Should search over regions and hands
    (obj, delta, prob) = args
    if not pbs.awayRegions():
        raw_input('Need some awayRegions')
        return
    domainPlaceVar = pbs.domainProbs.obsVarTuple     
    for ans,v in placeInRegionGenGen((obj, pbs.awayRegions(),
                                      domainPlaceVar, delta, prob),
                                     # preserve goalConds to get reachObsts
                                     goalConds, pbs, [], away=True, update=False):
        yield ans,v

placeInGenMaxPoses  = 50
placeInGenMaxPosesH = 10

def placeInGenTop(args, goalConds, pbs, outBindings,
                  regrasp=False, away = False, update=True):
    (obj, regShapes, graspB, placeB, base, prob) = args
    trace('placeInGen(%s,%s) h='%(obj,[x.name() for x in regShapes]), fbch.inHeuristic)
    skip = (fbch.inHeuristic and not debug('inHeuristic'))
    debugMsgSkip('placeInGen', skip,
             zip(('obj', 'regShapes', 'graspB', 'placeB', 'prob'), args),
             outBindings)
    if obj == 'none' or not regShapes:
        # Nothing to do
        trace('    placeInGen(%s,%s) h='%(obj,[x.name() for x in regShapes]), fbch.inHeuristic, 'nothing to do')
        return

    if goalConds and getConf(goalConds, None) and not away:
        # if conf is specified, just fail
        tracep('placeInGen', '    conf is specified so failing')
        return

    conf = None
    confAppr = None
    # Obstacles for all Reachable fluents
    reachObsts = getReachObsts(goalConds, pbs)
    if reachObsts == None:
        tracep('placeInGen', '    quitting because no path')
        return
    if debug('placeInGen', skip=skip) or debug('reachObsts', skip=skip):
        pbs.draw(prob, 'W')
        for _, obst in reachObsts: obst.draw('W', 'brown')
        raw_input('%d reachObsts - in brown'%len(reachObsts))

    newBS = pbs.copy()           #  not necessary
    # Shadow (at origin) for object to be placed.
    domainPlaceVar = newBS.domainProbs.obsVarTuple 
    pB = placeB.modifyPoseD(var=domainPlaceVar)
    nPoses = placeInGenMaxPosesH if fbch.inHeuristic else placeInGenMaxPoses
    poseGenLeft = Memoizer('regionPosesLeft',
                           potentialRegionPoseGen(newBS, obj, pB, graspB, prob, regShapes,
                                                  reachObsts, 'left', base,
                                                  maxPoses=nPoses))
    poseGenRight = Memoizer('regionPosesRight',
                            potentialRegionPoseGen(newBS, obj, pB, graspB, prob, regShapes,
                                                   reachObsts, 'right', base,
                                                   maxPoses=nPoses))

    mainLeftGen = placeInGenAux(newBS, poseGenLeft, goalConds, confAppr,
                                conf, placeB, graspB, 'left', base, prob,
                                regrasp=regrasp, away=away, update=update)
    mainRightGen = placeInGenAux(newBS, poseGenRight, goalConds, confAppr,
                                 conf, placeB, graspB, 'right', base, prob,
                                 regrasp=regrasp, away=away, update=update)
    mainGen = roundrobin(mainLeftGen, mainRightGen) \
               if pbs.useRight else mainLeftGen

    # Picks among possible target poses and then try to place it in region
    for ans,v in mainGen:
        if debug('traceGen'):
            (pB, gB, c, ca) = ans
            pose = pB.poseD.mode() if pB else None
            grasp = gB.grasp.mode() if gB else None
            sup = pB.support.mode() if pB else None
            pg = (sup, grasp)
            print '    placeInGen(%s,%s) h='%(obj,[x.name() \
                                                   for x in regShapes]), \
                  v.weight() if v else None, '(p,g)=', pg, pose
        if debug('placeInGen'):
            c.draw('W', 'green')
            raw_input('placeInGen result')
        #val.append((ans, v))
        yield ans,v

# Don't try to place all objects at once
def placeInGenAux(pbs, poseGen, goalConds, confAppr, conf, placeB, graspB,
                  hand, base, prob, regrasp=False, away=False, update=True):

    def placeBGen():
        for pose in poseGen.copy():
            yield placeB.modifyPoseD(mu=pose)
    skip = (fbch.inHeuristic and not debug('inHeuristic'))
    tries = 0
    gen = Memoizer('placeBGen_placeInGenAux1', placeBGen())
    for ans, viol, hand in placeGenTop((graspB.obj, graspB, gen, hand, base, prob),
                               goalConds, pbs, [], regrasp=regrasp, away=away, update=update):
        (gB, pB, cf, ca) = ans
        if debug('placeInGen', skip=skip):
            drawPoseConf(pbs, pB, cf, ca, prob, 'W', 'blue')
            debugMsg('placeInGen', ('-> cyan', ans))
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
                       # Pretend that the object has bigger delta
                       delta=tuple([o+l for (o,l) in zip(objDelta, lookDelta)]))

    # Don't try to look at the whole shadow
    placeB = placeB.modifyPoseD(var = (0.0001, 0.0001, 0.0001, 0.0005))


    for ans, viol in lookGenTop((obj, placeB, lookDelta, base, prob),
                                goalConds, pbs, outBindings):
        yield ans

def lookGenTop(args, goalConds, pbs, outBindings):

    def testFn(c):
        print 'Trying base conf', c['pr2Base']
        obst_rob = obst + [c.placement(shWorld.attached)]
        return visible(shWorld, c, sh, obst_rob, prob, moveHead=True)[0]

    (obj, placeB, lookDelta, base, prob) = args
    trace('lookGen(%s) h='%obj, fbch.inHeuristic)
    skip = (fbch.inHeuristic and not debug('inHeuristic'))
    newBS = pbs.copy()
    newBS = newBS.updateFromGoalPoses(goalConds)
    newBS.addAvoidShadow([obj])
    if placeB.poseD.mode() == None:
        tracep('lookGen', '    object is in the hand')
        return
    newBS.updatePermObjPose(placeB)
    rm = newBS.getRoadMap()
    shWorld = newBS.getShadowWorld(prob)
    if any(shWorld.attached.values()):
        trace('    attached=', shWorld.attached)
    shName = shadowName(obj)
    sh = shWorld.objectShapes[shName]
    obst = [s for s in shWorld.getNonShadowShapes() if s.name() != obj ]

    if goalConds and getConf(goalConds, None):
        # if conf is specified, just fail
        tracep('lookGen', '    Conf is specified so failing')
        return

    if obj in [newBS.held[hand].mode() for hand in ['left', 'right']]:
        tracep('lookGen', '    object is in the hand')
        return

    if base:
        # !! Could be more creative about the conf
        conf = rm.homeConf.set('pr2Base', base)
        path, viol = canReachHome(newBS, conf, prob, Violations(), moveBase=False)
        trace('    lookGen(%s) specified base viol='%obj, viol.weight() if viol else None)
        if not path:
            newBS.draw(prob, 'W')
            conf.draw('W', 'cyan')
            raw_input('Failed to find a path to look conf (in cyan) with specified base.')
            return
        conf = path[-1]
        if testFn(conf):
            lookConf = lookAtConfCanView(newBS, prob, conf, sh)
            if lookConf:
                trace('    Found a path to look conf with specified base.')
                if debug('lookGen', skip=skip):
                    newBS.draw(prob, 'W')
                    lookConf.draw('W', color='cyan', attached=shWorld.attached)
                    debugMsg('lookGen', ('-> cyan', lookConf.conf))
                yield (lookConf,), viol
        return

    # LPK did this
    tempPlaceB = copy.copy(placeB)
    tempPlaceB.poseD = copy.copy(placeB.poseD)
    tempPlaceB.modifyPoseD(var = pbs.domainProbs.obsVarTuple)
    # be smarter about this?
    tempPlaceB.delta = (.01, .01, .01, .01)
    shape = tempPlaceB.shadow(newBS.getShadowWorld(prob))

    # Check current conf
    curr = newBS.conf
    lookConf = lookAtConfCanView(newBS, prob, curr, shape)
    if lookConf and testFn(lookConf):
        trace('    Using current conf.')
        yield (lookConf,), rm.confViolations(curr, newBS, prob)
        
    world = newBS.getWorld()
    if obj in world.graspDesc and not fbch.inHeuristic:
        graspVar = 4*(0.001,)
        graspDelta = 4*(0.001,)   # put back to prev value
        graspB = ObjGraspB(obj, world.getGraspDesc(obj), None,
                           PoseD(None, graspVar), delta=graspDelta)
        # Use pbs to generate candidate confs, since they will need to
        # collide with shadow of obj.
        for gB in graspGen(pbs, obj, graspB):
            for hand in ['left', 'right']:
                for ans, viol in pickGenTop((obj, gB, tempPlaceB, hand, base,
                                             prob),
                                            goalConds, pbs, outBindings,
                                            onlyCurrent=True):
                    (pB, c, ca) = ans   # pB should be placeB
                    lookConf = lookAtConfCanView(pbs, prob, ca, shape)
                    if not lookConf:
                        trace('    lookGen(%s) canView failed clear')
                        continue
                    viol = rm.confViolations(lookConf, pbs, prob)
                    if testFn(lookConf):
                        trace('    lookGen(%s) canView cleared viol='%obj, viol.weight() if viol else None)
                        yield (lookConf,), viol
    lookConfGen = potentialLookConfGen(rm, sh, maxLookDist) # look unconstrained by base
    for ans in rm.confReachViolGen(lookConfGen, newBS, prob,
                                   testFn = testFn):
        viol, cost, path = ans
        trace('    lookGen(%s) viol='%obj, viol.weight() if viol else None)
        if not path:
            tracep('lookGen', 'Failed to find a path to look conf.')

            raw_input('Failed to find a path to look conf.')

            continue
        conf = path[-1]
        lookConf = lookAtConfCanView(newBS, prob, conf, sh)
        if lookConf:
            trace('    lookGen(%s) general conf viol='%obj, viol.weight() if viol else None)
            if debug('lookGen', skip=skip):
                pbs.draw(prob, 'W')
                lookConf.draw('W', color='cyan', attached=shWorld.attached)
                debugMsg('lookGen', ('-> cyan', lookConf.conf))
            yield (lookConf,), viol

def lookAtConfCanView(pbs, prob, conf, shape, hands=['left', 'right']):
    lookConf = lookAtConf(conf, shape)
    if not fbch.inHeuristic:
        for hand in hands:
            path = canView(pbs, prob, lookConf, hand, shape)
            if not path:
                raw_input('lookAtConfCanView failed path')
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
                attached[hand] = Box(0.1,0.05,0.1, None, name='virtualObject').applyLoc(gripperTip)
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
    placements = {}
    handObj = {}
    trace('lookHandGen(%s) h='%obj, fbch.inHeuristic)
    skip = (fbch.inHeuristic and not debug('inHeuristic'))
    newBS = pbs.copy()
    newBS = newBS.updateFromGoalPoses(goalConds)
    newBS.updateHeldBel(graspB, hand)
    shWorld = newBS.getShadowWorld(prob)
    if fbch.inHeuristic:
        lookConf = lookAtConf(newBS.conf, objInHand(newBS.conf, hand))
        if lookConf:
            yield (lookConf,), Violations()
        return
    if goalConds and getConf(goalConds, None):
        debugMsg('lookHandGen', 'conf is specified')
        # if conf is specified, just fail
        return
    rm = newBS.getRoadMap()
    obst = [s for s in shWorld.getNonShadowShapes() if s.name() != obj ]
    lookConfGen = potentialLookHandConfGen(newBS, prob, hand)
    for ans in rm.confReachViolGen(lookConfGen, newBS, prob,
                                   startConf = newBS.conf,
                                   testFn = testFn):
        viol, cost, path = ans
        trace('    lookHandGen(%s) viol='%obj, viol.weight() if viol else None)
        if not path:
            tracep('lookHandGen', 'Failed to find a path to look conf.')
            continue
        conf = path[-1]
        lookConf = lookAtConf(conf, objInHand(conf, hand))
        if lookConf is None: continue # can't look at it.
        if debug('lookHandGen', skip=skip):
            pbs.draw(prob, 'W')
            lookConf.draw('W', color='cyan', attached=shWorld.attached)
            debugMsg('lookHandGen', ('-> cyan', lookConf.conf))
        yield (lookConf,), viol

def moveOut(pbs, prob, obst, delta, goalConds):
    if debug('traceGen') or debug('canReachGen'):
        print '    canReachGen() obst:', obst
    domainPlaceVar = pbs.domainProbs.obsVarTuple 
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
    # Debug
    debugMsg('canReachGen', args)
    # Call
    def violFn(pbs):
        path, viol = canReachHome(pbs, conf, prob, Violations())
        return viol
    lookVar = bState.domainProbs.obsVarTuple
    for ans in canXGenTop(violFn, (cond, prob, lookVar),
                          goalConds, pbs, outBindings, 'canReachGen'):
        debugMsg('canReachGen', ('->', ans))
        yield ans
    debugMsg('canReachGen', 'exhausted')

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
    cppFluent = Bd([CanPickPlace([preconf, ppconf, hand, obj, pose, realPoseVar, poseDelta, poseFace,
                                  graspFace, graspMu, graspVar, graspDelta, op, cond]), True, prob], True)
    poseFluent = B([Pose([obj, poseFace]), pose, realPoseVar, poseDelta, prob], True)
    goalConds = goalConds + [cppFluent, poseFluent]
    # Debug
    skip = (fbch.inHeuristic and not debug('inHeuristic'))
    debugMsg('canPickPlaceGen', args)
    world = pbs.getWorld()
    lookVar = bState.domainProbs.obsVarTuple
    graspB = ObjGraspB(obj, world.getGraspDesc(obj), graspFace,
                       PoseD(graspMu, graspVar), delta= graspDelta)
    placeB = ObjPlaceB(obj, world.getFaceFrames(obj), poseFace,
                       PoseD(pose, realPoseVar), delta=poseDelta)
    # Initial test
    def violFn(pbs):
        return canPickPlaceTest(pbs, preconf, ppconf, hand,
                                graspB, placeB, prob, op=op)
    lookVar = bState.domainProbs.obsVarTuple
    for ans in canXGenTop(violFn, (cond, prob, lookVar),
                          goalConds, pbs, outBindings, 'canPickPlaceGen'):
        debugMsg('canPickPlacechGen', ('->', ans))
        yield ans
    debugMsg('canPickPlaceGen', 'exhausted')

def canXGenTop(violFn, args, goalConds, pbs, outBindings, tag):
    (cond, prob, lookVar) = args
    trace('%s() h='%tag, fbch.inHeuristic)
    skip = (fbch.inHeuristic and not debug('inHeuristic'))
    # Set up PBS
    newBS = pbs.copy()
    newBS = newBS.updateFromGoalPoses(goalConds)
    newBS = newBS.updateFromGoalPoses(cond, permShadows=True)
    # Initial test
    viol = violFn(newBS)
    if debug(tag):
        newBS.draw(prob, 'W')
    debugMsg(tag, ('viol', viol))
    if not viol:                  # hopeless
        tracep(tag, 'Impossible dream')
        return
    if viol.empty():
        tracep(tag, 'No obstacles or shadows; returning')
        return
    # If possible, it might be better to make the deltas big; but we
    # have to be sure to use the same delta when generating paths.
    objBMinDelta = newBS.domainProbs.minDelta
    objBMinVar = newBS.domainProbs.obsVarTuple
    objBMinProb = 0.95
    lookDelta = objBMinDelta
    moveDelta = objBMinDelta
    shWorld = newBS.getShadowWorld(prob)
    fixed = shWorld.fixedObjects
    # Try to fix one of the violations if any...
    obstacles = [o.name() for o in viol.allObstacles() \
                 if o.name() not in fixed]
    shadows = [sh.name() for sh in viol.allShadows() \
               if not sh.name() in fixed]
    if not (obstacles or shadows):
        debugMsg(tag, 'No movable obstacles or shadows to fix')
        return       # nothing available
    if obstacles:
        obstacleName = obstacles[0]
        for ans in moveOut(newBS, prob, obstacleName, moveDelta, goalConds):
            yield ans
        return
    if shadows:
        shadowName = shadows[0]
        obst = objectName(shadowName)
        placeB = newBS.getPlaceB(obst)
        # !! It could be that sensing is not good enough to reduce the
        # shadow so that we can actually achieve goal
        newBS2 = newBS.copy()
        placeB2 = placeB.modifyPoseD(var = lookVar)
        placeB2.delta = lookDelta
        newBS2.updatePermObjPose(placeB2)
        viol2 = violFn(newBS2)
        if viol2:
            if shadowName in [x.name() for x in viol2.allShadows()]:
                print 'could not reduce the shadow for', obst, 'enough to avoid'
                drawObjAndShadow(newBS, placeB, prob, 'W', color='red')
                print 'brown is as far as it goes'
                drawObjAndShadow(newBS2, placeB2, prob, 'W', color='brown')
                raw_input('Go?')
            if debug(tag, skip=skip):
                drawObjAndShadow(newBS, placeB, prob, 'W', color='red')
                debugMsg(tag, 'Trying to reduce shadow (on W in red) %s'%obst)
                trace('    %s() shadow:'%tag, obst)
            yield (obst, placeB.poseD.mode().xyztTuple(), placeB.support.mode(),
                   lookVar, lookDelta)
        # Either reducing the shadow is not enough or we failed and
        # need to move the object (if it's movable).
        if obst not in newBS.fixObjBs:
            for ans in moveOut(newBS, prob, obst, moveDelta, goalConds):
                yield ans

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
    trace('canSeeGen(%s) h='%obj, fbch.inHeuristic)
    skip = (fbch.inHeuristic and not debug('inHeuristic'))
    debugMsgSkip('canSeeGen', skip, ('args', args))

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
        tracep('canSeeGen', 'no occluders')
        return
    obst = occluders[0] # !! just pick one
    moveDelta = (0.01, 0.01, 0.01, 0.02)
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

