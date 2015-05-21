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
from pr2PlanBel import getConf
from shapes import Box

Ident = util.Transform(np.eye(4))            # identity transform

import pr2GenAux
from pr2GenAux import *

#  How many candidates to generate at a time...  Larger numbers will
#  generally lead to better solutions.
pickPlaceBatchSize = 1

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
    
    pbs = bState.pbs
    (obj, hand) = args
    trace('easyGraspGen(%s,%s) h='%(obj,hand), fbch.inHeuristic)
    if obj == 'none' or (goalConds and getConf(goalConds, None)):
        trace('easyGraspGen', 'obj is none or conf in goal conds')
        return
    prob = 0.75
    # Set up pbs
    newBS = pbs.copy()
    # Just placements specified in goal
    newBS = newBS.updateFromGoalPoses(goalConds) if goalConds else newBS
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
    graspB.grasp = UniformDist(range(len(graspB.graspDesc)))
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

def pickGen(args, goalConds, bState, outBindings, onlyCurrent = False):
    (obj, graspFace, graspPose,
     objV, graspV, objDelta, confDelta, graspDelta, hand, prob) = args

    base = sameBase(goalConds)          # base is (x, y, th)
    if base:
        print('Same base constraint in pickGen')

    debugMsg('pickGen', 'args', args, ('base', base))

    world = bState.pbs.getWorld()
    graspB = ObjGraspB(obj, world.getGraspDesc(obj), graspFace,
                       PoseD(util.Pose(*graspPose), graspV), delta=graspDelta)
    placeB = ObjPlaceB(obj, world.getFaceFrames(obj), None,
                       PoseD(None,  objV), delta=objDelta)
    for ans, viol in pickGenTop((obj, graspB, placeB, hand, base, prob,),
                                goalConds, bState.pbs, outBindings, onlyCurrent):
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
    if goalConds and getConf(goalConds, None):
        # if conf is specified, just fail
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
    newBS = newBS.updateFromGoalPoses(goalConds) if goalConds else newBS
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
    def pickable(ca, c, pB):
        return canPickPlaceTest(pbs, ca, c, hand, graspB, pB, prob, op='pick')

    def checkInfeasible(conf):
        newBS = pbs.copy()
        newBS.updateConf(conf)
        newBS.updateHeldBel(graspB, hand)
        viol, (rv, hv) = rm.confViolations(conf, newBS, prob)
        if not viol:                # was valid when not holding, so...
            trace('    pickGen: Held collision')
            if True: #debug('pickGen'):
                newBS.draw(prob, 'W')
                raw_input('held collision')
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
        debugMsg('pickTol', 'Shadow widths exceed tolerance in pickGen')
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
                    viol = pickable(ca, approached[ca], placeB)
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
        v = pickable(ca, cf, pB)
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
        # The regrasp option should never be cheaper than the non-regrasp.
        # penalty = viol.weight()+1 if viol else 1
        # debugMsg('pickGen', ('Adding penalty', penalty, 'to', viol.penalty, viol))
        yield (pB, cf, ca), viol
    tracep('pickGen', 'out of values')

# Returns (hand, graspMu, graspFace, graspConf,  preConf)
def placeGen(args, goalConds, bState, outBindings):
    (obj, hand, poses, support, objV, graspV, objDelta, graspDelta, confDelta,
     prob) = args

    base = sameBase(goalConds)
    if base:
        print('Same base constraint in placeGen')

    if not isinstance(poses[0], (list, tuple, frozenset)):
        poses = frozenset([poses])

    world = bState.pbs.getWorld()
    graspB = ObjGraspB(obj, world.getGraspDesc(obj), None,
                       PoseD(None, graspV), delta=graspDelta)
    def placeBGen():
        for pose in poses:
            yield ObjPlaceB(obj, world.getFaceFrames(obj), support,
                            PoseD(pose, objV), delta=objDelta)
    placeBs = Memoizer('placeBGen_placeGen', placeBGen())

    assert not (bState.pbs.useRight == False and hand == 'right')

    mustUseLeft = (hand == 'left' or not bState.pbs.useRight)
    mustUseRight = (hand == 'right')
    leftGen = placeGenTop((obj, graspB, placeBs, 'left', base, prob),
                                 goalConds, bState.pbs, outBindings)
    rightGen = placeGenTop((obj, graspB, placeBs, 'right', base, prob),
                                 goalConds, bState.pbs, outBindings)

    if mustUseLeft:
        gen = leftGen
    elif mustUseRight:
        gen = rightGen
    else:
        gen = roundrobin(leftGen, rightGen)

    for ans, viol, hand in gen:
        print 'placeGen ->', hand
        (gB, pB, c, ca) = ans
        yield (hand, gB.poseD.mode().xyztTuple(), gB.grasp.mode(), c, ca)

# returns values for (?graspPose, ?graspFace, ?conf, ?confAppr)
def placeGenTop(args, goalConds, pbs, outBindings, regrasp=False, away=False):
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
    if goalConds and getConf(goalConds, None) and not away:
        # if conf is specified, just fail
        tracep('placeGen', '    goal conf specified and not away')
        return

    # LPK!!  Changed this to allow regrasping.  Later code will be
    # sure to generate the current grasp first.
    graspB.grasp = UniformDist(range(0,len(graspB.graspDesc)))
        
    conf = None
    confAppr = None
    # Set up pbs
    newBS = pbs.copy()
    # Just placements specified in goal (and excluding obj)
    newBS = newBS.updateFromGoalPoses(goalConds, updateConf=not away) if goalConds else newBS
    newBS = newBS.excludeObjs([obj])
    if debug('placeGen', skip=skip):
        for gc in goalConds: print gc
        newBS.draw(prob, 'W')
        debugMsg('placeGen', 'Goal conditions')
    gen = placeGenAux(newBS, obj, confAppr, conf, placeBs.copy(), graspB, hand, base, prob,
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
    def placeable(ca, c):
        (pB, gB) = context[ca]
        ans = canPickPlaceTest(pbs, ca, c, hand, gB, pB, prob, op='place')
        return ans

    def checkRegraspable(pB):
        if pB in regraspablePB:
            return regraspablePB[pB]
        other =  [next(potentialGraspConfGen(pbs, pB, gBO, conf, hand, base, prob, nMax=1),
                       (None,None))[0] \
                  for gBO in gBOther]
        if any(other):
            if debug('placeGen', skip=skip):
                print 'Regraspable', pB.poseD.mode(), [gBO.grasp.mode() for gBO in gBOther]
                for x in other:
                    x.draw('W', 'green')
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
                return 0

        pB = pbsOrig.getPlaceB(obj, default=False) # check we know where obj is.
        if pbsOrig and pbsOrig.held[hand].mode() != obj and pB:
            if next(potentialGraspConfGen(pbsOrig, pB, gB, conf, hand, base,
                                          prob, nMax=1),
                    (None,None))[0]:
                return 1
            else:
                return 2
        else:
            return 1

    def placeApproachConfGen(gB):
        placeBsCopy = placeBs.copy()
        assert placeBsCopy.values is placeBs.values

        for pB in placeBsCopy:          # re-generate
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

    grasps = [(checkOrigGrasp(gB), gB) for gB in graspGen(pbs, obj, graspB)]
    grasps.sort()

    for (orig, gB) in grasps: # for now, just grasp type...
        if debug('placeGen', skip=skip):
            print '    placeGen: considering', gB, 'orig', orig

        targetConfs = placeApproachConfGen(gB)
        batchSize = pickPlaceBatchSize
        batch = 0
        while True:
            # Collect the next batach of trialConfs
            batch += 1
            trialConfs = []
            count = 0
            minCost = 1e6
            for ca in targetConfs:       # targetConfs is a generator
                viol = placeable(ca, approached[ca])
                if viol:
                    cost = viol.weight() + regraspCost(ca)
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
                yield ans, viol
    tracep('placeGen', 'out of values')

# Return objPose, poseFace.
def placeInRegionGen(args, goalConds, bState, outBindings, away = False):
    (obj, region, var, delta, prob) = args

    base = sameBase(goalConds)
    if base:
        raw_input('Same base constraint in placeInRegionGen')

    if not isinstance(region, (list, tuple, frozenset)):
        regions = frozenset([region])
    elif len(region) == 0:
        raise Exception, 'need a region to place into'
    else:
        regions = frozenset(region)

    skip = (fbch.inHeuristic and not debug('inHeuristic'))
    pbs = bState.pbs
    world = pbs.getWorld()

    # !! Should derive this from the clearance in the region
    domainPlaceVar = bState.domainProbs.obsVarTuple 

    # Reasonable?
    graspV = domainPlaceVar
    graspDelta = bState.domainProbs.pickStdev
    pose = None

    if bState.pbs.getPlaceB(obj, default=False):
        # If it is currently placed, use that support
        support = bState.pbs.getPlaceB(obj).support
        pose = pbs.getPlaceB(obj).poseD.mode()
    elif obj == bState.pbs.held['left'].mode():
        attachedShape = pbs.getRobot().attachedObj(pbs.getShadowWorld(prob),
                                                   'left')
        shape = pbs.getWorld().getObjectShapeAtOrigin(obj).\
                                        applyLoc(attachedShape.origin())
        support = supportFaceIndex(shape)
    elif obj == bState.pbs.held['right'].mode():
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

    shWorld = bState.pbs.getShadowWorld(prob)
    regShapes = [shWorld.regionShapes[region] for region in regions]
    if debug('placeInGen'):
        shWorld.draw('W')
        for rs in regShapes: rs.draw('W', 'purple')
        debugMsgSkip('placeInGen', skip, 'Target region in purple')

    gen = placeInGenTop((obj, regShapes, graspB, placeB, base, prob),
                          goalConds, bState.pbs, outBindings, away = away)
        
    for ans, viol in gen:
        (pB, gB, cf, ca) = ans
        yield (pB.poseD.mode().xyztTuple(), pB.support.mode())

def placeInGenAway(args, goalConds, pbs, outBindings):
    # !! Should search over regions and hands
    (obj, delta, prob) = args
    hand = 'left'
    if not pbs.awayRegions():
        raw_input('Need some awayRegions')
        return
    domainPlaceVar = pbs.domainProbs.obsVarTuple     
    for ans in placeInRegionGen((obj, pbs.awayRegions(),
                                          domainPlaceVar,
                                          delta, prob),
                          # preserve goalConds to get reachObsts
                          goalConds, pbs, [], away=True):
        yield ans

placeInGenMaxPoses  = 50
placeInGenMaxPosesH = 10

def placeInGenTop(args, goalConds, pbs, outBindings,
                  regrasp=False, away = False):
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

    if graspB.grasp is None:
        graspB.grasp = UniformDist(range(0,len(graspB.graspDesc)))
    conf = None
    confAppr = None
    # Obstacles for all Reachable fluents
    reachObsts = getReachObsts(goalConds, pbs)
    if reachObsts == None:
        tracep('placeInGen', '    quitting because no path')
        return
    if debug('placeInGen', skip=skip) or debug('reachObsts', skip=skip):
        for _, obst in reachObsts: obst.draw('W', 'brown')
        raw_input('%d reachObsts - in brown'%len(reachObsts))

    # placeInGenCache = pbs.beliefContext.genCaches['placeInGen']
    # key = (obj, tuple(regShapes), graspB, placeB, prob, regrasp, away, fbch.inHeuristic)
    # val = placeInGenCache.get(key, None)
    # if val != None:
    #     ff = placeB.faceFrames[placeB.support.mode()]
    #     objShadow = pbs.objShadow(obj, True, prob, placeB, ff)
    #     for ans in val:
    #         ((pB, gB, cf, ca), viol) = ans
    #         pose = pB.poseD.mode() if pB else None
    #         sup = pB.support.mode() if pB else None
    #         grasp = gB.grasp.mode() if gB else None
    #         pg = (sup, grasp)
    #         sh = objShadow.applyTrans(pose)
    #         if all(not sh.collides(obst) for (ig, obst) in reachObsts if obj not in ig):
    #             viol2 = canPickPlaceTest(pbs, ca, cf, gB, pB, prob, op='place')
    #             print 'viol', viol
    #             print 'viol2', viol2
    #             if viol2 and viol2.weight() <= viol.weight():
    #                 if debug('traceGen'):
    #                     w = viol2.weight() if viol2 else None
    #                     print '    reusing placeInGen',
    #                     print '    placeInGen(%s,%s,%s) h='%(obj,[x.name() for x in regShapes],hand), \
    #                           fbch.inHeuristic, 'v=', w, '(p,g)=', pg, pose
    #                 yield ans[0], viol2
    # else:
    #     placeInGenCache[key] = val
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
                                regrasp=regrasp, away=away)
    mainRightGen = placeInGenAux(newBS, poseGenRight, goalConds, confAppr,
                                 conf, placeB, graspB, 'right', base, prob,
                                 regrasp=regrasp, away=away)
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
        #val.append((ans, v))
        yield ans,v

# Don't try to place all objects at once
def placeInGenAux(pbs, poseGen, goalConds, confAppr, conf, placeB, graspB,
                  hand, base, prob, regrasp=False, away=False):

    def placeBGen():
        for pose in poseGen.copy():
            yield placeB.modifyPoseD(mu=pose)
    skip = (fbch.inHeuristic and not debug('inHeuristic'))
    tries = 0
    gen = Memoizer('placeBGen_placeInGenAux1', placeBGen())
    for ans, viol, hand in placeGenTop((graspB.obj, graspB, gen, hand, base, prob),
                               goalConds, pbs, [], regrasp=regrasp, away=away):
        (gB, pB, cf, ca) = ans
        if debug('placeInGen', skip=skip):
            drawPoseConf(pbs, pB, cf, ca, prob, 'W', 'blue')
            debugMsg('placeInGen', ('-> cyan', ans))
        yield (pB, gB, cf, ca), viol

maxLookDist = 1.5

# Returns lookConf
# The lookDelta is a slop factor.  Ideally if the robot is within that
# factor, visibility should still hold.
def lookGen(args, goalConds, bState, outBindings):
    (obj, pose, support, objV, objDelta, lookDelta, prob) = args
    world = bState.pbs.getWorld()

    base = sameBase(goalConds)
    if base and debug('lookGen'):
        raw_input('Same base constraint in lookGen -- how to pick conf??')

    if pose == '*':
        # This could produce a mode of None
        pB = bState.pbs.getPlaceB(obj, default=False)
        if pB == None:
            print 'Trying to reduce variance on object pose but obj is in hand'
            return
        poseD = pB.poseD if pB else PoseD(None, 4*(0.,))
    else: 
        poseD = PoseD(pose, objV)
    if isVar(support) or support == '*':
        support = bState.pbs.getPlaceB(obj).support.mode()
    if objDelta == '*':
        objDelta = lookDelta
        
    placeB = ObjPlaceB(obj, world.getFaceFrames(obj), support, poseD,
                       # Pretend that the object has bigger delta
                       delta=tuple([o+l for (o,l) in zip(objDelta, lookDelta)]))

    # Don't try to look at the whole shadow
    placeB = placeB.modifyPoseD(var = (0.0001, 0.0001, 0.0001, 0.0005))


    for ans, viol in lookGenTop((obj, placeB, lookDelta, base, prob),
                                goalConds, bState.pbs, outBindings):
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
    newBS = newBS.updateFromGoalPoses(goalConds) if goalConds else newBS
    newBS.updateAvoidShadow([obj])
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
        path, viol = canReachHome(newBS, conf, prob, Violations())
        trace('    lookGen(%s) specified base viol='%obj, viol.weight() if viol else None)
        if not path:
            tracep('lookGen', 'Failed to find a path to look conf with specified base.')
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
        graspB.grasp = UniformDist(range(len(graspB.graspDesc)))
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
                    viol = rm.confViolations(lookConf, pbs, prob)[0]
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

# Returns lookConf
def lookHandGen(args, goalConds, bState, outBindings):
    (obj, hand, graspFace, grasp, graspV, graspDelta, prob) = args
    world = bState.pbs.getWorld()
    if obj == 'none':
        graspB = None
    else:
        graspB = ObjGraspB(obj, world.getGraspDesc(obj), graspFace,
                           PoseD(grasp, graspV), delta=graspDelta)
    for ans, viol in lookHandGenTop((obj, hand, graspB, prob),
                                    goalConds, bState.pbs, outBindings):
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
    newBS = newBS.updateFromGoalPoses(goalConds) if goalConds else newBS
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

#canReachGenCache = {}

# returns
# ['Occ', 'Pose', 'PoseFace', 'PoseVar', 'PoseDelta']
# obj, pose, face, var, delta
def canReachGen(args, goalConds, bState, outBindings):
    (conf, hand, lobj, lgf, lgmu, lgv, lgd,
     robj, rgf, rgmu, rgv, rgd, prob, cond) = args
    debugMsg('canReachGen', args)
    world = bState.pbs.getWorld()
    lookVar = bState.domainProbs.obsVarTuple
    graspB1 = ObjGraspB(lobj, world.getGraspDesc(lobj), lgf,
                        PoseD(lgmu, lgv), delta=lgd) if lobj != 'none' else None
    graspB2 = ObjGraspB(robj, world.getGraspDesc(robj), rgf,
                        PoseD(rgmu, rgv), delta=rgd) if robj != 'none' else None
    for ans in canReachGenTop((conf, hand, graspB1, graspB2, cond, prob,
                               lookVar),
                              goalConds, bState.pbs, outBindings):

        # LPK temporary stuff to look for case of suggesting that we
        # move the same object
        (occ, occPose, occFace, occVar, occDelta) = ans
        for thing in cond:
            if thing.predicate == 'B' and thing.args[0].predicate == 'Pose' and\
                thing.args[0].args[0] == occ and thing.args[2][0] <= occVar[0]:
                print 'canReachGen suggesting a move of an object in cond'
                print 'CRH ans', ans
                print 'Suspect cond', thing
                raw_input('okay?')

        debugMsg('canReachGen', ('->', ans))
        yield ans
    debugMsg('canReachGen', 'exhausted')

def canReachGenTop(args, goalConds, pbs, outBindings):
    (conf, hand, graspB1, graspB2, cond, prob, lookVar) = args
    trace('canReachGen() h=', fbch.inHeuristic)
    skip = (fbch.inHeuristic and not debug('inHeuristic'))
    def moveOut(newBS, obst, delta):
        if debug('traceGen') or debug('canReachGen'):
            print '    canReachGen() obst:', obst
        domainPlaceVar = newBS.domainProbs.obsVarTuple 
        if not isinstance(obst, str):
            obst = obst.name()
        for ans in placeInGenAway((obst, delta, prob), goalConds, newBS, None):
            (pose, poseFace) = ans
            yield (obst, pose, poseFace, domainPlaceVar, delta)
    
    newBS = pbs.copy()
    newBS = newBS.updateFromGoalPoses(goalConds) if goalConds else newBS
    newBS = newBS.updateFromGoalPoses(cond) if cond else newBS
    newBS.updateHeldBel(graspB1, hand)
    newBS.updateHeldBel(graspB2, otherHand(hand))

    path, viol = canReachHome(newBS, conf, prob, Violations())
    if not viol:                  # hopeless
        tracep('canReachGen', 'Impossible dream')
        return
    if viol.empty():
        tracep('canReachGen', 'No obstacles or shadows; returning')
        return
    
    # If possible, it might be better to make the deltas big; but we
    # have to be sure to use the same delta when generating paths.

    objBMinDelta = newBS.domainProbs.minDelta
    objBMinVar = newBS.domainProbs.obsVarTuple
    objBMinProb = 0.95

    lookDelta = objBMinDelta
    moveDelta = objBMinDelta

    # Try to fix one of the violations if any...
    if viol.obstacles:
        obsts = [o.name() for o in viol.obstacles \
                 if o.name() not in newBS.fixObjBs]
        if not obsts:
            debugMsg('canReachGen', 'No movable obstacles to fix')
            return       # nothing available
        # !! How carefully placed this object needs to be
        for ans in moveOut(newBS, obsts[0], moveDelta):
            yield ans 
    else:
        shadowName = list(viol.shadows)[0].name()
        obst = objectName(shadowName)
        placeB = newBS.getPlaceB(obst)
        # !! It could be that sensing is not good enough to reduce the
        # shadow so that we can actually reach conf.
        newBS2 = newBS.copy()
        placeB2 = placeB.modifyPoseD(var = lookVar)
        placeB2.delta = lookDelta
        newBS2.updatePermObjPose(placeB2)
        path2, viol2 = canReachHome(newBS2, conf, prob, Violations())
        if path2 and viol2:
            if shadowName in [x.name() for x in viol2.shadows]:
                print 'canReachGen could not reduce the shadow for', obst
                drawObjAndShadow(newBS, placeB, prob, 'W', color='red')
                print 'brown is as far as it goes'
                drawObjAndShadow(newBS2, placeB2, prob, 'W', color='brown')
                raw_input('Go?')
            if debug('canReachGen', skip=skip):
                drawObjAndShadow(newBS, placeB, prob, 'W', color='red')
                debugMsg('canReachGen', 'Trying to reduce shadow (on W in red) %s'%obst)
                trace('    canReachGen() shadow:', obst)
            yield (obst, placeB.poseD.mode().xyztTuple(), placeB.support.mode(),
                   lookVar, lookDelta)
        # Either reducing the shadow is not enough or we failed and
        # need to move the object (if it's movable).
        if obst not in newBS.fixObjBs:
            for ans in moveOut(newBS, obst, moveDelta):
                yield ans

# LPK!! More efficient if we notice right away that we cannot ask to
# change the pose of an object that is in the hand in goalConds
def canPickPlaceGen(args, goalConds, bState, outBindings):
    (preconf, ppconf, hand,
     obj, pose, realPoseVar, poseDelta, poseFace,
     graspFace, graspMu, graspVar, graspDelta, oobj, oface, oGraspMu, oGraspVar,
     oGraspDelta, prob, cond, op) = args

    goalFluent = Bd([CanPickPlace([preconf, ppconf, hand, obj, pose, realPoseVar, poseDelta, poseFace,
                                   graspFace, graspMu, graspVar, graspDelta, oobj, oface, oGraspMu, oGraspVar,
                                   oGraspDelta, op, cond]), True, prob], True)

    goalConds = goalConds + [goalFluent]

    skip = (fbch.inHeuristic and not debug('inHeuristic'))
    
    def moveOut(newBS, obst, delta):
        if not isinstance(obst, str):
            obst = obst.name()
        domainPlaceVar = bState.domainProbs.obsVarTuple 
        for ans in placeInGenAway((obst, delta, prob), goalConds, newBS, None):
            (pose, poseFace) = ans
            trace('    canPickPlaceGen() obst:', obst, pose)
            yield (obst, pose, poseFace, domainPlaceVar, delta)
        trace('    canPickPlaceGen() obst:', obst, 'no more poses')

    debugMsg('canPickPlaceGen', args)
    trace('canPickPlaceGen() h=', fbch.inHeuristic)

    world = bState.pbs.getWorld()
    lookVar = bState.domainProbs.obsVarTuple

    graspB1 = ObjGraspB(obj, world.getGraspDesc(obj), graspFace,
                 PoseD(graspMu, graspVar), delta= graspDelta)
    graspB2 = ObjGraspB(oobj, world.getGraspDesc(oobj), oface,
                 PoseD(oGraspMu, oGraspVar), delta= oGraspDelta) \
                 if oobj != 'none' else None
    placeB = ObjPlaceB(obj, world.getFaceFrames(obj), poseFace,
                       PoseD(pose, realPoseVar), delta=poseDelta)
    newBS = bState.pbs.copy()   
    newBS = newBS.updateFromGoalPoses(goalConds) if goalConds else newBS
    newBS = newBS.updateFromGoalPoses(cond) if cond else newBS
    # Build the other hand's info into the bState
    newBS.updateHeldBel(graspB2, otherHand(hand))

    viol = canPickPlaceTest(newBS, preconf, ppconf, hand,
                             graspB1, placeB, prob, op=op)
    if debug('canPickPlaceGen'):
        newBS.draw(prob, 'W')
    debugMsg('canPickPlaceGen', ('viol', viol))
    if not viol:                  # hopeless
        tracep('canPickPlaceGen', 'Violation is permanent; returning')
        newBS.draw(prob, 'W')
        raw_input('Impossible CanPickPlace')
        return
    if viol.empty():
        tracep('canPickPlaceGen', 'No obstacles or shadows; returning')
        if debug('canPickPlaceGen'):
            glob.debugOn.append('confViol')
            viol = canPickPlaceTest(newBS, preconf, ppconf, hand,
                             graspB1, placeB, prob, op=op)
            raw_input('here is the conf viol')
        return
    
    objBMinDelta = newBS.domainProbs.minDelta
    objBMinVar = newBS.domainProbs.obsVarTuple
    objBMinProb = 0.95

    lookDelta = objBMinDelta
    moveDelta = objBMinDelta

    # Try to fix one of the violations if any...
    goalPoseObjs = getPoseObjs(goalConds)
    if viol.obstacles:
        obsts = [o.name() for o in viol.obstacles \
                 if o.name() not in newBS.fixObjBs.keys() + goalPoseObjs]
        if not obsts:
            tracep('canPickPlaceGen', 'No movable obstacles to remove')
            return       # nothing available
        # !! How carefully placed this object needs to be
        for ans in moveOut(newBS, obsts[0], moveDelta):
            yield ans 
    else:
        obsts = [objectName(o.name()) for o in viol.shadows \
                 if objectName(o.name()) not in goalPoseObjs]
        if not obsts:
            tracep('canPickPlaceGen', 'No varinances to reduce')
            return       # nothing available
        obst = obsts[0]                 # pick one
        pB = newBS.getPlaceB(obst)
        # !! It could be that sensing is not good enough to reduce the
        # shadow so that we can actually reach conf.
        newBS2 = newBS.copy()
        pB2 = pB.modifyPoseD(var = lookVar)
        pB2.delta = lookDelta
        newBS2.updatePermObjPose(pB2)
        viol2 = canPickPlaceTest(newBS2, preconf, ppconf, hand,
                                 graspB1, placeB, prob, op=op)
        debugMsg('canPickPlaceGen', ('viol2', viol2))
        if viol2:
            if debug('canPickPlaceGen', skip=skip):
                drawObjAndShadow(newBS2, pB2, prob, 'W', color='red')
                debugMsg('canPickPlaceGen',
                         'Trying to reduce shadow (on W in red) %s'%obst)
            trace('    canPickPlaceGen() shadow:', obst, pB.poseD.mode().xyztTuple())
            yield (obst, pB.poseD.mode().xyztTuple(), pB.support.mode(),
                   lookVar, lookDelta)
        # Either reducing the shadow is not enough or we failed and
        # need to move the object (if it's movable).
        if obst not in newBS.fixObjBs:
            for ans in moveOut(newBS, obst, moveDelta):
                yield ans
        else:
            tracep('canPickPlaceGen', 'found fixed obstacle', obst)

# returns
# ['Occ', 'PoseFace', 'Pose', 'PoseVar', 'PoseDelta']
def canSeeGen(args, goalConds, bState, outBindings):
    (obj, pose, support, objV, objDelta, lookConf, lookDelta, prob) = args
    world = bState.pbs.getWorld()

    if pose == '*':
        poseD = bState.pbs.getPlaceB(obj).poseD
    else: 
        poseD = PoseD(pose, objV)
    if isVar(support) or support == '*':
        support = bState.pbs.getPlaceB(obj).support.mode()
    if objDelta == '*':
        objDelta = lookDelta
    
    placeB = ObjPlaceB(obj, world.getFaceFrames(obj), support,
                       poseD,
                       # Pretend that the object has bigger delta
                       delta=tuple([o+l for (o,l) in zip(objDelta, lookDelta)]))

    for ans in canSeeGenTop((lookConf, placeB, [], prob),
                            goalConds, bState.pbs, outBindings):
        yield ans

def canSeeGenTop(args, goalConds, pbs, outBindings):
    (conf, placeB, cond, prob) = args
    obj = placeB.obj
    trace('canSeeGen(%s) h='%obj, fbch.inHeuristic)
    skip = (fbch.inHeuristic and not debug('inHeuristic'))
    debugMsgSkip('canSeeGen', skip, ('args', args))
    def moveOut(newBS, obst, delta):
        domainPlaceVar = pbs.domainProbs.obsVarTuple 
        if not isinstance(obst, str):
            obst = obst.name()
        for ans in placeInGenAway((obst, delta, prob), goalConds, newBS, None):
            (pose, poseFace) = ans
            trace('    canSeeGen(%s) obst='%obj, obst, pose)
            yield (obst, pose, poseFace, domainPlaceVar, delta)

    newBS = pbs.copy()
    newBS = newBS.updateFromGoalPoses(goalConds) if goalConds else newBS
    newBS = newBS.updateFromGoalPoses(cond) if cond else newBS
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
    for ans in moveOut(newBS, obst, moveDelta):
        yield ans 

###################################

# returns values for (?pose, ?poseFace, ?graspPose, ?graspFace, ?graspvar, ?conf, ?confAppr)
def placeInGenOld(args, goalConds, bState, outBindings,
               considerOtherIns = False, regrasp = False, away=False):
    (obj, region, pose, support, objV, graspV, objDelta,
     graspDelta, confDelta, hand, prob) = args
    if not isinstance(region, (list, tuple, frozenset)):
        regions = frozenset([region])
    else:
        regions = frozenset(region)

    skip = (fbch.inHeuristic and not debug('inHeuristic'))
    world = bState.pbs.getWorld()

    # !! Should derive this from the clearance in the region
    domainPlaceVar = bState.domainProbs.obsVarTuple 

    if isVar(graspV):
        graspV = domainPlaceVar
    if isVar(objV):
        objV = graspV
    if isVar(support):
        if bState.pbs.getPlaceB(obj, default=False):
            support = bState.pbs.getPlaceB(obj).support # !! Don't change support
        elif obj == bState.pbs.held[hand].mode():
            pbs = bState.pbs
            attachedShape = pbs.getRobot().attachedObj(pbs.getShadowWorld(prob), hand)
            shape = pbs.getWorld().getObjectShapeAtOrigin(obj).applyLoc(attachedShape.origin())
            support = supportFaceIndex(shape)
        else:
            assert None, 'Cannot determine support'

    graspB = ObjGraspB(obj, world.getGraspDesc(obj), None,
                       PoseD(None, graspV), delta=graspDelta)
    placeB = ObjPlaceB(obj, world.getFaceFrames(obj), support,
                       PoseD(None, objV), delta=objDelta)


    # If pose is specified, just call placeGen
    if pose and not isVar(pose):
        if debug('placeInGen'):
            bState.pbs.draw(prob, 'W')
            debugMsgSkip('placeInGen', skip, ('Pose specified', pose))
        oplaceB = placeB.modifyPoseD(mu=util.Pose(*pose))
        def placeBGen():
            yield oplaceB
        placeBs = Memoizer('placeBGen', placeBGen())
        for ans, viol in placeGenTop((obj, graspB, placeBs, hand, prob),
                                     goalConds, bState.pbs, [], away=away):
            (gB, pB, cf, ca) = ans
            yield (pose, oplaceB.support.mode(),
                   gB.poseD.mode().xyztTuple(), gB.grasp.mode(), graspV, cf, ca)
        return

    # !! Needs to consider uncertainty in region -- but how?

    shWorld = bState.pbs.getShadowWorld(prob)
    regShapes = [shWorld.regionShapes[region] for region in regions]
    if debug('placeInGen'):
        if len(regShapes) == 0:
            debugMsg('placeInGen', 'no region specified')
        shWorld.draw('W')
        for rs in regShapes: rs.draw('W', 'purple')
        debugMsgSkip('placeInGen', skip, 'Target region in purple')
    for ans, viol in placeInGenTop((obj, regShapes, graspB, placeB, hand, prob),
                                   goalConds, bState.pbs, outBindings, considerOtherIns,
                                   regrasp=regrasp, away=away):
        (pB, gB, cf, ca) = ans
        yield (pB.poseD.mode().xyztTuple(), pB.support.mode(),
               gB.poseD.mode().xyztTuple(), gB.grasp.mode(), graspV, cf, ca)
