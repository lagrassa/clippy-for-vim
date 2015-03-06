import numpy as np
import math
import random
import util
import copy
import windowManager3D as wm
from planGlobals import debugMsg, debugMsgSkip, debugDraw, debug, pause, torsoZ, debugOn
from miscUtil import isAnyVar, argmax, isGround, tuplify
from dist import DeltaDist, UniformDist
from pr2Robot import CartConf, gripperTip, gripperFaceFrame
from pr2Util import PoseD, ObjGraspB, ObjPlaceB, Violations, shadowName, objectName, \
     NextColor, supportFaceIndex
import fbch
from fbch import getMatchingFluents
from belief import Bd
from pr2Fluents import CanReachHome, canReachHome, inTest
from pr2Visible import visible, lookAtConf
from pr2PlanBel import getConf
from shapes import Box

Ident = util.Transform(np.eye(4))            # identity transform

import pr2GenAux2
from pr2GenAux2 import *
reload(pr2GenAux2)

pickPlaceSearch = True

# Generators:
#   INPUT:
#   list of specific args such as region, object(s), variance, probability
#   conditions from the goal state, e.g. Pose, Conf, Grasp, Reachable, In, are constraints
#   initial state
#   some pre-bindings of output variables.
#   OUTPUT:
#   ordered list of ordered value lists

def easyGraspGen(args, goalConds, bState, outBindings):
    def pickable(ca, c, pB, gB):
        #raw_input('pickable')
        return canPickPlaceTest(newBS, ca, c, hand, gB, pB, prob)

    def checkInfeasible(bState, graspB, conf):
        newBS = bState.copy()
        newBS.updateConf(conf)
        newBS.updateHeldBel(graspB, hand)
        viol, (rv, hv) = rm.confViolations(conf, newBS, prob,
                                           attached = newBS.getShadowWorld(prob).attached)
        if not viol:                # was valid when not holding, so...
            assert hv               # I hope...
            if debug('easyGraspGen'):
                newBS.draw(prob, 'W')
                debugMsg('easyGen', 'Held collision.')
            return True            # punt.

    def graspApproachConfGen(firstConf):
        if firstConf:
            yield firstConf
        for c in graspConfGen:
            ca = findApproachConf(newBS, obj, placeB, c, hand, prob)
            if ca:
                approached[ca] = c
                yield ca

    graspVar = 4*(0.001,)
    graspDelta = 4*(0.001,)
    
    bState = bState.pbs
    (obj, hand) = args
    if debug('traceGen'):
        print 'easyGraspGen(%s,%s) h='%(obj,hand), fbch.inHeuristic
    if obj == 'none' or (goalConds and getConf(goalConds, None)):
        return
    prob = 0.75
    # Set up bState
    newBS = bState.copy()
    # Just placements specified in goal
    newBS = newBS.updateFromGoalPoses(goalConds) if goalConds else newBS
    shWorld = newBS.getShadowWorld(prob)
    if obj == newBS.held[hand].mode():
        gB = newBS.graspB[hand]
        yield (gB.grasp.mode(), gB.poseD.mode().xyztTuple(),
               graspVar, graspDelta)
        return
    approached = {}
    rm = newBS.getRoadMap()
    placeB = newBS.getPlaceB(obj)
    graspB = ObjGraspB(obj, bState.getWorld().getGraspDesc(obj), None,
                       PoseD(None, graspVar), delta=graspDelta)
    graspB.grasp = UniformDist(range(len(graspB.graspDesc)))
    for gB in graspGen(newBS, obj, graspB):
        # print 'gB', gB
        graspConfGen = potentialGraspConfGen(newBS, placeB, gB, None, hand, prob)
        firstConf = next(graspApproachConfGen(None), None)
        if not firstConf: continue
        for ca in graspApproachConfGen(firstConf):
            if pickable(ca, approached[ca], placeB, gB):
                ans = (gB.grasp.mode(), gB.poseD.mode().xyztTuple(),
                       graspVar, graspDelta)
                if debug('traceGen'):
                    print '    easyGraspGen(%s,%s)='%(obj, hand), ans
                yield ans
                break
        if debug('traceGen'):
            print '    easyGraspGen(%s,%s)='%(obj, hand), None
        debugMsg('easyGraspGen', 'out of values')
        return

def pickGen(args, goalConds, bState, outBindings, onlyCurrent = False):
    (obj, graspFace, graspPose,
     objV, graspV, objDelta, confDelta, graspDelta, hand, prob) = args

    debugMsg('pickGen', 'args', args)

    world = bState.pbs.getWorld()
    graspB = ObjGraspB(obj, world.getGraspDesc(obj), graspFace,
                       PoseD(util.Pose(*graspPose), graspV), delta=graspDelta)
    placeB = ObjPlaceB(obj, world.getFaceFrames(obj), None,
                       PoseD(None,  objV), delta=objDelta)
    for ans, viol in pickGenTop((obj, graspB, placeB, hand, prob),
                                goalConds, bState.pbs, outBindings, onlyCurrent):
        (pB, c, ca) = ans
        yield (pB.poseD.mode().xyztTuple(), pB.support.mode(), c, ca)

def pickGenTop(args, goalConds, bState, outBindings,
               onlyCurrent = False):
    (obj, graspB, placeB, hand, prob) = args
    if debug('traceGen'):
        print 'pickGen(%s,%s,%d) h='%(obj,hand,graspB.grasp.mode()), fbch.inHeuristic
    debugMsgSkip('pickGen', fbch.inHeuristic,
                 zip(('obj', 'graspB', 'placeB', 'hand', 'prob'), args),
                 outBindings)
    if obj == 'none':                   # can't pick up 'none'
        return
    if goalConds and getConf(goalConds, None):
        # if conf is specified, just fail
        return
    if obj == bState.held[hand].mode():
        attachedShape = bState.getRobot().attachedObj(bState.getShadowWorld(prob), hand)
        shape = bState.getWorld().getObjectShapeAtOrigin(obj).applyLoc(attachedShape.origin())
        sup = supportFaceIndex(shape)
        #raw_input('Support = %d'%sup)
        pose = None
        conf = None
        confAppr = None
        debugMsg('pickGen', ('Trying to pick object already in hand -- support surface is', sup))
    else:
        # Use placeB from the current state
        pose = bState.getPlaceB(obj).poseD.mode()
        sup =  bState.getPlaceB(obj).support.mode()
        conf = None
        confAppr = None
    placeB.poseD = PoseD(pose, placeB.poseD.var) # record the pose
    placeB.support = DeltaDist(sup)                             # and supportFace
    debugMsgSkip('pickGen', fbch.inHeuristic, ('target placeB', placeB))
    # Set up bState
    newBS = bState.copy()
    # Just placements specified in goal
    newBS = newBS.updateFromGoalPoses(goalConds) if goalConds else newBS
    if debug('pickGen', skip=fbch.inHeuristic):
        newBS.draw(prob, 'W')
        debugMsg('pickGen', 'Goal conditions')
    gen = pickGenAux(newBS, obj, confAppr, conf, placeB, graspB, hand, prob,
                     goalConds, onlyCurrent=onlyCurrent)
    for x,v in gen:
        if debug('traceGen'):
            (pB, cf, ca) = x
            pose = pB.poseD.mode() if pB else None
            grasp = graspB.grasp.mode() if graspB else None
            print '    pickGen(%s) viol='%obj, v.weight() if v else None, grasp, pose
        yield x,v

def pickGenAux(bState, obj, confAppr, conf, placeB, graspB, hand, prob,
               goalConds, onlyCurrent = False):
    def pickable(ca, c, pB):
        return canPickPlaceTest(bState, ca, c, hand, graspB, pB, prob)

    def checkInfeasible(conf):
        newBS = bState.copy()
        newBS.updateConf(conf)
        newBS.updateHeldBel(graspB, hand)
        viol, (rv, hv) = rm.confViolations(conf, newBS, prob,
                                           attached = newBS.getShadowWorld(prob).attached)
        if not viol:                # was valid when not holding, so...
            assert hv               # I hope...
            if debug('pickGen'):
                newBS.draw(prob, 'W')
                debugMsg('pickGen', 'Held collision.')
            return True            # punt.

    def graspApproachConfGen(firstConf):
        if firstConf:
            yield firstConf
        for c in graspConfGen:
            ca = findApproachConf(bState, obj, placeB, c, hand, prob)
            if ca:
                approached[ca] = c
                yield ca
    
    shWorld = bState.getShadowWorld(prob)
    approached = {}
    rm = bState.getRoadMap()
    if placeB.poseD.mode() != None: # otherwise go to regrasp
        graspConfGen = potentialGraspConfGen(bState, placeB, graspB, conf, hand, prob)
        firstConf = next(graspApproachConfGen(None), None)
        if (not firstConf) or (firstConf and checkInfeasible(firstConf)):
            debugMsg('pickGen', 'No potential grasp confs, will need to regrasp')
        elif pickPlaceSearch:
            for ans in rm.confReachViolGen(graspApproachConfGen(firstConf), bState, prob,
                                           testFn = lambda ca: pickable(ca, approached[ca], placeB)):
                _, cost, path = ans
                if not path: continue
                ca = path[-1]
                c = approached[ca]
                viol = pickable(ca, approached[ca], placeB)
                ans = (placeB, c, ca)
                if debug('pickGen', skip=fbch.inHeuristic):
                    drawPoseConf(bState, placeB, c, ca, prob, 'W', color = 'navy')
                    debugMsg('pickGen', ('-> currently graspable', ans), ('viol', viol))
                    wm.getWindow('W').clear()
                yield ans, viol
        else:
            for ca in graspApproachConfGen(firstConf):
                viol = pickable(ca, approached[ca], placeB)
                if viol:
                    c = approached[ca]
                    ans = (placeB, c, ca)
                    if debug('pickGen', skip=fbch.inHeuristic):
                        drawPoseConf(bState, placeB, c, ca, prob, 'W', color = 'navy')
                        debugMsg('pickGen', ('-> currently graspable', ans), ('viol', viol))
                        wm.getWindow('W').clear()
                    yield ans, viol
        if onlyCurrent:
            debugMsg('pickGen', 'onlyCurrent: out of values')
            return
        
    # Try a regrasp... that is place the object somewhere else where it can be grasped.
    if fbch.inHeuristic:
        return
    print 'Calling for regrasping... h=', fbch.inHeuristic
    debugMsg('pickGen', 'Regrasp?')
    shWorld = bState.getShadowWorld(prob)
    # !! Needs to look for plausible regions...
    regShapes = regShapes = [shWorld.regionShapes[region] for region in bState.awayRegions()]
    plGen = placeInGenTop((obj, regShapes, graspB, placeB, hand, prob),
                          goalConds, bState, [],
                          considerOtherIns = False,
                          regrasp = True)
    for pl, viol in plGen:
        (pB, gB, cf, ca) = pl
        v = pickable(ca, cf, pB)
        if debug('pickGen'):
            bState.draw(prob, 'W')
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
    debugMsg('pickGen', 'out of values')

def placeGen(args, goalConds, bState, outBindings):
    (obj, poses, support, objV, graspV, objDelta, graspDelta, confDelta, hand, prob) = args
    world = bState.pbs.getWorld()
    graspB = ObjGraspB(obj, world.getGraspDesc(obj), None,
                       PoseD(None, graspV), delta=graspDelta)
    placeBs = [ObjPlaceB(obj, world.getFaceFrames(obj), support,
                         PoseD(pose, objV), delta=objDelta) for pose in poses]
    for ans, viol in placeGenTop((obj, graspB, placeBs, hand, prob),
                                 goalConds, bState.pbs, outBindings):
        (gB, pB, c, ca) = ans
        yield (gB.poseD.mode().xyztTuple(), gB.grasp.mode(), c, ca)

# returns values for (?graspPose, ?graspFace, ?conf, ?confAppr)
def placeGenTop(args, goalConds, bState, outBindings, regrasp=False, away=False):
    (obj, graspB, placeBs, hand, prob) = args
    if debug('traceGen'): print 'placeGen(%s,%s) h='%(obj,hand), fbch.inHeuristic
    debugMsgSkip('placeGen', fbch.inHeuristic,
                 zip(('obj', 'graspB', 'placeBs', 'hand', 'prob'), args),
                 ('outBindings', outBindings),
                 ('goalConds', goalConds),
                 ('moveObjBs', bState.moveObjBs),
                 ('fixObjBs', bState.fixObjBs),
                 ('held', (bState.held['left'].mode(),
                           bState.held['right'].mode(),
                           bState.graspB['left'],
                           bState.graspB['right'])))
    if obj == 'none' or not placeBs:
        return
    if goalConds and getConf(goalConds, None) and not away:
        # if conf is specified, just fail
        return
    # Have any output bindings been specified?
    graspB = copy.copy(graspB)
    if graspB.grasp.mode() is None:
        graspB.grasp = UniformDist(range(0,len(graspB.graspDesc)))
    conf = None
    confAppr = None
    # Set up bState
    newBS = bState.copy()
    # Just placements specified in goal (and excluding obj)
    newBS = newBS.updateFromGoalPoses(goalConds, updateConf=not away) if goalConds else newBS
    newBS = newBS.excludeObjs([obj])
    if debug('placeGen', skip=fbch.inHeuristic):
        for gc in goalConds: print gc
        newBS.draw(prob, 'W')
        debugMsg('placeGen', 'Goal conditions')
    gen = placeGenAux(newBS, obj, confAppr, conf, placeBs, graspB, hand, prob,
                      regrasp=regrasp, bStateOrig = bState)
    # !! double check reachObst collision?
    for x,v in gen:
        if debug('traceGen'):
            (gB, pB, c, ca) = x
            pose = pB.poseD.mode() if pB else None
            grasp = gB.grasp.mode() if gB else None
            print '    placeGen(%s,%s) viol='%(obj,hand), v.weight() if v else None, grasp, pose
        yield x,v

def placeGenAux(bState, obj, confAppr, conf, placeBs, graspB, hand, prob,
                regrasp=False, bStateOrig=None):
    def placeable(ca, c):
        (pB, gB) = context[ca]
        ans = canPickPlaceTest(bState, ca, c, hand, gB, pB, prob)
        return ans

    def checkRegraspable(pB):
        if pB in regraspablePB:
            return regraspablePB[pB]
        other =  [next(potentialGraspConfGen(bState, pB, gBO, conf, hand, prob, nMax=1), None) \
                  for gBO in gBOther]
        if any(other):
            if debug('placeGen', skip=fbch.inHeuristic):
                print 'Regraspable', pB.poseD.mode(), [gBO.grasp.mode() for gBO in gBOther]
                for x in other:
                    x.draw('W', 'green')
                    debugMsg('placeGen', 'other')
            regraspablePB[pB] = True
            return True
        else:
            regraspablePB[pB] = False
            if debug('placeGen', skip=fbch.inHeuristic): print 'Not regraspable'
            return False

    def checkOrigGrasp(gB):
        pB = bStateOrig.getPlaceB(obj, default=False) # check we know where obj is.
        if bStateOrig and bStateOrig.held[hand].mode() != obj and pB:
            if next(potentialGraspConfGen(bStateOrig, pB, gB, conf, hand, prob, nMax=1),
                    None):
                return 0
            else:
                return 1
        else: return 0

    def placeApproachConfGen(gB):
        for pB in placeBs:
            if debug('placeGen', skip=fbch.inHeuristic):
                print 'placeGen: considering grasps for ', pB
            if regrasp:
                checkRegraspable(pB)
            graspConfGen = potentialGraspConfGen(bState, pB, gB, conf, hand, prob)
            count = 0
            for c in graspConfGen:
                ca = findApproachConf(bState, obj, pB, c, hand, prob)
                if not ca: continue
                if debug('placeGen', skip=fbch.inHeuristic):
                    c.draw('W', 'orange')
                approached[ca] = c
                count += 1
                context[ca] = (pB, gB)
                yield ca
            if debug('placeGen', skip=fbch.inHeuristic):
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
        
    approached = {}
    context = {}
    regraspablePB = {}
    rm = bState.getRoadMap()
    if regrasp:
         graspBOther = copy.copy(graspB)
         otherGrasps = range(len(graspBOther.graspDesc))
         otherGrasps.remove(graspB.grasp.mode())
         if otherGrasps:
             graspBOther.grasp = UniformDist(otherGrasps)
             gBOther = list(graspGen(bState, obj, graspBOther))
         else:
             gBOther = []

    grasps = [(checkOrigGrasp(gB), gB) for gB in graspGen(bState, obj, graspB)]
    grasps.sort()

    for (orig, gB) in grasps: # for now, just grasp type...
        if debug('placeGen', skip=fbch.inHeuristic):
            print '    placeGen: considering', gB, 'orig', orig

        if pickPlaceSearch:
            # We have not chosen a grasp yet, so this search is done
            # without obj in the world and without it in the hand.  But,
            # the placeable test for success will consider the approach
            # with the relevant grasp.
            for ans in rm.confReachViolGen(placeApproachConfGen(gB), bState, prob,
                                           goalCostFn = regraspCost,
                                           testFn = lambda ca: placeable(ca, approached[ca])):
                _, cost, path = ans
                if path: 
                    ca = path[-1]
                    c = approached[ca]
                    viol = placeable(ca, approached[ca])
                    (pB, gB) = context[ca]
                    if debug('placeGen'):
                        if regrasp:
                            status = 'regraspable' if regraspablePB[pB] else 'not regraspable'
                        else:
                            status = 'not regrasping'
                        print 'pose=', pB.poseD.mode(), 'grasp=', gB.grasp.mode(), status
                    ans = (gB, pB, c, ca)
                    if debug('placeGen', skip=fbch.inHeuristic):
                        drawPoseConf(bState, pB, c, ca, prob, 'W', color='magenta')
                        debugMsg('placeGen', ('->', ans), ('viol', viol))
                        wm.getWindow('W').clear()
                    yield ans, viol
                else:
                    debugMsg('placeGen', 'No valid placements')
        else:
            targetConfs = placeApproachConfGen(gB)
            batchSize = 10
            batch = 0
            while True:
                # Collect the next batach of trialConfs
                batch += 1
                trialConfs = []
                count = 0
                for ca in targetConfs:       # targetConfs is a generator
                    trialConfs.append((regraspCost(ca), ca))
                    count += 1
                    if count == batchSize: break
                if count == 0: break
                trialConfs.sort()
                for _, ca in trialConfs:
                    viol = placeable(ca, approached[ca])
                    if viol:
                        (pB, gB) = context[ca]
                        c = approached[ca]
                        ans = (gB, pB, c, ca)
                        if debug('placeGen', skip=fbch.inHeuristic):
                            drawPoseConf(bState, pB, c, ca, prob, 'W', color='magenta')
                            debugMsg('placeGen', ('->', ans), ('viol', viol))
                            wm.getWindow('W').clear()
                        yield ans, viol
                    else:
                        debugMsg('placeGen', 'No valid placements')
    debugMsg('placeGen', 'out of values')

# Our choices are (generated by graspGen):
# 1. The grasp surface (graspDesc)
# !! 2. The pose on the grasp surface (3 params) -- for now -- skip this
# 3. The conf of the robot (and an approach conf, but we won't vary that)
def graspGen(bState, obj, graspB, placeB=None, conf=None, hand=None, prob=None):
    # LPK:  maybe wrong.  If we're holding the object, suggest that grasp first
    if hand and  bState.getHeld(hand).mode() == obj:
        gB = bState.getGraspB(obj, hand)
        yield gB  

    grasps = list(graspB.grasp.support())
    random.shuffle(grasps)
    for grasp in grasps:
        if debug('graspGen'):
            print 'graspGen: Generating grasp=', grasp
        # !! Should also sample a pose in the grasp face...
        gB = ObjGraspB(graspB.obj, graspB.graspDesc, grasp,
                       # !! Need to pick offset for grasp to be feasible
                       PoseD(graspB.poseD.mode() or util.Pose(0.0, -0.025, 0.0, 0.0),
                             graspB.poseD.var),
                       delta=graspB.delta)
        yield gB

# !! Should derive this from the clearance in the region
# Note that this is smaller than the variable in pr2Ops
maxGraspVar = (0.01**2, 0.01**2, 0.01**2, 0.02**2)

# returns values for (?pose, ?poseFace, ?graspPose, ?graspFace, ?graspvar, ?conf, ?confAppr)
def placeInGen(args, goalConds, bState, outBindings,
               considerOtherIns = False, regrasp = False, away=False):
    (obj, region, pose, support, objV, graspV, objDelta,
     graspDelta, confDelta, hand, prob) = args
    if not isinstance(region, (list, tuple)):
        regions = [region]
    else:
        regions = region

    world = bState.pbs.getWorld()

    if isAnyVar(graspV):
        graspV = maxGraspVar
    if isAnyVar(objV):
        objV = graspV
    if isAnyVar(support):
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
    if pose and not isAnyVar(pose):
        if debug('placeInGen'):
            bState.pbs.draw(prob, 'W')
            debugMsg('placeInGen', ('Pose specified', pose))
        oplaceB = placeB.modifyPoseD(mu=util.Pose(*pose))
        for ans, viol in placeGenTop((obj, graspB, [oplaceB], hand, prob),
                                     goalConds, bState.pbs, [], away=away):
            (gB, pB, cf, ca) = ans
            yield (pose, oplaceB.support.mode(),
                   gB.poseD.mode().xyztTuple(), gB.grasp.mode(), graspV, cf, ca)
        return

    # !! Needs to consider uncertainty in region -- but how?
    
    shWorld = bState.pbs.getShadowWorld(prob)
    regShapes = [shWorld.regionShapes[region] for region in regions]
    if debug('placeInGen'):
        shWorld.draw('W')
        for rs in regShapes: rs.draw('W', 'purple')
        debugMsg('placeInGen', 'Target region in purple')
    for ans, viol in placeInGenTop((obj, regShapes, graspB, placeB, hand, prob),
                                   goalConds, bState.pbs, outBindings, considerOtherIns,
                                   regrasp=regrasp, away=away):
        (pB, gB, cf, ca) = ans
        yield (pB.poseD.mode().xyztTuple(), pB.support.mode(),
               gB.poseD.mode().xyztTuple(), gB.grasp.mode(), graspV, cf, ca)

def placeInGenAway(args, goalConds, bState, outBindings):
    # !! Should search over regions and hands
    (obj, delta, prob) = args
    hand = 'left'
    assert bState.awayRegions(), 'Need some awayRegions'
    for ans in placeInGen((obj, bState.awayRegions(), '?x', '?x', '?x', '?x',
                           delta, delta, delta, hand, prob),
                          # preserve goalConds to get reachObsts
                          goalConds, bState, [], away=True):
        yield ans

maxPlaceVar = (0.001, 0.001, 0.001, 0.01)

def placeInGenTop(args, goalConds, bState, outBindings,
                  considerOtherIns = False, regrasp=False, away = False):
    (obj, regShapes, graspB, placeB, hand, prob) = args
    if debug('traceGen'):
        print 'placeInGen(%s,%s,%s) h='%(obj,[x.name() for x in regShapes],hand), fbch.inHeuristic
    debugMsgSkip('placeInGen', fbch.inHeuristic,
             zip(('obj', 'regShapes', 'graspB', 'placeB', 'hand', 'prob'), args),
             outBindings)
    if obj == 'none' or not regShapes:
        # Nothing to do
        if debug('traceGen'):
            print '    placeInGen(%s,%s,%s) h='%(obj,[x.name() for x in regShapes],hand), None
        return

    if goalConds and getConf(goalConds, None) and not away:
        # if conf is specified, just fail
        return

    if graspB.grasp is None:
        graspB.grasp = UniformDist(range(0,len(graspB.graspDesc)))
    conf = None
    confAppr = None
    # Obstacles for all Reachable fluents
    reachObsts = getReachObsts(goalConds, bState)
    if debug('placeInGen', skip=fbch.inHeuristic):
        for _, obst in reachObsts: obst.draw('W', 'brown')
        debugMsg('placeInGen', ('len(reachObsts) - in brown', len(reachObsts)))
    if reachObsts == None:
        debugMsg('placeInGen', 'quitting because no path')
        return
    # If we are not considering other objects, pick a pose and call placeGen
    if not considerOtherIns:
        newBS = bState.copy()           #  not necessary
        # Shadow (at origin) for object to be placed.
        pB = placeB.modifyPoseD(var=maxPlaceVar)
        poseGen = potentialRegionPoseGen(newBS, obj, pB, prob, regShapes, reachObsts, maxPoses=100)
        # Picks among possible target poses and then try to place it in region
        for ans,v in placeInGenAux1(newBS, poseGen, goalConds, confAppr, conf,
                                  placeB, graspB, hand, prob, regrasp=regrasp, away=away):
            if debug('traceGen'):
                (pB, gB, c, ca) = ans
                pose = pB.poseD.mode() if pB else None
                grasp = gB.grasp.mode() if gB else None
                print '    placeInGen(%s,%s,%s) h='%(obj,[x.name() for x in regShapes],hand), \
                      v.weight() if v else None, grasp, pose
            yield ans,v
    else:
        assert False           # !! not ready for this
        # Conditions for already placed objects
        goalInConds = getGoalInConds(goalConds)
        placed = [o for (o, r, p) in  goalInConds\
                  if o != obj and \
                  inTest(bState, o, r, p, bState.getPlaceB(o))]
        placedBs = dict([(o, bState.getPlaceB(o)) for o in placed])
        inConds = [(o, r, placeB,  p) for (o, r, p) in getGoalInConds(goalConds) \
                   if o != obj and o not in placed]
        # Set up bState
        newBS = bState.copy()
        # Just placements specified in goal
        newBS = newBS.updateFromGoalPoses(goalConds) if goalConds else newBS
        newBS.excludeObjs([obj] + placed)
        newBS.moveObjBs.update(placed)      # the placed are movable
        debugMsg('placeInGen', ('inConds', inConds), ('placed', placed), world)
        gen = placeInGenAux2(newBS, obj, regShapes, goalConds, confAppr, conf, placeB, graspB, hand, prob,
                             reachObsts, inConds, considerOtherIns)
        memo = Memoizer('placeInGen2', gen)
        for x in memo: yield x

# Don't try to place all objects at once
def placeInGenAux1(bState, poseGen, goalConds, confAppr, conf, placeB, graspB, hand, prob,
                   regrasp=False, away=False):
    def placeBGen():
        for pose in poseGen:
            yield placeB.modifyPoseD(mu=pose)
    tries = 0
    for ans, viol in placeGenTop((graspB.obj, graspB, placeBGen(), hand, prob),
                                 goalConds, bState, [], regrasp=regrasp, away=away):
        (gB, pB, cf, ca) = ans
        if debug('placeInGen', skip=fbch.inHeuristic):
            drawPoseConf(bState, pB, cf, ca, prob, 'W', 'blue')
            debugMsg('placeInGen', ('-> cyan', ans))
        yield (pB, gB, cf, ca), viol

# Try to place all objects at once
def placeInGenAux2(bState, obj, regShapes, goalConds, confAppr, conf, placeB, graspB, hand, prob,
                   reachObsts, inConds, considerOtherIns):
    if reachObsts: debugDraw('placeInGen', Shape([obst for (i, obst) in reachObsts]), 'W')
    thisInCond = [(obj, regShapes, placeB, prob)]
    # Pick an order for achieving other In conditions
    # Recall we're doing regression, so placement order is reverse of inConds
    if considerOtherIns:
        perm = permutations(inConds)
    else:
        perm = [[]]
    for otherInConds in perm:
        # Note that we have (pose + fixed) that constrain places and
        # paths and (reachable) that constrain places but not paths.
        # Returns only distinct choices for obj (first of inConds)
        gen = candidatePlaceH(bState, thisInCond + list(otherInConds),
                              goalConds, graspB, confAppr, conf, reachObsts, hand, prob)
        for ans, viol in gen:
            # !! need to return right stuff.
            assert None
            (pB, gB, cf, ca) = ans
            if debug('placeInGen'):
                drawObjAndShadow(bState, pB, prob, 'W', color='cyan')
                for rs in regShapes: rs.draw('W', 'magenta')
                wm.getWindow('W').update()
                debugMsg('placeInGen', ('->', ans))
            yield ans, viol
    debugMsg('placeInGenFail', 'out of values')

maxLookDist = 1.5

# Returns lookConf
# The lookDelta is a slop factor.  Ideally if the robot is within that
# factor, visibility should still hold.
def lookGen(args, goalConds, bState, outBindings):
    (obj, pose, support, objV, objDelta, lookDelta, prob) = args
    world = bState.pbs.getWorld()
    
    placeB = ObjPlaceB(obj, world.getFaceFrames(obj), support,
                       PoseD(pose, objV),
                       # Pretend that the object has bigger delta
                       delta=tuple([o+l for (o,l) in zip(objDelta, lookDelta)]))

    for ans, viol in lookGenTop((obj, placeB, lookDelta, prob),
                                goalConds, bState.pbs, outBindings):
        yield ans

def lookGenTop(args, goalConds, bState, outBindings):
    (obj, placeB, lookDelta, prob) = args
    if debug('traceGen'): print 'lookGen(%s) h='%obj, fbch.inHeuristic
    newBS = bState.copy()
    newBS = newBS.updateFromGoalPoses(goalConds) if goalConds else newBS
    newBS.updatePermObjPose(placeB)

    if goalConds and getConf(goalConds, None):
        # if conf is specified, just fail
        return

    shWorld = newBS.getShadowWorld(prob)
    shName = shadowName(obj)
    sh = shWorld.objectShapes[shName]
    obst = [s for s in shWorld.getNonShadowShapes() if s.name() != obj ]
    rm = newBS.getRoadMap()
    lookConfGen = potentialLookConfGen(rm, sh, maxLookDist)
    # !! BIG UGLY COMMENT -- THIS IS STUPID
    home = newBS.getRoadMap().homeConf
    curr = newBS.conf
    # print 'Home base conf', home['pr2Base'], 'curr base conf', curr['pr2Base']
    def testFn(c):
        if c['pr2Base'] == curr['pr2Base'] or c['pr2Base'] == home['pr2Base']:
            return False
        else:
            print 'Trying base conf', c['pr2Base']
            return visible(shWorld, c, sh, obst, prob)[0]
    for ans in rm.confReachViolGen(lookConfGen, newBS, prob,
                                   avoidShadow=[obj],
                                   testFn = testFn):
        viol, cost, path = ans
        if debug('traceGen'):
            print '    lookGen(%s) viol='%obj, viol.weight() if viol else None
        if not path:
            debugMsg('lookGen', 'Failed to find a path to look conf.')
            return
        conf = path[-1]
        lookConf = lookAtConf(conf, sh)
        if debug('lookGen', skip=fbch.inHeuristic):
            bState.draw(prob, 'W')
            lookConf.draw('W', color='cyan', attached=shWorld.attached)
            debugMsg('lookGen', ('-> cyan', lookConf.conf))
        yield (lookConf,), viol

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


def lookHandGenTop(args, goalConds, bState, outBindings):
    def objInHand(conf, hand):
        attached = shWorld.attached
        if not attached[hand]:
            attached = attached.copy()
            attached[hand] = Box(0.1,0.05,0.1, None, name='virtualObject').applyLoc(gripperTip)
        _, attachedParts = conf.placementAux(attached, getShapes=[])
        return attachedParts[hand]

    def testFn(c):
        ans = visible(shWorld, c, objInHand(c, hand),
                       [c.placement()]+obst, prob)[0]
        return ans

    (obj, hand, graspB, prob) = args
    if debug('traceGen'): print 'lookHandGen(%s) h='%obj, fbch.inHeuristic
    newBS = bState.copy()
    newBS = newBS.updateFromGoalPoses(goalConds) if goalConds else newBS
    newBS.updateHeldBel(graspB, hand)
    shWorld = newBS.getShadowWorld(prob)
    if fbch.inHeuristic:
        yield (lookAtConf(newBS.conf, objInHand(newBS.conf, hand)),), Violations()
        return
    if goalConds and getConf(goalConds, None):
        # if conf is specified, just fail
        return

    # key = (newBS, (obj, hand, graspB, prob))
    # if key in lookHandCache:
    #     print 'lookHandCache HIT'
    # else:
    #     print 'lookHandCache MISS'
    #     lookHandCache[key] = True

    rm = newBS.getRoadMap()
    obst = [s for s in shWorld.getNonShadowShapes() if s.name() != obj ]
    lookConfGen = potentialLookHandConfGen(newBS, prob, hand)
    for ans in rm.confReachViolGen(lookConfGen, newBS, prob,
                                   startConf = newBS.conf,
                                   testFn = testFn):
        viol, cost, path = ans
        if debug('traceGen'):
            print '    lookHandGen(%s) viol='%obj, viol.weight() if viol else None
        if not path:
            debugMsg('lookHandGen', 'Failed to find a path to look conf.')
            return
        conf = path[-1]
        lookConf = lookAtConf(conf, objInHand(conf, hand))
        if not all(lookConf.values()): continue # can't look at it.
        if debug('lookHandGen', skip=fbch.inHeuristic):
            bState.draw(prob, 'W')
            lookConf.draw('W', color='cyan', attached=shWorld.attached)
            debugMsg('lookHandGen', ('-> cyan', lookConf.conf))
        yield (lookConf,), viol

canReachGenCache = {}

# returns
# ['Occ', 'Pose', 'PoseFace', 'PoseVar', 'PoseDelta']
# obj, pose, face, var, delta
def canReachGen(args, goalConds, bState, outBindings):
    (conf, hand, lobj, lgf, lgmu, lgv, lgd,
     robj, rgf, rgmu, rgv, rgd, prob, cond) = args

    key = (tuplify(args),
           tuple(goalConds),
           bState)
    if key in canReachGenCache:
        print 'canReachGenCache hit'
        for val in canReachGenCache[key]:
            yield val
        return
    else:
        canReachGenCache[key] = []

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
        debugMsg('canReachGen', ('->', ans))
        canReachGenCache[key].append(ans)
        yield ans
    debugMsg('canReachGen', 'exhausted')


def canReachGenTop(args, goalConds, bState, outBindings):
    (conf, hand, graspB1, graspB2, cond, prob, lookVar) = args
    if debug('traceGen') or debug('canReachGen'): print 'canReachGen() h=', fbch.inHeuristic
    def moveOut(newBS, obst, delta):
        if debug('traceGen') or debug('canReachGen'):
            print '    canReachGen() obst:', obst
        if not isinstance(obst, str):
            obst = obst.name()
        for ans in placeInGenAway((obst, delta, prob), goalConds, newBS, None):
            (pose, poseFace, _, _, gV, _, _) = ans
            yield (obst, pose, poseFace, gV, delta)
    
    newBS = bState.copy()
    newBS = newBS.updateFromGoalPoses(goalConds) if goalConds else newBS
    newBS = newBS.updateFromGoalPoses(cond) if cond else newBS
    newBS.updateHeldBel(graspB1, hand)
    newBS.updateHeldBel(graspB2, otherHand(hand))

    path, viol = canReachHome(newBS, conf, prob, Violations())
    if not viol:                  # hopeless
        return
    if viol.empty():
        debugMsg('canReachGen', 'No obstacles or shadows; returning')
        return
    
    # This delta can actually be quite large; we aren't trying to
    # "find" this object in a specific position; mostly want to reduce
    # the variance.
    lookDelta = (0.01, 0.01, 0.01, 0.05)
    moveDelta = (0.01, 0.01, 0.01, 0.02)
    # Try to fix one of the violations if any...
    if viol.obstacles:
        obsts = [o.name() for o in viol.obstacles if o.name() not in newBS.fixObjBs]
        if not obsts: return       # nothing available
        # !! How carefully placed this object needs to be
        for ans in moveOut(newBS, obsts[0], moveDelta):
            yield ans 
    else:
        obst = objectName(list(viol.shadows)[0])
        placeB = newBS.getPlaceB(obst)
        # !! It could be that sensing is not good enough to reduce the
        # shadow so that we can actually reach conf.
        newBS2 = newBS.copy()
        placeB2 = placeB.modifyPoseD(var = lookVar)
        placeB2.delta = lookDelta
        newBS2.updatePermObjPose(placeB2)
        path2, viol2 = canReachHome(newBS2, conf, prob, Violations())
        if path2 and viol2:
            if debug('canReachGen', skip=fbch.inHeuristic):
                drawObjAndShadow(newBS2, placeB2, prob, 'W', color='red')
                debugMsg('canReachGen', 'Trying to reduce shadow (on W in red) %s'%obst)
            if debug('traceGen'):
                print '    canReachGen() shadow:', obst
            yield (obst, placeB.poseD.mode().xyztTuple(), placeB.support.mode(),
                   lookVar, lookDelta)
        # Either reducing the shadow is not enough or we failed and
        # need to move the object (if it's movable).
        if obst not in newBS.fixObjBs:
            for ans in moveOut(newBS, obst, moveDelta):
                yield ans

# returns
# ['Occ', 'PoseFace', 'Pose', 'PoseVar', 'PoseDelta']
def canSeeGen(args, goalConds, bState, outBindings):
    (obj, pose, support, objV, objDelta, lookConf, lookDelta, prob) = args
    world = bState.pbs.getWorld()
    
    placeB = ObjPlaceB(obj, world.getFaceFrames(obj), support,
                       PoseD(pose, objV),
                       # Pretend that the object has bigger delta
                       delta=tuple([o+l for (o,l) in zip(objDelta, lookDelta)]))

    for ans in canSeeGenTop((lookConf, placeB, [], prob),
                            goalConds, bState.pbs, outBindings):
        yield ans

def canSeeGenTop(args, goalConds, bState, outBindings):
    (conf, placeB, cond, prob) = args
    obj = placeB.obj
    if debug('traceGen') or debug('canSeeGen'):
        print 'canSeeGen(%s) h='%obj, fbch.inHeuristic
    debugMsg('canSeeGen', ('args', args))
    def moveOut(newBS, obst, delta):
        if debug('traceGen') or debug('canSeeGen'):
            print '    canSeeGen(%s) obst='%obj, obst
        if not isinstance(obst, str):
            obst = obst.name()
        for ans in placeInGenAway((obst, delta, prob), goalConds, newBS, None):
            (pose, poseFace, _, _, gV, _, _) = ans
            yield (obst, pose, poseFace, gV, delta)

    newBS = bState.copy()
    newBS = newBS.updateFromGoalPoses(goalConds) if goalConds else newBS
    newBS = newBS.updateFromGoalPoses(cond) if cond else newBS
    newBS = newBS.updatePermObjPose(placeB)

    shWorld = newBS.getShadowWorld(prob)
    shape = shWorld.objectShapes[placeB.obj]
    obst = [s for s in shWorld.getNonShadowShapes() \
            if s.name() != placeB.obj ]
    p, occluders = visible(shWorld, conf, shape, obst, prob)
    occluders = [oc for oc in occluders if oc not in newBS.fixObjBs]
    if not occluders:
        debugMsg('canSeeGen', 'no occluders')
        return
    obst = occluders[0] # !! just pick one
    moveDelta = (0.01, 0.01, 0.01, 0.02)
    for ans in moveOut(newBS, obst, moveDelta):
        yield ans 
