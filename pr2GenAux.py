import pdb
import numpy as np
import math
import random
import hu
from time import sleep
import copy
import windowManager3D as wm
import shapes
import planGlobals as glob
from planGlobals import torsoZ
from traceFile import debugMsg, debug
from miscUtil import argmax, isGround, isVar, argmax, squashOne
from dist import UniformDist, DDist
from geom import bboxCenter
from pr2Robot import CartConf, gripperFaceFrame
from planUtil import PoseD, ObjGraspB, ObjPlaceB, Violations, Response
from pr2Util import shadowName, objectName, Memoizer, inside
import fbch
from fbch import getMatchingFluents
from belief import Bd, B
from pr2Fluents import CanReachHome, canReachHome, In, Pose, CanPickPlace, \
    BaseConf, Holding, CanReachNB, Conf, CanPush, canPush
from transformations import rotation_matrix
from cspace import xyCI, CI, xyCOParts
from pr2Visible import visible, lookAtConf, viewCone, findSupportTableInPbs
from pr2RRT import planRobotGoalPath
from traceFile import tr
from miscUtil import roundrobin

Ident = hu.Transform(np.eye(4))            # identity transform
tiny = 1.0e-6

################
# Basic tests for pick and place
################

def legalGrasp(pbs, conf, hand, objGrasp, objPlace):
    deltaThreshold = (0.01, 0.01, 0.01, 0.02)
    # !! This should check for kinematic feasibility over a range of poses.
    of = objectGraspFrame(pbs, objGrasp, objPlace, hand)
    rf = robotGraspFrame(pbs, conf, hand)
    result = of.withinDelta(rf, deltaThreshold)
    return result

def objectGraspFrame(pbs, objGrasp, objPlace, hand):
    # Find the robot wrist frame corresponding to the grasp at the placement
    objFrame = objPlace.objFrame()
    graspDesc = objGrasp.graspDesc[objGrasp.grasp.mode()]
    faceFrame = graspDesc.frame.compose(objGrasp.poseD.mode())
    centerFrame = faceFrame.compose(hu.Pose(0,0,graspDesc.dz,0))
    graspFrame = objFrame.compose(centerFrame)
    # !! Rotates wrist frame to grasp face frame - defined in pr2Robot
    gT = gripperFaceFrame[hand]
    wristFrame = graspFrame.compose(gT.inverse())
    # assert wristFrame.pose()

    if debug('objectGraspFrame'):
        print 'objGrasp', objGrasp
        print 'objPlace', objPlace
        print 'objFrame\n', objFrame.matrix
        print 'grasp faceFrame\n', faceFrame.matrix
        print 'centerFrame\n', centerFrame.matrix
        print 'graspFrame\n', graspFrame.matrix
        print 'object wristFrame\n', wristFrame.matrix

    return wristFrame

def robotGraspFrame(pbs, conf, hand):
    robot = pbs.getRobot()
    _, frames = robot.placement(conf, getShapes=[])
    wristFrame = frames[robot.wristFrameNames[hand]]
    if debug('robotGraspFrame'):
        print 'robot wristFrame\n', wristFrame
    return wristFrame

###################
# !! Need implementations of InPickApproach and InPlaceDeproach

def inPickApproach(*x): return True
def InPlaceDeproach(*x): return True

###################

# Pick conditions
# pick1. move home->pre with obj at pick pose
# pick2. move home->pick without obj
# pick3. move home->pick with obj in hand 

# Place conditions are equivalent
# place1. move home->pre with obj in hand
# place1. move home->place with hand empty
# place2. move home->pre with obj at place pose

ppConfs = {}

def canPickPlaceTest(pbs, preConf, ppConf, hand, objGrasp, objPlace, p,
                     op='pick', quick = False):
    obj = objGrasp.obj
    collides = pbs.getRoadMap().checkRobotCollision
    if debug('canPickPlaceTest'):
        print zip(('preConf', 'ppConf', 'hand', 'objGrasp', 'objPlace', 'p', 'pbs'),
                  (preConf, ppConf, hand, objGrasp, objPlace, p, pbs))
    if not legalGrasp(pbs, ppConf, hand, objGrasp, objPlace):
        debugMsg('canPickPlaceTest', 'Grasp is not legal in canPickPlaceTest')
        return None, 'Legal grasp'
    # pbs.getRoadMap().approachConfs[ppConf] = preConf
    violations = Violations()           # cumulative
    # 1.  Can move from home to pre holding nothing with object placed at pose
    if preConf:
        pbs1 = pbs.copy().updatePermObjPose(objPlace).updateHeldBel(None, hand)
        if op == 'place': pbs1.addAvoidShadow([obj])
        if debug('canPickPlaceTest'):
            pbs1.draw(p, 'W')
            debugMsg('canPickPlaceTest', 'H->App, obj@pose (condition 1)')
        if quick:
            violations = pbs.getRoadMap().confViolations(preConf, pbs1, p, violations)
            path = [preConf]
        else:
            path, violations = canReachHome(pbs1, preConf, p, violations)
        if not violations:
            debugMsg('canPickPlaceTest', 'Failed H->App, obj=pose (condition 1)')
            return None, '1. H->App obj@pose'
        elif debug('canPickPlaceTest'):
            for c in path: c.draw('W', attached = pbs1.getShadowWorld(p).attached)
            debugMsg('canPickPlaceTest', 'path 1')

    # preConfShape = preConf.placement(attached = pbs1.getShadowWorld(p).attached)
    objShadow = objPlace.shadow(pbs1.getShadowWorld(p))
    # Check visibility at preConf (for pick)
    if op=='pick' and not (glob.inHeuristic or quick):
        path = canView(pbs1, p, preConf, hand, objShadow)
        if path:
            debugMsg('canPickPlaceTest', 'Succeeded visibility test for pick')
            preConfView = path[-1]
            if preConfView != preConf:
                path, violations = canReachHome(pbs1, preConfView, p, violations)
                if not violations:
                    debugMsg('canPickPlaceTest', 'Cannot reachHome with retracted arm')
                    pbs1.draw(p, 'W'); preConfView.draw('W', 'red')
                    raw_input('canPickPlaceTest - Cannot reachHome with retracted arm')
                    return None, 'Obj visibility'
        else:
            debugMsg('canPickPlaceTest', 'Failed visibility test for pick')
            return None, 'Obj visibility'
            
    # 2 - Can move from home to pre holding the object
    pbs2 = pbs.copy().excludeObjs([obj]).updateHeldBel(objGrasp, hand)
    if debug('canPickPlaceTest'):
        pbs2.draw(p, 'W'); preConf.draw('W', attached = pbs2.getShadowWorld(p).attached)
        debugMsg('canPickPlaceTest', 'H->App, obj=held (condition 2)')
    if quick:
        violations = pbs.getRoadMap().confViolations(preConf, pbs2, p, violations)
        path = [preConf]
    else:
        path, violations = canReachHome(pbs2, preConf, p, violations)
    if not violations:
        debugMsg('canPickPlaceTest' + 'Failed H->App, obj=held (condition 2)')
        return None, '2. H->App, held=obj'
    elif debug('canPickPlaceTest'):
        for c in path: c.draw('W', attached = pbs2.getShadowWorld(p).attached)
        debugMsg('canPickPlaceTest', 'path 2')

    # Check visibility of support table at preConf (for pick AND place)
    if op in ('pick', 'place') and not (glob.inHeuristic or quick):
        tableB = findSupportTableInPbs(pbs1, objPlace.obj) # use pbs1 so obj is there
        assert tableB
        if debug('canPickPlaceTest'): print 'Looking at support for', obj, '->', tableB.obj
        lookDelta = pbs2.domainProbs.minDelta
        lookVar = pbs2.domainProbs.obsVarTuple
        tableB2 = tableB.modifyPoseD(var = lookVar)
        tableB2.delta = lookDelta
        prob = 0.95
        shadow = tableB2.shadow(pbs2.updatePermObjPose(tableB2).getShadowWorld(prob))
        if collides(preConf, shadow, attached = pbs2.getShadowWorld(p).attached):
            preConfShape = preConf.placement(attached = pbs2.getShadowWorld(p).attached)
            pbs2.draw(p, 'W'); preConfShape.draw('W', 'cyan'); shadow.draw('W', 'cyan')
            raw_input('Preconf collides for place in canPickPlaceTest')
            return None, 'Support shadow collision'
        if collides(ppConf, shadow): # ppConfShape.collides(shadow):
            ppConfShape = ppConf.placement() # no attached
            pbs2.draw(p, 'W'); ppConfShape.draw('W', 'magenta'); shadow.draw('W', 'magenta')
            raw_input('PPconf collides for place in canPickPlaceTest')
            return None, 'Support shadow collision'
        if not canView(pbs2, p, preConf, hand, shadow):
            pbs2.draw(p, 'W'); preConfShape.draw('W', 'orange'); shadow.draw('W', 'orange')
            raw_input('Failing to view for place in canPickPlaceTest')
            return None, 'Support visibility'

    # 3.  Can move from home to pick with object placed at pose (0 var)
    oB = objPlace.modifyPoseD(var=4*(0.0,)) # ignore uncertainty
    oB.delta = 4*(0.0,)
    pbs3 = pbs.copy().updatePermObjPose(oB).updateHeldBel(None, hand)
    if debug('canPickPlaceTest'):
        pbs3.draw(p, 'W')
        debugMsg('canPickPlaceTest', 'H->Target, obj placed (0 var) (condition 3)')
    if quick:
        violations = pbs.getRoadMap().confViolations(ppConf, pbs3, p, violations)
        path = [ppConf]
    else:
        path, violations = canReachHome(pbs3, ppConf, p, violations)
    if not violations:
        debugMsg('canPickPlaceTest', 'Failed H->Target  (condition 3)')
        return None, '3. H->Target obj@pose 0var'
    elif debug('canPickPlaceTest'):
        for c in path: c.draw('W', attached = pbs3.getShadowWorld(p).attached)
        debugMsg('canPickPlaceTest', 'path 3')
    # 4.  Can move from home to pick while holding obj with zero grasp variance
    gB = objGrasp.modifyPoseD(var=4*(0.0,)) # ignore uncertainty
    gB.delta = 4*(0.0,)
    pbs4 = pbs.copy().excludeObjs([obj]).updateHeldBel(gB, hand)
    if debug('canPickPlaceTest'):
        pbs4.draw(p, 'W'); ppConf.draw('W', attached = pbs4.getShadowWorld(p).attached)
        debugMsg('canPickPlaceTest', 'H->Target, holding obj (0 var) (condition 4)')
    if quick:
        violations = pbs.getRoadMap().confViolations(ppConf, pbs4, p, violations)
        path = [ppConf]
    else:
        path, violations = canReachHome(pbs4, ppConf, p, violations)
    if not violations:
        debugMsg('canPickPlaceTest', 'Failed H->Target held=obj(condition 4)')
        return None, '4. H->Target held=obj 0var'
    elif debug('canPickPlaceTest'):
        for c in path: c.draw('W', attached = pbs4.getShadowWorld(p).attached)
        debugMsg('canPickPlaceTest', 'path 4')
    debugMsg('canPickPlaceTest', ('->', violations))

    if debug('lookBug'):
        base = tuple(preConf['pr2Base'])
        entry = (preConf, tuple([x.getShadowWorld(p) for x in (pbs1, pbs2, pbs3, pbs4)]))
        if base in ppConfs:
            ppConfs[base] = ppConfs[base].union(frozenset([entry]))
        else:
            ppConfs[base] = frozenset([entry])

    return violations, None


# Find a path to a conf such that the arm (specified) by hand does not
# collide with the view cone to the target shape and maybe shadow.
def canView(pbs, prob, conf, hand, shape,
            shapeShadow = None, maxIter = 50):
    def armShape(c, h):
        parts = dict([(o.name(), o) for o in c.placement(attached=attached).parts()])
        armShapes = [parts[pbs.getRobot().armChainNames[h]],
                     parts[pbs.getRobot().gripperChainNames[h]]]
        if attached[h]:
            armShapes.append(parts[attached[h].name()])
        return shapes.Shape(armShapes, None)
    collides = pbs.getRoadMap().checkRobotCollision
    robot = pbs.getRobot()
    vc = viewCone(conf, shape)
    if not vc: return None
    shWorld = pbs.getShadowWorld(prob)
    attached = shWorld.attached
    # confPlace = conf.placement(attached=attached)
    if not collides(conf, vc, attached=attached): # vc.collides(confPlace):
        if debug('canView'):
            print 'canView - no view cone collision'
        return [conf]
    # !! don't move arms to clear view of fixed objects
    if objectName(shape.name()) in pbs.getWorld().graspDesc:
        if debug('canView'):
            vc.draw('W', 'red')
            conf.draw('W', attached=attached)
            raw_input('ViewCone collision')
        if shapeShadow:
            avoid = shapes.Shape([vc, shape, shapeShadow], None)
        else:
            avoid = shapes.Shape([vc, shape], None)
        pathFull = []
        for h in ['left', 'right']:     # try both hands
            chainName = robot.armChainNames[h]
            armChains = [chainName, robot.gripperChainNames[h]]
            if not collides(conf, vc, attached=attached, selectedChains=armChains):  # vc.collides(armShape(conf, h)):
                continue
            if debug('canView'):
                print 'canView collision with', h, 'arm', conf['pr2Base']
            path, viol = planRobotGoalPath(pbs, prob, conf,
                                lambda c: not (collides(c, avoid, attached=attached, selectedChains=armChains) \
                                                          if glob.useCC else avoid.collides(armShape(c,h))),
                                           None, [chainName], maxIter = maxIter)
            if debug('canView'):
                pbs.draw(prob, 'W')
                if path:
                    for c in path: c.draw('W', 'blue', attached=attached)
                    path[-1].draw('W', 'orange', attached=attached)
                    vc.draw('W', 'green')
                    raw_input('canView - Retract arm')
            if debug('canView') or debug('canViewFail'):
                if not path:
                    pbs.draw(prob, 'W')
                    conf.draw('W', attached=attached)
                    vc.draw('W', 'red')
                    raw_input('canView - no path')
            if path:
                pathFull.extend(path)
            else:
                return []
        return pathFull
    else:
        if debug('canView'):
            print 'canView - ignore view cone collision for perm object'
        return [conf]

################
## GENERATORS
################

# This needs generalization

def findApproachConf(pbs, obj, placeB, conf, hand, prob):
    cached = pbs.getRoadMap().approachConfs.get(conf, False)
    if cached is not False: return cached
    robot = pbs.getRobot()
    cart = conf.cartConf()
    wristFrame = cart[robot.armChainNames[hand]]
    if abs(wristFrame.matrix[2,0]) < 0.1: # horizontal
        offset = hu.Pose(-glob.approachBackoff,0.,glob.approachPerpBackoff,0.)
    else:                               # vertical
        offset = hu.Pose(-glob.approachBackoff,0.,0.,0.)
    wristFrameBack = wristFrame.compose(offset)
    cartBack = cart.set(robot.armChainNames[hand], wristFrameBack)
    confBack = robot.inverseKin(cartBack, conf = conf)
    if not None in confBack.values():
        pbs.getRoadMap().approachConfs[conf] = confBack
        return confBack
    else:
        pbs.getRoadMap().approachConfs[conf] = None
        return None

# Our choices are (generated by graspGen):
# 1. The grasp surface (graspDesc)
# !! 2. The pose on the grasp surface (3 params) -- for now -- skip this
# 3. The conf of the robot (and an approach conf, but we won't vary that)
def graspGen(pbs, obj, graspB, placeB=None, conf=None, hand=None, prob=None):

    grasps = list(graspB.grasp.support())
    random.shuffle(grasps)
    for grasp in grasps:
        if debug('graspGen'):
            print 'graspGen: Generating grasp=', grasp
        # TODO: Should also sample a pose in the grasp face...
        gB = ObjGraspB(graspB.obj, graspB.graspDesc, grasp, None,
                       # !! Need to pick offset for grasp to be feasible
                       PoseD(graspB.poseD.mode() or hu.Pose(0.0, -0.025, 0.0, 0.0),
                             graspB.poseD.var),
                       delta=graspB.delta)
        yield gB

graspConfHistory = []

graspConfGenCache = {}
graspConfGenCacheStats = [0,0]

graspConfClear = 0.02

def potentialGraspConfGen(pbs, placeB, graspB, conf, hand, base, prob, nMax=None):
    tag = 'potentialGraspConfs'
    key = (pbs, placeB, graspB, conf, hand, tuple(base) if base else None, prob, nMax)
    cache = graspConfGenCache
    val = cache.get(key, None)
    graspConfGenCacheStats[0] += 1

    if val != None:
        graspConfGenCacheStats[1] += 1
        memo = val.copy()
        if debug(tag): print tag, 'cached gen with len(values)=', memo.values
    else:
        memo = Memoizer('potentialGraspConfGen',
                        potentialGraspConfGenAux(*key))
        cache[key] = memo
        if debug(tag): print tag, 'new gen'
    for x in memo:
        assert len(x) == 3 and x[-1] != None
        yield x

graspConfs = set([])

def graspConfForBase(pbs, placeB, graspB, hand, basePose, prob, wrist = None):
    robot = pbs.getRobot()
    rm = pbs.getRoadMap()
    if not wrist:
        wrist = objectGraspFrame(pbs, graspB, placeB, hand)
    basePose = basePose.pose()
    cart = CartConf({'pr2BaseFrame': basePose,
                     'pr2Torso':[torsoZ]}, robot)
    if hand == 'left':
        cart.conf['pr2LeftArmFrame'] = wrist 
        cart.conf['pr2LeftGripper'] = [0.08] # !! pick better value
    else:
        cart.conf['pr2RightArmFrame'] = wrist 
        cart.conf['pr2RightGripper'] = [0.08]
    # Check inverse kinematics
    conf = robot.inverseKin(cart,
                            complain=debug('potentialGraspConfs'))
    if None in conf.values(): return
    # Copy the other arm
    if hand == 'left':
        conf.conf['pr2RightArm'] = pbs.conf['pr2RightArm']
        conf.conf['pr2RightGripper'] = pbs.conf['pr2RightGripper']
    else:
        conf.conf['pr2LeftArm'] = pbs.conf['pr2LeftArm']
        conf.conf['pr2LeftGripper'] = pbs.conf['pr2LeftGripper']
    ca = findApproachConf(pbs, placeB.obj, placeB, conf, hand, prob)
    if ca:
        # Check for collisions, don't include attached...
        viol = rm.confViolations(ca, pbs, prob, ignoreAttached=True,
                                 clearance=graspConfClear)
        if not viol: return
        viol = rm.confViolations(conf, pbs, prob, initViol=viol,
                                 ignoreAttached=True, clearance=graspConfClear)
        if viol:
            if debug('potentialGraspConfsWin'):
                pbs.draw(prob, 'W')
                conf.draw('W','green')
                debugMsg('potentialGraspConfsWin', ('->', conf.conf))
            if debug('keepGraspConfs'):
                graspConfs.add((conf, ca, pbs, prob, viol))
            return conf, ca, viol
    else:
        if debug('potentialGraspConfs'):
            pbs.draw(prob, 'W'); conf.draw('W','red')

def potentialGraspConfGenAux(pbs, placeB, graspB, conf, hand, base, prob,
                             nMax=10):
    tag = 'potentialGraspConfs'
    if debug(tag): print 'Entering potentialGraspConfGenAux'
    if conf:
        ca = findApproachConf(pbs, placeB.obj, placeB, conf, hand, prob)
        if ca:
            viol = pbs.getRoadMap().confViolations(ca, pbs, prob,
                                                   ignoreAttached=True,
                                                   clearance=graspConfClear)
            if viol:
                yield conf, ca, viol
        tr(tag, 'Conf specified; viol is None or out of alternatives')
        return
    wrist = objectGraspFrame(pbs, graspB, placeB, hand)
    tr(tag, hand, placeB.obj, graspB.grasp, '\n', wrist,
       draw = [(pbs, prob, 'W')], snap = ['W'])
    count = 0
    tried = 0
    robot = pbs.getRobot()
    if base:
        (x,y,th) = base
        nominalBasePose = hu.Pose(x, y, 0.0, th)
    else:
        (x,y,th) = pbs.getShadowWorld(prob).robotConf['pr2Base']
        curBasePose = hu.Pose(x, y, 0.0, th)

    if debug('collectGraspConfs'):
        xAxisZ = wrist.matrix[2,0]
        bases = robot.potentialBasePosesGen(wrist, hand)
        if abs(xAxisZ) < 0.01:
            if not glob.horizontal:
                glob.horizontal = [(b,
                                    CartConf({'pr2BaseFrame': b,
                                              robot.armChainNames[hand]+'Frame' : wrist,
                                              'pr2Torso':[glob.torsoZ]}, robot)) \
                                   for b in bases]
        elif abs(xAxisZ + 1.0) < 0.01:
            if not glob.vertical:
                glob.vertical = [(b,
                                  CartConf({'pr2BaseFrame': b,
                                            robot.armChainNames[hand]+'Frame' : wrist,
                                            'pr2Torso':[glob.torsoZ]}, robot)) \
                                 for b in bases]
        else:
            assert None

    if base:
        for ans in [graspConfForBase(pbs, placeB, graspB, hand,
                                     nominalBasePose, prob, wrist)]:
            if ans:
                yield ans
        tr(tag, 'Base specified; out of grasp confs for base')
        return
    # Try current pose first
    ans = graspConfForBase(pbs, placeB, graspB, hand, curBasePose, prob, wrist)
    if ans: yield ans
    # Try the rest
    # !! Sample subset??
    for basePose in robot.potentialBasePosesGen(wrist, hand):
        if nMax and count >= nMax: break
        tried += 1
        ans = graspConfForBase(pbs, placeB, graspB, hand, basePose, prob, wrist)
        if ans:
            count += 1
            yield ans
    debugMsg('potentialGraspConfs',
             ('Tried', tried, 'Found', count, 'potential grasp confs'))
    return

def potentialLookConfGen(pbs, prob, shape, maxDist):
    def testPoseInv(basePoseInv):
        bb = shape.applyTrans(basePoseInv).bbox()
        return bb[0][0] > 0 and bb[1][0] > 0

    centerPoint = hu.Point(np.resize(np.hstack([bboxCenter(shape.bbox()), [1]]), (4,1)))
    tested = set([])
    rm = pbs.getRoadMap()
    for node in rm.nodes():             # !!
        nodeBase = tuple(node.conf['pr2Base'])
        if nodeBase in tested:
            continue
        else:
            tested.add(nodeBase)
        x,y,th = nodeBase
        basePose = hu.Pose(x,y,0,th)
        dist = centerPoint.distanceXY(basePose.point())
        if dist > maxDist:
            continue
        inv = basePose.inverse()
        if not testPoseInv(inv):
            # Rotate the base to face the center of the object
            center = inv.applyToPoint(centerPoint)
            angle = math.atan2(center.matrix[0,0], center.matrix[1,0])
            rotBasePose = basePose.compose(hu.Pose(0,0,0,-angle))
            par = rotBasePose.pose().xyztTuple()
            rotConf = node.conf.set('pr2Base', (par[0], par[1], par[3]))
            if debug('potentialLookConfs'):
                print 'basePose', node.conf['pr2Base']
                print 'center', center
                print 'rotBasePose', rotConf['pr2Base']
            if testPoseInv(rotBasePose.inverse()):
                if rm.confViolations(rotConf, pbs, prob):
                    yield rotConf
        else:
            if debug('potentialLookConfs'):
                node.conf.draw('W')
                print 'node.conf', node.conf['pr2Base']
                raw_input('potential look conf')
            if rm.confViolations(node.conf, pbs, prob):
                yield node.conf
    return

def otherHand(hand):
    return 'left' if hand == 'right' else 'right'

ang = -math.pi/2
rotL = hu.Transform(rotation_matrix(-math.pi/4, (1,0,0)))
def trL(p): return p.compose(rotL)
rotR = hu.Transform(rotation_matrix(math.pi/4, (1,0,0)))
def trR(p): return p.compose(rotR)
lookPoses = {'left': [trL(x) for x in [hu.Pose(0.4, 0.35, 1.0, ang),
                                       hu.Pose(0.4, 0.25, 1.0, ang),
                                       hu.Pose(0.5, 0.08, 1.0, ang),
                                       hu.Pose(0.5, 0.18, 1.0, ang)]],
             'right': [trR(x) for x in [hu.Pose(0.4, -0.35, 0.9, -ang),
                                        hu.Pose(0.4, -0.25, 0.9, -ang),
                                        hu.Pose(0.5, -0.08, 1.0, -ang),
                                        hu.Pose(0.5, -0.18, 1.0, -ang)]]}
def potentialLookHandConfGen(pbs, prob, hand):
    shWorld = pbs.getShadowWorld(prob)
    robot = pbs.conf.robot
    curCartConf = pbs.conf.cartConf()
    chain = robot.armChainNames[hand]
    baseFrame = curCartConf['pr2Base']
    for pose in lookPoses[hand]:
        if debug('potentialLookHandConfs'):
            print 'potentialLookHandConfs trying:\n', pose
        target = baseFrame.compose(pose)
        cartConf = curCartConf.set(chain, target)
        conf = robot.inverseKin(cartConf, conf=pbs.conf)
        if all(v for v in conf.conf.values()):
            if debug('potentialLookHandConfs'):
                conf.draw('W', 'blue')
                print 'lookPose\n', pose.matrix
                print 'target\n', target.matrix
                print 'conf', conf.conf
                print 'cart\n', cartConf[chain].matrix
                raw_input('potentialLookHandConfs')
            yield conf
    return

################
## SUPPORT FUNCTIONS
################

# returns lists of (poseB, graspB, conf, canfAppr)

# !! THIS IS NOT FINISHED!

def candidatePlaceH(pbs, inCondsRev, graspB, reachObsts, hand, prob):

    assert False

    # REVERSE THE INCONDS -- because regression is in opposite order
    inConds = inCondsRev[::-1]
    debugMsg('candidatePGCC', ('inConds - reversed', inConds))
    objs = [obj for (obj,_,_,_) in inConds]
    objPB = [opb for (_,_,opb,_,_) in inConds]
    shWorld = pbs.getShadowWorld(prob)
    # Shadows (at origin) for objects to be placed.
    ## !! Maybe give it a little cushion
    objShadows = [shWorld.world.getObjectShapesAtOrigin(o.name()) for o in shWorld.getShadowShapes()]
    newBS = pbs.copy()
    newBS.excludeObjs(objs)
    shWorld = newBS.getShadowWorld(prob)
    poseObsts = [sh for sh in shWorld.getShadowShapes() if not sh.name() in shWorld.fixedObjects]
    fixedObst = [sh for sh in shWorld.getShadowShapes() if sh.name() in shWorld.fixedObjects]
    # inCond is (obj, regShape, placeB, prob)
    allObsts = [fixedObst] + poseObsts + reachObsts
    robObsts = [fixedObst] + poseObsts
    regShapes = [regShape for (_,regShape,_,_) in inConds]
    robot = shWorld.robot
    # 1. Find plausible grasps -- could skip and just use legalGrasps
    # 2. Find combinations of poses and grasps

################
## Drawing
################
    
def drawPoseConf(pbs, placeB, conf, confAppr, prob, win, color = None):
    ws = pbs.getShadowWorld(prob)
    ws.world.getObjectShapeAtOrigin(placeB.obj).applyLoc(placeB.objFrame()).draw(win, color=color)
    conf.draw(win, color=color)

def drawObjAndShadow(pbs, placeB, prob, win, color = None):
    # Draw the object in its native color, but use the argument for the shadow
    ws = pbs.getShadowWorld(prob)
    obj = placeB.obj
    ws.world.getObjectShapeAtOrigin(obj).applyLoc(placeB.objFrame()).draw(win)
    if shadowName(obj) in ws.world.objects:
        ws.world.getObjectShapeAtOrigin(shadowName(obj)).applyLoc(placeB.objFrame()).draw(win, color=color)

################
## Looking at goal fluents
################
            
def getGoalInConds(goalConds, X=[]):
    # returns a list of [(obj, regShape, prob)]
    fbs = fbch.getMatchingFluents(goalConds,
                                  Bd([In(['Obj', 'Reg']), 'Val', 'P'], True))
    return [(b['Obj'], b['Reg'], b['P']) \
            for (f, b) in fbs if isGround(b.values())]

def sameBase(goalConds):
    # Return None if there is no sameBase requirement; otherwise
    # return base pose
    fbs = fbch.getMatchingFluents(goalConds,
                                  BaseConf(['B', 'D'], True))
    result = None
    for (f, b) in fbs:
        base = b['B']
        if not isVar(base):
            assert result is None, 'More than one Base fluent'
            result = tuple(base)
    return result

def targetConf(goalConds):
    fbs_conf = [(f, b) for (f, b) \
                in getMatchingFluents(goalConds,
                                      Conf(['Mu', 'Delta'], True))]
    fbs_crnb = fbch.getMatchingFluents(goalConds,
                                       Bd([CanReachNB(['Start', 'End', 'Cond']),
                                           True, 'P'], True))
    if not (fbs_conf and fbs_crnb): return None
    conf = None
    for (fconf, bconf) in fbs_conf:
        for (fcrnb, bcrnb) in fbs_crnb:
            confVar = fconf.args[0]
            if isVar(confVar) and fconf.args[0] == fcrnb.args[0].args[0]:
                conf = fcrnb.args[0].args[1]
    return conf

def pathShape(path, prob, pbs, name):
    assert isinstance(path, (list, tuple))
    attached = pbs.getShadowWorld(prob).attached
    return shapes.Shape([c.placement(attached=attached) for c in path], None, name=name)

def pathObst(cs, cd, p, pbs, name, start=None):
    newBS = pbs.copy()
    newBS = newBS.updateFromGoalPoses(cd, permShadows=True)
    key = (cs, newBS, p)
    if key in pbs.beliefContext.pathObstCache:
        return pbs.beliefContext.pathObstCache[key]
    path,  viol = canReachHome(newBS, cs, p, Violations(), homeConf = start)
    if debug('pathObst'):
        newBS.draw(p, 'W')
        cs.draw('W', 'red', attached=newBS.getShadowWorld(p).attached)
        print 'condition', cd
    if not path:
        if debug('pathObst'):
            print 'pathObst', 'failed to find path to conf in red', (cs, p, newBS)
        ans = None
    else:
        ans = pathShape(path, p, newBS, name)
    pbs.beliefContext.pathObstCache[key] = ans
    return ans

def getReachObsts(goalConds, pbs):
    fbs = fbch.getMatchingFluents(goalConds,
             Bd([CanPickPlace(['Preconf', 'Ppconf', 'Hand', 'Obj', 'Pose',
                               'Posevar', 'Posedelta', 'Poseface',
                               'Graspface', 'Graspmu', 'Graspvar', 'Graspdelta',
                                'Op', 'Inconds']),
                                True, 'P'], True))
    obstacles = []
    for (f, b) in fbs:
        crhObsts = getCRHObsts([Bd([fc, True, b['P']], True) \
                                for fc in f.args[0].getConds()], pbs)
        if crhObsts is None:
            return None
        obstacles.extend(crhObsts)

    # Now look for standalone CRH, CRNB and CP
    basicCRH = getCRHObsts(goalConds, pbs)
    if basicCRH is None: return None
    obstacles.extend(basicCRH)

    basicCRNB = getCRNBObsts(goalConds, pbs) 
    if basicCRNB is None: return None
    obstacles.extend(basicCRNB)

    basicCP = getCPObsts(goalConds, pbs) 
    if basicCP is None: return None
    obstacles.extend(basicCP)
        
    return obstacles

def pushPathObst(obj, hand, poseFace, prePose, pose, preConf, pushConf,
                 postConf, posevar, prePoseVar, poseDelta, cond, p, pbs, name):
    newBS = pbs.copy()
    newBS = newBS.updateFromGoalPoses(cond, permShadows=True)
    path,  viol = canPush(newBS, obj, hand, poseFace, prePose, pose,
                          preConf, pushConf, postConf, posevar,
                          prePoseVar, poseDelta, p, Violations())
    if debug('pathObst'):
        newBS.draw(p, 'W')
        cs.draw('W', 'red', attached=newBS.getShadowWorld(p).attached)
        print 'condition', cd
    if not path:
        if debug('pathObst'):
            print 'pathObst', 'failed to find path to conf in red', (cs, p, newBS)
        ans = None
    else:
        ans = pathShape(path, p, newBS, name)
    #pbs.beliefContext.pathObstCache[key] = ans
    return ans

def getCPObsts(goalConds, pbs):
    fbs = fbch.getMatchingFluents(goalConds,
                                  Bd([CanPush(['Obj', 'Hand', 'PoseFace', 'PrePose', 'Pose',
                                               'PreConf', 'PushConf',
                                               'PostConf', 'PoseVar', 'PrePoseVar', 'PoseDelta',
                                               'PreCond']),  True, 'Prob'], True))
    world = pbs.getWorld()
    obsts = []
    index = 0
    for (f, b) in fbs:
        if not isGround(b.values()): continue
        if debug('getReachObsts'):
            print 'GRO', f
        ignoreObjects = set([])
        obst = pushPathObst(b['Obj'], b['Hand'], b['PoseFace'], b['PrePose'], b['Pose'],
                            b['PreConf'], b['PushConf'],
                            b['PostConf'], b['PoseVar'], b['PrePoseVar'], b['PoseDelta'],
                            b['PreCond'], b['Prob'], pbs,
                            name= 'reachObst%d'%index)
        index += 1
        if not obst:
            debugMsg('getReachObsts', ('path fail', f, b.values()))
            return None
        # Look at Poses in conditions; they are exceptions
        pfbs = fbch.getMatchingFluents(b['PreCond'],
                                       B([Pose(['Obj', 'Face']), 'Mu', 'Var', 'Delta', 'P'], True))
        for (pf, pb) in pfbs:
            if isGround(pb.values()):
                ignoreObjects.add(pb['Obj'])
        obsts.append((ignoreObjects, obst))
    debugMsg('getReachObsts', ('->', len(obsts), 'CRH NB obsts'))
    return obsts

def getCRNBObsts(goalConds, pbs):
    fbs = fbch.getMatchingFluents(goalConds,
                             Bd([CanReachNB(['Start', 'End', 'Cond']),
                                 True, 'P'], True))
    world = pbs.getWorld()
    obsts = []
    index = 0
    for (f, b) in fbs:
        if not isGround(b.values()): continue
        if debug('getReachObsts'):
            print 'GRO', f
        ignoreObjects = set([])
        obst = pathObst(b['Start'], b['Cond'], b['P'], pbs,
                        name= 'reachObst%d'%index, start=b['End'])
        index += 1
        if not obst:
            debugMsg('getReachObsts', ('path fail', f, b.values()))
            return None
        # Look at Poses in conditions; they are exceptions
        pfbs = fbch.getMatchingFluents(b['Cond'],
                                       B([Pose(['Obj', 'Face']), 'Mu', 'Var', 'Delta', 'P'], True))
        for (pf, pb) in pfbs:
            if isGround(pb.values()):
                ignoreObjects.add(pb['Obj'])
        obsts.append((ignoreObjects, obst))
    debugMsg('getReachObsts', ('->', len(obsts), 'CRH NB obsts'))
    return obsts

def getCRHObsts(goalConds, pbs):
    fbs = fbch.getMatchingFluents(goalConds,
                             Bd([CanReachHome(['C', 'LAP', 'Cond']),
                                 True, 'P'], True))
    world = pbs.getWorld()
    obsts = []
    index = 0
    for (f, b) in fbs:
        if not isGround(b.values()): continue
        if debug('getReachObsts'):
            print 'GRO', f
        ignoreObjects = set([])
        obst = pathObst(b['C'], b['Cond'], b['P'], pbs, name= 'reachObst%d'%index)
        index += 1
        if not obst:
            debugMsg('getReachObsts', ('path fail', f, b.values()))
            return None
        # Look at Poses in conditions; they are exceptions
        pfbs = fbch.getMatchingFluents(b['Cond'],
                                  B([Pose(['Obj', 'Face']), 'Mu', 'Var', 'Delta', 'P'], True))
        for (pf, pb) in pfbs:
            if isGround(pb.values()):
                ignoreObjects.add(pb['Obj'])
        obsts.append((ignoreObjects, obst))
    debugMsg('getReachObsts', ('->', len(obsts), 'CRH obsts'))
    return obsts

# Returns (hand, obj) for Holding fluents.   Leave it out if obj is 'none'
def getHolding(goalConds):
    pfbs = fbch.getMatchingFluents(goalConds,
                                   Bd([Holding(['Hand']), 'Obj', 'P'], True))
    held = []
    for (pf, pb) in pfbs:
        if isGround(pb.values()):
            if pb['Obj'] != 'none':
                held.append((pb['Hand'], pb['Obj']))
    return held

def bboxRandomDrawCoords(bb):
    pt = tuple([random.uniform(bb[0,i], bb[1,i]) for i in xrange(3)])
    return pt[:2]+(bb[0,2],)

# find bbox for CI_1(2), that is, displacements of bb1 that place it
# inside bb2.  Assumes that bb1 has origin at 0,0.
def bboxInterior(bb1, bb2):
    for j in xrange(3):
        di1 = bb1[1,j] - bb1[0,j]
        di2 = bb2[1,j] - bb2[0,j]
        if di1 > di2: return None
    return np.array([[bb2[i,j] - bb1[i,j] for j in range(3)] for i in range(2)])

# !! Should pick relevant orientations... or more samples.
angleList = [-math.pi/2. -math.pi/4., 0.0, math.pi/4, math.pi/2]

def potentialRegionPoseGen(pbs, obj, placeB, graspB, prob, regShapes, reachObsts, hand, base,
                           maxPoses = 30):
    def interpolateVars(maxV, minV, n):
        deltaV = [(maxV[i]-minV[i])/(n-1.) for i in range(4)]
        if all([d < 0.001 for d in deltaV]):
            return [maxV]
        Vs = []
        for j in range(n):
            Vs.append(tuple([maxV[i]-j*deltaV[i] for i in range(4)]))
        return Vs
        
    maxVar = placeB.poseD.var
    minVar = pbs.domainProbs.obsVarTuple
    count = 0
    # Preferentially use large variance...
    for medVar in interpolateVars(maxVar, minVar, 4):
        pB = placeB.modifyPoseD(var=medVar)
        if debug('potentialRegionPoseGen'):
            print 'potentialRegionPoseGen var', medVar
        for pose in potentialRegionPoseGenAux(pbs, obj, placeB, graspB, prob, regShapes,
                                              reachObsts, hand, base, maxPoses):
            yield pose
            if count > maxPoses: return
            count += 1

# TODO: Should structure this as a mini-batch generator!

def potentialRegionPoseGenAux(pbs, obj, placeB, graspB, prob, regShapes, reachObsts, hand, base,
                              maxPoses = 30):
    def genPose(rs, angle, point):
        (x,y,z,_) = point
        # Support pose, we assume that sh is on support face
        pose = hu.Pose(x,y,z, 0.)     # shRotations is already rotated
        sh = shRotations[angle].applyTrans(pose)
        if debug('potentialRegionPoseGen'):
            sh.draw('W', 'brown')
            wm.getWindow('W').update()
            
        if inside(sh, rs) and \
           all(not sh.collides(obst) for (ig, obst) in reachObsts if obj not in ig):
            debugMsg('potentialRegionPoseGen', ('-> pose', pose))
            return pose
        else:
            debugMsg('potentialRegionPoseGen', ('fail pose', pose))
            pbs.draw(prob, 'W'); sh.draw('W', 'brown'); rs.draw('W', 'purple')
            #pdb.set_trace()

    def poseViolationWeight(pose):
        pB = placeB.modifyPoseD(mu=pose)
        for gB in graspGen(pbs, obj, graspB):
            c, ca, v = next(potentialGraspConfGen(pbs, pB, gB, None, hand, base, prob, nMax=1),
                            (None,None,None))
            if v:
                if debug('potentialRegionPoseGenWeight'):
                    pbs.draw(prob, 'W'); c.draw('W')
                    debugMsg('potentialRegionPoseGenWeight', 'v=%s'%v,
                             'weight=%s'%str(v.weight()), 'pose=%s'%pose)
                return v.weight() + baseDist(pbs.conf, ca)
        return None

    clearance = 0.01
    if debug('potentialRegionPoseGen'):
        pbs.draw(prob, 'W')
        for rs in regShapes: rs.draw('W', 'purple')
    ff = placeB.faceFrames[placeB.support.mode()]
    shWorld = pbs.getShadowWorld(prob)
    
    objShadow = pbs.objShadow(obj, shadowName(obj), prob, placeB, ff)
    if placeB.poseD.mode():
        tr('potentialRegionPoseGen', 'pose specified', placeB.poseD.mode(),
           ol = True)
        sh = objShadow.applyLoc(placeB.objFrame()).prim()
        verts = sh.vertices()
        if any(any(np.all(np.all(np.dot(r.planes(), verts) <= tiny, axis=1)) \
                   for r in rs.parts()) \
               for rs in regShapes)  and \
           all(not sh.collides(obst) for (ig, obst) in reachObsts if obj not in ig):
            tr('potentialRegionPoseGen',
               'pose specified and safely in region',
               placeB.poseD.mode(), ol = True)
            yield placeB.poseD.mode()
        else:
            tr('potentialRegionPoseGen',
               'pose specified and not safely in region')

    shRotations = dict([(angle, objShadow.applyTrans(hu.Pose(0,0,0,angle)).prim()) \
                        for angle in angleList])
    obstCost = 10.
    hyps = []                         # (index, cost)
    points = []                       # [(angle, xyz1)]
    count = 0
    world = pbs.getWorld()
    for rs in regShapes:
        tr('potentialRegionPoseGen', obj, rs.name(), hand, ol = True)
        if debug('potentialRegionPoseGen'):
            print 'Considering region', rs.name()
        for (angle, shRot) in shRotations.items():
            bI = CI(shRot, rs.prim())
            if bI is None:
                if debug('potentialRegionPoseGen'):
                    print 'bI is None for angle', angle
                    raw_input('bI')
                continue
            elif debug('potentialRegionPoseGen'):
                bI.draw('W', 'cyan')
                debugMsg('potentialRegionPoseGen', 'Region interior in cyan for angle', angle)
            coFixed = squashOne([xyCOParts(shRot, o) for o in shWorld.getObjectShapes() \
                                 if o.name() in shWorld.fixedObjects])
            coObst = squashOne([xyCOParts(shRot, o) for o in shWorld.getNonShadowShapes() \
                                if o.name() not in shWorld.fixedObjects])
            coShadow = squashOne([xyCOParts(shRot, o) for o in shWorld.getShadowShapes() \
                                  if o.name() not in shWorld.fixedObjects])
            if debug('potentialRegionPoseGen'):
                for co in coFixed: co.draw('W', 'red')
                for co in coObst: co.draw('W', 'brown')
                for co in coShadow: co.draw('W', 'orange')
            z0 = bI.bbox()[0,2] + clearance
            # for point in bboxGridCoords(bI.bbox(), res = 0.01, z=z0):
            for point in bboxRandomCoords(bI.bbox(), n=100, z=z0):
                pt = point.reshape(4,1)
                if any(np.all(np.dot(co.planes(), pt) <= tiny) for co in coFixed):
                    if debug('potentialRegionPoseGen'):
                        shapes.pointBox(pt.T[0]).draw('W', 'blue')
                    continue
                cost = 0
                for co in coObst:
                    if np.all(np.dot(co.planes(), pt) <= tiny): cost += obstCost
                for co in coShadow:
                    if np.all(np.dot(co.planes(), pt) <= tiny): cost += 0.5*obstCost
                points.append((angle, point.tolist()))

                # Randomized
                # hyp = (count, 1./cost if cost else 1.)
                hyp = (cost, rs, count)

                hyps.append(hyp)
                count += 1
    if hyps:
        # Randomized
        # pointDist = DDist(dict(hyps))
        # pointDist.normalize()
        hyps = sorted(hyps)
        # Randomize by regions
        levels = []
        values = []
        for p in hyps:
            (c, r, i) = p
            if not values or values[-1] != c:
                values.append(c)
                levels.append([p])
            else:
                levels[-1].append(p)
        for l in levels:
            random.shuffle(l)
        pointDist = []
        for l in levels: pointDist.extend(l)
        debugMsg('potentialRegionPoseGen', 'Invalid points in blue, len(valid)=%d'%len(pointDist))
    else:
        debugMsg('potentialRegionPoseGen', 'Invalid points in blue, no valid points in region')
        return
    count = 0
    maxTries = min(2*maxPoses, len(pointDist))
    if False: # glob.inHeuristic:
        tries = 0
        while count < maxPoses and tries < maxTries:
            tries += 1
            # Randomized
            # index = pointDist.draw()
            cost, rs, index = pointDist[tries]
            angle, point = points[index]
            pose = genPose(rs, angle, point)
            if not pose: continue
            count += 1
            if debug('potentialRegionPoseGen'):
                print '->', pose, 'cost=', cost
                # shRotations is already rotated
                (x,y,z,_) = pose.xyztTuple()
                shRotations[angle].applyTrans(hu.Pose(x,y,z, 0.)).draw('W', 'green')
            yield pose
    else:
        costHistory = []
        poseHistory = []
        historySize = 5
        tries = 0
        while count < maxPoses and tries < maxTries:
            # Randomized
            # index = pointDist.draw()
            hcost, rs, index = pointDist[tries]
            angle, point = points[index]
            tries += 1
            p = genPose(rs, angle, point)
            if not p: continue
            cost = poseViolationWeight(p)
            if cost is None: continue
            if len(costHistory) < historySize:
                costHistory.append(cost)
                poseHistory.append(p)
                continue
            elif cost > min(costHistory):
                minIndex = costHistory.index(min(costHistory))
                pose = poseHistory[minIndex]
                poseCost = costHistory[minIndex]
                if debug('potentialRegionPoseGen'):
                    print 'pose cost', costHistory[minIndex]
                costHistory[minIndex] = cost
                poseHistory[minIndex] = p
            else:                           # cost <= min(costHistory)
                pose = p
                poseCost = cost
                if debug('potentialRegionPoseGen'): print 'pose cost', cost
            count += 1
            if debug('potentialRegionPoseGen'):
                print '->', pose, 'cost=', poseCost
                # shRotations is already rotated
                (x,y,z,_) = pose.xyztTuple()
                shRotations[angle].applyTrans(hu.Pose(x,y,z, 0.)).draw('W', 'green')
            yield pose
    if True: # debug('potentialRegionPoseGen'):
        print 'Tried', tries, 'returned', count, 'for regions', [r.name() for r in regShapes]
    return

def baseDist(c1, c2):
    (x1,y1,_) = c1['pr2Base']
    (x2,y2,_) = c2['pr2Base']
    return ((x2-x1)**2 + (y2-y1)**2)**0.5
    
#############
# Selecting safe points in region

def bboxRandomCoords(bb, n=20, z=None):
    ((x0, y0, z0), (x1, y1, z1)) = tuple(bb)
    if z is None: z = z0
    points = []
    for i in xrange(n):
        x = random.uniform(x0, x1)
        y = random.uniform(y0, y1)
        points.append(np.array([x, y, z, 1.]))
    return points

def confDelta(c1, c2):
    return max([max([abs(x-y) for (x,y) in zip(c1.conf[k], c2.conf[k])]) \
                for k in c1 if k in c2])

def findGraspConfEntries(conf):
    return [(c, ca, pbs, prob, viol) \
            for (c, ca, pbs, prob, viol) in graspConfs \
            if confDelta(c, conf) < 0.001 or confDelta(ca, conf) < 0.001]


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
