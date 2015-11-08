import pdb
import numpy as np
import hu
import time
import shapes
from transformations import rotation_matrix
import planGlobals as glob
from traceFile import debugMsg, debug, tr
from pr2Robot import gripperFaceFrame
from planUtil import Violations
from pr2Util import shadowName, objectName 
from pr2Visible import visible, lookAtConf, viewCone, findSupportTableInPbs
from pr2RRT import planRobotGoalPath, interpolatePath

Ident = hu.Transform(np.eye(4))            # identity transform
tiny = 1.0e-6

################
# Basic tests for fluents and generators
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

def findRegionParent(pbs, region):
    # In case, this is a bState and not a pbs
    pbs = pbs.pbs
    regs = pbs.getWorld().regions
    for (obj, stuff) in regs.items():
        for (regName, regShape, regTr) in stuff:
            if regName == region:
                return obj
    raw_input('No parent object for region '+str(region))
    return None

# probability is: pObj * pParent * pObjFitsGivenRelativeVar = prob
def inTest(pbs, obj, regName, prob, pB=None):
    regs = pbs.getWorld().regions
    parent = findRegionParent(pbs, regName)
    pObj = pbs.poseModeProbs[obj]
    pParent = pbs.poseModeProbs[parent]
    pFits = prob / (pObj * pParent)
    if pFits > 1: return False

    # compute a shadow for this object
    placeB = pB or pbs.getPlaceB(obj)
    faceFrame = placeB.faceFrames[placeB.support.mode()]

    # !! Clean this up
    sh = pbs.objShadow(obj, shadowName(obj), pFits, placeB, faceFrame)
    shadow = sh.applyLoc(placeB.objFrame()) # !! is this right?
    shWorld = pbs.getShadowWorld(prob)
    region = shWorld.regionShapes[regName]

    ans = any([np.all(np.all(np.dot(r.planes(), shadow.prim().vertices()) <= tiny, axis=1)) \
               for r in region.parts()])

    tr('testVerbose', 'In test, shadow in orange, region in purple',
       (shadow, region, ans), draw = [(pbs, prob, 'W'),
                                      (shadow, 'W', 'orange'),
                                      (region, 'W', 'purple')], snap=['W'])
    return ans

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

ppConfs = {}

def canPickPlaceTest(pbs, preConf, ppConf, hand, objGrasp, objPlace, p,
                     op='pick', quick = False):
    tag = 'canPickPlaceTest'
    obj = objGrasp.obj
    collides = pbs.getRoadMap().checkRobotCollision
    if debug(tag):
        print zip(('preConf', 'ppConf', 'hand', 'objGrasp', 'objPlace', 'p', 'pbs'),
                  (preConf, ppConf, hand, objGrasp, objPlace, p, pbs))
    if not legalGrasp(pbs, ppConf, hand, objGrasp, objPlace):
        debugMsg(tag, 'Grasp is not legal in canPickPlaceTest')
        return None, 'Legal grasp'
    # pbs.getRoadMap().approachConfs[ppConf] = preConf
    violations = Violations()           # cumulative
    # 1.  Can move from home to pre holding nothing with object placed at pose
    if preConf:
        pbs1 = pbs.copy().updatePermObjBel(objPlace).updateHeldBel(None, hand)
        if op == 'pick':
            oB = objPlace.modifyPoseD(var=4*(0.0,)) # ignore uncertainty
            oB.delta = 4*(0.0,)
        else:
            oB = objPlace.modifyPoseD(var=pbs.domainProbs.placeVar)
            oB.delta = pbs.domainProbs.placeDelta
        pbs1 = pbs1.updatePermObjBel(oB).addAvoidShadow([obj])
        if debug(tag):
            pbs1.draw(p, 'W')
            debugMsg(tag, 'H->App, obj@pose (condition 1)')
        if quick:
            violations = pbs.getRoadMap().confViolations(preConf, pbs1, p, violations)
            path = [preConf]
        else:
            path, violations = canReachHome(pbs1, preConf, p, violations)
        if not violations:
            debugMsg(tag, 'Failed H->App, obj=pose (condition 1)')
            return None, '1. H->App obj@pose'
        elif debug(tag):
            for c in path: c.draw('W', attached = pbs1.getShadowWorld(p).attached)
            debugMsg(tag, 'path 1')

    # preConfShape = preConf.placement(attached = pbs1.getShadowWorld(p).attached)
    objShadow = objPlace.shadow(pbs1.getShadowWorld(p))
    # Check visibility at preConf (for pick)
    if op =='pick' and not (glob.inHeuristic or quick):
        path = canView(pbs1, p, preConf, hand, objShadow)
        if path:
            debugMsg(tag, 'Succeeded visibility test for pick')
            preConfView = path[-1]
            if preConfView != preConf:
                path, violations = canReachHome(pbs1, preConfView, p, violations)
                if not violations:
                    debugMsg(tag, 'Cannot reachHome with retracted arm')
                    pbs1.draw(p, 'W'); preConfView.draw('W', 'red')
                    debugMsg(tag, 'canPickPlaceTest - Cannot reachHome with retracted arm')
                    return None, 'Obj visibility'
        else:
            debugMsg(tag, 'Failed visibility test for pick')
            return None, 'Obj visibility'
            
    # 2 - Can move from home to pre holding the object
    pbs2 = pbs.copy().excludeObjs([obj]).updateHeldBel(objGrasp, hand)
    if debug(tag):
        pbs2.draw(p, 'W'); preConf.draw('W', attached = pbs2.getShadowWorld(p).attached)
        debugMsg(tag, 'H->App, obj=held (condition 2)')
    if quick:
        violations = pbs.getRoadMap().confViolations(preConf, pbs2, p, violations)
        path = [preConf]
    else:
        path, violations = canReachHome(pbs2, preConf, p, violations)
    if not violations:
        debugMsg(tag + 'Failed H->App, obj=held (condition 2)')
        return None, '2. H->App, held=obj'
    elif debug(tag):
        for c in path: c.draw('W', attached = pbs2.getShadowWorld(p).attached)
        debugMsg(tag, 'path 2')

    # Check visibility of support table at preConf (for pick AND place)
    if op in ('pick', 'place') and not (glob.inHeuristic or quick):
        tableB = findSupportTableInPbs(pbs1, objPlace.obj) # use pbs1 so obj is there
        assert tableB
        if debug(tag): print 'Looking at support for', obj, '->', tableB.obj
        lookDelta = pbs2.domainProbs.minDelta
        lookVar = pbs2.domainProbs.obsVarTuple
        tableB2 = tableB.modifyPoseD(var = lookVar)
        tableB2.delta = lookDelta
        prob = 0.95
        shadow = tableB2.shadow(pbs2.updatePermObjBel(tableB2).getShadowWorld(prob))
        if collides(preConf, shadow, attached = pbs2.getShadowWorld(p).attached):
            preConfShape = preConf.placement(attached = pbs2.getShadowWorld(p).attached)
            pbs2.draw(p, 'W'); preConfShape.draw('W', 'cyan'); shadow.draw('W', 'cyan')
            debugMsg('Preconf collides for place in canPickPlaceTest')
            return None, 'Support shadow collision'
        if collides(ppConf, shadow): # ppConfShape.collides(shadow):
            ppConfShape = ppConf.placement() # no attached
            pbs2.draw(p, 'W'); ppConfShape.draw('W', 'magenta'); shadow.draw('W', 'magenta')
            debugMsg(tag, 'PPconf collides for place in canPickPlaceTest')
            return None, 'Support shadow collision'
        if not canView(pbs2, p, preConf, hand, shadow):
            preConfShape = preConf.placement(attached = pbs2.getShadowWorld(p).attached)
            pbs2.draw(p, 'W'); preConfShape.draw('W', 'orange'); shadow.draw('W', 'orange')
            debugMsg(tag, 'Failing to view for place in canPickPlaceTest')
            return None, 'Support visibility'

    # 3.  Can move from home to pick with object placed at pose (0 var)
    oB = objPlace.modifyPoseD(var=4*(0.0,)) # ignore uncertainty
    oB.delta = 4*(0.0,)
    pbs3 = pbs.copy().updatePermObjBel(oB).updateHeldBel(None, hand)
    if debug(tag):
        pbs3.draw(p, 'W')
        debugMsg(tag, 'H->Target, obj placed (0 var) (condition 3)')
    if quick:
        violations = pbs.getRoadMap().confViolations(ppConf, pbs3, p, violations)
        path = [ppConf]
    else:
        path, violations = canReachHome(pbs3, ppConf, p, violations)
    if not violations:
        debugMsg(tag, 'Failed H->Target  (condition 3)')
        return None, '3. H->Target obj@pose 0var'
    elif debug(tag):
        for c in path: c.draw('W', attached = pbs3.getShadowWorld(p).attached)
        debugMsg(tag, 'path 3')
    # 4.  Can move from home to pick while holding obj with zero grasp variance
    gB = objGrasp.modifyPoseD(var=4*(0.0,)) # ignore uncertainty
    gB.delta = 4*(0.0,)
    pbs4 = pbs.copy().excludeObjs([obj]).updateHeldBel(gB, hand)
    if debug(tag):
        pbs4.draw(p, 'W'); ppConf.draw('W', attached = pbs4.getShadowWorld(p).attached)
        debugMsg(tag, 'H->Target, holding obj (0 var) (condition 4)')
    if quick:
        violations = pbs.getRoadMap().confViolations(ppConf, pbs4, p, violations)
        path = [ppConf]
    else:
        path, violations = canReachHome(pbs4, ppConf, p, violations)
    if not violations:
        debugMsg(tag, 'Failed H->Target held=obj(condition 4)')
        return None, '4. H->Target held=obj 0var'
    elif debug(tag):
        for c in path: c.draw('W', attached = pbs4.getShadowWorld(p).attached)
        debugMsg(tag, 'path 4')
    debugMsg(tag, ('->', violations))

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
            shapeShadow = None, maxIter = 50, findPath = True):
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
    if shapeShadow:
        avoid = shapes.Shape([vc, shape, shapeShadow], None)
    else:
        avoid = shapes.Shape([vc, shape], None)
    # confPlace = conf.placement(attached=attached)
    if not collides(conf, avoid, attached=attached):
        if debug('canView'):
            print 'canView - no collisions'
        return [conf]
    elif not findPath:
        return []
    # !! don't move arms to clear view of fixed objects
    if not permanent(objectName(shape.name())):
        if debug('canView'):
            avoid.draw('W', 'red')
            conf.draw('W', attached=attached)
            debugMsg('canView', 'ViewCone collision')
        pathFull = []
        for h in ['left', 'right']:     # try both hands
            chainName = robot.armChainNames[h]
            armChains = [chainName, robot.gripperChainNames[h]]
            if not (collides(conf, avoid, attached=attached, selectedChains=armChains) \
                   if glob.useCC else avoid.collides(armShape(conf, h))):
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
                    avoid.draw('W', 'green')
                    debugMsg('canView', 'Retract arm')
            if debug('canView') or debug('canViewFail'):
                if not path:
                    pbs.draw(prob, 'W')
                    conf.draw('W', attached=attached)
                    avoid.draw('W', 'red')
                    raw_input('canView - no path')
            if path:
                pathFull.extend(path)
            else:
                return []
        return pathFull
    else:
        if debug('canView'):
            print 'canView - ignore view cone collision for perm object', shape
        return [conf]


# Pushing                

# returns path, violations
pushStepSize = 0.01
def canPush(pbs, obj, hand, poseFace, prePose, pose,
            preConf, pushConf, postConf, poseVar, prePoseVar,
            poseDelta, prob, initViol, prim=False):
    tag = 'canPush'
    held = pbs.getHeld(hand)
    newBS = pbs.copy()
    if held != 'none':
        tr(tag, 'Hand=%s is holding %s in pbs'%(hand, held))
        newBS.updateHeld('none', None, None, hand, None)
    if obj in [newBS.getHeld(h) for h in ('left', 'right')]:
        tr(tag, '=> obj is in the other hand')
        # LPK!! Changed hand below to otherHand(hand)
        assert pbs.getHeld(otherHand(hand)) == obj
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
