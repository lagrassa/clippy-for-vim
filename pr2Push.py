
import math
from operator import itemgetter
from fbch import Function
from geom import bboxInside, bboxVolume, vertsBBox, bboxContains, bboxVolume
import numpy as np
from traceFile import debug, debugMsg, tr, trAlways
from shapes import thingFaceFrames, drawFrame
import hu
from planUtil import ObjGraspB, PoseD, PushResponse
from pr2Util import GDesc, trArgs, otherHand, bboxGridCoords
from pr2GenAux import *
from pr2PlanBel import getConf
from pr2Fluents import pushPath
from pr2PlanBel import getConf, getGoalPoseBels
import mathematica
reload(mathematica)
import windowManager3D as wm
from subprocess import call
import time
import pdb
import random

# Pick poses and confs for pushing an object

# For now, consider only a small set of stable push directions.
# Pretend that centroid of xyPrim is the center of friction.  The push
# path needs to stay inside the regions and the robot needs to be able
# to follow it without colliding with permanent obstacles.

# TODO: Pick a reasonable value

# How many paths to generate for a particular hand contact with the
# object, these will differ by the base placement.
maxPushPaths = 10
maxDone = 1

pushGenCacheStats = [0, 0]
pushGenCache = {}

minPushLength = 0.02

useDirectPush = False

class PushGen(Function):
    def fun(self, args, goalConds, bState):
        for ans in pushGenGen(args, goalConds, bState):
            # Verify that it's feasible
            (obj, pose, support, poseVar, poseDelta, confdelta, prob) = args
            (hand, prePose, preConf, pushConf, postConf) = ans.pushTuple()
            path, viol = canPush(bState.pbs, obj, hand, support, hu.Pose(*prePose),
                                 pose, preConf, pushConf, postConf, poseVar,
                                 poseVar, poseDelta, prob,
                                 Violations(), prim=False)
            if viol == None:
                print 'PushGen generated infeasible answer'
            else:
                tr('pushGen', '->', 'final', str(ans))
                yield ans.pushTuple()
        tr('pushGen', '-> completely exhausted')

def pushGenGen(args, goalConds, bState):
    (obj, pose, support, posevar, posedelta, confdelta, prob) = args
    tag = 'pushGen'
    pbs = bState.pbs.copy()
    world = pbs.getWorld()
    # This is the target placement
    placeB = ObjPlaceB(obj, world.getFaceFrames(obj), support,
                       PoseD(pose, posevar), delta=posedelta)
    # Figure out whether one hand or the other is required;  if not, do round robin
    leftGen = pushGenTop((obj, placeB, 'left', prob),
                         goalConds, pbs)
    rightGen = pushGenTop((obj, placeB, 'right', prob),
                          goalConds, pbs)
    # Run the push generator with each of the hands
    for ans in chooseHandGen(pbs, goalConds, obj, None, leftGen, rightGen):
        yield ans

def pushGenTop(args, goalConds, pbs,
               partialPaths=False, reachObsts=[]):
    (obj, placeB, hand, prob) = args
    startTime = time.clock()
    tag = 'pushGen'
    trArgs(tag, ('obj', 'placeB', 'hand', 'prob'), args, goalConds, pbs)
    if obj == 'none' or not placeB:
        tr(tag, '=> obj is none or no placeB, failing')
        return
    if goalConds:
        if getConf(goalConds, None):
            tr(tag, '=> goal conf specified, failing')
            return
        for (h, o) in getHolding(goalConds):
            if h == hand:
                # TODO: we could push with the held object
                tr(tag, '=> Hand=%s is holding in goal, failing'%hand)
                return
        base = sameBase(goalConds)
    else:
        base = None
    # Set up pbs
    newBS = pbs.copy()
    # Just placements specified in goal
    newBS = newBS.updateFromGoalPoses(goalConds)
    tr(tag, 'Goal conditions', draw=[(newBS, prob, 'W')], snap=['W'])
    held = newBS.held[hand].mode()
    if held != 'none':
        # Get rid of held object in newBS, we'll need to add this to
        # violations in canPush...
        tr(tag, 'Hand=%s is holding %s in pbs'%(hand,held))
        newBS.updateHeld('none', None, None, hand, None)
    if held == obj:
        curPB = None
    elif obj in [h.mode() for h in newBS.held.values()]:
        tr(tag, 'obj is in other hand')
        newBS.updateHeld('none', None, None, otherHand(hand), None)
        curPB = None
    else:
        curPB = newBS.getPlaceB(obj, default=False)
        if curPB:
            tr(tag, 'obj is placed')
    # Check the cache, otherwise call Aux
    pushGenCacheStats[0] += 1
    key = (newBS, placeB, hand, base, prob, partialPaths, glob.inHeuristic)
    val = pushGenCache.get(key, None)
    if val != None:
        pushGenCacheStats[1] += 1
        # re-start the generator
        memo = val.copy()
        if debug(tag):
            print tag, 'cached, with len(values)=', len(memo.values)
    else:
        gen = pushGenAux(newBS, placeB, hand, base, curPB, prob,
                         partialPaths=partialPaths, reachObsts=reachObsts)
        memo = Memoizer(tag, gen)
        pushGenCache[key] = memo
        if debug(tag):
            print tag, 'creating new pushGenAux generator'
    rm = newBS.getRoadMap()
    for ans in memo:
        (hand, prePoseTuple, preConf, pushConf, postConf) = ans.pushTuple()
        prePose = hu.Pose(*prePoseTuple)
        # Double check that preConf is safe - this should not be necessary...
        testBS = newBS.copy()
        testBS.updatePermObjPose(placeB.modifyPoseD(prePose)) # obj at prePose
        testBS.draw(prob, 'W'); preConf.draw('W')
        viol = rm.confViolations(preConf, testBS, prob)

        if not glob.inHeuristic:
            print 'pushGen violations', viol
        if not viol:
            if debug(tag):
                raw_input('Collision in pushPath')
            continue
        # End double check
        yield ans
    tr(tag, '=> pushGenTop exhausted for', hand)

def choosePrim(shape):
    return sorted(shape.toPrims(),
                  key = lambda p: bboxVolume(p.bbox()), reverse=True)[0]

def pushGenAux(pbs, placeB, hand, base, curPB, prob,
               partialPaths=False, reachObsts=[]):
    tag = 'pushGen'
    # The shape at target, without any shadow
    shape = placeB.shape(pbs.getWorld())
    if curPB:
        curPose = curPB.poseD.mode()
        targetPose = placeB.poseD.mode()
        # If the current position is almost at the target, return
        if curPose.distance(targetPose)  < minPushLength:
            tr(tag, '=> Target pose is too close to current pose, failing',
               curPose, targetPose, curPose.distance(targetPose))
    else:
        curPose = None
    # TODO: this should really look for a large Cspace surface
    # Choose a prim for contact
    prim = choosePrim(shape)
    # The xyPrim is extrusion of xy convex hull, so side faces are
    # perp to support - assume original faces were almost that way.
    xyPrim = shape.xyPrim()
    # Location of center at placeB; we'll treat as COM
    center =  np.average(xyPrim.vertices(), axis=1)
    # This should identify arbitrary surfaces, e.g. in shelves.  The
    # bottom of the region is the support polygon.
    supportRegion = findSupportRegionInPbs(pbs, prob, xyPrim)
    # find tool point frames (with Z opposing normal, like face
    # frames) that brings the fingers in contact with the object and
    # span the centroid.
    potentialContacts = []
    for vertical in (True, False):
        potentialContacts.extend(handContactFrames(prim, center, vertical))
    # sort contacts and compute a distance when applicable, entries
    # are: (distance, vertical, contactFrame, width)
    # Sort contacts by nearness to current pose of object
    sortedContacts = sortPushContacts(potentialContacts, placeB.poseD.mode(),
                                      curPose)
    if not sortPushContacts:
        tr(tag, '=> No sorted contacts')
    # Now we have frames for contact with object, we have to generate
    # potential answers following the face normal in the given
    # direction.
    rm = pbs.getRoadMap()               # save typing...
    for (dist, vertical, contactFrame, width) in sortedContacts:
        if debug(tag): print 'dist=', dist
        # construct a graspB corresponding to the push hand pose,
        # determined by the contact frame
        graspB = graspBForContactFrame(pbs, prob, contactFrame,
                                       0.0,  placeB, hand, vertical)
        # This is negative z axis of face
        direction = -contactFrame.matrix[:3,2].reshape(3)
        direction[2] = 0.0            # we want z component exactly 0.
        prePoseOffset = hu.Pose(*(dist*direction).tolist()+[0.0])
        # initial pose for object along direction
        prePose = prePoseOffset.compose(placeB.poseD.mode())
        if debug(tag):
            pbs.draw(prob, 'W')
            shape.draw('W', 'blue'); prim.draw('W', 'green')
            print 'vertical', vertical, 'dist', dist
            drawFrame(contactFrame)
            raw_input('graspDesc frame')
        count = 0                       # how many tries
        doneCount = 0                   # how many went all the way
        pushPaths = []                  # for different base positions
        # Generate confs to place the hand at graspB
        # Try going directly to goal and along the direction of face
        # The direct route only works for vertical pushing...
        for direct in (True, False) if (useDirectPush and vertical and curPose) else (False,):
            for ans in potentialConfs(pbs, prob, placeB,
                                      curPose.pose() if direct else prePose.pose(),
                                      graspB, hand, base):
                if not ans:
                    tr(tag+'Path', 'potential grasp conf is empy')
                    continue
                (c, ca, viol) = ans         # conf, approach, violations
                pushConf = gripSet(c, hand, 2*width) # open fingers
                if debug(tag+'_kin'):
                    pushConf.draw('W', 'orange')
                    raw_input('Candidate conf')
                count += 1
                pathAndViols, reason = pushPath(pbs, prob, graspB, placeB, pushConf,
                                                curPose.pose() if direct else prePose.pose(),
                                                xyPrim, supportRegion, hand, reachObsts=reachObsts)
                if reason == 'done':
                    doneCount +=1 
                    pushPaths.append((pathAndViols, reason))
                elif pathAndViols and partialPaths:
                    pushPaths.append((pathAndViols, reason))
                tr(tag, 'pushPath reason = %s, path len = %d'%(reason, len(pathAndViols)))
                if doneCount >= maxDone and not partialPaths: break
                if count > maxPushPaths: break
            if doneCount >= maxDone and not partialPaths: break
            if count > maxPushPaths: break
        if count == 0 and not glob.inHeuristic:
            print tag, 'Could not find conf for push along', direction[:2]
        # Sort the push paths by violations
        sorted = sortedPushPaths(pushPaths, curPose)
        for i in range(min(len(sorted), maxDone)):
            pp = sorted[i]
            ppre, cpre, ppost, cpost = getPrePost(pp)
            if not ppre or not ppost: continue
            if debug(tag):
                robot = cpre.robot
                print 'pre pose\n', ppre.matrix
                print 'pre conf tool'
                print cpre.cartConf()[robot.armChainNames[hand]].compose(robot.toolOffsetX[hand]).matrix
                print 'post conf tool'
                print cpost.cartConf()[robot.armChainNames[hand]].compose(robot.toolOffsetX[hand]).matrix
            tr(tag, 'pre conf (blue), post conf (pink)',
               draw=[(pbs, prob, 'W'),
                     (cpre, 'W', 'blue'), (cpost, 'W', 'pink')], snap=['W'])

            viol = Violations()
            for (c,v,p) in pathAndViols:
                viol.update(v)
            yield PushResponse(placeB.modifyPoseD(ppre.pose()),
                               placeB.modifyPoseD(ppost.pose()),
                               cpre, cpost, cpre, viol, hand,
                               placeB.poseD.var, placeB.delta)

    tr(tag, '=> pushGenAux exhausted')
    return

def potentialConfs(pbs, prob, placeB, prePose, graspB, hand, base):
    def preGraspConfGen():
        for (c, ca, v) in potentialGraspConfGen(pbs, prePB, graspB, None, hand, base, prob):
            (x, y, th) = c['pr2Base']
            basePose = hu.Pose(x, y, 0, th)
            ans = graspConfForBase(pbs, placeB, graspB, hand, basePose, prob)
            if debug('pushGen_kin'):
                c.draw('W', 'cyan'); print 'ans=', ans
                raw_input('Candidate at prePose')
            if ans: yield ans
    prePB = placeB.modifyPoseD(prePose)
    postGen = potentialGraspConfGen(pbs, placeB, graspB, None, hand, base, prob)
    preGen = preGraspConfGen()
    return roundrobin(postGen, preGen)

# Path entries are (robot conf, violations, object pose)
# Paths can end or start with a series of steps that do not move the
# object, indocated by None for object pose.
def getPrePost(pp):
    cpost, _, _ = pp[-1]
    cpre, _, _ = pp[0]
    for i in range(1, len(pp)):
        _, _, ppost = pp[-i]
        if ppost: break
    for j in range(len(pp)):
        _, _, ppre = pp[j]
        if ppre: break
    return (ppre, cpre, ppost, cpost)

def sortedPushPaths(pushPaths, curPose):
    scored = []
    for (pathAndViols, reason) in pushPaths:
        ppre, cpre, ppost, cpost = getPrePost(pathAndViols)
        vmax = max(v.weight() for (c,v,p) in pathAndViols)
        if ppre and curPose:
            scored.append((vmax + curPose.totalDist(ppre), pathAndViols))
        else:
            scored.append((vmax, pathAndViols))
    scored.sort()
    return [pv for (s, pv) in scored]
                
def findSupportRegionInPbs(pbs, prob, shape):
    tag = 'findSupportRegionInPbs'
    shWorld = pbs.getShadowWorld(prob)
    bbox = shape.bbox()
    bestRegShape = None
    bestVol = 0.
    if debug(tag): print 'find support region for', shape
    for regShape in shWorld.regionShapes.values():
        if debug(tag): print 'considering', regShape
        regBB = regShape.bbox()
        if inside(shape, regShape):
            # TODO: Have global support(region, shape) test?
            if debug(tag): print 'shape in region'
            if 0 <= bbox[0,2] - regBB[0,2] <= 0.02:
                if debug(tag): print 'bottom z is close enough'
                vol = bboxVolume(regBB)
                if vol > bestVol:
                    bestRegShape = regShape
                    bestVol = vol
                    if debug(tag): print 'bestRegShape', regShape
    if not bestRegShape:
        print 'Could not find supporting region for %s'%shape.name()
        shape.draw('W', 'magenta')
        for regShape in shWorld.regionShapes.values():
            print regShape.draw('W', 'cyan')
        print 'gonna fail!'
        raise Exception
    return bestRegShape

# Potential contacts
fingerTipThick = 0.02
fingerTipWidth = 0.02
fingerLength = 0.045                    # it's supposed to be 0.06

def handContactFrames(shape, center, vertical):
    tag = 'handContactFrames'
    planes = shape.planes()
    verts = shape.vertices()
    faceFrames = thingFaceFrames(planes, shape.origin())
    pushCenter = center.copy().reshape(4)
    if vertical:
        minPushZ = shape.bbox()[1,2] - fingerLength
        # TODO : max(minPushZ...)?
        pushCenter[2] = shape.bbox()[0,2] + pushHeight(vertical)
    else:
        pushCenter[2] = shape.bbox()[0,2] + pushHeight(vertical)
    if debug(tag):
        print 'pushCenter', pushCenter
    contactFrames = []
    for f, face in enumerate(shape.faces()):
        # face is array of indices for verts in face
        if abs(planes[f,2]) > 0.01: continue # not a vertical face
        # frame is centered on face
        frame = shape.origin().compose(faceFrames[f])
        frameInv = frame.inverse()
        if debug(tag):
            print 'consider face', f, 'face frame:\n', frame.matrix
        if  abs(frame.matrix[2,1]) > 0.01:
            print frame.matrix
            trAlways('The y axis of face frame should be parallel to support')
        c = np.dot(frameInv.matrix, pushCenter.reshape((4,1)))
        c[2] = 0.0                      # project to face plane

        if 'pushSimSkew' in glob.debugOn:
            c[1] -= 0.07
        
        faceVerts = np.dot(frameInv.matrix, verts)
        faceBB = vertsBBox(faceVerts, face)
        faceBB[0,2] = faceBB[1,2] = 0.0
        if debug(tag):
            print 'center of face', c.tolist(), '\nfaceBB', faceBB.tolist()
        if not bboxContains(faceBB, c.reshape(4)):
            if debug(tag): print 'face does not contain center projection'
            continue
        # fingers can contact the region while spanning the center
        # TODO: use this to open the fingers as wide as possible
        wx = float(c[0] - faceBB[0,0]), float(faceBB[1,0] - c[0])
        wy = float(c[1] - faceBB[0,1]), float(faceBB[1,1] - c[1])
        if min(wx) >= 0.5 * fingerTipThick and \
           min(wy) >= 0.5 * fingerTipThick:
            cf = frame.compose(hu.Pose(c[0], c[1], 0., 0.))
            # wy is the y range (side to side)
            width = min(wy) - 0.5 * fingerTipThick

            if 'pushSimSkew' in glob.debugOn:
                width=0.

            if debug(tag):
                print faceBB
                print 'width', width, 'valid contact frame\n', cf.matrix
                # raw_input('Target')
            contactFrames.append((vertical, cf, width))
        else:
            if debug(tag): print 'face is too small'
    return contactFrames

# TODO: This should be a property of robot -- and we should extend to held objects
def pushHeight(vertical):
    if vertical:

#############
# HACK SO THAT WE CAN PUSH TALL OBJECTS WITHOUT THE ARM COLLIDING!!!        
#############

        return 0.1                 # tool tip above the table
    else:                           
        return 0.06                 # wrist needs to clear the support

# We're treating the push as a "virtual grasp".
# This is how a graspFrame is obtained
#     objFrame = objPlace.objFrame()
#     graspDesc = objGrasp.graspDesc[objGrasp.grasp.mode()]
#     faceFrame = graspDesc.frame.compose(objGrasp.poseD.mode())
#     centerFrame = faceFrame.compose(hu.Pose(0,0,graspDesc.dz,0))
#     graspFrame = objFrame.compose(centerFrame)

horizGM = np.array([(-1.,0.,0.,0.),
                   (0.,0.,1.,0.),
                   (0.,1.,0.,0.),
                   (0.,0.,0.,1.)], dtype=np.float64)
vertGM = np.array([(0.,-1.,0.,0.),
                    (0.,0.,-1.,0.),
                    (1.,0.,0.,0.),
                    (0.,0.,0.,1.)], dtype=np.float64)

def graspBForContactFrame(pbs, prob, contactFrame, zOffset, placeB, hand, vertical):
    tag = 'graspBForContactFrame'
    # TODO: what should these values be?
    graspVar = 4*(0.01**2,)
    graspDelta = 4*(0.00,)
    obj = placeB.obj
    objFrame = placeB.objFrame()
    if debug(tag): print 'objFrame\n', objFrame.matrix
    (tr, 'pushGen', 'Using pushBuffer', glob.pushBuffer)
    # Displacement of finger tip from contact face (along Z of contact frame)
    zOff = zOffset + (-fingerTipWidth if vertical else 0.) # - glob.pushBuffer
    displacedContactFrame = contactFrame.compose(hu.Pose(0.,0.,zOff,0.))
    if debug(tag):
        print 'displacedContactFrame\n', displacedContactFrame.matrix
        drawFrame(displacedContactFrame)
        placeB.shape(pbs.getWorld()).draw('W')
        raw_input('displacedContactFrame')
    graspB = None
    # consider flips of the hand (mapping one finger to the other)
    for angle in (0, np.pi):
        displacedContactFrame = displacedContactFrame.compose(hu.Pose(0.,0.,0.,angle))
        if debug(tag):
            pbs.draw(prob); drawFrame(displacedContactFrame)
            raw_input('displacedContactFrame (rotated)')
        gM = displacedContactFrame.compose(hu.Transform(vertGM if vertical else horizGM))
        if debug(tag):
            print gM.matrix
            print 'vertical =', vertical
            pbs.draw(prob); drawFrame(gM)
            raw_input('gM')
        gT = objFrame.inverse().compose(gM) # gM relative to objFrame
        # TODO: find good values for dx, dy, dz must be 0.
        graspDescList = [GDesc(obj, gT, 0.0, 0.0, 0.0)]
        graspDescFrame = objFrame.compose(graspDescList[-1].frame)
        if debug(tag): print 'graspDescFrame\n', graspDescFrame.matrix
        # -1 grasp denotes a "virtual" grasp
        gB = ObjGraspB(obj, graspDescList, -1, placeB.support.mode(),
                       PoseD(hu.Pose(0.,0.,0.,0), graspVar), delta=graspDelta)
        wrist = objectGraspFrame(pbs, gB, placeB, hand)
        if any(pbs.getWorld().robot.potentialBasePosesGen(wrist, hand, complain=False)):
            if debug(tag): print 'valid graspB'
            graspB = gB
            break
    if debug(tag):
        print graspB
        raw_input('graspB')
    assert graspB
    return graspB

def sortPushContacts(contacts, targetPose, curPose):
    bad = []
    good = []
    if curPose:
        offset = targetPose.inverse().compose(curPose).pose()
        offsetPt = offset.matrix[:,3].reshape((4,1)).copy()
        offsetPt[3,0] = 0.0                   # displacement
        for (vertical, contact, width) in contacts:
            cinv = contact.inverse() 
            ntr = np.dot(cinv.matrix, offsetPt)
            ntrz = ntr[2,0]
            if debug('pushGenDetail'):
                print 'push contact, vertical=', vertical, '\n', contact.matrix
                print 'offset', offset
                print 'offset z in contact frame\n', ntr
                raw_input('Next?')
            if ntrz >= 0.:
                # bad ones require "pulling"
                bad.append((0, None, vertical, contact, width))
            elif -ntrz >= minPushLength:
                # distance negated...
                score = 5 * width - ntrz # prefer wide faces and longer pushes
                good.append((score, -ntrz, vertical, contact, width))
                # if abs(ntrz) > 0.1:
                #     score = 5 * width - 0.5*ntrz # prefer wide faces and longer pushes
                #     good.append((score, -0.5*ntrz, vertical, contact, width))
    else:                               # no pose, just use width
        for (vertical, contact, width) in contacts:
            if debug('pushGenDetail'):
                print 'push contact, vertical=', vertical, '\n', contact.matrix
                raw_input('Next?')
            score = width
            for dist in [0.05, 0.1, 0.25]:
                # TODO: Pick distances to meaningful locations!!
                # dist = random.uniform(0.05, 0.25)
                good.append((score, dist, vertical, contact, width))
    good.sort(reverse=True)             # z distance first
    if debug('pushGen'):
        print 'push contacts sorted by push distance'
        for x in good:
            print x
    # remove score before returning
    return [x[1:] for x in good]

def gripSet(conf, hand, width=0.08):
    return conf.set(conf.robot.gripperChainNames[hand], [min(width, 0.08)])

# =================================================================
# PushInRegionGen
# =================================================================

class PushInRegionGen(Function):
    # Return objPose, poseFace.
    def fun(self, args, goalConds, bState):
        for ans in pushInRegionGenGen(args, goalConds, bState, away = False):
            assert isinstance(ans, tuple)
            yield ans.poseInTuple()

def pushInRegionGenGen(args, goalConds, bState, away = False):
    (obj, region, var, delta, prob) = args
    tag = 'pushInGen'
    pbs = bState.pbs.copy()
    world = pbs.getWorld()
    # Get the regions
    if not isinstance(region, (list, tuple, frozenset)):
        regions = frozenset([region])
    elif len(region) == 0:
        raise Exception, 'Need a region to push into'
    else:
        regions = frozenset(region)
    shWorld = pbs.getShadowWorld(prob)
    regShapes = [shWorld.regionShapes[region] for region in regions]
    tr(tag, 'Target region in purple',
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
        shape = pbs.getObjectShapeAtOrigin(obj).\
                applyLoc(attachedShape.origin())
        support = supportFaceIndex(shape)
    elif obj == pbs.held['right'].mode():
        attachedShape = pbs.getRobot().attachedObj(pbs.getShadowWorld(prob),
                                                   'right')
        shape = pbs.getObjectShapeAtOrigin(obj).\
                applyLoc(attachedShape.origin())
        support = supportFaceIndex(shape)
    else:
        raise Exception('Cannot determine support')

    # Check if object pose is specified in goalConds
    poseBels = getGoalPoseBels(goalConds, world.getFaceFrames)
    if obj in poseBels:
        pB = poseBels[obj]
        shw = shadowWidths(pB.poseD.var, pB.delta, prob)
        shwMin = shadowWidths(graspV, graspDelta, prob)
        if any(w > mw for (w, mw) in zip(shw, shwMin)):
            args = (obj, pose, support, var, delta, None, prob)
            gen = pushGenGen(args, goalConds, bState)
            for ans in gen:
                regions = [x.name() for x in regShapes]
                tr(tag, str(ans), 'regions=%s'%regions,
                   draw=[(pbs, prob, 'W')] + [(rs, 'W', 'purple') for rs in regShapes],
                   snap=['W'])
                yield ans
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

    gen = pushInGenTop((obj, regShapes, placeB, prob),
                          goalConds, pbs, away = away)
    for ans in gen:
        yield ans

pushVarIncreaseFactor = 3 # was 2

def pushInGenAway(args, goalConds, pbs):
    (obj, delta, prob) = args
    placeB = pbs.getPlaceB(obj, default=False)
    assert placeB, 'Need to know where object is to get support region'
    shape = placeB.shape(pbs.getWorld())
    regShape = findSupportRegionInPbs(pbs, prob, shape.xyPrim())
    tr('pushInGenAway', zip(('obj', 'delta', 'prob'), args),
       draw=[(pbs, prob, 'W')], snap=['W'])
    targetPushVar = tuple([pushVarIncreaseFactor * x \
                            for x in pbs.domainProbs.obsVarTuple])
    for ans in pushInRegionGenGen((obj, regShape.name(),
                                   targetPushVar, delta, prob),
                                   # preserve goalConds to get reachObsts
                                   goalConds, pbs, away=True):
        yield ans

def pushOut(pbs, prob, obst, delta, goalConds):
    tr('pushOut', 'obst=%s'%obst)
    domainPlaceVar = tuple([pushVarIncreaseFactor * x \
                            for x in pbs.domainProbs.obsVarTuple])
    if not isinstance(obst, str):
        obst = obst.name()
    for ans in pushInGenAway((obst, delta, prob), goalConds, pbs):
        ans = ans.copy()
        ans.var = domainPlaceVar; ans.delta = delta
        yield ans

pushInGenMaxPoses  = 50
pushInGenMaxPosesH = 10

def pushInGenTop(args, goalConds, pbs, away = False):
    (obj, regShapes, placeB, prob) = args
    tag = 'pushInGen'
    regions = [x.name() for x in regShapes]
    tr(tag, '(%s,%s) h=%s'%(obj,regions, glob.inHeuristic))
    tr(tag, 
       zip(('obj', 'regShapes', 'placeB', 'prob'), args))
    if obj == 'none' or not regShapes:
        # Nothing to do
        tr(tag, '=> object is none or no regions, failing')
        return
    if goalConds and getConf(goalConds, None) and not away:
        # if conf is specified, just fail
        tr(tag, '=> conf is specified, failing')
        return

    conf = None
    confAppr = None
    # Obstacles for all Reachable fluents
    reachObsts = getReachObsts(goalConds, pbs)
    if reachObsts is None:
        tr(tag, '=> No path for reachObst, failing')
        return
    tr(tag, '%d reachObsts - in brown'%len(reachObsts),
       draw=[(pbs, prob, 'W')] + [(obst, 'W', 'brown') for _,obst in reachObsts],
       snap=['W'])
    shWorld = pbs.getShadowWorld(prob)
    nPoses = pushInGenMaxPosesH if glob.inHeuristic else pushInGenMaxPoses

    leftGen = pushInGenAux(pbs, prob, goalConds, placeB, regShapes, reachObsts, 'left')
    rightGen = pushInGenAux(pbs, prob, goalConds, placeB, regShapes, reachObsts, 'right')
    # Figure out whether one hand or the other is required;  if not, do round robin
    mainGen = chooseHandGen(pbs, goalConds, obj, None, leftGen, rightGen)

    # Picks among possible target poses and then try to push it in region
    for ans in mainGen:
        yield ans

# placeB is current place for object
# regShapes is a list of (one) target region
# reachObsts are regions where placements are forbidden
def pushInGenAux(pbs, prob, goalConds, placeB, regShapes, reachObsts, hand):
    shWorld = pbs.getShadowWorld(prob)
    for pB in awayTargetPB(pbs, prob, placeB, regShapes):
        for ans in pushGenTop((placeB.obj, pB, hand, prob),
                              goalConds, pbs,
                              # TODO: should this be true or false?
                              partialPaths=True, reachObsts=reachObsts):

            tr('pushInGen', ('->', str(ans)),
               draw=[(pbs, prob, 'W'),
                     (ans.prePB.shape(shWorld), 'W', 'blue'),
                     (ans.postPB.shape(shWorld), 'W', 'pink'),
                     (ans.preConf, 'W', 'blue', shWorld.attached),
                     (ans.pushConf, 'W', 'pink', shWorld.attached)] \
               + [(rs, 'W', 'purple') for rs in regShapes],
               snap=['W'])

            yield ans
        # pdb.set_trace()

# Find target placements in region.  Object starts out in placeB.
def regionPB(pbs, prob, placeB, regShapes):
    shWorld = pbs.getShadowWorld(prob)
    shape = pbs.getObjectShapeAtOrigin(placeB.obj)
    regShape = regShapes[0]
    mode = placeB.poseD.mode().pose()
    bbox = bboxInterior(shape.bbox(), regShape.bbox())
    n = len(bboxGridCoords(bbox, res=0.2))
    targets = sortedTargets(pbs, prob, placeB, bboxRandomCoords(bbox, n=n))
    for point in targets:
        newMode = hu.Pose(point[0], point[1], mode.z, mode.theta)
        newPB = placeB.modifyPoseD(newMode, var=4*(0.02**2,))
        newPB.delta = delta=4*(0.001,)
        newShape = newPB.makeShadow(pbs, prob) 
        if inside(newShape, regShape, strict=True):
            if debug('pushInGen'):
                pbs.draw(prob, 'W'); newShape.draw('W'); regShape.draw('W', 'purple')
                raw_input('target pose in region for push')
            yield newPB

def sortedTargets(pbs, prob, placeB, targetPoints):
    shape = placeB.shape(pbs.getWorld())
    prim = choosePrim(shape)
    xyPrim = shape.xyPrim()
    center =  np.average(xyPrim.vertices(), axis=1)
    potentialContacts = []
    for vertical in (True, False):
        potentialContacts.extend(handContactFrames(prim, center, vertical))
    mode = placeB.poseD.mode().pose()
    targets = []
    for point in targetPoints:
        targets.append(hu.Pose(point[0], point[1], mode.z, mode.theta))
    return sortedTargetsAux(potentialContacts, targets, placeB.poseD.mode())

def sortedTargetsAux(contacts, targetPoses, curPose):
    good = []
    for targetPose in targetPoses:
        offset = targetPose.inverse().compose(curPose).pose()
        offsetPt = offset.matrix[:,3].reshape((4,1)).copy()
        offsetPt[3,0] = 0.0                   # displacement
        for (vertical, contact, width) in contacts:
            cinv = contact.inverse() 
            ntr = np.dot(cinv.matrix, offsetPt)
            ntrz = ntr[2,0]
            if ntrz < 0.:
                score = math.sqrt(ntr[1,0]**2 + ntr[2,0]**2)
                good.append((score, np.array(targetPose.xyztTuple()[:3])))
    good.sort(key=itemgetter(0))
    # remove score before returning
    return removeDuplicates([x[1] for x in good])

def removeDuplicates(pointList):
    points = []
    for i, point1 in enumerate(pointList):
        match = False
        for j, point2 in enumerate(pointList[i+1:]):
            if all(point1==point2): match=True; break
        if not match:
            points.append(point1)
    return points

def awayTargetPB(pbs, prob, placeB, regShapes):
    shape = placeB.shape(pbs.getWorld())
    regShape = regShapes[0]
    prim = choosePrim(shape)
    xyPrim = shape.xyPrim()
    center =  np.average(xyPrim.vertices(), axis=1)
    potentialContacts = []
    for vertical in (True, False):
        potentialContacts.extend(handContactFrames(prim, center, vertical))
    targets = []
    for (vertical, contactFrame, width) in potentialContacts:
        # This is z axis of face, push will be in that direction 
        direction = contactFrame.matrix[:3,2].reshape(3).copy()
        direction[2] = 0.0            # we want z component exactly 0.
        for newPB in maxDistInRegionPB(pbs, prob, placeB, direction, regShape):
            if newPB not in targets:
                targets.append(newPB)
                if debug('pushGen'):
                    print 'direction', direction, 'newPB', newPB
                    newShape = newPB.makeShadow(pbs, prob)
                    pbs.draw(prob, 'W'); drawFrame(contactFrame); newShape.draw('W', 'magenta')
                    raw_input('Go?')
    return targets

def maxDistInRegionPB(pbs, prob, placeB, direction, regShape, mind = 0.0, maxd = 1.0):
    dist = 0.5*(mind+maxd)
    postPoseOffset = hu.Pose(*(dist*direction).tolist()+[0.0])
    postPose = postPoseOffset.compose(placeB.poseD.mode())
    newPB = placeB.modifyPoseD(postPose.pose(), var=4*(0.01**2,))
    newPB.delta = delta=4*(0.001,)
    if inside(newPB.makeShadow(pbs, prob), regShape, strict=True):
        if maxd-mind < 0.01:
            postPoseOffset = hu.Pose(*((dist*0.5)*direction).tolist()+[0.0])
            postPose = postPoseOffset.compose(placeB.poseD.mode())
            newPB2 = placeB.modifyPoseD(postPose.pose(), var=4*(0.01**2,))
            newPB2.delta = delta=4*(0.001,)
            return [newPB2, newPB]
        else:
            return maxDistInRegionPB(pbs, prob, placeB, direction, regShape, dist, maxd)
    else:
        return maxDistInRegionPB(pbs, prob, placeB, direction, regShape, mind, dist)
