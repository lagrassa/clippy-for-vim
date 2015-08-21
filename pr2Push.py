from fbch import Function
from geom import bboxInside, bboxVolume, vertsBBox, bboxContains, bboxVolume
import numpy as np
from traceFile import debug, debugMsg, tr, trAlways
from shapes import thingFaceFrames, drawFrame
import hu
from planUtil import ObjGraspB, PoseD, Response
from pr2Util import GDesc, trArgs, otherHand
from pr2GenAux import *
from pr2PlanBel import getConf
from pr2Fluents import pushPath
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
maxPushPaths = 50
maxDone = 1

pushGenCacheStats = [0, 0]
pushGenCache = {}

minPushLength = 0.01

useDirectPush = False

class PushGen(Function):
    def fun(self, args, goalConds, bState):
        for ans in pushGenGen(args, goalConds, bState):
            tr('pushGen', '->', 'final', str(ans))
            if debug('pushGen'):
                # Verify that it's feasible
                (obj, pose, support, poseVar, poseDelta, confdelta, prob) = args
                (hand, prePose, preConf, pushConf, postConf) = ans
                path, viol = canPush(bState.pbs, obj, hand, support, prePose,
                                     pose, preConf, pushConf, postConf, poseVar,
                                    poseVar, poseDelta, prob,
                                    Violations(), prim=False)
                assert viol != None
            yield ans
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

def pushGenTop(args, goalConds, pbs):
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
    if obj in [h.mode() for h in newBS.held.values()]:
        tr(tag, 'obj is in other hand')
        newBS.updateHeld('none', None, None, otherHand(hand), None)
        curPB = None
    else:
        curPB = newBS.getPlaceB(obj, default=False)
        assert curPB, 'Object needs to be somewhere'
        tr(tag, 'obj is placed')
    # Check the cache, otherwise call Aux
    pushGenCacheStats[0] += 1
    key = (newBS, placeB, hand, base, prob)
    val = pushGenCache.get(key, None)
    if val != None:
        pushGenCacheStats[1] += 1
        # re-start the generator
        memo = val.copy()
        if debug(tag):
            print tag, 'cached, with len(values)=', len(memo.values)
    else:
        gen = pushGenAux(newBS, placeB, hand, base, curPB, prob)
        memo = Memoizer(tag, gen)
        pushGenCache[key] = memo
        if debug(tag):
            print tag, 'creating new pushGenAux generator'
    rm = newBS.getRoadMap()
    for ans in memo:
        (hand, prePose, preConf, pushConf, postConf) = ans
        # Double check that preConf is safe - this should not be necessary...
        testBS = newBS.copy()
        testBS.updatePermObjPose(placeB.modifyPoseD(prePose)) # obj at prePose
        testBS.draw(prob, 'W'); preConf.draw('W')
        viol = rm.confViolations(preConf, testBS, prob)
        assert viol                     # no permanent collisions
        # End double check
        yield ans
    tr(tag, '=> pushGenTop exhausted for', hand)

def choosePrim(shape):
    return sorted(shape.toPrims(),
                  key = lambda p: bboxVolume(p.bbox()), reverse=True)[0]

def pushGenAux(pbs, placeB, hand, base, curPB, prob):
    tag = 'pushGen'
    # The shape at target, without any shadow
    shape = placeB.shape(pbs.getWorld())
    if curPB:
        curPose = curPB.poseD.mode()
        targetPose = placeB.poseD.mode()
        # If the current position is almost at the target, return
        if curPose.distance(targetPose)  < minPushLength:
            tr(tag, '=> Target pose is too close to current pose, failing',
               curPose, targetose, curPose.distance(targetPose))
    else:
        curPOse = None
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
        for (contactFrame, width) in handContactFrames(prim, center, vertical):
            potentialContacts.append((vertical, contactFrame, width))
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
        for ans in potentialGraspConfGen(pbs, placeB, graspB,
                                         None, hand, base, prob):
            if not ans:
                tr(tag+'Path', 'potential grasp conf is empy')
                continue
            (c, ca, viol) = ans         # conf, approach, violations
            pushConf = gripSet(c, hand, 2*width) # open fingers
            if debug(tag+'_kin'):
                pushConf.draw('W', 'orange')
                # raw_input('Candidate conf')
                wm.getWindow('W').update()
                
            count += 1
            # Try going directly to goal and along the direction of face
            # The direct route only works for vertical pushing...
            for direct in (True, False) if (useDirectPush and vertical and curPose) else (False,):
                pathAndViols, reason = pushPath(pbs, prob, graspB, placeB, pushConf,
                                                curPose.pose() if direct else prePose.pose(),
                                                xyPrim, supportRegion, hand)
                if reason == 'done':
                    doneCount +=1 
                    pushPaths.append((pathAndViols, reason))
                tr(tag+'Path', 'pushPath reason = %s, path len = %d'%(reason, len(pathAndViols)))
            if doneCount >= maxDone: break
            if count > maxPushPaths: break
        if count == 0 and not glob.inHeuristic:
            print tag, 'No potentialGraspConfGen results'
        # Sort the push paths by violations
        sorted = sortedPushPaths(pushPaths, curPose)
        for i in range(min(len(sorted), maxDone)):
            pp = sorted[i]              # path is reversed (post...pre)
            ppre, cpre, ppost, cpost = getPrePost(pp)
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
            yield (hand, ppre.pose().xyztTuple(), cpre, cpost, cpre)
    tr(tag, '=> pushGenAux exhausted')
    return

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
                pdb.set_trace()
            contactFrames.append((cf, width))
        else:
            if debug(tag): print 'face is too small'
    return contactFrames

# TODO: This should be a property of robot -- and we should extend to held objects
def pushHeight(vertical):
    if vertical:
        return 0.04                 # tool tip above the table
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
            else:
                score = 5 * width - ntrz # prefer wide faces and longer pushes
                # distance negated...
                good.append((score, -ntrz, vertical, contact, width))
    else:                               # no pose, just use width
        for (vertical, contact, width) in contacts:
            if debug('pushGenDetail'):
                print 'push contact, vertical=', vertical, '\n', contact.matrix
                raw_input('Next?')
            score = width
            for k in range(3):
                # TODO: Pick distances to meaningful locations!!
                dist = random.uniform(0.05, 0.25)
                good.append((score, -dist, vertical, contact, width))
    good.sort(reverse=True)             # z distance first
    if debug('pushGen'):
        print 'push contacts sorted by push distance'
        for x in good:
            print x
    # remove score before returning
    return [x[1:] for x in good]

def gripSet(conf, hand, width=0.08):
    return conf.set(conf.robot.gripperChainNames[hand], [min(width, 0.08)])
