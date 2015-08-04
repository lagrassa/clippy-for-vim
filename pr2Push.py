from fbch import Function
from geom import bboxInside, bboxVolume, vertsBBox, bboxContains, bboxVolume
import numpy as np
from traceFile import debug, debugMsg, tr, trAlways
from shapes import thingFaceFrames, drawFrame
import hu
from planUtil import ObjGraspB, PoseD, Response
from pr2Util import GDesc
from pr2GenAux import *
from pr2PlanBel import getConf
from pr2Fluents import pushPath
import mathematica
reload(mathematica)
import windowManager3D as wm
from subprocess import call
import time
import pdb

# Pick poses and confs for pushing an object

# For now, consider only a small set of stable push directions.
# Pretend that centroid of xyPrim is the center of friction.  The push
# path needs to stay inside the regions and the robot needs to be able
# to follow it without colliding with permanent obstacles.

# TODO: Pick a reasonable value

# How many paths to generate for a particular hand contact with the
# object, these will differ by the base placement.
maxPushPaths = 50

pushGenCacheStats = [0, 0]
pushGenCache = {}

class PushGen(Function):
    def fun(self, args, goalConds, bState):
        for ans in pushGenGen(args, goalConds, bState):
            tr('pushGen', str(ans))
            yield ans

def pushGenGen(args, goalConds, bState):
    (obj, pose, posevar, posedelta, confdelta, prob) = args
    tag = 'pushGen'
    base = sameBase(goalConds)
    tr(tag, 'obj=%s, pose=%s, base=%s'%(obj, pose, base))
    if goalConds:
        if getConf(goalConds, None):
            tr(tag, '=> conf is already specified, failing')
            return
    pbs = bState.pbs.copy()
    world = pbs.getWorld()
    support = pbs.getPlaceB(obj).support.mode()
    placeB = ObjPlaceB(obj, world.getFaceFrames(obj), support,
                       PoseD(pose, posevar), delta=posedelta)

    if pbs.getPlaceB(placeB.obj).poseD.mode().distance(placeB.poseD.mode()) < 0.01:
        tr(tag, 'Target pose is too close to current pose, failing')
        return

    # Figure out whether one hand or the other is required;  if not, do round robin
    leftGen = pushGenTop((obj, placeB, 'left', base, prob),
                         goalConds, pbs)
    rightGen = pushGenTop((obj, placeB, 'right', base, prob),
                          goalConds, pbs)

    for ans in chooseHandGen(pbs, goalConds, obj, None, leftGen, rightGen):
        
        yield ans
    tr(tag, '=> Exhausted')

def pushGenTop(args, goalConds, pbs):
    (obj, placeB, hand, base, prob) = args
    startTime = time.clock()
    tag = 'pushGen'
    tr(tag, '(%s,%s) h=%s'%(obj,hand, glob.inHeuristic))
    tr(tag, 
       zip(('obj', 'placeB', 'hand', 'prob'), args),
       ('goalConds', goalConds),
       ('moveObjBs', pbs.moveObjBs),
       ('fixObjBs', pbs.fixObjBs),
       ('held', (pbs.held['left'].mode(),
                 pbs.held['right'].mode(),
                 pbs.graspB['left'],
                 pbs.graspB['right'])))
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
    # Set up pbs
    newBS = pbs.copy()
    # Just placements specified in goal
    newBS = newBS.updateFromGoalPoses(goalConds)
    tr(tag, 'Goal conditions', draw=[(newBS, prob, 'W')], snap=['W'])
    if newBS.held[hand].mode() != 'none':
        tr(tag, '=> Hand=%s is holding in pbs, failing'%hand)
        return
    if obj in [h.mode() for h in newBS.held.values()]:
        tr(tag, '=> obj is in the hand, failing')
        return
    pushGenCacheStats[0] += 1
    key = (newBS, placeB, hand, base, prob)
    val = pushGenCache.get(key, None)
    if val != None:
        if debug(tag): print tag, 'cached'
        pushGenCacheStats[1] += 1
        memo = val.copy()
    else:
        gen = pushGenAux(newBS, placeB, hand, base, prob)
        memo = Memoizer(tag, gen)
        pushGenCache[key] = memo
    for ans in memo:
        tr(tag, str(ans) +' (t=%s)'%(time.clock()-startTime))
        yield ans
    tr(tag, '=> pushGenTop exhausted')

def pickPrim(shape):
    return sorted(shape.toPrims(),
                  key = lambda p: bboxVolume(p.bbox()), reverse=True)[0]

def pushGenAux(pbs, placeB, hand, base, prob):
    tag = 'pushGen'
    shape = placeB.shape(pbs.getWorld())
    prim = pickPrim(shape)
    xyPrim = shape.xyPrim()
    # Location of center at placeB
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
            # construct a graspB corresponding to the push hand pose,
            # determined by the contact frame
            potentialContacts.append((vertical, contactFrame, width))
    # Sort contacts by nearness to current pose of object
    curPose = pbs.getPlaceB(placeB.obj).poseD.mode()
    # sort contacts and compute a distance when applicable, entries
    # are: (distance, vertical, contactFrame)
    sortedContacts = sortPushContacts(potentialContacts, placeB.poseD.mode(), curPose)
    # Now we have frames and confs for contact with object, we have to
    # generate potential answers following the face normal in the
    # given direction.  We'll generate answers in order of distance
    # from placeB.
    for (dist, vertical, contactFrame, width) in sortedContacts:
        graspB = graspBForContactFrame(pbs, contactFrame,
                                       0.0,  placeB, hand, vertical)
        gf = placeB.objFrame().compose(graspB.graspDesc[-1].frame)
        # This is negative z axis
        direction = -contactFrame.matrix[:3,2].reshape(3)
        direction[2] = 0.0
        if debug(tag):
            pbs.draw(prob, 'W')
            shape.draw('W', 'blue'); prim.draw('W', 'green')
            print 'vertical', vertical, 'dist', dist
            drawFrame(contactFrame)
            raw_input('graspDesc frame')
        count = 0
        doneCount = 0
        rm = pbs.getRoadMap()
        pushPaths = []                  # for different base positions
        for ans in potentialGraspConfGen(pbs, placeB, graspB,
                                         None, hand, base, prob):
            if not ans:
                tr(tag+'Path', 'potential grasp conf is empy')
                continue
            (c, ca, viol) = ans
            c = gripSet(c, hand, width)
            # Make sure that we don't collide with the object to be pushed
            newBS = pbs.copy().updatePermObjPose(placeB)
            viol = rm.confViolations(c, newBS, prob)
            if not viol:
                print 'Conf collides in pushPath'
                continue
            count += 1
            pathAndViols, reason = pushPath(newBS, prob, graspB, placeB, c,
                                            direction, dist, xyPrim, supportRegion, hand)
            if reason == 'done':
                doneCount +=1 
                pushPaths.append((pathAndViols, reason))
            tr(tag+'Path', 'pushPath reason = %s, path len = %d'%(reason, len(pathAndViols)))
            if doneCount >= 2: break
            if count > maxPushPaths: break
        sorted = sortedPushPaths(pushPaths)
        for i in range(min(len(sorted), 2)):
            pp = sorted[i]              # path is reversed (post...pre)
            ptarget = placeB.poseD.mode()
            for i in range(len(pp)):
                cpost, vpost, ppost = pp[i]
                if ppost: break
            for j in range(1, len(pp)):
                _, _, ppre = pp[-j]
                if ppre: break
            if not ppre: continue
            cpre, _, _ = pp[-1]
            if debug(tag):
                robot = cpre.robot
                print 'pre pose\n', ppre.matrix
                print 'pre conf tool'
                print cpre.cartConf()[robot.armChainNames[hand]].compose(robot.toolOffsetX[hand]).matrix
                print 'post conf tool'
                print cpost.cartConf()[robot.armChainNames[hand]].compose(robot.toolOffsetX[hand]).matrix
                raw_input('Yield this?')
            tr(tag, 'pre conf (blue), post conf (pink)',
               draw=[(pbs, prob, 'W'),
                     (cpre, 'W', 'blue'), (cpost, 'W', 'pink')], snap=['W'])
            yield (hand, ppre.pose().xyztTuple(), cpre, cpost, cpre)
    tr(tag, '=> pushGenAux exhausted')
    return


def sortedPushPaths(pushPaths):
    scored = []
    for (pathAndViols, reason) in pushPaths:
        if reason == 'done':
            scored.append((0., pathAndViols))
        elif pathAndViols:
            vmax = max(v.weight() for (c,v,p) in pathAndViols)
            scored.append((max(1, vmax), pathAndViols))
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
            width = min(wx) - 0.5 * fingerTipThick
            if debug(tag):
                print 'width', width, 'valid contact frame\n', cf.matrix
                raw_input('Target')
            contactFrames.append((cf, width))
        else:
            if debug(tag): print 'face is too small'
    return contactFrames

# TODO: This should be a property of robot -- and we should extend to held objects
def pushHeight(vertical):
    if vertical:
        return 0.02                 # tool tip above the table
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
pushBuffer = 0.05
def graspBForContactFrame(pbs, contactFrame, zOffset, placeB, hand, vertical):
    tag = 'graspBForContactFrame'
    # TODO: what should these values be?
    graspVar = 4*(0.01**2,)
    graspDelta = 4*(0.00,)
    obj = placeB.obj

    # TODO: straighten this out...
    objFrame = placeB.objFrame()
    # objFrame = placeB.poseD.mode()

    if debug(tag): print 'objFrame\n', objFrame.matrix

    (tr, 'pushGen', 'Using pushBuffer', pushBuffer)
    # Displacement of finger tip from contact face (along Z of contact frame)
    zOff = zOffset + (-fingerTipWidth if vertical else 0.) - pushBuffer
    displacedContactFrame = contactFrame.compose(hu.Pose(0.,0.,zOff,0.))
    if debug(tag):
        print 'displacedContactFrame\n', displacedContactFrame.matrix
        drawFrame(displacedContactFrame)
        raw_input('displacedContactFrame')
    graspB = None
    # consider flips of the hand (mapping one finger to the other)
    for angle in (0, np.pi):
        displacedContactFrame = displacedContactFrame.compose(hu.Pose(0.,0.,0.,angle))
        if debug(tag):
            pbs.draw(0.9); drawFrame(displacedContactFrame)
            raw_input('displacedContactFrame (rotated)')
        gM = displacedContactFrame.compose(hu.Transform(vertGM if vertical else horizGM))
        if debug(tag):
            print gM.matrix
            print 'vertical =', vertical
            pbs.draw(0.9); drawFrame(gM)
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
            bad.append((None, vertical, contact, width))
        else:
            # distance negated...
            good.append((-ntrz, vertical, contact, width))
    good.sort(reverse=True)             # largest z distance first
    if debug('pushGen'):
        print 'push contacts sorted by push distance'
        for x in good:
            print x
    return good                         # bad ones require "pulling"

def gripSet(conf, hand, width=0.08):
    return conf.set(conf.robot.gripperChainNames[hand], [width])
