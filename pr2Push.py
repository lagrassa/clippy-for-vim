from fbch import Function
from geom import bboxInside, bboxVolume, vertsBBox, bboxContains
from pr2GenAux import inside
import numpy as np
from planGlobals import debug, debugMsg, useROS
from traceFile import tr
from shapes import thingFaceFrames, drawFrame
import hu
from planUtil import ObjGraspB, PoseD, Response
from pr2Util import GDesc
from pr2GenAux import potentialGraspConfGen, objectGraspFrame, robotGraspFrame
from pr2Robot import gripperFaceFrame
import mathematica
reload(mathematica)
import windowManager3D as wm
from subprocess import call

# Pick poses and confs for push, either source or destination.

# For now, consider only a small set of stable push directions.
# Pretend that centroid of xyPrim is the center of friction.  The push
# path needs to stay inside the regions and the robot needs to be able
# to follow it without colliding with permanent obstacles.

# Direction is +1, -1 relative to face frame, which opposes face
# normal.  That is, direction = -1 is away from the face, so this
# would return places that the object could be pushed FROM.  The +1
# direction is towards the face and returns places that the object
# could be pushed TO.
directionFrom = 1                       # along faceFrame z
directionTo = -1                        # opposite faceFrame z

maxPushPaths = 10

class PushGen(Function):
    def fun(self, args, goalConds, bState):
        for ans in pushGenGen(args, goalConds, bState):
            tr('pushGen', 1, str(ans))
            yield ans

def pushGenGen(args, goalConds, bState):
    (obj, pose, posevar, posedelta,confdelta) = args
    tag = 'pushGen'
    base = sameBase(goalConds)
    tr(tag, 0, 'obj=%s, base=%s'%(obj, base))
    if goalConds:
        if getConf(goalConds, None):
            tr(tag, 1, '=> conf is already specified, failing')
            return
    pbs = bState.pbs.copy()
    world = pbs.getWorld()
    support = pbs.getPlaceB(obj).support.mode()
    placeB = ObjPlaceB(obj, world.getFaceFrames(obj), support,
                       PoseD(pose, objV), delta=objDelta)
    # Figure out whether one hand or the other is required;  if not, do round robin
    leftGen = pushGenTop((obj, placeB, 'left', base, prob),
                         goalConds, pbs)
    rightGen = pushGenTop((obj, placeB, 'right', base, prob),
                          goalConds, pbs)

    for ans in chooseHandGen(pbs, goalConds, obj, hand, leftGen, rightGen):
        yield ans

def pushGenTop(args, goalConds, pbs):
    (obj, placeB, hand, base, prob) = args
    startTime = time.clock()
    tag = 'pushGen'
    tr(tag, 0, '(%s,%s) h=%s'%(obj,hand, glob.inHeuristic))
    tr(tag, 2, 
       zip(('obj', 'placeB', 'hand', 'prob'), args),
       ('goalConds', goalConds),
       ('moveObjBs', pbs.moveObjBs),
       ('fixObjBs', pbs.fixObjBs),
       ('held', (pbs.held['left'].mode(),
                 pbs.held['right'].mode(),
                 pbs.graspB['left'],
                 pbs.graspB['right'])))
    if obj == 'none' or not placeB:
        tr(tag, 1, '=> obj is none or no placeB, failing')
        return
    if goalConds:
        if getConf(goalConds, None):
            tr(tag, 1, '=> goal conf specified, failing')
            return
        for (h, o) in getHolding(goalConds):
            if h == hand:
                # TODO: we could push with the held object
                tr(tag, 1, '=> Hand=%s is Holding, failing'%hand)
                return
    # Set up pbs
    newBS = pbs.copy()
    # Just placements specified in goal
    newBS = newBS.updateFromGoalPoses(goalConds, updateConf=not away)
    tr(tag, 2, 'Goal conditions', draw=[(newBS, prob, 'W')], snap=['W'])
    gen = pushGenAux(newBS, placeB, hand, base, prob)
    for ans in gen:
        tr(tag, 1, str(ans) +' (t=%s)'%(time.clock()-startTime))
        yield ans

def pushGenAux(pbs, placeB, hand, base, prob):
    tag = 'pushGen'
    shape = placeB.shape(pbs.getWorld())
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
        for contactFrame in handContactFrames(xyPrim, center, vertical):
            # construct a graspB corresponding to the push hand pose,
            # determined by the contact frame
            pbs.draw(prob, 'W')
            potentialContacts.append((vertical, contactFrame))
    # Sort contacts by nearness to current pose of object
    curPose = pbs.getPlaceB(placeB.obj).poseD.mode()
    # sort contacts and compute a distance when applicable, entries
    # are: (vertical, distance, contactFrame)
    sortedContacts = sortPushContacts(potentialContacts, curPose)
    # Now we have frames and confs for contact with object, we have to
    # generate potential answers following the face normal in the
    # given direction.  We'll generate answers in order of distance
    # from placeB.
    for (dist, vertical, contactFrame) in sortedContacts:
        graspB = graspBForContactFrame(pbs, contactFrame,
                                       0.0,  placeB, hand, vertical)
        pbs.draw(prob, 'W')
        gf = placeB.objFrame().compose(graspB.graspDesc[-1].frame)
        drawFrame(gf)
        raw_input('graspDesc frame')
        count = 0
        pushPaths = []                  # for different base positions
        for ans in potentialGraspConfGen(pbs, placeB, graspB,
                                         None, hand, base, prob):
            if not ans: continue
            (c, ca, viol) = ans
            # TODO: pick hand opening better
            c = gripSet(c, hand, 0.0)
            ca = gripSet(ca, hand, 0.0)
            ans = Response(placeB, graspB, c, ca, viol, hand)
            count += 1
            pathAndViols, reason = pushPath(pbs, prob, ans, contactFrame, dist,
                                            xyPrim, supportRegion, hand)
            pushPaths.append((pathAndViols, reason))
            print 'pushPath reason =', reason, len(pathAndViols)
            if count > maxPushPaths: break
        sorted = sortedPushPaths(pushPaths)
        for i in range(min(len(sorted), 2)):
            pp = sorted[i]              # path is reversed (post...pre)
            cpost, vpost, ppost = pp[0]
            cpre, vpre, ppre = pp[-1]
            yield (hand, ppre, cpre, cpost)
    return

def sortedPushPaths(pushPaths):
    scored = []
    for (pathAndViols, reason) in pushPaths:
        if reason == 'done':
            scored.append((0., [pv for pv in pathAndViols]))
        else:
            vmin = min(v.weight() for (c,v,p) in pathAndViols)
            trim = []
            for pv in pathAndViols:
                (c,v,p) = pv
                trim.append(pv)
                if v.weight() == vmin:
                    scored.append(trim)
                    break
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
    assert bestRegShape, 'Could not find supporting region for %s'%shape.name()
    return bestRegShape

# Potential contacts
fingerTipThick = 0.02
fingerTipWidth = 0.05
fingerLength = 0.05                     # it's really 0.06

def handContactFrames(shape, center, vertical):
    tag = 'handContactFrames'
    planes = shape.planes()
    verts = shape.vertices()
    faceFrames = thingFaceFrames(planes, shape.origin())
    pushCenter = center.copy().reshape(4)
    if vertical:
        minPushZ = shape.bbox()[1,2] - fingerLength
        pushCenter[2] = max(minPushZ, shape.bbox()[0,2] + pushHeight(vertical))
    else:
        pushCenter[2] = shape.bbox()[0,2] + pushHeight(vertical)
    if debug(tag):
        print 'pushCenter', pushCenter
    contactFrames = []
    for f, face in enumerate(shape.faces()):
        # face is array of indices for verts in face
        if abs(planes[f,2]) > 0.01: continue # not a vertical face
        frame = shape.origin().compose(faceFrames[f])
        frameInv = frame.inverse()
        if debug(tag):
            print 'consider face', f, 'face frame:\n', frame.matrix
        if  abs(frame.matrix[2,1]) > 0.01:
            print frame.matrix
            raw_input('The y axis of face frame should be parallel to support')
        c = np.dot(frameInv.matrix, pushCenter.reshape((4,1)))
        c[2] = 0.0                      # project to face plane
        faceVerts = np.dot(frameInv.matrix, verts)
        faceBB = vertsBBox(faceVerts, face)
        if debug(tag):
            print 'center of face', c.tolist(), '\nfaceBB', faceBB.tolist()
        if not bboxContains(faceBB, c.reshape(4)):
            if debug(tag): print 'face does not contain center projection'
            continue
        # fingers can contact the region while spanning the center
        # TODO: use this to open the fingers as wide as possible
        if c[0] - faceBB[0,0] >= fingerTipThick and faceBB[1,0] - c[0] >= fingerTipThick \
           and c[1] - faceBB[0,1] >= fingerTipThick and faceBB[1,1] - c[1] >= fingerTipThick:
            cf = frame.compose(hu.Pose(c[0], c[1], 0., 0.))
            if debug(tag):
                print 'valid contact frame\n', cf.matrix
            contactFrames.append(cf)
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
def graspBForContactFrame(pbs, contactFrame, zOffset, placeB, hand, vertical):
    tag = 'graspBForContactFrame'
    # TODO: what should these values be?
    graspVar = 4*(0.0,)
    graspDelta = 4*(0.0,)
    obj = placeB.obj
    objFrame = placeB.objFrame()
    if debug(tag): print 'objFrame\n', objFrame.matrix
    zOff = zOffset + (-fingerTipWidth if vertical else 0.)
    displacedContactFrame = contactFrame.compose(hu.Pose(0.,0.,zOff,0.))
    if debug(tag):
        print 'displacedContactFrame\n', displacedContactFrame.matrix
        drawFrame(displacedContactFrame)
        raw_input('displacedContactFrame')
    graspB = None
    for angle in (0, np.pi):
        displacedContactFrame = displacedContactFrame.compose(hu.Pose(0.,0.,0.,angle))
        gM = displacedContactFrame.compose(hu.Transform(vertGM if vertical else horizGM))
        gT = objFrame.inverse().compose(gM)
        # TODO: find good values for dx, dy, dz
        graspDescList = [GDesc(obj, gT, 0.0, 0.0, 0.0)]
        graspDescFrame = objFrame.compose(graspDescList[-1].frame)
        if debug(tag): print 'graspDescFrame\n', graspDescFrame.matrix
        # -1 grasp denotes a "virtual" grasp
        gB = ObjGraspB(obj, graspDescList, -1,
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

pushStepSize = 0.02

def pushPath(pbs, prob, resp, contactFrame, dist, shape, regShape, hand):
    tag = 'pushPath'
    # This is negative z axis
    direction = -contactFrame.matrix[:3,2].reshape(3)
    direction[2] = 0.0
    newBS = pbs.copy()
    newBS = newBS.updateHeldBel(resp.gB, hand)
    newBS = newBS.excludeObjs([resp.pB.obj])
    shWorld = newBS.getShadowWorld(prob)
    attached = shWorld.attached
    if debug(tag): newBS.draw(prob, 'W'); raw_input('Go?')
    rm = pbs.getRoadMap()
    conf = resp.c
    pathViols = []
    reason = 'done'
    dist = dist or 1.0
    for step in np.arange(0., dist + 0.001, pushStepSize):
        offsetPose = hu.Pose((step*direction).tolist()+[0.0])
        nshape = shape.applyTrans(offsetPose)
        if not inside(nshape, regShape):
            reason = 'outside'
            break
        nconf = displaceHand(conf, hand, offsetPose)
        if not nconf:
            reason = 'invkin'
            break
        viol = rm.confViolations(nconf, newBS, prob)
        if debug('pushPath'):
            print viol
            wm.getWindow('W').startCapture()
            newBS.draw(prob, 'W')
            nconf.draw('W', 'cyan', attached)
            mathematica.mathFile(wm.getWindow('W').stopCapture(),
                                 view = "ViewPoint -> {2, 0, 2}",
                                 filenameOut='./pushPath.m')
            raw_input('Next?')
        if viol is None:
            reason = 'collide'
            break
        pathViols.append((nconf, viol, resp.pB.poseD.mode().compose(offsetPose)))
    if debug('pushPath'):
        raw_input('Path:'+reason)
    return pathViols, reason
        
def displaceHand(conf, hand, offsetPose, nearTo=None):
    cart = conf.cartConf()
    handFrameName = conf.robot.armChainNames[hand]
    trans = cart[handFrameName]
    nTrans = offsetPose.compose(trans)
    nCart = cart.set(handFrameName, nTrans)
    nConf = conf.robot.inverseKin(nCart, conf=(nearTo or conf)) # use conf to resolve
    if all(nConf.values()):
        return nConf

def gripSet(conf, hand, width=0.08):
    return conf.set(conf.robot.gripperChainNames[hand], [width])

def sortPushContacts(contacts, pose):
    bad = []
    good = []
    for (vertical, contact) in contacts:
        ntr = contact.inverse().compose(pose)
        ntrz = ntr.matrix[2,3]
        if ntrz > 0.:
            bad.append((None, vertical, contact))
        else:
            good.append((-ntrz, vertical, contact))
    good.sort()                         # smallest z distance first
    return good + bad

# returns path, violations
def canPush(pbs, obj, hand, prePose, pose,
            preConf, postConf, prePoseVar, poseVar,
            poseDelta, prob, initViol):
    tag = 'canpPush'
    # direction from post to pre 
    direction = (prePose.point().matrix - postPose.point.matrix).reshape(4)[:3]
    direction[2] = 0.0
    # placeB - ignore support
    placeB = ObjPlaceB(obj, pbs.getWorld().faceFrames, None, PoseD(pose, poseVar), poseDelta)
    objFrame = placeB.objFrame()
    # graspB - from hand and objFrame
    # TODO: what should these values be?
    graspVar = 4*(0.0,)
    graspDelta = 4*(0.0,)
    wrist = robotGraspFrame(pbs, postConf, hand)
    faceFrame = objFrame.inverse.compose(wrist.compose(gripperFaceFrame[hand]))
    graspDescList = [GDesc(obj, faceFrame, 0.0, 0.0, 0.0)]
    graspDescFrame = objFrame.compose(graspDescList[-1].frame)
    graspB =  ObjGraspB(obj, graspDescList, -1,
                        PoseD(hu.Pose(0.,0.,0.,0), graspVar), delta=graspDelta)
    newBS = pbs.copy()
    newBS = newBS.updateHeldBel(graspB, hand)
    newBS = newBS.excludeObjs([obj])
    shWorld = newBS.getShadowWorld(prob)
    attached = shWorld.attached
    if debug(tag): newBS.draw(prob, 'W'); raw_input('Go?')
    rm = pbs.getRoadMap()
    conf = postConf
    path = []
    viol = initViol
    for step in np.arange(0., dist+pushStepSize-0.001, pushStepSize):
        offsetPose = hu.Pose((step*direction).tolist()+[0.0])
        nconf = displaceHand(conf, hand, offset)
        if not nconf:
            return None, None
        viol = rm.confViolations(nconf, newBS, prob, initViol=viol)
        if debug('canPush'):
            print viol
            wm.getWindow('W').startCapture()
            newBS.draw(prob, 'W')
            nconf.draw('W', 'cyan', attached)
            mathematica.mathFile(wm.getWindow('W').stopCapture(),
                                 view = "ViewPoint -> {2, 0, 2}",
                                 filenameOut='./canPush.m')
            raw_input('Next?')
        if viol is None:
            return None, None
        path.append(nconf)
    tr(tag, 1, 'path=%s, viol=%s'%(path, viol))
    return path, viol
