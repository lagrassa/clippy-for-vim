import time
import pdb
import random
import math
from operator import itemgetter
import planGlobals as glob
from fbch import Function
from geom import bboxInside, bboxVolume, vertsBBox, bboxContains, bboxIsect, bboxOverlap
import numpy as np
from traceFile import debug, debugMsg, tr, trAlways
from shapes import thingFaceFrames, drawFrame, Shape
import hu
from planUtil import ObjGraspB, ObjPlaceB, Violations, PoseD, PushResponse
from pr2Util import GDesc, trArgs, otherHand, bboxGridCoords, bboxRandomCoords,\
     supportFaceIndex, inside, Memoizer, objectGraspFrame, robotGraspFrame
from pr2GenUtils import getRegions, getPoseAndSupport, fixed, chooseHandGen, minimalConf
from pr2GenGrasp import potentialGraspConfGen, graspConfForBase
from pr2GenPose import potentialRegionPoseGen
from pr2GenTests import lookAtConfCanView
from pr2Robot import gripperPlace, gripperFaceFrame
from cspace import xyCO
import mathematica
reload(mathematica)
from miscUtil import roundrobin
from transformations import rotation_matrix
import windowManager3D as wm
import ucSearchPQ as search
reload(search)

# Pick poses and confs for pushing an object

# For now, consider only a small set of stable push directions.
# Pretend that centroid of prim is the center of friction.  The push
# path needs to stay inside the regions and the robot needs to be able
# to follow it without colliding with permanent obstacles.

# TODO: Pick a reasonable value

# How many paths to generate for a particular hand contact with the
# object, these will differ by the base placement.
maxPushPaths = 10
maxDone = 1

pushGenCacheStats = [0, 0]
pushGenCache = {}

minPushLength = 0.00999

# Needs work... the hand needs to turn in the direction of the push?
useDirectPush = False
useHorizontalPush = False #True
useVerticalPush = True

class PushGen(Function):
    def fun(self, args, goalConds, bState):
        if glob.inHeuristic:
            glob.pushGenCallsH += 1
        else:
            glob.pushGenCalls += 1

        pbs = bState.pbs
        cpbs = pbs.conditioned(goalConds, [])
        for ans in pushGenGen(args, pbs, cpbs):
            if debug('pushGen'):
                (obj, pose, support, poseVar, poseDelta, confdelta, prob) = args
                (hand, prePose, preConf, pushConf, postConf) = ans.pushTuple()
                preConf.cartConf().prettyPrint('Pre Conf')
                pushConf.cartConf().prettyPrint('Push Conf')
                postConf.cartConf().prettyPrint('Post Conf')
                tr('pushGen', '->', 'final', str(ans))
            yield ans.pushTuple()
            
        if glob.inHeuristic:
            glob.pushGenFailH += 1
        else:
            glob.pushGenFail += 1
        tr('pushGen', '-> completely exhausted')

def pushGenGen(args, pbs, cpbs):
    (obj, pose, support, posevar, posedelta, confdelta, prob) = args
    tag = 'pushGen'
    world = pbs.getWorld()
    # This is the target placement
    placeB = ObjPlaceB(obj, world.getFaceFrames(obj), support,
                       PoseD(pose, posevar), delta=posedelta)
    # Figure out whether one hand or the other is required;  if not, do round robin
    leftGen = pushGenTop((obj, placeB, 'left', prob),
                         pbs, cpbs)
    rightGen = pushGenTop((obj, placeB, 'right', prob),
                          pbs, cpbs)
    # Run the push generator with each of the hands
    for ans in chooseHandGen('push', pbs, cpbs, obj, None, leftGen, rightGen):
        yield ans

def pushGenTop(args, pbs, cpbs, away=False):
    (obj, placeB, hand, prob) = args
    startTime = time.clock()
    tag = 'pushGen'
    trArgs(tag, ('obj', 'placeB', 'hand', 'prob'), args, pbs)
    if obj == 'none' or not placeB:
        tr(tag, '=> obj is none or no placeB, failing')
        return
    if fixed(cpbs.conf) and not away:
        tr(tag, '=> goal conf specified and not away, failing')
        return
    if fixed(cpbs.getHeld(hand)):
        # TODO: we could push with the held object
        tr(tag, '=> Hand=%s is holding in goal, failing'%hand)
        return
    base = cpbs.getBase()
    tr(tag, 'Goal conditions', draw=[(cpbs, prob, 'W')], snap=['W'])
    held = cpbs.getHeld(hand)
    if held != 'none':
        # Get rid of held object in pbs, we'll need to add this to
        # violations in canPush...
        tr(tag, 'Hand=%s is holding %s in pbs'%(hand,held))
        cpbs.updateHeld('none', None, None, hand, None)
    if held == obj:
        curPB = None
    elif obj in [cpbs.getHeld(h) for h in ('left', 'right')]:
        tr(tag, 'obj is in other hand')
        cpbs.updateHeld('none', None, None, otherHand(hand), None)
        curPB = None
    else:
        curPB = cpbs.getPlaceB(obj, default=False)
        if curPB:
            tr(tag, 'obj is placed')

    # Are we in a pre-push?
    ans = push(cpbs, prob, obj, hand)
    if ans:
        shWorld = cpbs.getShadowWorld(prob)
        tr(tag, 'Cached push ->' + str(ans), 'viol=%s'%ans.viol,
           draw=[(cpbs, prob, 'W'),
                 (ans.prePB.shape(shWorld), 'W', 'magenta'),
                 (ans.preConf, 'W', 'magenta', shWorld.attached)],
           snap=['W'])
        yield ans
        
    # Check the cache, otherwise call Aux
    pushGenCacheStats[0] += 1
    key = (cpbs, placeB, hand, base, prob, away, glob.inHeuristic)
    val = pushGenCache.get(key, None)
    if val != None:
        if glob.inHeuristic:
            glob.pushGenCacheH += 1
        else:
            glob.pushGenCache += 1
        
        pushGenCacheStats[1] += 1
        # re-start the generator
        memo = val.copy()
        if debug(tag):
            print tag, 'cached, with len(values)=', len(memo.values)
    else:
        if glob.inHeuristic:
            glob.pushGenCacheMissH += 1
        else:
            glob.pushGenCacheMiss += 1

        gen = pushGenAux(cpbs, placeB, hand, base, curPB, prob,
                         away=away)
        memo = Memoizer(tag, gen)
        pushGenCache[key] = memo
        if debug(tag):
            print tag, 'creating new pushGenAux generator'
    for ans in memo:
        (hand, prePoseTuple, preConf, pushConf, postConf) = ans.pushTuple()
        prePose = hu.Pose(*prePoseTuple)
        # Double check that preConf is safe - this should not be necessary...
        if debug(tag):
            testBS = cpbs.copy()
            testBS.updatePermObjBel(placeB.modifyPoseD(prePose)) # obj at prePose
            testBS.draw(prob, 'W'); preConf.draw('W')
            viol = testBS.confViolations(preConf, prob)
            if not glob.inHeuristic and debug(tag):
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

def getPotentialContacts(prim):
    # Location of center at placeB; we'll treat as COM
    center =  np.average(prim.vertices(), axis=1)
    # find tool point frames (with Z opposing normal, like face
    # frames) that brings the fingers in contact with the object and
    # span the centroid.
    potentialContacts = []
    values = []
    if useVerticalPush: values.append(True)
    if useHorizontalPush: values.append(False)
    for vertical in values:
        potentialContacts.extend(handContactFrames(prim, center, vertical))
    return potentialContacts

def pushGenAux(cpbs, placeB, hand, base, curPB, prob,
               away=False):
    tag = 'pushGen'

    if glob.traceGen:
        print '***', tag+'Aux', glob.inHeuristic, hand, placeB.poseD.mode(), curPB.poseD.mode()

    # The shape at target, without any shadow
    shape = placeB.shape(cpbs.getWorld())
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
    # This should identify arbitrary surfaces, e.g. in shelves.  The
    # bottom of the region is the support polygon.
    supportRegion = cpbs.findSupportRegion(prob, prim, fail = False)
    if not supportRegion:
        trAlways('Target pose is not supported')
        pdb.set_trace()
        return
    potentialContacts = getPotentialContacts(prim)

    for ans in pushGenPaths(cpbs, prob, potentialContacts, placeB, curPB,
                            hand, base, prim, supportRegion.xyPrim(), away=away):
        if glob.traceGen:
            print '***', tag+'Aux', glob.inHeuristic, hand, placeB.poseD.mode(), curPB.poseD.mode(), '->', ans
        yield ans

    tr(tag, '=> pushGenAux exhausted')
    if glob.traceGen:
        print '***', tag+'Aux', glob.inHeuristic, hand, placeB.poseD.mode(), curPB.poseD.mode(), '-> None'
    return

# Return (preConf, pushConf) - conf at prePose and conf at placeB (pushConf)
def potentialConfs(cpbs, prob, placeB, prePose, appPose, graspB, hand, base):
    def graspConfGen(appPB, prePB, targetPB, rev):
        for (c1, ca1, v1) in \
                potentialGraspConfGen(newBS, targetPB if rev else appPB, graspB,
                                      None, hand, base, prob, findApproach=False):
            (x, y, th) = c1['pr2Base']
            basePose = hu.Pose(x, y, 0, th)
            if debug('pushPath'):
                print 'Testing base', basePose, 'at other end of push'
            ans = graspConfForBase(newBS, appPB if rev else targetPB, graspB,
                                   hand, basePose, prob, findApproach=False)
            if ans:
                c2, ca2, v2 = ans
                appConf = c2 if rev else c1
                targetConf = c1 if rev else c2
                pb = prePB.modifyPoseD(var=(1.0e-05, 1.0e-05, 1.0000000000000002e-10, 2.0e-05))
                newPBS = cpbs.copy()
                newPBS.updatePermObjBel(pb)
                newPBS.updateAvoidShadow([pb.obj])
                pb = prePB.modifyPoseD(var=(0.0025, 0.0025, 0.0025, 0.01))
                sh = pb.shape(cpbs.getWorld())
                sw = pb.makeShadow(cpbs, prob)
                if not lookAtConfCanView(newPBS, prob, appConf, sh, shapeShadow=sw, findPath=True):
                    if debug('pushPathLook'):
                        appConf.draw('W', 'orange')
                        raw_input('Pushconf failed look')
                    continue
                if debug('pushPath') or debug('pushPathLook'):
                    # the first should be a appConf, the second a pushConf
                    appConf.draw('W', 'blue'); targetConf.draw('W', 'pink')
                    raw_input('potentialConfs')
                yield (appConf, targetConf)
    prePB = placeB.modifyPoseD(prePose) # pb for initial contact
    appPB = placeB.modifyPoseD(appPose) # pb for approach to object (no contact)
    newBS = cpbs.copy().excludeObjs([placeB.obj])
    if debug('pushPath'):
        print 'potentialConfs for', appPose, prePose, placeB.poseD.mode()
    return roundrobin(graspConfGen(appPB, prePB, placeB, True),
                      graspConfGen(appPB, prePB, placeB, False))

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

def reverseConf(pp, hand):
    cpost, _, _ = pp[-1]                # end of the path, contact with object
    handFrameName = cpost.robot.armChainNames[hand]
    tpost = cpost.cartConf()[handFrameName] # final hand position
    for i in range(2, len(pp)):
        c, _, _ = pp[-i]
        t = c.cartConf()[handFrameName] # hand position
        if t.distance(tpost) > 0.1:
            return c
    return pp[0][0]

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

        pushCenter[2] = max(shape.bbox()[0,2] + minPushHeight(vertical),
                            shape.bbox()[1,2] - 0.1) # avoid colliding with forearm
    else:
        pushCenter[2] = shape.bbox()[0,2] + minPushHeight(vertical)
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
def minPushHeight(vertical):
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

def graspBForContactFrame(cpbs, prob, contactFrame, zOffset, placeB, hand, vertical):
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
        placeB.shape(cpbs.getWorld()).draw('W')
        raw_input('displacedContactFrame')
    graspB = None
    # consider flips of the hand (mapping one finger to the other)
    for angle in (0, np.pi):
        displacedContactFrame = displacedContactFrame.compose(hu.Pose(0.,0.,0.,angle))
        if debug(tag):
            cpbs.draw(prob); drawFrame(displacedContactFrame)
            raw_input('displacedContactFrame (rotated)')
        gM = displacedContactFrame.compose(hu.Transform(vertGM if vertical else horizGM))
        if debug(tag):
            print gM.matrix
            print 'vertical =', vertical
            cpbs.draw(prob); drawFrame(gM)
            raw_input('gM')
        gT = objFrame.inverse().compose(gM) # gM relative to objFrame
        # TODO: find good values for dx, dy, dz must be 0.
        graspDescList = [GDesc(obj, gT, 0.0, 0.0, 0.0)]
        graspDescFrame = objFrame.compose(graspDescList[-1].frame)
        if debug(tag): print 'graspDescFrame\n', graspDescFrame.matrix
        # -1 grasp denotes a "virtual" grasp
        gB = ObjGraspB(obj, graspDescList, -1, placeB.support.mode(),
                       PoseD(hu.Pose(0.,0.,0.,0), graspVar), delta=graspDelta)
        wrist = objectGraspFrame(cpbs, gB, placeB, hand)
        if any(cpbs.getWorld().robot.potentialBasePosesGen(wrist, hand, complain=False)):
            if debug(tag): print 'valid graspB'
            graspB = gB
            break
    if debug(tag):
        print graspB
        raw_input('graspB')
    assert graspB
    return graspB

def gripSet(conf, hand, width=0.08):
    return conf.set(conf.robot.gripperChainNames[hand], [min(width, 0.08)])

# =================================================================
# PushInRegionGen
# =================================================================

class PushInRegionGen(Function):
    # Return objPose, poseFace.
    def fun(self, args, goalConds, bState):
        pbs = bState.pbs
        cpbs = pbs.conditioned(goalConds, [])
        for ans in pushInRegionGenGen(args, pbs, cpbs, away = False):
            assert isinstance(ans, tuple)
            yield ans.poseInTuple()

def pushInRegionGenGen(args, pbs, cpbs, away = False):
    if debug('disablePush'): return 
    (obj, region, var, delta, prob) = args

    if glob.traceGen:
        print '***', 'pushInRegionGenGen', obj, region.name()

    tag = 'pushInGen'
    world = pbs.getWorld()
    # Get the regions
    regions = getRegions(region)
    shWorld = pbs.getShadowWorld(prob)
    regShapes = [shWorld.regionShapes[region].xyPrim() if isinstance(region, str) else region\
                 for region in regions]
    regions = [x.name() for x in regShapes]
    tr(tag, 'Target region in purple',
       draw=[(pbs, prob, 'W')] + [(rs, 'W', 'purple') for rs in regShapes], snap=['W'])
    pose, support = getPoseAndSupport(tag, obj, pbs, prob) # from original pbs
    # Check if object pose is specified in goal
    if obj in cpbs.objectBs and fixed(cpbs.objectBs[obj]):
        pB = cpbs.getPlaceB(obj)
        shw = shadowWidths(pB.poseD.var, pB.delta, prob)
        shwMin = shadowWidths(graspV, graspDelta, prob)
        if any(w > mw for (w, mw) in zip(shw, shwMin)):
            args = (obj, pose, support, var, delta, None, prob)
            gen = pushGenGen(args, pbs, cpbs)
            for ans in gen:
                tr(tag, str(ans), 'regions=%s'%regions,
                   draw=[(cpbs, prob, 'W')] + [(rs, 'W', 'purple') for rs in regShapes],
                   snap=['W'])
                yield ans
            return
        else:
            # If pose is specified and variance is small, return
            tr(tag, '=> Pose is specified and variance is small - fail')
            return

    # The normal case

    # Use the input var and delta to select candidate poses in the
    # region.  We will use smaller values (in general) for actually
    # placing.
    placeB = ObjPlaceB(obj, world.getFaceFrames(obj), support,
                       PoseD(pose, var), delta=delta)

    gen = pushInGenTop((obj, regShapes, placeB, prob),
                          pbs, cpbs, away = away)
    for ans in gen:
        tr(tag, ans)
        yield ans
    tr(tag, 'No answers remaining')

pushVarIncreaseFactor = 3 # was 2

def pushInGenAway(args, pbs):
    (obj, delta, prob) = args
    placeB = pbs.getPlaceB(obj, default=False)
    assert placeB, 'Need to know where object is to get support region'
    shape = placeB.shape(pbs.getWorld())
    regShape = pbs.findSupportRegion(prob, shape.xyPrim()).xyPrim()
    tr('pushInGenAway', zip(('obj', 'delta', 'prob'), args),
       draw=[(pbs, prob, 'W')], snap=['W'])
    targetPushVar = tuple([pushVarIncreaseFactor * x \
                            for x in pbs.domainProbs.obsVarTuple])
    for ans in pushInRegionGenGen((obj, regShape,
                                   targetPushVar, delta, prob),
                                   pbs, pbs, away=True):
        yield ans

def pushOut(pbs, prob, obst, delta):
    tr('pushOut', 'obst=%s'%obst)
    domainPlaceVar = tuple([pushVarIncreaseFactor * x \
                            for x in pbs.domainProbs.obsVarTuple])
    if not isinstance(obst, str):
        obst = obst.name()
    for ans in pushInGenAway((obst, delta, prob), pbs):
        ans = ans.copy()
        ans.var = pbs.domainProbs.objBMinVar(obst)
        ans.delta = delta
        yield ans

pushInGenMaxPoses  = 50
pushInGenMaxPosesH = 10

def connectedSupport(reg, support):
    isectBB = bboxIsect([reg.bbox(), support.bbox()])
    return bboxVolume(isectBB) > 0 and abs(isectBB[0,2] - support.bbox()[0,2]) <= 0.005

def pushInGenTop(args, pbs, cpbs, away = False):
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
    if fixed(cpbs.conf) and not away:
        # if conf is specified, just fail
        tr(tag, '=> conf is specified, failing')
        return
    supportRegion = cpbs.findSupportRegionForObj(prob, obj,
                                                strict=True, fail=False)
    regShapes = [r.xyPrim() for r in regShapes if connectedSupport(r, supportRegion)]
    if not regShapes:
        tr(tag, '=> target regions do no share support')
        return
    conf = None
    confAppr = None
    shWorld = cpbs.getShadowWorld(prob)
    nPoses = pushInGenMaxPosesH if glob.inHeuristic else pushInGenMaxPoses

    # Check the cache, otherwise call Aux
    leftGen = pushInGenAux(pbs, cpbs, prob, placeB, regShapes,
                           'left', away=away)
    rightGen = pushInGenAux(pbs, cpbs, prob, placeB, regShapes,
                            'right', away=away)
    # Figure out whether one hand or the other is required;  if not, do round robin
    mainGen = chooseHandGen('push', pbs, cpbs, obj, None, leftGen, rightGen)
    # Picks among possible target poses and then try to push it in region
    for ans in mainGen:
        yield ans

# placeB is current place for object
# regShapes is a list of (one) target region
def pushInGenAux(pbs, cpbs, prob, placeB, regShapes, hand,
                 away=False):
    def feasiblePBS(pB):
        if pbs.conditions:
            newBS = pbs.copy()
            newBS.updatePlaceB(pB)
            return newBS.feasible()           # check conditioned fluents
        else:
            return True
    tag = 'pushInGen'
    shWorld = cpbs.getShadowWorld(prob)
    # Get a pose in one of the regions.  If away=True, has to be one
    # push away.
    for pB in regionTargetPB(cpbs, prob, placeB, regShapes, hand, away=away):
        if feasiblePBS(pB):
            tr(tag, 'target is feasible:', pB)
        else:
            tr(tag, 'target is not feasible:', pB)
            continue
        tr(tag, 'target', pB,
           draw=[(cpbs, prob, 'W'),
                 (pB.makeShadow(pbs, prob), 'W', 'pink')] \
                 + [(rs, 'W', 'purple') for rs in regShapes],
           snap=['W'])

        if debug(tag+'Vefify'):
            print 'pushInGen asks', pB.poseD.mode().xyztTuple()
            cpbs.draw(prob, 'W');
            for rs in regShapes: rs.draw('W', 'purple')
            pB.makeShadow(pbs, prob).draw('W', 'pink')
            raw_input(tag)

        if not any(inside(pB.makeShadow(cpbs, prob), regShape, strict=True) \
                   for regShape in regShapes):
           for rs in regShapes: rs.draw('W', 'purple')
           cpbs.draw(prob, 'W'); pB.makeShadow(pbs, prob).draw('W', 'pink')
           print 'pushInGen target not in region'
           pdb.set_trace()

        for ans in pushGenTop((placeB.obj, pB, hand, prob),
                              pbs, cpbs, away=away):
            if debug(tag+'Vefify'):
                print 'pushGen achieves', ans.postPB.poseD.mode().xyztTuple()
            tr(tag, '->', str(ans),
               draw=[(cpbs, prob, 'W'),
                     (ans.prePB.shape(shWorld), 'W', 'blue'),
                     (ans.postPB.shape(shWorld), 'W', 'pink'),
                     (ans.preConf, 'W', 'blue', shWorld.attached),
                     (ans.pushConf, 'W', 'pink', shWorld.attached)] \
               + [(rs, 'W', 'purple') for rs in regShapes],
               snap=['W'])

            yield ans
    tr(tag, 'No targets left')

def regionTargetPB(pbs, prob, placeB, regShapes, hand,
                   nPoses=30, away=False):
    def contactGraspBGen():
        shape = placeB.shape(pbs.getWorld())
        potentialContacts = getPotentialContacts(choosePrim(shape))
        # random.shuffle(potentialContacts)
        for (vertical, contactFrame, width) in potentialContacts:
            graspB = graspBForContactFrame(pbs, prob, contactFrame,
                                           0.0,  placeB, hand, vertical)
            yield graspB
    for pose in potentialRegionPoseGen(pbs, placeB,
                                       Memoizer('graspB', contactGraspBGen()),
                                       prob, regShapes,
                                       hand, None,
                                       maxPoses=nPoses, angles=[placeB.poseD.mode().pose().theta]):
        yield placeB.modifyPoseD(pose)

PushCache = {}
def cachePushResponse(pr):
    minConf = minimalConf(pr.preConf, pr.hand)
    if minConf not in PushCache:
        PushCache[minConf] = pr
    else:
        old = PushCache[minConf]
        if old.prePB.poseD.mode() != pr.prePB.poseD.mode() or \
           old.postPB.poseD.mode() != pr.postPB.poseD.mode():
            print 'PushCache collision'
            pdb.set_trace()

def push(pbs, prob, obj, hand):
    minConf = minimalConf(pbs.getConf(), hand)
    if minConf in PushCache:
        pr = PushCache[minConf]
    else:
        return
    assert pr.hand == hand
    path, viol = canPush(pbs, obj, hand, pr.prePB.support.mode(),
                         pr.prePB.poseD.mode().xyztTuple(), pr.postPB.poseD.mode().xyztTuple(),
                         pr.preConf, pr.pushConf, pr.postConf, pr.var,
                         pr.var, pr.delta, prob,
                         Violations(), prim=False)
    if viol:
        print '*** Cached push ***'
        return pr
    else:
        print 'Cached push failed'
        pdb.set_trace()


def displaceHand(conf, hand, offsetPose):
    cart = conf.cartConf()
    handFrameName = conf.robot.armChainNames[hand]
    trans = cart[handFrameName]         # initial hand position
    nTrans = offsetPose.compose(trans)  # final hand position
    nCart = cart.set(handFrameName, nTrans)
    nConf = conf.robot.inverseKin(nCart, conf=conf) # use conf to resolve
    if all(nConf.values()):
        return nConf

# find bbox for CI_1(2), that is, displacements of bb1 that place it
# inside bb2.  Assumes that bb1 has origin at 0,0.
def bboxInterior(bb1, bb2):
    for j in xrange(3):
        di1 = bb1[1,j] - bb1[0,j]
        di2 = bb2[1,j] - bb2[0,j]
        if di1 > di2: return None
    return np.array([[bb2[i,j] - bb1[i,j] for j in range(3)] for i in range(2)])


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
    # Return
    return (appDir, appDist, pushDir, pushDist)

# The conf in the input is the robot conf in contact with the object
# at the destination pose.

pushStepSize = 0.01

def pushPath(pbs, prob, gB, pB, conf, initPose, preConf, regShape, hand):
    tag = 'pushPath'
    def finish(reason, gloss, pathViols=[], cache=True):
        if debug(tag):
            print '->', reason, gloss, 'path len=', len(pathViols)
            debugMsg(reason)
        ans = (pathViols, reason)
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
    key = (postPose, baseSig, initPose, hand, glob.pushBuffer, glob.inHeuristic)
    names =('postPose', 'base', 'initPose', 'hand', 'pushBuffer', 'inHeuristic')
    cached = checkPushPathCache(key, names, pbs, prob, gB, conf, newBS)
    if cached is not None:
        pathViols, reason = cached
        return finish(reason, 'Cached answer', pathViols, cache=False)
    if debug(tag): newBS.draw(prob, 'W'); raw_input('pushPath: Go?')
    # Check there is no permanent collision at the goal
    viol = newBS.confViolations(conf, prob)
    if not viol:
        return finish('collide', 'Final conf collides in pushPath')
    preViol = newBS.confViolations(preConf, prob)
    if not preViol:
        return finish('collide', 'Pre-conf collides in pushPath')
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
    # The minimal shadow, at the destination
    shape = pB.makeShadow(pbs, 0.0)
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
    pathViols = [(preConf, preViol, None)]
    reason = 'done'                     # done indicates success
    #######################
    # Set up state for the approach scan
    #######################
    # Number of steps for approach displacement
    nsteps = int(appDist / pushStepSize)
    delta = appDist / nsteps
    stepVals = [0, nsteps] if glob.inHeuristic else xrange(nsteps+1)
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
            drawState(newBS, prob, nconf, shape)
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
        stepVals = [0, nsteps] if glob.inHeuristic else xrange(nsteps+1)
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
        hOffsetPose = hu.Pose(*(((step - handTiltOffset)*pushDir).tolist()+[0.0]))
        nconf = displaceHandRot(initConf, hand, hOffsetPose,
                                tiltRot = tiltRot, angle=step*deltaAngle)
        if not nconf:
            reason = 'invkin'; break
        if step_i == nsteps:
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
            drawState(newBS, prob, nconf, nshape)
        pathViols.append((nconf, viol, offsetPB.poseD.mode()))
    #######################
    # Prepare ans
    #######################
    if not pathViols:
        reason = 'no path'
    return finish(reason, 'Final pushPath', pathViols)

def armCollides(conf, objShape, hand):
    armShape = conf.placement()
    parts = dict([(part.name(), part) for part in armShape.parts()])
    gripperName = conf.robot.gripperChainNames[hand]
    return any(objShape.collides(parts[name]) for name in parts if name != gripperName)

def drawState(pbs, prob, conf, shape=None):
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

"""
A simple xy planner to push from a specified target location towards a
specified final location.  In the away mode, then we must reach the
initial state.  Note that all positions need to remain inside a given
region shape.

A (small) set of pushing directions are allowed.  Note that these are
one-way, pushing opposite to a face normal.

A set of C-obstacles are defined by the pushed object and the hand at
the push pose.  The tangents of the obstacles for each of the push
directions (and rays through the start and target define a sort of
grid that is searched for a path.

Turns in a path are expensive because they require breaking contact
and moving to a new contact.

Each path segment can be checked for feasibility (and mapped to robot)
separately - canReachHome at beginning and end and then kinematics and
collision checks along the segment - without moving the base.  If
kinematics fails partway, we can break the segment and check
separately for feasibility of the rest.  That way, we may be able to
move the base to make the kinematics feasible.
"""

# Returns a list of dictionaries of C-obstacles, one for each
# potential push.  We don't distinguish permanents from not.
def CObstacles(pbs, prob, potentialPushes, pB, hand, supportRegShape):
    shWorld = pbs.getShadowWorld(prob)
    objShape = pB.makeShadow(pbs, prob)
    (x,y,z,_) = pB.poseD.mode().pose().xyztTuple()
    centeringOffset = hu.Pose(-x,-y,-z,0.)

    xyObst = {name: obst.xyPrim() \
              for (name, obst) in shWorld.objectShapes.iteritems()}
    CObsts = {}
    for (dirx, vert), (gB, width) in potentialPushes.iteritems():
        wrist = objectGraspFrame(pbs, gB, pB, hand)
        # the gripper at push pose relative to pB
        gripShape = gripperPlace(pbs.getConf(), hand, wrist)
        # the combinaed gripper and object, origin is that of object
        objGripShape = Shape([gripShape, objShape], \
                             objShape.origin(),
                             name='gripper').xyPrim() # note xyPrim
        # center the shape at the origin (only in x,y translation)
        objGripShapeCtr = objGripShape.applyTrans(centeringOffset)
        bb = bboxExtendXY(supportRegShape.bbox(), objGripShapeCtr.bbox())
        CO = {}
        for (name, obst) in xyObst.iteritems():
            if bboxOverlap(bb, obst.bbox()):
                c = xyCO(objGripShapeCtr, obst)
                CO[name] = c
                c.properties['name'] = name

                if 'table' in name:
                    pdb.set_trace()

            else:
                CO[name] = None
        CObsts[(dirx, vert)] = CO

    return CObsts
        
def bboxExtendXY(bbB, bbA):
    bb = np.array([bbB[0] - bbA[0], bbB[1] + bbA[1]])
    bb[0,2] = bbB[0,2]
    bb[1,2] = bbB[1,2]
    return bb

# Given obst shape and dirx vector [x,y,0], returns two line
# equations [-y,x,0,-dmin] and [-y,x,0,-dmax]
def tangents(obst, dirx, eps = 0.):
    x = dirx[0]; y = dirx[1]            # has been normalized
    verts = obst.getVertices()
    ext = [float('inf'), -float('inf')]
    for i in xrange(verts.shape[1]):
        d = -verts[0,i]*y + verts[1,i]*x
        ext[0] = min(ext[0], d)
        ext[1] = max(ext[1], d)
    return [np.array([-y,x,0,-(ext[0]-eps)]), np.array([-y,x,0,-(ext[1]+eps)])]

def lineThruPoint(pt, dirx):
    x = dirx[0]; y = dirx[1]            # has been normalized
    d = -pt[0]*y + pt[1]*x
    return np.array([-y,x,0,-d])
    
def edgeCross((A,B), (C,D)):
    """
Suppose the two segments have endpoints A,B and C,D. The numerically
robust way to determine intersection is to check the sign of the four
determinants:

| Ax-Cx  Bx-Cx |    | Ax-Dx  Bx-Dx |
| Ay-Cy  By-Cy |    | Ay-Dy  By-Dy |

| Cx-Ax  Dx-Ax |    | Cx-Bx  Dx-Bx |
| Cy-Ay  Dy-Ay |    | Cy-By  Dy-By |

For intersection, each determinant on the left must have the opposite
sign of the one to the right,
"""
    def det(a,b,c):
        return (a[0]-c[0])*(b[1]-c[1]) - (a[1]-c[1])*(b[0]-c[0])
    return (np.sign(det(A,B,C)) == -np.sign(det(A,B,D))) and \
           (np.sign(det(C,D,A)) == -np.sign(det(C,D,B)))

def edgeCollides(p0, p1, poly):
    # Are bounding boxes disjoint?
    bb = poly.bbox()
    if min(p0[0], p1[0]) >= bb[1][0] or \
       min(p0[1], p1[1]) >= bb[1][1] or \
       max(p0[0], p1[0]) <= bb[0][0] or \
       max(p0[1], p1[1]) <= bb[0][1]:
        return False
    # Check in case edge is completely inside
    (z0, z1) = poly.zRange()
    p = np.array([p0[0], p0[1], 0.5*(z0+z1), 1.0])
    if np.all(np.dot(poly.planes(),
                     np.resize(p,(4,1)))<=0): return True
    p = np.array([p1[0], p1[1], 0.5*(z0+z1), 1.0])
    if np.all(np.dot(poly.planes(),
                     np.resize(p,(4,1)))<=0): return True
    # Check for crossings
    verts = poly.vertices()
    face0 = poly.faces()[0]
    nv = len(face0)
    for v in xrange(nv):
        if edgeCross((p0, p1), (verts[:,v], verts[:,(v+1)%nv])):
            return True
    return False

def lineIsectDist(pt, dirx, line):
    cos = sum([dirx[i]*line[i] for i in (0,1)])
    if abs(cos) < 1e-6: return None
    return -(sum([line[i]*pt[i] for i in (0,1)]) + line[3]) / cos

# tangents is a dictionary {dirx: list of tangent lines}
# When searching backwards from the target, set reverse=True
def nextCrossing(pt, dirx, tangents, goal, regShape, reverse = False):
    if reverse:
        tdirx = [-d for d in dirx]
    else:
        tdirx = dirx
    # distance to goal along tdirx
    d = sum([(goal[i] - pt[i])*tdirx[i] for i in (0,1)])
    if d >= -1e-6 and all(abs(pt[i] + tdirx[i]*d - goal[i]) < 1e-4 for i in (0,1)):
        # aligned with goal
        bestDist = d
        bestDirs = ['goal']
    else:
        bestDist = float('inf')
        bestDirs = []
    for (ndirx, nvert), tanLines in tangents.iteritems():
        if abs(np.dot(tdirx, ndirx)) == 1.0: continue
        for line in tanLines:
            d = lineIsectDist(pt, tdirx, line)
            if d and d > 1e-6:
                if abs(d - bestDist) < 1e-6 and bestDirs != ['goal']:
                    if ndirx not in bestDirs:
                        bestDirs = bestDirs + [ndirx]
                elif d < bestDist:
                    bestDist = d
                    bestDirs = [ndirx]
    if bestDist < float('inf'):
        (z0, z1) = regShape.zRange()
        p = np.array([pt[i] + bestDist*tdirx[i] for i in (0,1)] + [0.5*(z0+z1), 1.])
        if np.all(np.dot(regShape.planes(), np.resize(p,(4,1)))<=0.):
            if bestDirs == ['goal']:
                return goal, bestDirs, bestDist    # return goal exactly
            else:
                # print 'bestDirs', bestDirs
                return (p[0],p[1]), bestDirs, bestDist
    return None, None, None

class PushState:
    def __init__(self, pt, dirx, vert, coll):
        self.pt = pt
        self.dirx = dirx
        self.vert = vert
        self.coll = coll
    def __str__(self):
        return 'PushState(%s,%s,%s,%s)'%(self.pt, self.dirx, self.vert, self.coll)
    __repr__ = __str__

class PushSegment:
    def __init__(self, pt0, pt1, dirx, vert, coll):
        self.pt0 = pt0
        self.pt1 = pt1
        self.dirx = dirx
        self.vert = vert
        self.coll = coll
    def __str__(self):
        return 'PushSegment(%s,%s,%s,%s,%s)'%(self.pt0, self.pt1, self.dirx, self.vert,self.coll)
    __repr__ = __str__

turnCost = 1.0
def searchObjPushPath(pbs, prob, potentialPushes, targetPB, curPB,
                      hand, supportRegShape, away=False):
    curPt = curPB.poseD.mode().pose().xyztTuple()[:2]
    targetPt = targetPB.poseD.mode().pose().xyztTuple()[:2]
    # Use a fixed shadow in computing the COs to avoid growth in COs
    # in successive calls
    ntargetPB = targetPB.modifyPoseD(var=(0.0001,0.0001, 1e-10,0.0004))
    ntargetPB.delta = (0.01,0.01,0.0002,0.02)
    COs = CObstacles(pbs, prob, potentialPushes, ntargetPB,
                     hand, supportRegShape)
    dirxs = []
    tLines = {}
    for (dirx, vert), (gB, width) in potentialPushes.iteritems():
        dirxs.append((dirx, vert))
        tanLines = []
        for c in COs[(dirx, vert)].values():
            if c:
                tanLines.extend(tangents(c, dirx, 0.001))
        tanLines.append(lineThruPoint(curPt, dirx))
        tanLines.append(lineThruPoint(targetPt, dirx)) 
        tLines[(dirx, vert)] = tanLines
    fixed = pbs.getShadowWorld(prob).fixedObjects
    def countViols(state, p1):
        count = 0
        viols = []
        for name, shape in COs[(state.dirx, state.vert)].iteritems():
            if shape and name not in state.coll and \
                   edgeCollides(state.pt, p1, shape):
                viols.append(name)
                if name in fixed:
                    return None, None
                if 'shadow' in name: count += 0.5
                else: count += 1.0
        return count, viols
    def actions(state):
        if state is None:               # initial state
            return [(PushState(targetPt, dx, vert, []), 0.) for dx,vert in dirxs]
        elif state.dirx == 'goal':
            return []
        else:
            p, dxs, d = nextCrossing(state.pt, state.dirx, tLines,
                                    curPt, supportRegShape, reverse=True)
            if debug('searchObjPushPath'):
                print '  Next:', state.pt, state.dirx, '->', p
            if p:
                vcost, nviol = countViols(state, p)
                if debug('searchObjPushPath'):
                    print '  Cost:', vcost, nviol
                if vcost is not None: # None means permanent collision
                    # switch direction or stay the course
                    if dxs == ['goal']:
                        return [(PushState(p, dxs[0], state.vert, state.coll+nviol), d + vcost)]
                    else:
                        return [(PushState(p, dx, state.vert, state.coll+nviol), d + vcost + turnCost) \
                                for dx in dxs] + \
                                [(PushState(p, state.dirx, state.vert, state.coll+nviol), d + vcost)]
            return []
    def successor(state, action):
        return action                   # state, cost
    def heuristic(s, g):
        if s is None:
            s = targetPt
        return sum([abs(si - gi) for si, gi in zip(s[0:2], g[0:2])])
    
    if debug('pushGen'): print 'searchObjPushPath...'
    print 'searchObjPushPath, target=', targetPt, 'cur=', curPt
    gen = search.searchGen(None, [curPt], actions, successor,
                           heuristic,
                           goalKey = lambda x: x.pt if x else x,
                           printFinal = debug('pushGen'),
                           verbose = debug('pushGen'),
                           fail = False)
    for (path, costs) in gen:
        if debug('pushGen'):
            print '...yielding', path
            if path is None:
                pdb.set_trace()
            else:
                pbs.draw(prob, 'W')
                targetPose = targetPB.poseD.mode()
                for (a, s) in path:
                    if not s: continue
                    pose = hu.Pose(s.pt[0], s.pt[1], targetPose.z, targetPose.theta)
                    targetPB.modifyPoseD(pose).makeShadow(pbs, prob).draw('W', 'blue')
                raw_input('Push path')
        print '->'
        if path and path[1:]:
            for p in path[1:]: print '  ', p[0]
        else:
            print 'path=', path
            pdb.set_trace()
        yield path
    print 'searchObjPushPath end'

def segmentPath(path):
    segments = []
    curseg = None
    for (a, s) in path:
        if not s:                       # filler
            continue
        elif curseg and s.dirx == curseg.dirx:     # continue seg
            curseg.pt0 = s.pt           # start pt
            curseg.coll.update(s.coll)
        else:                           # open new seg, save old
            if curseg:
                curseg.pt0 = s.pt       # start pt
                curseg.coll.update(s.coll)
                segments.append(curseg)
            curseg = PushSegment(None, s.pt, s.dirx, s.vert, set(s.coll))
    # save the last seg
    segments.append(curseg)
    return segments

def pushGenPaths(pbs, prob, potentialContacts, targetPB, curPB,
                 hand, base, prim, supportRegShape, away=False):
    tag = 'pushGen'
    newBS = pbs.copy().excludeObjs([targetPB.obj])
    # Define a collection of potential (straight line) pushes, defined
    # by the direction and the "grasp", the position of the hand
    # relative to the object.
    potentialPushes = {}                # dirx: (graspB, width)
    frames = {}
    for (vertical, contactFrame, width) in potentialContacts:
        # construct a graspB corresponding to the push hand pose,
        # determined by the contact frame
        graspB = graspBForContactFrame(newBS, prob, contactFrame,
                                       0.0,  targetPB, hand, vertical)
        # contactFrame z is negative z normal of face
        direction = contactFrame.matrix[:3,2].copy().reshape(3)
        direction[2] = 0.0            # we want z component exactly 0.
        direction /= np.dot(direction,direction)**0.5 # normalize
        # Use tuple for direction so we can hash on it.
        dirx = (direction[0], direction[1])
        potentialPushes[(dirx, vertical)] = (graspB, width)
        frames[(dirx, vertical)] = contactFrame
    def split(seg):
        return PushSegment([0.5*(x0+x1) for x0,x1 in zip(seg.pt0, seg.pt1)],
                           seg.pt1, seg.dirx, seg.vert, seg.coll)
    # if curPB is None, then use targetPB -- does this make sense?
    for path in searchObjPushPath(newBS, prob, potentialPushes, targetPB, curPB or targetPB,
                                  hand, supportRegShape, away=away):
        if not path: return
        pathSegs = segmentPath(path)
        for i in range(len(pathSegs)):
            ps = pathSegs[i]
            if not ps.pt0:
                print 'degenerate path segment', ps
                continue     # degenerate path
            maxPush = max([abs(a-b) for (a,b) in zip(ps.pt0,ps.pt1)])
            minDelta = min(targetPB.delta[0], targetPB.delta[1])
            print 'maxPush', maxPush, 'minDelta', minDelta
            if maxPush < minDelta:
                print 'maxPush < minDelta', ps
                continue
            else:
                break
        graspB, width = potentialPushes[(ps.dirx, ps.vert)]
        for seg in (ps, split(ps)):
            # Use pbs with object in it
            print '  ', seg
            for ans in pushGenPathsAux(pbs, newBS, prob, seg, graspB, width, targetPB, curPB,
                                       hand, base, supportRegShape,
                                       frame=frames[(ps.dirx,ps.vert) ]):
                print '->', ans
                yield ans
            print '  -> end'
        
def pushGenPathsAux(pbs, newBS, prob, pathSeg, graspB, width, targetPB, curPB,
                    hand, base, supportRegion, frame=None):
    tag = 'pushGen'
    dirx = pathSeg.dirx
    direction = np.array([dirx[0], dirx[1], 0.0])
    curPose = curPB.poseD.mode()
    targetPose = targetPB.poseD.mode()
    assert abs(targetPose.x - pathSeg.pt1[0]) <= targetPB.delta[0] and \
           abs(targetPose.y - pathSeg.pt1[1]) <= targetPB.delta[1]
    if debug(tag):
        newBS.draw(prob, 'W')
        shape = targetPB.shape(newBS.getWorld())
        shape.draw('W', 'pink');
        shape = curPB.shape(newBS.getWorld())
        shape.draw('W', 'blue');
        if frame: drawFrame(frame)
        raw_input('object push')
    # initial pose for object along direction
    prePose = hu.Pose(pathSeg.pt0[0], pathSeg.pt0[1], targetPose.z, targetPose.theta)
    pushPaths = []                  # for different base positions
    # Generate confs to place the hand at graspB
    count = 0                       # how many tries
    doneCount = 0                   # how many went all the way
    appPoseOffset = hu.Pose(*(-glob.pushBuffer*direction).tolist()+[0.0])
    appPose = appPoseOffset.compose(prePose).pose()
    if debug(tag):
        print 'Calling potentialConfs, with appPose', appPose
    # Passing pbs with object still init
    for ans in potentialConfs(pbs, prob, targetPB, prePose, appPose, graspB, hand, base):
        if not ans:
            tr(tag+'_kin', 'potential grasp conf is empy')
            continue
        (prec, postc) = ans         # conf, approach, violations
        preConf = gripSet(prec, hand, 2*width) # open fingers
        pushConf = gripSet(postc, hand, 2*width) # open fingers
        if debug(tag+'_kin'):
            pushConf.draw('W', 'orange')
            raw_input('Candidate conf')
        count += 1
        pathAndViols, reason = pushPath(newBS, prob, graspB, targetPB, pushConf,
                                        prePose, preConf, supportRegion, hand)
        if reason == 'done':
            pushPaths.append((pathAndViols, reason))
            doneCount +=1 
            if doneCount >= maxDone: break
        tr(tag, 'pushPath reason = %s, path len = %d'%(reason, len(pathAndViols)))
        if count > maxPushPaths: break
    pose1 = targetPB.poseD.mode()
    pose2 = appPose
    if debug(tag):
        if count == 0:
            print 'No push', direction[:2], 'between', pose1, pose2, 'with', hand
            debugMsg('pushFail', 'Could not find conf for push along %s'%direction[:2])
        else:
            print 'Found conf for push', direction[:2], 'between', pose1, pose2 
    # Sort the push paths by violations
    sorted = sortedPushPaths(pushPaths, curPose)
    for i in range(min(len(sorted), maxDone)):
        pp = sorted[i]
        ppre, cpre, ppost, cpost = getPrePost(pp)
        if not ppre or not ppost: continue
        crev = reverseConf(pp, hand)
        if debug(tag):
            robot = cpre.robot
            handName = robot.armChainNames[hand]
            print 'pre pose\n', ppre.matrix
            for (name, conf) in (('pre', cpre), ('post', cpost), ('rev', crev)):
                print name, 'conf tool'
                print conf.cartConf()[handName].compose(robot.toolOffsetX[hand]).matrix
        tr(tag, 'pre conf (blue), post conf (pink), rev conf (green)',
           draw=[(newBS, prob, 'W'),
                 (cpre, 'W', 'blue'), (cpost, 'W', 'pink'),
                 (crev, 'W', 'green')], snap=['W'])
        viol = Violations()
        for (c,v,p) in pathAndViols:
            viol.update(v)
        ans = PushResponse(targetPB.modifyPoseD(ppre.pose()),
                           targetPB.modifyPoseD(ppost.pose()),
                           cpre, cpost, crev, viol, hand,
                           targetPB.poseD.var, targetPB.delta)
        if debug('pushFail'):
            print 'Yield push', ppre.pose(), '->', ppost.pose()
        cachePushResponse(ans)
        yield ans


# Pushing                

# returns path, violations
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

print 'Loaded pr2Push.py'        

