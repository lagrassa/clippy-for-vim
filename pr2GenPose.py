import pdb
import random
import math
import hu
from cspace import CI
from pr2Util import checkCache, shadowName, inside, bboxRandomCoords
from pr2GenUtils import sortedHyps, baseDist, graspGen
from pr2GenGrasp import potentialGraspConfGen
import planGlobals as glob
from traceFile import tr, debug, debugMsg

# TODO: Should pick relevant orientations... or more samples.
angleList = [-math.pi/2. -math.pi/4., 0.0, math.pi/4, math.pi/2]

# (regShapeName, obj, hand, grasp) : (n_attempts, [relPose, ...])
regionPoseCache = {}

# Generate poses
# Requirements: inside region, graspable, pbs is feasible
# Preferences:  bigger shadows, fewer violations, grasp conf close to current conf
def potentialRegionPoseGen(pbs, placeB, graspBGen, prob, regShapes, hand, base,
                           maxPoses = 30, angles = angleList):
    obj = placeB.obj
    if glob.traceGen:
        print ' **', 'potentialRegionPoseGen', obj, placeB.poseD.mode(), hand
    # Consider a range of variances
    maxVar = placeB.poseD.var
    minVar = pbs.domainProbs.obsVarTuple
    pBs = [placeB.modifyPoseD(var=medVar) \
           for medVar in interpolateVars(maxVar, minVar, 4)]
    count = 0
    for pose in potentialRegionPoseGenAux(pbs, obj, pBs, graspBGen, prob, regShapes,
                                          hand, base, maxPoses=maxPoses, angles=angles):
        yield pose
        if count > maxPoses: return
        count += 1

# Intermediate variances between minV and maxV
def interpolateVars(maxV, minV, n):
    deltaV = [(maxV[i]-minV[i])/(n-1.) for i in range(4)]
    if all([d < 0.001 for d in deltaV]):
        return [maxV]
    Vs = []
    for j in range(n):
        Vs.append(tuple([maxV[i]-j*deltaV[i] for i in range(4)]))
    return Vs

class PoseHyp(object):
    def __init__(self, pB, shadow):
        self.pB = pB
        self.shadow = shadow
        self.conf = None
        self.viol = None
    def getPose(self):
        return self.pB.poseD.mode()

# Generator for poses
def potentialRegionPoseGenAux(pbs, obj, placeBs, graspBGen, prob, regShapes, hand, base,
                              maxPoses = 30, angles = angleList):
    def validTestFn(hyp):
        return checkValidHyp(hyp, pbsCopy, graspBGen, prob, regShapes, hand, base)
    def costFn(hyp):
        return scoreValidHyp(hyp, pbsCopy, graspBGen, prob)
    tag = 'potentialRegionPoseGen'
    pbsCopy = pbs.copy()                # so it can be modified 
    hypGen = regionPoseHypGen(pbsCopy, prob, placeBs, regShapes,
                              maxTries=2*maxPoses, angles=angles)
    for hyp in sortedHyps(hypGen, validTestFn, costFn, maxPoses, 2*maxPoses):
        if debug(tag):
            pbs.draw(prob, 'W'); hyp.conf.draw('W', 'green')
            debugMsg(tag, 'v=%s'%hyp.viol, 'weight=%s'%str(hyp.viol.weight()),
                     'pose=%s'%hyp.getPose())
        yield hyp.getPose()

def checkValidHyp(hyp, pbs, graspBGen, prob, regShapes, hand, base):
    return poseGraspable(hyp, pbs, graspBGen, prob, hand, base) and \
           feasiblePBS(hyp, pbs)

# We want to minimize this score, optimzal value is 0.
def scoreValidHyp(hyp, pbs, graspBGen, prob):
    # ignores size of shadow.
    return 5*hyp.viol.weight() + baseDist(pbs.getConf(), hyp.conf)

# A generator for hyps that meet the minimal requirement - object is
# in region and does not have permanent collisions with other objects.
def regionPoseHypGen(pbs, prob, placeBs, regShapes,
                     maxTries = 10, angles = angleList):
    def makeShadow(pB, angle):
        # placed, pbs, prob and ff drawn from environment
        shadow = pbs.objShadow(pB.obj, shadowName(pB.obj), prob, pB, ff).prim()
        if placed:
            angle = pB.poseD.mode().pose().theta
        return shadow.applyTrans(hu.Pose(0,0,0,angle))
    def makeBI(shadow, rs):
        ci = CI(shadow, rs)
        if ci: return ci.bbox()
    def placeShadow(pB, angle, rs):
        # Shadow for placement, rotated by angle
        shadow = checkCache(objShadows, (pB, angle), makeShadow)
        if placed:
            (x, y, z, angle) = pB.poseD.mode().pose().xyztTuple()
            npB = pB.modifyPoseD(hu.Pose(x, y, z, angle))
        else:
            # The CI bbox (positions that put object inside region)
            bI = checkCache(regBI, (shadow, rs), makeBI)
            if bI is None: return None, None
            z0 = bI[0,2] + clearance
            # Sampled (x,y) point
            (x, y, _, _) = next(bboxRandomCoords(bI, n=1, z=z0))
            npB = pB.modifyPoseD(hu.Pose(x, y, z0, angle))
        return npB, shadow.applyTrans(npB.poseD.mode())
    # initialize
    tag = 'potentialRegionPoseGen'
    ff = placeBs[0].faceFrames[placeBs[0].support.mode()]
    regPrims = [rs.prim() for rs in regShapes]
    print 'Region prims:'
    for rs, rp in zip(regShapes, regPrims):
        print rs, rp
    objShadows = dict()
    regBI = dict()
    clearance = 0.01
    shWorld = pbs.getShadowWorld(prob)
    # Check if the pose is specified, if so try that first
    if any(pB.poseD.mode() for pB in placeBs):
        placedBs = [pB for pB in placeBs if pB.poseD.mode()]
        placed = True
        if glob.traceGen: print '    ', tag, 'start: placed=True'
    else:
        placed = False
    # generate samples
    for trial in xrange(maxTries):
        if glob.traceGen: print '    ', tag, 'trial: placed=', placed
        # placement depends on region, pB and angle
        rs = random.choice(regPrims)
        if debug(tag):
            print '... random region prim', rs
        pB = random.choice(placedBs if placed else placeBs)
        angle = random.choice(angles)
        npB, placedShadow = placeShadow(pB, angle, rs)
        if not placedShadow: continue
        # Check conditions
        if debug(tag):
            pbs.draw(prob, 'W'); placedShadow.draw('W', 'orange')
            debugMsg(tag, 'candiate pose=%s, angle=%.2f'%(npB.poseD.mode(), angle))
        if inside(placedShadow, rs, strict=True) and \
               legalPlace(npB.obj, placedShadow, shWorld):
            debugMsg(tag, 'valid hyp pose=%s angle=%.2f'%(npB.poseD.mode(), angle))
            yield PoseHyp(npB, placedShadow)
        else:
            debugMsg(tag, 'invalid hyp pose=%s'%npB.poseD.mode())
        if placed:
            placed = False              # only try placed this once

# A shadow placement illegal if it collides with a fixed object.
def legalPlace(obj, shadow, shWorld):
    return not any(o.collides(shadow) for o in shWorld.getObjectShapes() \
                   if ((o.name() in shWorld.fixedObjects) and (o.name() != obj)))

def poseGraspable(hyp, pbs, graspBGen, prob, hand, base):
    tag = 'poseGraspable'
    for gB in graspBGen.copy():
        pB = hyp.pB
        grasp = gB.grasp.mode()
        cb = pbs.getConf()['pr2Base']
        # Try with a specified base
        c, ca, v = next(potentialGraspConfGen(pbs, pB, gB, None, hand, (base or cb), prob, nMax=1),
                        (None,None,None))
        if v is None:
            # try without specifying base
            c, ca, v = next(potentialGraspConfGen(pbs, pB, gB, None, hand, None, prob, nMax=1),
                            (None,None,None))
        if v:
            hyp.conf = ca
            hyp.viol = v
            pbs.draw(prob, 'W'); pB.shape(pbs).draw('W', 'green'); ca.draw('W', 'green')
            debugMsg(tag, 'candiate won pose=%s, grasp=%s'%(pB.poseD.mode(), gB.grasp.mode()))
            return hyp
        else:
            debugMsg(tag, 'candiate failed pose=%s, grasp=%s'%(pB.poseD.mode(), gB.grasp.mode()))

def feasiblePBS(hyp, pbs):
    if pbs.conditions:
        print '*** Testing feasibibility with %d conditions'%(len(pbs.conditions))
        pbs.updatePlaceB(hyp.pB)
        feasible = pbs.feasible()   # check conditioned fluents
        print '*** feasible =>', feasible
        return feasible
    else:
        return True
