import pdb
import random
import math
import hu
from cspace import CI
from geom import bboxIsect
from pr2Util import checkCache, shadowName, objectName, inside, bboxRandomCoords
from pr2GenUtils import sortedHyps, baseDist, graspGen, feasiblePBS, inflatedBS
from pr2GenGrasp import potentialGraspConfGen
from dist import DDist
import planGlobals as glob
from traceFile import tr, debug, debugMsg

# TODO: Should pick relevant orientations... or more samples.
angleList = [-math.pi/2. -math.pi/4., 0.0, math.pi/4, math.pi/2]

# (regShapeName, obj, hand, grasp) : (n_attempts, [relPose, ...])
regionPoseCache = {}

# (obj, pose) : (score, grasp)
poseGraspCache = {}

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
    newBS = inflatedBS(pbs, prob)
    for pose in potentialRegionPoseGenAux(newBS, obj, pBs, graspBGen, prob, regShapes,
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
    def __init__(self, pB, shadow, region):
        self.pB = pB
        self.shadow = shadow
        self.region = region
        self.conf = None
        self.viol = None
        self.gB = None
        self.cost = None
    def getPose(self):
        return self.pB.poseD.mode()
    def __str__(self):
        return 'PoseHyp(%s,%s,%s)'%(self.pB.poseD.mode(), self.region.name(), self.viol)
    __repr__ = __str__
    
# Generator for poses
def potentialRegionPoseGenAux(pbs, obj, placeBs, graspBGen, prob, regShapes, hand, base,
                              maxPoses = 30, angles = angleList):
    def validTestFn(hyp):
        return checkValidHyp(hyp, pbs, graspBGen, prob, regShapes, hand, base)
    def costFn(hyp):
        return scoreValidHyp(hyp, pbs, graspBGen, prob)
    tag = 'potentialRegionPoseGenFinal'
    hypGen = regionPoseHypGen(pbs, prob, placeBs, regShapes,
                              maxTries=2*maxPoses, angles=angles)
    for hyp in sortedHyps(hypGen, validTestFn, costFn, maxPoses, 2*maxPoses,
                          size=(2 if glob.inHeuristic else 20)):
        if debug(tag):
            pbs.draw(prob, 'W')
            hyp.conf.draw('W', 'magenta')
            hyp.pB.makeShadow(pbs,prob).draw('W', 'magenta')
            debugMsg(tag, 'obj=%s'%obj, 'r=%s'%hyp.region, 'v=%s'%hyp.viol, 'weight=%s'%str(hyp.viol.weight()),
                     'cost=%s'%hyp.cost, 'pose=%s'%hyp.getPose())
        if not feasiblePBS(hyp.pB, pbs): continue # check PBS
        pose = hyp.getPose()
        key = (obj, pose)
        entry = poseGraspCache.get(key, None)
        if entry:
            entry.append((hyp.cost, hyp.gB))
        else:
            poseGraspCache[key] = [(hyp.cost, hyp.gB.grasp.mode())]
        yield pose

def checkValidHyp(hyp, pbs, graspBGen, prob, regShapes, hand, base):
    return poseGraspable(hyp, pbs, graspBGen, prob, hand, base)

# We want to minimize this score, optimal value is 0.
def scoreValidHyp(hyp, pbs, graspBGen, prob):
    # ignores size of shadow.
    obj = hyp.pB.obj
    shWorld = pbs.getShadowWorld(prob)
    placeWeight = 0.
    placeCollisions = []
    for o in shWorld.getObjectShapes():
        name = o.name()
        # Don't place it in the same place it is now, so don't ignore "self collisions"
        # if name in shWorld.fixedObjects or objectName(name) == objectName(obj): continue
        if o.collides(hyp.shadow):
            placeCollisions.append(o.name())
            if 'shadow' in name:
                placeWeight += 0.5
            else:
                placeWeight += 1.0
    confWeight = hyp.viol.weight()

    placeWeight *= 5
    confWeight *= 5

    objd = placeBDist(pbs.getPlaceB(hyp.pB.obj), hyp.pB)
    based = baseDist(pbs.getConf(), hyp.conf)
    if debug('sortedPoseHyps'):
        pbs.draw(prob, 'W')
        pbs.getConf().draw('W', 'blue')
        hyp.conf.draw('W', 'pink')
        if confWeight > 0: print 'confViol', hyp.viol
        if placeWeight > 0: print 'placeCollide', placeCollisions
        print 'objd=', objd, 'based=', based, 'conf=', confWeight, 'place', placeWeight
        raw_input('score=%f'%(objd+based+confWeight+placeWeight))
    hyp.cost = objd + based + confWeight + placeWeight
    return hyp.cost

def placeBDist(pB1, pB2):
    if pB1 and pB2:
        p1 = pB1.poseD.mode().pose()
        p2 = pB2.poseD.mode().pose()
        return p1.totalDist(p2)         # angleScale = 1
    else:
        return 0.

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
    if debug(tag):
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
    # Build distribution based on available area
    regDist = regionDDist(regPrims, shWorld)
    # generate samples
    for trial in xrange(maxTries):
        if glob.traceGen: print '    ', tag, 'trial: placed=', placed
        # placement depends on region, pB and angle
        rs = regDist.draw()
        if debug(tag):
            print '... random region prim', rs
        pB = random.choice(placedBs if placed else placeBs)
        angle = random.choice(angles)
        npB, placedShadow = placeShadow(pB, angle, rs)
        if not placedShadow: continue
        # Check conditions
        if debug(tag):
            pbs.draw(prob, 'W'); placedShadow.draw('W', 'orange')
            debugMsg(tag, 'candidate pose=%s, angle=%.2f'%(npB.poseD.mode(), angle))
        if inside(placedShadow, rs, strict=True) and \
               legalPlace(npB.obj, placedShadow, shWorld):
            debugMsg(tag, 'valid hyp pose=%s angle=%.2f'%(npB.poseD.mode(), angle))
            yield PoseHyp(npB, placedShadow, rs)
        else:
            debugMsg(tag, 'invalid hyp pose=%s'%npB.poseD.mode())
        if placed:
            placed = False              # only try placed this once

# A shadow placement illegal if it collides with a fixed object.
def legalPlace(obj, shadow, shWorld):
    return not any(o.collides(shadow) for o in shWorld.getObjectShapes() \
                   if ((o.name() in shWorld.fixedObjects) and (o.name() != obj)))

def regionDDist(regPrims, shWorld):
    def bboxXYArea(bb): return max(0., (bb[1][0] - bb[0][0])*(bb[1][1] - bb[0][1]))
    def overlapArea(bb1, bb2): return bboxXYArea(bboxIsect([bb1, bb2]))
    def availableArea(r):
        rbb = r.bbox()
        return max(1.0e-4, bboxXYArea(rbb) - \
                   sum([overlapArea(rbb, ob) for ob in obsts]))
    obsts = [o.bbox() for o in shWorld.getNonShadowShapes()]
    return DDist({r : availableArea(r) for r in regPrims}).normalize()

def poseGraspable(hyp, pbs, graspBGen, prob, hand, base):
    tag = 'poseGraspable'
    for gB in graspBGen.copy():
        pB = hyp.pB
        grasp = gB.grasp.mode()
        cb = pbs.getConf().baseConf()
        # Try with a specified base
        c, ca, v = next(potentialGraspConfGen(pbs, pB, gB, None, hand, (base or cb), prob, nMax=1),
                        (None,None,None))
        if v is None:
            # try without specifying base
            c, ca, v = next(potentialGraspConfGen(pbs, pB, gB, None, hand, None, prob, nMax=1),
                            (None,None,None))
        if v:
            hyp.conf = c
            hyp.viol = v
            hyp.gB = gB
            pbs.draw(prob, 'W'); pB.shape(pbs).draw('W', 'green'); ca.draw('W', 'green')
            debugMsg(tag, 'candidate won pose=%s, grasp=%s'%(pB.poseD.mode(), gB.grasp.mode()))
            return True
        else:
            debugMsg(tag, 'candidate failed pose=%s, grasp=%s'%(pB.poseD.mode(), gB.grasp.mode()))
    return False
