import random
from pr2Util import checkCache
from pr2GenUtils import sortedHyps, baseDist

# TODO: Should pick relevant orientations... or more samples.
angleList = [-math.pi/2. -math.pi/4., 0.0, math.pi/4, math.pi/2]

# (regShapeName, obj, hand, grasp) : (n_attempts, [relPose, ...])
regionPoseCache = {}

# Generate poses
# Requirements: inside region, graspable, pbs is feasible
# Preferences:  bigger shadows, fewer violations, grasp conf close to current conf
def potentialRegionPoseGen(pbs, obj, placeB, graspB, prob, regShapes, hand, base,
                           maxPoses = 30):
    if traceGen:
        print ' **', 'potentialRegionPoseGen', placeB.poseD.mode(), graspB.grasp.mode(), hand
    # Consider a range of variances
    maxVar = placeB.poseD.var
    minVar = pbs.domainProbs.obsVarTuple
    pBs = [placeB.modifyPoseD(var=medVar) \
           for medVar in interpolateVars(maxVar, minVar, 4)]
    count = 0
    for pose in potentialRegionPoseGenAux(pbs, obj, pBs, graspB, prob, regShapes,
                                          hand, base, maxPoses):
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
def potentialRegionPoseGenAux(pbs, obj, placeBs, graspB, prob, regShapes, hand, base,
                              maxPoses = 30):
    def validTestFn(hyp):
        return checkValidHyp(hyp, pbsCopy, graspB, prob, regShapes, hand, base)
    def costFn(hyp):
        return scoreValidHyp(hyp, pbsCopy, graspB, prob)
    tag = 'potentialRegionPoseGen'
    pbsCopy = pbs.copy()                # so it can be modified 
    hypGen = regionPoseHypGen(pbsCopy, probs, placeBs, regShapes)
    for hyp in sortedHyp(hypGen, validTestFn, costFn, maxPoses, 2*maxPoses):
        if debug(tag):
            pbs.draw(prob, 'W'); hyp.conf.draw('W', 'green')
            debugMsg(tag, 'v=%s'%hyp.viol, 'weight=%s'%str(hyp.viol.weight()),
                     'pose=%s'%hyp.getPose(), 'grasp=%s'%graspB.grasp.mode())
        yield hyp.getPose()

def checkValidHyp(hyp, pbs, graspB, prob, regShapes, hand, base):
    return poseGraspable(hyp, pbs, graspB, prob, hand, base) and \
           feasiblePBS(hyp, pbs)

# We want to minimize this score, optimzal value is 0.
def scoreValidHyp(hyp, pbs, graspB, prob):
    # ignores size of shadow.
    return hyp.viol.weight() + baseDist(pbs.getConf(), hyp.conf)

# A generator for hyps that meet the minimal requirement - object is
# in region and does not have permanent collisions with other objects.
def regionPoseHypGen(pbs, prob, placeBs, regShapes, maxTries = 10):
    def makeShadow(pB, angle):
        # placed, pbs, prob and ff drawn from environment
        shadow = pbs.objShadow(pB.obj, shadowName(pB.obj), prob, pB, ff).prim()
        if not placed:
            shadow = shadow.applyTrans(hu.Pose(0,0,0,angle))
        return shadow
    def makeBI(shadow, rs):
        return CI(shadow, rs).bbox()
    def placeShadow(pB, angle, rs):
        # Shadow for placement
        shadow = checkCache(objShadows, (pB, angle), makeShadow)
        if placed:
            return shadow
        else:
            # The CI bbox (positions that put object inside region)
            bI = checkCache(regBI, (shadow, rs), makeBI)
            if bI is None: continue
            z0 = bI[0,2] + clearance
            # Sampled (x,y) point
            pt = next(bboxRandomCoords(bI.bbox(), n=1, z=z0))
            return shadow.applyTrans(hu.Pose(x, y, 0, 0))
    # initialize 
    ff = placeB.faceFrames[placeBs[0].support.mode()]
    regPrims = [rs.prim() for rs in regShapes]
    objShadows = dict()
    regBI = dict()
    clearance = 0.01
    shWorld = pbs.getShadowWorld(prob)
    # Check if the pose is specified, if so try that first
    if any(pB.poseD.mode() for pB in placeBs):
        placedBs = [pB for pB in placeBs if pB.poseD.mode()]
        placed = True
    # generate samples
    for trial in xrange(maxTries):
        # placement depends on region, pB and angle
        rs = random.choice(regPrims)
        pB = random.choice(placedBs if placed else placeBs)
        angle = random.choice(angleList)
        placedShadow = placeShadow(pB, angle, rs)
        # Check conditions
        if inside(placedShadow, rs) and \
               legalPlace(pB.obj, placedShadow, shWorld):
            yield PoseHyp(pB, placedShadow)
        if placed:
            placed = False              # only try placed this once

# A shadow placement illegal if it collides with a fixed object.
def legalPlace(obj, shadow, shadowWorld):
    return not any(o.collides(shadow) for o in shWorld.getObjectShapes() \
                   if ((o.name() in shWorld.fixedObjects) and (o.name() != obj)))

def poseGraspable(hyp, pbs, graspB, prob, hand, base):
    for gB in graspGen(pbs, graspB):
        grasp = gB.grasp.mode()
        cb = pbs.getConf()['pr2Base']
        # Try with a specified base
        c, ca, v = next(potentialGraspConfGen(pbs, pB, gB, None, hand, (base or cb), prob, nMax=1),
                        (None,None,None))
        if v is None:
            # try without specifying base
            c, ca, v = next(potentialGraspConfGen(pbs, pB, gB, None, hand, None, prob, nMax=1),
                            (None,None,None))
        debugMsg('poseGraspable', 'v=%s'%v, 'pose=%s'%hyp.getPose(), 'grasp=%s'%grasp)
        if v:
            hyp.conf = ca
            return hyp

def feasiblePBS(hyp, pbs):
    if pbs.conditions:
        pbs = pbs.updatePlaceB(hyp.pB)
        return pbs.feasible()           # check conditioned fluents
    else:
        return True
