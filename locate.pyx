import math
import time
import planGlobals as glob
reload(glob)
from planGlobals import debugMsg, debug

# Can be cimports
cimport numpy as np
import numpy as np
cimport util
import util
cimport geom
import geom
cimport shapes
cimport pointClouds as pc
from cpython cimport bool

import windowManager3D as wm
windowName = 'MAP'
tiny = 1.0e-6

# Shape is the refObj, centered on origin.  We'll be matching the
# individual bbox of the parts.
def bestShape(zone, shape, pointCloud, exclude,
                    minPoints = 50, res=0.01, zthr = 0.03, angles = []):
    
    cdef int p, size, score, bestScore
    cdef float height, radius, thrHi, thrLo, angle
    cdef list good, below
    cdef np.ndarray[np.float64_t, ndim=1] pt, center
    cdef np.ndarray[np.float64_t, ndim=2] bb, points, eye, badPoints, goodPoints, badOffset, bbGood
    cdef bool inside
    cdef util.Transform headTrans
    cdef util.Transform pose, bestPose
    
    height = shape.zRange()[1] - shape.zRange()[0]
    bb = shape.bbox()
    if debug('locate'): print 'shape bbox=\n', bb
    radius = 0.75*math.sqrt((bb[1,0] - bb[0,0])**2 + (bb[1,1] - bb[0,1])**2)
    good = []
    below = []
    thrHi = height + zthr
    thrLo = height - zthr
    points = pointCloud.vertices
    headTrans = pointCloud.headTrans
    for p in range(1, points.shape[1]):   # index 0 is eye
        if thrLo <= points[2,p] <= thrHi: # points on the plane
            pt = points[:,p]
            if not np.all(np.dot(zone.planes(), pt) <= tiny):
                continue
            inside = False
            for obj in exclude:
                if np.all(np.dot(obj.planes(), pt) <= tiny):
                    inside = True
                    break
            if not inside:
                good.append(p)          
        elif -0.25 <= points[2,p] < thrLo: # ignore the NaN => 10m points
            below.append(p)             # points below the plane

    if debug('locate'):
        print 'len(good)=', len(good), 'len(below)=', len(below)
    if len(good) < minPoints:
        return 0, None
    eye = headTrans.point().matrix.reshape(4,1)
    # project bad points (along vector to eye) to the plane
    badPoints = points[:,below]
    badOffset = eye - badPoints
    for p in range(len(below)):
        badOffset[:,p] *= (height-badPoints[2,p])/(eye[2,0] - badPoints[2,p])
    badPoints += badOffset

    goodPoints = points[:, good]
    bbGood = geom.vertsBBox(goodPoints, None) # Johnny, bbGood!
    radius += max(bbGood[1,0] - bbGood[0,0], bbGood[1,1] - bbGood[0,1])/2.
    center = geom.bboxCenter(bbGood)

    if debug('locate'):
        zone.draw(windowName, 'purple')
        pc.Scan(headTrans, None, goodPoints).draw(windowName, 'green')
        pc.Scan(headTrans, None, badPoints).draw(windowName, 'red')
        debugMsg('locate', 'good=green, bad=red')

    # create the summed area shape
    size = 1+2*int(math.ceil(radius/res))
    print 'size=', size
    summed = (np.zeros((size, size), dtype=np.int),
              np.zeros((size, size), dtype=np.int))
    # Try rotations to find best fit placement of the shape
    bestScore = -1000
    if not angles:
        angles = anglesList(30)
    for angle in angles:
        (score, pose) = bestShapePose(shape, center, angle,
                                      goodPoints, badPoints, summed, res, size)
        if score > bestScore:
            bestScore = score
            bestPose = pose
        if debug('locate'):
            shape.applyTrans(pose).draw(windowName, 'pink')
            shape.applyTrans(bestPose).draw(windowName, 'blue')
            wm.getWindow(windowName).update()
    if bestScore > minPoints:
        return bestScore, shape.applyTrans(bestPose)
    else:
        return bestScore, None
    
cdef fillSummed(util.Transform centerPoseInv, tuple ci,
               np.ndarray[np.float64_t, ndim=2] pts,
               np.ndarray[np.int_t, ndim=2] summed,
               float res, int size):

    cdef int pi, i, j, p, npts
    cdef np.ndarray[np.int_t, ndim=1] si, si_1, ind
    cdef np.ndarray[np.float64_t, ndim=2] centered
    
    pi = 0                              # index into points
    centered = np.dot(centerPoseInv.matrix, pts)
    centered /= res
    pc = np.array([ci[0], ci[1]], dtype=np.int).reshape(2,1) + np.rint(centered[:2]) # x,y ints
    legal = np.where((pc[0] >= 0) & (pc[0] < size) & (pc[1] >= 0) & (pc[1] < size))
    points = pc[:, legal[0]]
    points = np.c_[points, np.array([size,size]).reshape(2,1)] # guard
    ind = np.lexsort((points[1], points[0])) # sort by x, then by y
    npts = points.shape[1]
    for i in range(size):
        si = summed[i]
        si_1 = summed[i-1] if i > 0 else None
        for j in range(size):
            p = 0
            while (pi < npts) and (points[0,ind[pi]] == i and points[1,ind[pi]] == j):
                p = 1                   # found point, only count once.
                pi += 1                 # advance index
            if si_1 != None and j > 0:
                si[j] = p + si_1[j] + si[j-1] - si_1[j-1]
            elif j > 0:
                si[j] = p + si[j-1]
            else:
                si[j] = p

cdef int scoreShape1(np.ndarray[np.int_t, ndim=2] summed, int i, int j, int dimI, int dimJ):
    cdef int scoreA, scoreB, scoreC, scoreD
    scoreA = summed[i,j]
    scoreC = summed[i+dimI,j+dimJ]
    scoreB = summed[i,j+dimJ]
    scoreD = summed[i+dimI,j]
    return scoreC + scoreA - scoreB - scoreD

cdef tuple scoreShape(tuple summed, int i, int j, int dimI, int dimJ):
    return (scoreShape1(summed[0], i, j, dimI, dimJ),
            scoreShape1(summed[1], i, j, dimI, dimJ))

cpdef tuple bestShapePose(shapes.Thing shape, np.ndarray[np.float64_t, ndim=1 ]center,
                          float angle,
                          np.ndarray[np.float64_t, ndim=2] good,
                          np.ndarray[np.float64_t, ndim=2] bad,
                          tuple summed, float res, int size):

    cdef util.Transform centerPose, centerPoseInv, pose, cpose
    cdef tuple ci, bestPlace
    cdef np.ndarray[np.float64_t, ndim=2] bb
    cdef int dimI, dimJ, bestScore
    cdef int i, j, score, shrink, maxShrink, scoreGood, scoreBad
    
    centerPose = util.Pose(*[center[0], center[1], 0., angle])
    centerPoseInv = centerPose.inverse()
    ci = ((size-1)/2, (size-1)/2)
    fillSummed(centerPoseInv, ci, good, summed[0], res, size)
    fillSummed(centerPoseInv, ci, bad, summed[1], res, size)
    bb = shape.bbox()
    dimI = int(math.floor((bb[1,0] - bb[0,0])/res))
    dimJ = int(math.floor((bb[1,1] - bb[0,1])/res))
    bestScore = -1000
    bestPlace = None
    maxShrink = int(math.ceil(glob.shapeMaxShrink / glob.cloudPointsResolution))
    for i in range(size - dimI):
        for j in range(size - dimJ):
            for shrink in range(maxShrink):
                scoreGood, scoreBad = scoreShape(summed, i, j, dimI-shrink, dimJ-shrink)
                score = scoreGood - glob.shapeBadWeight*scoreBad
                if score > bestScore:
                    bestScore = score
                    bestScoreGood = scoreGood
                    bestScoreBad = scoreBad
                    bestPlace = (i, j)
                    bestShrink = shrink
    # displacement pose for shape center
    pose = util.Pose((bestPlace[0] + int((dimI-bestShrink)/2.0) - ci[0])*res,
                     (bestPlace[1] + int((dimJ-bestShrink)/2.0) - ci[1])*res,
                     0, 0)
    cpose = centerPose.compose(pose)
    if debug('locate'):
        print 'bestPlace', bestPlace, 'bestShrink', bestShrink,
        print 'score', bestScore, 'scoreGood', bestScoreGood, 'scoreBad', bestScoreBad
    else:
        print '.',
    return (bestScore, cpose)

def anglesList(n = 8):
    # cdef float delta
    delta = math.pi/n
    return [i*delta for i in xrange(n)]

# obsTargets is dict: {name:ObjPlaceB or None, ...}
def getShapeDetections(world, obsPlaceBs, pointCloud):
    shapes = []
    exclude = []
    # A zone of interest
    zone = shapes.BoxAligned(np.array([(0, -2, 0), (3, 2, 1.5)]), None)
    allAngles = anglesList(30)
    for placeB in obsPlaceBs:
        objName = placeB.obj
        if 'table' in objName:
            startTime = time.time()
            shape = world.getObjectShapeAtOrigin(objName)
            zr = shape.zRange()
            height = 0.5*(zr[1] - zr[0])
            dz = height - shape.origin().matrix[2,3]
            shape = shape.applyTrans(util.Pose(0, 0, dz, 0))
            # Use the mode and variance to limit angles to consider.
            pose = placeB.poseD.mode().pose()
            var = placeB.poseD.variance()
            std = var[-1]**0.5          # std for angles
            angles = [angle for angle in allAngles if \
                      abs(util.angleDiff(angle, pose.theta)) <= 4*std] + [pose.theta]
            angles.sort()
            if debug('locate'):
                print 'Candidate shape angles:', angles
            score, detection = \
                   bestShape(zone, shape, pointCloud, exclude,
                             angles = angles, zthr = 0.05)
            print 'Shape detection', detection, 'with score', score
            print 'Running time for shape detections =',  time.time() - startTime
            if detection:
                shapes.append((score, detection))
                exclude.append(detection)
                detection.draw('MAP', 'blue')
                debugMsg('locate', 'Detection for shape=%s'%objName)
    return shapes
