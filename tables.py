from cpython cimport bool
import math
cimport geom
import planGlobals as glob
reload(glob)
from glob import debug, debugMsg

# Can be cimports
import numpy as np
import util
import shapes
import pointClouds as pc

windowName = 'MAP'

# Table is the refObj, centered on origin
def tuple bestTable(zone, table, pointCloud, exclude,
                    res=0.01, zthr = 0.03, angles = []):
    
    # cdef int minTablePoints, p, size, score, bestScore
    # cdef float height, radius, thrHi, thrLo, angle
    # cdef list good, below
    # cdef np.ndarray[np.float64_t, ndim=1] pt, center
    # cdef np.ndarray[np.float64_t, ndim=2] bb, points, eye, badPoints, goodPoints, badOffset, bbGood
    # cdef bool inside
    # cdef util.Transform headTrans
    # cdef util.Pose pose, bestPose
    
    minTablePoints = int(glob.minTableDim / glob.cloudPointsResolution)
    height = table.zRange()[1] - table.zRange()[0]
    bb = table.bbox()
    radius = 0.75*math.sqrt((bb[1,0] - bb[0,0])**2 + (bb[1,1] - bb[0,1])**2)
    good = []
    below = []
    thrHi = height + zthr
    thrLo = height - zthr
    points = pointCloud.vertices
    headTrans = pointCloud.headTrans
    for p in points.shape[1]:
        if thrLo <= points[2,p] <= thrHi:       # points on the plane
            pt = points[:,p]
            if not zone.containsPt(pt):
                continue
            inside = False
            for obj in exclude:
                if obj.containsPt(pt):
                    inside = True
                    break
            if not inside:
                good.append(p)          
        elif -0.25 <= points[2,p] < thrLo: # ignore the NaN => 10m points
            below.append(p)             # points below the plane
    if len(good) < minTablePoints:
        return 0, None
    eye = headTrans.eye.matrix.reshape(4,1)
    # project bad points (along vector to eye) to the plane
    badPoints = points[:,below]
    badOffset = eye - badPoints
    for p in len(bad):
        badOffset[:,p] *= (height-badPoints[2,p])/(eye[2,0] - badPoints[2,p])
    badOffset += badPoints

    goodPoints = points[:, good]
    bbGood = geom.bbox(goodPoints)      # Johnny, bbGood!
    radius += max(bbGood[1,0] - bbGood[0,0], bbGood[1,1] - bbGood[0,1])/2.
    center = geom.bboxCenter(bbGood)

    if debug('tables'):
        zone.draw(windowName, 'purple')
        pc.Scan(headTrans, None, goodPoints).draw(windowName, 'green')
        pc.Scan(headTrans, None, belowPoints).draw(windowName, 'orange')
        pc.Scan(headTrans, None, badPoints).draw(windowName, 'red')
        debugMsg('tables', 'good=green, below=organge, bad=red')

    # create the summed area table
    size = 1+2*int(math.ceil(radius/res))
    summed = (np.zeros((size, size), dtype=np.int_t),
              np.zeros((size, size), dtype=np.int_t))
    # Try rotations to find best fit placement of the table
    bestScore = -1000
    if not angles:
        angles = anglesList(30)
    for angle in angles:
        (score, pose) = bestTablePose(table, center, angle,
                                      goodPoints, badPoints, summed, res, size)
        if score > bestScore:
            bestScore = score
            bestPose = pose
    if bestScore > minTablePoints:
        return bestScore, table.applyPose(bestPose)
    else:
        return bestScore, None
    
def fillSummed(centerPoseInv, ci, pts, summed, res, size):

    # cdef int pi, i, j, p, npts
    # cdef np.ndarray[np.int_t, ndim=1] si, si_1, ind
    # cdef np.ndarray[np.float64_t, ndim=2] centered
    
    pi = 0                              # index into points
    centered = np.dot(centerPoseInv.matrix, pts)
    centered /= res
    pc = np.array([ci[0], ci[1]], dtype=np.int_t) + np.rint(centered[:2]) # x,y ints
    legal = np.intersect1d(np.intersect1d(np.where(pc[0] >= 0), np.where(pc[0] < size)),
                           np.intersect1d(np.where(pc[1] >= 0), np.where(pc[1] < size)))
    points = pc[legal]
    points = np.c_[legal, np.array([size,size])] # guard
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
            if si_1 and j > 0:
                si[j] = p + si_1[j] + si[j-1] - si_1[j-1]
            elif j > 0:
                si[j] = p + si[j-1]
            else:
                si[j] = p

def tuple scoreTable1(summed, i, j, dimI, dimJ):
    # cdef list scoreA, scoreB, scoreC, scoreD
    scoreA = summed[i,j]
    scoreC = summed[i+dimI,j+dimJ]
    scoreB = summed[i,j+dimJ]
    scoreD = summed[i+dimI,j]
    return scoreC + scoreA - scoreB - scoreD

def tuple scoreTable(summed, i, j, dimI, dimJ):
    return (scoreTable1(summed[0], i, j, dimI, dimJ),
            scoreTable1(summed[1], i, j, dimI, dimJ))

def tuple bestTablePose(table, center, angle, good, bad, summed, res, size):

    # cdef util.Pose centerPose, centerPoseInv, pose, cpose
    # cdef tuple ci, bestPlace
    # cdef np.ndarray[np.float64_t, ndim=2] bb
    # cdef int dimI, dimJ, bestScore
    # cdef int i, j, score, shrink, maxShrink, scoreGood, scoreBad
    
    centerPose = util.Pose(center.x, center.y, 0.0, angle)
    centerPoseInv = centerPose.inverse()
    ci = ((size-1)/2, (size-1)/2)
    fillSummed(centerPoseInv, ci, good, summed[0], res, size)
    fillSummed(centerPoseInv, ci, bad, summed[1], res, size)
    bb = table.bbox()
    dimI = int(math.floor((bb[1,0] - bb[0,0])/res))
    dimJ = int(math.floor((bb[1,1] - bb[0,1])/res))
    bestScore = -1000
    bestPlace = None
    maxShrink = int(math.ceil(glob.tableMaxShrink / glob.cloudPointsResolution))
    for i in range(size - dimI):
        for j in range(size - dimJ):
            for shrink in range(maxShrink):
                scoreGood, scoreBad = scoreTable(summed, i, j, dimI-shrink, dimJ-shrink)
                score = scoreGood - glob.tableBadWeight*scoreBad
                if score > bestScore:
                    bestScore = score
                    bestScoreGood = scoreGood
                    bestScoreBad = scoreBad
                    bestPlace = (i, j)
                    bestShrink = shrink
    # displacement pose for table center
    pose = util.Pose((bestPlace[0] + int((dimI-bestShrink)/2.0) - ci[0])*res,
                     (bestPlace[1] + int((dimJ-bestShrink)/2.0) - ci[1])*res,
                     0, 0)
    cpose = centerPose.compose(pose)
    if debug('tables'):
        print 'bestPlace', bestPlace, 'bestShrink', bestShrink,
        print 'score', bestScore, 'scoreGood', bestScoreGood, 'scoreBad', bestScoreBad
    else:
        print '.',
    return (bestScore, cpose)

def list anglesList(n = 8):
    # cdef float delta
    delta = math.pi/n
    return [i*delta for i in xrange(n)]
