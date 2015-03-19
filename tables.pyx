cimport util
cimport kContents as KC
cimport objModels as om
from cpython cimport bool

import math
import planGlobals as glob
reload(glob)

windowName = 'MAP'
#windowName = 'PR2'

cpdef list emptyPair (int i, int j):
    return [0, 0]

# Table is the refObj, centered on origin
cpdef tuple bestTable(om.Polygon zone, om.Object table, KC.Scan scan, list exclude,
                      float res=0.01, float zthr = 0.03, list angles = [],
                      bool debug = False):
    name = 'bestTable'

    cdef int minTablePoints = int(glob.minTableDim / glob.cloudPointsResolution)
    cdef float height = table.zRange()[1] - table.zRange()[0]
    cdef tuple bb = table.bbox()
    cdef float radius = 0.75*math.sqrt((bb[1][0] - bb[0][0])**2 + (bb[1][1] - bb[0][1])**2)
    cdef list good = []
    cdef list below = []
    cdef float thrHi = height + zthr
    cdef float thrLo = height - zthr
    cdef util.Point p
    cdef bool inside
    for p in scan.points:
        if thrLo <= p.z <= thrHi:       # points on the plane
            if not zone.contains(p):
                continue
            inside = False
            for obj in exclude:
                if obj.contains(p):
                    inside = True
                    break
            if not inside:
                good.append(p)          
        elif -0.25 <= p.z < thrLo:      # ignore the NaN => 10m points
            below.append(p)             # points below the plane
    if len(good) < minTablePoints:
        return 0, None
    cdef util.Point eye = scan.eye
    # project (along vector to eye) to the plane
    cdef list bad = [p+((height - p.z)/(eye.z - p.z))*(eye-p) for p in below]
    cdef tuple bbGood = om.bbox(good)
    radius += max(bbGood[1][0] - bbGood[0][0], bbGood[1][1] - bbGood[0][1])/2
    cdef util.Point center = om.bboxCenter(bbGood)
    if debug:
        zone.draw(windowName, 'purple')
        KC.Scan(scan.headTrans, good, (0.3, 0.2, 0.2, 5, 30)).draw(windowName, 'green')
        KC.Scan(scan.headTrans, below, (0.3, 0.2, 0.2, 5, 30)).draw(windowName, 'orange')
        KC.Scan(scan.headTrans, bad, (0.3, 0.2, 0.2, 5, 30)).draw(windowName, 'red')
    # create the summed area table
    cdef int size = 1+2*int(math.ceil(radius/res))
    cdef list summed = util.make2DArrayFill(size, size, emptyPair)
    # Try rotations to find best fit placement of the table
    cdef int bestScore = -1000
    cdef util.Pose pose, bestPose
    cdef float angle
    cdef int score
    if not angles:
        angles = anglesList(30)
    for angle in angles:
        (score, pose) = bestTablePose(table, center, angle,
                                      good, bad, summed, res, size, debug)
        if score > bestScore:
            bestScore = score
            bestPose = pose
    if debug: raw_input('Continue?')
    if bestScore > minTablePoints:
        return bestScore, table.applyPose(bestPose)
    else:
        return bestScore, None

cpdef fillSummed(util.Pose centerPoseInv, tuple ci, list good, list bad, list summed, float res, int size):
    cdef list ngood = sorted([coord(p, centerPoseInv, res, ci, size) for p in good])
    cdef list nbad = sorted([coord(p, centerPoseInv, res, ci, size) for p in bad])
    ngood.append((size, size))          # guard
    nbad.append((size, size))           # guard
    fillSummedIndex(ngood, 0, summed, size)
    fillSummedIndex(nbad, 1, summed, size)

cpdef fillSummedIndex(list points, int id, list summed, int size):
    cdef int pi = 0                     # index into points
    cdef int i, j, p
    cdef list si, si_1, sij
    for i in xrange(size):
        si = summed[i]
        si_1 = summed[i-1] if i > 0 else None
        for j in xrange(size):
            sij = si[j]
            p = 0
            while (points[pi][0] == i and points[pi][1] == j):
                p = 1                   # found point, only count once.
                pi += 1                 # advance index
            if si_1 and j > 0:
                sij[id] = p + si_1[j][id] + si[j-1][id] - si_1[j-1][id]
            elif j > 0:
                sij[id] = p + si[j-1][id]
            else:
                sij[id] = p

cpdef tuple scoreTable(list summed, int i, int j, int dimI, int dimJ):
    cdef list scoreA, scoreB, scoreC, scoreD
    cdef int scoreGood, scoreBad
    scoreA = summed[i][j]
    scoreC = summed[i+dimI][j+dimJ]
    scoreB = summed[i][j+dimJ]
    scoreD = summed[i+dimI][j]
    scoreGood = scoreC[0] + scoreA[0] - scoreB[0] - scoreD[0]
    scoreBad = scoreC[1] + scoreA[1] - scoreB[1] - scoreD[1]
    return scoreGood, scoreBad    

cpdef tuple bestTablePose(om.Object table, util.Point center, float angle,
                          list good, list bad, list summed, float res, int size,
                          bool debug):
    cdef util.Pose centerPose = util.Pose(center.x, center.y, 0.0, angle)
    cdef util.Pose centerPoseInv = centerPose.inverse()
    cdef tuple ci = ((size-1)/2, (size-1)/2)
    fillSummed(centerPoseInv, ci, good, bad, summed, res, size)
    cdef tuple bb = table.bbox()
    cdef int dimI = int(math.floor((bb[1][0] - bb[0][0])/res))
    cdef int dimJ = int(math.floor((bb[1][1] - bb[0][1])/res))
    cdef int bestScore = -1000
    cdef tuple bestPlace = None
    cdef int i, j, score, shrink, maxShrink, scoreGood, scoreBad
    maxShrink = int(math.ceil(glob.tableMaxShrink / glob.cloudPointsResolution))
    for i in xrange(size - dimI):
        for j in xrange(size - dimJ):
            for shrink in xrange(maxShrink):
                scoreGood, scoreBad = scoreTable(summed, i, j, dimI-shrink, dimJ-shrink)
                score = scoreGood - glob.tableBadWeight*scoreBad
                if score > bestScore:
                    bestScore = score
                    bestScoreGood = scoreGood
                    bestScoreBad = scoreBad
                    bestPlace = (i, j)
                    bestShrink = shrink
    # displacement pose for table center
    cdef util.Pose pose = util.Pose((bestPlace[0] + int((dimI-bestShrink)/2.0) - ci[0])*res,
                                    (bestPlace[1] + int((dimJ-bestShrink)/2.0) - ci[1])*res,
                                    0, 0)
    cdef util.Pose cpose = centerPose.compose(pose)
    if debug:
        print 'bestPlace', bestPlace, 'bestShrink', bestShrink, 'score', bestScore, 'scoreGood', bestScoreGood, 'scoreBad', bestScoreBad
        # guess = table.applyPose(cpose)
        # guess.draw('PR2', 'brown')
        # raw_input('Guess with score=%f'%bestScore)
    else:
        print '.',
    return (bestScore, cpose)
    
cdef tuple coord (util.Point pt, util.Pose trans, float res, tuple ci, int size):
    npt = trans.applyToPoint(pt)
    x = ci[0]+int(round(npt.x/res))
    y = ci[1]+int(round(npt.y/res))
    if 0 <= x < size and 0 <= y < size:
        return (x, y)
    else:
        # send them out of bounds...
        return (size, size)

cpdef list anglesList(int n = 8):
    cdef float delta = math.pi/n
    return [i*delta for i in xrange(n)]
