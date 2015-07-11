import numpy as np
from math import sqrt
from scipy.spatial import cKDTree
from scipy.optimize import fmin, fmin_powell
from shapes import pointBox
import pointClouds as pc
from traceFile import tr
import time
#from planGlobals import debug, debugMsg
import windowManager3D as wm
import hu
from pr2Util import colorGen
from dist import MultivariateGaussianDistribution
MVG = MultivariateGaussianDistribution


laserScanParams = (0.3, 0.1, 0.1, 2., 20)
def getObjectDetections(lookConf, placeB, pbs, pointCloud):
    startTime = time.time()
    objName = placeB.obj
    pose = placeB.objFrame().pose()     # nominal pose
    shWorld = pbs.getShadowWorld(0.95)
    world = pbs.getWorld()
    objShape = placeB.shape(pbs.getWorld()) # shape at nominal pose
    objShadow = placeB.shadow(shWorld)
    # Simulated scan for object at nominal pose
    objScan = pc.simulatedScan(lookConf, laserScanParams, [objShape])
    contacts = np.array([bool(c) for c in objScan.contacts])
    # The point cloud corresponding to contacts with the object
    objCloud = objScan.vertices[:, contacts]
    # drawVerts(objCloud, 'W', 'green')
    # raw_input('objCloud')
    var = placeB.poseD.variance()
    # trans is a Pose displacement from the nominal one
    score, trans, detection = \
           bestObjDetection(placeB, objShape, objShadow, objCloud, pointCloud, var)
    tr('detect', 1, 'Obj detection', detection, 'with score', score, ol = True)
    tr('detect', 1, 'Running time for obj detections =',  time.time() - startTime,
       ol = True)
    tr('locate', 0, 'Detection for obj=%s'%objName)
    if score is None:
        return (None, None, None)
    else:
        return (score, trans.compose(placeB.poseD.mode()), detection)

def bestObjDetection(placeB, objShape, objShadow, objCloud, pointCloud, variance,
                     thr = 0.02):
    good = [0]                          # include eye
    points = pointCloud.vertices
    headTrans = pointCloud.headTrans
    minZ = objShadow.bbox()[0,2] + thr
    for p in range(1, points.shape[1]):   # index 0 is eye
        pt = points[:,p]
        for sh in objShadow.parts():
            if pt[2] > minZ and \
                   np.all(np.dot(sh.planes(), pt) <= thr): # note use of thr
                good.append(p); break
    good = np.array(good)               # indices of points in shadow
    goodCloud = pointCloud.vertices[:, good]
    if objCloud.shape[1] < 10 or goodCloud.shape[1] < 0.5*objCloud.shape[1]:
        return (None, None, None)
    goodScan = pc.Scan(pointCloud.headTrans, None, verts=goodCloud)

    # print 'initial score', depthScore(objShape, goodScan, thr)

    # Try to align the model to the relevant data
    # score, trans = locate(placeB, objCloud, goodCloud[:,1:], variance, thr, objShape)
    score, trans = locateByFmin(objCloud, goodCloud[:,1:], variance, shape=objShape)
    tr('locate', 2, 'best score', score, 'best trans', trans, ol = True)
    transShape = objShape.applyTrans(trans)
    transShape.draw('W', 'purple')
    tr('locate', 0, 'transShape (in purple)', snap = ['W'])

    # Evaluate the result by comparing the depth map from the point
    # cloud to the simulated depth map for the detection.
    # score = depthScore(transShape, goodScan, thr)
    # raw_input('score=%f'%score)
    return score, trans, transShape

def diagToSq(d):
    return [[(d[i] if i==j else 0.0) \
             for i in range(len(d))] for j in range(len(d))]

def locateByFmin(model_verts, data_verts, variance, shape=None):
    win = wm.getWindow('W')
    data = data_verts.T[:,:3]
    drawVerts(data_verts, 'W', colorGen.next()); win.update()
    kdTree = cKDTree(data, 20)
    frac = 0.9
    Nt = int(frac*model_verts.shape[1])
    def poseScore(xv):
        (x, y, z, th) = xv
        pose = hu.Pose(x,y,z,th)
        model_verts_trans = np.dot(pose.matrix, model_verts)
        model_trans = model_verts_trans.T[:, :3]
        dists, nbrs = kdTree.query(model_trans)
        sdists = np.sort(dists)[:Nt]
        err = sum(np.multiply(sdists, sdists))
        rmse = sqrt(err/Nt)
        # print '    ', xv, '->', rmse
        return rmse
    mvg = MVG((0.,0.,0.,0.), diagToSq(variance))
    best_score = None
    best_trans = None
    no_improvement = 0
    for attempt in xrange(30):
        initial = mvg.draw()
        print 'initial', initial
        final = fmin_powell(poseScore, initial,
                            xtol = 0.001, ftol = 0.001)
        score = poseScore(final)
        tr('locate', 3, 'score', score, 'final', final, ol = True)
        tr('locate', 3,  'best_score', best_score, 'best_trans', best_trans,
           ol = True)
        if best_score is None or score < best_score:
            no_improvement = 0
            best_score = score
            best_trans = hu.Pose(*final)
            if shape:
                shape.applyTrans(best_trans).draw('W', 'orange'); win.update()
            # ans = raw_input('Continue?')
            # if not ans: break
        else:
            no_improvement += 1
            if no_improvement >= 10: break
    return (best_score, best_trans)
    
def depthScore(shape, scan, thr):
    dmScan = scan.depthMap()        # actual depths
    # dmShape is normalized depths
    dmShape, contacts = pc.simulatedDepthMap(scan, [shape])
    dmShape = dmScan * dmShape
    drawVerts(vertsForDM(dmShape, scan), 'W', 'cyan')
    score = 0.
    diff = dmScan - dmShape
    for i in range(diff.shape[0]):
        if dmShape[i] > 5.: continue    # no prediction
        # if scan point is farther from eye than prediction from
        # shape, then we saw through the object, this is very bad.
        if diff[i] > thr: score -= 1.0
        # if scan is closer to eye, then it could be an occlusion,
        # which is penalized but not as much.
        elif diff[i] < -thr: score -= 0.1
        else: score += 1.0
    return score

def vertsForDM(dm, scan):
    verts = np.zeros((4, dm.shape[0]))
    scanVerts = scan.vertices
    scanDM = scan.depthMap()
    for i in xrange(dm.shape[0]):
        uv = scanVerts[:,i] - scanVerts[:,0]
        if scanDM[i] != 0.:
            uv = uv/scanDM[i]
            verts[:,i] = scanVerts[:,0] + dm[i]*uv
        else:
            verts[:,i] = scanVerts[:,0]
    return verts

def drawVerts(verts, win = 'W', color='blue'):
    for v in xrange(verts.shape[1]):
        pointBox(verts[:3,v],r=0.01).draw(win, color)

def drawMat(mat, win = 'W', color='blue'):
    for v in xrange(mat.shape[0]):
        pointBox(np.array(mat[v])[0],r=0.01).draw(win, color)
