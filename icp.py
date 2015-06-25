import numpy as np
from math import sqrt
from scipy.spatial import cKDTree
from transformations import euler_matrix
from shapes import toPrims, pointBox, readOff
import pointClouds as pc
import time
from planGlobals import debug, debugMsg
import util

# Input: expects Nx3 matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector

def rigid_transform_3D(A, B, trans_only=False):
    if not trans_only:
        assert len(A) == len(B)

    N = A.shape[0]; # total points

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    
    if trans_only:
        t = -centroid_A.T + centroid_B.T
        return np.mat(np.identity(t.shape[0])), t

    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = np.transpose(AA) * BB

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T * U.T

    # special reflection case
    if np.linalg.det(R) < 0:
       print "Reflection detected"
       Vt[Vt.shape[0]-1,:] *= -1
       R = Vt.T * U.T

    t = -R*centroid_A.T + centroid_B.T

    return R, t

# Test with random data
def test_transform():
    # Random rotation and translation
    R = np.mat(np.random.rand(3,3))
    t = np.mat(np.random.rand(3,1))

    # make R a proper rotation matrix, force orthonormal
    U, S, Vt = np.linalg.svd(R)
    R = U*Vt

    # remove reflection
    if np.linalg.det(R) < 0:
       Vt[2,:] *= -1
       R = U*Vt

    R = np.mat(euler_matrix(*((random.rand(3)).tolist()), axes='sxyz'))[:3,:3]
    t = 0.1*np.mat(np.random.rand(3,1))

    # number of points
    n = 10

    A = np.mat(np.random.rand(n,3));
    B = R*A.T + np.tile(t, (1, n))
    B = B.T;

    # recover the transformation
    ret_R, ret_t = rigid_transform_3D(A, B)

    A2 = (ret_R*A.T) + np.tile(ret_t, (1, n))
    A2 = A2.T

    # Find the error
    err = A2 - B

    err = np.multiply(err, err)
    err = sum(err)
    rmse = sqrt(err/n);

    print "Points A"
    print A
    print ""

    print "Points B"
    print B
    print ""

    print "Rotation"
    print R
    print ""

    print "Translation"
    print t
    print ""

    print "RMSE:", rmse
    print "If RMSE is near zero, the function is correct!"

    return icp(A, B, 0.25)

def icp0(model, data, dmax, niter=100, trans_only=False):
    kdTree = cKDTree(data, 20)

    n = model.shape[0]
    # if trans_only:
    #     R, t = rigid_transform_3D(model, data, trans_only=True)
    # else:
    #     R, t, _ = icp(model, data, dmax, niter=4, trans_only=True)
    R = np.mat(np.eye(3))
    t = np.mat(np.zeros(3)).T
    print 'Initial'
    print R
    print t
    for iter in range(niter):
        trans_model = (R*model.T + np.tile(t, (1, n))).T
        dmx = dmax - iter*0.001
        dists, nbrs = kdTree.query(trans_model, distance_upper_bound=dmx)
        print 'Iter', iter
        # print 'dists', dists
        # print 'nbrs', nbrs
        valid = nbrs < data.shape[0]
        validData = set(nbrs[valid])
        print 'aligning', dists[valid].shape[0], 'model points to', len(validData), 'data points'
        err = sum(np.multiply(dists[valid], dists[valid]))
        rmse = sqrt(err/n);
        print 'rmse', rmse
        if rmse < 0.001 or len(validData) < 10 or dmx < 0.02: break
        if trans_only:
            R, t = rigid_transform_3D(model[valid], data[nbrs[valid]], trans_only=trans_only)
        else:
            R, t = rigid_transform_3D(model[valid], data[nbrs[valid]], trans_only=True)
            R2, t2 = rigid_transform_3D(model[valid][:,:2], data[nbrs[valid]][:,:2], trans_only=trans_only)
            R[:2,:2] = R2
        print R
        print t
    return R, t, trans_model[valid]

def icp(model, data, trimFactor=0.9, niter=10, trans_only=False):
    kdTree = cKDTree(model, 20)         # initialize with model
    R = np.mat(np.eye(3))
    t = np.mat(np.zeros(3)).T
    trans_data = data                   # we'll transform data
    N = data.shape[0]
    Nt = trimFactor*N
    print 'N=', N, 'Nt=', Nt
    e_old = 0.0
    for iter in range(niter):
        print 'Iter', iter
        dists, nbrs = kdTree.query(trans_data) # find assignments for data
        sdists = np.sort(dists)
        err = sum(np.multiply(sdists, sdists)[:Nt])
        e = sqrt(err/Nt)                # rmse for trim fraction of the data
        rel_de = abs(e - e_old)/e
        print 'rmse', e, 'rel change in rmse', rel_de
        if e < 0.001 or rel_de < 0.0001: break
        e_old = e
        valid = dists <= sdists[Nt-1]   # relevant subset of data
        # Compute trans based on relevant subset
        if trans_only:
            R, t = rigid_transform_3D(data[valid], model[nbrs[valid]], trans_only=trans_only)
        else:
            R, t = rigid_transform_3D(data[valid], model[nbrs[valid]], trans_only=True)
            R2, t2 = rigid_transform_3D(data[valid][:,:2], model[nbrs[valid]][:,:2], trans_only=trans_only)
            R[:2,:2] = R2
        print R
        print t
        trans_data = (R*data.T + np.tile(t, (1, N))).T
    return R, t, e

# Construct rotated model for a range of theta
# Do ICP with translation only to get initial alignment for each theta
# Pick best alignment as the starting point for a full search
# or simply do bisection search on the angle.

def drawVerts(verts, win = 'W', color='blue'):
    for v in xrange(verts.shape[1]):
        pointBox(verts[:3,v],r=0.01).draw(win, color)

def drawMat(mat, win = 'W', color='blue'):
    for v in xrange(mat.shape[0]):
        pointBox(np.array(mat[v])[0],r=0.01).draw(win, color)

# model and scan are vertex arrays
def icpLocate(model, data):
    model_mat = np.matrix(model.T[:,:3])
    data_mat = np.matrix(data.T[:,:3])
    drawMat(data_mat, 'orange')
    best_score = None
    best_R = None
    best_t = None
    for frac in np.arange(0.4, 1.0, 0.1):
        R, t, e = icp(model_mat, data_mat, trimFactor=frac)
        score = e*(frac**(-3))
        if best_score == None or score < best_score:
            best_score = score
            best_R = R
            best_t = t
            trans = np.matrix(np.eye(4))
            trans[:3,:3] = best_R
            trans[:3, 3] = best_t
    return trans

def getObjectDetections(placeB, pbs, pointCloud):
    startTime = time.time()
    objName = placeB.obj
    pose = placeB.objFrame().pose()
    shWorld = pbs.getShadowWorld(0.95)
    world = pbs.getWorld()
    objShape = placeB.shape(pbs.getWorld()) # nominal shape
    objShadow = placeB.shadow(shWorld)
    objCloud = world.typePointClouds[world.objectTypes[objName]]
    objCloud = np.dot(np.array(pose.matrix), objCloud)
    var = placeB.poseD.variance()
    std = var[-1]**0.5          # std for angles
    angles = [pose.theta+d for d in (-3*std, -2*std, std, 0., std, 2*std, 3*std)]    
    if debug('icp'):
        print 'Candidate obj angles:', angles
    score, detection = \
           bestObjDetection(objShape, objShadow, objCloud, pointCloud, angles = angles, thr = 0.05)
    print 'Obj detection', detection, 'with score', score
    print 'Running time for obj detections =',  time.time() - startTime
    if detection:
        detection.draw('MAP', 'blue')
        debugMsg('icp', 'Detection for obj=%s'%objName)
        return (score, detection)

def bestObjDetection(objShape, objShadow, objCloud, pointCloud, angles = [0.], thr = 0.02):
    good = [0]                          # include eye
    points = pointCloud.vertices
    headTrans = pointCloud.headTrans
    for p in range(1, points.shape[1]):   # index 0 is eye
        pt = points[:,p]
        for sh in objShadow.parts():
            if np.all(np.dot(sh.planes(), pt) <= thr): # note use of thr
                good.append(p); break
    good = np.array(good)               # indices of points in shadow
    goodCloud = pointCloud.vertices[:, good]
    goodScan = pc.Scan(pointCloud.headTrans, None, verts=goodCloud)

    print 'initial score', depthScore(objShape, goodScan, thr)

    # Try to align the model to the relevant data
    atrans = icpLocate(objCloud, goodCloud[:,1:])
    trans = util.Transform(np.array(atrans)).inverse()
    transShape = objShape.applyTrans(trans)
    transShape.draw('W', 'green')
    raw_input('transShape (in green)')

    # Evaluate the result by comparing the depth map from the point
    # cloud to the simulated depth map for the detection.
    score = depthScore(transShape, goodScan, thr)

    raw_input('score=%f'%score)
    return score, transShape

def depthScore(shape, scan, thr):
    dmScan = scan.depthMap()        # actual depths
    # dmShape is normalized depths
    dmShape, contacts = pc.simulatedDepthMap(scan, [shape])
    dmShape = dmScan * dmShape
    drawVerts(vertsForDM(dmShape, scan), 'cyan')
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
