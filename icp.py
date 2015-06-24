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
       Vt[2,:] *= -1
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

def icp(model, data, dmax, niter=10, trans_only=False):
    kdTree = cKDTree(data, 20)
    n = model.shape[0]
    if trans_only:
        R, t = rigid_transform_3D(model, data, trans_only=True)
    else:
        R, t, _ = icp(model, data, dmax, niter=4, trans_only=True)
    print 'Initial'
    print R
    print t
    for iter in range(niter):
        trans_model = (R*model.T + np.tile(t, (1, n))).T
        dists, nbrs = kdTree.query(trans_model, distance_upper_bound=dmax)
        print 'Iter', iter
        # print 'dists', dists
        # print 'nbrs', nbrs
        valid = nbrs < data.shape[0]
        print 'aligning', dists[valid].shape[0], 'points'
        err = sum(np.multiply(dists[valid], dists[valid]))
        rmse = sqrt(err/n);
        print 'rmse', rmse
        if rmse < 0.001: break
        if trans_only:
            R, t = rigid_transform_3D(model[valid], data[nbrs[valid]], trans_only=trans_only)
        else:
            R, t = rigid_transform_3D(model[valid], data[nbrs[valid]], trans_only=True)
            R2, t2 = rigid_transform_3D(model[valid][:,:2], data[nbrs[valid]][:,:2], trans_only=trans_only)
            R[:2,:2] = R2
        print R
        print t
    return R, t, trans_model[valid]

# Construct rotated model for a range of theta
# Do ICP with translation only to get initial alignment for each theta
# Pick best alignment as the starting point for a full search
# or simply do bisection search on the angle.

def drawVerts(verts, color='blue'):
    for v in xrange(verts.shape[1]):
        pointBox(verts[:3,v],r=0.01).draw('W', color)

def drawMat(mat, color='blue'):
    for v in xrange(mat.shape[0]):
        pointBox(np.array(mat[v])[0],r=0.01).draw('W', color)

# model and scan are vertex arrays
def icpLocate(model, data):
    model_mat = np.matrix(model.T[:,:3])
    data_mat = np.matrix(data.T[:,:3])
    drawMat(data_mat, 'orange')
    R, t, validModel = icp(model_mat, data_mat, 0.05)
    drawMat(validModel, 'green')
    raw_input('Matched model points')
    trans = np.matrix(np.eye(4))
    trans[:3,:3] = R
    trans[:3, 3] = t
    r_model = np.dot(np.array(trans), model)
    drawVerts(r_model)
    raw_input('icp')
    return trans

def getObjectDetections(placeB, pbs, pointCloud):
    startTime = time.time()
    objName = placeB.obj
    pose = placeB.poseD.mode().pose()
    shWorld = pbs.getShadowWorld(0.95)
    world = pbs.getWorld()
    objShape = placeB.shape(shWorld)
    objShadow = placeB.shadow(shWorld)
    objCloud = world.typePointClouds[world.objectTypes[objName]]

    # Rotate the objCloud to the poseD.mode()
    
    var = placeB.poseD.variance()
    std = var[-1]**0.5          # std for angles
    angles = [pose.theta+d for d in (-3*std, -2*std, std, 0., std, 2*std, 3*std)]    
    if debug('objDetections'):
        print 'Candidate obj angles:', angles
    score, detection = \
           bestObjDetection(objShape, objShadow, objCloud, pointCloud, angles = angles, thr = 0.05)
    print 'Obj detection', detection, 'with score', score
    print 'Running time for obj detections =',  time.time() - startTime
    if detection:
        detection.draw('MAP', 'blue')
        debugMsg('objDetections', 'Detection for obj=%s'%objName)
        return (score, detection)

def bestObjDetection(objShape, objShadow, objCloud, pointCloud, angles = [0.], thr = 0.02):
    good = []
    points = pointCloud.vertices
    headTrans = pointCloud.headTrans
    for p in range(1, points.shape[1]):   # index 0 is eye
        pt = points[:,p]
        for sh in objShadow.parts():
            if np.all(np.dot(sh.planes(), pt) <= thr): # note use of thr
                good.append(p); break
    good = np.array(good)               # indices of points in shadow
    goodCloud = pointCloud.vertices[:, good]

    # Try to align the model to the relevant data
    trans = util.Transform(np.array(icpLocate(objCloud, goodCloud)))
    transShape = objShape.applyLoc(trans)
    transShape.draw('W', 'green')
    raw_input('transShape (in green)')

    # Evaluate the result by comparing the depth map from the point
    # cloud to the simulated depth map for the detection.
    goodScan = pc.Scan(pointCloud.headTrans, None, verts=goodCloud)
    dmScan = goodScan.depthMap()
    dmShape, contacts = pc.simulatedDepthMap(goodScan, [transShape])
    score = 0.
    diff = dmScan - dmShape
    for i in range(diff.shape[0]):
        if dmShape[i] == 10.: continue  # no prediction
        if diff[i] > thr: score -= 0.1
        elif diff[i] < -thr: score -= 1.0
        else: score += 1.0

    return score, transShape


                  
