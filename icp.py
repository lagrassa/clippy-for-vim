from numpy import *
from math import sqrt
from scipy.spatial import cKDTree
from transformations import euler_matrix
from shapes import toPrims, pointBox, readOff

# Input: expects Nx3 matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector

def rigid_transform_3D(A, B, trans_only=False):
    if not trans_only:
        assert len(A) == len(B)

    N = A.shape[0]; # total points

    centroid_A = mean(A, axis=0)
    centroid_B = mean(B, axis=0)
    
    if trans_only:
        t = -centroid_A.T + centroid_B.T
        return mat(identity(t.shape[0])), t

    # centre the points
    AA = A - tile(centroid_A, (N, 1))
    BB = B - tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = transpose(AA) * BB

    U, S, Vt = linalg.svd(H)

    R = Vt.T * U.T

    # special reflection case
    if linalg.det(R) < 0:
       print "Reflection detected"
       Vt[2,:] *= -1
       R = Vt.T * U.T

    t = -R*centroid_A.T + centroid_B.T

    return R, t

# Test with random data
def test_transform():
    # Random rotation and translation
    R = mat(random.rand(3,3))
    t = mat(random.rand(3,1))

    # make R a proper rotation matrix, force orthonormal
    U, S, Vt = linalg.svd(R)
    R = U*Vt

    # remove reflection
    if linalg.det(R) < 0:
       Vt[2,:] *= -1
       R = U*Vt

    R = mat(euler_matrix(*((random.rand(3)).tolist()), axes='sxyz'))[:3,:3]
    t = 0.1*mat(random.rand(3,1))

    # number of points
    n = 10

    A = mat(random.rand(n,3));
    B = R*A.T + tile(t, (1, n))
    B = B.T;

    # recover the transformation
    ret_R, ret_t = rigid_transform_3D(A, B)

    A2 = (ret_R*A.T) + tile(ret_t, (1, n))
    A2 = A2.T

    # Find the error
    err = A2 - B

    err = multiply(err, err)
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
        trans_model = (R*model.T + tile(t, (1, n))).T
        dists, nbrs = kdTree.query(trans_model, distance_upper_bound=dmax)
        print 'Iter', iter
        print 'dists', dists
        print 'nbrs', nbrs
        valid = nbrs < data.shape[0]
        print 'aligning', dists[valid].shape[0], 'points'
        err = sum(multiply(dists[valid], dists[valid]))
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
        pointBox(array(mat[v])[0],r=0.01).draw('W', color)

def icpLocate(file, scan, bbox):
    model = readOff(file, name='shelves')
    model_mat = matrix(model.T[:,:3])
    scan_mat = matrix(scan.vertices.T[1:,:3])
    inside = [greater_equal(scan_mat[i], bbox[0]).all() and less_equal(scan_mat[i],bbox[1]).all() \
              for i in xrange(scan_mat.shape[0])]
    data_mat = scan_mat[array(inside)]
    drawMat(data_mat, 'orange')
    R, t, validModel = icp(model_mat, data_mat, 0.05)
    drawMat(validModel, 'green')
    raw_input('Matched model points')
    trans = matrix(eye(4))
    trans[:3,:3] = R
    trans[:3, 3] = t
    r_model = dot(array(trans), model)
    drawVerts(r_model)
    raw_input('icp')
    return trans
