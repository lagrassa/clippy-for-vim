import math
import numpy as np
cimport numpy as np
from cpython cimport bool

cimport util
import util

cimport shapes
import shapes

cimport geom
from geom import vertsBBox, bboxContains

######################################
# Simulated point clouds
######################################

cdef class Scan:
    # The input verts includes eye as vertex 0
    def __init__(self, util.Transform headTrans, tuple scanParams,
                 verts = None, str name='scan', str color=None):
        self.name = name
        self.color = color
        self.headTrans = headTrans
        self.headTransInverse = headTrans.inverse()
        self.eye = headTrans.point()
        if verts == None:
            self.vertices = scanVerts(scanParams, headTrans)
        else:
            self.vertices = verts
        edges = np.zeros((self.vertices.shape[1], 2), dtype=np.int)
        for i from 1 <= i < self.vertices.shape[1]:
            edges[i,0] = 0; edges[i,1] = i
        self.edges = edges
        self.bbox = vertsBBox(self.vertices, None)
        self.scanParams = scanParams

    def __str__(self):
        return 'Scan:'+str(self.name)+'_'+str(self.eye)
    __repr__ = __str__

    cpdef Scan applyTrans(self, util.Transform trans):
        return Scan(trans.compose(self.headTrans), self.scanParams,
                    np.dot(trans.matrix, self.vertices), self.name, self.color)

    cpdef bool visible(self, util.Point pt):
        cdef float height, width, length, focal
        cdef int n
        (focal, height, width, length, n) = self.scanParams
        cdef:
            util.Point ptLocal = self.headTransInverse.applyToPoint(pt)
            float t
        if ptLocal.x <= focal: return False
        t = focal/ptLocal.x
        return abs(t*ptLocal.y) <= width \
               and abs(t*ptLocal.z) <= height \
               and (ptLocal.x**2 + ptLocal.y**2 + ptLocal.z**2 < length**2)

    cpdef draw(self, window, str color = None):
        cdef:
            float r = 0.01
            np.ndarray[np.float64_t, ndim=2] v
        if self.scanParams:
            (focal, height, width, length, n) = self.scanParams
        else:
            length = 3.0
        ray = shapes.BoxAligned(np.array([(-r, -r, -r), (length/3., r, r)]), None)
        ray = ray.applyTrans(self.headTrans)
        ray.draw(window, color='red')

        ray = shapes.BoxAligned(np.array([(-r, -r, -r), (r, length/3., r)]), None)
        ray = ray.applyTrans(self.headTrans)
        ray.draw(window, color='green')

        ray = shapes.BoxAligned(np.array([(-r, -r, -r), (r, r, length/3.)]), None)
        ray = ray.applyTrans(self.headTrans)
        ray.draw(window, color='blue')

        pointBox = shapes.BoxAligned(np.array([(-r, -r, -r), (r, r, r)]), None)
        v = self.vertices
        for i from 0 <= i < v.shape[1]:
            pose = util.Pose(v[0,i], v[1,i], v[2,i], 0.0)
            pointBox.applyTrans(pose).draw(window, color or self.color)

cpdef np.ndarray[np.float64_t, ndim=2] scanVerts(tuple scanParams, util.Transform pose):
    cdef:
       float focal, height, width, length, deltaX, deltaY, dirX, dirY, y, z, x, w
       set points
       int iX, iY, n, nv, i
       tuple pt
    (focal, height, width, length, n) = scanParams
    deltaX = width/n                    # or half
    deltaY = height/n                   # or half
    nv = 0
    points = set([])
    for dirX in (-1.0, 0.0, 1.0):
        for iX from 1 <= iX < n+1:
            for dirY in (-1.0, 0.0, 1.0):
                for iY from 1 <= iY < n+1:
                    # rearrange for x front
                    y = iX * dirX * deltaX
                    z = iY * dirY * deltaY
                    x = focal
                    pt = (x, y, z)
                    if pt in points: continue
                    points.add(pt)
                    nv += 1
    verts = np.zeros((4, nv+1), dtype=np.float64)
    v = np.zeros((4,1), dtype=np.float64); v[3,0] = 1.
    verts[:, 0] = np.dot(pose.matrix, v)[:,0]   # eye
    for i, (x, y, z) in enumerate(points):
        w = length/math.sqrt(x*x + y*y + z*z)
        v[0,0] = x * w; v[1,0] = y * w; v[2,0] = z * w
        verts[:, i+1] = np.dot(pose.matrix, v)[:,0]
    return verts

######################################
# Below is based on collision.pyx
######################################

# Computes an array of normalized depths [0,1] along given "edge" segments
# t1 is "fake" object with only vertices and edges, but no faces.
cpdef bool updateDepthMap(Scan scan, shapes.Thing thing,
                          np.ndarray[np.float64_t, ndim=1] depth,
                          list contacts, int objId, list onlyUpdate = []):
    cdef:
        np.ndarray[np.float64_t, ndim=2] f2xv1, verts, bb
        np.ndarray[np.float64_t, ndim=1] crossPlanes, p0, p1, diff, pt
        np.ndarray[np.int_t, ndim=2] edges
        np.ndarray[np.int_t, ndim=1] indices
        int e
        bool ans
        float d0, d1, prod, t
    verts = scan.vertices             # 4xn array
    edges = scan.edges                # ex2 array
    f2xv1 = np.dot(thing.planes(), verts);
    ans = False                         # was an update made?
    bb = thing.bbox()
    for e in range(edges.shape[0]):
        if onlyUpdate:
            if not (contacts[e] and contacts[e][1] in onlyUpdate): continue
        crossPlanes = f2xv1[:, edges[e,0]] * f2xv1[:, edges[e,1]]
        indices = np.where(crossPlanes < 0)[0]
        if np.any(indices):
            p0 = verts[:, edges[e,0]]
            p1 = verts[:, edges[e,1]]
            if min(p0[0],p1[0])>bb[1,0] or max(p0[0],p1[0])<bb[0,0]: continue
            if min(p0[1],p1[1])>bb[1,1] or max(p0[1],p1[1])<bb[0,1]: continue
            if min(p0[2],p1[2])>bb[1,2] or max(p0[2],p1[2])<bb[0,2]: continue
            dots = f2xv1[np.ix_(indices, edges[e])]
            diff = p1 - p0
            for i in range(dots.shape[0]):
                d0 = dots[i, 0]; d1 = dots[i, 1]
                prod = d0 * d1
                if prod >= 0: continue  # same side
                t = - d0/(d1 - d0)
                assert 0 <= t <= 1
                pt = p0 + (t+1.0e-5)*diff
                if thing.containsPt(pt): # brute force test, could be optimized
                    if t < depth[e]:
                        depth[e] = t
                        contacts[e] = (pt, objId)
                        ans = True
    return ans

# This has been merged into the code above...
cpdef float edgeCross(np.ndarray[np.float64_t, ndim=1] p0, # row vector
                      np.ndarray[np.float64_t, ndim=1] p1, # row vector
                      np.ndarray[np.float64_t, ndim=2] dots,
                      shapes.Thing thing):
    cdef:
        np.ndarray[np.float64_t, ndim=1] diff, pt
        float d0, d1, prod, t
        int i
    diff = p1 - p0
    for i in range(dots.shape[0]):
        d0 = dots[i, 0]; d1 = dots[i, 1]
        prod = d0 * d1
        if prod >= 0: continue          # same side
        t = - d0/(d1 - d0)
        assert 0 <= t <= 1
        pt = p0 + t*diff
        if thing.containsPt(pt):        # brute force test, could be optimized
            return t
    return 10.0                         # out of bounds


