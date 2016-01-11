import math
import numpy as np
cimport numpy as np
from cpython cimport bool

cimport hu
import hu
Ident = hu.Transform(np.eye(4))            # identity transform
import transformations as transf
cimport shapes
import shapes

cimport geom
from geom import vertsBBox, bboxContains
import windowManager3D as wm

tiny = 1.0e-6

######################################
# Simulated point clouds
######################################

cdef class Scan:
    # The input verts includes eye as vertex 0
    def __init__(self, hu.Transform headTrans, tuple scanParams,
                 verts = None, str name='scan', str color=None, contacts = None):
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
        for i in range(1, self.vertices.shape[1]):
            edges[i,0] = 0; edges[i,1] = i
        self.edges = edges
        self.contacts = contacts
        self.bbox = vertsBBox(self.vertices, None)
        self.scanParams = scanParams

    def __str__(self):
        return 'Scan:'+str(self.name)+'_'+str(self.eye)
    __repr__ = __str__

    cpdef np.ndarray depthMap(self):
        n = self.edges.shape[0]
        v = self.vertices.T
        dm = (v[self.edges[:,1]] - v[0])**2
        dm = np.sum(dm, axis=1)
        dm = np.sqrt(dm)
        return dm

    cpdef Scan applyTrans(self, hu.Transform trans):
        return Scan(trans.compose(self.headTrans), self.scanParams,
                    np.dot(trans.matrix, self.vertices), self.name, self.color)

    cpdef bool visible(self, hu.Point pt):
        cdef double height, width, length, focal
        cdef int n
        (focal, height, width, length, n) = self.scanParams
        cdef:
            hu.Point ptLocal = self.headTransInverse.applyToPoint(pt)
            double t, x, y, z
        x,y,z = [ptLocal.matrix[i,0] for i in (0,1,2)]
        if x <= focal: return False
        t = focal/x
        return abs(t*y) <= width \
               and abs(t*z) <= height \
               and (x**2 + y**2 + z**2 < length**2)

    def draw(self, window, str color = None):
        if self.scanParams:
            (focal, height, width, length, n) = self.scanParams
        else:
            length = 3.0

        # Draw a coordinate frame (RGB = XYZ)
        r = 0.01
        ray = shapes.BoxAligned(np.array([(-r, -r, -r), (length/3., r, r)]), None)
        ray.applyTrans(self.headTrans).draw(window, color='red')
        ray = shapes.BoxAligned(np.array([(-r, -r, -r), (r, length/3., r)]), None)
        ray.applyTrans(self.headTrans).draw(window, color='green')
        ray = shapes.BoxAligned(np.array([(-r, -r, -r), (r, r, length/3.)]), None)
        ray.applyTrans(self.headTrans).draw(window, color='blue')

        wm.getWindow(window).draw(self.vertices, color or self.color)

        # pointBox = shapes.BoxAligned(np.array([(-r, -r, -r), (r, r, r)]), None)
        # v = self.vertices
        # for i from 0 <= i < v.shape[1]:
        #     pose = util.Pose(v[0,i], v[1,i], v[2,i], 0.0)
        #     pointBox.applyTrans(pose).draw(window, color or self.color)


cpdef np.ndarray scanVerts(tuple scanParams, hu.Transform pose):
    cdef:
       double focal, height, width, length, deltaX, deltaY, dirX, dirY, y, z, x, w
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

scanCache = {}

def simulatedScan(conf, scanParams, objects, name='scan', color=None):
    lookCartConf = conf.robot.forwardKin(conf)
    headTrans = lookCartConf['pr2Head']
    if scanParams in scanCache:
        laserScan = scanCache[scanParams]
    else:
        laserScan = Scan(Ident, scanParams)
        scanCache[scanParams] = laserScan
    scanTrans = headTrans.compose(hu.Transform(transf.rotation_matrix(-math.pi/2, (0,1,0))))
    scan = laserScan.applyTrans(scanTrans)
    n = scan.edges.shape[0]
    dm = np.zeros(n); dm.fill(10.0)
    contacts = n*[None]
    for i, shape in enumerate(objects):
        for objPrim in shape.toPrims():
            updateDepthMap(scan, objPrim, dm, contacts, i)
    verts = np.zeros((4, n))
    for i in range(n):
        if contacts[i]:
            verts[:,i] = contacts[i][0]
        else:
            verts[:, i] = scan.vertices[:, i]
    return Scan(headTrans, scanParams, verts=verts,
                name=name, color=color, contacts=contacts)

def simulatedDepthMap(scan, objects):
    n = scan.edges.shape[0]
    dm = np.zeros(n); dm.fill(10.0)
    contacts = n*[None]
    for i, shape in enumerate(objects):
        for objPrim in shape.toPrims():
            updateDepthMap(scan, objPrim, dm, contacts, i)
    return dm, contacts

######################################
# Below is based on collision.pyx
######################################

# Computes an array of normalized depths [0,1] along given "edge" segments
# t1 is "fake" object with only vertices and edges, but no faces.
cpdef bool updateDepthMap(Scan scan, shapes.Prim thing,
                          np.ndarray[np.float64_t, ndim=1] depth,
                          list contacts, int objId, list onlyUpdate = []):
    cdef:
        np.ndarray[np.float64_t, ndim=2] f2xv1, verts, bb
        np.ndarray[np.float64_t, ndim=1] crossPlanes, p0, p1, diff, pt
        np.ndarray[np.int_t, ndim=2] edges
        np.ndarray[np.int_t, ndim=1] indices
        int e
        bool ans
        double d0, d1, prod, t
    verts = scan.vertices             # 4xn array
    edges = scan.edges                # ex2 array
    f2xv1 = np.dot(thing.planes(), verts)
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
                if np.all(np.dot(thing.planes(), pt.reshape(4,1)) <= tiny):
                    # brute force test, could be optimized
                    if t < depth[e]:
                        depth[e] = t
                        contacts[e] = (pt, objId)
                        ans = True
    return ans

# This has been merged into the code above...
cpdef double edgeCross(np.ndarray[np.float64_t, ndim=1] p0, # row vector
                      np.ndarray[np.float64_t, ndim=1] p1, # row vector
                      np.ndarray[np.float64_t, ndim=2] dots,
                      shapes.Prim thing):
    cdef:
        np.ndarray[np.float64_t, ndim=1] diff, pt
        double d0, d1, prod, t
        int i
    diff = p1 - p0
    for i in range(dots.shape[0]):
        d0 = dots[i, 0]; d1 = dots[i, 1]
        prod = d0 * d1
        if prod >= 0: continue          # same side
        t = - d0/(d1 - d0)
        assert 0 <= t <= 1
        pt = p0 + t*diff
        if np.all(np.dot(thing.planes(), pt.reshape(4,1)) <= tiny):
            # brute force test, could be optimized
            return t
    return 10.0                         # out of bounds

######################################
# raster version of depth map
######################################

# Should be in-line...
cdef inline double edgeFunction(np.ndarray[np.float64_t, ndim=1] a,
                                np.ndarray[np.float64_t, ndim=1] b,
                                np.ndarray[np.float64_t, ndim=1] c):
    return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])

cdef class Raster:
    def __init__(self,
                 screenWidth, screenHeight, imageWidth, imageHeight,
                 nearClippingPlane, farClippingPLane, focalLength):
        self.screenWidth = screenWidth
        self.screenHeight = screenHeight
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight
        self.nearClippingPlane = nearClippingPlane
        self.farClippingPLane = farClippingPLane
        self.focalLength = focalLength
        screenAspectRatio = float(screenWidth)/float(screenHeight)
        imageAspectRatio = float(imageWidth) / float(imageHeight)
        top = ((screenHeight/2.) / focalLength) * nearClippingPlane
        right = ((screenWidth/2.) / focalLength) * nearClippingPlane
        # field of view (horizontal)
        xscale = 1.; yscale = 1.
        fov = 2*180/math.pi*math.atan((screenWidth / 2.) / focalLength)
        print 'fov=', fov
        if screenAspectRatio > imageAspectRatio:
            xscale = imageAspectRatio / screenAspectRatio
        else:
            yscale = screenAspectRatio / imageAspectRatio
        right *= xscale
        top *= yscale
        bottom = -top
        left = -right
        self.screenCoordinates = (left, right, top, bottom)
        size = imageWidth * imageHeight
        # self.depthBuffer = np.full(size, farClippingPLane, dtype=np.float64)
        self.depthBuffer = np.zeros(size, dtype=np.float64)
        for i in xrange(size): self.depthBuffer[i] = farClippingPLane
        self.frameBuffer = np.zeros(size, dtype=np.int)

    cpdef int convertToRaster(self,
                             np.ndarray[np.float64_t, ndim=1] vertexCamera,
                             np.ndarray[np.float64_t, ndim=1] vertexRaster):
        cdef:
            double l, r, t, b, vScreen_x, vScreen_y, vNDC_x, vNDC_y
        (l, r, t, b) = self.screenCoordinates
        vScreen_x = self.nearClippingPlane * vertexCamera[0] / -vertexCamera[2]
        vScreen_y = self.nearClippingPlane * vertexCamera[1] / -vertexCamera[2]
        # from screen to NDC (range [-1,1])
        vNDC_x = 2. * vScreen_x / (r - l) - (r + l) / (r - l)
        vNDC_y = 2. * vScreen_y / (t - b) - (t + b) / (t - b)
        vertexRaster[0] = (vNDC_x + 1) / 2 * self.imageWidth
        # in raster space y is down
        vertexRaster[1] = (1 - vNDC_y) / 2 * self.imageHeight
        vertexRaster[2] = - vertexCamera[2]
        return 0

    cpdef int reset(self):
        cdef:
            int size = self.imageWidth * self.imageHeight
            int i
        for i in xrange(size):
            self.depthBuffer[i] = self.farClippingPLane
            self.frameBuffer[i] = 0.
        return 0

    cpdef int update(self, prim, int objId, onlyUpdate=None):
        cdef:
            np.ndarray[np.int_t, ndim=1] fbuff = self.frameBuffer
            np.ndarray[np.float64_t, ndim=1] dbuff = self.depthBuffer
            np.ndarray[np.float64_t, ndim=1] v0, v1, v2, v0Raster, v1Raster, v2Raster, pixelSample
            np.ndarray[np.float64_t, ndim=2] verts
            np.ndarray[np.int_t, ndim=1] face
            double xmin, ymin, xmax, ymax, w0, w1, w2, z, near
            int x0, x1, y0, y1, x, y, curId, v
            set updates = set([]) if onlyUpdate is None else onlyUpdate
        
        assert objId > 0
        v0Raster = np.zeros(3, dtype=np.float64)
        v1Raster = np.zeros(3, dtype=np.float64)
        v2Raster = np.zeros(3, dtype=np.float64)
        pixelSample = np.zeros(2, dtype=np.float64)
        near = self.nearClippingPlane

        verts = prim.vertices()
        # Punt on prims too near or behing the eye
        for v in xrange(verts.shape[1]):
            if -verts[2,v] < near: return 1
        for face in prim.faces():
            # face is array of indices for verts in face
            v0 = verts[:, face[0]]
            for tri in xrange(face.shape[0] - 2):
                v1 = verts[:, face[tri+1]]
                v2 = verts[:, face[tri+2]]
                # Convert the vertices of the triangle to raster space
                self.convertToRaster(v0, v0Raster)
                self.convertToRaster(v1, v1Raster)
                self.convertToRaster(v2, v2Raster)

                # window = wm.getWindow('Raster')
                # window.window.drawLineSeg(v0Raster[0], v0Raster[1], v1Raster[0], v1Raster[1])
                # window.window.drawLineSeg(v1Raster[0], v1Raster[1], v2Raster[0], v2Raster[1])
                # window.window.drawLineSeg(v2Raster[0], v2Raster[1], v0Raster[0], v0Raster[1])
                # raw_input('Tri')
                
                # Precompute reciprocal of vertex z-coordinate
                v0Raster[2] = 1. / v0Raster[2]
                v1Raster[2] = 1. / v1Raster[2]
                v2Raster[2] = 1. / v2Raster[2]

                xmin = min(v0Raster[0], v1Raster[0], v2Raster[0])
                ymin = min(v0Raster[1], v1Raster[1], v2Raster[1])
                xmax = max(v0Raster[0], v1Raster[0], v2Raster[0])
                ymax = max(v0Raster[1], v1Raster[1], v2Raster[1])

                # the triangle is out of screen
                if (xmin > self.imageWidth - 1 or xmax < 0 \
                    or ymin > self.imageHeight - 1 or ymax < 0): continue

                # be careful xmin/xmax/ymin/ymax can be negative
                x0 = int(max(0, math.floor(xmin)))
                x1 = int(min(self.imageWidth - 1, math.floor(xmax)))
                y0 = int(max(0, math.floor(ymin)))
                y1 = int(min(self.imageHeight - 1, math.floor(ymax)))

                area = edgeFunction(v0Raster, v1Raster, v2Raster)

                # Inner loop
                for y in xrange(y0, y1+1):
                    for x in xrange(x0, x1+1):
                        pixelSample[0] = x + 0.5; pixelSample[1] = y + 0.5
                        w0 = edgeFunction(v1Raster, v2Raster, pixelSample);
                        w1 = edgeFunction(v2Raster, v0Raster, pixelSample);
                        w2 = edgeFunction(v0Raster, v1Raster, pixelSample);
                        if (w0 >= 0 and w1 >= 0 and w2 >= 0):
                            w0 /= area;
                            w1 /= area;
                            w2 /= area;
                            oneOverZ = v0Raster[2] * w0 + v1Raster[2] * w1 + v2Raster[2] * w2;
                            z = 1. / oneOverZ;
                            # Depth-buffer test
                            off = y * self.imageWidth + x
                            if (z < dbuff[off]):
                                dbuff[off] = z;
                                curId = fbuff[off]
                                if (not updates) or (curId and curId in updates):
                                    fbuff[off] = objId;
        return 0

    cpdef int countId(self, int objId):
        cdef int i, count = 0
        cdef np.ndarray[np.int_t, ndim=1] buff = self.frameBuffer
        for i in xrange(self.imageWidth * self.imageHeight):
            if buff[i] == objId: count += 1
        return count

    def draw(self, win):
        colors = ['red', 'green', 'blue', 'orange', 'cyan', 'purple']
        objColors = {}
        size = self.imageWidth * self.imageHeight
        window = wm.getWindow(win)
        window.clear()
        iw = float(self.imageWidth)
        for i in xrange(size):        
            val = self.frameBuffer[i]
            if val in objColors:
                color = objColors[val]
            else:
                color = colors[val%len(colors)]
                objColors[val] = color
            if val:
                y = int(i/iw)
                x = i - y*self.imageWidth
                window.window.drawPoint(x,self.imageHeight-y,color=color)
        window.update()
