import math

import numpy as np
cimport numpy as np

import util
cimport util

from cpython cimport bool
from collision cimport thingThingCollides
from geom cimport *
from cut cimport *

import windowManager3D as win
import transformations as transf

#################################
# Object classes: Thing, Prim, Shape
#################################

cdef float tiny = 1.0e-6

cdef class Thing:
    """Most general class of object, characterized by a bbox. The more specific
    types (Prim) are subclasses.  All Things also have a properties dictionary,
    which includes a name."""
    def __init__(self,
                 np.ndarray[np.float64_t, ndim=2] bbox,
                 util.Transform origin,
                 **props):
        self.properties = props.copy()
        if not 'name' in self.properties:
            self.properties['name'] = util.gensym('Thing')
        self.thingBBox = bbox
        if origin:
            self.thingOrigin = origin
        else:
            trans = np.eye(4, dtype=np.float64)
            trans[:3, 3] = bboxCenter(bbox)[:3]
            self.thingOrigin = util.Transform(trans)
        self.thingCenter = None
        self.thingVerts = None
        self.thingPlanes = None
        self.thingEdges = None
        self.thingPrim = None
        self.thingFaceFrames = None
        self.thingString = None

    cpdef str name(self):
        return self.properties.get('name', 'noName')

    cpdef list faceFrames(self):
        if self.thingFaceFrames is None:
            self.thingFaceFrames = thingFaceFrames(self.planes(),
                                                   self.thingOrigin)
        return self.thingFaceFrames

    cpdef util.Transform origin(self):
        return self.thingOrigin

    cpdef np.ndarray[np.float64_t, ndim=2] bbox(self):
        """Returns bbox (not a copy)."""
        return self.thingBBox

    cpdef tuple zRange(self):
        """Summarizes z range of bbox by a tuple"""
        return (self.thingBBox[0,2], self.thingBBox[1,2])

    cpdef np.ndarray[np.float64_t, ndim=1] center(self):
        """Returns a point at the center of the bbox."""
        if self.thingCenter is None:
            self.thingCenter = bboxCenter(self.thingBBox)
        return self.thingCenter

    cpdef np.ndarray[np.float64_t, ndim=2] vertices(self):
        cdef:
            float xlo, ylo, zlo, xhi, yhi, zhi
            np.ndarray[np.float64_t, ndim=2] points, verts, bb
        if self.thingVerts is None:
            bb = self.bbox()
            xlo = bb[0,0]; ylo = bb[0,1]; zlo = bb[0,2]
            xhi = bb[1,0]; yhi = bb[1,1]; zhi = bb[1,2]
            points = np.array([[xlo, ylo, zlo, 1.], [xhi, ylo, zlo, 1.],
                               [xhi, yhi, zlo, 1.], [xlo, yhi, zlo, 1.]]).T
            verts = vertsFrom2D(points, zlo, zhi)
            self.thingVerts = verts
        return self.thingVerts

    cpdef np.ndarray[np.float64_t, ndim=2] planes(self):
        cdef:
            float xlo, ylo, zlo, xhi, yhi, zhi
            np.ndarray[np.float64_t, ndim=2] bb
        if self.thingPlanes is None:
            bb = self.bbox()
            self.thingPlanes = np.array([[-1.,0.,0., bb[0,0]], [1.,0.,0., -bb[1,0]],
                                         [0.,-1,0., bb[0,1]], [0.,1.,0., -bb[1,1]],
                                         [0.,0.,-1., bb[0,2]], [0.,0.,1., -bb[1,2]]])
        return self.thingPlanes

    cpdef np.ndarray[np.int_t, ndim=2] edges(self):
        if self.thingEdges is None:
            self.thingEdges = np.array([[3, 2], [2, 6], [6, 7], [1, 5], [4, 5], [2, 1],
                                        [7, 4], [5, 6], [3, 7], [0, 3], [1, 0], [4, 0]], dtype=np.int)
        return self.thingEdges    

    cpdef Prim prim(self):
        """Constructs a Prim that matches the bbox.  Useful when computing
        collisions, etc."""
        if self.thingPrim is None:
            self.thingPrim = BoxAligned(self.bbox(), self.thingOrigin, **self.properties)
        return self.thingPrim

    cpdef list parts(self):
        return [self]

    cpdef Thing applyTransMod(self, util.Transform trans, Thing shape, str frame='unspecified'):
        """Displace the Thing; returns a Prim."""
        return self.prim().applyTransMod(trans, shape, frame)

    cpdef Thing applyLocMod(self, util.Transform trans, Thing shape, str frame='unspecified'):
        """Displace the Thing to a location; returns a Prim."""
        return self.applyTransMod(trans.compose(self.thingOrigin.inverse()), shape, frame)
    
    cpdef Thing applyTrans(self, util.Transform trans, str frame='unspecified'):
        """Displace the Thing; returns a Prim."""
        return self.prim().applyTrans(trans, frame)

    cpdef Thing applyLoc(self, util.Transform trans, str frame='unspecified'):
        """Displace the Thing to a location; returns a Prim."""
        return self.applyTrans(trans.compose(self.thingOrigin.inverse()), frame)

    cpdef bool containsPt(self, np.ndarray[np.float64_t, ndim=1] pt):
        """Test whether the Thing's bbox contains the Point pt."""
        return bboxContains(self.bbox(), pt)

    cpdef bool collides(self, Thing obj):
        """Test whether the Thing's collides with another obj, that
        could be any of the types of Thing."""
        if self.bbox() is None or obj.bbox() is None: return False
        if  bboxOverlap(self.bbox(), obj.bbox()): 
            if isinstance(obj, (Prim, Shape)): # more general type, pass the buck
                return obj.collides(self)
            elif isinstance(obj, Thing):
                return True             # already checked bbox
            else:
                raise Exception, 'Unknown obj type'%str(obj)
        return False

    cpdef Shape cut(self, Thing obj, bool isect = False):
        if not (obj.bbox() is None) and bboxOverlap(self.bbox(), obj.bbox()):
            if isinstance(obj, Shape):
                return self.prim().cut(obj, isect=isect)
            else:
                ans = primPrimCut(self.prim(), obj.prim(), isect=isect)
                if ans:
                    return Shape([ans], self.origin(), **self.properties)
        return None if isect else self

    cpdef Prim xyPrim(self):
        return self.prim()

    cpdef Prim boundingRectPrim(self):
        return self.prim()

    cpdef draw(self, str window, str color = 'black', float opacity = 1.0):
        """Ask the window to draw this object."""
        win.getWindow(window).draw(self, color, opacity)
        
    def __str__(self):
        if not self.thingString:
            self.thingString = self.properties['name']+':'+str(self.bbox().tolist())
        return self.thingString
    def __repr__(self):
        return str(self)
    def __hash__(self):
        return repr(self).__hash__()
    def __richcmp__(self, other, int op):
        if not (other and isinstance(other, Thing)):
            return True if op == 3 else False
        if op == 2:
            ans = self.name() == other.name() and repr(self) == repr(other)
        elif op == 3:
            ans = self.name() != other.name() or repr(self) != repr(other)
        else:
            ans = False
        return ans

# Prim class: convex "chunk", has 3D description and we can get 2.5D
# approx via convex hull of projection.

cdef class Prim(Thing):
    def __init__(self,
                 np.ndarray[np.float64_t, ndim=2] verts,
                 list faces,            # since faces have variable length
                 util.Transform origin,
                 **props):
        self.primVerts = verts
        self.primFaces = faces
        self.primPlanes = None
        self.primEdges = None
        Thing.__init__(self, vertsBBox(verts, None), origin, **props)
        if not 'name' in self.properties:
            self.properties['name'] = util.gensym('Prim')

    cpdef Prim prim(self):
        return self

    cpdef list parts(self):
        return [self]

    cpdef np.ndarray[np.float64_t, ndim=2] vertices(self):
        return self.primVerts

    cpdef list faces(self):
        return self.primFaces

    cpdef np.ndarray[np.float64_t, ndim=2] planes(self):
        if self.primPlanes is None:
            self.primPlanes = primPlanes(self.vertices(), self.faces())
        return self.primPlanes

    cpdef np.ndarray[np.int_t, ndim=2] edges(self):
        if self.primEdges is None:
            self.primEdges = primEdges(self.vertices(), self.faces())
        return self.primEdges

    cpdef Thing applyTrans(self, util.Transform trans, str frame='unspecified',):
        return Prim(np.dot(trans.matrix, self.vertices()),
                    self.faces(),
                    trans.compose(self.thingOrigin),
                    **mergeProps(self.properties, {'frame':frame}))

    cpdef Thing applyTransMod(self, util.Transform trans, Thing shape, str frame='unspecified',):
        shape.primVerts = np.dot(trans.matrix, self.vertices())
        shape.thingOrigin = trans.compose(self.thingOrigin)
        shape.thingBBox = vertsBBox(shape.primVerts, None)
        shape.primPlanes = None         # could be transformed...
        shape.thingFaceFrames = None
        shape.thingCenter = None

    cpdef bool containsPt(self, np.ndarray[np.float64_t, ndim=1] pt):
        # Due to a Cython bug... cannot convert numpy.bool_ to bool
        return True if np.all(np.dot(self.planes(), pt) <= tiny) else False

    cpdef np.ndarray containsPts(self, np.ndarray[np.float64_t, ndim=2] pts):
        """Returns array of booleans"""
        return np.all(np.dot(self.planes(), pts) <= tiny, axis=0)

    cpdef bool collides(self, Thing obj):
        if self.bbox() is None or obj.bbox() is None: return False
        if bboxOverlap(self.bbox(), obj.bbox()):
            if isinstance(obj, Shape): # more general type, pass the buck
                return obj.collides(self)
            else:
                return thingThingCollides(self, obj)
        return False

    cpdef Shape cut(self, Thing obj, bool isect = False):
        if bboxOverlap(self.bbox(), obj.bbox()):
            if isinstance(obj, Shape):  # Shape is a union of objects
                ans = []
                p1 = self
                if bboxOverlap(p1.bbox(), obj.bbox()):
                    if isect:
                        # We want union of intersections.
                        for p2 in obj.parts():
                            cut = p1.cut(p2, isect).parts()
                            ans.extend(cut)
                    else:
                        # For diff, find pieces of p1 outside all of obj
                        p1Parts = [p1]
                        for p2 in obj.parts():
                            temp = [] # will hold pieces of p1 outside of p2
                            for p in p1Parts: # loop over every piece of p1
                                cut = p.cut(p2, isect).parts()
                                temp.extend(cut) # add result to temp
                            p1Parts = temp       # set up p1 parts for next p2
                        ans.extend(p1Parts)
                elif not isect:     # doing diff
                    ans.append(p1)  # keep p1, else ignore for isect
                return Shape(ans, self.origin(), **self.properties) if ans else None
            else:
                ans = primPrimCut(self, obj.prim(), isect=isect)
                if ans:
                    Shape([ans], self.origin(), **self.properties)
        return None if isect else Shape([self], self.origin(), **self.properties)
    
    # Compute XY convex hull
    cpdef Prim xyPrim(self):
        return xyPrimAux(self.vertices(), self.zRange(), self.thingOrigin, self.properties)

    # Compute least inertia box
    cpdef Prim boundingRectPrim(self):
        return boundingRectPrimAux(self.vertices(), self.thingOrigin, self.properties)

    # def __repr__(self):
    #     if not self.thingString:
    #         self.thingString = 'Prim('+str(self.primVerts.tolist())+','+str(self.primFaces)+','+str(self.properties)+')'
    #     return self.thingString

cdef class Shape(Thing):
    def __init__(self, list parts, util.Transform origin, **props):
        self.compParts = parts
        if parts:
            self.compVerts = np.hstack([p.vertices() for p in parts \
                                        if not p.vertices() is None])
            Thing.__init__(self, vertsBBox(self.compVerts, None), origin, **props)
            if not 'name' in self.properties:
                self.properties['name'] = util.gensym('Shape')
        else:
            self.compVerts = None
            Thing.__init__(self, np.array([3*[0.0], 3*[0.0]]), origin, **props)

    cpdef emptyP(self):
        return not self.compParts

    cpdef list parts(self):
        return self.compParts

    cpdef np.ndarray[np.float64_t, ndim=2] vertices(self):
        return self.compVerts

    cpdef Thing applyTrans(self, util.Transform trans, str frame='unspecified'):
        return Shape([p.applyTrans(trans, frame) for p in self.parts()],
                     trans.compose(self.thingOrigin),
                     **mergeProps(self.properties, {'frame':frame}))

    cpdef Thing applyTransMod(self, util.Transform trans, Thing shape, str frame='unspecified'):
        if not shape.compVerts is None:
            shape.compVerts = np.dot(trans.matrix, self.vertices())
            shape.thingBBox = vertsBBox(shape.compVerts, None)
            shape.thingOrigin = trans.compose(self.thingOrigin)
            shape.thingFaceFrames = None
            shape.thingCenter = None
        for p, pm in zip(self.parts(), shape.parts()):
            p.applyTransMod(trans, pm)

    cpdef bool containsPt(self, np.ndarray[np.float64_t, ndim=1] pt):
        for p in self.parts():
            if p.containsPt(pt): return True
        return False

    cpdef np.ndarray containsPts(self, np.ndarray[np.float64_t, ndim=2] pts):
        """Returns array of booleans"""
        return np.array([self.containsPt(pts[i]) for i in range(pts.shape[0])])

    cpdef bool collides(self, Thing obj):
        cdef Thing p1, p2, op
        # Is there any pairwise collision
        if self.bbox() is None or obj.bbox() is None: return False
        if bboxOverlap(self.bbox(), obj.bbox()):
            for p1 in self.parts():
                if p1.bbox() is None: continue
                if bboxOverlap(p1.bbox(), obj.bbox()):
                    for p2 in obj.parts():
                        if p1.collides(p2): return True
        return False

    cpdef Shape cut(self, Thing obj, bool isect = False):
        # Shape is a union of objects
        cdef Prim p1, p2, op
        if bboxOverlap(self.bbox(), obj.bbox()):
            ans = []
            for p1 in self.parts(): # loop over parts of self
                if bboxOverlap(p1.bbox(), obj.bbox()):
                    if isect:
                        # We want union of intersections.
                        for p2 in obj.parts():
                            cut = p1.cut(p2, isect).parts()
                            ans.extend(cut)
                    else:
                        # For diff, find pieces of p1 outside all of obj
                        p1Parts = [p1]
                        for p2 in obj.parts():
                            temp = [] # will hold pieces of p1 outside of p2
                            for p in p1Parts: # loop over every piece of p1
                                cut = p.cut(p2, isect).parts()
                                temp.extend(cut) # add result to temp
                            p1Parts = temp       # set up p1 parts for next p2
                        ans.extend(p1Parts)
                elif not isect:     # doing diff
                    ans.append(p1)  # keep p1, else ignore for isect
            return Shape(ans, self.origin(), **self.properties) if ans else None
        return None if isect else self

    # Compute 3d convex hull
    cpdef Prim prim(self):
        return convexHullPrim(self.vertices(), self.thingOrigin) \
               if self.parts() else None

    # Compute XY convex hull
    cpdef Prim xyPrim(self):
        return xyPrimAux(self.vertices(), self.zRange(), self.thingOrigin, self.properties) \
               if self.parts() else None

    # Compute least inertia box
    cpdef Prim boundingRectPrim(self):
        return boundingRectPrimAux(self.vertices(), self.thingOrigin, self.properties) \
               if self.parts() else None

    # def __repr__(self):
    #     if not self.thingString:
    #         self.thingString = 'Shape('+repr(self.parts())+','+str(self.properties)+')'
    #     return self.thingString

#################################
# Object creation: Prims
#################################

cdef class Box(Prim):
    def __init__(self, float dx, float dy, float dz, util.Transform origin, **props):
        cdef:
            float hdx = 0.5*dx
            float hdy = 0.5*dy
            float hdz = 0.5*dz
            np.ndarray[np.float64_t, ndim=2] points
        if not 'name' in props:
            props = mergeProps(props, {'name':util.gensym("box")})
        points = np.array([[-hdx, -hdy, -hdz, 1.], [hdx, -hdy, -hdz, 1.],
                           [hdx, hdy, -hdz, 1.], [-hdx, hdy, -hdz, 1.]]).T
        Prim.__init__(self,
                      vertsFrom2D(points, -hdz, hdz),
                      facesFrom2D(<int>points.shape[1]),
                      origin,
                      **props)

cdef class BoxScale(Prim):
    def __init__(self, float dx, float dy, float dz, util.Transform origin, float scale, **props):
        cdef:
            float hdx = 0.5*dx
            float hdy = 0.5*dy
            float hdz = 0.5*dz
            np.ndarray[np.float64_t, ndim=2] points
        if not 'name' in props:
            props = mergeProps(props, {'name':util.gensym("box")})
        points = np.array([[-hdx, -hdy, -hdz, 1.], [hdx, -hdy, -hdz, 1.],
                           [hdx, hdy, -hdz, 1.], [-hdx, hdy, -hdz, 1.]]).T
        Prim.__init__(self,
                      vertsFrom2DScale(points, -hdz, hdz, scale),
                      facesFrom2D(<int>points.shape[1]),
                      origin,
                      **props)

cdef class Ngon(Prim):
    def __init__(self, float r, dz, int nsides, util.Transform origin, **props):
        cdef:
            float hdz, ang
            int i
            np.ndarray[np.float64_t, ndim=2] points
        hdz = 0.5*dz
        ang = 2*math.pi/nsides
        if not 'name' in props:
            props = mergeProps(props, {'name':util.gensym("ngon")})
        points = np.array([[r*math.cos(i*ang), r*math.sin(i*ang), -hdz, 1.] \
                           for i in range(nsides)]).T
        Prim.__init__(self,
                      vertsFrom2D(points, -hdz, hdz),
                      facesFrom2D(<int>points.shape[1]),
                      origin,
                      **props)

cdef class BoxAligned(Prim):
    def __init__(self, np.ndarray[np.float64_t, ndim=2] bbox, util.Transform origin, **props):
        cdef:
            float xlo, ylo, zlo, xhi, yhi, zhi
            np.ndarray[np.float64_t, ndim=2] points
        if not 'name' in props:
            props = mergeProps(props, {'name':util.gensym("box")})
        ((xlo, ylo, zlo), (xhi, yhi, zhi)) = bbox.tolist()
        points = np.array([[xlo, ylo, zlo, 1.], [xhi, ylo, zlo, 1.],
                           [xhi, yhi, zlo, 1.], [xlo, yhi, zlo, 1.]]).T
        Prim.__init__(self,
                      vertsFrom2D(points, zlo, zhi),
                      facesFrom2D(<int>points.shape[1]),
                      origin,
                      **props)

cpdef Thing pointBox(pt, r = 0.02):
    return Thing(np.array([(pt[0]-r, pt[1]-r, pt[2]-r), (pt[0]+r, pt[1]+r, pt[2]+r)]),
                 util.Ident)

cdef class Polygon(Prim):
    def __init__(self, np.ndarray[np.float64_t, ndim=2] verts,
                 tuple zr, util.Transform origin, **props):
        if not 'name' in props:
            props = mergeProps(props, {'name':util.gensym("box")})
        Prim.__init__(self,
                      vertsFrom2D(verts, zr[0], zr[1]),
                      facesFrom2D(<int>verts.shape[1]),
                      origin,
                      **props)

#################################################################################
# Prim code
#################################################################################

cpdef np.ndarray[np.float64_t, ndim=2] vertsFrom2D(np.ndarray[np.float64_t, ndim=2] verts,
                                                   float zlo, float zhi):
    cdef:
        np.ndarray[np.float64_t, ndim=2] vertsLo, vertsHi
        int i
    vertsLo = verts.copy()
    vertsHi = verts.copy()
    for i in xrange(verts.shape[1]):
        vertsLo[2,i] = zlo
        vertsHi[2,i] = zhi
    return np.hstack([vertsLo, vertsHi])

cpdef np.ndarray[np.float64_t, ndim=2] vertsFrom2DScale(np.ndarray[np.float64_t, ndim=2] verts,
                                                        float zlo, float zhi, float scale):
    cdef:
        np.ndarray[np.float64_t, ndim=2] vertsLo, vertsHi
        int i
    vertsLo = verts.copy()
    vertsHi = verts.copy()
    for i in xrange(verts.shape[1]):
        if vertsLo[0,i] > 0.:
            vertsHi[0,i] = vertsLo[0,i]*scale
        else:
            vertsHi[0,i] = vertsLo[0,i]
        vertsHi[1,i] = vertsLo[1,i]*scale
        vertsLo[2,i] = zlo
        vertsHi[2,i] = zhi
    return np.hstack([vertsLo, vertsHi])

cpdef list facesFrom2D(int n):
    cdef:
        list faces
        int i, ip1
    faces = []
    # Bottom face, order is reversed
    faces.append(np.array(range(n-1, -1, -1), dtype=np.int)) # reversed for bottom face
    faces.append(np.array(range(n,2*n,1), dtype=np.int))     # top face 
    for i in range(n):
        ip1 = (i+1)%n
        faces.append(np.array([i, ip1, n+ip1, n+i], dtype=np.int))
    return faces

# Returns an array of planes, one for each face
# The plane equations is n x + d = 0.  n;d are rows of matrix.
cdef np.ndarray[np.float64_t, ndim=2] primPlanes(np.ndarray[np.float64_t, ndim=2] verts,
                                                 list faces):
    cdef:
        float mag, d
        np.ndarray[np.float64_t, ndim=1] n
        np.ndarray[np.float64_t, ndim=2] planes
        np.ndarray[np.int_t, ndim=1] face
    planes = np.zeros((len(faces), 4), dtype=np.float64)
    for f, face in enumerate(faces):
        n = np.cross(verts[:, face[1]][:3] - verts[:, face[0]][:3],
                     verts[:, face[2]][:3] - verts[:, face[1]][:3])
        mag = np.linalg.norm(n)
        if mag > 0:                     # otherwise, leave as all zeros
            n /= mag
            d = (np.dot(n, verts[:, face[0]][:3]) + \
                 np.dot(n, verts[:, face[1]][:3]) + \
                 np.dot(n, verts[:, face[2]][:3]))/3.0
            planes[f,:3] = n            # unit normal
            planes[f,3] = -d            # -distance from origin
    return planes

# The indices of edges
cdef np.ndarray[np.int_t, ndim=2] primEdges(np.ndarray[np.float64_t, ndim=2] verts,
                                            list faces):
    cdef:
        np.ndarray[np.int_t, ndim=2] edges
        np.ndarray[np.int_t, ndim=1] face
        set done
        int f, v, v1, tail, head, i
    done = set([])
    for f in range(len(faces)):
        face = faces[f]
        k = face.shape[0]
        for v in range(k):
            v1 = (v+1)%k
            tail = face[v]; head = face[v1]
            if not ((tail, head) in done or (head, tail) in done):
                done.add((tail, head))
    edges = np.zeros((len(done), 2), dtype=np.int)
    for i, (tail, head) in enumerate(done):
        edges[i,0] = tail
        edges[i,1] = head
    return edges

cdef Prim xyPrimAux(np.ndarray[np.float64_t, ndim=2] verts,
                    tuple zr, util.Transform origin, dict props):
    cdef:
        np.ndarray[np.float64_t, ndim=2] points
    points = convexHullVertsXY(verts)
    return Prim(vertsFrom2D(points, zr[0], zr[1]),
                facesFrom2D(<int>points.shape[1]),
                origin,
                **props)

cdef Prim boundingRectPrimAux(np.ndarray[np.float64_t, ndim=2] verts,
                              util.Transform origin, dict props):
    cdef:
        np.ndarray[np.float64_t, ndim=2] mu, centered, u, v, bbox
        np.ndarray[np.float64_t, ndim=1] l
    mu = np.resize(np.mean(verts, axis=1), (4,1))
    centered = verts-mu
    (u, l, v) = np.linalg.svd(centered)
    bbox = vertsBBox(np.dot(u.T, centered), None)
    tr = np.hstack([u[:,:3], mu])
    return BoxAligned(bbox, origin).applyTrans(util.Transform(tr),
                                               props.get('frame', 'unspecified'))

# values in d2 take precedence
cdef mergeProps(dict d1, dict d2):
    cdef dict d
    if d1:
        d = d1.copy()
        d.update(d2)
        return d
    else:
        return d2.copy()

def toPrims(obj):
    if isinstance(obj, Shape):
        prims = []
        for p in obj.parts():
            prims.extend(toPrims(p))
        return prims
    elif isinstance(obj, Prim):
        return [obj]
    else:
        return [obj.prim()]

# Returns a list of transforms for the faces.
cpdef list thingFaceFrames(np.ndarray[np.float64_t, ndim=2] planes,
                         util.Transform origin):
    cdef:
        float d, cr
        int i, yi
        list vo
        np.ndarray[np.float64_t, ndim=1] x, y, z
        np.ndarray[np.float64_t, ndim=2] mat
    vo = [origin.matrix[:, i][:3] for i in range(4)]
    frames = []
    for f in range(planes.shape[0]):
        tr = util.Transform(np.eye(4))
        mat = tr.matrix
        z = -planes[f, :3]
        d = -planes[f,3]
        for i in (1, 0, 2):
            cr = abs(np.linalg.norm(np.cross(z,vo[i])))
            yi = i
            if cr >= 0.1: break
        x = -np.cross(z, vo[yi])
        y = np.cross(z, x)
        p = vo[3] - (np.dot(z,vo[3])+d)*z
        for i, v in enumerate((x, y, z, p)):
            mat[:3,i] = v
        frames.append(origin.inverse().compose(tr))
    return frames

# Writing OFF files for a union of convex prims
def writeOff(obj, filename, scale = 1):
    prims = toPrims(obj)
    nv = sum([len(o.vertices()) for o in prims])
    nf = sum([len(o.faces()) for o in prims])
    ne = nv + nf - 2                    # Euler's formula...
    f = open(filename, 'w')
    f.write('OFF\n')
    f.write('%d %d %d\n'%(nv, nf, ne))
    for o in prims:
        verts = o.vertices()
        for p in verts.shape[1]:
            f.write('  %6.3f %6.3f %6.3f\n'%tuple([x*scale for x in verts[0:3,p]]))
    v = 0
    for o in prims:
        faces = o.faces()
        for f in range(len(faces)):
            face = faces[f]
            f.write('  %d'%face.shape[0])
            for k in face.shape[0]:
                f.write(' %d'%(v+face[k]))
            f.write('\n')
        v += len(o.faces())
    f.close()
