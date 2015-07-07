import math

import numpy as np
cimport numpy as np

import util
cimport util

from cpython cimport bool
from geom cimport *
from cut cimport *

import windowManager3D as win
import transformations as transf
from planGlobals import debug, debugMsg

cimport collision
import collision
from collision import primPrimCollides

#################################
# Object classes: Thing, Prim, Shape
#################################

cdef float tiny = 1.0e-6

cdef class Thing:
    """Most unspecific class of object, characterized by a bbox. All
    Things also have a properties dictionary, which includes a name."""
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
            trans[:3, 3] = bboxCenter(bbox)[:3] if (not bbox is None) else np.zeros(3)
            self.thingOrigin = util.Transform(trans)
        self.thingVerts = None
        self.thingPlanes = None
        self.thingEdges = None

    cpdef str name(self):
        return self.properties.get('name', 'noName')

    cpdef util.Transform origin(self):
        return self.thingOrigin

    cpdef np.ndarray[np.float64_t, ndim=2] bbox(self):
        """Returns bbox (not a copy)."""
        return self.thingBBox

    cpdef list parts(self):
        return [self]

    cpdef tuple zRange(self):
        """Summarizes z range of bbox by a tuple"""
        cdef np.ndarray[np.float64_t, ndim=2] bb
        bb = self.bbox()
        return (bb[0,2], bb[1,2])

    cpdef np.ndarray[np.float64_t, ndim=1] center(self):
        """Returns a point at the center of the bbox."""
        cdef np.ndarray[np.float64_t, ndim=1] center = bboxCenter(self.bbox())
        return center

    cpdef np.ndarray[np.float64_t, ndim=2] vertices(self):
        cdef:
            float xlo, ylo, zlo, xhi, yhi, zhi
            np.ndarray[np.float64_t, ndim=2] points, bb
        if self.thingVerts is None:
            bb = self.bbox()
            xlo = bb[0,0]; ylo = bb[0,1]; zlo = bb[0,2]
            xhi = bb[1,0]; yhi = bb[1,1]; zhi = bb[1,2]
            points = np.array([[xlo, ylo, zlo, 1.], [xhi, ylo, zlo, 1.],
                               [xhi, yhi, zlo, 1.], [xlo, yhi, zlo, 1.]]).T
            self.thingVerts = vertsFrom2D(points, zlo, zhi)
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
            self.thingEdges= np.array([[3, 2], [2, 6], [6, 7], [1, 5], [4, 5], [2, 1],
                                       [7, 4], [5, 6], [3, 7], [0, 3], [1, 0], [4, 0]], dtype=np.int)
        return self.thingEdges

    cpdef Thing applyTrans(self, util.Transform trans, str frame='unspecified'):
        """Displace the Thing; returns a Prim."""
        return self.prim().applyTrans(trans, frame)

    cpdef Thing applyLoc(self, util.Transform trans, str frame='unspecified'):
        """Displace the Thing to a location; returns a Prim."""
        return self.applyTrans(trans.compose(self.origin().inverse()), frame)

    # cpdef bool containsPt(self, np.ndarray[np.float64_t, ndim=1] pt):
    #     """Test whether the Thing's bbox contains the Point pt."""
    #     return bboxContains(self.bbox(), pt)

    # cpdef Prim prim(self):
    #     """Constructs a Prim that matches the bbox.  Useful when computing
    #     collisions, etc."""
    #     prim = BoxAligned(self.bbox(), self.thingOrigin, **self.properties)
    #     return prim

    # cpdef bool collides(self, Thing obj):
    #     """Test whether the Thing's collides with another obj, that
    #     could be any of the types of Thing."""
    #     if self.bbox() is None or obj.bbox() is None: return False
    #     if  bboxOverlap(self.bbox(), obj.bbox()): 
    #         if isinstance(obj, (Prim, Shape)): # more general type, pass the buck
    #             return obj.collides(self)
    #         elif isinstance(obj, Thing):
    #             return True             # already checked bbox
    #         else:
    #             raise Exception, 'Unknown obj type'%str(obj)
    #     return False

    # cpdef Shape cut(self, Thing obj, bool isect = False):
    #     if not (obj.bbox() is None) and bboxOverlap(self.bbox(), obj.bbox()):
    #         if isinstance(obj, Shape):
    #             return self.prim().cut(obj, isect=isect)
    #         else:
    #             ans = primPrimCut(self.prim(), obj.prim(), isect=isect)
    #             if ans:
    #                 return Shape([ans], self.origin(), **self.properties)
    #     return None if isect else self

    # cpdef Prim xyPrim(self):
    #     return self.prim()

    # cpdef Prim boundingRectPrim(self):
    #     return self.prim()

    cpdef draw(self, str window, str color = 'black', float opacity = 1.0):
        """Ask the window to draw this object."""
        win.getWindow(window).draw(self, color, opacity)
        
    def __str__(self):
        return self.properties['name']+':'+str(self.bbox().tolist())
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

# This is the base (unchanging) description of a convex chunk.
cdef class BasePrim:
    def __init__(self,
                 np.ndarray[np.float64_t, ndim=2] verts,
                 list faces,
                 **props):
        cdef np.ndarray[np.float64_t, ndim=2] bb
        self.properties = props.copy()
        if not 'name' in self.properties:
            self.properties['name'] = util.gensym('Base')
        self.baseBBox = bb = vertsBBox(verts, None)   # the bbox for original verts
        self.baseVerts = verts
        self.basePlanes = primPlanes(self.baseVerts, faces)
        self.baseEdges = primEdges(self.baseVerts, faces)
        bbPlanes = np.array([[-1.,0.,0., bb[0,0]], [1.,0.,0., -bb[1,0]],
                             [0.,-1,0., bb[0,1]], [0.,1.,0., -bb[1,1]],
                             [0.,0.,-1., bb[0,2]], [0.,0.,1., -bb[1,2]]])
        self.baseFaceFrames = thingFaceFrames(bbPlanes, util.Ident)
        self.baseString = self.properties['name']+':'+str(self.baseBBox.tolist())
    def __str__(self):
        return self.baseString
    def __repr__(self):
        return self.baseString

# This is a located convex chunk, need to provide (verts, faces, origin) or a BasePrim
cdef class Prim:
    def __init__(self,
                 np.ndarray[np.float64_t, ndim=2] verts,
                 list faces,            # since faces have variable length
                 util.Transform origin,
                 BasePrim bs,
                 **props):
        self.properties = props.copy()
        self.primOrigin = origin or util.Ident
        self.basePrim = bs or BasePrim(np.dot(self.primOrigin.inverse().matrix, verts), faces, **props)
        if not 'name' in self.properties:
            self.properties['name'] = util.gensym('Prim')
        self.primVerts = None
        self.primPlanes = None
        self.primBBox = None

    cpdef Prim prim(self):
        return self

    cpdef list parts(self):
        return [self]

    cpdef str name(self):
        return self.properties.get('name', 'noName')

    cpdef util.Transform origin(self):
        return self.primOrigin

    cpdef np.ndarray[np.float64_t, ndim=2] vertices(self):
        if self.primVerts is None:
            self.primVerts = np.dot(self.primOrigin.matrix, self.basePrim.baseVerts)
        return self.primVerts

    cpdef np.ndarray[np.float64_t, ndim=2] planes(self):
        if self.primPlanes is None:
            self.primPlanes = np.dot(self.basePrim.basePlanes,
                                     self.primOrigin.inverse().matrix)
        return self.primPlanes

    cpdef list faces(self):
        return self.basePrim.baseFaces

    cpdef np.ndarray[np.int_t, ndim=2] edges(self):
        return self.basePrim.baseEdges

    cpdef np.ndarray[np.float64_t, ndim=2] bbox(self):
        if self.primBBox is None:
            self.primBBox = vertsBBox(self.vertices(), None)
        return self.primBBox

    cpdef tuple zRange(self):
        """Summarizes z range of bbox by a tuple"""
        cdef np.ndarray[np.float64_t, ndim=2] bb
        bb = self.bbox()
        return (bb[0,2], bb[1,2])

    cpdef Prim applyTrans(self, util.Transform trans, str frame='unspecified',):
        return Prim(None, None,         # basePrim has the relevant info
                    trans.compose(self.primOrigin),
                    self.basePrim,
                    **mergeProps(self.properties, {'frame':frame}))

    cpdef Prim applyLoc(self, util.Transform trans, str frame='unspecified'):
        """Displace the Thing to a location; returns a Prim."""
        return self.applyTrans(trans.compose(self.origin().inverse()), frame)

    cpdef list faceFrames(self):
        return [self.primOrigin.compose(fr) for fr in self.basePrim.baseFaceFrames]

    # cpdef bool containsPt(self, np.ndarray[np.float64_t, ndim=1] pt):
    #     return True if np.all(np.dot(self.planes(), pt.reshape(4,1)) <= tiny) else False

    # cpdef np.ndarray containsPts(self, np.ndarray[np.float64_t, ndim=2] pts):
    #     """Returns array of booleans"""
    #     return np.all(np.dot(self.planes(), pts) <= tiny, axis=0)

    cpdef bool collides(self, obj):
        if isinstance(obj, Shape):  # more general type, pass the buck
            return obj.collides(self)
        else:
            if not bboxGrownOverlap(self.bbox(), obj.bbox()):
                return False
            return primPrimCollides(self, obj)

    cpdef Shape cut(self, obj, bool isect = False):
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
        return xyPrimAux(self.vertices(), self.zRange(), self.primOrigin, self.properties)

    # Compute least inertia box
    cpdef Prim boundingRectPrim(self):
        return boundingRectPrimAux(self.vertices(), self.primOrigin, self.properties)

    cpdef draw(self, str window, str color = 'black', float opacity = 1.0):
        """Ask the window to draw this object."""
        win.getWindow(window).draw(self, color, opacity)

    cpdef tuple desc(self):
        return (tuple([tuple(x) for x in self.basePrim.baseBBox.tolist()]), self.primOrigin)
    def __str__(self):
        return self.properties['name']+':'+str(self.desc())
    def __repr__(self):
        return str(self)
    def __hash__(self):
        return hash(self.desc())
    def __richcmp__(self, other, int op):
        if not (other and isinstance(other, Prim)):
            return True if op == 3 else False
        if op == 2:
            ans = self.name() == other.name() and self.desc() == other.desc()
        elif op == 3:
            ans = self.name() != other.name() or self.desc() != other.desc()
        else:
            ans = False
        return ans

cdef class Shape:
    def __init__(self, list parts, util.Transform origin, **props):
        self.properties = props.copy()
        self.shapeParts = parts
        if origin:
            self.shapeOrigin = origin
        elif parts:
            self.shapeOrigin = util.Transform(bboxOrigin(bboxUnion([p.bbox() for p in parts])))
        else:
            self.shapeOrigin = util.Ident
        if not 'name' in self.properties:
            self.properties['name'] = util.gensym('Shape')
        self.shapeBBox = None

    cpdef list parts(self):
        return self.shapeParts

    cpdef str name(self):
        return self.properties.get('name', 'noName')

    cpdef util.Transform origin(self):
        return self.shapeOrigin

    cpdef np.ndarray[np.float64_t, ndim=2] vertices(self):
        raw_input('Calling for vertices of compound shape')
        return None

    cpdef Shape applyTrans(self, util.Transform trans, str frame='unspecified'):
        return Shape([p.applyTrans(trans, frame) for p in self.parts()],
                     trans.compose(self.shapeOrigin),
                     **mergeProps(self.properties, {'frame':frame}))

    cpdef Shape applyLoc(self, util.Transform trans, str frame='unspecified'):
        """Displace the Thing to a location; returns a Prim."""
        return self.applyTrans(trans.compose(self.origin().inverse()), frame)

    cpdef np.ndarray[np.float64_t, ndim=2] bbox(self):
        if self.shapeBBox is None:
            self.shapeBBox = bboxUnion([x.bbox() for x in self.parts()])
        return self.shapeBBox

    # cpdef bool containsPt(self, np.ndarray[np.float64_t, ndim=1] pt):
    #     for p in self.parts():
    #         if p.containsPt(pt): return True
    #     return False

    # cpdef np.ndarray containsPts(self, np.ndarray[np.float64_t, ndim=2] pts):
    #     """Returns array of booleans"""
    #     return np.array([self.containsPt(pts[i]) for i in range(pts.shape[0])])

    cpdef list faceFrames(self):
        # Use faceFrames for bounding box -- FOR NOW -- should be for convex hull?
        cdef:
            np.ndarray[np.float64_t, ndim=2] bb
        bb = self.bbox()
        bbPlanes = np.array([[-1.,0.,0., bb[0,0]], [1.,0.,0., -bb[1,0]],
                             [0.,-1,0., bb[0,1]], [0.,1.,0., -bb[1,1]],
                             [0.,0.,-1., bb[0,2]], [0.,0.,1., -bb[1,2]]])
        return thingFaceFrames(bbPlanes, self.shapeOrigin)

    cpdef bool collides(self, obj):
        if not bboxGrownOverlap(self.bbox(), obj.bbox()):
            return False
        # Is there any pairwise collision
        for p1 in toPrims(self):
            for p2 in toPrims(obj):
                if p1.collides(p2): return True
        return False

    cpdef Shape cut(self, obj, bool isect = False):
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
        verts = np.hstack([p.vertices() for p in toPrims(self) \
                           if not p.vertices() is None])
        return convexHullPrim(verts, self.shapeOrigin) \
               if self.parts() else None

    # Compute XY convex hull
    cpdef Prim xyPrim(self):
        cdef np.ndarray[np.float64_t, ndim=2] bb
        verts = np.hstack([p.vertices() for p in toPrims(self) \
                           if not p.vertices() is None])
        bb = vertsBBox(verts, None)
        zr = (bb[0,2], bb[1,2])
        return xyPrimAux(verts, zr, self.shapeOrigin, self.properties) \
               if self.parts() else None

    # Compute least inertia box
    cpdef Prim boundingRectPrim(self):
        return boundingRectPrimAux(self.vertices(), self.shapeOrigin, self.properties) \
               if self.parts() else None

    cpdef tuple desc(self):
        return tuple([p.desc() for p in self.parts()])

    cpdef draw(self, str window, str color = 'black', float opacity = 1.0):
        """Ask the window to draw this object."""
        win.getWindow(window).draw(self, color, opacity)
    
    def __str__(self):
        return self.properties['name']+':'+str(self.desc())
    def __repr__(self):
        return str(self)
    def __hash__(self):
        return hash(self.desc())
    def __richcmp__(self, other, int op):
        if not (other and isinstance(other, Shape)):
            return True if op == 3 else False
        if op == 2:
            ans = self.name() == other.name() and self.desc() == other.desc()
        elif op == 3:
            ans = self.name() != other.name() or self.desc() != other.desc()
        else:
            ans = False
        return ans


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
                      origin, None,
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
                      origin, None,
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
                      origin, None,
                      **props)

cdef class BoxAligned(Prim):
    def __init__(self, np.ndarray[np.float64_t, ndim=2] bbox, util.Transform origin, **props):
        cdef:
            float xlo, ylo, zlo, xhi, yhi, zhi
            np.ndarray[np.float64_t, ndim=2] points
        if not 'name' in props:
            props = mergeProps(props, {'name':util.gensym("box")})
        center = util.Transform(bboxOrigin(bbox))
        ((xlo, ylo, zlo), (xhi, yhi, zhi)) = bbox.tolist()
        points = np.array([[xlo, ylo, zlo, 1.], [xhi, ylo, zlo, 1.],
                           [xhi, yhi, zlo, 1.], [xlo, yhi, zlo, 1.]]).T
        if origin:
            center = origin.compose(center)
        Prim.__init__(self,
                      vertsFrom2D(points, zlo, zhi),
                      facesFrom2D(<int>points.shape[1]),
                      center, None,
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
                      origin, None,
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
        vertsHi[0,i] = vertsLo[0,i]*scale
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
                origin, None,
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
    nv = sum([o.vertices().shape[1] for o in prims])
    nf = sum([len(o.faces()) for o in prims])
    ne = 0       # nv + nf - 2                    # Euler's formula...
    fl = open(filename, 'w')
    fl.write('OFF\n')
    fl.write('%d %d %d\n'%(nv, nf, ne))
    for o in prims:
        verts = o.vertices()
        for p in range(verts.shape[1]):
            fl.write('  %6.3f %6.3f %6.3f\n'%tuple([x*scale for x in verts[0:3,p]]))
    v = 0
    for o in prims:
        verts = o.vertices()
        faces = o.faces()
        for f in range(len(faces)):
            face = faces[f]
            fl.write('  %d'%face.shape[0])
            for k in range(face.shape[0]):
                fl.write(' %d'%(v+face[k]))
            fl.write('\n')
        v += verts.shape[1]
    fl.close()

# Reading OFF files
def readOff(filename, name='offObj', scale=1.0):
    fl = open(filename)
    assert fl.readline().split()[0] == 'OFF'
    (nv, nf, ne) = [int(x) for x in fl.readline().split()]
    vl = []
    for v in range(nv):
        vl.append(np.array([scale*float(x) for x in fl.readline().split()]+[1.0]))
    verts = np.vstack(vl).T
    if nf == 0:
        return verts
    faces = []
    for f in range(nf):
        faces.append(np.array([int(x) for x in fl.readline().split()][1:]))
    return shapes.Prim(verts, faces, util.Pose(0,0,0,0), None, name=name)


