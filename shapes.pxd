import math
import numpy as np
cimport numpy as np
import util
cimport util
from cpython cimport bool

cdef class Thing:
    cdef public dict properties
    cdef public np.ndarray thingVerts, thingBBox, thingCenter, thingPlanes, thingEdges
    cdef public list thingFaceFrames
    cdef public Prim thingPrim
    cdef public util.Transform thingOrigin

    cpdef np.ndarray[np.float64_t, ndim=2] bbox(self)
    cpdef str name(self)
    cpdef list faceFrames(self)
    cpdef util.Transform origin(self)
    cpdef list parts(self)
    cpdef tuple zRange(self)
    cpdef np.ndarray[np.float64_t, ndim=1] center(self)
    cpdef np.ndarray[np.float64_t, ndim=2] vertices(self)
    cpdef np.ndarray[np.float64_t, ndim=2] planes(self)
    cpdef np.ndarray[np.int_t, ndim=2] edges(self)
    cpdef Prim prim(self)
    cpdef Thing applyTransMod(self, util.Transform trans, Thing shape, str frame=*)
    cpdef Thing applyLocMod(self, util.Transform trans, Thing shape, str frame=*)
    cpdef Thing applyTrans(self, util.Transform trans, str frame=*)
    cpdef Thing applyLoc(self, util.Transform trans, str frame=*)
    cpdef bool containsPt(self, np.ndarray[np.float64_t, ndim=1] pt)
    cpdef bool collides(self, Thing obj)
    cpdef Shape cut(self, Thing obj, bool isect = *)
    cpdef draw(self, str window, str color = *, float opacity = *)
    cpdef Prim xyPrim(self)
    cpdef Prim boundingRectPrim(self)

cdef class Prim(Thing):
    cdef public np.ndarray primVerts, primPlanes, primEdges
    cdef public list primFaces

    cpdef Prim prim(self)
    cpdef list parts(self)
    cpdef np.ndarray[np.float64_t, ndim=2] vertices(self)
    cpdef list faces(self)
    cpdef np.ndarray[np.float64_t, ndim=2] planes(self)
    cpdef np.ndarray[np.int_t, ndim=2] edges(self)
    cpdef Thing applyTrans(self, util.Transform trans, str frame=*)
    cpdef Thing applyTransMod(self, util.Transform trans, Thing shape, str frame=*)
    cpdef bool containsPt(self, np.ndarray[np.float64_t, ndim=1] pt)
    cpdef Shape cut(self, Thing obj, bool isect = *)
    cpdef np.ndarray containsPts(self, np.ndarray[np.float64_t, ndim=2] pts)
    cpdef bool collides(self, Thing obj)
    cpdef Prim xyPrim(self)
    cpdef Prim boundingRectPrim(self)


cdef class Shape(Thing):
    cdef public list compParts
    cdef public np.ndarray compVerts
    cpdef list parts(self)
    cpdef emptyP(self)
    cpdef np.ndarray[np.float64_t, ndim=2] vertices(self)
    cpdef Thing applyTrans(self, util.Transform, str frame=*)
    cpdef Thing applyTransMod(self, util.Transform, Thing shape, str frame=*)
    cpdef bool containsPt(self, np.ndarray[np.float64_t, ndim=1] pt)
    cpdef np.ndarray containsPts(self, np.ndarray[np.float64_t, ndim=2] pts)
    cpdef bool collides(self, Thing obj)
    cpdef Shape cut(self, Thing obj, bool isect = *)
    cpdef Prim prim(self)
    cpdef Prim xyPrim(self)
    cpdef Prim boundingRectPrim(self)

cdef class Box(Prim):
    cdef nothing

cdef class Ngon(Prim):
    cdef nothing

cdef class BoxAligned(Prim):
    cdef nothing

cdef class Polygon(Prim):
    cdef nothing
