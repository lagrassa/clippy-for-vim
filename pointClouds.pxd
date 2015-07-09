import numpy as np
cimport numpy as np
from cpython cimport bool
cimport shapes
cimport hu

cdef class Scan:
    cdef public:
        str name
        str color
        hu.Transform headTrans
        hu.Transform headTransInverse
        hu.Point eye
        np.ndarray vertices, edges, bbox
        tuple scanParams
        list contacts
    cpdef Scan applyTrans(self, hu.Transform trans)
    cpdef np.ndarray depthMap(self)
    cpdef bool visible(self, hu.Point pt)
    cpdef draw(self, window, str color = *)

cpdef np.ndarray[np.float64_t, ndim=2] scanVerts(tuple scanParams, hu.Transform pose)
cpdef bool updateDepthMap(Scan thing1, shapes.Thing thing2,
                          np.ndarray[np.float64_t, ndim=1] depth,
                          list contacts, int objId, list onlyUpdate = *)


