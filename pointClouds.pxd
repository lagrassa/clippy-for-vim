import numpy as np
cimport numpy as np
from cpython cimport bool
cimport shapes
cimport util

cdef class Scan:
    cdef public:
        str name
        str color
        util.Transform headTrans
        util.Transform headTransInverse
        util.Point eye
        np.ndarray vertices, edges, bbox
        tuple scanParams
        list contacts
    cpdef Scan applyTrans(self, util.Transform trans)
    cpdef np.ndarray depthMap(self)
    cpdef bool visible(self, util.Point pt)
    # cpdef draw(self, window, str color = *)

cpdef np.ndarray scanVerts(tuple scanParams, util.Transform pose)
cpdef bool updateDepthMap(Scan thing1, shapes.Prim thing2,
                          np.ndarray[np.float64_t, ndim=1] depth,
                          list contacts, int objId, list onlyUpdate = *)


