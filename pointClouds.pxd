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
    # cpdef draw(self, window, str color = *)

cpdef np.ndarray scanVerts(tuple scanParams, hu.Transform pose)
cpdef bool updateDepthMap(Scan thing1, shapes.Prim thing2,
                          np.ndarray[np.float64_t, ndim=1] depth,
                          list contacts, int objId, list onlyUpdate = *)

cdef class Raster:
     cdef public:
          double screenWidth, screenHeight, nearClippingPlane, farClippingPLane, focalLength
          int imageWidth, imageHeight
          tuple screenCoordinates
          np.ndarray depthBuffer, frameBuffer
     cpdef int convertToRaster(self, np.ndarray vertexCamera, np.ndarray vertexRaster)
     cpdef int reset(self)
     cpdef int update(self, prim, int objId, onlyUpdate=*)
     cpdef int countId(self, int objId)

