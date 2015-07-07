# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np

cimport shapes

cimport c_gjk
from c_gjk cimport Object_structure, gjk_distance

cpdef double gjkDist(shapes.Prim prim1, shapes.Prim prim2):
    cdef Object_structure obj1
    cdef Object_structure obj2
    cdef np.ndarray[double, ndim=2, mode="c"] bv1, bv2, tr1, tr2
    cdef np.ndarray[double, ndim=1, mode="c"] wp1, wp2

    bv1 = np.ascontiguousarray(prim1.basePrim.baseVerts[:3,:].T, dtype=np.double)
    bv2 = np.ascontiguousarray(prim2.basePrim.baseVerts[:3,:].T, dtype=np.double)

    obj1.numpoints = bv1.shape[0]
    obj2.numpoints = bv2.shape[0]
    obj1.vertices = <double (*)[3]>bv1.data
    obj2.vertices = <double (*)[3]>bv2.data
    obj1.rings = NULL
    obj2.rings = NULL

    tr1 = np.ascontiguousarray(prim1.origin().matrix, dtype=np.double)
    tr2 = np.ascontiguousarray(prim2.origin().matrix, dtype=np.double)

    wp1 = np.zeros(3, dtype=np.double)
    wp2 = np.zeros(3, dtype=np.double)

    ans = gjk_distance(&obj1, <double (*)[4]>tr1.data, &obj2, <double (*)[4]>tr2.data,
                        <double *>wp1.data, <double *>wp2.data, NULL, 0)

    return ans


