from cpython cimport bool
import numpy as np
cimport numpy as np

cdef class Transform(object):
    cdef public float eqDistEps, eqAngleEps
    cdef public np.ndarray matrix
    cdef np.ndarray matrixInv
    cdef Quat q 
    cdef Point pt
    cdef str reprString

    cpdef Transform inverse(self)
    cpdef Transform invertCompose(self, Transform trans)
    cpdef Transform compose(self, Transform trans)
    cpdef Pose pose(self, float zthr = *, bool fail = *)
    cpdef Point point(self)
    cpdef Quat quat(self)
    cpdef Point applyToPoint(self, Point point)
    cpdef bool near(self, Transform trans, float distEps, float angleEps)
    cpdef bool withinDelta(self, Transform trans, tuple delta, pr=*)
    cpdef float distance(self, Transform tr)

cdef class Pose(Transform):
    cdef public float x,y,z,theta

    cpdef initTrans(self)
    cpdef setX(self, float x)
    cpdef setY(self, float y)
    cpdef setZ(self, float z)
    cpdef setTheta(self, float theta)
    cpdef Pose average(self, Pose other, float alpha)
    cpdef Point point(self)
    cpdef Pose diff(self, Pose pose)
    cpdef float totalDist(self, Pose pose, float angleScale = *)
    cpdef Pose inversePose(self)
    cpdef tuple xyztTuple(self)
    cpdef Pose corrupt(self, float e, float eAng = *, bool noZ = *)
    cpdef Pose corruptGauss(self, float mu, tuple stdDev, bool noZ = *)

cdef class Point:
    cdef public np.ndarray matrix

    cpdef bool isNear(self, Point point, float distEps)
    cpdef float distance(self, Point point)
    cpdef float distanceXY(self, Point point)
    cpdef float distanceSq(self, Point point)
    cpdef float distanceSqXY(self, Point point)
    cpdef float magnitude(self)
    cpdef tuple xyzTuple(self)
    cpdef Pose pose(self, float angle = *)
    cpdef Point point(self)
    cpdef float angleToXY(self, Point p)
    cpdef Point add(self, Point point)
    cpdef Point sub(self, Point point)
    cpdef Point scale(self, float s)
    cpdef float dot(self, Point p)

cdef class Quat:
    cdef public np.ndarray matrix

cpdef Transform Ident

################################################################################

cdef class SymbolGenerator:
     cdef  dict counts
     cpdef str gensym(self, str prefix = *)

cpdef list smash(list lists)
cpdef bool within(float v1, float v2, float eps)
cpdef bool nearAngle(float a1, float a2, float eps)
cpdef bool nearlyEqual(float x, float y)
cpdef float fixAnglePlusMinusPi(float a)
cpdef fixAngle02Pi(a)
cpdef argmax(list l, f)
cpdef tuple argmaxWithVal(list l, f)
cpdef tuple argmaxIndex(list l, f)
cpdef tuple argmaxIndexWithTies(list l, f)
cpdef float clip(float v, vMin, vMax)
cpdef int sign(float x)
cpdef str prettyString(struct)
cpdef float logGaussian(float x, float mu, float sigma)
cpdef float gaussian(float x, float mu, float sigma)
cpdef list lineIndices(tuple one, tuple two)
cpdef float angleDiff(float x, float y)
cpdef bool inRange(v, tuple r)
cpdef bool rangeOverlap(tuple r1, tuple r2)
cpdef tuple rangeIntersect(tuple r1, tuple r2)
cpdef float average(list stuff)
cpdef tuple tuplify(x)
cpdef list squash(list listOfLists)
cpdef float angleAverage(float th1, float th2, float alpha)
cpdef list floatRange(float lo, float hi, float stepsize)
cpdef pop(x)


