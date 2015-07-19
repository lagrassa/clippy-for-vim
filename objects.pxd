import transformations as transf
import hu
cimport hu
import math
import random 
import numpy as np
cimport numpy as np
import xml.etree.ElementTree as ET
import shapes
import windowManager as win
from cpython cimport bool
from hu cimport Transform
from hu import Transform

cdef class MultiChain:
     cdef public type, chainsInOrder, chainsByName, fnames, baseFname, name
     cpdef placement(self, base, conf, getShapes=*)
     cpdef placementMod(self, base, conf, place, getShapes=*)
cpdef int chainCmp(c1, c2)

cdef class Chain:
     cdef public name, baseFname, joints, links, fnames, chainsInOrder, chainsByName, jointLimits, movingJoints
     cpdef frameTransforms(self, base, jointValues)
     cpdef limits(self)
     cpdef randomValues(self)
     cpdef bool valid(self, list jointValues)
     cpdef placement(self, base, jointValues, getShapes=*)
     cpdef placementMod(self, base, jointValues, place)
     cpdef forwardKin(self, base, jointValues)
     cpdef targetPlacement(self, base, targetValue)
     cpdef stepAlongLine(self, jvf, jvi, stepSize)
     cpdef interpolate(self, jvf, jvi, ratio, stepSize)
     cpdef dist(self, jvf, jvi)
     cpdef normalize(self, jvf, jvi)
     cpdef inverseKin(self, base, targetValue)

cdef class Movable(Chain):
     cpdef inverseKin(self, base, tr)

cdef class Planar(Chain):
     cpdef inverseKin(self, base, tr)

cdef class XYZT(Chain):
     cpdef inverseKin(self, base, tr)

cdef class Permanent(Chain):
     cpdef inverseKin(self, base, target)

cdef class RevoluteDoor(Chain):
     cpdef inverseKin(self, base, tr)

cdef class GripperChain(Chain):
     cpdef frameTransforms(self, base, jointValues)
     cpdef limits(self)
     cpdef bool valid(self, list jointValues)
     cpdef placement(self, base, jointValues, getShapes=*)
     cpdef placementMod(self, base, jointValues, place)
     cpdef stepAlongLine(self, jvf, jvi, stepSize)
     cpdef interpolate(self, jvf, jvi, ratio, stepSize)
     cpdef dist(self, jvf, jvi)
     cpdef normalize(self, jvf, jvi)
     cpdef forwardKin(self, base, jointValues)
     cpdef targetPlacement(self, base, targetValue)
     cpdef inverseKin(self, base, targetValue)

cdef class Joint:
     cdef public name, trans, limits, axis, normalized, subclasses, rotMat

cdef class Prismatic(Joint):
     cpdef np.ndarray matrix(self, val)
     cpdef Transform transform(self, val)
     cpdef bool valid(self, double val)
     cpdef diff(self, a, b)

cdef class Revolute(Joint):
     cpdef np.ndarray matrix(self, val)
     cpdef Transform transform(self, val)
     cpdef bool valid(self, double val)
     cpdef diff(self, a, b)

cdef list normalizedAngleLimits(tuple limits)

cdef class General(Joint):
     cpdef np.ndarray matrix(self, val)
     cpdef Transform transform(self, val)
     cpdef bool valid(self, val)
     cpdef diff(self, a, b)

cdef class Rigid(Joint):
     cpdef np.ndarray matrix(self, val=*)
     cpdef Transform transform(self, val=*)
     cpdef bool valid(self, val=*)
     cpdef diff(self, a, b)
