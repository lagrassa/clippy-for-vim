from cpython cimport bool

cdef class Hash(object):
    cdef hashValue, descValue

cdef class PoseD(Hash):
    cdef public mu, muTuple, var
     
    cpdef mean(self)
    cpdef mode(self)
    cpdef tuple meanTuple(self)
    cpdef tuple modeTuple(self)
    cpdef tuple variance(self)
    cpdef tuple varTuple(self)
    cpdef tuple desc(self)

cdef class ObjGraspB(Hash):
    cdef public obj, graspDesc, grasp, poseD, delta, support

    cpdef ObjGraspB copy(self)
    cpdef ObjGraspB modifyPoseD(self, mu=*, var=*)
    cpdef tuple desc(self)

cdef class ObjPlaceB(Hash):
    cdef public obj, faceFrames, support, poseD, delta

    cpdef ObjPlaceB copy(self)
    cpdef objFrame(self)
    cpdef ObjPlaceB modifyPoseD(self, mu=*, var=*)
    cpdef tuple desc(self)
    cpdef shape(self, ws)
    cpdef shadow(self, ws)

cdef class Violations(Hash):
    cdef public frozenset obstacles, shadows
    cdef public tuple heldObstacles, heldShadows

    cpdef list allObstacles(self)
    cpdef list allShadows(self)
    cpdef Violations combine(self, obstacles, shadows, heldObstacles=*, heldShadows=*)
    cpdef Violations update(self, viol)
    cpdef double weight(self, weights=*)
    cpdef bool LEQ(self, other)
    cpdef tuple desc(self)
    cpdef tuple names(self)

cpdef list upd(curShapes, newShapes)
