from cpython cimport bool
cimport shapes

cdef class Window3D:
     cdef public list capture
     cdef public bool capturing
     cdef public tuple xzOffset
     cdef public window

     cpdef draw(self, shapes.Thing thing, str color = *, float opacity = *)
     cpdef update(self)