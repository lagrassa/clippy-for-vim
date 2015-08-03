cimport shapes
from cpython cimport bool

cpdef double gjkDist(shapes.Prim prim1, shapes.Prim prim2)
cpdef bool chainCollides(tuple CC1, list chains1, tuple CC2, list chains2, double minDist = *, bool ignoreBBox = *)
cpdef bool confSelfCollide(tuple compiledChains, tuple heldCC, double minDist = *)
cpdef tuple confPlaceChains(conf, tuple compiledChains)
cpdef dict chainBBoxes(tuple compiledChains)