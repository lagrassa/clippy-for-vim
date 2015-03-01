import math
import numpy as np
cimport numpy as np
from cpython cimport bool
cimport shapes

cpdef bool thingThingCollides(shapes.Thing p1, shapes.Thing p2)
cpdef bool thingThingCollidesAux(shapes.Thing thing1, shapes.Thing thing2, 
                               np.ndarray[np.float64_t, ndim=2] f2xv1)
cpdef bool edgeCross(np.ndarray[np.float64_t, ndim=1] p0, # row vector
                     np.ndarray[np.float64_t, ndim=1] p1, # row vector
                     np.ndarray[np.float64_t, ndim=2] planes,
                     shapes.Thing thing)

