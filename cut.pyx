import math
import numpy as np
cimport numpy as np
from cpython cimport bool
cimport shapes
import shapes
cimport geom
cimport util
from scipy.spatial import ConvexHull
from planGlobals import debug

######################################################################
# Cutting one polyhedron with another
######################################################################
cdef float tiny = 1.0e-6

cpdef shapes.Shape primPrimCut(shapes.Prim p1, shapes.Prim p2, bool isect = False):
    """
    Return a list of prims representing (p1 - p2) or (p1 isect p2).
    """
    cdef:
        np.ndarray[np.float64_t, ndim=2] verts1, verts2, planes1, planes2, fxv
    if not geom.bboxOverlap(p1.bbox(), p2.bbox()):
        # no overlap
        return shapes.Shape([], util.Ident) if isect else shapes.Shape([p1], p1.origin())
    planes1 = p1.planes(); verts2 = p2.vertices()
    fxv = np.dot(planes1, verts2)       # nf x nv - signed distance of vert to face
    if np.any(np.all(fxv >= -tiny, axis=1)):
        # some plane of p1 separates the prims
        return shapes.Shape([], util.Ident) if isect else shapes.Shape([p1], p1.origin())
    planes2 = p2.planes(); verts1 = p1.vertices()
    fxv = np.dot(planes2, verts1)
    if np.any(np.all(fxv >= -tiny, axis=1)):
        # some plane of p2 separates
        return shapes.Shape([], util.Ident) if isect else shapes.Shape([p1], p1.origin()) 
    if np.all(np.all(fxv < tiny, axis=0)):
        # p1 is completely inside p2 (all verts inside)
        return shapes.Shape([p1], p1.origin()) if isect else shapes.Shape([], util.Ident)
    return shapes.Shape(primPrimCutAux(p1, p2, isect), p1.origin())

cdef list primPrimCutAux(shapes.Prim p1, shapes.Prim p2, bool isect):
    """
    Return a list of prims representing (p1 - p2) or (p1 isect p2).
    We've now tested the simple cases.
    """
    cdef:
        list ans
        shapes.Prim outP, outPn, inP, inPn # or None
        int f
        np.ndarray[np.float64_t, ndim=2] planes2
    if debug('cut'):
        p1.draw('W', 'blue')
        p2.draw('W', 'red')
        raw_input('Cut red out of blue')
    ans = []                            # build up answer
    inP = p1                            # cut this prim with each face
    planes2 = p2.planes()               # mx4 array
    for f in range(planes2.shape[0]):   # iterate over faces of p2
        (outPn, inPn) = facePrimCut(planes2[f], inP)
        if outPn is None and inPn is None:
            # Should not happen...
            print 'Cut returned both in and out as empty'
            continue
        (outP, inP) = (outPn, inPn)
        if not (outP is None or isect):
            ans.append(outP)            # When doing diff, keep outP
        if not inP:                     # inP is gone
            break
    if isect:                           # return intersection, if any
        if not inP is None: return [inP]
        else: return []
    else:                               # return diff, if any
        if debug('cut'):
            [a.draw('W', 'green') for a in ans]
            raw_input('answer in green')
        return ans

cdef tuple facePrimCut(np.ndarray[np.float64_t, ndim=1] plane,
                       shapes.Prim prim):
    """
    Return a tuple of two prims (out,in) from cutting prim with plane.
    """
    cdef:
        np.ndarray[np.float64_t, ndim=2] verts, cutV
        np.ndarray[np.float64_t, ndim=1] dots, pt, p0, p1
        np.ndarray[np.int_t, ndim=2] edges
        np.ndarray[np.int_t, ndim=1] inv, outv
        float d0, d1, prod, t
        int e
    verts = prim.vertices()
    edges = prim.edges()
    dots = np.dot(plane, verts)         # nv -- plane distances for each v
    inv = np.where(dots <= tiny)[0]        # verts inside plane
    outv = np.where(dots >= -tiny)[0]       # verts outside plane
    if np.any(inv) and not np.any(outv): # everything inside
        return (None, geom.convexHullPrim(verts[:, inv], prim.origin()))
    elif np.any(outv) and not np.any(inv): # everything outside
        return (geom.convexHullPrim(verts[:, outv], prim.origin()), None)
    cutVerts = []                       # find verts from cutting edges with plane
    for e in range(edges.shape[0]):
        d0 = dots[edges[e,0]]; d1 = dots[edges[e,1]]
        prod = d0 * d1
        if prod >= 0: continue          # same side, skip
        t = - d0/(d1 - d0)              # fraction to intersection
        assert 0 <= t <= 1
        p0 = verts[:, edges[e,0]]       # tail
        p1 = verts[:, edges[e,1]]       # head
        pt = p0 + t*(p1 - p0)           # isect point
        pt.resize((4,1))                # column vector
        cutVerts.append(pt)             # remember it
    if not cutVerts:
        raise Exception, 'Inconsistent!'
    cutV = np.hstack(cutVerts)          # matrix of cutverts
    # Return convex hull of vertex sets
    print 'cutting with', plane
    print 'calling ConvexHull'
    print np.hstack([verts[:, outv], cutV])
    outPrim = geom.convexHullPrim(np.hstack([verts[:, outv], cutV]), prim.origin())
    outPrim.draw('W', 'purple')
    print 'returned from ConvexHull'
    print 'calling ConvexHull'
    np.hstack([verts[:, inv], cutV])
    inPrim = geom.convexHullPrim(np.hstack([verts[:, inv], cutV]), prim.origin())
    inPrim.draw('W', 'orange')
    raw_input('returned from ConvexHull')
    return (outPrim, inPrim)
