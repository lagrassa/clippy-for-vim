import numpy as np
cimport numpy as np
from cpython cimport bool
cimport shapes
import shapes
cimport util

from scipy.spatial import ConvexHull

#################################################################################
# Convex Hull operations -- uses Numpy arrays
#################################################################################

cpdef np.ndarray convexHullVertsXY(np.ndarray[np.float64_t, ndim=2] verts):
    """
    Compute vertices of convex hull of an array of points projected on xy plane.
    The returned points are sorted by angle, suitable for using as polygon.
    """
    cdef:
        np.ndarray[np.float64_t, ndim=2] xyVerts, chVerts, vecs
        np.ndarray[np.float64_t, ndim=1] angles
        np.ndarray[np.int_t, ndim=1] indices
        list ch
    xyVerts = verts[0:2,:]              # the xy coordinates (2xn)
    # Note the Transpose, points are rows in the input
    ch = ConvexHull(xyVerts.T).simplices.flatten().tolist()
    indices = np.array(sorted(list(set(ch))))  # unique indices of ch vertices
    chVerts = xyVerts[:, indices]      # the xy verts of ch (2xm)
    vecs = chVerts - np.resize(np.mean(chVerts, axis=1), (2,1))
    angles = np.arctan2(vecs[1,:], vecs[0,:])
    return verts[:, indices[np.argsort(angles)]]

cpdef shapes.Prim convexHullPrim(np.ndarray[np.float64_t, ndim=2] verts,
                                 util.Transform origin):
    """
    Return a Prim that is the convex hull of the input verts.
    """
    cdef np.ndarray[np.int_t, ndim=1] indices
    ch = ConvexHull(verts[:3,:].T)
    # unique indices of ch vertices
    indices = np.array(sorted(list(set(ch.simplices.flatten().tolist()))))
    # create the prim from verts and faces
    pr = shapes.Prim(verts[:, indices], chFaces(ch, indices), origin, None)
    # pr.draw('W')
    # raw_input('Next')
    return pr

cdef list chFaces(ch, np.ndarray indices, merge = True):
    """
    Unfortunately ConvexHull triangulates faces, so we have to undo it...
    """
    cdef:
        int fi, i
        bool found
        dict mapping
        list group, mergedFaces, remaining, normSimp, groups
    if merge:
        # This cleans up the faces, but it really slows things down a lot.
        # Normalize the faces, a list of lists
        normSimp = normSimplices(ch)
        # Group simplices on each face
        groups = groupSimplices(ch)
        # Merge the normed simplices; this is reduce with mergedFace
        mergedFaces = []
        for group in groups:
            face = normSimp[group[0]]
            remaining = group[1:]
            while remaining:
                found = False
                for fi in remaining:
                    if adjacent(face, normSimp[fi]):
                        face = mergedFace(face, normSimp[fi])
                        remaining.remove(fi)
                        found = True
                        break
                if not found:
                    raise Exception, 'Could not find adjacent face'
            mergedFaces.append(face)
        # The simplex specifies indices into the full point set.
    else:
        mergedFaces = ch.simplices.tolist()
    mapping = dict([(indices[i], i) for i in range(indices.shape[0])])
    faces = [np.array([mapping[fi] for fi in face]) for face in mergedFaces]
    return faces

cdef bool adjacent(s1, s2):             # two faces share an edge
    return len(set(s1).intersection(set(s2))) == 2 

cdef list normSimplices(ch):
    """
    Sometimes the simplices don't go around CCW, this reverses them if
    they don't.  Stupid wasteful...
    """
    cdef:
        int f
        list normSimp
        np.ndarray[np.int32_t, ndim=2] simplices
        np.ndarray[np.float64_t, ndim=2] eqns, points
        np.ndarray[np.float64_t, ndim=1] normal
    points = ch.points
    simplices = ch.simplices
    eqns = ch.equations
    normSimp = []
    for f in range(simplices.shape[0]):
        normal = np.cross(points[simplices[f][1]] - points[simplices[f][0]],
                          points[simplices[f][2]] - points[simplices[f][1]])
        if np.dot(normal, eqns[f][:3]) > 0:
            normSimp.append(simplices[f].tolist())
        else:
            normSimp.append(simplices[f].tolist()[::-1]) # reverse
    return normSimp

cdef list groupSimplices(ch):
    """
    Find all the simplices that share a face.  Only works for convex
    solids, that is, we assume those simplices are adjacent.
    """
    cdef:
        list groups, face
        set done
        int ne
        np.ndarray[np.int32_t, ndim=2] simplices
        np.ndarray[np.float64_t, ndim=2] eqns
    groups = []
    simplices = ch.simplices
    eqns = ch.equations
    ne = eqns.shape[0]
    face = []                           # current "face"
    done = set([])                      # faces already done
    for e in range(ne):                 # loop over eqns
        if e in done: continue
        face = [e]
        done.add(e)
        for e1 in range(e+1, ne):       # loop for remaining eqns
            if e1 in done: continue
            if np.all(np.abs(eqns[e] - eqns[e1]) < 1e-6):
                # close enough
                face.append(e1)
                done.add(e1)
        # remember this "face"
        groups.append(face)
    # all the faces.
    return groups

cdef list mergedFace(list face, list simplex):
    """
    Insert the new vertex from the simplex into the growing face.
    """
    cdef:
        set setFace, setSimp, diff
        list newFace
        int n, i
    setFace = set(face)
    setSimp = set(simplex)
    common = setSimp.intersection(setFace)
    diff = setSimp.difference(setFace)
    assert len(diff)==1 and len(common)==2
    newFace = face[:]                   # copy
    n = len(face)
    for i in range(n):
        if newFace[i] in common and newFace[(i+1)%n] in common:
            newFace[i+1:] = [diff.pop()] + newFace[i+1:]
            break
    return newFace


#################################################################################
# BBox operations -- uses Numpy arrays
#################################################################################

# verts is a 4xn array
cpdef np.ndarray vertsBBox(np.ndarray[np.float64_t, ndim=2] verts,
                           np.ndarray[np.int_t, ndim=1] indices):
    cdef np.ndarray[np.float64_t, ndim=2] selected
    if indices is None:
        # Do min and max along the points dimension
        return np.array([np.min(verts, axis=1)[:3], np.max(verts, axis=1)[:3]])
    else:
        selected = verts[:, indices]    # pick out some of the points
        return np.array([np.min(selected, axis=1)[:3], np.max(selected, axis=1)[:3]])

cpdef np.ndarray bboxCenter(np.ndarray[np.float64_t, ndim=2] bb,
                            bool base = False):
    cdef np.ndarray[np.float64_t, ndim=1] c
    c =  0.5*(bb[0] + bb[1])
    if base: c[2] = bb[0,2]             # bottom z value
    return c

cpdef np.ndarray bboxDims(np.ndarray[np.float64_t, ndim=2] bb):
    return bb[1] - bb[0]

cpdef np.ndarray bboxUnion(list bboxes):
    minVals = 3*[float('inf')]
    maxVals = 3*[-float('inf')]
    for bb in bboxes:
        for i in range(3):
            minVals[i] = min(minVals[i], bb[0][i])
            maxVals[i] = max(maxVals[i], bb[1][i])
    return np.array([minVals, maxVals])

cpdef np.ndarray bboxIsect(list bboxes):
    return np.array([np.max(np.vstack([bb[0] for bb in bboxes])),
                     np.min(np.vstack([bb[1] for bb in bboxes]))])

cpdef bool bboxOverlap(np.ndarray[np.float64_t, ndim=2] bb1,
                       np.ndarray[np.float64_t, ndim=2] bb2):
    # Touching is not overlap
    # return not (np.any(bb1[0] >= bb2[1]) or np.any(bb1[1] <= bb2[0]))
    # Due to a Cython bug... cannot convert numpy.bool_ to bool
    return False if \
           bb1[0][0] >= bb2[1][0] or bb1[1][0] <= bb2[0][0] or \
           bb1[0][1] >= bb2[1][1] or bb1[1][1] <= bb2[0][1] or \
           bb1[0][2] >= bb2[1][2] or bb1[1][2] <= bb2[0][2] else True

cpdef bool bboxIn(np.ndarray[np.float64_t, ndim=2] bb,
                  np.ndarray[np.float64_t, ndim=1] p):
    # Due to a Cython bug... cannot convert numpy.bool_ to bool
    return True if np.all(p >= bb[0]) and np.all(p <= bb[1]) else False

cpdef bool bboxContains(np.ndarray[np.float64_t, ndim=2] bb,
                        np.ndarray[np.float64_t, ndim=1] pt):
    return bboxIn(bb, pt[:3])

# bb1 inside bb2?
cpdef bool bboxInside(np.ndarray[np.float64_t, ndim=2] bb1,
                      np.ndarray[np.float64_t, ndim=2] bb2):
    return bboxIn(bb2, bb1[0]) and bboxIn(bb2, bb1[1])

cpdef float bboxVolume(np.ndarray[np.float64_t, ndim=2] bb):
    return np.prod(bb[1]-bb[0])

cpdef np.ndarray bboxGrow(np.ndarray[np.float64_t, ndim=2] bb,
                          np.ndarray[np.float64_t, ndim=1] off):
    return np.array([bb[0]-off, bb[1]+off])

# Produces xy bbox, by setting z values to 0
cpdef np.ndarray bboxZproject(np.ndarray[np.float64_t, ndim=2] bb):
    cdef np.ndarray[np.float64_t, ndim=2] bbc
    bbc = bb.copy()
    bbc[0,2] = bbc[1,2] = 0.
    return bbc

cpdef np.ndarray bboxMinkowskiXY(np.ndarray[np.float64_t, ndim=2] bb1,
                                 np.ndarray[np.float64_t, ndim=2] bb2):
    cdef np.ndarray[np.float64_t, ndim=2] b
    b = bboxGrow(bb2, 0.5*bboxDims(bb1))
    b[0,2] = min(bb1[0,2], bb2[0,2])
    b[1,2] = max(bb1[1,2], bb2[1,2])
    return b

# The bbox referenced to its base center
cpdef np.ndarray bboxRefXY(np.ndarray[np.float64_t, ndim=2] bb):
    return bb - bboxCenter(bboxZproject(bb))

cpdef np.ndarray bboxOrigin(np.ndarray[np.float64_t, ndim=2] bb):
    trans = np.eye(4, dtype=np.float64)
    trans[:3, 3] = bboxCenter(bb)[:3]
    return trans

cpdef bool bboxGrownOverlap(np.ndarray[np.float64_t, ndim=2] bb1,
                            np.ndarray[np.float64_t, ndim=2] bb2,
                            float delta = 0.01):
    # Touching is not overlap
    # return not (np.any(bb1[0] >= bb2[1]) or np.any(bb1[1] <= bb2[0]))
    # Due to a Cython bug... cannot convert numpy.bool_ to bool
    return False if \
           bb1[0][0]-delta >= bb2[1][0]+delta or bb1[1][0]+delta <= bb2[0][0]-delta or \
           bb1[0][1]-delta >= bb2[1][1]+delta or bb1[1][1]+delta <= bb2[0][1]-delta or \
           bb1[0][2]-delta >= bb2[1][2]+delta or bb1[1][2]+delta <= bb2[0][2]-delta else True
