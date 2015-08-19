import numpy as np
cimport numpy as np

import objects
cimport objects

from geom cimport bboxUnion
from cpython cimport bool
from cython cimport boundscheck, wraparound

cimport shapes

cimport c_gjk
from c_gjk cimport Object_structure, gjk_distance

from traceFile import debugMsg, debug
import windowManager3D as wm
import hu

@boundscheck(False)
@wraparound(False)
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

cdef inline bool bboxOverlap(np.ndarray[np.float64_t, ndim=2] bb1,
                             np.ndarray[np.float64_t, ndim=2] bb2,
                             double eps):
    # Grow the bbox by eps
    # Due to a Cython bug... cannot convert numpy.bool_ to bool
    return False if \
           bb1[0,0]-eps >= bb2[1,0]+eps or bb1[1,0]+eps <= bb2[0,0]-eps or \
           bb1[0,1]-eps >= bb2[1,1]+eps or bb1[1,1]+eps <= bb2[0,1]-eps or \
           bb1[0,2]-eps >= bb2[1,2]+eps or bb1[1,2]+eps <= bb2[0,2]-eps else True


gwp1 = np.zeros(3, dtype=np.double)
gwp2 = np.zeros(3, dtype=np.double)

cpdef bool chainCollides(tuple CC1, list chains1, tuple CC2, list chains2,
                         double minDist = 1.0e-6):
    cdef Object_structure obj1
    cdef Object_structure obj2
    cdef list framesList1, framesList2, verts
    cdef str frame1, frame2, cname
    cdef dict frames1, frames2
    cdef double dist, md
    cdef np.ndarray[double, ndim=2] bb1, bb2
    cdef np.ndarray[double, ndim=2, mode="c"] ov1, ov2, tr1, tr2
    cdef np.ndarray[double, ndim=1, mode="c"] wp1, wp2

    if CC1 is None or CC2 is None:
        return False

    obj1.rings = NULL
    obj2.rings = NULL
    wp1 = gwp1
    wp2 = gwp2
    
    (frames1, framesList1, chainNames1, frameChain1) = CC1
    (frames2, framesList2, chainNames2, frameChain2) = CC2

    if chains1 is None:
        chains1 = chainNames1.keys()
    if chains2 is None:
        chains2 = chainNames2.keys()

    minDist = max(minDist, 1.0e-6)
    md = minDist*0.5                    # clearance for bbox

    # Check collisions between chains
    for frame2 in framesList2:
        if not frameChain2[frame2] in chains2: continue
        entry2 = frames2[frame2]
        tr2 = entry2.frame
        verts2 = entry2.linkVerts
        if verts2 is None: continue
        bb2 = entry2.bbox
        for ov2 in verts2:
            obj2.numpoints = ov2.shape[0]
            obj2.vertices = <double (*)[3]>ov2.data
            for frame1 in framesList1:
                if not frameChain1[frame1] in chains1: continue
                entry1 = frames1[frame1]
                verts1 = entry1.linkVerts
                if verts1 is None: continue
                bb1 = entry1.bbox
                if not bboxOverlap(bb2, bb1, md): continue
                tr1 = entry1.frame
                for ov1 in verts1:
                    obj1.numpoints = ov1.shape[0]
                    obj1.vertices = <double (*)[3]>ov1.data
                    dist = gjk_distance(&obj1, <double (*)[4]>tr1.data, &obj2, <double (*)[4]>tr2.data,
                                        <double *>wp1.data, <double *>wp2.data, NULL, 0)
                    if dist < minDist:
                        return True
    return False

# Don't let the arms of the robot get too close
minSelfDistance = 0.1

cdef list left_gripper = ['pr2LeftGripper']
cdef list left_arm = ['pr2LeftArm']
cdef list right_gripper = ['pr2RightGripper']
cdef list right_arm = ['pr2RightArm']

cpdef bool confSelfCollide(tuple compiledChains, tuple heldCC,
                           double minDist = minSelfDistance):
    minDist = minDist*minDist  # gjk_distance returns squared distance
    return chainCollides(compiledChains, left_gripper, compiledChains, right_gripper, minDist) \
           or chainCollides(heldCC[1], None, heldCC[0], None, minDist) \
           or chainCollides(compiledChains, left_gripper, compiledChains, right_arm) \
           or chainCollides(compiledChains, right_gripper, compiledChains, left_arm) \
           or chainCollides(compiledChains, left_arm, heldCC[1], None) \
           or chainCollides(compiledChains, right_arm, heldCC[0], None)

cdef dict confCache = {}

cpdef tuple confPlaceChains(conf, tuple compiledChains):
    cdef list framesList, q, qc, vals
    cdef str frame, base, chain
    cdef dict frames
    cdef np.ndarray[double, ndim=2] baseFrame, origin
    cdef objects.Joint joint

    (frames, framesList, chainNames, _) = compiledChains

    if conf in confCache:
        vals = confCache[conf]
        for frame, (fr, bb) in zip(framesList, vals):
            entry = frames[frame]
            entry.frame = fr
            entry.bbox = bb
        return compiledChains
    
    q = []
    for name in chainNames:
        if 'Gripper' in name:
            qc = conf[name]
            q.extend([qc[0]*0.5, qc[0]])
        else:
            q.extend(conf[name])
    vals = []
    for frame in framesList:
        entry = frames[frame]
        baseFrame = frames[entry.base].frame
        if entry.qi is not None:
            origin = np.dot(baseFrame, entry.joint.matrix(q[entry.qi]))
        elif entry.joint is not None:
            origin = np.dot(baseFrame, entry.joint.matrix())
        else:                           # attached objects don't have joint
            origin = baseFrame
        entry.frame = np.ascontiguousarray(origin, dtype=np.double)
        entry.bbox = frameBBoxRad(entry) if entry.link else None
        vals.append((entry.frame, entry.bbox))

        if entry.link and debug('confPlaceChains'):
            print frame
            print entry.link.origin().matrix
            if len(entry.link.parts()) > 1:
                for p in entry.link.parts(): print '+', p.origin().matrix
            entry.link.applyTrans(hu.Transform(origin)).draw('W', 'blue')
    confCache[conf] = vals

    if debug('confPlaceChains'):
        conf.draw('W', 'cyan')
        raw_input('Cyan from conf, Blue from chains - Next?')
        wm.getWindow('W').clear()
    
    return compiledChains

cdef inf = float('inf')
cdef maxBBox = np.array(((-inf,-inf,-inf),(inf,inf,inf)))

cpdef dict chainBBoxesNull(tuple compiledChains):
    cdef dict chainBB
    cdef str chain
    if compiledChains is None: return None
    chainNames = compiledChains[2]
    chainBB = {}
    for (chain, _) in chainNames.iteritems():
        chainBB[chain] = maxBBox
    return chainBB

# This computes the actual bboxes from scratch, but it's too slow.
cpdef dict chainBBoxesFull(tuple compiledChains):
    cdef list framesList, vertFrames, frl, vlist
    cdef str chain, fr
    cdef dict frames
    cdef np.ndarray[double, ndim=2] bbox, verts, mat
    if compiledChains is None: return None
    (frames, framesList, chainNames, _, chainBB) = compiledChains
    for (chain, frl) in chainNames.iteritems():
        vertFrames = [frame for frame in frl if frames[frame].linkVerts]
        # Transform the base vertices into current frame
        vlist = []
        for frame in vertFrames:
            entry = frames[frame]
            mat = entry.frame
            stacked = np.vstack(entry.linkVerts)
            stacked4T = np.vstack([stacked.T, np.ones(stacked.shape[0])])
            vlist.append(np.dot(mat, stacked4T))
        verts = np.hstack(vlist)
        bbox = np.array([np.min(verts, axis=1)[:3], np.max(verts, axis=1)[:3]])
        chainBB[chain] = bbox

    return chainBB

cpdef dict chainBBoxes(tuple compiledChains):
    cdef list framesList, frl, vertFrames
    cdef str chain
    cdef dict frames
    cdef np.ndarray[double, ndim=2] bbox, mat
    if compiledChains is None: return None
    (frames, framesList, chainNames, _, chainBB) = compiledChains
    for (chain, frl) in chainNames.iteritems():
        vertFrames = [frame for frame in frl if frames[frame].linkVerts]
        bbox = np.array(((inf,inf,inf),(-inf,-inf,-inf)))
        for frame in vertFrames:
            entry = frames[frame]
            mat = entry.frame
            rad = entry.radius
            for i in range(3):
                bbox[0,i] = min(bbox[0,i], mat[i,3]-rad)
                bbox[1,i] = max(bbox[1,i], mat[i,3]+rad)
        chainBB[chain] = bbox

    return chainBB

cpdef np.ndarray[double, ndim=2] frameBBoxRad(entry):
    cdef np.ndarray[double, ndim=2] bbox, mat
    cdef double rad
    bbox = np.array(((inf,inf,inf),(-inf,-inf,-inf)))
    mat = entry.frame
    rad = entry.radius
    for i in range(3):
        bbox[0,i] = min(bbox[0,i], mat[i,3]-rad)
        bbox[1,i] = max(bbox[1,i], mat[i,3]+rad)
    return bbox

cpdef np.ndarray[double, ndim=2] frameBBoxFull(entry):
    cdef np.ndarray[double, ndim=2] stacked, stacked4T, verts, mat
    mat = entry.frame
    stacked = np.vstack(entry.linkVerts)
    stacked4T = np.vstack([stacked.T, np.ones(stacked.shape[0])])
    verts = np.dot(mat, stacked4T)
    return np.array([np.min(verts, axis=1)[:3], np.max(verts, axis=1)[:3]])

