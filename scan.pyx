from cpython cimport bool
#import copy
import math
import numpy as np
cimport numpy as np
cimport hu
#cimport shapes
from shapes cimport *
from geom cimport *
from ranges cimport realRange, angleRange, splits
#from heapq import heappush, heappop
from collections import deque
import windowManager as win

PI2 = 2*math.pi
DEBUG = False

# thing is an instance of class Prim that is a z extrusion.  There are n scan
# lines between ylo and yhi. Returns a scan map - an array of xlo, xhi for each
# scan line (maybe both be zero when no crossing).

# Edge table
cdef class Edge:
    def __init__(self, x, y, ymax, invSlope):
        self.ymax = ymax
        self.y = y
        self.x = x
        self.invSlope = invSlope
    def __repr__(self):
        args = (self.x, self.y, self.ymax, self.invSlope)
        return "Edge(x=%.2f,y=%.2f,ymax=%.2f,invSlope=%.2f)"%args
    __str__ = __repr__

cdef class Scan:
    def __init__(self, scanArray, ylo, yhi):
        self.ylo = ylo
        self.yhi = yhi
        self.scanArray = scanArray
    def __repr__(self):
        args = (self.ylo, self.yhi)
        return "Scan(y=[%.2f,%.2f])"%args
    __str__ = __repr__

# Global variables, there's probably a way of working around these, but later.
spanCount = 0
# The obsts attribute is [robotObsts, gripObsts]
# robotObsts = {objName: [colliding k values], ...}
# gripObsts = {objName: [colliding g values], ...}
# The objnames are objects at their initial locations, including 'perm'
cdef class Span:
    def __init__(self, lo, hi, yi, xyMap):
        global spanCount
        self.index = spanCount          # for hashing
        spanCount += 1
        # these define the span geometrically
        self.lo = lo
        self.hi = hi
        self.yi = yi
        self.xyMap = xyMap
        # the K obstacle map (obstName : kSet) and G obstacle map (obstName : gSet)
        self.obsts = [{}, {}]           # (k, g)
        self.prev = {}                  # indexed by gXobj condition
        self.kSet = {}                  # reachable k values, indexed by gXobj
        # this must be filled in before a search
        self.nbrs = []                  # neighboring Spans
        # temporary state used by searches
        self.reach = self.lo            # x value reached by search
    cpdef Span copy(self, FreeSpaceMapXY newMap = None):
        c = Span(self.lo, self.hi, self.yi, newMap or self.xyMap)
        # needed when splitting spans
        c.obsts = [self.obsts[0].copy(), self.obsts[1].copy()]
        # The other properties are not copied!!
        return c
    cpdef initializeSearch(self, tuple searchCondition):
        cdef set rk
        cdef str obst
        cdef int k
        self.prev[searchCondition] = None
        rk = set(range(len(self.xyMap.parent.kHandCspaces)))
        for obst in self.obsts[0]:
            if obst in searchCondition[1]:
                for k in self.obsts[0].get(obst, set([])):
                    if k in rk: rk.remove(k)
        self.kSet[searchCondition] = rk
    cpdef bool reachFrom(self, Span other, tuple searchCondition):
        cdef set rk, kObsts
        cdef str obst
        cdef set empty = set([])
        # For now, let's pretend that every K can reach every other
        rk = other.kSet[searchCondition]
        if not rk: return False
        for obst in searchCondition[1]: # obsts
            if searchCondition[0] in self.obsts[1].get(obst, empty):
                # grasp object collision with relevant obstacle
                return False
            rk = rk.difference(self.obsts[0].get(obst, empty))
            if not rk: return False
        self.kSet[searchCondition] = rk
        self.prev[searchCondition] = other
        if other.reach < self.lo:
            self.reach = self.lo
        elif other.reach > self.hi:
            self.reach = self.hi
        else:
            self.reach = other.reach
        return True 
    def __repr__(self):
        args = (self.lo, self.hi)
        return "S(%.2f,%.2f)"%args
    def __hash__(self):
        return self.index

cdef list edgeTable = [None]
cdef bool VERBOSE = False

cpdef scanConvertXYPrim(Prim prim, float ymin, float ymax, int n, verbose = False):
    global edgeTable
    cdef int i, yi, yiStart, yiEnd
    cdef float delta, hix, lox, ytop, ybot
    cdef np.ndarray[np.float64_t, ndim=2] scanArray
    cdef list oldxr, newxr, internal, activeEdges
    cdef Edge edge

    VERBOSE = verbose

    # More generally we should search for the correct face...
    # assert abs(1.0 - prim.planes()[1,2]) < 1.0e-3 # upward facing z face
    if VERBOSE: prim.draw('Lab', 'red')
    if n != len(edgeTable):
        edgeTable = n*[ None ]
    for i in range(n): edgeTable[i] = []
    delta = (ymax - ymin)/float(n)
    scanArray = np.zeros((n,2))              # output
    # xr is the initial xrange for the first scan line (if any)
    yiStart, oldxr = fillEdgeTable(prim, ymin, ymax, delta, edgeTable)
    yiEnd = -1000
    activeEdges = []
    ybot = yiStart*delta + ymin
    for yi in range(yiStart, n):
        ytop = (yi+1)*delta + ymin      # top y for y scanline
        internal = [1000,-1000]
        newxr = [1000,-1000]
        activeEdges = activeEdges + edgeTable[yi]
        # if VERBOSE: print 'yi', yi, 'ytop', ytop, activeEdges
        if not activeEdges: break
        for edge in activeEdges:
            if edge.y > ybot or edge.ymax < ytop: # internal vertex
                if edge.x < internal[0]: internal[0] = edge.x
                if edge.x > internal[1]: internal[1] = edge.x
            if edge.ymax >= ytop:
                edge.x = (ytop - edge.y)*edge.invSlope + edge.x
                edge.y = ytop
                if edge.x < newxr[0]: newxr[0] = edge.x
                if edge.x > newxr[1]: newxr[1] = edge.x
        # if VERBOSE: print '    oldxr', oldxr, 'internal', internal, 'newxr', newxr
        lox = min(oldxr[0], internal[0], newxr[0])
        hix = max(oldxr[1], internal[1], newxr[1])
        if hix >= lox:
            scanArray[yi,0] = lox
            scanArray[yi,1] = hix
            if yi > yiEnd: yiEnd = yi
            if VERBOSE: BoxAligned(np.array([[lox,ytop-delta,0],[hix,ytop,0.1]])).draw('Lab', 'blue')
        # if VERBOSE: print ' -> scanArray', scanArray[yi]
        # update state for next scanline, keep only continuing edges
        activeEdges = [ae for ae in activeEdges if ae.ymax > ytop+0.00001]
        oldxr = newxr
        ybot = ytop
    if VERBOSE: raw_input('ScanArray')
    return Scan(scanArray, yiStart, yiEnd)

cpdef fillEdgeTable(Prim prim, float ymin, float ymax, float delta, list et):
    cdef np.ndarray[np.float64_t, ndim=2] verts, bbox
    cdef np.ndarray[np.int_t, ndim=1] vertIndices
    cdef int i, nverts, yiStart, tail, head, yiLo
    cdef float taily, heady, tailx, headx, invSlope
    cdef list bottomxr
    verts = prim.vertices()
    vertIndices = prim.faces()[1]       # vertex index array
    nverts = vertIndices.shape[0]
    bbox = prim.bbox()
    yiStart = 1000
    bottomxr = [1000, -1000]
    for i in range(nverts):
        tail = vertIndices[i]
        head = vertIndices[(i+1)%nverts]
        taily = verts[1,tail]
        heady = verts[1,head]
        if heady == taily: continue
        if taily > heady:
            (taily, heady) = (heady, taily)
            (tail, head) = (head, tail)
        if taily > ymax or heady < ymin: continue
        if VERBOSE: print 'tail', tail, verts[:,tail]
        if VERBOSE: print 'head', head, verts[:,head]
        tailx = verts[0,tail]
        headx = verts[0,head]
        invSlope = (headx - tailx)/(heady - taily)
        if taily < ymin:
            if VERBOSE: print 'below ymin', tailx, taily
            tailx = tailx + (ymin - taily)*invSlope
            taily = ymin
            bottomxr = [min(bottomxr[0], tailx), max(bottomxr[1], tailx)]
            if VERBOSE: print 'tailx', tailx, 'taily', taily, 'bottomxr', bottomxr
        # Insert into the edgeTable
        yiLo = int((taily - ymin)//delta)
        if 0 <= yiLo < len(edgeTable):  # out of range, ignore it.
            yiStart = min(yiLo, yiStart)
            edgeTable[yiLo] += [Edge(tailx, taily, heady, invSlope)]
    return yiStart, bottomxr

cdef class FreeSpaceMap:
    def __init__(self, bbox, nSplits, objNames, kBaseCspaces, kHandCspaces, gCspaces,
                 nRows = None, thRanges = None):
        self.kBaseCspaces = kBaseCspaces
        self.kHandCspaces = kHandCspaces
        self.gCspaces = gCspaces
        self.ranges = [realRange(bbox[0,i], bbox[1,i])
                       for i in range(3)] + [angleRange(0, 2*math.pi)]
        self.mySplitRanges = [splits(r, n) for r,n in zip(self.ranges, nSplits)]
        self.nSplits = nSplits
        if thRanges:
            self.thRanges = thRanges
            self.mySplitRanges[3] = thRanges
        else:
            self.thRanges = self.mySplitRanges[3]
        self.zRanges = None             # see initialize
        self.zThMap = None              # see initialize
        # nRows is the internal y sampling, nSplits[1] is the "external" grid
        if nRows is None: nRows = nSplits[1]
        self.conditions = set([])       # processed conditions
        empty = frozenset([])
        self.initialize(bbox, nRows)
        self.home = None
        self.homeEntry = None

    cpdef bool setHome(self, homePose):
        self.home = homePose
        vals = homePose.xyztTuple()
        homeArgs = self.coords(*vals)
        xyMap = self.zThMap[homeArgs[2]][homeArgs[3]]
        row = xyMap.map[homeArgs[1]]
        entry = None
        for span in row:
            if span.lo < vals[0] <= span.hi:
                entry = span
                break
        if not entry:
            return False
        self.homeEntry = entry
        entry.reach = vals[0]
        return True

    # This does the actual work of creating the free space map
    cpdef initialize(self, np.ndarray bbox, int nRows):
        cdef int zi, thi, oi, k, nk
        cdef realRange zr, bboxZr
        cdef angleRange thr, thr0
        cdef np.ndarray bb
        cdef list cos, obstaclePrims, initThMaps, zR_obsts
        cdef tuple tag, tagi

        nk = len(self.kBaseCspaces)
        thr0 = self.thRanges[0]
        obstaclePrims = []
        for tag, cspace in \
                [((0, k), cs) for (k,cs) in enumerate(self.kBaseCspaces)]:
            cslice = cspace.getSlice((thr0.lo, thr0.hi))
            assert cslice
            obstaclePrims.extend([((tag, obj), prim) for (obj, prim) in cslice.CObstaclePrims()])
        # initialize the maps for all Z with the base obstacles
        initThMaps = len(self.thRanges)*[None]
        for thi, thr in enumerate(self.thRanges):
            print thi, 'base thr', thr
            cos = []
            for cspace in self.kBaseCspaces:
                cslice = cspace.getSlice((thr.lo, thr.hi))
                assert cslice
                cos.extend(cslice.CObstaclePrims())
            bb = bbox.copy()
            bb[0,2], bb[1,2] = 0.01, 0.011 # just at floor level..
            xyMap = FreeSpaceMapXY(bb, thr, nRows, self)
            initThMaps[thi] = xyMap
            for oi in range(len(obstaclePrims)): # loop over obstacles in zrange
                tagi, o = obstaclePrims[oi]
                obj, co = cos[oi]
                if obj.name() == 'floor': continue
                scan = scanConvertXYPrim(co, bb[0,1], bb[1,1], nRows)
                xyMap.addObjectScan(tagi, scan)
            xyMap.filter(nk)
        # Compute the z ranges -- use obstacles from each hand and grasp cspace
        # (and initialize with base obstacles).
        bboxZr = realRange(bbox[0,2], bbox[1,2])
        obstaclePrims = []
        # Label the primitive COs with ((label, value), original_obj)
        # So, entry in obstaclePrims is (((label, value), original_obj), co_prim)
        # label in {0,1}, 0=k, 1=g; value is one of the k or g indices.
        # Note that this is done for a single theta range, since theta range
        # does not affect the z ranges.
        for tag, cspace in \
                [((0, k), cs) for (k,cs) in enumerate(self.kHandCspaces)] + \
                [((1, g), cs) for (g,cs) in enumerate(self.gCspaces)]:
            cslice = cspace.getSlice((thr0.lo, thr0.hi))
            assert cslice
            obstaclePrims.extend([((tag, obj), prim) for (obj, prim) in cslice.CObstaclePrims()])
        zR_obsts = [(r,o) for (r,o) in obstacleZRanges(obstaclePrims) \
                        if r.overlaps(bboxZr)]
        self.zRanges = [x[0] for x in zR_obsts]
        # update the split ranges
        self.mySplitRanges[2] = self.zRanges
        # create the map
        self.zThMap = [len(self.thRanges)*[None] for i in range(len(self.zRanges))]
        for thi, thr in enumerate(self.thRanges):
            print thi, 'hand thr', thr
            # Get the COs for this theta range
            cos = []
            for cspace in self.kHandCspaces + self.gCspaces:
                cslice = cspace.getSlice((thr.lo, thr.hi))
                cos.extend(cslice.CObstaclePrims())
            for zi, (zr, obsti) in enumerate(zR_obsts):
                bb = bbox.copy()
                bb[0,2], bb[1,2] = zr.lo, zr.hi
                xyMap = FreeSpaceMapXY(bb, thr, nRows, self, initMap=initThMaps[thi])
                self.zThMap[zi][thi] = xyMap
                for oi in obsti:        # loop over obstacles in zrange
                    tagi, obj = obstaclePrims[oi]
                    obj, co = cos[oi]
                    scan = scanConvertXYPrim(co, bb[0,1], bb[1,1], nRows)
                    xyMap.addObjectScan(tagi, scan)
                xyMap.filter(nk)
        self.getNbrs()

    cpdef splitRanges(self):
        return self.mySplitRanges

    cpdef tuple getXYMap(self, float z, float th):
        cdef int zi, thi
        zi = indexInRanges(z, self.zRanges)
        if zi is None: raise Exception, "z value %f not in ranges"%z
        thi = indexInRanges(hu.fixAngle02Pi(th), self.thRanges)
        if thi is None: raise Exception, "theta value %f not in ranges"%th
        xyMap = self.zThMap[zi][thi]
        return zi, thi, xyMap

    cpdef accessXYMap(self, FreeSpaceMapXY xyMap, float x, float y):
        cdef int yi, label
        cdef Span entry
        if xyMap.ylo <= y < xyMap.yhi:
            yi = int((y - xyMap.ylo)//xyMap.yDelta)
            row = xyMap.map[yi]
            for entry in row:
                if entry.lo > x: break
                if entry.hi <= x: continue
                return entry
            return False
        raise Exception, "y value %f not in range"%y

    cpdef markReach(self, tuple condition):
        cdef Span n, entry
        cdef set visited
        self.conditions.add(condition)
        print 'Starting markReach for', condition
        start = self.homeEntry
        start.initializeSearch(condition)
        q = deque([start])
        visited = set([start])
        while len(q) > 0:
            entry = q.popleft()
            for n in entry.nbrs:
                if n in visited: continue
                visited.add(n)
                if n.reachFrom(entry, condition):
                    q.append(n)
        print '...finished markReach'
        
    cpdef list coords(self, float x, float y, float z, float th):
        return [indexInRanges(v, l) for v,l in zip((x,y,z,th), self.splitRanges())]

    cpdef tuple testAccessibleIndices(self, indices, condition):
        (xi, yi, zi, thi) = indices
        ranges = self.splitRanges()
        return self.zThMap[zi][thi].testAccessible(ranges[0][xi], ranges[1][yi], condition)

    cpdef tuple testAccessible(self, realRange xRange, realRange yRange,
                               realRange zRange, angleRange thRange,
                               tuple condition, propagate=True):
        cdef int zi, thi
        cdef realRange zr
        cdef angleRange thr
        if propagate and not condition in self.conditions:
            self.markReach(condition)
        for zi, zr in enumerate(self.zRanges):
            if zRange.overlaps(zr, min(zr.width()/4., zRange.width()/4.)):
                for thi, thr in enumerate(self.thRanges):
                    if thRange.overlaps(thr, min(thr.width()/4., thRange.width()/4.)):
                        xyMap = self.zThMap[zi][thi]
                        if propagate:
                            ans = xyMap.testAccessible(xRange, yRange, condition)
                        else:
                            if DEBUG: print 'zi', zi, 'thi', thi
                            ans = xyMap.testAccessibleLocal(xRange, yRange, condition)
                        if ans and ans[0]: return ans
        return (None, set([]))

    cpdef getNbrs(self):
        ziMax = len(self.zRanges)
        thiMax = len(self.thRanges)
        for zi in range(ziMax):
            for thi in range(thiMax):
                xyMap = self.zThMap[zi][thi]
                xyMap.getNbrs()         # mark neighbors in xyMap
                for nzi, nthi in [(zi-1, thi), (zi+1, thi), (zi, thi-1), (zi, thi+1)]:
                    if 0 > nzi or nzi >= ziMax: continue
                    if nthi < 0: nthi = thiMax # wraparound theta
                    if nthi >= thiMax: nthi = 0 # wraparound theta
                    xyMap.propagate(self.zThMap[nzi][nthi])

    def __str__(self):
        return 'FSM:'+self.name
    __repr__ = __str__

xyMapIndex = 0
count = [0,0]
cdef class FreeSpaceMapXY:
    def __init__(self, bbox, thRange, nRows, parent, initMap = None):
        global xyMapIndex
        self.mapIndex = xyMapIndex
        xyMapIndex += 1
        self.parent = parent
        self.bbox = bbox
        self.thRange = thRange
        self.nRows = nRows
        self.xlo = bbox[0,0]; self.xhi = bbox[1,0]
        self.ylo = bbox[0,1]; self.yhi = bbox[1,1]
        self.yDelta = (self.yhi - self.ylo)/float(self.nRows)
        self.connected = []
        # This is an xy map of the free space.
        if initMap:
            self.map = [[xl.copy(self) for xl in entry] for entry in initMap.map]
        else:
            self.map = [[Span(self.xlo, self.xhi, i, self)] \
                        for i in range(nRows)]

    cdef incRefSet(self, Span span, int label, str objName, int value):
        cdef frozenset old = span.obsts[label][objName]
        # cdef frozenset new, refNew
        # if value in old: return
        # new = old.union(frozenset([value]))
        # refNew = refSets.get(new, None)
        # # count[label] += 1
        # if refNew is None:
        #     # print label, '>', len(refSets), '/', count[label]
        #     refSets[new] = new
        #     refNew = new
        # span.obsts[label][objName] = refNew
        span.obsts[label][objName] = old.union(frozenset([value]))

    cpdef addObjectScan(self, tuple tag, Scan scan):
        cdef int yi, i, value, label
        cdef float x0, x1
        cdef list mapEntry
        cdef Span span
        cdef str objName
        cdef float thr = 0.01
        ((label, value), obj) = tag
        objName = obj.name()
        empty = frozenset([])
        for yi in range(scan.ylo, scan.yhi+1):
            x0, x1 = scan.scanArray[yi] # scan
            if x1 - x0 <= thr: continue
            mapEntry = self.map[yi]
            i = 0
            # print 'scan', (x0, x1), value, objName
            while True:
                # note than entries could be deleted, so len changes
                if i >= len(mapEntry): break
                span = mapEntry[i]

                # print i, mapEntry, span, span.obsts
                # raw_input('Go?')

                if x1 < span.lo: break       # span is to the right of scan
                elif x0 >= span.hi:
                    # span is to the left of scan or scan does not add new obst
                    i += 1
                    continue
                # Overlap
                if not objName in span.obsts[label]:
                    span.obsts[label][objName] = empty
                elif value in span.obsts[label][objName]:
                    i += 1
                    continue
                # Update span
                if x0 <= span.lo and x1 >= span.hi: # scan contains span; just add value
                    self.incRefSet(span, label, objName, value)
                elif x0 <= span.lo and x1 < span.hi: # split at bottom end and stop
                    if x1 - span.lo > thr:
                        left = span.copy()
                        left.hi = x1
                        self.incRefSet(left, label, objName, value)
                        span.lo = x1
                        mapEntry[i:i] = [left]
                    break
                elif x0 > span.lo and x1 >= span.hi: # split at top end and keep going
                    if span.hi - x0 > thr:
                        right = span.copy()
                        right.lo = x0
                        self.incRefSet(right, label, objName, value)
                        span.hi = x0
                        mapEntry[i+1:i+1] = [right]
                        i += 1   # can skip an extra one (the one we just added)
                    i += 1
                    continue
                else:                   # split into three parts...
                    assert x0 > span.lo and x1 < span.hi
                    new = []
                    if x0 - span.lo > thr:
                        left = span.copy()
                        left.hi = x0
                        span.lo = x0
                        new.append(left)
                    new.append(span)
                    if span.hi - x1 > thr:
                        right = span.copy()
                        span.hi = x1
                        right.lo = x1
                        new.append(right)
                    self.incRefSet(span, label, objName, value)
                    mapEntry[i:i+1] = new
                    break

    cpdef filter(self, int nk):
        cdef frozenset fullSet
        cdef int yi, i
        cdef list row, full
        cdef Span entry
        cdef dict kObsts
        cdef str obst
        cdef float thr = 0.01
        fullSet = frozenset(range(nk))
        for yi in range(self.nRows):
            full = []
            row = self.map[yi]
            for i, entry in enumerate(row):
                kObsts = entry.obsts[0]
                if kObsts['perm'] == fullSet:
                    full.append(i)
                if i > 0 and not i in full:
                    prev = row[i-1]
                    if prev.hi - prev.lo < thr or entry.obsts == prev.obsts:
                        entry.lo = prev.lo
                        full.append(i-1)
            for i in full[::-1]:        # delete back to front
                del row[i]

    cpdef getNbrs(self):
        cdef list prevRow, row
        cdef int yi, i
        cdef Span entry
        prevRow = []
        verbose = False
        for yi in range(0, self.nRows):
            if verbose: print 'yi', yi
            row = self.map[yi]
            for j in range(1, len(row)):
                if abs(row[j].lo - row[j-1].hi) < 0.001:
                    row[j].nbrs.append(row[j-1])
                    row[j-1].nbrs.append(row[j])
            for entry in row:
                if verbose: print 'entry', entry
                i = 0
                if verbose: print 'prevRow', prevRow
                while True:
                    if verbose: print 'i', i, 'entry', entry
                    if i >= len(prevRow):
                        break
                    lei = prevRow[i]
                    if lei.hi <= entry.lo:     # prevRow is to the left
                        i += 1                 # keep going
                    elif lei.lo >= entry.hi: # prevRow is to the right
                        break                # stop
                    else:
                        if lei is entry:
                            print yi, lei, entry
                            raw_input('Whoa -- self loop in xyMap getNbrs')
                        lei.nbrs.append(entry)
                        entry.nbrs.append(lei)
                        i += 1      # keep going
            prevRow = self.map[yi]

    cpdef list propagate(self, FreeSpaceMapXY other):
        cdef int yi
        cdef list thisRow, otherRow
        cdef Span thisEntry, otherEntry
        if other is self or other in self.connected: return
        self.connected.append(other)
        other.connected.append(self)
        for yi in range(self.nRows):
            thisRow = self.map[yi]
            otherRow = other.map[yi]
            for thisEntry in thisRow:
                for otherEntry in otherRow: # Could make this incremental
                    if otherEntry.lo < thisEntry.hi and otherEntry.hi > thisEntry.lo: # overlap
                        if thisEntry is otherEntry:
                            raw_input('Whoa -- self loop in propagate')
                        thisEntry.nbrs.append(otherEntry)
                        otherEntry.nbrs.append(thisEntry)

    cpdef float yValue(self, int yi):
        return self.ylo + yi*self.yDelta

    cpdef list yIndexRange(self, realRange yRange):
        cdef int yiLo, yiHi
        yiLo = int((max(yRange.lo, self.ylo) - self.ylo)//self.yDelta)
        yiHi = int((min(yRange.hi, self.yhi) - self.ylo)//self.yDelta)
        if yiHi == yiLo and yRange.hi > yRange.lo:
            yiHi += 1
        return range(yiLo, yiHi)

    cpdef Span testAccessibleSome(self, realRange xRange, realRange yRange, tuple condition):
        cdef int yi, label
        cdef list row
        for yi in self.yIndexRange(yRange):
            if yi < 0 or yi >= len(self.map): continue
            row = self.map[yi]
            for entry in row:
                if entry.lo >= xRange.hi: break
                if entry.hi <= xRange.lo: continue
                if entry.kSet.get(condition, False):
                    return entry
        return None

    cpdef tuple testAccessible(self, realRange xRange, realRange yRange, tuple condition):
        cdef int yi, label
        cdef list row
        for yi in self.yIndexRange(yRange):
            if yi < 0 or yi >= len(self.map): continue
            row = self.map[yi]
            for entry in row:
                if entry.lo >= xRange.hi: break
                if entry.hi <= xRange.lo: continue
                kSet = entry.kSet.get(condition, False)
                if kSet:  return entry, kSet
        return None, set([])

    cpdef set feasibleEntry(self, Span entry, tuple condition):
        cdef set rk
        rk = set(range(len(self.parent.kHandCspaces)))
        for obst in condition[1]: # obsts
            gObsts = entry.obsts[1].get(obst, set([]))
            if condition[0] in gObsts:
                # grasp object collision with relevant obstacle
                return set([])
            kObsts = entry.obsts[0].get(obst, set([])) # k obsts
            rk = rk.difference(kObsts)
            if not rk:
                return set([])
        return rk

    cpdef tuple testAccessibleLocal(self, realRange xRange, realRange yRange, tuple condition):
        cdef int yi, label
        cdef list row
        for yi in self.yIndexRange(yRange):
            if yi < 0 or yi >= len(self.map): continue
            row = self.map[yi]
            for entry in row:
                if entry.lo >= xRange.hi: break
                if entry.hi <= xRange.lo: continue
                kSet = self.feasibleEntry(entry, condition)
                if DEBUG: print '... yi', yi, entry, condition, entry.obsts, 'kSet', kSet
                if kSet:  return entry, kSet
        return None, set([])

    cpdef FreeSpaceMapXY copy(self):
        f = FreeSpaceMapXY(self.bbox, self.thRange, self.nRows, self.parent, initMap=self)
        return f
                
    cpdef draw(self, window, condition, color = 'red', all = False):
        if not all and not condition in self.parent.conditions:
            self.parent.markReach(condition)
        dy = self.yDelta
        y = self.ylo
        for yi in range(self.nRows):
            for entry in self.map[yi]:
                if all or entry.kSet.get(condition, False):
                    Thing(np.array([[entry.lo, y, self.bbox[0,2]],
                                    [entry.hi, y+dy, self.bbox[1,2]]])).draw(window, color)
            y += dy

    def __hash__(self):
        return hash(self.mapIndex)

    def __repr__(self):
        args = (self.bbox.tolist(), self.thRange)
        return 'FreeSpaceMapXY(%s,%s)'%args

cdef int indexInRanges(float val, list rList):
    cdef int i
    for i in range(len(rList)):
        if rList[i].lo <= val < rList[i].hi: return i

cpdef list obstacleZRanges(list obstacles, float eps = 0.001):
    """
    Creates a list of zRanges based on the obstacles present in the world.
    It does some reasoning to merge zRanges if they're really close
    that is, if the start or the end of a certain obstacle is within eps of another one, we merge the start or end, respectively.
    In the end, the function returns a list of zRanges, coupled with the obstacles that exist in each zRange.
    """
    cdef:
        list zValues = []
        int obstIndex, prevObst, curObst, curType, prevType, i, n
        float prevZ, curZ
        list obstList, active
        list zR, zRanges, prev
        tuple zRt

    # We start by putting all the obstacles zRanges into a list and sort it.
    # We sort by zValue and type (start/end) - this works because sorting is lexicographic
    # It is important that, for the same zValue, the start comes before the end, to make it easy to merge.
    for obstIndex,(tag, obst) in enumerate(obstacles):
        zRt = obst.zRange()
        zValues.append([zRt[0], 0, obstIndex])
        zValues.append([zRt[1], 1, obstIndex])

    zValues.sort()

    # Now we merge zValues of the same type that are within eps distance of each other.
    # The zR list now has a different format. Instead of a list of [zRange, type, obstIndex], it is now a list of [zRange, type, [obstIndex]], due to the merging.
    (prevZ, prevType, prevObst) = zValues[0]
    zR = [[prevZ, prevType, [prevObst]]]
    for i from 1 <= i < len(zValues):
        (prevZ, prevType, obstList) = prev =  zR[-1]
        (curZ, curType, curObst) = zValues[i]
        if abs(prevZ-curZ) <= eps and prevType == curType:
            # We only merge zRange values of the same type (start/end) and within eps of each other.
            # If the type is 0, all we need to do is to append the new obstacle to the previous zRange
            # otherwise, we also need to update the end point to the current zRange's end point.
            if prevType == 1:
                prev[0] = curZ
            obstList.append(curObst)
        else: # If we don't merge, we just add the new zRange value to the list.
            zR.append([curZ, curType, [curObst]])

    # Now that we're done merging, we must create the final zRanges, and maintain which obstacles are in each.
    # We use an active list that keeps track of the obstacles that have "started" but not ended, so that we know which obstacles are in the current zRange.
    (curZ, curType, obstList) = zR[0]

    if curType != 0:
        raise Exception, 'Ill formed z Ranges: ' + str(zR)

    zRanges = [[curZ, None]]
    active = obstList[:]

    n = len(zR)
    for i from 1 <= i < n:
        # Right now, the type only matters to maintain the active obstacles list, and doesn't dictate which values start/end a zRange.
        # Whenever a new start is found, a new zRange is created, for it contains a new obstacle, and therefore that start value is used to
        # end the previous zRange.
        # The list of active obstacles is also added, so we know what obstacles exist in that zRange.
        (curZ, curType, obstList) = zR[i]
        zRanges[-1][0] = realRange(zRanges[-1][0], curZ)
        zRanges[-1][1] = active[:]

        # If we aren't in the last entry, we add a new zRange
        if i < n-1:
            zRanges.append([curZ,None])
        # If the type is 0, then this is a start and has obstacles to be added.
        # Otherwise, there are obstacles to be removed
        if curType ==0:
            active.extend(obstList)
        else:
            for obstIndex in obstList:
                active.remove(obstIndex)

    if active:
        print 'Obstacles are still active at end:', active
        raise Exception, 'Ill formed z Ranges: ' + str(zR)
    return zRanges

def printMap(msg, xyMap, scan):
    print msg
    for yi in range(len(xyMap.map)):
        print yi, xyMap.map[yi]
        if scan:
            print '  scan', scan.scanArray[yi]
