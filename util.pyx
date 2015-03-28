import math
import numpy as np
cimport numpy as np
import random
import operator
from cpython cimport bool
import transf as transf
cimport transf as transf

# Types for transform matrices, for points and point matrices
# Transform is a 4x4 matrix
# Point is a column vector, (4x1)
# Quaternion is a row vector, 1x4

#################################
# Transform
#################################

cdef class Transform:
    """
    Rotation and translation represented as 4 x 4 matrix
    """
    def __init__(self,
                 np.ndarray[np.float64_t, ndim=2] m = None,
                 np.ndarray[np.float64_t, ndim=2] p = None,
                 np.ndarray[np.float64_t, ndim=1] q = None):
        assert (not m is None) or ((not q is None) and (not p is None)) # two ways to specify
        self.eqDistEps = 0.001                 # for equality
        self.eqAngleEps = 0.01                 # for equality
        self.matrix = m
        self.matrixInv = None
        if (not q is None) and (not p is None):
            self.q = Quat(q)
            self.pt = Point(p)
            if m is None:
                self.matrix = np.dot(transf.translation_matrix(p.T[0]),
                                     transf.quaternion_matrix(q))
        else:
            self.q = None
            self.pt = None

    cpdef Transform inverse(self):
        """
        Returns transformation matrix that is the inverse of this one
        """
        if self.matrixInv is None:
            # self.matrixInv =  np.linalg.inv(self.matrix)
            self.matrixInv = np.eye(4)
            self.matrixInv[:3,:3] = self.matrix[:3,:3].T
            self.matrixInv[:3,3] = -np.dot(self.matrixInv[:3,:3], self.matrix[:3,3])
        return Transform(self.matrixInv)

    cpdef Transform invertCompose(self, Transform trans):
        """
        Returns transformation matrix that is the inverse of this one composed with trans
        """
        if self.matrixInv is None:
            self.matrixInv = np.eye(4)
            self.matrixInv[:3,:3] = self.matrix[:3,:3].T
            self.matrixInv[:3,3] = -np.dot(self.matrixInv[:3,:3], self.matrix[:3,3])
        return Transform(np.dot(self.matrixInv, trans.matrix))
    
    def __neg__(self):
        return self.inverse()

    cpdef Transform compose(self, Transform trans):
        """
        Returns composition of self and trans
        """
        cdef Transform tr = Transform(np.dot(self.matrix, trans.matrix))
        return tr

    def __mul__(self, other):
        return self.compose(other)

    cpdef Pose pose(self, float zthr = 0.01, bool fail = True):
        """
        Convert to Pose
        """
        cdef float theta
        if abs(1.0 - self.matrix[2,2]) < zthr:
            theta = math.atan2(self.matrix[1,0], self.matrix[0,0])
            return Pose(self.matrix[0,3], self.matrix[1,3], self.matrix[2,3], theta)
        elif fail:
            print self.matrix
            raise Exception, "Not a valid 2.5D Pose"
        else:
            return None

    cpdef Point point(self):
        if self.pt is None:
            self.pt = Point(np.vstack([self.matrix[:3, 3:4], np.array([1.], dtype=np.float64)]))
        return self.pt

    cpdef Quat quat(self):
        if self.q is None:
            self.q = Quat(transf.quaternion_from_matrix(self.matrix))
        return self.q

    cpdef Point applyToPoint(self, Point point):
        """
        Transform a point into a new point.
        """
        return Point(np.dot(self.matrix, point.matrix))

    cpdef bool near(self, Transform trans, float distEps, float angleEps):
        """
        Return True if point of transform is within distEps of self
        and the quaternion distance is with angleEps.
        """
        # Check the distance between the centers
        if not self.point().isNear(trans.point(), distEps):
            return False
        # Check the angle between the quaternions
        dot = np.dot(trans.quat().matrix, self.quat().matrix)
        dot = max(-1.0, min(1.0, dot))
        if 2*abs(math.acos(abs(dot))) > angleEps:
            return False
        return True
    
    cpdef bool withinDelta(self, Transform trans, tuple delta, pr=False):
        """
        Return True if [x,y,z] of transform is within delta[:3] of self
        and the quaternion distance is with delta[3]
        """
        for i in range(3):
            if abs(self.matrix[i,3] - trans.matrix[i,3]) > delta[i]:
                if pr:
                    print 'xyz'[i], abs(self.matrix[i,3] - trans.matrix[i,3]), '>', delta[i]
                return False
        # The angle between the quaternions
        q1 = trans.quat().matrix
        q2 = self.quat().matrix
        if all(q1 == q2):
            return True
        dot = max(-1.0, min(1.0, np.dot(q1, q2)))
        if (1-dot) <= 1.0e-8:
            return True
        if 2*abs(math.acos(abs(dot))) > delta[3]:
            if pr:
                print 'theta', 2*abs(math.acos(abs(dot))), '>', delta[3]
            return False
        return True

    cpdef float distance(self, Transform tr):
        """
        Return the distance between the x,y,z part of self and the x,y,z
        part of pose.
        """
        return self.point().distance(tr.point())

    def __call__(self, Point point):
        return self.applyToPoint(point)

    def __repr__(self):
        return 'util.Transform(p='+str(self.point().matrix.T)+', q='+str(self.quat().matrix)+')'
    def __str__(self):
        return str(self.matrix)
    def __copy__(self):
        return Transform(self.matrix.copy())
    def __richcmp__(self, other, int op):
        if not (other and isinstance(other, Transform)):
            return True if op == 3 else False
        # print 'Testing equality for:'
        # print self
        # print other
        # print 'epsilons', self.eqDistEps, self.eqAngleEps
        ans = self.near(other, self.eqDistEps, self.eqAngleEps) if op == 2 else False
        # print 'ans=', ans
        return ans
    def __hash__(self):
        return id(self)
    __deepcopy__ = __copy__
    copy = __copy__

cdef Transform Ident = Transform(np.eye(4))            # identity transform

#################################
# Pose
#################################

cdef class Pose(Transform):             # 2.5D transform
    """
    Represent the x,y,z,theta pose of an object in 2.5D space
    """
    def __init__(self, float x, float y, float z, float theta):
        self.x = x
        """x coordinate"""
        self.y = y
        """y coordinate"""
        self.z = z
        """z coordinate"""
        # self.theta = fixAngle02Pi(theta)
        self.theta = theta
        """rotation in radians"""
        self.initTrans()

    cpdef initTrans(self):
        cdef float cosTh
        cdef float sinTh
        cosTh = math.cos(self.theta)
        sinTh = math.sin(self.theta)
        self.reprString = None
        Transform.__init__(self, np.array([[cosTh, -sinTh, 0.0, self.x],
                                           [sinTh, cosTh, 0.0, self.y],
                                           [0.0, 0.0, 1.0, self.z],
                                           [0, 0, 0, 1]], dtype=np.float64))

    cpdef setX(self, float x):
        raw_input('Modifying Pose... not a good idea')
        self.x = x
        self.initTrans()

    cpdef setY(self, float y):
        raw_input('Modifying Pose... not a good idea')
        self.y = y
        self.initTrans()

    cpdef setZ(self, float z):
        raw_input('Modifying Pose... not a good idea')
        self.z = z
        self.initTrans()

    cpdef setTheta(self, float theta):
        raw_input('Modifying Pose... not a good idea')
        self.theta = theta
        self.initTrans()

    cpdef Pose average(self, Pose other, float alpha):
        """
        Weighted average of this pose and other
        """
        return Pose(alpha * self.x + (1 - alpha) * other.x,
                    alpha * self.y + (1 - alpha) * other.y,
                    alpha * self.z + (1 - alpha) * other.z,
                    angleAverage(self.theta, other.theta, alpha))

    cpdef Point point(self):
        """
        Return just the x, y, z parts represented as a C{Point}
        """
        return Point(np.array([[self.x], [self.y], [self.z], [1.0]], dtype=np.float64))

    cpdef Pose diff(self, Pose pose):
        """
        Return a pose that is the difference between self and pose (in
        x, y, z, and theta)
        """
        return Pose(self.x-pose.x,
                    self.y-pose.y,
                    self.z-pose.z,
                    fixAnglePlusMinusPi(self.theta-pose.theta))

    cpdef float totalDist(self, Pose pose, float angleScale = 1):
        return self.distance(pose) + \
               abs(fixAnglePlusMinusPi(self.theta-pose.theta)) * angleScale

    cpdef Pose inversePose(self):
        """
        Return a transformation matrix that is the inverse of the
        transform associated with this pose.
        """
        return super(Pose, self).inverse().pose()

    cpdef tuple xyztTuple(self):
        """
        Representation of pose as a tuple of values
        """
        return (self.x, self.y, self.z, self.theta)

    cpdef Pose corrupt(self, float e, float eAng = 0.0, bool noZ = False):
        """
        Corrupt with a uniformly distributed Pose.
        """
        eAng = eAng or e
        return Pose(np.random.uniform(-e, e),
                    np.random.uniform(-e, e),
                    0.0 if noZ else np.random.uniform(-e, e),
                    np.random.uniform(-eAng, eAng)).compose(self)

    cpdef Pose corruptGauss(self, float mu, tuple stdDev, bool noZ = False):
        """
        Corrupt with a Gaussian distributed Pose.
        """
        perturbation = Pose(random.gauss(mu, stdDev[0]),
                            random.gauss(mu, stdDev[1]),
                            0.0 if noZ else random.gauss(mu, stdDev[2]),
                            random.gauss(mu, stdDev[3]))
        return self.compose(perturbation).pose()

    def __copy__(self):
        return Pose(self.x, self.y, self.z, self.theta)
    def __deepcopy__(self):
        return Pose(self.x, self.y, self.z, self.theta)

    def __str__(self):
        if not self.reprString:
            # An attempt to make string equality useful
            self.reprString = 'Pose[' + prettyString(self.x) + ', ' +\
                              prettyString(self.y) + ', ' +\
                              prettyString(self.z) + ', ' +\
                              (prettyString(self.theta) \
                              if self.theta <= 6.283 else prettyString(0.0))\
                              + ']'
        return self.reprString
    def __repr__(self):
        return 'util.Pose(' + \
               repr(self.x) + ',' + \
               repr(self.y) + ',' + \
               repr(self.z) + ',' + \
               repr(self.theta) + ')'

#################################
# Point
#################################

cdef class Point:
    """
    Represent a point with its x, y, z values
    """
    def __init__(self, np.ndarray[np.float64_t, ndim=2] p): # column matrix
        self.matrix = p

    cpdef bool isNear(self, Point point, float distEps):
        """
        Return true if the distance between self and point is less
        than distEps
        """
        return self.distance(point) < distEps

    cpdef float distance(self, Point point):
        """
        Euclidean distance between two points
        """
        return np.linalg.norm((self.matrix - point.matrix)[:3])

    cpdef float distanceXY(self, Point point):
        """
        Euclidean distance between two XY points
        """
        return np.linalg.norm((self.matrix - point.matrix)[:2])

    cpdef float distanceSq(self, Point point):
        """
        Euclidean distance (squared) between two points
        """
        cdef np.ndarray[np.float64_t, ndim=2] delta
        delta = (self.matrix - point.matrix)[:3]
        return np.dot(delta.T, delta)

    cpdef float distanceSqXY(self, Point point):
        """
        Euclidean distance (squared) between two XY points
        """
        cdef np.ndarray[np.float64_t, ndim=2] delta
        delta = (self.matrix - point.matrix)[:2]
        return np.dot(delta.T, delta)

    cpdef float magnitude(self):
        """
        Magnitude of this point, interpreted as a vector in 3-space
        """
        return np.linalg.norm(self.matrix[:3])

    cpdef tuple xyzTuple(self):
        """
        Return tuple of x, y, z values
        """
        return tuple(self.matrix[:3])

    cpdef Pose pose(self, float angle = 0.0): #Pose
        """
        Return a pose with the position of the point.
        """
        return Pose(self.matrix[0,0], self.matrix[1,0], self.matrix[2,0], angle)

    cpdef Point point(self):
        """
        Return a point, that is, self.
        """
        return self

    def __str__(self):
        w = self.matrix[3]
        if w == 1:
            return 'Point'+ prettyString(self.xyzTuple())
        if w == 0:
            return 'Delta'+ prettyString(self.xyzTuple())
        else:
            return 'PointW'+ prettyString(tuple(self.matrix))

    def __repr__(self):
        return 'util.Point(np.array(' + str(self.matrix) + '))'

    cpdef float angleToXY(self, Point p):
        """
        Return angle in radians of vector from self to p (in the xy projection)
        """
        cdef np.ndarray[np.float64_t, ndim=2] delta
        delta = p.matrix - self.matrix
        return math.atan2(delta[1,0], delta[0,0])

    cpdef Point add(self, Point point):
        """
        Vector addition
        """
        cdef np.ndarray[np.float64_t, ndim=2] summ
        summ = self.matrix + point.matrix
        summ[3,0] = 1.0
        return Point(summ)
    def __add__(self, point):
        return self.add(point)

    cpdef Point sub(self, Point point):
        """
        Vector subtraction
        """
        cdef np.ndarray[np.float64_t, ndim=2] diff
        diff = self.matrix - point.matrix
        diff[3,0] = 1.0
        return Point(diff)
    def __sub__(self, point):
        return self.sub(point)
    cpdef Point scale(self, float s):
        """
        Vector scaling
        """
        cdef np.ndarray[np.float64_t, ndim=2] sc
        sc = self.matrix * s
        sc[3,0] = 1.0
        return Point(sc)
    def __mul__(self, s):
        return self.scale(s)
    cpdef float dot(self, Point p):
        """
        Dot product
        """
        return np.dot(self.matrix[:3].T, p.matrix[:3])

    def __copy__(self):
        cp = Point(self.matrix)
        cp.minAngle = self.minAngle
        cp.topAngle = self.topAngle
        return cp
    __deepcopy__ = __copy__
    copy = __copy__

#################################
# Quat
#################################

cdef class Quat:
    def __init__(self, np.ndarray[np.float64_t, ndim=1] quat):
        self.matrix = quat

######################################################################
# Miscellaneous utilities  - are they used?
######################################################################
    
cpdef list smash(list lists):
    return [item for sublist in lists for item in sublist]

cpdef bool within(float v1, float v2, float eps):
    """
    Return True if v1 is with eps of v2. All params are numbers
    """
    return abs(v1 - v2) < eps

cpdef bool nearAngle(float a1, float a2, float eps):
    """
    Return True if angle a1 is within epsilon of angle a2  Don't use
    within for this, because angles wrap around!
    """
    return abs(fixAnglePlusMinusPi(a1-a2)) < eps

cpdef bool nearlyEqual(float x, float y):
    """
    Like within, but with the tolerance built in
    """
    return abs(x-y)<.0001

cpdef float fixAnglePlusMinusPi(float a):
    """
    A is an angle in radians;  return an equivalent angle between plus
    and minus pi
    """
    cdef float pi2
    cdef int i
    pi2 = 2.0* math.pi
    i = 0
    while abs(a) > math.pi:
        if a > math.pi:
            a = a - pi2
        elif a < -math.pi:
            a = a + pi2
        i += 1
        if i > 10: break                # loop found
    return a

cpdef fixAngle02Pi(a):
    """
    A is an angle in radians;  return an equivalent angle between 0
    and 2 pi
    """
    cdef float pi2 = 2.0* math.pi
    cdef int i = 0
    while a < 0 or a > pi2:
        if a < 0:
            a = a + pi2
        elif a > pi2:
            a = a - pi2
        i += 1
        if i > 10: break                # loop found
    return a

cpdef argmax(list l, f):
    """
    @param l: C{List} of items
    @param f: C{Procedure} that maps an item into a numeric score
    @returns: the element of C{l} that has the highest score
    """
    cdef list vals = [f(x) for x in l]
    return l[vals.index(max(vals))]

cpdef tuple argmaxWithVal(list l, f):
    """
    @param l: C{List} of items
    @param f: C{Procedure} that maps an item into a numeric score
    @returns: the element of C{l} that has the highest score and the score
    """
    best = l[0]
    cdef:
        float bestScore = f(best)
        float xScore
    for x in l:
        xScore = f(x)
        if xScore > bestScore:
            best, bestScore = x, xScore
    return (best, bestScore)

cpdef tuple argmaxIndex(list l, f): #edit - f = lambda x: x
    """
    @param l: C{List} of items
    @param f: C{Procedure} that maps an item into a numeric score
    @returns: the index of C{l} that has the highest score
    """
    cdef:
        int i
        int best = 0
        float bestScore = f(l[best])
        float xScore
    for i from 0 <= i < len(l):
        xScore = f(l[i])
        if xScore > bestScore:
            best, bestScore = i, xScore
    return (best, bestScore)

cpdef tuple argmaxIndexWithTies(list l, f): #edit - f = lambda x: x
    """
    @param l: C{List} of items
    @param f: C{Procedure} that maps an item into a numeric score
    @returns: the index of C{l} that has the highest score
    """
    cdef:
        list best = []
        float bestScore = f(l[0])
        float xScore
        int i
    for i from 0 <= i < len(l):
        xScore = f(l[i])
        if xScore > bestScore:
            best, bestScore = [i], xScore
        elif xScore == bestScore:
            best, bestScore = best + [i], xScore
    return (best, bestScore)

cpdef float clip(float v, vMin, vMax):
    """
    @param v: number
    @param vMin: number (may be None, if no limit)
    @param vMax: number greater than C{vMin} (may be None, if no limit)
    @returns: If C{vMin <= v <= vMax}, then return C{v}; if C{v <
    vMin} return C{vMin}; else return C{vMax}
    """
    if vMin == None:
        if vMax == None:
            return v
        else:
            return min(v, vMax)
    else:
        if vMax == None:
            return max(v, vMin)
        else:
            return max(min(v, vMax), vMin)

cpdef int sign(float x):
    """
    Return 1, 0, or -1 depending on the sign of x
    """
    if x > 0.0:
        return 1
    elif x == 0.0:
        return 0
    else:
        return -1

cpdef str prettyString(struct):
    """
    Make nicer looking strings for printing, mostly by truncating
    floats
    """
    if type(struct) == list:
        return '[' + ', '.join([prettyString(item) for item in struct]) + ']'
    elif type(struct) == tuple:
        return '(' + ', '.join([prettyString(item) for item in struct]) + ')'
    elif type(struct) == dict:
        return '{' + ', '.join([str(item) + ':' +  prettyString(struct[item]) \
                                             for item in struct]) + '}'
    elif type(struct) == float or type(struct) == np.float64:
        struct = round(struct, 3)
        if struct == 0: struct = 0      #  catch stupid -0.0
        return "%5.3f" % struct
    else:
        return str(struct)

cdef class SymbolGenerator:
    """
    Generate new symbols guaranteed to be different from one another
    Optionally, supply a prefix for mnemonic purposes
    Call gensym("foo") to get a symbol like 'foo37'
    """
    def __init__(self): 
        self.counts = {}
    cpdef str gensym(self, str prefix = 'i'):
        count = self.counts.get(prefix, 0)
        self.counts[prefix] = count + 1
        return prefix + '_' + str(count)

gensym = SymbolGenerator().gensym
"""Call this function to get a new symbol"""

cpdef float logGaussian(float x, float mu, float sigma):
    """
    Log of the value of the gaussian distribution with mean mu and
    stdev sigma at value x
    """
    return -((x-mu)**2 / (2*sigma**2)) - math.log(sigma*math.sqrt(2*math.pi))

cpdef float gaussian(float x, float mu, float sigma):
    """
    Value of the gaussian distribution with mean mu and
    stdev sigma at value x
    """
    return math.exp(-((x-mu)**2 / (2*sigma**2))) /(sigma*math.sqrt(2*math.pi))

cpdef list lineIndices(tuple one, tuple two):
    (i0, j0) = one
    (i1, j1) = two
    """
    Takes two cells in the grid (each described by a pair of integer
    indices), and returns a list of the cells in the grid that are on the
    line segment between the cells.
    """
    cdef:
        list ans = [(i0,j0)]
        int di = i1 - i0
        int dj = j1 - j0
        float t = 0.5
        float m
    if abs(di) > abs(dj):               # slope < 1
        m = float(dj) / float(di)       # compute slope
        t += j0
        if di < 0: di = -1
        else: di = 1
        m *= di
        while (i0 != i1):
            i0 += di
            t += m
            ans.append((i0, int(t)))
    else:
        if dj != 0:                     # slope >= 1
            m = float(di) / float(dj)   # compute slope
            t += i0
            if dj < 0: dj = -1
            else: dj = 1
            m *= dj
            while j0 != j1:
                j0 += dj
                t += m
                ans.append((int(t), j0))
    return ans

cpdef float angleDiff(float x, float y):
    cdef:
        float twoPi = 2*math.pi
        float z = (x - y)%twoPi
    if z > math.pi:
        return z - twoPi
    else:
        return z

cpdef bool inRange(v, tuple r):
    return r[0] <= v <= r[1]

cpdef bool rangeOverlap(tuple r1, tuple r2):
    return r2[0] <= r1[1] and r1[0] <= r2[1]

cpdef tuple rangeIntersect(tuple r1, tuple r2):
    return (max(r1[0], r2[0]), min(r1[1], r2[1]))

cpdef float average(list stuff):
    return sum(stuff) * (1.0 / float(len(stuff)))

cpdef tuple tuplify(x):
    if type(x) in (tuple, list):
        return tuple([tuplify(y) for y in x])
    else:
        return x

cpdef list squash(list listOfLists):
    return reduce(operator.add, listOfLists)

# Average two angles
cpdef float angleAverage(float th1, float th2, float alpha):
    return math.atan2(alpha * math.sin(th1) + (1 - alpha) * math.sin(th2),
                      alpha * math.cos(th1) + (1 - alpha) * math.cos(th2))

cpdef list floatRange(float lo, float hi, float stepsize):
    """
    @returns: a list of numbers, starting with C{lo}, and increasing
    by C{stepsize} each time, until C{hi} is equaled or exceeded.

    C{lo} must be less than C{hi}; C{stepsize} must be greater than 0.
    """
    if stepsize == 0:
       print 'Stepsize is 0 in floatRange'
    cdef list result = []
    cdef float v = lo
    while v <= hi:
        result.append(v)
        v += stepsize
    return result

cpdef pop(x):
    if isinstance(x, list):
        if len(x) > 0:
            return x.pop(0)
        else:
            return None
    else:
        try:
            return x.next()
        except StopIteration:
            return None

def tangentSpaceAdd(a, b):
    res = a + b
    for i in range(3, len(res), 4):
        res[i, 0] = fixAnglePlusMinusPi(res[i, 0])
    return res
