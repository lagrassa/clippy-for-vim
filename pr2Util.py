import pdb
import math
import hu
import random
import copy
from colors import RGBToPyColor, HSVtoRGB
from dist import chiSqFromP
import numpy as np
from traceFile import debug, debugMsg, tr
from planUtil import PoseD, Violations
from pr2Robot import gripperFaceFrame
import planGlobals as glob

class Hashable:
    def __init__(self):
        self.hashValue = None
        self.descValue = None
    def _desc(self):
        if self.descValue is None:
            self.descValue = self.desc()
        return self.descValue
    def __eq__(self, other):
        return hasattr(other, 'desc') and self._desc() == other._desc()
    def __ne__(self, other):
        return not self == other
    def __hash__(self):
        if self.hashValue is None:
            self.hashValue = hash(self._desc())
        return self.hashValue
    def __str__(self):
        return self.__class__.__name__+str(self._desc())
    __repr__ = __str__

# Useful as a default
defaultPoseD = PoseD(hu.Pose(0.0, 0.0, 0.0, 0.0),
                     (0.0, 0.0, 0.0, 0.0))

# Rect grasp set.
# !! Could also have a cylindrical one.  What's the API?
class GDesc(Hashable):
    def __init__(self, obj, frame, dx, dy, dz):
        self.obj = obj                    # ??
        self.frame = frame
        # y is approach direction, x is transverse, z is grip
        self.dx = dx                      # half-widths of a box
        self.dy = dy
        self.dz = dz
        Hashable.__init__(self)
    def desc(self):
        return (self.obj, self.frame, (self.dx, self.dy, self.dz))

def combineViols(*viols):
    v = Violations()
    for viol in viols:
        if viol == None:
            return None
        v = v.update(viol)
    return v

def shadowp(obj):
    if isinstance(obj, str):
        return obj[-7:] == '_shadow'
    else:
        return obj.name()[-7:] == '_shadow'

def shadowName(obj):
    name = obj if isinstance(obj, str) else obj.name()
    return name+'_shadow'

def objectName(obj):
    name = obj if isinstance(obj, str) else obj.name()
    return name[:-7] if '_shadow' in name else name

class NextColor:
    def __init__(self, num, s = .4, v = .99):
        self.max = num
        self.current = 0
        self.s = s
        self.v = v
    def next(self):
        h = (float(self.current) / self.max) * 360
        self.current = (self.current + 1) % self.max
        col = RGBToPyColor(HSVtoRGB(h, self.s, self.v))
        return col

def supportFaceIndex(shape):
    origin = shape.origin()
    for f, ff in enumerate(shape.faceFrames()):
        ffo = origin.compose(ff)
        if abs(1.0 - ffo.matrix[2,2]) < 0.01:
            return f

colorGen = NextColor(20)

def drawPath(path, viol=None, attached=None):
    c = colorGen.next()
    for conf in path:
        conf.draw('W', color=c, attached=attached)
    if viol:
        for v in viol.obstacles:
            v.draw('W', 'red')
        for v in viol.shadows:
            v.draw('W', 'orange')


def graspable(thingName):
    # These global lists are defined when obhects are defined

    if thingName == 'objHeavy': return False

    for prefix in glob.graspableNames:
        if thingName[0:len(prefix)] == prefix:
            return True

def pushable(thingName):

    if thingName == 'objHeavy': return True
        
    # These global lists are defined when objects are defined
    for prefix in glob.pushableNames:
        if thingName[0:len(prefix)] == prefix:
            return True

def permanent(thingName):
    return not (graspable(thingName) or pushable(thingName))

def crashable(thingName):
    for prefix in glob.crashableNames:
        if thingName[0:len(prefix)] == prefix:
            return True



######################################################################
# Store probabilities to describe the domain
######################################################################

class DomainProbs:
    # obsVar, pickVar, and placeVar are diagonal cov 4-tuples (fix this!)
    # pickTolerance is a distance error in x, y, z, theta
    def __init__(self, odoError, obsVar, obsTypeErrProb,
                 pickFailProb, placeFailProb, pushFailProb,
                 pickVar, placeVar, pushVar, pickTolerance,
                 maxGraspVar = (0.015**2, .015**2, .015**2, .03**2),
                 maxPushVar = (0.01**2, .01**2, .01**2, .02**2),
                 placeDelta = (0.005, 0.005, 1.0e-6, 0.005),
                 graspDelta = (0.005, 0.005, 1.0e-6, 0.005),
                 shadowDelta = (0.005, 0.005, 1.0e-6, 0.005),
                 moveConfDelta = (0.001, 0.001, 0.0, 0.002)):
        self.odoError = odoError # std dev per meter / radian in base motion
        self.obsVar = np.diag(obsVar) # error in observations
        self.obsVarTuple = obsVar     # error in observations
        self.obsTypeErrProb = obsTypeErrProb
        self.pickVar = pickVar # error introduced by the picking operation
        self.pickStdev = tuple([np.sqrt(x) for x in pickVar])
        # error introduced by the placing operator;  should have 0 error in z
        self.placeVar = placeVar
        self.placeStdev = tuple([np.sqrt(x) for x in placeVar])
        self.pushVar = pushVar
        self.pushStdev = tuple([np.sqrt(x) for x in pushVar])
        # size of basin of attraction of pick operator
        self.pickTolerance = pickTolerance
        # Don't allow a bigger grasp variance than this
        self.maxGraspVar = maxGraspVar
        self.maxPushVar = maxPushVar
        # Bad failures, like dropping
        self.pickFailProb = pickFailProb
        self.placeFailProb = placeFailProb
        self.pushFailProb = pushFailProb
        self.placeDelta = placeDelta
        self.graspDelta = graspDelta
        self.shadowDelta = shadowDelta
        self.moveConfDelta = moveConfDelta
        minDelta = [2*x for x in self.placeStdev]
        minDelta[2] = 1e-3
        self.minDelta = tuple(minDelta)

    def objBMinVar(self, objName, specialG = None):
        notCrashable = not crashable(objName)
        # Error on a 1 meter move
        objBMinVarStatic = tuple([o**2 for o in self.odoError])
        # Error after two looks
        objBMinVarGrasp = specialG if specialG \
                       else tuple([x/2 for x in self.obsVarTuple])
        # take the max for static objects
        static = tuple([max(a, b) for (a, b) in zip(objBMinVarGrasp,
                                                    objBMinVarStatic)])
        return objBMinVarGrasp if notCrashable else static

######################################################################
        
def shadowWidths(variance, delta, probability):
    numStdDevs =  math.sqrt(chiSqFromP(1-probability, 3))
    assert all([v >= 0 for v in variance])
    return [numStdDevs*(v**0.5)+d for (v,d) in zip(variance, delta)]


memoizerBufferN = 5
class MemoizerViol:
    def __init__(self, name, generator, values = None, bufN = memoizerBufferN):
        self.name = name
        self.generator = generator               # shared
        self.values = values if values else [] # shared
        self.bufN = bufN                       # shared
        self.done = set([])             # not shared
    def __iter__(self):
        return self
    def copy(self):
        # shares the generator and values list, only index differs.
        new = Memoizer(self.name, self.generator, self.values, self.bufN)
        return new
    def next(self):
        dif = len(self.values) - len(self.done)
        # Fill up the buffer, if possible
        if dif < self.bufN:
            for i in range(self.bufN - dif):
                try:
                    val = self.generator.next()
                    self.values.append(val)
                    if val[1].weight() < 1.0: break
                except StopIteration:
                    break
        if len(self.values) > len(self.done):
            elegible = set(range(len(self.values))) - self.done
            # Find min weight index among elegible
            nextI = hu.argmax(list(elegible), lambda i: -self.values[i][1].weight())
            self.done.add(nextI)
            chosen = self.values[nextI]
            debugMsg('Memoizer',
                     self.name,
                     ('weights', [self.values[i][1].weight() for i in elegible]),
                     ('chosen', chosen[1].weight()))
            # if chosen[1].weight() > 5:
            #    raw_input('Big weight - Ok?')
            return chosen
        else:
            raise StopIteration

class Memoizer:
    def __init__(self, name, generator, values = None):
        self.name = name
        self.generator = generator
        self.values = values if values else [name]
        self.i = 0
    def __iter__(self):
        return self
    def copy(self):
        # shares the generator and values list, only index differs.
        if glob.traceGen:
            print '  * Initializing gen =', self.name
        new = Memoizer(self.name, self.generator, self.values)
        return new
    def next(self):
        if self.i < len(self.values)-1:
            i = self.i
            self.i += 1
            return self.values[i+1]
        else:
            val = self.generator.next()
            self.values.append(val)
            self.i += 1
            return val

def bigAngleWarn(conf1, conf2, thr = math.pi/8.):
    if not debug('bigAngleChange'): return
    for chain in ['pr2LeftArm', 'pr2RightArm']:
        joint = 0
        for angle1, angle2 in zip(conf1[chain], conf2[chain]):
            if abs(hu.angleDiff(angle1, angle2)) >= thr:
                print chain, joint, angle1, angle2
                debugMsg('bigAngleChange', 'Big angle change')
            joint += 1

tiny = 1.0e-6
def inside(shape, reg, strict=False):
    bbox = shape.bbox()
    if strict:
        buffer = 0.001
    else:
        buffer = min([0.5*(bbox[1][i] - bbox[0][i]) for i in [0,1]])
    return any(all(insideAux(s, r, buffer) \
                   for s in shape.parts()) for r in reg.parts())

def insideAux(shape, reg, buffer=tiny):
    # all([np.all(np.dot(reg.planes(), p) <= 1.0e-6) for p in shape.vertices().T])
    verts = shape.vertices()
    for i in xrange(verts.shape[1]):
        if not np.all(np.dot(reg.planes(), verts[:,i].reshape(4,1)) <= buffer):
            return False
    return True

def bboxGridCoords(bb, n=5, z=None, res=None):
    eps = 0.001
    ((x0, y0, z0), (x1, y1, z1)) = tuple(bb)
    x0 += eps; y0 += eps
    x1 -= eps; y1 -= eps
    if res:
        dx = res
        dy = res
        nx = int(float(x1 - x0)/res)
        ny = int(float(y1 - y0)/res)
        if nx*ny > n*n:
            for point in bboxGridCoords(bb, n=n, z=z, res=None):
                yield point
    else:
        dx = float(x1 - x0)/n
        dy = float(y1 - y0)/n
        nx = ny = n
    if z is None: z = z0
    for i in range(nx+1):
        x = x0 + i*dx
        for j in range(ny+1):
            y = y0 + j*dy
            yield np.array([x, y, z, 1.])

# Assume an implicit grid 0.01 on a side
def bboxRandomGridCoords(bb, n=5, z=None):
    ((x0, y0, z0), (x1, y1, z1)) = tuple(bb)
    x0 = round(x0, 2)
    y0 = round(y0, 2)
    x1 = round(x1, 2)
    y1 = round(y1, 2)
    if z is None: z = z0
    nx = int(round((x1-x0)/0.01))
    ny = int(round((y1-y0)/0.01))
    maxn = nx*ny
    vals = set([])
    count = 0
    while count < n and len(vals) < maxn:
        i = random.randint(0, nx)
        j = random.randint(0, ny)
        if (i, j) in vals: continue
        else:
            vals.add((i,j))
            count += 1
            x = round(x0 + i*0.01, 2)
            y = round(y0 + j*0.01, 2)
            yield np.array([x, y, z, 1.])

def bboxRandomCoords(bb, n=20, z=None):
    ((x0, y0, z0), (x1, y1, z1)) = tuple(bb)
    if z is None: z = z0
    for i in xrange(n):
        x = random.uniform(x0, x1)
        y = random.uniform(y0, y1)
        yield np.array([x, y, z, 1.])

# prob is probability of generatng a grid point, 1-prob is for random
def bboxMixedCoords(bb, prob, n=20, z=None):
    grid = bboxRandomGridCoords(bb, n=n, z=z)
    rand = bboxRandomCoords(bb, n=n, z=z)
    for i in xrange(n):
        if random.random() <= prob:
            yield next(grid, next(rand))
        else:
            yield next(rand)

def trArgs(tag, names, args, pbs):
    tr(tag, 
       zip(names, args),
       ('objectBs', pbs.objectBs),
       ('held', (pbs.held['left'], pbs.held['right'],
                 pbs.getGraspB('left'), pbs.getGraspB('right'))))

def otherHand(hand):
    return 'left' if hand == 'right' else 'right'

def checkCache(cache, key, valueFn):
    if key not in cache:
        cache[key] = valueFn(*key)
    return cache[key]

def baseConfWithin(bc1, bc2, delta):
    (x1, y1, t1) = bc1
    (x2, y2, t2) = bc2
    bp1 = hu.Pose(x1, y1, 0, t1)
    bp2 = hu.Pose(x2, y2, 0, t2)
    return bp1.near(bp2, delta[0], delta[-1])
    
def confWithin(c1, c2, delta):
    def withinDelta(a, b):
        if isinstance(a, list):
            dd = delta[0]                # !! hack
            return all([abs(a[i] - b[i]) <= dd \
                        for i in range(min(len(a), len(b)))])
        else:
            return a.withinDelta(b, delta)

    if not all([d >= 0 for d in delta]):
        return False

    # We only care whether | conf - targetConf | <= delta
    # Check that the moving frames are all within specified delta
    c1CartConf = c1.cartConf()
    c2CartConf = c2.cartConf()
    robot = c1.robot

    # Also be sure two head angles are the same
    (c1h1, c1h2) = c1['pr2Head']
    (c2h1, c2h2) = c2['pr2Head']

    # Also look at gripper

    return all([withinDelta(c1CartConf[x],c2CartConf[x]) \
                for x in robot.moveChainNames]) and \
                hu.nearAngle(c1h1, c2h1, delta[-1]) and \
                hu.nearAngle(c1h2, c2h2, delta[-1])


def objectGraspFrame(pbs, objGrasp, objPlace, hand):
    # Find the robot wrist frame corresponding to the grasp at the placement
    objFrame = objPlace.objFrame()
    graspDesc = objGrasp.graspDesc[objGrasp.grasp.mode()]
    faceFrame = graspDesc.frame.compose(objGrasp.poseD.mode())
    centerFrame = faceFrame.compose(hu.Pose(0,0,graspDesc.dz,0))
    graspFrame = objFrame.compose(centerFrame)
    # !! Rotates wrist frame to grasp face frame - defined in pr2Robot
    gT = gripperFaceFrame[hand]
    wristFrame = graspFrame.compose(gT.inverse())
    # assert wristFrame.pose()

    if debug('objectGraspFrame'):
        print 'objGrasp', objGrasp
        print 'objPlace', objPlace
        print 'objFrame\n', objFrame.matrix
        print 'grasp faceFrame\n', faceFrame.matrix
        print 'centerFrame\n', centerFrame.matrix
        print 'graspFrame\n', graspFrame.matrix
        print 'object wristFrame\n', wristFrame.matrix

    return wristFrame

def robotGraspFrame(pbs, conf, hand):
    robot = pbs.getRobot()
    _, frames = robot.placement(conf, getShapes=[])
    wristFrame = frames[robot.wristFrameNames[hand]]
    if debug('robotGraspFrame'):
        print 'robot wristFrame\n', wristFrame
    return wristFrame

def removeDuplicateConfs(path):
    inp = []
    for p in path:
        if not inp or inp[-1] != p:
            inp.append(p)
    return inp
