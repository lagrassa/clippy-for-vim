import math
import util
import copy
from colors import RGBToPyColor, HSVtoRGB
from dist import chiSqFromP
import numpy as np
import dist

class Hashable:
    def __eq__(self, other):
        return hasattr(other, 'desc') and self.desc() == other.desc()
    def __neq__(self, other):
        return not self == other
    def __hash__(self):
        return hash(self.desc())
    def __str__(self):
        return self.__class__.__name__+str(self.desc())
    __repr__ = __str__

# Ultimately, we will also need a discrete value to indicate which
# stable face the object is resting on.
class PoseD(Hashable):
    def __init__(self, mu, var):
        if isinstance(mu, tuple):
            mu = util.Pose(*mu)
        # assert isinstance(mu, util.Pose)
        self.mu = mu
        self.muTuple = self.mu.xyztTuple() if mu else None
        assert isinstance(var, tuple)
        self.var = var
        
    def mean(self):
        return self.mu
    mode = mean

    def meanTuple(self):
        return self.mu.xyztTuple()
    modeTuple = meanTuple

    def variance(self):
        return self.var
    varTuple = variance

    def desc(self):
        return (self.muTuple, self.var)

# Useful as a default
defaultPoseD = PoseD(util.Pose(0.0, 0.0, 0.0, 0.0),
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
    def desc(self):
        return (self.obj, self.frame, (self.dx, self.dy, self.dz))

# These represet SETS of Gaussian distributions

# This represent poses for the finger tip in the contact face.
class ObjGraspB(Hashable):
    def __init__(self, obj, graspDesc, grasp, poseD, var = None, delta = None):
        self.obj = obj
        self.graspDesc = graspDesc
        # grasp is an index into the graspDesc
        if  isinstance(grasp, dist.DDist):
            self.grasp = grasp                # this is a DDist for index
        else:
            self.grasp = dist.DeltaDist(grasp)
        # This is PoseD for the mode of the grasp
        if isinstance(poseD, (util.Pose, util.Transform)):
            self.poseD = PoseD(poseD.pose(), var or 4*(0.0,))
        elif poseD == None or hasattr(poseD, 'muTuple'):
            self.poseD = poseD
        else:
            print 'Unknown poseD type'
            assert None
        self.delta = delta or 4*(0.0,)
    def modifyPoseD(self, mu=None, var=None):
        gB = copy.copy(self)
        gB.poseD = PoseD(mu or self.poseD.mu,
                         var or self.poseD.var)
        return gB
    def desc(self):
        if self.obj == 'none':
            return None
        else:
            return (self.obj, self.grasp.mode(), self.poseD, self.delta)
    def __eq__(self, other):
        if other is None and self.obj == 'none': return True
        return hasattr(other, 'desc') and self.desc() == other.desc()
    def __neq__(self, other):
        return not self == other
    
# This represents poses for the face frame of the support face
class ObjPlaceB(Hashable):
    def __init__(self, obj, faceFrames, support, poseD, var = None, delta = None):
        self.obj = obj
        self.faceFrames = faceFrames
        # support is an index into the faceFrames
        if isinstance(support, dist.DDist):
            self.support = support            # this is a DDist for index
        else:
            self.support = dist.DeltaDist(support)
        # This is PoseD for the mode of the support
        if isinstance(poseD, (util.Pose, util.Transform)):
            self.poseD = PoseD(poseD.pose(), var or 4*(0.0,))
        elif poseD == None or hasattr(poseD, 'muTuple'):
            self.poseD = poseD
        else:
            print 'Unknown poseD type'
            assert None
        self.delta = delta or 4*(0.0,)
    # get the origin pose for the object corresponding to mode of poseD.
    def objFrame(self):
        faceFrame = self.faceFrames[self.support.mode()]
        return self.poseD.mode().compose(faceFrame.inverse())
    def modifyPoseD(self, mu=None, var=None):
        pB = copy.copy(self)
        pB.poseD = PoseD(mu or self.poseD.mu,
                         var or self.poseD.var)
        return pB
    def desc(self):
        return (self.obj, self.support.mode(), self.poseD, self.delta)
    def shape(self, ws):                # in WorldState, e.g. shadow world
        return ws.world.getObjectShapeAtOrigin(self.obj).applyLoc(self.objFrame())
    def shadow(self, ws):
        if shadowName(self.obj) in ws.objectShapes:
            return ws.objectShapes[shadowName(self.obj)].applyLoc(self.objFrame())

# represent the "removable" collisions wth obstacles and shadows.
class Violations(Hashable):
    def __init__(self, obstacles=frozenset([]), shadows=frozenset([]), penalty=0.0):
        self.obstacles = frozenset(obstacles)
        self.shadows = frozenset(shadows)
        assert len(set([o.name() for o in self.obstacles])) == len(self.obstacles)
        assert len(set([o.name() for o in self.shadows])) == len(self.shadows)
        self.penalty = penalty
    def empty(self):
        return (not self.obstacles) and (not self.shadows)
    def combine(self, obstacles, shadows):
        return Violations(frozenset(self.obstacles.union(obstacles)),
                          frozenset(self.shadows.union(shadows)))
    def update(self, viol):
        def upd(curShapes, newShapes):
            curDict = dict([(o.name(), o) for o in curShapes])
            newDict = dict([(o.name(), o) for o in newShapes])
            curDict.update(newDict)
            return curDict.values()
        return Violations(frozenset(upd(self.obstacles, viol.obstacles)),
                          frozenset(upd(self.shadows, viol.shadows)))
    def weight(self):
        return len(self.obstacles) + 0.5*len(self.shadows)+self.penalty
    def LEQ(self, other):
        return self.weight() <= other.weight()
    def desc(self):
        return (self.obstacles, self.shadows, self.penalty)
    def __repr__(self):
        return 'Violations%s'%str(self.desc())
    __str__ = __repr__

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

######################################################################
# Store probabilities to describe the domain
######################################################################

class DomainProbs:
    # obsVar, pickVar, and placeVar are diagonal cov 4-tuples (fix this!)
    # pickTolerance is a distance error in x, y, z, theta
    def __init__(self, odoError, obsVar, obsTypeErrProb,
                 pickFailProb, placeFailProb,
                 pickVar, placeVar, pickTolerance):
        self.odoError = odoError # std dev per meter / radian in base motion
        self.obsVar = np.diag(obsVar) # error in observations
        self.obsVarTuple = obsVar     # error in observations
        self.obsTypeErrProb = obsTypeErrProb
        self.pickVar = pickVar # error introduced by the picking operation
        self.pickStdev = tuple([np.sqrt(x) for x in pickVar])
        # error introduced by the placing operator;  should have 0 error in z
        self.placeVar = placeVar
        self.placeStdev = tuple([np.sqrt(x) for x in placeVar])
        # size of basin of attraction of pick operator
        self.pickTolerance = pickTolerance 
        # Bad failures, like dropping
        self.pickFailProb = pickFailProb
        self.placeFailProb = placeFailProb
        
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
            nextI = argmax(list(elegible), lambda i: -self.values[i][1].weight())
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
        self.values = values if values else []
        self.i = 0
    def __iter__(self):
        return self
    def copy(self):
        # shares the generator and values list, only index differs.
        new = Memoizer(self.name, self.generator, self.values)
        return new
    def next(self):
        if self.i < len(self.values):
            i = self.i
            self.i += 1
            return self.values[i]
        else:
            val = self.generator.next()
            self.values.append(val)
            self.i += 1
            return val
