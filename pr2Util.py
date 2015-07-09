import math
import hu
import copy
from colors import RGBToPyColor, HSVtoRGB
from dist import chiSqFromP
import numpy as np
import dist
from planGlobals import debug, debugMsg
from planUtil import PoseD, Violations

class Hashable:
    def __init__(self):
        self.hashValue = None
        self.descValue = None
    def __eq__(self, other):
        return hasattr(other, 'desc') and self.desc() == other.desc()
    def __neq__(self, other):
        return not self == other
    def __hash__(self):
        if self.hashValue == None:
            self.hashValue = hash(self.desc())
        return self.hashValue
    def __str__(self):
        return self.__class__.__name__+str(self.desc())
    __repr__ = __str__

# MOVED TO planUtil.pyx

# # Ultimately, we will also need a discrete value to indicate which
# # stable face the object is resting on.
# class PoseD(Hashable):
#     def __init__(self, mu, var):
#         if isinstance(mu, tuple):
#             mu = hu.Pose(*mu)
#         # assert isinstance(mu, hu.Pose)
#         self.mu = mu
#         self.muTuple = self.mu.xyztTuple() if mu else None
#         assert isinstance(var, tuple)
#         self.var = var
#         Hashable.__init__(self)
        
#     def mean(self):
#         return self.mu
#     mode = mean

#     def meanTuple(self):
#         return self.mu.xyztTuple()
#     modeTuple = meanTuple

#     def variance(self):
#         return self.var
#     varTuple = variance

#     def desc(self):
#         return (self.muTuple, self.var)

# # This represent poses for the finger tip in the contact face.
# class ObjGraspB(Hashable):
#     def __init__(self, obj, graspDesc, grasp, poseD, var = None, delta = None):
#         self.obj = obj
#         self.graspDesc = graspDesc
#         # grasp is an index into the graspDesc
#         if isinstance(grasp, dist.DDist):
#             self.grasp = grasp                # this is a DDist for index
#         elif grasp == None:
#             self.grasp = dist.UniformDist(range(len(self.graspDesc)))
#         else:
#             self.grasp = dist.DeltaDist(grasp)
#         # This is PoseD for the mode of the grasp
#         if isinstance(poseD, (hu.Pose, hu.Transform)):
#             self.poseD = PoseD(poseD.pose(), var or 4*(0.0,))
#         elif poseD == None or hasattr(poseD, 'muTuple'):
#             self.poseD = poseD
#         else:
#             print 'Unknown poseD type'
#             assert None
#         self.delta = delta or 4*(0.0,)
#         Hashable.__init__(self)
#     def modifyPoseD(self, mu=None, var=None):
#         gB = copy.copy(self)
#         gB.poseD = PoseD(mu or self.poseD.mu,
#                          var or self.poseD.var)
#         return gB
#     def desc(self):
#         if not self.descValue:
#             self.descValue = (self.obj, self.grasp, self.poseD, self.delta) # .mode() for grasp?
#         return self.descValue
#     def __eq__(self, other):
#         if other is None and self.obj == 'none': return True
#         return hasattr(other, 'desc') and self.desc() == other.desc()
#     def __neq__(self, other):
#         return not self == other
    
# # This represents poses for the face frame of the support face
# class ObjPlaceB(Hashable):
#     def __init__(self, obj, faceFrames, support, poseD, var = None, delta = None):
#         self.obj = obj
#         self.faceFrames = faceFrames
#         # support is an index into the faceFrames
#         if isinstance(support, dist.DDist):
#             self.support = support            # this is a DDist for index
#         else:
#             self.support = dist.DeltaDist(support)
#         # This is PoseD for the mode of the support
#         if isinstance(poseD, (hu.Pose, hu.Transform)):
#             self.poseD = PoseD(poseD.pose(), var or 4*(0.0,))
#         elif poseD == None or hasattr(poseD, 'muTuple'):
#             self.poseD = poseD
#         else:
#             print 'Unknown poseD type'
#             assert None
#         self.delta = delta or 4*(0.0,)
#         Hashable.__init__(self)
#     # get the origin pose for the object corresponding to mode of poseD.
#     def objFrame(self):
#         faceFrame = self.faceFrames[self.support.mode()]
#         return self.poseD.mode().compose(faceFrame.inverse())
#     def modifyPoseD(self, mu=None, var=None):
#         pB = copy.copy(self)
#         pB.poseD = PoseD(mu or self.poseD.mu,
#                          var or self.poseD.var)
#         return pB
#     def desc(self):
#         if not self.descValue:
#             self.descValue = (self.obj, self.support, self.poseD, self.delta) # .mode() for support
#         return self.descValue
#     def shape(self, ws):                # in WorldState, e.g. shadow world
#         return ws.world.getObjectShapeAtOrigin(self.obj).applyLoc(self.objFrame())
#     def shadow(self, ws):
#         if shadowName(self.obj) in ws.objectShapes:
#             return ws.objectShapes[shadowName(self.obj)].applyLoc(self.objFrame())

# # represent the "removable" collisions wth obstacles and shadows.
# # This has to be hashable so use tuples and frozensets
# class Violations(Hashable):
#     def __init__(self, obstacles=[], shadows=[],
#                  heldObstacles=None, heldShadows=None):
#         obst = obstacles[:]
#         sh = shadows[:]
#         if not heldObstacles: heldObstacles = ([], [])
#         if not heldShadows: heldShadows = ([], [])
#         self.obstacles = frozenset(obst)
#         # Collisions with only shadows, remove collisions with objects as well
#         self.shadows = frozenset([o for o in sh if not o in self.obstacles])
#         ao = self.obstacles.union(self.shadows)
#         ho = ([],[])
#         if heldObstacles:
#             for h in (0,1):
#                 for o in heldObstacles[h]:
#                     if o not in ao: ho[h].append(o)
#         hs = ([],[])
#         if heldShadows:
#             for h in (0,1):
#                 for o in heldShadows[h]:
#                     if o not in ao: hs[h].append(o)
#         self.heldObstacles = tuple([frozenset(ho[h]) for h in (0,1)])
#         # Collisions only with heldShadow, remove collisions with heldObject as well
#         self.heldShadows = tuple([frozenset([o for o in hs[h] \
#                                              if not o in self.heldObstacles[h]]) \
#                                   for h in (0,1)])
#         Hashable.__init__(self)
#     def allObstacles(self):
#         obst = list(self.obstacles)
#         for h in (0,1):
#             for o in self.heldObstacles[h].union(self.heldShadows[h]):
#                 if not shadowp(o): obst.append(o)
#         return obst
#     def allShadows(self):
#         shad = list(self.shadows)
#         for h in (0,1):
#             for o in self.heldObstacles[h].union(self.heldShadows[h]):
#                 if shadowp(o): shad.append(o)
#         return shad
#     def empty(self):
#         return (not self.obstacles) and (not self.shadows) \
#                and (not any(x for x in self.heldObstacles)) \
#                and (not any(x for x in self.heldShadows))
#     def combine(self, obstacles, shadows, heldObstacles=None, heldShadows=None):
#         return self.update(Violations(obstacles, shadows, heldObstacles, heldShadows))
#     def update(self, viol):
#             return Violations(upd(self.obstacles, viol.obstacles),
#                               upd(self.shadows, viol.shadows),
#                               (upd(self.heldObstacles[0], viol.heldObstacles[0]),
#                                upd(self.heldObstacles[1], viol.heldObstacles[1])),
#                               (upd(self.heldShadows[0], viol.heldShadows[0]),
#                                upd(self.heldShadows[1], viol.heldShadows[1])))
#     def weight(self, weights=(1.0, 0.5, 1.0, 0.5)):
#         return weights[0]*len(self.obstacles) + \
#                weights[1]*len(self.shadows) + \
#                weights[2]*sum([1 if ho else 0 for ho in self.heldObstacles]) +\
#                weights[3]*sum([1 if hs else 0 for hs in self.heldShadows])
#     def LEQ(self, other):
#         return self.weight() <= other.weight()
#     def desc(self):
#         return (self.obstacles, self.shadows, self.heldObstacles, self.heldShadows)
#     def names(self):
#         return (frozenset([x.name() for x in self.obstacles]),
#                 frozenset([x.name() for x in self.shadows]),
#                 tuple([frozenset([x.name() for x in ho]) for ho in self.heldObstacles]),
#                 tuple([frozenset([x.name() for x in hs]) for hs in self.heldShadows]))
#     def __repr__(self):
#         return 'Violations%s'%str(([x.name() for x in self.obstacles],
#                                    [x.name() for x in self.shadows],
#                                    [[x.name() for x in ho] for ho in self.heldObstacles],
#                                    [[x.name() for x in hs] for hs in self.heldShadows]))
#     __str__ = __repr__

# def upd(curShapes, newShapes):
#     curDict = dict([(o.name(), o) for o in curShapes])
#     newDict = dict([(o.name(), o) for o in newShapes])
#     curDict.update(newDict)
#     return curDict.values()

# ========================

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

######################################################################
# Store probabilities to describe the domain
######################################################################

class DomainProbs:
    # obsVar, pickVar, and placeVar are diagonal cov 4-tuples (fix this!)
    # pickTolerance is a distance error in x, y, z, theta
    def __init__(self, odoError, obsVar, obsTypeErrProb,
                 pickFailProb, placeFailProb,
                 pickVar, placeVar, pickTolerance,
                 maxGraspVar = (0.015**2, .015**2, .015**2, .03**2),
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
        # size of basin of attraction of pick operator
        self.pickTolerance = pickTolerance
        # Don't allow a bigger grasp variance than this
        self.maxGraspVar = maxGraspVar
        # Bad failures, like dropping
        self.pickFailProb = pickFailProb
        self.placeFailProb = placeFailProb
        self.placeDelta = placeDelta
        self.graspDelta = graspDelta
        self.shadowDelta = shadowDelta
        self.moveConfDelta = moveConfDelta
        minDelta = [2*x for x in self.placeStdev]
        minDelta[2] = 1e-3
        self.minDelta = tuple(minDelta)

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

