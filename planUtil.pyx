from cpython cimport bool
import hu
import dist

# Copied from pr2Util.py
def shadowp(obj):
    if isinstance(obj, str):
        return obj[-7:] == '_shadow'
    else:
        return obj.name()[-7:] == '_shadow'

# Copied from pr2Util.py
def shadowName(obj):
    name = obj if isinstance(obj, str) else obj.name()
    return name+'_shadow'

cdef class Hash(object):
    def __init__(self):
        self.hashValue = None
        self.descValue = None
    def __str__(self):
        return self.__class__.__name__+str(self.desc())
    def __repr__(self):
        return self.__class__.__name__+str(self.desc())
    def __hash__(self):
        if self.hashValue == None:
            self.hashValue = self.desc().__hash__()
        return self.hashValue
    def __richcmp__(self, other, int op):
        if op == 2:
            return isinstance(other, Hash) and self.desc() == other.desc()
        elif op == 3:
            return not (isinstance(other, Hash) and self.desc() == other.desc())
        else:
            return False

# Ultimately, we will also need a discrete value to indicate which
# stable face the object is resting on.
cdef class PoseD(Hash):
    def __init__(self, mu, var):
        if isinstance(mu, tuple):
            mu = hu.Pose(*mu)
        # assert isinstance(mu, hu.Pose)
        self.mu = mu
        self.muTuple = self.mu.xyztTuple() if mu else None
        assert isinstance(var, tuple)
        self.var = var
        Hash.__init__(self)
    cpdef mean(self):
        return self.mu
    cpdef mode(self):
        return self.mu
    cpdef tuple meanTuple(self):
        return self.mu.pose().xyztTuple()
    cpdef tuple modeTuple(self):
        return self.mu.pose().xyztTuple()
    cpdef tuple variance(self):
        return self.var
    cpdef tuple varTuple(self):
        return self.var
    cpdef tuple desc(self):
        return (self.muTuple, self.var)

# This represent poses for the finger tip in the contact face.
cdef class ObjGraspB(Hash):
    def __init__(self, obj, graspDesc, grasp, support, poseD, var = None, delta = None):
        self.obj = obj
        self.graspDesc = graspDesc
        # grasp is an index into the graspDesc
        if hasattr(grasp, 'maxProbElt'):
            self.grasp = grasp                # this is a DDist for index
        elif grasp == None:
            self.grasp = dist.UniformDist(range(len(self.graspDesc)))
        else:
            self.grasp = dist.DeltaDist(grasp)
        # This is PoseD for the mode of the grasp
        if hasattr(poseD, 'matrix'):
            self.poseD = PoseD(poseD.pose(), var or 4*(0.0,))
        elif poseD == None or hasattr(poseD, 'muTuple'):
            self.poseD = poseD
        else:
            print 'Unknown poseD type', poseD
            assert None
        self.delta = delta or 4*(0.0,)
        self.support = support
        Hash.__init__(self)
    cpdef ObjGraspB copy(self):
        return ObjGraspB(self.obj, self.graspDesc, self.grasp, self.support,self.poseD, delta=self.delta)
    cpdef ObjGraspB modifyPoseD(self, mu=None, var=None):
        gB = self.copy()
        gB.poseD = PoseD(mu or self.poseD.mu,
                         var or self.poseD.var)
        return gB
    cpdef tuple desc(self):
        if not self.descValue:
            self.descValue = (self.obj, self.grasp, tuple(self.graspDesc),
                              self.poseD, self.delta) # .mode() for grasp?
        return self.descValue
    def __richcmp__(self, other, int op):
        if op == 2:
            if other is None and self.obj == 'none': return True
            return isinstance(other, ObjGraspB) and self.desc() == other.desc()
        elif op == 3:
            return not (isinstance(other, ObjGraspB) and self.desc() == other.desc())
        else:
            return False
    def __hash__(self):
        if self.hashValue == None:
            self.hashValue = self.desc().__hash__()
        return self.hashValue

# This represents poses for the face frame of the support face
cdef class ObjPlaceB(Hash):
    def __init__(self, obj, faceFrames, support, poseD, var = None, delta = None):
        self.obj = obj
        self.faceFrames = faceFrames
        # support is an index into the faceFrames
        if hasattr(support, 'maxProbElt'):
            self.support = support            # this is a DDist for index
        else:
            self.support = dist.DeltaDist(support)
        # This is PoseD for the mode of the support
        if hasattr(poseD, 'matrix'):
            self.poseD = PoseD(poseD.pose(), var or 4*(0.0,))
        elif poseD == None or hasattr(poseD, 'muTuple'):
            self.poseD = poseD
        else:
            print 'Unknown poseD type'
            assert None
        self.delta = delta or 4*(0.0,)
        Hash.__init__(self)
    cpdef ObjPlaceB copy(self):
        return ObjPlaceB(self.obj, self.faceFrames, self.support, self.poseD, delta=self.delta)
    # get the origin pose for the object corresponding to mode of poseD.
    cpdef objFrame(self):
        faceFrame = self.faceFrames[self.support.mode()]
        return self.poseD.mode().compose(faceFrame.inverse())
    cpdef ObjPlaceB modifyPoseD(self, mu=None, var=None):
        pB = self.copy()
        pB.poseD = PoseD(mu or self.poseD.mu,
                         var or self.poseD.var)
        return pB
    cpdef tuple desc(self):
        if not self.descValue:
            self.descValue = (self.obj, self.support, self.poseD, self.delta) # .mode() for support
        return self.descValue
    cpdef shape(self, ws):                # in WorldState, e.g. shadow world, or world
        return ws.getObjectShapeAtOrigin(self.obj).applyLoc(self.objFrame())
    cpdef shadow(self, ws):
        if shadowName(self.obj) in ws.objectShapes:
            return ws.objectShapes[shadowName(self.obj)].applyLoc(self.objFrame())
    cpdef makeShadow(self, pbs, prob):
        faceFrame = self.faceFrames[self.support.mode()]
        sh = pbs.objShadow(self.obj, shadowName(self.obj), prob, self, faceFrame)
        return sh.applyLoc(self.objFrame())
    def __richcmp__(self, other, int op):
        if op == 2:
            return isinstance(other, ObjPlaceB) and self.desc() == other.desc()
        elif op == 3:
            return not (isinstance(other, ObjPlaceB) and self.desc() == other.desc())
        else:
            return False
    def __hash__(self):
        if self.hashValue == None:
            self.hashValue = self.desc().__hash__()
        return self.hashValue

# represent the "removable" collisions wth obstacles and shadows.
# This has to be hashable so use tuples and frozensets
cdef class Violations(Hash):
    def __init__(self, obstacles=[], shadows=[],
                 heldObstacles=None, heldShadows=None):
        obst = obstacles[:]
        sh = shadows[:]
        if not heldObstacles: heldObstacles = ([], [])
        if not heldShadows: heldShadows = ([], [])
        self.obstacles = frozenset(obst)
        # Collisions with only shadows, remove collisions with objects as well
        self.shadows = frozenset([o for o in sh if not o in self.obstacles])
        ao = self.obstacles.union(self.shadows)
        ho = ([],[])
        if heldObstacles:
            for h in (0,1):
                for o in heldObstacles[h]:
                    if o not in ao: ho[h].append(o)
        hs = ([],[])
        if heldShadows:
            for h in (0,1):
                for o in heldShadows[h]:
                    if o not in ao: hs[h].append(o)
        self.heldObstacles = tuple([frozenset(ho[h]) for h in (0,1)])
        # Collisions only with heldShadow, remove collisions with heldObject as well
        self.heldShadows = tuple([frozenset([o for o in hs[h] \
                                             if not o in self.heldObstacles[h]]) \
                                  for h in (0,1)])
        Hash.__init__(self)
    cpdef list allObstacles(self):
        obst = list(self.obstacles)
        for h in (0,1):
            for o in self.heldObstacles[h].union(self.heldShadows[h]):
                if not shadowp(o): obst.append(o)
        return obst
    cpdef list allShadows(self):
        shad = list(self.shadows)
        for h in (0,1):
            for o in self.heldObstacles[h].union(self.heldShadows[h]):
                if shadowp(o): shad.append(o)
        return shad
    def empty(self):
        return (not self.obstacles) and (not self.shadows) \
               and (not any(x for x in self.heldObstacles)) \
               and (not any(x for x in self.heldShadows))
    cpdef Violations combine(self, obstacles, shadows, heldObstacles=None, heldShadows=None):
        return self.update(Violations(obstacles, shadows, heldObstacles, heldShadows))
    cpdef Violations update(self, viol):
            return Violations(upd(self.obstacles, viol.obstacles),
                              upd(self.shadows, viol.shadows),
                              (upd(self.heldObstacles[0], viol.heldObstacles[0]),
                               upd(self.heldObstacles[1], viol.heldObstacles[1])),
                              (upd(self.heldShadows[0], viol.heldShadows[0]),
                               upd(self.heldShadows[1], viol.heldShadows[1])))
    cpdef double weight(self, weights=(1.0, 0.5, 1.0, 0.5)):
        return weights[0]*len(self.obstacles) + \
               weights[1]*len(self.shadows) + \
               weights[2]*sum([len(ho) for ho in self.heldObstacles]) +\
               weights[3]*sum([len(hs) for hs in self.heldShadows])
    cpdef bool LEQ(self, other):
        return self.weight() <= other.weight()
    cpdef tuple desc(self):
        return (self.obstacles, self.shadows, self.heldObstacles, self.heldShadows)
    cpdef tuple names(self):
        return (frozenset([x.name() for x in self.obstacles]),
                frozenset([x.name() for x in self.shadows]),
                tuple([frozenset([x.name() for x in ho]) for ho in self.heldObstacles]),
                tuple([frozenset([x.name() for x in hs]) for hs in self.heldShadows]))
    def __repr__(self):
        return 'Violations%s'%str(([x.name() for x in self.obstacles],
                                   [x.name() for x in self.shadows],
                                   [[x.name() for x in ho] for ho in self.heldObstacles],
                                   [[x.name() for x in hs] for hs in self.heldShadows]))
    def __str__(self):
        return self.__repr__()

cpdef list upd(curShapes, newShapes):
    curDict = dict([(o.name(), o) for o in curShapes])
    newDict = dict([(o.name(), o) for o in newShapes])
    curDict.update(newDict)
    return curDict.values()

class PPResponse:
    def __init__(self, pB, gB, c, ca, viol, hand, var=None, delta=None):
        self.pB = pB
        self.gB = gB
        self.c = c
        self.ca = ca
        self.viol = viol
        self.hand = hand
        self.var = var
        self.delta = delta
    def easyGraspTuple(self):
        return (self.gB.grasp.mode(), self.gB.poseD.mode().xyztTuple(),
                self.gB.poseD.var, self.gB.delta)
    def pickTuple(self):
        return (self.pB.poseD.mode().xyztTuple(), self.pB.support.mode(), self.c, self.ca)
    def placeTuple(self):
        return (self.hand, self.gB.poseD.mode().xyztTuple(), self.gB.grasp.mode(),
                self.c, self.ca,
                self.pB.poseD.mode().xyztTuple(), self.pB.support.mode())
    def placeInTuple(self):
        return (self.pB.poseD.mode().xyztTuple(), self.pB.support.mode())
    def canXGenTuple(self):
        return (self.pB.obj, self.pB.poseD.mode().xyztTuple(), self.pB.support.mode(),
                self.var, self.delta)
    def copy(self):
        return PPResponse(self.pB, self.gB, self.c, self.ca, self.viol, self.hand)
    def __str__(self):
        obj = (self.pB or self.gB).obj
        pose = self.pB.poseD.mode().xyztTuple() if self.pB else None
        grasp = self.gB.grasp.mode() if self.gB else None
        pg = (self.pB.support.mode(), grasp)
        w = self.viol.weight() if self.viol else None
        return '%s %s v=%s (p,g)=%s, pose=%s'%(obj, self.hand, w, pg, pose)
    def __repr__(self):
        return str(self)
    
class PushResponse:
    def __init__(self, prePB, postPB, preConf, pushConf, postConf, viol, hand,
                 var=None, delta=None):
        self.prePB = prePB
        self.postPB = postPB
        self.preConf = preConf
        self.pushConf = pushConf
        self.postConf = postConf
        self.viol = viol
        self.hand = hand
        self.var = var
        self.delta = delta
    def pushTuple(self):
        return (self.hand, self.prePB.poseD.mode().xyztTuple(),
                self.preConf, self.pushConf, self.postConf)
    def pushInTuple(self):
        return (self.postPB.poseD.mode().xyztTuple(), self.postPB.support.mode())
    def canXGenTuple(self):
        return tuple()
    def copy(self):
        return PushResponse(self.prePB, self.postPB, self.preConf, self.pushConf, self.postConf,
                            self.viol, self.hand)
    def __str__(self):
        obj = (self.prePB or self.postPB).obj
        pose = self.prePB.poseD.mode().xyztTuple() if self.prePB else None
        w = self.viol.weight() if self.viol else None
        return '%s %s v=%s pose=%s'%(obj, self.hand, w,  pose)
    def __repr__(self):
        return str(self)
    
