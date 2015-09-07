import pdb
import copy
import traceFile
import fbch
reload(fbch)
from fbch import Operator, Fluent, Function, State, \
     planBackward, getMatchingFluents, makePlanObj
from miscUtil import within, isGround, isVar
import planGlobals as glob
reload(glob)

class W(Fluent):
    predicate = 'W'
    def test(self, details):
        (fluent, value, delta) = self.args
        return within(fluent.test(details), value, delta)

    def getGrounding(self, bstate):
        (fluent, value, delta) = self.args
        return {value : fluent.test(bstate.details)}

    def update(self):
        self.args[0].value = self.args[1]
        self.args[0].update()
        super(W, self).update()

    def glb(self, other, details):
        if other.predicate != 'W':
            return {self, other}, {}
     
        (sf, sv, sd) = self.args
        (of, ov, od) = other.args
        sfn, ofn = sf.copy(), of.copy()
        sfn.value = sv; ofn.value = ov
        fglb, b = sfn.glb(ofn, details)
        if fglb == False:
            # contradiction
            return False, {}
        if isinstance(fglb, set):
            # no glb at rfluent level
            return {self, other}, {}

        if self.isGround() and other.isGround():
            # Find interval intersection
            smin, smax = sv - sd, sv + sd
            omin, omax = ov - od, ov + od
            if smin > omax or omin > smax:
                # No interesection, contradiction
                return False, {}
            if smin == omin and smax == omax:
                return self, b
            nmin, nmax = max(smin, omin), min(smax, omax)
            nv, nd = (nmax + nmin) / 2.0, (nmax - nmin) / 2.0
            return W([fglb, nv, nd], True), b

        # Deal with variables
        if isVar(sv): b[sv] = ov
        elif isVar(ov): b[ov] = sv
        if isVar(sd): b[sd] = od
        elif isVar(od): b[od] = sd

        if isGround(sd) and isGround(od):
            # value is not bound but both deltas are;  take smaller
            nd = min(sd, od)
            nf = W([fglb, self.args[1], nd], True)
            return nf.applyBindings(b), b
            
        return self.applyBindings(b), b
    
class Pose(Fluent):
    predicate = 'Pose'
    def test(self, details):
        return details.pose

class Conf(Fluent):
    predicate = 'Conf'
    def test(self, details):
        return details.conf

class Grasp(Fluent):
    predicate = 'Grasp'
    def test(self, details):
        return details.pose - details.conf

class Painted(Fluent):
    predicate = 'Painted'
    def test(self, details):
        return details.painted

class Holding(Fluent):
    predicate = 'Holding'
    def test(self, details):
        return details.holding
            
class SimpleBel:
    def __init__(self):
        self.pose = 2
        self.conf = 1
        self.holding = False
        self.painted = False

eps = 1e-2
minConfDelta = 0.1
paintLoc = 3
paintDelta = 0.5
maxGraspDelta = 0.5
movePrecision = 0.05    

class MovePrevDelta(Function):
    @staticmethod    
    def fun(args, goal, start):
        (ed,) = args
        result = ed - movePrecision
        return [[result]] if result > 0 else []

class PickGen(Function):
    @staticmethod    
    def fun(args, goal, start):
        (grasp, graspDelta) = args
        # constraints:
        #  - poseDelta + confDelta < graspDelta
        #  - conf + grasp = pose

        # pose in goal, if there is one
        fbp = getMatchingFluents(goal, W([Pose([]), 'P', 'PD'], True))
        if fbp:
            # added constraint: poseDelta < maxPoseDelta
            pose, poseDelta = fbp[0][1]['P'], fbp[0][1]['PD']
            conf = pose - grasp
            # Strategy below is legal, but might make confDelta too small for
            # comfort.
            confDelta = max(graspDelta - poseDelta, minConfDelta)
            poseDelta = graspDelta - confDelta
            return [(conf, confDelta, pose, poseDelta)]

        results = []
        # initial pose, pose for painting
        # cool to make a generator for random poses after these
        poses = [start.pose, paintLoc]
        poseDelta = (graspDelta / 2.0)
        confDelta = poseDelta
        for pose in poses:
            conf = pose - grasp
            results.append((conf, confDelta, pose, poseDelta))
        return results

class PlaceGen(Function):
    @staticmethod    
    def fun(args, goal, start):
        (pose, poseDelta) = args
        # constraints:
        #  - graspDelta + confDelta < poseDelta
        #  - conf + grasp = pose
        #  - graspDelta < maxGraspDelta

        # grasp in goal, if there is one
        fbp = getMatchingFluents(goal, W([Grasp([]), 'G', 'GD'], True))
        if fbp:
            grasp, graspDelta = fbp[0][1]['G'], fbp[0][1]['GD']
            conf = pose - grasp
            # Strategy below is legal, but might make confDelta too small for
            # comfort.
            confDelta = max(poseDelta - graspDelta, minConfDelta)
            graspDelta = poseDelta - confDelta
            return [(conf, confDelta, pose, poseDelta)]

        results = []
        # In this domain, grasp 0 is the only one that makes sense
        grasps = [0.0]
        # Divide by 2 is easier, but try divide by 4 here (to give
        # more slack to the conf)
        graspDelta = min((poseDelta / 2.0), maxGraspDelta)
        confDelta = poseDelta - graspDelta
        for grasp in grasps:
            conf = pose - grasp
            results.append((conf, confDelta, grasp, graspDelta))
        return results

# It would be ugly if we had to generate starting confs SC.  This
# leaves that variable unbound and uses matching to handle it.
move = Operator('Move', ['SC', 'SD', 'EC', 'ED'],
                {0 : {W([Conf([]), 'SC', 'SD'], True)}},
                [({W([Conf([]), 'EC', 'ED'], True)},{})],
                [MovePrevDelta(['SD'], ['ED'])])

pick = Operator('Pick', ['C', 'CD', 'P', 'PD', 'G', 'GD'],
                {0 : {W([Pose([]), 'P', 'PD'], True),
                      W([Conf([]), 'C', 'CD'], True),
                      Holding([], False)}},
                [({Holding([], True),
                   W([Grasp([]), 'G', 'GD'], True)},{})],
                [PickGen(['C', 'CD', 'P', 'PD'], ['G', 'GD'])])


place = Operator('Place', ['C', 'CD', 'P', 'PD', 'G', 'GD'],
                {0 : {W([Grasp([]), 'G', 'GD'], True),
                      W([Conf([]), 'C', 'CD'], True),
                      Holding([], True)}},
                [({Holding([], False)}, {}),
                 ({W([Pose([]), 'P', 'PD'], True)},{})],
                [PlaceGen(['C', 'CD', 'G', 'GD'], ['P', 'PD'])])

paint = Operator('Paint', [],
                 {0 : {W([Pose([]), paintLoc, paintDelta], True),
                       Holding([], False)}},
                 [({Painted([], True)}, {})])


######################################################################
#                         Planning
######################################################################                    
ops = [move, pick, place, paint]

def runTest(init, goal, skeleton = None):
    h = lambda s, g, o, a: g.easyH(s)
    p = planBackward(init, goal, ops, h = h, 
                     fileTag = 'interval', maxNodes = 100,
                     skeleton = skeleton)
    if p:
        planObj = makePlanObj(p, init)
        planObj.printIt(verbose = False)

# Just move        
def it1():
    init = State([], SimpleBel())
    goal = State([W([Conf([]), 5,  0.2], True)])
    runTest(init, goal)

# Move the object
def it2():
    init = State([], SimpleBel())
    goal = State([W([Pose([]), 3,  0.5], True)])
    runTest(init, goal)

# Paint the object
def it3():
    init = State([], SimpleBel())
    goal = State([Painted([], True)])
    runTest(init, goal)

# Painted and at another loc
def it4():
    skeleton = [place, move, pick, paint, place, move, pick, move]
    init = State([], SimpleBel())
    goal = State([Painted([], True),
                  W([Pose([]), 5, 1.0], True)])
    runTest(init, goal)
            

print 'interval loaded'
