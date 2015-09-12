import pdb
import copy
import traceFile
import fbch
reload(fbch)
from fbch import Operator, Fluent, Function, State, \
     planBackward, getMatchingFluents, makePlanObj, simplifyCond
from miscUtil import within, isGround, isVar, customCopy, prettyString, \
     lookup
import planGlobals as glob
reload(glob)

######################################################################
#
# Infrastructure
#
######################################################################

class W(Fluent):
    predicate = 'W'
    def test(self, details):
        (fluent, value, delta) = self.args
        fint = fluent.interval(details)
        return withinInt(fluent.interval(details), (value, delta))

    def getGrounding(self, bstate):
        (fluent, value, delta) = self.args
        return {value : fluent.interval(bstate.details)[0]}

    def isImplicit(self):
        return self.args[0].isImplicit()

    def isConditional(self):
        return self.args[0].isConditional()

    def sameFluent(self, other):
        return self.isGround() and  other.isGround() and \
             other.predicate == 'W' and \
             self.args[0].matchArgs(other.args[0]) != None

    def getIsGround(self):
        return self.args[0].isGround() and \
            all([not isVar(a) for a in self.args[1:]]) \
            and not isVar(self.value)

    def getIsPartiallyBound(self):
        b0 = self.args[0].isPartiallyBound()
        g0 = self.args[0].isGround()
        v0 = not b0 and not g0 # rf has no bindings
        av = [v0] + [isVar(a) for a in self.args[1:]]
        return (True in av) and (False in av)

    def argsGround(self):
        return self.args[0].isGround()

    def getVars(self):
        return set([a for a in self.args[1:] if isVar(a)]).union(\
                                                self.args[0].getVars())

    def argString(self, eq):
        return '['+ self.args[0].prettyString(eq, includeValue = False) +', ' +\
                  ', '.join([prettyString(a, eq) for a in self.args[1:]]) + ']'

    def applyBindings(self, bindings):
        if self.isGround():
            return self.copy()
        return W([self.args[0].applyBindings(bindings)] + \
                  [lookup(a, bindings) for a in self.args[1:]],
                  lookup(self.value, bindings))

    def copy(self):
        return W(customCopy(self.args), customCopy(self.value))

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
            if disjointInt((sv, sd), (ov, od)):
                return False, {}
            smin, smax = sv - sd, sv + sd
            omin, omax = ov - od, ov + od
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

def withinInt((m1, d1), (m2, d2)):
    l1, u1 = m1-d1, m1+d1
    l2, u2 = m2-d2, m2+d2
    return l1 >= l2 and u1 <= u2

def overlapsInt(i1, i2):
    return not disjointInt(i1, i2)
def disjointInt((m1, d1), (m2, d2)):
    l1, u1 = m1-d1, m1+d1
    l2, u2 = m2-d2, m2+d2
    return l1 > u2 or l2 > u1

######################################################################
#
# Fluents
#
######################################################################

class Pose(Fluent):
    predicate = 'Pose'
    def interval(self, details):
        (obj,) = self.args
        return details.objects[obj]   # (center, width) of interval

class Conf(Fluent):
    predicate = 'Conf'
    def interval(self, details):
        return details.conf   # interval

class Grasp(Fluent):
    predicate = 'Grasp'
    def interval(self, details):
        return details.grasp   # interval

class Painted(Fluent):
    predicate = 'Painted'
    def test(self, details):
        (obj,) = self.args
        return details.painted[obj]  # boolean

class Holding(Fluent):
    predicate = 'Holding'
    def test(self, details):
        return details.holding   # object name

class CanPaint(Fluent):
    implicit = True
    conditional = True
    predicate = 'CanPaint'

    def conditionOn(self, f):
        return f.predicate == 'W' and f.args[0].predicate in ('PaintLevel',)

    def test(self, details):
        (obj, cond) = self.args
        pbs = details.updateFromCond(cond)
        return True # Boolean

class CanPlace(Fluent):
    implicit = True
    conditional = True
    predicate = 'CanPlace'

    def conditionOn(self, f):
        return f.predicate == 'W' and f.args[0].predicate in ('Pose',)
    
    def test(self, details):
        (obj, poseMid, poseDelta, cond) = self.args
        pbs = details.updateFromCond(cond)
        return pbs.intervalFree((poseMid, poseDelta + (objSize / 2.0)))
            
class SimpleBel:
    def __init__(self, objects):
        self.objects = objects # dictionary maping names to (center, width)
        self.conf = (1, 0)
        self.grasp = (0, 0)
        self.holding = 'none'
        self.painted = dict([(o, False) for o in objects.keys()])
        self.paintSupply = (0, 0)
        self.propellantSupply = (0, 0)

    def copy(self):
        new = copy.copy(self)
        new.objects = copy.copy(self.objects)
        new.painted = copy.copy(self.painted)
        return new

    def updateFromCond(self, conditions):
        newBS = self.copy()
        for c in conditions:
            if c.predicate == 'W' and c.args[0].predicate == 'Pose':
                # Override object pose
                newBS.objects[c.args[0].args[0]] = c.args[1:]
            elif c.predicate == 'PaintSupply':
                newBS.paintSupply = c.args[0]
            else:
                print "Don't know how to handle cond", c
        return newBS

    def intervalFree(self, interval):
        return all([not overlapsInt(oInterval, interval) \
                    for oInterval in self.objects.values()])

######################################################################
#
#  Generators
#
######################################################################

eps = 1e-2
minConfDelta = 0.1
paintLoc = 5
paintDelta = 0.5
maxGraspDelta = 0.5
movePrecision = 0.05
outDelta = 0.5
objSize = 1 # constant for now

class MovePrevDelta(Function):
    @staticmethod    
    def fun(args, goal, start):
        (ed,) = args
        result = ed - movePrecision
        return [[result]] if result > 0 else []

class PickGen(Function):
    @staticmethod    
    def fun(args, goal, start):
        (obj, grasp, graspDelta) = args
        # constraints:
        #  - poseDelta + confDelta < graspDelta
        #  - conf + grasp = pose

        # pose in goal, if there is one
        fbp = getMatchingFluents(goal, W([Pose([obj]), 'P', 'PD'], True))
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
        poses = [start.objects[obj][0], paintLoc]
        poseDelta = (graspDelta / 2.0)
        confDelta = poseDelta
        for pose in poses:
            conf = pose - grasp
            results.append((conf, confDelta, pose, poseDelta))
        return results

class PlaceGen(Function):
    @staticmethod    
    def fun(args, goal, start):
        (obj, pose, poseDelta) = args
        # constraints:
        #  - graspDelta + confDelta < poseDelta
        #  - conf + grasp = pose
        #  - graspDelta < maxGraspDelta

        # grasp in goal, if there is one
        fbg = getMatchingFluents(goal, W([Grasp([]), 'G', 'GD'], True))
        fbh = getMatchingFluents(goal, Holding([], obj))
        if fbh:
            assert(fbg)
            grasp, graspDelta = fbg[0][1]['G'], fbp[0][1]['GD']
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

class AchCanPlaceGen(Function):
    @staticmethod    
    def fun(args, goal, start):
        (obj, pose, poseDelta, cond) = args
        pbs = start.updateFromCond(cond)
        intervalWidth = poseDelta + (objSize / 2.0)
        if pbs.intervalFree((pose, intervalWidth)):
            # No suggestions, it's already true
            return
        for (o, oInt) in pbs.objects.items():
            if overlapsInt(oInt, (pose, intervalWidth)):
                for loc in outLoc((pose, intervalWidth), pbs):
                    yield (o, loc)
        return

def outLoc(interval, pbs):
    newBS = pbs.copy()
    newBS.objects['fizz'] = interval
    maxOcc = max([p + d + (objSize / 2.0) for (p, d) in newBS.objects.values()])
    val = maxOcc + outDelta + (objSize / 2.0)
    for i in range(5):
       yield val
       val += (outDelta + objSize) * 2

class AddPoseCond(Function):
    @staticmethod    
    def fun(args, goal, start):
        (postCond, obj, pose, delta) = args
        cond = W([Pose([obj]), pose, delta], True)
        return [[simplifyCond(postCond, [cond])]]
           
    
######################################################################
#
#  Operators
#
######################################################################

# It would be ugly if we had to generate starting confs SC.  This
# leaves that variable unbound and uses matching to handle it.
move = Operator('Move', ['SC', 'SD', 'EC', 'ED'],
                {0 : {W([Conf([]), 'SC', 'SD'], True)}},
                [({W([Conf([]), 'EC', 'ED'], True)},{})],
                [MovePrevDelta(['SD'], ['ED'])])

pick = Operator('Pick', ['O', 'C', 'CD', 'P', 'PD', 'G', 'GD'],
                {0 : {W([Pose(['O']), 'P', 'PD'], True),
                      W([Conf([]), 'C', 'CD'], True),
                      Holding([], 'none')}},
                [({Holding([], 'O'),
                   W([Grasp([]), 'G', 'GD'], True)},{})],
                [PickGen(['C', 'CD', 'P', 'PD'], ['O', 'G', 'GD'])])


place = Operator('Place', ['O', 'C', 'CD', 'P', 'PD', 'G', 'GD'],
                {0 : {W([Grasp([]), 'G', 'GD'], True),
                      W([Conf([]), 'C', 'CD'], True),
                      CanPlace(['O','P', 'PD', []], True),
                      Holding([], 'O')}},
                [({Holding([], 'none')}, {}),
                 ({W([Pose(['O']), 'P', 'PD'], True)},{})],
                [PlaceGen(['C', 'CD', 'G', 'GD'], ['O', 'P', 'PD'])])

# Precond really should be that the robot is not holding O
paint = Operator('Paint', ['O'],
                 {0 : {W([Pose(['O']), paintLoc, paintDelta], True),
                       Holding([], 'none')}}, 
                 [({Painted(['O'], True)}, {})])

achCanPlace = Operator('AchCanPlace', ['O', 'P', 'PD',
                                       'Occ', 'OccP', 
                                       'PreCond', 'PostCond'],
                {0 : {CanPlace(['O','P', 'PD', 'PreCond'], True),
                      W([Pose(['Occ']), 'OccP', outDelta], True)}},
                [({CanPlace(['O','P', 'PD', 'PostCond'], True)},{})],
                [AchCanPlaceGen(['Occ', 'OccP'], ['O', 'P', 'PD', 'PostCond']),
                 AddPoseCond(['PreCond'],
                             ['PostCond', 'Occ', 'OccP', outDelta])])

######################################################################
#
#                         Planner call and tests
#
######################################################################

ops = [move, pick, place, paint, achCanPlace]

def runTest(goal, skeleton = None, objects = {'a' : (2, 0)}):
    h = lambda s, g, o, a: g.easyH(s)
    init = State([], SimpleBel(copy.copy(objects)))
    p = planBackward(init, goal, ops, h = h, 
                     fileTag = 'interval', maxNodes = 100,
                     skeleton = skeleton)
    if p:
        planObj = makePlanObj(p, init)
        planObj.printIt(verbose = False)

# Just move        
def it1():
    goal = State([W([Conf([]), 5,  0.2], True)])
    runTest(goal)

# Move the object
def it2():
    skeleton = [place, move, pick, move]
    goal = State([W([Pose(['a']), 5,  0.5], True)])
    runTest(goal, skeleton = skeleton)

# Paint the object
def it3():
    goal = State([Painted(['a'], True)])
    runTest(goal)

# Painted and at another loc
def it4():
    skeleton = [place, move, pick, paint, place, move, pick, move]
    goal = State([Painted(['a'], True),
                  W([Pose(['a']), 10, 1.0], True)])
    runTest(goal)

# Move something out of the way
def it5():
    skeleton = [place, move, pick, move, achCanPlace,
                place, move, pick, move]
    objects = {'a' : (2, 0), 'b' : (10, 0)}
    goal = State([W([Pose(['a']), 10, 1.0], True)])
    runTest(goal, objects = objects)
                

print 'interval loaded'
