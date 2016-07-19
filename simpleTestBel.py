import pdb
import math
import numpy as np
import copy
import fbch
reload(fbch)
from fbch import Fluent, planBackward, State, Operator, \
     makePlanObj, HPN, Function, getMatchingFluents, simplifyCond

import local
reload(local)

import miscUtil
reload(miscUtil)
from miscUtil import isVar, lookup, isGround

from operator import mul

import planGlobals as glob
reload(glob)

import traceFile
reload(traceFile)
from traceFile import debugMsg, debug, debugDraw

import belief
reload(belief)
from belief import Bd, BBhAddBackBSet

import dist
reload(dist)
from dist import DDist as DD
from dist import UniformDist

import cff
reload(cff)
from cff import CFF, CFFConstraint

maxRequirableCertainty = 0.99
minAllowableProb = .1

glob.rebindPenalty = 10

######################################################################
#
# Progression procedures update belief state
#
######################################################################

# Belief update rule

def stateTransModel(oldState, state, failProb, relevantProb):
    # probability of a state change is (1 - failProb) * relevantProb
    # probability of becoming dirty is relevantProb * failProb / 2
    
    if oldState == state:
        d = DD({state : 1.0 - 0.5*failProb*relevantProb})
        d.addProb('dirty', 0.5*failProb*relevantProb)
    elif oldState in precondStates[state]:
        d = DD({'dirty' : relevantProb * 0.5 * failProb})
        d.addProb(state, (1 - failProb)*relevantProb)
        d.addProb(oldState,
                  1 - relevantProb * 0.5 * failProb - \
                  (1 - failProb)*relevantProb)
    else:
        d = DD({oldState : 1.0 - 0.5*failProb*relevantProb})
        d.addProb('dirty', 0.5*failProb*relevantProb)
    return d

def dirtyTransModel(oldState, dirtyProb):
    if oldState == 'dirty':
        return DD({'dirty' : 1.0})       # no change
    else:
        d = DD({oldState: 1 - dirtyProb})
        d.addProb('dirty', dirtyProb)
        return d
    
def stateBProgress(state, loc, failProb):
    # Have to take into account the likelihood that the object is in the
    # location where this state transformation can happen and that the
    # precond is satisfied
    
    def bProgress(details, args, obs = None):
        # TLP - In the continuous case, this should affect the
        # position as well as state.  Should the probability
        # arguments enter into this??
        # P = P1*P2*(1-failProb)

        print 'Progressing belief state', state, loc
        
        for obj in details.objLoc.keys():
            relevantProb = details.objLocDist(obj).prob(loc)
            transModel = \
                lambda oldState: stateTransModel(oldState, state, failProb,
                                                 relevantProb)
            print 'before update', details.objState[obj]
            details.objState[obj].transitionUpdate(transModel)
            print 'after update', details.objState[obj]

    return bProgress

def moveTransModel(oldLoc, start, dest, failProb):
    if oldLoc == dest or oldLoc != start:
        return DD({oldLoc : 1.0})
    else:
        return DD({oldLoc: failProb,               # no change
                      dest : 1-failProb})

dirtyProb = 0.1

def moveBProgress(details, args, obs = None):
    [_, start, dest, _, _, _, _, _, _, _] = args
    moveProb = 1 - failProbs['Move']
    print 'moveBProgress', args, 'success prob', moveProb
    objClearProbs = dict([(o, 1 - d.prob(dest))\
                           for (o, d) in details.objLoc.items()])
    
    for obj in details.objLoc.keys():
        if obs not in ('bump', 'whiff'):
            clearProb = reduce(mul,
                               [objClearProbs[o] \
                                for o in details.objLoc.keys() if o != obj], 1)
            fp = 1 - (moveProb * clearProb)     # move wins, dest is clear
            transModel = \
              lambda oldLoc: moveTransModel(oldLoc, start, dest, fp)
            details.objLoc[obj].transitionUpdate(transModel)
            
        ## !! fix this
        dp = dirtyProb * details.objLocDist(obj).prob(start)
        dirtyModel = \
          lambda oldState: dirtyTransModel(oldState, dp)
        details.objState[obj].transitionUpdate(dirtyModel)
    # Update occs at start and dest
    # This would be nicer if we had a general DBN strategy for doing
    # transitions (so a var can have more than one parent)

    # Cheap way of making transition depend on whether the dest was
    # previously occupied.
    msp = moveProb * details.locOccDist(dest).prob(False)
    def occStartModel(oldOcc):
        if oldOcc:
            return DD({False: msp, True: 1-msp})
        else:
            return DD({False: 1.0})

    # Cheap way of making transition depend on whether the start was
    # previously occupied.
    mdp = moveProb * details.locOccDist(start).prob(True)
    def occDestModel(oldOcc):
        if oldOcc:
            return DD({True: 1.0})
        else:
            return DD({True: mdp, False: 1-mdp})

    def destObsModel(occ):
        fp = failProbs['Look']
        if occ:
            return DD({'bump' : 1 - fp, None : fp})
        else:
            return DD({None : 1 - fp, 'bump' : fp})

    def startObsModel(occ):
        fp = failProbs['Look']
        if occ:
            return DD({None : 1 - fp, 'whiff' : fp})
        else:
            return DD({'whiff' : 1 - fp, None : fp})
        
    # incorporate observations first, because they are about the
    # state before transition
    startObs = obs if obs != 'bump' else None
    details.occupancies[start].obsUpdate(startObsModel, startObs)
    destObs = obs if obs != 'whiff' else None
    details.occupancies[dest].obsUpdate(destObsModel, destObs)
    # transition update
    details.occupancies[start].transitionUpdate(occStartModel)
    details.occupancies[dest].transitionUpdate(occDestModel)

# this model assumes that we can aim the sensor at the location
# accurately.  
def lookObsModelTarget(trueLoc, lookLoc, failProb):
    # We're doing, Look(obj, lookLoc), this is update for objLoc[obj]
    if trueLoc == lookLoc:
        # nominal case 
        return DD({True: 1 - failProb, False: failProb})
    else:
        # if obj is elsewhere, we only see it here as a false positive
        return DD({True: failProb, False: 1 - failProb})

def lookStateObsModel(trueState, pObjAtLookLoc, failProb):
    objObsDist = dist.MixtureDist(dist.DeltaDist(trueState),
                                  dist.UniformDist(possibleStates),
                                  1 - failProb)
    # This should really integrate over other objecs that might be there
    otherObjDist = dist.UniformDist(possibleStates)
    return dist.MixtureDist(objObsDist, otherObjDist, pObjAtLookLoc)
    
def lookObsModelOther(oldLoc, loc, failProb, noiseProb):
    # We're doing, Look(obj, loc), this is update for objLoc[obj']
    # That is, what we learn about other objects when looking for obj
    if oldLoc == loc:
        # We get True, when looking for obj in loc, only when look fails
        return DD({True: failProb, False: 1 - failProb})
    else:
        # obj' is elsewhere
        return DD({True: noiseProb, False: 1 - noiseProb})
    
def lookBProgress(details, args, obs = None):
    print 'Observation', obs
    failProb = failProbs['Look']
    [targetObj, lookLoc, _, _] = args
    # Do the update at the target loc
    obsModel = lambda trueLoc: lookObsModelTarget(trueLoc, lookLoc, failProb)
    details.objLoc[targetObj].obsUpdate(obsModel, obs)

    # If I look for object a at location l, how should that change my
    # beliefs about the locations of other objects?
    # Leave it alone for now, and just do the obvious update on the
    # object we were looking for.
    
def lookNotBProgress(details, args, obs = None):
    # Trying to convince ourselves that object is not at lookLoc
    print 'Observation', obs
    failProb = failProbs['Look']
    [targetObj, lookLoc, _, _] = args
    obsModel = lambda trueLoc: lookObsModelTarget(trueLoc, lookLoc, failProb)
    details.objLoc[targetObj].obsUpdate(obsModel, obs)

def lookStateBProgress(details, args, obs = None):
    print 'Observation', obs
    failProb = failProbs['Look']
    [targetObj, lookLoc, _, _, _, _] = args
    assert isGround(args)
    pObjAtLookLoc = details.objLoc[targetObj].prob(lookLoc)
    obsModel = lambda trueState: lookStateObsModel(trueState, pObjAtLookLoc,
                                                   failProb)
    details.objState[targetObj].obsUpdate(obsModel, obs)
    

######################################################################
#
# Prim functions map an operator's arguments to some parameters that
#  are used during execution
#
######################################################################

def lookPrim(args, details):
    return args[:2]  # obj and loc


######################################################################
#
#  Generators and cost functions
#
######################################################################

# LPK: The form of these has gotten complicated for dumb reasons...

class GenNone(Function): 
    @staticmethod
    def fun(args, goal, start):
        return None

class GenObjAtLoc(Function):
    @staticmethod
    def fun(args, goal, start):
        (loc, conds) = args
        # Dict of possible objects to move out of the way
        objsAtLoc = dict(start.getObjsPossiblyAtLoc(loc))
        # Devalue ones we're already committed to moving
        gb = getMatchingFluents(conds,
                    Bd([NotObjLoc(['Obj', 'Loc']), True,'P'], True))
        for (fluent, bindings) in gb:
            if bindings['Loc'] == loc:
                obj = bindings['Obj']
                objsAtLoc[obj] = min(objsAtLoc[obj], 1 - bindings['P'])
        oList = objsAtLoc.items()
        oList.sort(key = lambda x: x[1], reverse = True)
        # Sort from most to least likely
        answer = [[o] for (o, p) in oList]
        debugMsg('genObjAtLoc', loc, oList, answer)
        return answer

class GenObjToClearLook(Function):
    @staticmethod
    def fun(args, goal, start):
        (loc,) = args
        # List of possible objects to try to remove by looking
        # Most likely one first
        objsAtLoc = start.getObjsPossiblyAtLoc(loc)
        objsAtLoc.sort(key = lambda x: x[1])
        debugMsg('genObjAtLoc', loc, objsAtLoc)
        return objsAtLoc

# Is this object actually required to be somewhere by these conditions?
def objLocatedIn(obj, conditions):
    for c in conditions:
        if c.predicate == 'Bd' and c.args[0].predicate == 'ObjLoc' and \
                            c.args[0].args[0] == obj:
            return True
    return False


class GenLikelyLoc(Function):
    @staticmethod
    def fun(args, goal, start):
        # When we're looking for an object, what's the best place to start?
        (obj,) = args
        objLocDist = start.objLocDist(obj)

        # Keep clears.
        gbs = getMatchingFluents(goal,
                            Bd([Clear(['Loc', 'Cond']), True, 'P'], True))
        kc = [b['Loc'] for (f, b) in gbs if not objLocatedIn(obj, b['Cond'])]
    
        candidates = [(loc, objLocDist.prob(loc)) \
                      for loc in objLocDist.support() if not loc in kc]

        candidates.sort(key = lambda x: x[1], reverse = True)
        return [[l] for (l, p) in candidates]

class GenLookStateLoc(Function):
    @staticmethod
    def fun(args, goal, start):
        # When we're looking at an object trying to assess its state,
        # where should we look?
        (obj,) = args
        objLocDist = start.objLocDist(obj)

        candidates = [(loc, objLocDist.prob(loc)) \
                      for loc in objLocDist.support() + opLocs.values()]

        candidates.sort(key = lambda x: x[1], reverse = True)
        return [[l] for (l, p) in candidates]

usefulLocs = ['painter', 'washer', 'dryer']

class GenDest(Function):
    @staticmethod
    def fun(args, goal, start):
        (loc, obj) = args

        gbs = getMatchingFluents(goal,
                                Bd([Clear(['Loc', 'Cond']), True, 'P'], True))
        kc = [b['Loc'] for (f, b) in gbs]

        # Possible destinations that we are not trying to keep clear
        dests = [l for l in start.locations if not l in kc and l != loc]
    
        candidates = []
        for dest in dests:
            for obj in start.objLoc.keys():
                objClearProbs = [1 - d.prob(dest) \
                                for (o, d) in start.objLoc.items() if o != obj]
            clearProb = reduce(mul, objClearProbs, 1)
            if dest in usefulLocs:
                # Penalize useful locations
                clearProb -= 0.5
            candidates.append((clearProb, dest))
        debugMsg('genDest', 'candidates', sorted(candidates, reverse = True))
        # Heuristic value ordering, by clear probability
        return [[dest] for (p, dest) in sorted(candidates, reverse = True)]

class GenClearPrecond(Function):
    @staticmethod
    def fun(args, goal, start):
        (loc, obj, postCond, prob) = args
        return [[simplifyCond(postCond, [Bd([NotObjLoc([obj, loc]), True, prob],
                                        True)])]]

class MoveStateRegress(Function):
    @staticmethod
    def fun(args, goal, start):
        # post = (1 - dirtyProb) * prior
        # prior = post / (1 - dirtyProb)
        (posterior,) = args
        prior = posterior / (1.0 - dirtyProb)
        if prior > 1:
            return None
        return [[prior]]

class PrPerObj(Function):
    @staticmethod
    def fun(args, goal, start):
        (pr,) = args
        n = len(start.objLoc)
        return [[np.power(pr, 1.0/n)]]

class RegressProb(Function):
    isNecessary = True
    # Compute the nth root of the maximum defined prob value.  Also, attenuated
    # by an error probability found in domainProbs
    def __init__(self, outVars, inVars, isNecessary = False, fp = None):
        self.n = len(outVars)
        self.failProb = fp
        super(RegressProb, self).__init__(outVars, inVars, isNecessary)
    def fun(self, args, goal, start):
        failProb = self.failProb if (self.failProb is not None) else 0.0
        pr = max([a for a in args if not isVar(a)]) / (1 - failProb)
        # noinspection PyTypeChecker
        val = np.power(pr, 1.0/self.n)
        if val < maxRequirableCertainty:
            return [[val]*self.n]
        else:
            return []
    def applyBindings(self, bindings):
        return self.__class__([lookup(v, bindings) for v in self.outVars],
                              [lookup(v, bindings) for v in self.inVars],
                              self.isNecessary, self.failProb)
    

class LookRegressProb(Function):
    @staticmethod
    def fun(args, goal, start):
        (pr,) = args
        if pr < minAllowableProb:
            return None
        # assume we see the object
        # pr = (1 - fp) * pp / ((1 - fp)*pp + fp*(1-pp))
        fp = failProbs['Look']
        pp = fp * pr / (1 - fp - pr + 2 * fp * pr)
        if pp <= 0:
            pp = minAllowableProb
        return [[pp]]

class LookStateRegressProb(Function):
    @staticmethod
    def fun(args, goal, start):
        (pr,) = args
        # assume we see the object
        # pr = (1 - fp) * pp / ((1 - fp)*pp + fp*(1-pp))
        locProb = math.sqrt(pr)
        # Fails if look doesn't work or the object wasn't really there
        fp = 1 - (1 - failProbs['Look']) * locProb
        pp = fp * pr / (1 - fp - pr + 2 * fp * pr)
        if pp <= 0 or pr < minAllowableProb:
            return None
        return [[pp, locProb]]

class LookNegRegressProb(Function):
    @staticmethod
    def fun(args, goal, start):
        (pr,) = args
        # assume we see the object
        # pr = (1 - fp) * pp / ((1 - fp)*pp + fp*(1-pp))
        if pr < minAllowableProb:
            return None
        fp = failProbs['Look']
        pp = fp * pr / (1 - fp - pr + 2 * fp * pr)
        if pp <= 0:
            pp = minAllowableProb
        return [[pp]]

    
# How much do we think it will cost to replan and achieve the goal
# from the resulting state
replanCost = 10
# If we try to move and mess up, what will that cost us?
moveDamageCost = 10 # was 50
# Weight on observation costs
obsProbCostFactor = 6


# Consequences of a failed move are greater than of a failed look

opCosts = {'Move' : 1, 'Paint' : 1, 'Dry' : 1, 'Wash' : 1, 'Look': 1}

def costFun(op):
    def cf(absLevel, args, details):
        return opCosts[op]
    return cf

def moveCostFun(absLevel, args, details):
    (o, s, d, loc, state, pr, pr1, pr2, pr3, pr4) = args
    return opCosts['Move'] + (1 - pr) * moveDamageCost

def lookCostFun(absLevel, args, details):
    (obj, loc, pr, pr1) = args
    # Prob of failure is prob we don't get the observation we
    # want.  That is bounded below by the prior prob times (1 -
    # failProb), but that's often too conservative.  So let's try
    # pr times (1 - failProb)
    obsProb = pr1 * (1 - failProbs['Look'])
    #return opCosts['Look'] + (1 - obsProb) * replanCost
    # previous one was willing to do ridiculously low prob looks
    # return opCosts['Look'] + replanCost / obsProb
    return opCosts['Look'] + replanCost - obsProbCostFactor * math.log(obsProb)

def lookStateCostFun(absLevel, args, details):
    (obj, loc, state, pr, pr1, pr2) = args
    # Prob of failure is prob we don't get the observation we
    # want.  That is bounded below by the prior prob times (1 -
    # failProb), but that's often too conservative.  
    # Multiply by pr2 (obj has to be at the right location)
    
    obsProb = pr1 * (1 - failProbs['Look']) * pr2
    
    #return opCosts['Look'] + (1 - obsProb) * replanCost
    # previous one was willing to do ridiculously low prob looks
    # return opCosts['Look'] + replanCost / obsProb
    return opCosts['Look'] + replanCost - obsProbCostFactor * math.log(obsProb)

######################################################################
#
# Belief
# 
######################################################################

class Belief:
    def __init__(self, objLoc, objState, locations, occupancies = None):
        self.objLoc = objLoc              # dict: {obj : locDist}
        self.objState = objState          # dict: {obj : stateDist}
        self.locations = locations
        if occupancies:
            self.occupancies = occupancies
        else:
            # Assume uniform
            occProb = len(objLoc)/float(len(locations))
            self.occupancies = dict([(l, DD({True : occProb,
                                             False: 1 - occProb})) \
                                 for l in self.locations])

        # Non-overlapping object constraints
        noFun = lambda a, b: 0 if a == b else 1
        noC = [CFFConstraint((o1, o2), noFun, 'locMutex') \
               for o1 in self.objLoc.keys() \
               for o2 in self.objLoc.keys() if o1 != o2]
        # occupancy constraints
        def occFun(l): 
            return lambda loc, occ: 0 if (loc == l and not occ) else 1
        def somewhereFun(l):
            def f(*args):
                objLocs = args[:-1]
                occ = args[-1]
                if occ == 1:
                    # some object should be at l
                    return any([ol == l for ol in objLocs])
                else:
                    # no object should be at l
                    return not any([ol == l for ol in objLocs])
            return f

        # If an object is in a location, then that location is occupied
        occC1 = [CFFConstraint([obj, loc], occFun(loc), 'locImpliesOcc') \
                for obj in self.objLoc.keys() \
                for loc in self.locations]
        # If a location is occupied, some object is in it
        # This wouldn't be true if there could be extra stuff in the world
        occC2 = [CFFConstraint(self.objLoc.keys() + [loc],
                               somewhereFun(loc), 'occImpliesSomeLoc') \
                 for loc in self.locations]

        self.cff = CFF(dict(objLoc.items() + self.occupancies.items()),
                       noC + occC1 + occC2)
        # Show off
        self.propagate()
        self.draw()

    def objLocDist(self, obj):
        return self.cff.marginals[obj]
    def locOccDist(self, loc):
        return self.cff.marginals[loc]

    def getObjsPossiblyAtLoc(self, loc):
        return [(obj, self.objLocDist(obj).prob(loc)) \
                for obj in self.objLoc.keys() \
                if loc in self.objLocDist(obj).support()]
    

    def propagate(self):
        # Compute and store marginal distributiosn as needed
        self.cff.computeMarginals()

    def draw(self):
        print '**************************************************'
        print ' New Belief'
        for obj in self.objLoc.keys():
            print 'Obj', obj
            print '    LocDist', self.objLocDist(obj)
            print '    StateDist', self.objState[obj]
        for loc in self.occupancies.keys():
            print 'LocOcc', loc, self.locOccDist(loc)
        print '**************************************************'
        print '**************************************************'
        print 'Prim belief components'
        for obj in self.objLoc.keys():
            print 'Obj', obj
            print '    LocDist', self.objLoc[obj]
        for loc in self.occupancies.keys():
            print 'LocOcc', loc, self.occupancies[loc]
        print '**************************************************'
        debugMsg('newBelief')

def worldFromBelief(b):
    def legal(stuff):
        locs = [l for (s, l) in stuff.values()]
        return len(set(locs)) == len(locs)
    stuff = dict([(o, [b.objState[o].draw(), b.objLocDist(o).draw()]) \
                  for o in b.objLoc.keys()])
    # Terrible rejection sampling.  Should do it internal to picking locations. 
    while not legal(stuff):
        stuff = dict([(o, [b.objState[o].draw(), b.objLocDist(o).draw()]) \
                  for o in b.objLoc.keys()])
    return World(stuff)

######################################################################
#
# Fluents
# 
######################################################################

class ObjState(Fluent):
    predicate = 'ObjState'
    
    def dist(self, details):
        (obj,) = self.args
        return details.objState[obj]

class ObjLoc(Fluent):
    predicate = 'ObjLoc'

    def dist(self, details):
        (obj,) = self.args
        return details.objLocDist(obj)

    def contradicts(self, other, details = None):
        if super(ObjLoc, self).contradicts(other, details):
            return True
        (sObj,) = self.args
        if other.predicate == 'ObjLoc':
            (oObj,) = other.args
            # Asserts that a different obj is in location, which holds only one
            return sObj != oObj and self.value == other.value
        elif other.predicate == 'Clear':
            (oLoc, conds) = other.args
            return oLoc == self.value

class NotObjLoc(Fluent):
    predicate = 'NotObjLoc'

    def dist(self, details):
        (obj, loc) = self.args
        p = details.objLocDist(obj).prob(loc)
        return DD({True: 1 - p, False: p})

    def contradicts(self, other, details = None):
        if super(NotObjLoc, self).contradicts(other, details):
            return True
        (sObj, sLoc) = self.args
        if other.predicate == 'ObjLoc':
            (oObj,) = other.args
            return sObj == oObj and sLoc == other.value

class Clear(Fluent):
    predicate = 'Clear'
    implicit = True
    conditional = True
    
    def bTest(self, details, v, p):
        assert v == True
        (loc, conds) = self.args

        # Dict - for each obj, prob it is not in loc
        objClearProbs = dict([(o, 1 - details.objLocDist(o).prob(loc)) \
                              for o in details.objLoc.keys()])

        # Remove ones that have been conditionalized to be elsewhere
        gb = getMatchingFluents(conds,
                                Bd([ObjLoc(['Obj']), 'Loc', 'P'], True))
        for (fluent, bindings) in gb:
            obj = bindings['Obj']
            if bindings['Loc'] == loc:
                # We think this object *is* in loc.  So, prob it is not in
                # the loc is lower.
                objClearProbs[obj] = min(1 - bindings['P'],
                                         objClearProbs[obj])
            else:
                # We think the object *is not* in loc.  So prob it is not in
                # the loc is higher.
                objClearProbs[obj] = max(bindings['P'],
                                         objClearProbs[obj])
        gb = getMatchingFluents(conds,
                                Bd([NotObjLoc(['Obj', 'Loc']), True,'P'], True))
        for (fluent, bindings) in gb:
            obj = bindings['Obj']
            if bindings['Loc'] == loc:
                objClearProbs[obj] = max(bindings['P'],
                                         objClearProbs[obj])

        objClearProb = reduce(mul, objClearProbs.values(), 1)
        directClearProb = details.locOccDist(loc).prob(False)

        # conservative doesn't work...
        clearProb = max(directClearProb, objClearProb)

        debugMsg('clearTest', ('args', self.args),
                 objClearProbs, directClearProb,
                 ('result', clearProb, p, clearProb >= p))
        
        return clearProb >= p

    def contradicts(self, other, details = None):
        if super(Clear, self).contradicts(other, details):
            return True
        return other.predicate == 'ObjLoc' and other.value == self.args[0]

######################################################################
#
# Operators
# 
######################################################################

# Operators have a prim, which is a function, run at execution time, from
# the current belief details to parameters of the operation

def defaultPrim(args, details):
    return None

# Make the cost of a move operation depend on how certain we are about
# its initial pose.  Story is that it is more risky (even if the bel
# space update is deterministic.)

def makeOperators(failProbs):
    # We don't want to establish a the goal of moving an object unless
    # its pose is moderately well known ** in the now **

    # A bit difficult because we actually let Start be picked by the
    # next oepration, through backtracking.
    move = Operator(\
            'Move', ['Obj', 'Start', 'Dest', 'Loc', 'State',
                     'PR', 'PR1', 'PR2', 'PR3', 'PR4'],
            # Pre
            {0 : {Bd([Clear(['Dest', []]), True, 'PR2'], True)},
             1 : {Bd([ObjLoc(['Obj']), 'Start', 'PR1'], True)}},
            # Results, each with private preconds
            # Main result: object gets moved
            [({Bd([ObjLoc(['Obj']), 'Dest', 'PR'], True)}, {}),
             # In the process of moving, may degrade state
             ({Bd([ObjState(['Obj']), 'State', 'PR3'], True)},
              {0 : {Bd([ObjState(['Obj']), 'State', 'PR4'], True)}})],
            functions = [\
                GenNone(['Dest'], []),
                GenObjAtLoc(['Obj'], ['Loc']),
                MoveStateRegress(['PR4'], ['PR3']),
                RegressProb(['PR1', 'PR2'], ['PR'], fp = failProbs['Move'])],
            cost = moveCostFun,
            f = moveBProgress,
            prim = defaultPrim,
            ignorableArgs = range(3, 7))

    wash = Operator(\
            'Wash', ['Obj', 'PR', 'PR1'],
            {0 : set([]),
             1 : {Bd([ObjLoc(['Obj']), 'washer', 'PR1'], True)}},
            [({Bd([ObjState(['Obj']), 'clean', 'PR'], True)}, {})],
            functions = [\
                RegressProb(['PR1'], ['PR'], fp = failProbs['Wash'])],
            cost = costFun('Wash'),
            f = stateBProgress('clean', 'washer', failProbs['Wash']),
            prim = defaultPrim,
            ignorableArgs = range(1, 3))

    paint = Operator(\
            'Paint', ['Obj', 'PR', 'PR1', 'PR2'],
            {0 : {Bd([ObjState(['Obj']), 'clean', 'PR1'], True)},
             1 : {Bd([ObjLoc(['Obj']), 'painter', 'PR2'], True)}},
            [({Bd([ObjState(['Obj']), 'wetPaint', 'PR'], True)}, {})],
            functions = [\
                RegressProb(['PR1', 'PR2'], ['PR'], fp = failProbs['Paint'])],
            cost = costFun('Paint'),
            f = stateBProgress('wetPaint', 'painter', failProbs['Paint']),
            prim = defaultPrim,
            ignorableArgs = range(1, 4))
            

    dry = Operator(\
            'Dry', ['Obj', 'PR', 'PR1', 'PR2'],
            {0 : set(),
             1 : {Bd([ObjState(['Obj']), 'wetPaint', 'PR1'], True)},
             2 : {Bd([ObjLoc(['Obj']), 'dryer', 'PR2'], True)}},
            [({Bd([ObjState(['Obj']), 'dryPaint', 'PR'], True)},{})],
            functions = [\
                RegressProb(['PR1', 'PR2'], ['PR'], fp = failProbs['Dry'])],
            cost = costFun('Dry'),
            f = stateBProgress('dryPaint', 'dryer', failProbs['Dry']),
            prim = defaultPrim,
            ignorableArgs = range(1, 4))
            

    look = Operator(\
            'Look', ['Obj', 'Loc', 'PR', 'PR1'],
            #{0 : set([]),
             #1 : {BVd([ObjLoc(['Obj']), 'PVR'], True)},
             #1 : {Bd([ObjLoc(['Obj']), 'Loc', 'PR1'], True)}},
            {0 : {Bd([ObjLoc(['Obj']), 'Loc', 'PR1'], True)}},
            [({Bd([ObjLoc(['Obj']), 'Loc', 'PR'], True)}, {})],
            functions = [GenLikelyLoc(['Loc'], ['Obj']),
                         LookRegressProb(['PR1'], ['PR'])],
            cost = lookCostFun,
            prim = lookPrim,
            f = lookBProgress,
            ignorableArgs = 2)
            

    lookState = Operator(\
            'LookState', ['Obj', 'Loc', 'State', 'PR', 'PR1', 'PR2'],
            {0 : {Bd([ObjState(['Obj']), 'State', 'PR1'], True)},
             1 : {Bd([ObjLoc(['Obj']), 'Loc', 'PR2'], True)}},
            [({Bd([ObjState(['Obj']), 'State', 'PR'], True)}, {})],
            functions = [\
                         GenNone(['State'], ['Obj']),
                         LookStateRegressProb(['PR1', 'PR2'], ['PR'])],
            cost = lookStateCostFun,
            prim = lookPrim,
            f = lookStateBProgress,
            ignorableArgs = range(3, 6))
            
    
    # Systematic approach to clearing a location
    # 1.  Move an object out of the location
    # 2.  Look at the location with the goal of seeing that it's clear
    #       (not strictly necessary)
    # 3.  Look for a particular object there, to verify it is not there.

    clearObj = Operator(\
                'ClearObj',
                ['Loc', 'Obj', 'Conditions', 'PreConditions', 'PR', 'PR1'],
                {0 : set(),
                 1 : {Bd([NotObjLoc(['Obj', 'Loc']), True, 'PR1'], True),
                      Bd([Clear(['Loc', 'PreConditions']), True, 'PR'], True)}},
                [({Bd([Clear(['Loc', 'Conditions']), True, 'PR'], True)}, {})],
                functions = [\
                    GenObjAtLoc(['Obj'], ['Loc', 'Conditions']),
                    PrPerObj(['PR1'], ['PR']),
                    GenClearPrecond(['PreConditions'],
                                    ['Loc', 'Obj', 'Conditions', 'PR1'])],
                cost = lambda al, args, details: 0.1,
                ignorableArgs = range(2, 5))

    # Could wish to have a guard condition here.  B obj at loc.   We
    # don't really have a way of doing this.  So, we really do have to
    # replan.
    moveC = Operator(\
                'MoveToClear',
                ['Loc', 'Obj', 'Dest', 'PR'],
                {0 : {Bd([ObjLoc(['Obj']), 'Dest', 'PR'], True)}},
                [({Bd([NotObjLoc(['Obj', 'Loc']), True, 'PR'], True)}, {})],
                functions = [\
                    GenDest(['Dest'], ['Loc', 'Obj'])],
                cost = lambda al, args, details: 0.1,
                ignorableArgs = range(3, 4))
                
                
    lookC = Operator(\
            'LookToClear', ['Obj', 'Loc', 'PR', 'PR1'],
            {0 : set(),
             1 : {Bd([NotObjLoc(['Obj', 'Loc']), True, 'PR1'], True)}},
            [({Bd([NotObjLoc(['Obj', 'Loc']), True, 'PR'], True)}, {})],
            functions = [\
                         LookNegRegressProb(['PR1'], ['PR'], True)],
            cost = lookCostFun,
            prim = lookPrim,
            f = lookBProgress,
            ignorableArgs = range(2, 4))
    
    return dict([(op.name, op) for op in move, wash, paint, dry, look, moveC,
                 lookC, clearObj, lookState])

######################################################################
#
# Simulator
#
######################################################################

# Wash, paint, dry, get soap, get paint, put on truck
# have to be sure there's space in dryer...will be ruined if dry doesn't
# happen right after paint

# Objects is a dictionary from names to object (state, loc)
# object state can be 'dirty', 'clean', 'wetPaint', 'dryPaint'
# loc can be: 'inventory', 'washer', 'painter', 'dryer', 'truck'
# only one object can be in 'washer', 'painter', 'dryer'
# objects must be washed before painted, painted before dried

# Operators are: move, wash, paint, dry

possibleStates = {'clean', 'dirty', 'wetPaint', 'dryPaint'}
opLocs = {'Wash' : 'washer', 'Paint' : 'painter', 'Dry' : 'dryer'}
precondStates = {'wetPaint' : {'clean'}, 'dryPaint' : {'wetPaint'},
                'clean' : {'dirty', 'wetPaint', 'dryPaint'}}
 
class World:
    def __init__(self, objects):
        # This is a dictionary of {obj: (state, loc)}
        self.objects = objects

    def objAt(self, loc):
        return [o for (o, (s, l)) in self.objects.items() if l == loc]

    def objState(self, obj):
        return self.objects[obj][0]
    def setObjState(self, obj, s):
        self.objects[obj][0] = s

    def objLoc(self, obj):
        return self.objects[obj][1]
    def setObjLoc(self, obj, l):
        self.objects[obj][1] = l

    resultProps = {'Wash' : 'clean', 'Paint' : 'wetPaint', 'Dry' : 'dryPaint'}

    # Side effects the world state
    def executePrim(self, op, params = None):
        dirtyProb = 0.1
        obs = None
        if op.name == 'Move':
            (_, start, dest, _, _, _, _, _, _, _) = op.args
            obj = self.objAt(start)
            if obj:
                o = obj[0]
            occluder = self.objAt(dest)
            print 'move objat', obj
            if obj == []:
                obs = 'whiff'
            elif occluder != []:
               # no move if dest is occupied
                obs = 'bump'
            else:
                resultLoc = DD({dest : 1 - failProbs['Move'],
                                start : failProbs['Move']})
                self.objects[o][1] = resultLoc.draw()

            print 'Move obs', obs

            if obj:
                # some chance it gets dirty
                objState = self.objects[o][0]
                resultState = DD({objState : 1 - dirtyProb})
                resultState.addProb('dirty' , dirtyProb)
                self.objects[o][0] = resultState.draw()

        elif op.name in ['Wash', 'Paint', 'Dry']:
            opLoc = opLocs[op.name]
            prop = self.resultProps[op.name]
            for obj in self.objAt(opLoc):
                if (op.name == 'Paint' and self.objState(obj) != 'clean') or \
                   (op.name == 'Dry' and self.objState(obj) != 'wetPaint'):
                    print 'Precond not met', op.name, obj, self.objState(obj)
                else:
                    resultDist = stateTransModel(self.objState(obj), prop,
                                             failProbs[op.name], 1)
                    self.setObjState(obj, resultDist.draw())

        elif op.name in ('Look', 'LookToClear'):
            (obj, loc) = params
            trueLoc = self.objLoc(obj)
            fp = failProbs['Look']
            if trueLoc == loc:
                obs = DD({True: 1 - fp, False: fp}).draw()
            else:
                obs = DD({True: fp, False: 1 - fp}).draw()

        elif op.name in ('LookState'):
            (obj, loc) = params
            trueLoc = self.objLoc(obj)
            trueState = self.objState(obj)
            fp = failProbs['Look']
            if trueLoc == loc:
                obs = dist.MixtureDD(dist.DeltaDist(trueState),
                                       dist.UniformDist(possibleStates),
                                       1-fp).draw()
            else:
                objAtLoc = self.objAt(loc)
                if len(objAtLoc) > 0:
                    observedObj = objAtLoc.pop()
                    observedObjState = self.objState(observedObj)
                    obs = dist.MixtureDD(dist.DeltaDist(observedObjState),
                                           dist.UniformDist(possibleStates),
                                        1-fp).draw()
                else:
                    obs = dist.UniformDist(possibleStates).draw()
                
        else:
            raise Exception, 'Unknown operator: '+str(op)
        print 'executePrim', op, 'obs', obs
        self.draw()
        return obs

    def copy(self):
        return copy.copy(self)

    def draw(self):
        print '**************************************************'
        print ' New World State'
        for obj in self.objects:
            print 'Obj', obj, self.objLoc(obj),  self.objState(obj)
        print '**************************************************'
        debugMsg('newState')
    

######################################################################
#
# Testing stuff
# 
######################################################################

writeSearch = True

s = None
operators = None

def planTest(name, objects, beliefDetails,
             goal, skeleton = None, hpn = True, h = None):
    global s
    global operators

    operatorDict = makeOperators(failProbs)
    operators = operatorDict.values()
    s = State([], beliefDetails)
    if hpn:
        world = World(objects)
        if skeleton != None:
            skeleton = [[operatorDict[opName] for opName in ski] \
                             for ski in skeleton]
        HPN(s, goal, operators, world,
            skeleton = skeleton,
            h = h,
            hpnFileTag = name,
            fileTag = name if writeSearch else None)
    else:
        if skeleton != None:
            skeleton = [operatorDict[opName] for opName in skeleton]
        p = planBackward(s, goal, operators,
                         skeleton = skeleton, 
                         h = h,
                         fileTag = name if writeSearch else None)
        if p:
            makePlanObj(p, s).printIt()
        else:
             print 'Planning failed'

def planTestRandom(name, bDetails, goal, h, failProbs):
    global s
    global operators

    operatorDict = makeOperators(failProbs)
    operators = operatorDict.values()
    s = State([], bDetails)
    world = worldFromBelief(bDetails)
    HPN(s, goal, operators, world, h = h,
        hpnFileTag = name,
        fileTag = name if writeSearch else None)

def objDictCopy(od):             
    return dict([(o, copy.copy(s)) for (o, s) in od.items()])

objects1 = {'a' : ['dirty', 'i1'],
            'b' : ['clean', 'i2']}

objects2 = {'a': ['dirty', 'dryer'], 'b': ['dirty', 'inventory']}    

locations = ['washer', 'painter', 'dryer', 'i1', 'i2', 'i3', 'i4']
#failProbs = {'Move' : 0, 'Paint' : 0, 'Dry' : 0, 'Wash' : 0, 'Look': 0}


# Three locations, one object
def test0(hpn = True, useSkeleton = False, hierarchical = False,
          heuristic = BBhAddBackBSet):
    global failProbs
    name = 'test0'
    fbch.flatPlan = not hierarchical
    locations = ('i0', 'i1', 'i2')
    failProbs = {'Move' : .1, 'Paint' : 0, 'Dry' : 0, 'Wash' : 0, 'Look': .1}
    objects0 = {'a' : ['dirty', 'i2']}

    bDetails = Belief({'a' : UniformDist(locations)},
                      {'a' : DD({'dirty' : 1.0})},
                      locations)
    
    goal = State([Bd([ObjLoc(['a']), 'i0', 0.95], True)])
    planTest(name, objDictCopy(objects0), bDetails, goal, hpn = hpn,
             h = heuristic)

def test1(hpn = True, useSkeleton = False, hierarchical = False,
          heuristic = BBhAddBackBSet):
    global failProbs
    name = 'test1'
    fbch.flatPlan = not hierarchical
    failProbs = {'Move' : .1, 'Paint' : .2, 'Dry' : .3, 'Wash' : .1, 'Look': .2}

    def op(l):
        return 0.99 if l in ('i1', 'i2') else 0.01

    occDists = dict([(l, DD({True : op(l), False: 1 - op(l)})) \
                        for l in locations])

    bDetails = Belief({'a' : DD({'i1' : 1.0}),
                       'b' : DD({'i2' : 1.0})},
                      {'a' : DD({'dirty' : 1.0}),
                       'b' : DD({'dirty' : 0.5, 'clean' : 0.5})},
                      locations,
                      occDists)
    
    goal = State([Bd([ObjState(['a']), 'clean', 0.7], True)])
    skeleton = [['Wash', 'Move']]
    planTest(name, objDictCopy(objects1), bDetails, goal, hpn = hpn,
             h = heuristic,
             skeleton = skeleton if useSkeleton else None)

def test2(hpn = True, useSkeleton = False, hierarchical = True,
          heuristic = BBhAddBackBSet):
    # Do a bunch of steps.  Only uncertainty is in moving and looking
    global failProbs
    name = 'test2'
    fbch.flatPlan = not hierarchical
    failProbs = {'Move' : .2, 'Paint' : 0, 'Dry' : 0, 'Wash' : 0, 'Look': 0.2}

    bDetails = Belief({'a' : DD({'i1' : 1.0}),
                       'b' : DD({'i2' : 1.0})},
                      {'a' : DD({'dirty' : 1.0}),
                       'b' : DD({'dirty' : 0.5, 'clean' : 0.5})},
                      locations)
    
    goal = State([Bd([ObjState(['a']), 'dryPaint', 0.8], True)])
    #goal = State([Bd([ObjState(['a']), 'wetPaint', 0.8], True)])
    #goal = State([Bd([ObjState(['a']), 'clean', 0.8], True)])

    planTest(name, objDictCopy(objects1), bDetails, goal, hpn = hpn,
             h = heuristic,
             skeleton = skeleton if useSkeleton else None)

def test2h(p, verbose = True):
    # Do a bunch of steps.  Only uncertainty is in moving and looking
    global failProbs
    failProbs = {'Move' : .2, 'Paint' : 0, 'Dry' : 0, 'Wash' : 0, 'Look': 0.2}

    bDetails = Belief({'a' : DD({'i1' : 1.0}),
                       'b' : DD({'i2' : 1.0})},
                      {'a' : DD({'dirty' : 1.0}),
                       'b' : DD({'dirty' : 0.5, 'clean' : 0.5})},
                      locations)
    
    hTest(bDetails, Bd([ObjState(['a']), 'dryPaint', p], True), verbose = verbose)
    
def test3(hpn = True, useSkeleton = False, hierarchical = True,
          heuristic = BBhAddBackBSet):
    # Object in the way; initial poses known
    objects2 = {'a': ['dirty', 'dryer'], 'b': ['dirty', 'washer']}
    global failProbs
    name = 'test3'
    fbch.flatPlan = not hierarchical
    failProbs = {'Move' : .2, 'Paint' : 0, 'Dry' : 0, 'Wash' : 0, 'Look': 0.2}

    bDetails = Belief({'a' : DD({'dryer' : 1.0}),
                       'b' : DD({'washer' : 1.0})},
                      {'a' : DD({'dirty' : 1.0}),
                       'b' : DD({'dirty' : 0.5, 'clean' : 0.5})},
                      locations)
    
    goal = State([Bd([Clear(['washer', []]), True, 0.8], True)])
    planTest(name, objDictCopy(objects2), bDetails, goal, hpn = hpn,
             h = heuristic,
             skeleton = skeleton if useSkeleton else None)

def test4(hpn = True, useSkeleton = False, hierarchical = True,
          heuristic = BBhAddBackBSet):
    #raw_input('Will fail in monotonic mode')
    
    # Object in the way; initial poses known
    objects2 = {'a': ['dirty', 'dryer'], 'b': ['dirty', 'washer']}
    global failProbs
    name = 'test4'
    fbch.flatPlan = not hierarchical
    failProbs = {'Move' : .2, 'Paint' : 0, 'Dry' : 0, 'Wash' : 0, 'Look': 0.2}

    bDetails = Belief({'a' : DD({'dryer' : 1.0}),
                       'b' : DD({'washer' : 1.0})},
                      {'a' : DD({'dirty' : 1.0}),
                       'b' : DD({'dirty' : 0.5, 'clean' : 0.5})},
                      locations)
    
    goal = State([Bd([ObjState(['a']), 'dryPaint', 0.8], True)])
    skeleton = ['Wash', 'Move']
    planTest(name, objDictCopy(objects2), bDetails, goal, hpn = hpn,
             h = heuristic,
             skeleton = skeleton if useSkeleton else None)

def test5(hpn = True, useSkeleton = False, hierarchical = True,
          heuristic = BBhAddBackBSet):
    # Treausre hunt
    objects2 = {'a': ['dirty', 'dryer'], 'b': ['dirty', 'washer']}
    global failProbs
    name = 'test5'
    fbch.flatPlan = not hierarchical
    failProbs = {'Move' : .2, 'Paint' : 0, 'Dry' : 0, 'Wash' : 0, 'Look': 0.2}
    failProbs = {'Move' : .05, 'Paint' : .02, 'Dry' : .02, 'Wash' : .02,
                 'Look': 0.05}


    bDetails = Belief({'a' : dist.UniformDist(locations),
                       'b' : DD({'washer' : 1.0})},
                      {'a' : DD({'dirty' : 1.0}),
                       'b' : DD({'dirty' : 0.5, 'clean' : 0.5})},
                      locations)
    
    goal = State([Bd([ObjLoc(['a']), 'painter', 0.8], True)])
    goal = State([Bd([ObjState(['a']), 'dryPaint', 0.8], True)])
    skeleton = None
    planTest(name, objDictCopy(objects2), bDetails, goal, hpn = hpn,
             h = heuristic,
             skeleton = skeleton if useSkeleton else None)


def test6(hpn = True, useSkeleton = False, hierarchical = True,
          heuristic = BBhAddBackBSet):
    # Treausre hunt
    objects2 = {'a': ['dirty', 'dryer'], 'b': ['dirty', 'washer'],
                'c' : ['clean', 'painter']}
    global failProbs
    name = 'test6'
    fbch.flatPlan = not hierarchical
    failProbs = {'Move' : .2, 'Paint' : 0, 'Dry' : 0, 'Wash' : 0, 'Look': 0.2}
    failProbs = {'Move' : .05, 'Paint' : .0, 'Dry' : .0, 'Wash' : .0,
                 'Look': 0.05}

    bDetails = Belief({'a' : dist.UniformDist(locations),
                       'b' : DD({'washer' : 0.9, 'i1' : 0.1}),
                       'c' : DD({'dryer' : 0.9, 'painter' : 0.1})},
                      {'a' : DD({'dirty' : 1.0}),
                       'b' : DD({'dirty' : 0.5, 'clean' : 0.5}),
                       'c' : DD({'dirty' : 0.5, 'clean' : 0.5})},
                      locations)
    
    goal = State([Bd([ObjState(['a']), 'dryPaint', 0.9], True),
                  Bd([ObjState(['b']), 'dryPaint', 0.9], True),
                  Bd([ObjState(['c']), 'dryPaint', 0.9], True)])

    goal = State([Bd([ObjState(['a']), 'dryPaint', 0.95], True),
                  Bd([ObjState(['b']), 'dryPaint', 0.95], True),
                  Bd([ObjState(['c']), 'dryPaint', 0.95], True)])
    
    planTest(name, objDictCopy(objects2), bDetails, goal, hpn = hpn,
             h = heuristic,
             skeleton = skeleton if useSkeleton else None)


# To debug lookState
def test7(hpn = True, useSkeleton = False, hierarchical = True,
          heuristic = BBhAddBackBSet):
    objects2 = {'a': ['dirty', 'dryer'], 'b': ['dirty', 'washer'],
                'c' : ['clean', 'painter']}
    global failProbs
    name = 'test7'
    fbch.flatPlan = not hierarchical
    failProbs = {'Move' : .00001, 'Paint' : .2, 'Dry' : .2, 'Wash' : .2, 'Look': 0.05}

    bDetails = Belief({'a' : DD({'dryer' : 1.0}),
                       'b' : DD({'washer' : 1.0}),
                       'c' : DD({'painter' : 0.9, 'painter' : 0.1})},
                      {'a' : DD({'dirty' : 1.0}),
                       'b' : DD({'dirty' : 0.5, 'clean' : 0.5}),
                       'c' : DD({'dirty' : 0.5, 'clean' : 0.5})},
                      locations)
    
    goal = State([Bd([ObjState(['a']), 'dryPaint', 0.8], True),
                  Bd([ObjState(['b']), 'dryPaint', 0.8], True),
                  Bd([ObjState(['c']), 'dryPaint', 0.8], True)])

    goal = State([Bd([ObjState(['b']), 'clean', 0.9], True)])

    planTest(name, objDictCopy(objects2), bDetails, goal, hpn = hpn,
             h = heuristic,
             skeleton = skeleton if useSkeleton else None)
        


# Lots of objects

def test8(hpn = True, useSkeleton = False, hierarchical = True,
          heuristic = BBhAddBackBSet):
    someLocations = ['washer', 'painter', 'dryer', 'i1', 'i2', 'i3', 'i4']
    allLocations = someLocations + ['i5', 'i6', 'i7', 'i8', 'i9', 'i10',\
                                    'i11', 'i12', 'i13', 'i14', 'i15']
    # Treausre hunt
    objects2 = {'a': ['dirty', 'dryer'], 'b': ['dirty', 'washer']}
    global failProbs
    name = 'test8'
    fbch.flatPlan = not hierarchical
    failProbs = {'Move' : .2, 'Paint' : 0, 'Dry' : 0, 'Wash' : 0, 'Look': 0.2}
    failProbs = {'Move' : .05, 'Paint' : .02, 'Dry' : .02, 'Wash' : .02,
                 'Look': 0.05}


    bDetails = Belief({'a' : dist.UniformDist(someLocations),
                       'b' : DD({'washer' : 1.0})},
                      {'a' : DD({'dirty' : 1.0}),
                       'b' : DD({'dirty' : 0.5, 'clean' : 0.5})},
                      allLocations)
    
    goal = State([Bd([ObjState(['a']), 'dryPaint', 0.95], True)])
    skeleton = None
    planTest(name, objDictCopy(objects2), bDetails, goal, hpn = hpn,
             h = heuristic,
             skeleton = skeleton if useSkeleton else None)

failProbs = {}
    
def randomTest(numObjects = 3, numLocs = 8, numTrials = 10, highVarBel = False,
               savedState = None):
    global failProbs
    failProbs = {'Move' : .05, 'Paint' : .02, 'Dry' : .02, 'Wash' : .02,
                 'Look': 0.05}

    objNames = ['o'+str(i) for i in range(numObjects)]
    locNames = ['l'+str(i) for i in range(numLocs)] + \
                      ['washer', 'painter', 'dryer']
    stateNames = ['dirty', 'clean', 'wetPaint', 'dryPaint']

    name = 'randomTest_'+str(numObjects)+'_'+str(numLocs)

    if highVarBel:
        objLocDists = dict([(oname, dist.UniformDist(locNames))\
                            for oname in objNames])
        objStateDists = dict([(oname, dist.UniformDist(stateNames))\
                            for oname in objNames])
    else:
        objLocDists = dict([(oname,dist.MixtureDD(dist.DeltaDist(locNames[i]),
                                                     dist.UniformDist(locNames),
                                                     0.9)) \
                             for (i, oname) in enumerate(objNames)])
        objStateDists = dict([(oname,dist.MixtureDD(dist.DeltaDist(\
                                                            stateNames[i%4]),
                                                 dist.UniformDist(stateNames),
                                                     0.9)) \
                             for (i, oname) in enumerate(objNames)])

    if savedState:
        bDetails = savedState.details
    else:
        bDetails = Belief(objLocDists, objStateDists, locNames)

    goal = State([Bd([ObjState([o]), 'dryPaint', 0.95], True) \
                      for o in objNames])
    planTestRandom(name, bDetails, goal, h = BBhAddBackBSet, failProbs = failProbs)
    
        
