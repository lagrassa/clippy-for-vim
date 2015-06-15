import copy
import numpy as np

import planGlobals as glob
from planGlobals import debug, debugMsg

import fbch
from fbch import Fluent, applyBindings, hCache, State, applicableOps,\
    Operator

import miscUtil
from miscUtil import isVar, isAnyVar, floatify, isStruct, customCopy,\
    lookup, squash, squashSets, prettyString, isGround, matchLists


######################################################################
#
# Belief fluents.  First a generic class

class BFluent(Fluent):
    
    def isImplicit(self):
        return self.args[0].isImplicit()
    def isConditional(self):
        return self.args[0].isConditional()

    def conditionOn(self, f):
        return f.predicate in ('B', 'Bd') and \
          not ('*' in f.args) and \
          self.args[0].conditionOn(f.args[0]) 

    def addConditions(self, newConds, details = None):
        addConds = [c for c in newConds if self.conditionOn(c)]
        self.args[0].addConditions(addConds, details)

    def sameFluent(self, other):
        return self.isGround() and other.isGround() and \
          self.args[0].matchArgs(other.args[0]) != None

    def getIsGround(self):
        return self.args[0].isGround() and \
               all([not isAnyVar(a) for a in self.args[1:]]) \
               and not isAnyVar(self.value)

    def getIsPartiallyBound(self):
        b0 = self.args[0].isPartiallyBound()
        g0 = self.args[0].isGround()
        v0 = not b0 and not g0 # rf has no bindings
        av = [v0] + [isVar(a) for a in self.args[1:]]
        return (True in av) and (False in av)

    def couldClobber(self, other, details = None):
        if self.contradicts(other, details):
            return True
        if not other.predicate in ('B', 'Bd'):
            return False
        return self.args[0].couldClobber(other.args[0])

    def argsGround(self):
        return self.args[0].isGround()

    def getVars(self):
        return set([a for a in self.args[1:] if isAnyVar(a)]).union(\
                                                self.args[0].getVars())

    def shortName(self):
        return self.predicate + '(' + self.args[0].shortName() + \
                   ', ' + prettyString(self.args[-1]) +  ')'

    def update(self):
        assert isVar(self.args[-1]) or (0 <= self.args[-1] <= 1) or \
          self.args[-1] == None
        # Set the value in the embedded fluent
        self.args[0].value = self.args[1]
        # Generate the string
        self.args[0].update()
        # Do the update on this fluent (should call the parent method)
        self.isGroundStored = self.getIsGround()
        self.isPartiallyBoundStored = self.getIsPartiallyBound()
        self.strStored = self.getStr()

    # Avoid printing the value of the embedded rfluent
    def argString(self, eq):
        return '['+ self.args[0].prettyString(eq, includeValue = False) +', ' +\
                  ', '.join([prettyString(a, eq) for a in self.args[1:]]) + ']'

class Bd(BFluent):
    predicate = 'Bd'
    # Fluent for the planner to use.  Derived from an RFluent with a
    # Bernoulli distribution
    # Parameters are fluent, v, p

    def removeProbs(self):
        return Bd([self.args[0], self.args[1], None], self.value)

    def getGrounding(self, details):
        (rFluent, v, p) = self.args
        if not isGround(rFluent.args):
            return None
        dv = rFluent.dist(details)
        b = {}
        if isAnyVar(v):
            b[v] = dv.mode()
        if isAnyVar(p):
            b[p] = dv.prob(dv.mode())     
        return b

    def applyBindings(self, bindings):
        if self.isGround():
            return self.copy()
        return Bd([self.args[0].applyBindings(bindings)] + \
                  [lookup(a, bindings) for a in self.args[1:]],
                  lookup(self.value, bindings))

    def copy(self):
        return Bd(customCopy(self.args), customCopy(self.value))

    # True if the rFluent has value v with probability greater than p
    def test(self, b):
        (rFluent, v, p) = self.args
        assert isVar(p) or (0 <= p <= 1) or p == None

        if hasattr(rFluent, 'bTest'):
            return rFluent.bTest(b, v, p)

        if v == '*':
            (mpe, mp) = rFluent.dist(b).maxProbElt()
            return mp >= p
        else:
            return rFluent.dist(b).prob(v) >= p

    def feasible(self, details):
        (rFluent, v, p) = self.args
        return rFluent.feasible(details, v, p)


    def heuristicVal(self, details):
        (rFluent, v, p) = self.args
        return rFluent.heuristicVal(details, v, p)

    def beliefMode(self, details = None):
        if hasattr(rFluent, 'bTest'):
            return rFluent.bTest(details, v, 0.5)
        
        # Mode of the rfluent's value distribution
        return self.args[1].dist().mode()

    def glb(self, other, details = None):
        (sf, sv, sp) = self.args
        if other.predicate == 'Bd':
            (of, ov, op) = other.args
        elif other.predicate == 'B':
            (of, ov, ovar, odelta, op) = other.args
        else:
            return {self, other}, {}

        fglb, b = sf.glb(of, details)
        if fglb == False:
            # contradiction
            return False, {}
        if isinstance(fglb, set):
            # no glb at rfluent level
            return {self, other}, {}
        needNewFluent = (fglb != sf)

        if isVar(sp):
            b[sp] = op
        elif isVar(op):
            b[op] = sp

        if isGround((sp, op)):
            newP = max(op, sp)
            needNewFluent = needNewFluent or (newP != sp)
        else:
            newP = sp

        if needNewFluent:
            assert fglb.predicate != 'Pose'   # LPK chasing stupid bug
            return Bd([fglb, fglb.value, newP], True), b
        else:
            # This is pure entailment
            return self, b

class B(BFluent):
    predicate = 'B'
    # Fluent for the planner to use.  Derived from an RFluent with a
    # continuous distribution.
    # Args are: phi, value, var, delta, p
    # Semantics: there is a mode in the distribution of rf whose
    # - mean is within delta of v
    # - variance is less than var
    # - mixture weight is greater than p

    # If value is '*', then we are just checking variance and p

    def __init__(self, (f, val, var, delta, p), value = None):
        # Equal lengths
        assert isVar(val) or isVar(var) or isVar(delta) or \
               val == None or var == None or delta == None or val == '*' or \
                 len(val) == len(var) == len(delta)
        # No negative deltas!
        assert isVar(delta) or delta == None or delta == '*' or \
                     all([float(dv) >= 0.0 for dv in delta])

        # Make sure numeric args are floats.  Allow None.
        def g(v):
            return None if v == None else floatify(v)
        # super(B, self).__init__([f, g(val), g(var), g(delta), g(p)],
        #                         value = value)
        self.args = [f, g(val), g(var), g(delta), g(p)]
        self.value = value
        self.predicate = 'B'
        self.update()

    def removeProbs(self):
        return B([self.args[0], self.args[1], None, None, None], self.value)

    def applyBindings(self, bindings):
        if self.isGround(): return self
        return B([self.args[0].applyBindings(bindings)] + \
                  [lookup(a, bindings) for a in self.args[1:]],
                  lookup(self.value, bindings))

    # Print stdev!!
    def argString(self, eq = True):
        # Args: fluent, mean, var, delta, p
        (fluent, mean, var, delta, p) = self.args
        stdev = tuple([np.sqrt(v) for v in var]) \
                         if (not var == None and not isVar(var)) else var
        return '['+ fluent.prettyString(eq, includeValue = False) + ',' + \
         ', '.join([prettyString(a, eq) for a in [mean, stdev, delta, p]]) + ']'

    def copy(self):
        return B(customCopy(self.args), customCopy(self.value))

    def heuristicVal(self, details):
        (rFluent, v, var, delta, p) = self.args
        return  rFluent.heuristicVal(details, v, var, delta, p)

    def test(self, b):
        (rFluent, v, var, delta, p) = self.args
        dv = rFluent.dist(b)
        (dc, dp) = dv.mlc()  # most likely mixture component

        if debug('testVerbose'):
            print 'B fluent test', rFluent
            print '    ModeProb', dp
            print '    Actual mode', dc
            print '    Target mean', v
            if v != '*':
                print '    Mean diff', abs(dc.mode() - np.array(v))
                print '    Delta', delta
                print '    Mean diff < delta', \
                 (abs(dc.mode() - np.array(v)) <= np.array(delta))
            print '    Target var', var
            print '    Sigma < var', (dc.sigma.diagonal() <= np.array(var))
            print '    Target p', p
            print '    Modep >= p', dp >= p

            print '    value with star', \
              (dc.sigma.diagonal() <= np.array(var)).all() \
                   and dp >= p
            raw_input('okay?')

        if v == '*':
            # Just checking variance and p
            return (dc.sigma.diagonal() <= np.array(var)).all() \
                   and dp >= p
        else:
             return (abs(dc.mode() - np.array(v)) <= np.array(delta)).all() \
                   and (dc.sigma.diagonal() <= np.array(var)).all() \
                   and dp >= p

    def getGrounding(self, details):
        (rFluent, v, var, delta, p) = self.args
        if not isGround(rFluent.args):
            return None
        dv = rFluent.dist(details)
        b = {}
        if isAnyVar(v):
            b[v] = dv.modeList()  # hack
        if isAnyVar(var):
            b[var] = dv.varTuple()
        if isAnyVar(delta):
            b[delta] = [0.0]*len(dv.modeTuple())
        if isAnyVar(p):
            b[p] = .999   # bogus
        return b

    def beliefMode(self, details = None):
        # Mode of the rfluent's value distribution
        return self.args[1].dist().mode()
    
    def glb(self, other, details = None):
        # Quick test
        if self == other:
            return self, {}

        if other.predicate != 'B':
            return other.glb(self, details)

        (sf, sval, svar, sdelta, sp) = self.args
        (of, oval, ovar, odelta, op) = other.args

        needNewFluent = False

        newSF = copy.copy(sf)
        newSF.value = sval
        newOF = copy.copy(of)
        newOF.value = oval
        fglb, b = newSF.glb(newOF, details)
        if fglb == False:
            # contradiction
            return False, {}
        if isinstance(fglb, set):
            # no glb at rfluent level
            return {self, other}, {}
        if isVar(sval):
            b[sval] = oval
        elif isVar(oval):
            b[oval] = sval

        if isVar(sdelta):
            b[sdelta] = odelta
        elif isVar(odelta):
            b[odelta] = sdelta

        # Find the glb.  Find interval of overlap, then pick mean, to get
        # new values of val and delta
        if sval == '*' and not isVar(oval):
            newVal, newDelta = oval, odelta
            needNewFluent = True
        elif oval == '*' and not isVar(sval):
            newVal, newDelta = sval, sdelta
            needNewFluent = True
        elif isGround((sval, oval, sdelta, odelta)):
            newVal, newDelta = getOverlap(sval, oval, sdelta, odelta)
            if newVal == False:
                return False, {}
            needNewFluent = (newVal != sval) or (newDelta != sdelta)
        else:
            newVal, newDelta = sval, sdelta
        
        # take the min of the variances
        if isVar(svar):
            b[svar] = ovar
        elif isVar(ovar):
            b[ovar] = svar

        if isGround((svar, ovar)):
            newVar = tuple(np.minimum(np.array(svar), np.array(ovar)).tolist())
            needNewFluent = needNewFluent or (newVar != svar)
        else:
            newVar = svar

        # take the max of the probs
        if isVar(sp):
            b[sp] = op
        elif isVar(op):
            b[op] = sp

        if isGround((sp, op)):
            newP = max(op, sp)
            needNewFluent = needNewFluent or (newP != sp)
        else:
            newP = sp

        if needNewFluent:
            return B([fglb, newVal, newVar, newDelta, newP], True), b
        else:
            # This is pure entailment
            return self.applyBindings(b), b

def getOverlap(vl1, vl2, dl1, dl2):
    def go(v1, v2, d1, d2):
        min1 = v1 - d1
        min2 = v2 - d2
        newMin = max(min1, min2)
        max1 = v1 + d1
        max2 = v2 + d2
        newMax = min(max1, max2)
        if newMin > newMax:
            return 'bad', 'bad'
        return (newMin + newMax)/2.0, (newMax - newMin) / 2.0

    result = [go(v1, v2, d1, d2) for (v1, v2, d1, d2) in \
              zip(vl1, vl2, dl1, dl2)]
    newMean = tuple([m1 for (m1, d1) in result])
    newDelta = tuple([d1 for (m1, d1) in result])
    if 'bad' in newMean:
        return False, False
    else:
        return newMean, newDelta

######################################################################
#
# Belief version of hAddBack
#
######################################################################

hCache2 = {}
hCacheID = {}

def hCacheReset():
    fbch.hCache.clear()
    hCache2.clear()
    hCacheID.clear()

fbch.hCacheReset = hCacheReset

def hCacheDel(f):
    if f in fbch.hCache:
        del fbch.hCache[f]
    for i in hCacheID.keys():
        hCacheID[i].discard(f)

fbch.hCacheDel = hCacheDel


# hCache maps each fluent to the set of actions needed to achieve it
# This can be saved until the next action is taken
# hCache[f] = (totalCost, operators)

# For belief, make it two-level:
#   Top level: replace probability arguments with None
#   next level: map probability args to (totalCost, operators)
# At query time, look for lowest prob greater than query prob and return
#   the associated result

# So, we'll have two caches:  hCache and hCache2

# Could do more to keep hCache2 minimal.

# One more cache for iterative deepening:  hCacheID
# maps level k into the set of fluents that have infinite cost at that depth

def removeProbs(f):
    if f.predicate in ('B', 'Bd'):
        return f.removeProbs()
    else:
        return f

def addToCachesSet(fs, cost, actSet, k):
    if cost == float('inf'):
        hCacheID[k].add(fs)
    else:
        hCache[fs] = (cost, actSet)
        fsStripped = frozenset([removeProbs(f) for f in fs])
        if not fsStripped in hCache2:
            hCache2[fsStripped] = {}
        hCache2[fsStripped][fs] = (cost, actSet)

def inCache(fs, k):
    return (fs in hCacheID[k]) or (fs in hCache)

def hCacheLookup(fs, k):
    if fs in hCache:
        return hCache[fs]
    elif fs in hCacheID[k]:
        return (float('inf'), set())
    else:
        return False

def hCacheEntailsSet(fs, k, debug = False):
    fsStripped = frozenset([removeProbs(f) for f in fs])
    if not fsStripped in hCache2:
        return False
    bestCost = float('inf')
    bestActSet = set({})
    for (ff, (cost, actSet)) in hCache2[fsStripped].items():
        if setEntails(ff, fs) != False:
            if cost < bestCost:
                (bestCost, bestActSet) = (cost, actSet)
                if cost == 0 and debug:
                     print 'hces cost 0'
                     for thing in ff:
                         print thing.prettyString()
                     glob.rememberMe = ff
                     raw_input('go?')
    if bestCost < float('inf'):
        return (bestCost, bestActSet)
    return False

# Is each of the fluents in f2 entailed by some fluent in f1?
def setEntails(fs1, fs2):
    return all([any([f1.entails(f2) != False for f1 in fs1]) for f2 in fs2])

maxHeuristicValue = float('inf')
    
# See if we can get some branch-and-bound action to work.  Advantage is
# finishing early;  risk is not filling up cache effectively

hAddBackEntail = True

def hAddBackBSet(start, goal, operators, ancestors, idk, maxK = 30,
                 staticEval = lambda f: float('inf'),
                 ddPartitionFn = lambda fs: [set([f]) for f in fs]):

    # Always address immutable functions first.
    def partitionFn(fs):
        immu = frozenset([f for f in fs if f.immutable])
        notImmu = [f for f in fs if not f.immutable]
        return [frozenset([immuF]) for immuF in immu if immuF.isGround()] + \
                     ddPartitionFn(notImmu)

    # fUp is a set of fluents
    # cache entries will be frozen sets of fluents
    # Return a set of actions.
    def aux(fUp, k, minSoFar):
        spaces = '\n'+' '*(idk - k)
        spaces = ' '*(idk - k)
        # See if it's in the main cache
        if inCache(fUp, idk):
            if debug('hAddBackV'): print 'c',
            return hCacheLookup(fUp, idk)

        g = State(fUp)
        if start.satisfies(g):
            if debug('hAddBackV'): print 's',
            addToCachesSet(fUp, 0, set(), idk)
            return hCacheLookup(fUp, idk)

        # See if it's in level 2 cache
        # This makes it inadmissible
        result = hCacheEntailsSet(fUp, idk) if hAddBackEntail else False
        if result != False:
            (c, a) = result
            if c < float('inf'):
                if c == 0:
                    assert start.satisfies(g)
                addToCachesSet(fUp, c, a, idk)
                if debug('hAddBackV'): print 'e',
            return result

        if k == 0:
            if debug('hAddBackV'): print 'l',
            if idk == maxK:
                # Last round of iterative deepening
                # Get a final value. At a leaf.  Use static eval
                v = staticEval(fUp)
                dummyO = Operator('dummy'+prettyString(v), [], {}, [])
                dummyO.instanceCost = v
                addToCachesSet(fUp, v, set([dummyO]), idk)
                if v == 0:
                    assert start.satisfies(g)
            else:
                addToCachesSet(fUp, float('inf'), set(), idk)
            return hCacheLookup(fUp, idk)

        elif all([not f.isGround() for f in fUp]):
            # None of these fluents are ground; assume they can be
            # made true by matching
            return 0, set()
        else:
            # If this group has only 1 fluent and a special way to
            # compute the value, then use that.
            if len(fUp) == 1:
                f = list(fUp)[0]
                hv =  f.heuristicVal(start.details)
                if hv != False:
                    # We actually do have a special value
                    (cost, ops) = hv
                    addToCachesSet(fUp, cost, ops, idk)
                    # if cost == 0:
                    #     assert start.satisfies(g)
                    if debug('hAddBackV') or True: print 'hv',
                    return cost,ops

            # Otherwise, do regression

            # If it's not cheaper than the upper bound, we'll quit
            totalCost = minSoFar
            store = False

            # OR loop over operators and ways of achieving them
            # Need to pass in ancestors to use the appropriate level of
            # abstraction in computing the heuristic (does this make sense?)
            ops = applicableOps(g, operators, start, ancestors,
                                monotonic = False)
            if len(ops) == 0:
                debugMsg('hAddBack', 'no applicable ops', g)
            for o in ops:
                if debug('primitiveHeuristicAlways'):
                    o.abstractionLevel = o.concreteAbstractionLevel
                pres = o.regress(g, start)
                if len(pres) == 1:
                    debugMsg('hAddBack', 'no preimage', g, o)
                for pre in pres[:-1]:
                    (preImage, newOpCost) = pre
                    newActSet = set([preImage.operator])
                    children = []
                    partialCost = preImage.operator.instanceCost
                    assert partialCost < float('inf')
                    # AND loop over preconditions
                    store = True
                    for ff in partitionFn(preImage.fluents):
                        if partialCost >= totalCost:
                            # This op can never be cheaper than the best we
                            # have found so far in this loop
                            if debug('hAddBackV'): print 'BB!',
                            store = False # This is not a good cost estimate
                            break

                        children.append(ff)
                        if debug('hAddBack'):
                            print spaces+'C Aux:', k-1,\
                                 prettyString(totalCost - partialCost),\
                                   [f.shortName() for f in ff]
                            for f in ff: print spaces+'--'+f.prettyString()
    
                                   
                        subCost, subActSet = aux(ff, k-1,
                                                 totalCost - partialCost)
                        if debug('hAddBack'):
                            print spaces+'R Aux:', k-1, prettyString(subCost),\
                                   [f.shortName() for f in ff]

                        # make this side effect
                        if subCost == float('inf'):
                            partialCost, newActSet = (float('inf'), set())
                        else:
                            newActSet = newActSet.union(subActSet)
                            partialCost = sum([op.instanceCost \
                                               for op in newActSet])

                    newTotalCost = partialCost

                    if debug('hAddBack'):
                        childCosts = [aux(c, k-1, float('inf'))[0] \
                                      for c in children]
                        print spaces+'goal', k, prettyString(minSoFar),\
                                    [f.shortName() for f in fUp]
                        print spaces+'op', o.name
                        print spaces+'Children:'
                        for (x,y) in zip(childCosts, children):
                            print spaces+'    ', x, [f.shortName() for f in y]

                    if store and newTotalCost < totalCost:
                        addToCachesSet(fUp, newTotalCost, newActSet, idk)
                        if newTotalCost == 0:
                            assert start.satisfies(g)
                        (totalCost, actSet) = (newTotalCost, newActSet)
                        if debug('hAddBackV'):
                            print '\n'+spaces+'H', prettyString(totalCost), \
                                  [f.shortName() for f in fUp]
                        if debug('hAddBack'):
                            print spaces+'stored value', k, idk, newTotalCost
                            print spaces+'actSet', [a.name for a in actSet]
                            raw_input('okay?')
                    elif not store:
                        if debug('hAddBack'):
                            print spaces+'BB prevented value'
                    else:
                        if debug('hAddBack'):
                            print spaces+'New Cost', newTotalCost, \
                              'not better than', totalCost

            assert totalCost >= 0
            if totalCost == float('inf'):
                print 'Infinite cost goal set'
                actSet = set()
                for f in fUp:
                    print '    ', f.shortName()
                debugMsg('hAddBackInfV',
                         'Warning: storing infinite value in hCache')
                addToCachesSet(fUp, totalCost, set(), idk)
                if totalCost == 0:
                    assert start.satisfies(State(fUp))

            thing = hCacheLookup(fUp, idk)
            # If it's not in the cache, we bailed out before computing a good
            # value.  Just return inf
            return thing if thing != False else (float('inf'), set())

    totalActSet = set()
    # AND loop over fluents
    #fbch.inHeuristic = True
    for ff in partitionFn(goal.fluents):
        (ic, actSet) = aux(ff, idk, float('inf'))
        if ic == float('inf'):
            return ic
        totalActSet = totalActSet.union(actSet)
    totalCost = sum([op.instanceCost for op in totalActSet])

    if totalCost < float('inf') and debug('hAddBack'):
        print '** Final **', prettyString(totalCost)
        for thing in goal.fluents: print '   ', thing.shortName()
        print '     acts'
        for op in totalActSet: print '        ', \
              prettyString(op.instanceCost), op.name, op.args[0]
        debugMsg('hAddBack', 'final')
    if totalCost < float('inf') and not debug('hAddBack'):
        def n(f):
            return f.args[0].predicate if f.predicate in ('B', 'Bd') \
                      else f.predicate

        print 'H', prettyString(totalCost)
    return totalCost

def hAddBackBSetID(start, goal, operators, ancestors, maxK = 30,
                   staticEval = lambda f: 500,
                   ddPartitionFn = lambda fs: [set([f]) for f in fs]):
    fbch.inHeuristic = True
    startDepth = maxK-1
    for k in range(startDepth, maxK):
        hCacheID[k] = set()
        vk = hAddBackBSet(start, goal, operators, ancestors, k,
                          ddPartitionFn = ddPartitionFn)
        if vk < float('inf'):
            break
    result = min(vk, maxHeuristicValue)
    if vk == float('inf') and debug('hAddBackInf'):
        print '**** Final heuristic value is infinite ****'
        print 'Searched to depth', maxK
        for thing in goal.fluents: print thing
        debugMsg('hAddBackInf', 'Bad if this is the root')
        return vk
    fbch.inHeuristic = False
    return result
