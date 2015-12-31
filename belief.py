import copy
import numpy as np

from traceFile import tr, trAlways, debug, debugMsg

import planGlobals as glob

import fbch
from fbch import Fluent, hCache, State, applicableOps, Operator,\
     AddPreConds

from miscUtil import isVar, isAnyVar, floatify, customCopy,\
    lookup, prettyString, isGround, SetWithEquality, timeString

import local


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
        return self.args[0].couldClobber(other.args[0], details)

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
        self.strStored = {True:None, False:None}

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

    def getGrounding(self, bstate):
        (rFluent, v, p) = self.args
        if not isGround(rFluent.args):
            return None
        dv = rFluent.dist(bstate.details)
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
        # (rFluent, v, p) = self.args
        # return rFluent.feasible(details, v, p)
        fs = frozenset((self,))
        if glob.inHeuristic:
            if not inCache(fs):
                if inNHCache(fs):  # okay to use real value for heuristic
                    return nhCacheLookup(fs)[0] < float('inf')
                (cost, ops) = self.heuristicVal(details)
                addToCachesSet(fs, cost, ops)
            return hCacheLookup(fs)[0] < float('inf')
        else:
            if not inNHCache(fs):
                (cost, ops) = self.heuristicVal(details)
                addToNHCache(fs, cost, ops)
            return nhCacheLookup(fs)[0] < float('inf')

    def feasiblePBS(self, pbs):
        (rFluent, v, p) = self.args
        return rFluent.feasiblePBS(pbs, v, p)

    def heuristicVal(self, details):
        (rFluent, v, p) = self.args
        return rFluent.heuristicVal(details, v, p)

    def beliefMode(self, details = None):
        (rFluent, v, p) = self.args
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

        tr('testVerbose',
           ('B fluent test', rFluent),
           ('    ModeProb', dp),
           ('    Actual mode', dc),
           ('    Target mean', v),
           ('    Mean diff', abs(dc.mode() - np.array(v)) if v!='*' else ''),
           ('    Delta', delta),
           ('    Mean diff < delta',
            (abs(dc.mode() - np.array(v)) <= np.array(delta)) \
                if v != '*' else ''),
           ('    Target var', var),
           ('    Sigma < var', (dc.sigma.diagonal() <= np.array(var))),
           ('    Target p', p),
           ('    Modep >= p', dp >= p),
           ('    value with star', \
              (dc.sigma.diagonal() <= np.array(var)).all() \
                   and dp >= p), ol = False)

        if v == '*':
            # Just checking variance and p
            return (dc.sigma.diagonal() <= np.array(var)).all() \
                   and dp >= p
        else:
             return (abs(dc.mode() - np.array(v)) <= np.array(delta)).all() \
                   and (dc.sigma.diagonal() <= np.array(var)).all() \
                   and dp >= p

    def getGrounding(self, bstate):
        (rFluent, v, var, delta, p) = self.args
        if not isGround(rFluent.args):
            return None
        dv = rFluent.dist(bstate.details)
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
        if str(self) == str(other) and self != other:
            trAlways('Strings match but not equal', self,
                     self.args, other.args, ol = True)

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
# Belief meta-operator
#
######################################################################

# Needs special treatment in regression to take the binding of
# 'NewCond' and add it to the preconditions

# assumes generator returns a list of lists (or sets) of fluents

class BMetaOperator(Operator):
    def __init__(self, name, fluentClass, args, generator,
                 argsToPrint = None):
        super(BMetaOperator, self).__init__(\
            name,
            args + ['PreCond', 'NewCond', 'PostCond', 'P'],
            {0 : {},
             1 : {Bd([fluentClass(args + ['PreCond']), True, 'P'], True)}},
            [({Bd([fluentClass(args + ['PostCond']),  True, 'P'], True)}, {})],
            functions = [
               generator(['NewCond'], args + ['P', 'PostCond'], True),
               AddPreConds(['PreCond'],['PostCond', 'NewCond'], True)],
            argsToPrint = range(len(args)) if argsToPrint == None else \
                           argsToPrint,
            ignorableArgs = range(len(args), len(args) + 4),
            ignorableArgsForHeuristic = range(len(args), len(args) + 4),
            conditionOnPreconds = True,
            metaOperator = True)

######################################################################
#
# Belief version of hAddBack
#
######################################################################

hCache2 = {}
hCacheInf = {}  # Maps fluent sets to the level at which they have
                # generated infinite values
nhCache = {}

def hCacheReset():
    fbch.hCache.clear()
    hCache2.clear()
    hCacheInf.clear()
    nhCache.clear()

fbch.hCacheReset = hCacheReset

def hCacheDel(f):
    if f in fbch.hCache:
        del fbch.hCache[f]
    for i in hCacheID.keys():
        hCacheInf.discard(f)

fbch.hCacheDel = hCacheDel

hAddBackEntail = True


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

# One more cache for infinities:  hCacheInf
# maps level k into the set of fluents that have infinite cost at that depth

def removeProbs(f):
    if f.predicate in ('B', 'Bd'):
        return f.removeProbs()
    else:
        return f

def addToNHCache(fs, cost, actSet):
    nhCache[fs] = (cost, actSet)

def addToCachesSet(fs, cost, actSet):
    hCache[fs] = (cost, actSet)
    if hAddBackEntail: 
        fsStripped = frozenset([removeProbs(f) for f in fs])
        if not fsStripped in hCache2:
            hCache2[fsStripped] = {}
        hCache2[fsStripped][fs] = (cost, actSet)

def inCache(fs):
    return fs in hCache
def inNHCache(fs):
    return fs in nhCache

def hCacheLookup(fs):
    return hCache[fs] if fs in hCache else False
def nhCacheLookup(fs):
    return nhCache[fs] if fs in nhCache else False

def hCacheEntailsSet(fs):
    fsStripped = frozenset([removeProbs(f) for f in fs])
    if not fsStripped in hCache2:
        return False
    bestCost = float('inf')
    bestActSet = set({})
    for (ff, (cost, actSet)) in hCache2[fsStripped].items():
        if setEntails(ff, fs) != False:
            if cost < bestCost:
                (bestCost, bestActSet) = (cost, actSet)
    if bestCost < float('inf'):
        return bestCost, bestActSet
    return False

# Is each of the fluents in f2 entailed by some fluent in f1?
def setEntails(fs1, fs2):
    return all([any([f1.entails(f2) != False for f1 in fs1]) for f2 in fs2])

maxHeuristicValue = float('inf')
    
# See if we can get some branch-and-bound action to work.  Advantage is
# finishing early;  risk is not filling up cache effectively

cacheHits = 0
easy = 0
regress = 0

# Set of actions;  equality is based on ignoring values of many arguments
class ActSet(SetWithEquality):
    @staticmethod
    def testF(a, b):
        return a.ignoreIgnorableForH() == b.ignoreIgnorableForH()
    
    def __init__(self, elts = None):
        self.elts = [] if elts is None else elts
        self.test = [self.testF] # horrible workaround should be a static method

def BBhAddBackBSet(start, goal, operators, ancestors, maxK = 30,
                 staticEval = lambda f: float('inf'),
                 ddPartitionFn = lambda fs: [frozenset([f]) for f in fs],
                 feasibleOnly = True,
                 primitiveHeuristicAlways = debug('primitiveHeuristicAlways')):

    global cacheHits, easy, regress
    cacheHits = 0
    easy = 0
    regress = 0

    # Always address immutable functions first.
    def partitionFn(fs):
        immu = frozenset([f for f in fs if f.immutable])
        notImmu = [f for f in fs if not f.immutable]
        return [frozenset([immuF]) for immuF in immu if immuF.isGround()] + \
                     ddPartitionFn(notImmu)

    # fUp is a set of fluents
    # cache entries will be frozen sets of fluents
    # Return a set of actions.
    # ha is heuristic ancestors
    def aux(g, k, bound, ha, fp = None):
        global cacheHits, easy

        fUp = frozenset(g.fluents)
        # If fUp is in ancestors, return infinite score.  Stops stupid
        # endless backchaining
        if fUp in ha:
            if fp is not None: writeHNode(fp, g, 'inf', loopStyle)
            return float('inf'), None
        
        # See if it's in the main cache
        if inCache(fUp):
            cval = hCacheLookup(fUp)
            cacheHits += 1
            if fp is not None: writeHNode(fp, g, cval[0], cacheStyle)
            return cval

        # Try to find a quick answer
        cost, ops = None, None
        # If this group has only 1 fluent and a special way to
        # compute the value, then use that.
        if len(fUp) == 1:
            f = list(fUp)[0]
            hv =  f.heuristicVal(start.details)
            if hv:
                # We actually do have a special value
                (cost, ops) = hv
                if fp is not None: writeHNode(fp, g, cost, specialStyle)
        # See if it's true in start state
        if cost is None and start.satisfies(g):
            cost, ops = 0, ActSet()
            if fp is not None: writeHNode(fp, g, cost, initStyle)
        # See if it's in level 2 cache.   This makes it inadmissible
        if cost == None and hAddBackEntail:
            result = hCacheEntailsSet(fUp)
            if result:
                if fp is not None: writeHNode(fp, g, cost, l2CacheStyle)
                # print 'L2'
                (cost, ops) = result
        # At a leaf.  Use static eval.
        if cost == None and k == 0:
            cost = staticEval(fUp)
            dummyO = Operator('dummy'+prettyString(cost), [], {}, [])
            dummyO.instanceCost = cost
            ops = ActSet([dummyO])
            if fp is not None: writeHNode(fp, g, cost, leafStyle)
        if cost == None and all([not f.isGround() for f in fUp]):
            # None of these fluents are ground; assume they can be
            # made true by matching
            cost, ops = 0, ActSet()
            if fp is not None: writeHNode(fp, g, cost, nonGroundStyle)

        if cost != None:
            # We found a cheap answer.  Put in cache and return
            if cost != float('inf'):
                addToCachesSet(fUp, cost, ops)
            easy += 1
            return cost, ops

        # Otherwise, do regression
        newHA = ha.union({fUp})

        # If we find it has to be more expensive than this upper bound, quit
        # Set to true if we compute a reliable value for fUp that should be
        # put into the cache
        bestActSet = None

        # OR loop over operators and ways of achieving them
        # Need to pass in ancestors to use the appropriate level of
        # abstraction in computing the heuristic 
        ops = applicableOps(g, operators, start, ancestors,
                                monotonic = False)
        pres = set()
        for o in ops:
            if primitiveHeuristicAlways:
                o.abstractionLevel = o.concreteAbstractionLevel
            # TODO : LPK : domain-dependent hack
            n = glob.numOpInstances if o.name in ['Push', 'Place'] else 1
            opres = o.regress(g, start, numResults = n)
            if len(opres) > 0:
                pres = pres.union(set(opres[:-1]))
        if len(pres) == 0: debugMsg('hAddBack', 'no applicable ops', g)

        minSoFar = bound
        for pre in pres:
            preImage, newOpCost = pre
            newActSet = ActSet([preImage.operator])
            partialCost = preImage.operator.instanceCost
            # AND loop over preconditions.  See if we have a good value
            # for this operator
            bb = False
            for fff in partitionFn(preImage.fluents):
                if partialCost >= minSoFar:
                    # This op can never be cheaper than the best we
                    # have found so far in this loop
                    bb = True
                    break
                fffs = State(fff)
                if fp is not None: writeSearchArc(fp, preImage, fffs)
                subCost, subActSet = \
                  aux(fffs, k-1, minSoFar - partialCost, newHA, fp)
                if subCost == float('inf'):
                    partialCost, newActSet = (float('inf'), ActSet())
                else:
                    newActSet = newActSet.union(subActSet)
                    partialCost = sum([opr.instanceCost \
                                           for opr in newActSet.elts])
                        
            if fp is not None:
                style = bbStyle if bb else \
                         (domStyle if partialCost >= minSoFar else andStyle)
                cost = ('?' if bb else partialCost, minSoFar)
                writeHNode(fp, preImage, cost, style)
                writeSearchArc(fp, g, preImage, preImage.operator)

            # At this point, partialCost is the total cost for this operator
            if partialCost < minSoFar:
                # This is a better way of achieving fUp; a reliable value
                minSoFar = partialCost
                bestActSet = newActSet
                  
                if feasibleOnly:
                    # Good enough for us!
                    addToCachesSet(fUp, minSoFar, bestActSet)
                    return (minSoFar, bestActSet)

        # Done with all the ways of achieving fUp.  Store the result
        if bestActSet is not None:
            addToCachesSet(fUp, minSoFar, bestActSet)

        # Return the value in the cache
        result = hCacheLookup(fUp)

        if fp is not None:
            cost = ('?' if result == False else result[0], bound) 
            writeHNode(fp, g, cost, orStyle)
        
        # If it's not in the cache, we bailed out before computing a good
        # value.  Just return inf
        return result if result != False else (float('inf'), ActSet())

    def topLevel(maxK, writeFile = False):
        fp = openHFile() if writeFile else None
        try:
            totalActSet = ActSet()
            infCost = False
            totalCost = '???'
            for ff in partitionFn(goal.fluents):
                sff = State(ff)
                (ic, actSet) = aux(sff, maxK, float('inf'), set(), fp)
                if ic == float('inf'):
                    infCost = True
                    break
                else:
                    totalActSet = totalActSet.union(actSet)
                    if fp is not None:
                        writeHNode(fp, sff, ic, orStyle)
                        writeSearchArc(fp, goal, sff)
                
            totalCost = sum([op.instanceCost for op in totalActSet.elts]) \
                        if not infCost else float('inf')
        
        finally:
            if fp is not None:
                writeHNode(fp, goal, totalCost, andStyle)
                closeHFile(fp)
        return totalCost, totalActSet

    ### Procedure body ####
    writeFile = debug('alwaysWriteHFile')
    glob.inHeuristic = True
    (totalCost, totalActSet) = topLevel(maxK, writeFile = writeFile)
    if totalCost == float('inf') and debug('infiniteHeuristic'):
        print '** Found infinite heuristic value, recomputing **'
        # Could flush cache
        (h2, as2) = topLevel(maxK, writeFile = True)
        print 'New heuristic value', h2
        # totalActSet = ActSet()
        totalCost = h2
        if totalCost == float('inf'):
            raw_input('Heuristic value is still infinite - continue?')
        totalActSet = as2
    glob.inHeuristic = False
    return totalCost, totalActSet.elts

def writeHNode(f, s, c, styleStr):
    f.write('    "'+s.uniqueStr()+\
            styleStr +\
             prettyString(c, True) + '\\n' + s.prettyString(False)+'"];\n')

def writeSearchArc(f, s1, s2, op = None):
    opStr = op.prettyString(False) if op is not None else ''
    f.write('    "'+s1.uniqueStr()+'" -> "'+s2.uniqueStr()+'"[label="'+\
            opStr+'"];\n')

def openHFile():
    fp = open(local.outDir+'Heur'+'_'+timeString()+'.dot', 'w')
    fp.write('digraph G {\n')
    fp.write('    ordering=out;\n')
    fp.write('    node [fontname=HelveticaBold];\n')
    return fp

def closeHFile(fp):
    fp.write('}\n')
    fp.close()
    print 'Wrote heuristic file'
            
# red
loopStyle = \
  '" [shape=box, style=filled, colorscheme=pastel16, color=1, label="Loop cost='
# blue
initStyle = \
  '" [shape=box, style=filled, colorscheme=pastel16, color=2, label="Init cost='
# yellow
cacheStyle = \
 '" [shape=box, style=filled, colorscheme=pastel16, color=6, label="Cache cost='
l2CacheStyle = \
 '" [shape=box, style=filled, colorscheme=pastel16, color=6, label="L2 Cache cost='
# purple
andStyle = \
  '" [shape=box, style=filled, colorscheme=pastel16, color=4, label="cost='
# green
orStyle = \
  '" [shape=box, style=filled, colorscheme=pastel16, color=3, label="cost='
# orange
specialStyle = \
 '" [shape=box, style=filled, colorscheme=pastel16, color=5, label="Special cost='
# brown
leafStyle = \
  '" [shape=box, style=filled, colorscheme=pastel19, color=7, label="Leaf cost='
# pink
nonGroundStyle = \
  '" [shape=box, style=filled, colorscheme=pastel19, color=8, label="NonGround cost='
# gray
domStyle = \
  '" [shape=box, style=filled, colorscheme=pastel19, color=9, label="Dominated cost='
# clear
bbStyle = \
  '" [shape=box, label="Pruned cost='
