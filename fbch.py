import copy
import itertools
import os

import planGlobals as glob
from planGlobals import debugMsg, debug

import ucSearchPQ as ucSearch
reload(ucSearch)

import miscUtil
reload(miscUtil)
from miscUtil import applyBindings, timeString, prettyString,\
     isVar, squash, Stack, tuplify, gensym, matchLists, makeVar,\
     matchTerms, combineBindings, isAnyVar, \
     getCVBindings, getRegularBindings, extractVars, isGround, undiag,\
     makeDiag, isStruct, customCopy, lookup, squashSets, powerset, \
     mergeDicts, squashOne

import local

inHeuristic = False

flatPlan = False   # Set to False if you want hierarchy
maxHDelta = None      # check that heuristic does not change radically
hmax = float('inf')  # prune a node if the h value is higher than this

# Make the planner slightly greedy wrt heuristic
plannerGreedy = 0.5  # UC = 0, A* = 0.5, BestFirst = 1.0

# Stop monotonic planning if cost is bigger than this.
maxMonoCost = 300

######################################################################
## Planner support classes
######################################################################

class Primitive:
    numPrims = 0
    def __init__(self):
        self.primNum = Primitive.numPrims
        Primitive.numPrims = Primitive.numPrims + 1
    def __str__(self):
        return self.prettyString(True)+str(self.primNum)

class State:
    numStates = 0
    def __init__(self, fluents, details = None):
        self.details = details
        self.num = State.numStates; State.numStates += 1
        self.depth = 0
        self.suspendedOperator = None
        self.bindingsAlreadyTried = [] 
        self.rebind = False
        self.operator = None
        self.bindings = {}  # just used for some bookkeeping in planning
        self.valueCache = {} # mapping from fluents to values
        self.relaxedValueCache = {} # mapping from fluents to values
        # Add fluents serially to get rid of redundancy
        self.fluents = set()
        for f in fluents:
            self.add(f, details)
        internalTest(self.fluents, details)

    def flushCache(self):
        self.valueCache = {}
        self.relaxedValueCache = {}

    # Quick heuristic
    def easyH(self, start, defaultFluentCost = 5):
        # Find how many false fluents there are.  Better to partition
        # into groups.  Charge them their heuristicVal if it's
        # defined, else the default
        total = 0
        actSet = set()
        fluentGroups = start.details.partitionFn(self.fluents)
        for fg in fluentGroups:
            for f in fg:
                maxCostInGroup = 0
                groupActSet = set()
                
                if not f.isGround():
                    # See if it can be satisfied
                    fSat = start.satisfies(State([f]))
                else:
                    fSat = (start.fluentValue(f) == f.value)
                
                if not fSat:
                    # Either False (if not defined), else a cost and a
                    # list of actions
                    hv = f.heuristicVal(start)
                    if hv == False:
                        if defaultFluentCost > maxCostInGroup:
                            maxCostInGroup = defaultFluentCost
                            groupActSet = set()
                    elif hv[0] != float('inf'):
                        totalCost = sum([a.instanceCost for a in hv[1]])
                        if totalCost > maxCostInGroup:
                            maxCostInGroup = totalCost
                            groupActSet = hv[1]
                    else:
                        # We got trouble, right here in River City
                        maxCostInGroup = float('inf')
            # Per fluent group
            if actSet == set():
                total += maxCostInGroup
            else:
                actSet = actSet.union(groupActSet)

        result = total + sum([o.instanceCost for o in actSet])
        if result == 0:
            # one more sanity check
            if not start.satisfies(self):
                numNonGround = len([thing for thing in self.fluents \
                                    if not thing.isGround()])
                # There is no other reason why this value should be 0
                if numNonGround == 0:
                    raw_input('Heuristic 0 but not sure why')
                result = numNonGround * defaultFluentCost
        return result

    def updateStateEstimate(self, op, obs=None):
        # Assumes that we're not progressing fluents, so just update
        # the details, which keeps distribution on objects, etc.
        op.f(self.details, op.args, obs) # perform the action update
        if hasattr(self.details, 'propagate'):
            self.details.propagate()
        self.details.draw() # show new belief
        self.flushCache() # get rid of memorized fluent values
        hCacheReset() # flush heuristic values

    def addSet(self, newFs, moreDetails = None, noBindings = False):
        details = self.details or moreDetails
        b = {}
        for newF in newFs:
            newB = self.add(newF, details, noBindings = noBindings)
            b.update(newB)
        return b

    def add(self, newF, moreDetails = None, noBindings = False):
        details = self.details or moreDetails
        (newFluentSet,newBindings) = addFluentToSet(self.fluents, newF,
                                                    details, noBindings)
        internalTest(newFluentSet, details) 
        self.fluents = newFluentSet
        return newBindings
        
    def internalConsistencyTest(self, details):
        for f1 in self.fluents:
            for f2 in self.fluents:
                if f1.glb(f2, details)[0] == False:
                    raise Exception, 'State internally inconsistent'

    def remove(self, f):
        self.fluents.remove(f)
    
    def copy(self):
        # Copies the details as well
        s = State(copy.copy(self.fluents),
                  self.details.copy() if self.details else self.details)
        s.depth = self.depth
        s.suspendedOperator = self.suspendedOperator
        s.bindingsAlreadyTried = copy.copy(self.bindingsAlreadyTried)
        s.rebind = self.rebind
        s.bindings = copy.copy(self.bindings)
        return s

    def applyBindings(self, bindings):
        s = State([f.applyBindings(bindings) for f in self.fluents],
                  self.details)
        if self.operator == None:
            s.operator = None
        else:
            s.operator = self.operator.applyBindings(bindings)
        s.depth = self.depth
        s.bindings = self.bindings 
        return s

    def isConsistent(self, fluentList, details = None):
        return not any([self.contradicts(f, details) for f in fluentList])

    # Return True if any fluent in the list binds with some fluent in the state
    def couldBeClobbered(self, fluentList, details = None):
        return any([f1.couldClobber(f2, details) for f1 in fluentList \
                    for f2 in self.fluents])
    
    # Return the value of the fluent in this state.
    def fluentValue(self, fluent, recompute = False):
        cache = None if recompute else \
                 (self.relaxedValueCache if inHeuristic else self.valueCache)
        return self.fluentValueWithCache(fluent, cache)

    # Terribly terribly inefficient.
    def fluentValueWithCache(self, fluent, cache):
        if cache != None and (fluent in cache):
            return cache[fluent]

        # Only useful when there are some aspects of the initial state
        # described only in fluents (not details).  Could delete
        for f in self.fluents:
            # See if it matches a fluent in the state
            if fluent.predicate == f.predicate and \
                         fluent.args == f.args:

                if self.details != None and debug('fluentCache'):
                    fval = fluent.valueInDetails(self.details)
                    assert fval == f.getValue(), 'fluent cache inconsistent'
                return f.getValue()

        # Now, look in the details
        if self.details != None:
            fval = fluent.valueInDetails(self.details)
            if cache != None: cache[fluent] = fval
            return fval

        raise Exception, 'No details or matching fluent'

    def satisfies(self, goal, ed = None):
        # just do simple entailment if we don't have details
        if not self.details:
            # ed is extra details we might sometimes have, just for
            # looking up shapes, etc.
            return all([self.entails(bf, ed) != False for bf in goal.fluents])
        
        goalFluents = goal.fluents
        extraBindings = {}
        if not all([bf.isGround() for bf in goalFluents]):
            b = getGrounding(goalFluents, self.details)
            if b == None:
                debugMsg('satisfies', 'failed to find satisfiable grounding',
                         goalFluents)
                return False
            else:
                debugMsg('satisfies', 'update bindings to satisfy in b0',
                         b)
                extraBindings.update(b)
                oldGoalFluents = goalFluents
                goalFluents = [f.applyBindings(b) for f in goalFluents]

        # Fail if we have an inconsistency.
        # Kind of lame because:  it doesn't try to find different
        # bindings if this one is inconsistent; also because it
        # doesn't test for inconsistency within the set of bound goalFluents.
        if not goal.isConsistent(goalFluents):
            debugMsg('satisfies', 'found grounding but it is inconsistent')
            print 'satisfies found grounding but it is inconsistent'
            return False
                
        failed = False
        for bf in goalFluents:
            assert(bf.isGround())
            v = bf.getValue()
            fv = self.fluentValue(bf)
            if fv != v:
                debugMsg('satisfies', 'failed', bf, v, fv)
                failed = True
        if failed:
            return False
        debugMsg('satisfies', 'satisfied', goalFluents,
                 ('extra bindings', extraBindings))
        return True

    def entails(self, otherFluent, details = None):
        for f in self.fluents:
            b = f.entails(otherFluent, details)
            if b != False:
                return b
        return False

    def contradicts(self, otherFluent, details = None):
        return any([f.contradicts(otherFluent, details) \
                   for f in self.fluents])

    def __str__(self):
        fluentString = '\\n'.join(sorted([str(y) for y in \
                        self.fluents]))

        # Try using prettyString so that real values will be truncated
        fluentString = self.prettyString(True)
                        
        # Rebind nodes look like duplicates, so be sure they're never pruned
        restartString = str(self.suspendedOperator) + str(self.num) if \
                        self.rebind else ''
        return fluentString+restartString

    def prettyString(self, eq = True, start = None, hFun = None):
        # Nicely formatted list of fluents with returns
        fluentString = '\\n'.join(sorted([y.prettyString(eq, start, hFun) \
                                          for y in self.fluents]))
        if self.rebind:
            end = '\\n ** try local rebinding'
        else:
            end = ''
        if self.rebind:
            return self.suspendedOperator.name + end
        return fluentString + '\\n' + end 

    # Use when we need a unique identifier, per state, to make the dot files
    def uniqueStr(self):
        return str(self.num)

    def goalName(self):
        # Only defined for states that have been used as goals in the planner
        return str(self) + str(self.planNum)

    __repr__ = __str__
    signature = __str__
    def __eq__(self, other):
        
        return other != None and self.signature() == other.signature()
    def __ne__(self, other):
        return other == None or self.signature() != other.signature()
    def __hash__(self):
        return self.signature().__hash__()

# Test a set of fluents for redudancy and consistency
def internalTest(fluents, details):
    if not debug('extraTests'): return True
    for f1 in fluents:
        for f2 in fluents:
            if f1 == f2: continue
                
            glb1 = f1.glb(f2, details)[0]
            glb2 = f2.glb(f1, details)[0]
            assert glb1 == glb2
            if glb1 == False:
                print f1, f2
                raise Exception, 'State internally inconsistent'
            elif type(glb1) != set:
                print f1, f2
                print 'glb', glb1
                raise Exception, 'State internally redundant'
    if debug('stateConsistency'):
        print 'Passed test'
        for f in self.fluents:
            print '    ', f
        raw_input('okay?')
    

# Should be consistent
def addFluentToSet(fluentSet, newF, details = None, noBindings = False):
    if fluentSet == set():
        return set([newF]), {}
    
    allB = {}
    removal = set()
    additions = set()
    addNew = True
    for oldF in fluentSet:
        #print 'oldF', oldF
        boundF = oldF.applyBindings(allB)
        glb, b = boundF.glb(newF, details)
        if glb == False:
            # Contradiction
            raw_input('adding fluent causes contradiction')
            return False
        elif b != {} and noBindings:
            # just add it
            b = {}
            pass
        elif isinstance(glb, set):
            # Just add the new guy
            #print 'no interesting glb'
            pass
        elif glb == oldF:
            # New guy is subsumed by oldF
            # print 'oldF subsumes new'
            addNew = False
        else:
            # in either case, remove oldF and add the glb
            # print 'adding glb to fluent set'
            # print 'glb', glb
            additions, addB = addFluentToSet(additions, glb,
                                                details, noBindings)
            allB.update(addB)
            addNew = False
            removal.add(oldF)
        allB.update(b)
    if addNew:
        (additions, addB) = addFluentToSet(additions, newF, details, noBindings)
        allB.update(addB)
    # Remove fluents that were entailed; add new ones
    fluentSet = (fluentSet - removal) | additions
    # Apply bindings
    fluentSet = set([f.applyBindings(allB) for f in fluentSet])
    return fluentSet, allB

# Return a list of fluent, bindings pairs
def getMatchingFluents(fluentList, pattern):
    result = [(f, pattern.matchAll(f)) for f in fluentList]
    return [(f, b) for (f, b) in result if b != None]

class Fluent(object):
    implicit = False
    immutable = False
    conditional = False
    fglb = None # define this method for a fluent specific glb

    def __init__(self, args, value = None, predicate = None):
        self.args = args
        self.value = value
        if predicate != None: self.predicate = predicate
        self.update()

    def update(self):
        self.isGroundStored = self.getIsGround()
        self.isPartiallyBoundStored = self.getIsPartiallyBound()
        self.strStored = self.getStr()

    def isImplicit(self):
        return self.implicit
    def isConditional(self):
        return self.conditional
    def heuristicVal(self, details, *args):
        return False

    def addConditions(self, newConds, details = None):
        assert self.isConditional()
        cond = self.args[-1]
        self.args[-1] = simplifyCond(cond, newConds, details)
        self.update()

    def shortName(self):
        return self.predicate

    def copy(self):
        newFluent = copy.copy(self)
        newFluent.args = customCopy(self.args)
        return newFluent

    def getIsGround(self):
        return self.argsGround() and not isVar(self.value)

    # If some args are bound and some are not
    def getIsPartiallyBound(self):
        argB = [isVar(a) for a in self.args]
        return (True in argB) and (False in argB)

    def getVars(self):
        valVars = [self.value] if isVar(self.value) else []
        if self.isConditional():
            condVars = set([self.args[-1]]) if isVar(self.args[-1])\
                        else squashSets([c.getVars() for c in self.args[-1]])
            return set([a for a in self.args[:-1] if isVar(a)]+valVars).\
                      union(condVars)
        else:
            return set([a for a in self.args if isVar(a)] + valVars)

    def argsGround(self):
        if self.isConditional():
            return not isVar(self.args[-1]) and \
                   all([not isVar(a) for a in self.args[:-1]]) and \
                   all([c.isGround() for c in self.args[-1]])
        else:
            return all([not isVar(a) for a in self.args])
                   
    def isGround(self):
        return self.isGroundStored

    def isPartiallyBound(self):
        return self.isPartiallyBoundStored

    # For a fluent that is ground except for the value, get the value
    def getGrounding(self, details):
        if not isGround(self.args):
            return None
        v = self.valueInDetails(details)
        b = {}
        if isVar(self.value):
            b[self.value] = v
        return b

    def getValue(self):
        return self.value

    def __str__(self):
        if debug('extraTests'):
            assert self.strStored == self.getStr()
        return self.strStored

    def getStr(self):
        # return self.predicate + prettyString(self.args) + \
        #   ' = ' + prettyString(self.value)
        return self.predicate + self.argString(True) +\
          ' = ' + prettyString(self.value)

    def argString(self, eq):
        return '['+ ', '.join([prettyString(a, eq) for a in self.args]) + ']'

    # Needs to be ground if we are going to compute the *
    def prettyString(self, eq = True, state = None, heuristic = None,
                     includeValue = True):
        isBold = (state and self.isGround() and \
                  state.fluentValue(self) != self.getValue())
        if isBold:
            # Used to compute heuristic for individual fluents and show the
            # values here, but not good now that some fluents only make sense
            # in groups
            if False: 
                h = heuristic(State([self]))
                bold = 'H['+prettyString(h, True)+'] '
            else:
                bold = '* '
        else:
            bold = ''

        pArgs = self.argString(eq)
        pValue = (' = ' + prettyString(self.value, eq)) if includeValue \
                                 else ''
        return bold + self.predicate + pArgs + pValue
    
    __repr__ = __str__
    def __hash__(self):
        return str(self).__hash__()
    def __eq__(self, other):
        return self.__hash__() == other.__hash__()
    def __ne__(self, other):
        return self.__hash__() != other.__hash__()

    def getPredicate(self):
        return self.predicate

    # Predicate, values and arguments match
    def matchAll(self, other, bindings = {}):
        return matchTerms(self.getValue(), other.getValue(),
                          self.matchArgs(other, bindings))

    # Predicate and arguments match
    def matchArgs(self, other, bindings = {}):
        if self.predicate == other.predicate:
            return matchLists(self.args, other.args, bindings)
        else:
            return None

    def valueInDetails(self, details):
        return self.test(details)

    def applyBindings(self, bindings):
        if self.isGround(): return self
        # Have to copy to get the right subclass, methods, etc.
        newF = copy.copy(self)
        if self.isConditional():
            # Dig one level deeper into last arg
            newF.args = [lookup(a, bindings) for a in self.args[:-1]] + \
                     ([lookup(self.args[-1], bindings)] \
                      if isVar(self.args[-1]) else \
                      [[c.applyBindings(bindings) for c in self.args[-1]]])
        else:
            newF.args = [lookup(a, bindings) for a in self.args]
        newF.value = lookup(self.value, bindings)
        newF.update()
        return newF

    # Rules are universally quantified
    # Fluents (states) are existentially quantified

    # Three cases:
    # - self and other can be locally unified:  return (newF, bindings)
    # - self and other are in contradiction: return (False, {})
    # - otherwise: return ({self, other}, {})
    
    def glb(self, other, details = None):
        b = copy.copy(self.matches(other))
        bnv = copy.copy(self.matches(other, noValue = True))
        b2 = copy.copy(other.matches(self))
        bnv2 = copy.copy(other.matches(self, noValue = True))
        if b != None:
            result = self.applyBindings(b), b
        elif b2 != None:
            result = other.applyBindings(b2), b2
        elif bnv != None or bnv2 != None:
            # Fluents match but values disagree
            result = (False, {})
        elif self.fglb != None:
            # See if we have a domain-specific method
            result = self.fglb(other, details)
        elif other.fglb != None:
            result = other.fglb(self, details)
        else:
            # Assume they can get along
            result = {self, other}, {}
        debugMsg('glb', self, other, result)
        return result

    def entails(self, other, details = None):
        glb, b = self.glb(other, details)
        if glb == self:
            return b
        else:
            return False

    def couldClobber(self, other, details = None):
        b = self.entails(other, details)
        return b != False and b != {}

    def contradicts(self, other, details = None):
        glb, b = self.glb(other, details)
        return glb == False

    def matches(self, other, b = {}, noValue = False):
        if self.predicate != other.predicate or \
           len(self.args) != len(other.args):
            return None
        elif self == other:
            return b
        else:
            a1 = self.args + ([self.value] if not noValue else [])
            a2 = other.args + ([other.value] if not noValue else [])
            result = matchLists(a1, a2, b)
            return result

# Preconditions are a dictionary, mapping abstraction level to a set
# of fluents.  Level 0 is most abstract.  Preconditions do not need to
# be repeated...automatically inherited at lower levels.

# Each result is a pair: fluent and a dictionary of specific
# preconditions (not already in the general precondition list

class Operator(object):
    num = 0
    # Lists of fluents
    def __init__(self, name, args, 
                 preconditions, results, functions = None,
                 f = None, 
                 cost = lambda al, args, details: 1,
                 prim = None,
                 sideEffects = None,
                 ignorableArgs = [],
                 argsToPrint = None,
                 specialRegress = None):
        self.name = name # string
        self.args = args # list of vars or constants
        self.preconditions = preconditions
        self.functions = functions if functions != None else []

        assert type(results) == list
        if len(results) > 0:
            assert type(results[0]) == tuple
            assert type(results[0][0]) == set
            assert type(results[0][1]) == dict
            if results[0][0] != set():
                # Should be a set of fluents
                assert type(list(results[0][0])[0]) != list

        self.results = results
        self.f = f  # function from args and details to details
        self.cost = cost # function from abs level and args to number
        self.concreteAbstractionLevel = max(preconditions.keys()) if \
                                        preconditions else 0
        self.abstractionLevel = \
                         self.concreteAbstractionLevel if flatPlan else 0
        self.sideEffects = sideEffects if sideEffects != None else {}
        self.prim = prim
        self.verifyArgs()
        # Unique ID for drawing graphs
        self.incrementNum()
        self.argsToPrint = argsToPrint
        self.ignorableArgs = ignorableArgs
        self.instanceCost = 'none'
        self.specialRegress = specialRegress

    def verifyArgs(self):
        varsUsed = set({})
        for (al, ps) in self.preconditions.items():
            for pre in ps:
                varsUsed = varsUsed.union(pre.getVars())
        for f in self.functions:
            for v in f.outVars:
                if isVar(v): varsUsed.add(v)
            for v in f.inVars:
                if isVar(v): varsUsed.add(v)
        for (rset, conds) in self.results:
            for r in rset:
                varsUsed = varsUsed.union(r.getVars())
                for (al, c) in conds.items():
                    for cc in c:
                        varsUsed.union(cc.getVars())

        varsDeclared = set([v for v in self.args if isVar(v)])

        usedMinusDeclared = varsUsed.difference(varsDeclared)
        declaredMinusUsed = varsDeclared.difference(varsUsed)

        if usedMinusDeclared:
            print self
            print 'Vars used without being declared', usedMinusDeclared
            raise Exception

        if declaredMinusUsed and debug('declaredVars'):
            print 'Warning'
            print self
            print 'Declared vars are not used', declaredMinusUsed
            raw_input('You have been warned.')

    def evalPrim(self, details):
        if self.prim:
            return self.prim(self.args, details)
        else:
            return None

    def incrementNum(self):
        self.num = Operator.num
        Operator.num += 1

    def allArgs(self):
        return self.args

    def allResultFluents(self):
        return squashSets([r for (r, ps) in self.results])
    
    def isGround(self):
        return not any([isVar(a) for a in self.args])

    def preconditionVars(self, includeFunctionVars = True):
        result = []
        for f in self.preconditionSet():
            result.extend(f.getVars())
        if includeFunctionVars:
            for f in self.functions:
                result.extend(f.outVars + f.inVars)
        debugMsg('precondVars', self.preconditionSet(), self.functions,
                 set(result))
        return set(result)

    def resultVars(self):
        result = []
        for f in self.results:
            for arg in f.args:
                if hasattr(arg, 'args'):
                    result.extend([v for v in arg.args if isVar(v)])
                elif isVar(arg):
                    result.append(arg)
            if isVar(f.value):
                result.append(f.value)
        return set(result)

    def isAbstract(self):
        return self.abstractionLevel < self.concreteAbstractionLevel

    # Stupidly slow.  Fix to precompute 
    def preconditionSet(self, allLevels = False):
        maxRange = (self.concreteAbstractionLevel + 1) if allLevels \
                   else (self.abstractionLevel + 1)
        result = []
        for i in range(maxRange):
            result.extend(self.preconditions[i])
        return result

    def sideEffectSet(self, allLevels = False):
        maxRange = (self.concreteAbstractionLevel + 1) if allLevels \
                   else (self.abstractionLevel + 1)
        result = []
        # Side effects up to level i
        for i in range(maxRange):
            result.extend(self.sideEffects.get(i, set({})))
        # Preconditions from level i+1 to max
        for i in range(maxRange, self.concreteAbstractionLevel + 1):
            result.extend(self.preconditions[i])
        return result

    def matchesForAbstraction(self, a):
        if self.name != a.name:
            return False
        selfAllArgs = self.allArgs()
        aAllArgs = a.allArgs()
        selfArgs = []
        aArgs = []
        for i in range(len(selfAllArgs)):
            if i not in self.ignorableArgs:
                selfArgs.append(selfAllArgs[i])
                aArgs.append(aAllArgs[i])
        return matchLists(selfArgs, aArgs, starIsWild = False) != None

    # Horrible.  Make more efficient.
    # Ancestors is the stack of plans above this one.
    def setAbstractionLevel(self, ancestors):
        ancestorList = squashOne(ancestors)
        if self.abstractionLevel < self.concreteAbstractionLevel:
            lastAL = -1
            for a in ancestorList:
                if self.matchesForAbstraction(a):
                    lastAL = max(lastAL, a.abstractionLevel)
            level = min(lastAL+1, self.concreteAbstractionLevel)
            if ancestorList:
                debugMsg('abstractionLevel', self, ancestorList,
                              level)    
            self.abstractionLevel = level

    def uniqueStr(self):
        return self.name + str(self.num)
        #return self.name + prettyString(self.allArgs(), eq =True)+str(self.num)
    def __str__(self):
        return self.name + prettyString(self.allArgs(), eq = True)
    __repr__ = __str__

    def __eq__(self, other):
        return str(self) == str(other)
    def __hash__(self):
        return str(self).__hash__()

    def copy(self):
        return self.applyBindings({})

    def applyBindings(self, bindings, rename = False):
        rb = copy.copy(bindings)
        if rename:
            for v in self.allArgs():
                if isVar(v) and not v in bindings:
                    rb[v] = makeVar(v)

        op = Operator(self.name,
                      [lookup(a, rb) for a in self.args],
                      dict([(v, [f.applyBindings(rb) for f in preConds]) \
                            for (v, preConds) in self.preconditions.items()]),
                      applyBindings(self.results, rb),
                      [f.applyBindings(rb) for f in self.functions],
                      self.f,
                      self.cost,
                      self.prim,
                      dict([(v, [f.applyBindings(rb) for f in preConds]) \
                            for (v, preConds) in self.sideEffects.items()]),
                      self.ignorableArgs,
                      self.argsToPrint,
                      self.specialRegress)
        op.abstractionLevel = self.abstractionLevel
        op.instanceCost = self.instanceCost
        return op

    def applyBindingsSideEffect(self, rb):
        self.args = [lookup(a, rb) for a in self.args]
        self.preconditions = dict([(v, [f.applyBindings(rb) for f in preConds])\
                            for (v, preConds) in self.preconditions.items()])
        self.results = [f.applyBindings(rb) for f in self.results]
        self.functions = [f.applyBindings(rb) for f in self.functions]
        self.sideEffects = [f.applyBindings(rb) for f in self.sideEffects]

    def getNecessaryFunctions(self):
        preconditions = self.preconditionSet()
        neededVars = set(self.preconditionVars(False))
        result = []
        for f in reversed(self.functions):
            outVarSet = set([v for v in f.outVars if isVar(v)])
            if neededVars.intersection(outVarSet) or f.isNecessary:
                neededVars = neededVars.difference(outVarSet)
                inVarSet = set([v for v in f.inVars if isVar(v)])
                neededVars = neededVars.union(inVarSet)
                result.append(f)
        result.reverse()
        return result


    # Returns a list of (goal, cost) pairs
    # Operators used in hierarchical heuristic
    def regress(self, goal, startState = None, heuristic = None):
        tag = 'regression'

        # Stop right away if an immutable precond is false
        if any([(p.immutable and p.isGround() and\
                 not startState.fluentValue(p) == p.getValue()) \
                for p in self.preconditionSet()]):
                debugMsg('regression:fail', 'immutable precond is false',
                         [p for p in self.preconditionSet() if \
                          (p.immutable and p.isGround() and \
                            not startState.fluentValue(p) == p.getValue())])
                return []

        results = squashSets([r for (r, c) in self.results])

        # Discharge conditions that are entailed by this result
        # It's an experiment to do this here, before getting all the bindings
        pendingFluents = []
        for gf in goal.fluents: 
            entailed = False
            for rf in results:
                bTemp = rf.entails(gf, startState.details)
                debugMsg('regression:entails',
                         'entails', rf, gf, bTemp, bTemp != False)
                if bTemp != False:
                    entailed = True
                    break
            if not entailed:
                pendingFluents.append(gf.copy())

        # Figure out which variables, and therefore which functions, we need.
        necessaryFunctions = self.getNecessaryFunctions()

        # Call backtracker to get bindings of unbound vars
        newBindings = btGetBindings(necessaryFunctions,
                                    pendingFluents, #goal.fluents,
                                    startState.details,
                                    avoid = goal.bindingsAlreadyTried)
        # Debugging stuff
        debugMsg('regression:bind', 'getting new bindings',
                 self.functions, goal.fluents, ('result', newBindings))
        if newBindings == None:
            if not inHeuristic or debug('debugInHeuristic'):
                debugMsg('regression:fail', self, 'could not get bindings',
                     'h = '+str(inHeuristic))
            if debug('regression:fail') and \
                    (not inHeuristic or debug('debugInHeuristic')):
                glob.debugOn = glob.debugOn + ['btbind']
                newBindings = btGetBindings(necessaryFunctions,
                                            goal.fluents,
                                            startState.details,
                                            avoid = goal.bindingsAlreadyTried)
                debugMsg('regression:fail', 'hope that was helpful')
                glob.debugOn = glob.debugOn[:-1]
            return []

        br = set()
        # Get rid of entailments *within* the results.  Kind of ugly.
        for f in results:
            bf = f.applyBindings(newBindings)
            addF = True
            for f2 in br:
                if f2.entails(bf, startState):
                    addF = False; break
                if bf.entails(f2, startState):
                    br.remove(f2)
            if addF: br.add(bf)
        boundResults = list(br)

        # Be sure the result is consistent
        if not goal.isConsistent(boundResults, startState.details):
            if not inHeuristic or debug('debugInHeuristic'):
                debugMsg('regression:inconsistent', self,
                         'results inconsistent with goal')
            # This is not a fatal flaw;  just a problem with these bindings

            return []

        # Some bindings that we make after this might apply to previous steps
        # in the plan, so we have to accumulate and store then
        bp = {}
        
        # Discharge conditions that are entailed by this result and
        # apply the special regression function if there is one
        newFluents = []
        for gf in goal.fluents: 
            entailed = False
            for rf in boundResults:
                bTemp = rf.entails(gf, startState.details)
                debugMsg('regression:entails',
                         'entails', rf, gf, bTemp, bTemp != False)
                if bTemp != False:
                    entailed = True
                    newBindings.update(bTemp)
                    bp.update(bTemp)
                    # print 'discharged', gf
                    break
            if not entailed:
                if self.specialRegress:
                    nf = self.specialRegress(gf, startState.details)
                else:
                    nf = gf.copy()
                if nf == None:
                    if not inHeuristic or debug('debugInHeuristic'):
                        print 'special regression fail'
                        debugMsg('regression:fail', 'special regress failure')
                    return []
                newFluents.append(nf)

        newBoundFluents = [f.applyBindings(newBindings) for f in newFluents]

        # Regress the implicit predicates by adding the bound results of this
        # operator to the conditions of those predicates.
        # Bound preconditions, too!
        boundPreconds = [f.applyBindings(newBindings) \
                         for f in self.preconditionSet()]

        boundSE = [f.applyBindings(newBindings) for f in self.sideEffectSet()]

        explicitResults = [r for r in boundResults \
                           if r.isGround() and not r.isImplicit()]
        explicitSE = [r for r in boundSE \
                           if r.isGround() and not r.isImplicit()]
        explicitPreconds = [r for r in boundPreconds \
                           if r.isGround() and not r.isImplicit()]

        # Add conditions and make new state.  Put these fluents into
        # the new state sequentially to get rid of redundancy.  Make
        # this code nicer.
        newGoal = State([])
        for f in newBoundFluents:
            if f.isConditional():

                #print 'adding conditions to', f.prettyString()
                # Preconds will not override results
                #v0 = f.valueInDetails(startState.details)
                f.addConditions(explicitResults, startState.details)
                #v1 = f.valueInDetails(startState.details)
                f.addConditions(explicitPreconds, startState.details)
                #v2 = f.valueInDetails(startState.details)
                f.addConditions(explicitSE, startState.details)
                #v3 = f.valueInDetails(startState.details)
                # if v0 == True and v3 == False:
                #     print 'value in details:', v0, v1, v2, v3
                #     raw_input('made it false!')
                f.update()
            newGoal.add(f, startState.details)

        debugMsg('regression:entails', 'Discharge entailed', ('Op', self),
                 ('goal', goal.fluents),
                 ('newBindings', newBindings),
                 ('newGoal', newGoal.fluents))

        # Stop right away if an immutable precond is false
        if any([(p.immutable and p.isGround() and \
                 not startState.fluentValue(p) == p.getValue()) \
                for p in boundPreconds]):
            if not inHeuristic or debug('debugInHeuristic'):
                debugMsg('regression:fail', 'immutable precond is false')
            return []

        # Could fold the boundPrecond part of this in to addSet later
        if not newGoal.isConsistent(boundPreconds, startState.details):
            if not inHeuristic or debug('debugInHeuristic'):
                if debug('regression:inconsistent'):
                    for f1 in boundPreconds:
                        for f2 in newGoal.fluents:
                            if f1.contradicts(f2, startState.details):
                                print '    contradiction\n', f1, '\n', f2
                    debugMsg('regression:inconsistent', self,
                             'preconds inconsistent with goal',
                             ('newGoal', newGoal), ('preconds', boundPreconds))
            bindingsNoGood = True
        elif newGoal.couldBeClobbered(boundSE, startState.details):
            if not inHeuristic or debug('debugInHeuristic'):
                if debug('regression:inconsistent'):
                    for f1 in boundSE:
                        for f2 in newGoal.fluents:
                            if f1.couldClobber(f2, startState.details):
                                print '    might clobber\n', f1, '\n', f2
                    debugMsg('regression:inconsistent', self,
                             'side effects may be inconsistent with goal',
                             ('newGoal', newGoal), ('sideEffects', boundSE))
            # Idea!  If this happens, try the operator at a more concrete level
            #bindingsNoGood = True
            print 'possible clobbering...ignoring for now'
            bindingsNoGood = False
        else:
            bindingsNoGood = False

        # Make another result, which is a place-holder for rebinding
        rebindLater = goal.copy()
        rebindLater.suspendedOperator = self.copy()
        rebindLater.bindingsAlreadyTried.append(newBindings)
        rebindLater.rebind = True
        rebindCost = glob.rebindPenalty

        if bindingsNoGood:
            rebindLater.suspendedOperator.instanceCost = rebindCost
            return [[rebindLater, rebindCost]]

        # Add in the preconditions.  We can make new bindings between
        # new and old preconds, but not between new ones!
        bTemp = newGoal.addSet(boundPreconds, moreDetails = startState.details)

        bp.update(bTemp)
        newBindings.update(bp)

        newGoal.depth = goal.depth + 1
        newGoal.operator = self.applyBindings(newBindings)

        newGoal.bindings = copy.copy(goal.bindings)
        newGoal.bindings.update(newBindings)


        if self.abstractionLevel == self.concreteAbstractionLevel:
            cost = self.cost(self.abstractionLevel,
                         [lookup(arg, newBindings) for arg in self.args],
                         startState.details)
        else:
            # Idea is to use heuristic difference as an estimate of operator
            # cost.
            hh = heuristic if heuristic else lambda s: s.easyH(startState)
            hNew = hh(newGoal)
            primOp = self.applyBindings(newBindings) # was self.copy()
            primOp.abstractionLevel = primOp.concreteAbstractionLevel

            numRegressSamples = 5
            tempGoal = goal
            for i in range(numRegressSamples):
                primOpRegr = primOp.regress(tempGoal, startState)
                if len(primOpRegr) == 0:
                    print 'Prim op regr: no applicable op: failing'
                    break
                elif len(primOpRegr) == 1:
                    print 'Retrying primOpRegr:  rebind op only'
                    tempGoal = primOpRegr[0][0]
                    continue
                else:
                    (sp, cp) = primOpRegr[0]
                    primPrecondCost = hh(sp)
                    if primPrecondCost == float('inf'):
                        print 'Retrying primOpRegr:  preconds infeasible'
                        tempGoal = primOpRegr[1][0]
                    else:
                        # good to go
                        break
 
            if hNew == float('inf'):
                # This is hopeless.  Give up now.
                debugMsg('infeasible', 'New goal is infeasible', newGoal)
                cost = float('inf')
            elif len(primOpRegr) < 2:
                # Looks good abstractly, but can't apply concrete op
                if not inHeuristic or debug('debugInHeuristic'):
                    debugMsg('infeasible', 'Concrete op not applicable',
                             goal)
                cost = float('inf')
            else:
                # Do one step of primitive regression on the old
                # state and then compute the heuristic on that.
                # This is an estimate of how hard it is to
                # establish all the preconds
                # Cost to get from start to one primitive step before
                # newGoal, plus the cost of the last step
                if primPrecondCost == float('inf'):
                    debugMsg('infeasible', 'Prim preconds infeasible', sp)
    
                hOld = primPrecondCost + cp
                # Difference between that cost, and the cost of
                # the regression of the abstract action.  This is
                # an estimate of the cost of the abstract action.
                cost = hOld - hNew

                if cost < 0:
                    cost = cp

                if hOld != float('inf'):
                    rebindCost = hOld + rebindCost
                else:
                    # Try a smaller penalty here to encourage rebinding
                    rebindCost = 10

                # LPK!!  This is a way to cut down on generator calls
                # but is potentially risky.  Disabled for now.
                # Store the bindings we
                # made in this process!  But keep the abstract
                # preconditions
                if False: #not inHeuristic:
                    psb = primOpRegr[0][0].bindings
                    newOp = newGoal.operator.applyBindings(psb)
                    newGoal.operator = newOp
                
                debugMsg('abstractCost',
                         ('with heuristic', heuristic != None),
                         cost)
        if not inHeuristic or debug('debugInHeuristic'):
            debugMsg(tag, 'Final regression result', ('Op', self),
                     ('cost', cost),
                     ('goal',  goal.prettyString(False, startState)),
                     ('newGoal', newGoal.prettyString(False, startState)))

        rebindLater.suspendedOperator.instanceCost = rebindCost
        if cost == float('inf'):
            if not inHeuristic or debug('debugInHeuristic'):
                debugMsg('regression:fail', 'infinite cost')
            return [[rebindLater, rebindCost]]
        newGoal.operator.instanceCost = cost

        return [[newGoal, cost], [rebindLater, rebindCost]]

    def prettyString(self, eq = True):
        # Make unbound vars anonymous
        def argS(a):
            return prettyString(a, eq) if not isVar(a) else '_'

        argsToPrint = self.args if (eq or self.argsToPrint == None) else \
                     [self.args[ai] for ai in self.argsToPrint]
        argStr = '(' + ', '.join([argS(v) for v in argsToPrint]) + ')'
        return self.name + argStr + ':'+str(self.abstractionLevel)

def simplifyCond(oldFs, newFs, details = None):
    result = State(list(oldFs))
    for f in newFs:
        if (not f.isGround()) or f.isImplicit() or f.immutable:
            continue
        if result.isConsistent([f], details):
            result.add(f, details)
    debugMsg('simplifyCond', ('oldFs', oldFs),
             ('newFs', newFs), ('result', result.fluents))
    resultList = list(result.fluents)
    resultList.sort(key = str)
    return tuple(resultList)
    
def btGetBindings(functions, goalFluents, start, avoid = []):
    # Avoid is list of dictionaries
    avoid = set(tuplify(b) for b in avoid)
    # Helper fun to find a set of bindings that hasn't been found before
    def gnb(funs, sofar):
        if funs == []:
            if not tuplify(sofar) in avoid:
                debugMsg('btbind', 'returning', sofar)
                return sofar
            else:
                debugMsg('btbind', 'hit duplicate', sofar)
                return None
        else:
            f = funs[0]
            values  = f.fun([lookup(v, sofar) for v in f.inVars],
                            [z.applyBindings(sofar) for z in goalFluents],
                            start,
                            [lookup(v, sofar) for v in f.outVars])
            if values == None or values == []:
                debugMsg('btbind', 'fun failed', f, f.inVars, sofar)
                return None
            for val in values:
                assert len(f.outVars) == len(val)
                sf = copy.copy(sofar)
                for (var, vv) in zip(f.outVars, val):
                    if isVar(var) and not var == vv:
                        assert not isVar(vv)
                        sf[var] = vv
                result = gnb(funs[1:], sf)
                if result != None:
                    return result
            debugMsg('btbind', 'ran out of values', f, sofar)
            return None

    funsInOrder = [f for f in functions if \
                   f.isNecessary or not isGround(f.outVars)]
    novelBindings = gnb(funsInOrder, {})
    return novelBindings

class RebindOp:
    name = 'Rebind'
    def regress(self, goal, startState, heuristic = None):
        g = goal.copy()
        op = goal.suspendedOperator
        debugMsg('rebind', 'about to try local rebinding', op, g.bindings)
            
        g.suspendedOperator = None
        g.rebind = False
        results = op.regress(g, startState, heuristic)

        if len(results) > 0:
            debugMsg('rebind', 'successfully rebound local vars',
                     'costs', [c for (s, c) in results], 'minus',
                     glob.rebindPenalty)
            results[0][1] -= op.instanceCost
        else:
            debugMsg('rebind', 'failed to rebind local vars')
        return results

    def __str__(self):
        return 'RebindVars'
    def prettyString(self, eq):
        return str(self)

class Function:
    def __init__(self, outVars, inVars, fun, funName, isNecessary = False):
        assert isStruct(outVars)
        self.outVars = outVars
        self.inVars = inVars
        self.fun = fun
        self.funName = funName
        # True if we should always evaluate this function, no matter the
        # abstraction level
        self.isNecessary = isNecessary

    def applyBindings(self, bindings):
        return Function([lookup(v, bindings) for v in self.outVars],
                        [lookup(v, bindings) for v in self.inVars],
                        self.fun,
                        self.funName,
                        self.isNecessary)

    def prettyString(self, eq = True):
        return 'Fun:'+str(self.outVars)+' = '+self.funName+\
               prettyString(self.inVars, eq)
    __str__ = prettyString
    __repr__ = __str__

        
######################################################################
# Heuristic
######################################################################

# New heuristic.  Independent backchaining, summing costs.

hCache = {}
def hCacheReset():
    hCache.clear()

def hCacheDel(f):
    if f in hCache:
        del hCache[f]

# hCache maps each fluent to the set of actions needed to achieve it
# This can be saved until the next action is taken

# hCache[f] = (totalCost, operators)

# At AND nodes, take union
# At OR nodes, take option with least summed cost
                
def hAddBack(start, goal, operators, minK = 20, maxK = 30,
             staticEval = lambda f: float("inf")):

    raw_input('This version of hAddBack probably needs to have bugs fixed'+\
              '; see the version in belief.py for details')

    # Return a set of actions
    def aux(fl, k):
        if fl in hCache:
            if debug('hAddBackV'): print 'c',
            return hCache[fl]
        
        g = State([fl])
        if start.satisfies(g):
            if debug('hAddBackV'): print 's',
            hCache[fl] = (0, set())
            return hCache[fl]
        elif k == 0:
            if debug('hAddBackV'): print 'l',
            # Get a final value. At a leaf.  Use static eval
            v = staticEval(fl)
            dummyO = Operator('dummy'+prettyString(v), [], {}, [])
            dummyO.instanceCost = v
            hCache[fl] = (v, set([dummyO]))
            return hCache[fl]
        else:
            # If totalCost = infinity, ignore action set
            totalCost, actSet = (float('inf'), set())

            # OR loop over operators and ways of achieving them
            for o in applicableOps(g, operators, start):
                o.abstractionLevel = o.concreteAbstractionLevel
                pres = o.regress(g, start)
                for pre in pres[:-1]:
                    (preImage, newOpCost) = pre
                    newActSet = set([preImage.operator])
                    canonicalChildren = []
                    # AND loop over preconditions
                    for ff in preImage.fluents:
                        canonicalChildren.append(ff)
                        subCost, subActSet = aux(ff, k-1)
                        # make this side effect
                        newActSet = newActSet.union(subActSet)
                    newTotalCost = sum([op.instanceCost for op in newActSet])

                    if newTotalCost < totalCost:
                        hCache[fl] = (newTotalCost, newActSet)
                        print '**Added', len(hCache), newTotalCost, fl
                        
                        (totalCost, actSet) = (newTotalCost, newActSet)
                        if debug('hAddBackV'):
                            print ' '
                            print o
                            print 'Children:'
                            for c in children:
                                print '    ', hCache[c][0], c
                            
                        debugMsg('hAddBackV', ('stored', k,
                                 newTotalCost, newOpCost, fl))
            return (totalCost, actSet)

    totalActSet = set()
    # AND loop over fluents
    for ff in goal.fluents:
        (ic, actSet) = aux(ff, maxK)
        if ic == float('inf'):
            debugMsg('hAddBackInf', ('infinite cost', ff))
            print ic
            return ic
        debugMsg('hAddBackV', ff, ('actSet', actSet))
        totalActSet = totalActSet.union(actSet)
    totalCost = sum([op.instanceCost for op in totalActSet])
    debugMsg('hAddBackV', ('instancecosts',
                           [op.instanceCost for op in totalActSet]))
    if debug('hAddBackPrintVal'):
        print totalCost
    return totalCost


######################################################################
# Execution
######################################################################

# Abstract subtask for initializing the plan stack
top = Operator('Top', [], {1:[]}, [], [], None)
top.abstractionLevel = 0
nop = Operator('Nop', [], {1:[]}, [], [], None)

# Skeleton is a list of lists of operators
# First is used in the first planning problem, etc.

# Only pop a level if we fall out of the enveope or satisfy the subgoal
def HPNCommit(s, g, ops, env, h = None, fileTag = None, hpnFileTag = None,
        skeleton = None, nonMonOps = []):
    f = writePreamble(hpnFileTag or fileTag)
    ps = Stack()
    ancestors = []
    ps.push(Plan([(nop, State([])), (top, g)]))
    try:
        while not ps.isEmpty():
            p = ps.top()
            (op, subgoal) = p.nextStep(s)
            if not op:
                # Plan is not applicable in state s
                ps.pop()
                if ancestors: ancestors.pop()
            else:
                if op.isAbstract():
                    # Plan again at a more concrete level
                    ancestors += [op]
                    writeSubgoalRefinement(f, p, subgoal)
                    p = planBackward(s, subgoal, ops, ancestors, h, fileTag,
                                     skeleton = skeleton[subgoal.planNum]\
                                     if skeleton else None,
                                     nonMonOps = nonMonOps)
                    assert p, 'Planning failed.'
                    planObj = makePlanObj(p, s)
                    planObj.printIt()
                    ps.push(planObj)
                    writeSubtasks(f, planObj, subgoal)
                else:
                    # Execute
                    executePrim(op, s, env, f)
    finally:
        writeCoda(f)

def HPN(s, g, ops, env, h = None, fileTag = None, hpnFileTag = None,
        skeleton = None, verbose = False, nonMonOps = []):
    f = writePreamble(hpnFileTag or fileTag)
    ps = PlanStack()
    ancestors = []
    ps.push(Plan([(nop, State([])), (top, g)]))
    try:
        (op, subgoal) = (top, g)
        while not ps.isEmpty() and op != None:
            if op.isAbstract():
                # Plan again at a more concrete level
                parent = ps.guts()[-1]
                writeSubgoalRefinement(f, parent, subgoal)
                p = planBackward(s, subgoal, ops, ancestors, h, fileTag,
                                 lastOp = op,
                                 skeleton = skeleton[subgoal.planNum]\
                                            if (skeleton and \
                                                len(skeleton)>subgoal.planNum) \
                                                else None,
                                 nonMonOps = nonMonOps)
                assert p, 'Planning failed.'
                planObj = makePlanObj(p, s)
                planObj.printIt(verbose = verbose)
                ps.push(planObj)
                ancestors.append(planObj.getOps())
                writeSubtasks(f, planObj, subgoal)
            elif op.prim != None:
                # Execute
                executePrim(op, s, env, f)
                assert len(s.valueCache) == 0
                
            # Decide what to do next
            # will pop levels we don't need any more, so that p is on the top
            # op will be None if we are done
            (op, subgoal) = ps.nextStep(s, f)
            # Possibly pop ancestors
            ancestors = ancestors[0:ps.size()]
    finally:
        writeCoda(f)
        
def executePrim(op, s, env, f = None):
    # The refinement procedure should extract whatever information is
    # necessary from belief.details and store it in the prim.
    params = op.evalPrim(s.details)
    # Print compactly unless we are debugging
    print 'PRIM:', op.prettyString(eq = debug('prim'))
    if debug('prim'): print '     params', params
    writePrimRefinement(f, op)
    obs = env.executePrim(op, params)
    s.updateStateEstimate(op, obs)
    debugMsg('prim', 'done')

class PlanStack(Stack):
    # Return op and subgoal to be addressed next by HPN.
    # Go down layers (from most abstract to least)
    # Ask each layer what it wants its next step to be
    # If it's different than the subgoal at the next layer below, pop all
    #    lower layers
    def nextStep(self, s, f = None):
        layers = self.guts()
        numLayers = len(layers)
        preImages = self.computePreimages()
        # Subgoal layer i-1 is executing
        (upperOp, upperSubgoal) = self.nextLayerStep(layers[0], preImages[0],
                                                     s, f)
        for i in range(1, len(layers)):
            # Op and subgoal layer i wants to execute
            (op, subgoal) = self.nextLayerStep(layers[i], preImages[i], s, f)
            # Ultimate goal of layer i
            lowerGoal = layers[i].steps[-1][1]
            if not op or upperSubgoal != lowerGoal:
                # Layer i doesn't have an action to execute OR
                # subgoal at i-1 doesn't equal actual goal at i
                debugMsg('nextStep', ('op', op), ('upperSG', upperSubgoal),
                         ('lowerGoal', lowerGoal))
                # This is a surprise if previous is not current - 1
                # Could actually check this condition going down farther in
                # order to see where the surprise originated.

                if lowerGoal and upperSubgoal:
                    previousUpperIndex = layers[i-1].subgoalIndex(lowerGoal)
                    currentUpperIndex = layers[i-1].subgoalIndex(upperSubgoal)

                    if previousUpperIndex != currentUpperIndex -1 :
                        writeSurprise(f, layers[i-1], previousUpperIndex,
                                      currentUpperIndex)
                        debugMsg('executionSurprise', ('layer', i-1),
                             ('prevIndex', previousUpperIndex),
                             ('currIndex', currentUpperIndex),
                             ('popping layers', i, 'through', len(layers)-1))
            
                # For purposes of drawing in the tree, find the next
                # step at each level below
                for j in range(i+1, len(layers)):
                    self.nextLayerStep(layers[j], preImages[j], s, f,
                                       quiet = True)

                # Get rid of layers i and below
                self.popTo(i)
                # Replan again for upperSubgoal
                return (upperOp, upperSubgoal)
            elif i == len(layers)-1:
                # bottom layer, return subgoal at level i
                debugMsg('nextStep', 'bottomLayer', op, subgoal)
                assert op.isAbstract() or op.prim != None, \
                             'Selected inferential op for execution'
                return (op, subgoal)
            else:
                # go down a level
                (upperOp, upperSubgoal) = (op, subgoal)
        return (upperOp, upperSubgoal)

    # Return op and subgoal in this layer to be addressed next
    def nextLayerStep(self, layer, preImages, s, f = None, quiet = False):
        # If the final subgoal holds, then we're done
        if any([s.satisfies(sg) for sg in preImages[-1]]):
            if debug('nextStep'):
                for sg in preImages[-1]:
                    print sg, s.satisfies(sg)
                raw_input('already satisfied')
            return None, None
        # Work backwards to find the latest subgoal that's satisfied
        for i in range(layer.length-1, -1, -1):
            if not quiet:
                debugMsg('nextStep', [s.satisfies(sg) for sg in preImages[i]])
            if any([s.satisfies(sg) for sg in preImages[i]]):
                layer.lastStepExecuted = i+1
                if not quiet:
                    debugMsg('nextStep', 'returning', layer.steps[i+1])

                (op, _) = layer.steps[i+1]
                if op.prim == None and not (op.isAbstract() or op == top):
                    print 'Selecting an inferential operator for execution'
                    print op
                    preImage = preImages[i]
                    postCond = preImages[i+1]
                    print 'Post conditions not satisfied'
                    for thing in postCond:
                        for fl in thing.fluents:
                            #if not fl.valueInDetails(s.details) == True:
                            if not s.fluentValue(fl) == True:
                                print fl
                    raw_input('Continue?')
                return layer.steps[i+1]
        # Not in the envelope
        debugMsg('nextStep', 'not in envelope')
        

        fooFluents = []
        for fl in layer.steps[layer.lastStepExecuted][1].fluents:
            fv = s.fluentValue(fl, recompute = True)
            if fl.value != fv:
                fooFluents.append(fl)
        writeFailure(f, layer, fooFluents)
        
        if debug('executionFail') and not quiet:
            print 'Next step: failed to satisfy any pre-image'
            print 'Was expecting to satisfy preimage', layer.lastStepExecuted
            glob.debugOn.append('testVerbose')
            foundError = False
            for fl in layer.steps[layer.lastStepExecuted][1].fluents:
                fv = s.fluentValue(fl, recompute = True)
                if fl.value != fv:
                    print 'wanted:', fl.value, 'got:', fv
                    print '    ', fl.prettyString()
                    foundError = True
            glob.debugOn.pop()
            assert foundError, 'inconsistency!?'
            raw_input('execution fail')
        return None, None
        

    def computePreimages(self):
        # include the envelope of the plan at the layer below in the
        # pre-image above, in order to avoid the problem of a
        # pre-image going false during hierarchical execution
        layers = self.guts()
        # preimages will be a list of lists of states
        #   - one list per level
        #   - one list per step in the level representing pre-image 
        preImages = []
        # handle bottom level separately.  no disjunction.
        preImages.append([[subgoal] for (op, subgoal) in layers[-1].steps])
        # print 'Started with layer', len(layers)-1
        # print preImages[0]
        for i in range(len(layers)-2, -1, -1):
            # print 'Working on layer', i
            lowerSubgoal = preImages[-1][-1][0]
            layer = layers[i].steps
            # print '    steps:', layer
            layerPreImage = []
            for j in range(1, len(layer)):
                # print '    Working on step', j-1
                (opj, subgoalj) = layer[j]
                (opjm1, subgoaljm1) = layer[j-1]                 
                if subgoalj == lowerSubgoal:
                    # print '        getting elts from below'
                    unionPreImage = squash(preImages[-1][:-1])
                    # print '        ', unionPreImage
                    layerPreImage.append(unionPreImage + [subgoaljm1])
                else:
                    # print '        getting regular elt'
                    # print '        ', subgoaljm1
                    layerPreImage.append([subgoaljm1])
            # print '        adding last subgoal'
            # print '        ', subgoalj
            layerPreImage.append([subgoalj])
            # print '     preimages for whole layer'
            # print '    ', layerPreImage
            preImages.append(layerPreImage)
        preImages.reverse()
        # print 'returning'
        # print preImages
        # raw_input('ok go?')
        # Top-level first
        return preImages
    
    # Remove levels i and greater
    def popTo(self, level):
        while len(self.guts()) > level:
            self.pop()
        

# Convert from the output of a regresion plan
# [(None, goal), (op_n, sg_n), ..., (op_0, planPreimage)]
# to a plan instance

def makePlanObj(regressionPlanUnbound, start):
    # Result is supposed to be a list of (action, state)
    planPreImage = regressionPlanUnbound[-1][1]
    b = planPreImage.bindings
    grounding = getGrounding(planPreImage.fluents, start.details)
    b.update(grounding)
    plan = applyBindings(regressionPlanUnbound, b)
    n = len(plan)
    regressionPlan = [plan[i] for i in xrange(n) \
                      if i == n-1 or plan[i+1][0].name not in ['RebindVars', 'Rebind']]
    planPreImage = applyBindings(planPreImage, b)
    result = [(None, planPreImage)]

    for i in range(len(regressionPlan)-1, 0, -1):
        subgoal = regressionPlan[i-1][1]
        operator = regressionPlan[i][0]
        preImage = regressionPlan[i][1]
        operator = preImage.operator
        # assert operator.isGround() or \
        #   operator.abstractionLevel < operator.concreteAbstractionLevel

        # There are really two different kinds of variables: ones that
        # are critical for defining the operation and ones that are
        # used for bookkeeping.  We really only need for the first
        # ones to be bound; other ones might not be.

        pi = preImage.fluents
        pc = operator.preconditionSet()
        assert preImage == regressionPlan[i][1]
        success = State(pi).satisfies(State(pc), start.details)
        assert preImage == regressionPlan[i][1]
        if not success:
            for pcf in pc:
                if not State(pi).satisfies(State([pcf]), start.details):
                    print pcf
                    assert preImage == regressionPlan[i][1]
        assert success, 'preimage does not satisfy preconds'
        debugMsg('planObj', 'plan step', i, operator,
                 subgoal.fluents)
        result.append((operator, subgoal))

    return Plan(result)

class Plan:
    def __init__(self, opGoalList):
        # List of (o, g) pairs
        self.steps = opGoalList
        self.length = len(opGoalList)
        self.lastStepExecuted = 0

    # Return (None, None) if not applicable Only used in the HPNCommit
    # version, which commits to a layer until the envelope is exited
    # (even if it exits at the level above).
    def nextStep(self, s):
        # If the final subgoal holds, then we're done
        if s.satisfies(self.steps[self.length-1][1]):
            return None, None
        # Work backwards to find the latest subgoal that's satisfied
        for i in range(self.length-1, -1, -1):
            if s.satisfies(self.steps[i][1]):
                self.lastStepExecuted = i+1
                return self.steps[i+1]
        # No subgoals are satisfied, not even the first pre-image
        print 'Next step: failed to satisfy any pre-image'
        print 'Was expecting to satisfy preimage', self.lastStepExecuted
        if debug('executionFail'):
            for f in self.steps[self.lastStepExecuted][1].fluents:
                fv = s.fluentValue(f)
                if f.value != fv:
                    print 'wanted:', f.value, 'got:', fv
                    print '    ', f
                    raw_input('go?')

        return None, None

    # return index of this subgoal in the plan
    # None if it's not there
    def subgoalIndex(self, subgoal):
        for i in range(self.length):
            if self.steps[i][1] == subgoal:
                return i
        return None

    def getOps(self):
        return [o for (o, g) in self.steps[1:]]

    def printIt(self, verbose = True):
        totalCost = 0
        print '** Start plan **'
        if verbose:
            print '    Initial preimage :', \
              self.steps[0][1].prettyString().replace('\\n','\n')
        for i in range(1, self.length):
            print '    step', i, ':', \
                  self.steps[i][0].prettyString(eq = verbose)
            print '        cost = ', prettyString(self.steps[i][0].instanceCost)
            totalCost += self.steps[i][0].instanceCost
            if verbose:
                print '    result', i, ':', \
                  self.steps[i][1].prettyString().replace('\\n','\n')
        print '** End plan **'
        print '===== Total Cost =', prettyString(totalCost), '====='
        debugMsg('displayPlan', 'Continue?')

# Return an instance of Plan
# P is a list of (op, state)

# p coming in is a list of (o, s) pairs, which each operation o
# leading to a state s
# need to return a list of (o_i, g_i) pairs where g_i is the preimage
# of g_{i+1} under o_{i+1}

def computePreimages(p, ops, g):
    # Because we have implicit predicates, it's tricky to figure out
    # who produced the critical results when.
    # !! Currently only works with explicit preds
    preImages = []
    start = p[0][1]
    sg = set(g.fluents)
    for (o, s) in reversed(p):
        preImages.append((o, State(list(sg))))
        if o:
            sg = sg.difference(o.results).union(o.preconditionSet())
    preImages.reverse()
    return Plan(preImages)

######################################################################
# Planning
######################################################################

# What should the heuristic be like in belief space?  To work best, we
# would like to allow the details to have multiple locations with
# multiple distributions for objects and the robot.

def hNum(start, goal, operators):
    # ignore non-ground fluents
    # return num of fluents that are false in start state
    return len([f for f in goal.fluents if \
                f.isGround() and start.fluentValue(f) != f.value])

# Handling all binding choices (including objects) by the rebinding mechanism. 
# Allow some operations to be nonmonotonic if they're on the nonMonOps list.
def applicableOps(g, operators, startState, ancestors = [], skeleton = None,
                  monotonic = True, lastOp = None, nonMonOps = []):

    tag = 'applicableOps'
    result = set([])
    if skeleton:
        #monotonic = False
        if g.depth < len(skeleton):
            ops = [skeleton[g.depth]]
            debugMsg('skeleton', g.depth, skeleton[g.depth])
        else:
            debugMsg('skeleton', 'Skeleton exhausted', g.depth)
            glob.debugOn = glob.debugOn + ['satisfies']
            startState.satisfies(g)
            debugMsg('skeleton', 'Skeleton exhausted', g.depth)
            glob.debugOn = glob.debugOn[:-1]
            return []
    elif lastOp and g.depth == 0 and lastOp != top:
        # At least ensure we try these bindings
        ops = [lastOp] + operators
    else:
        ops = operators

    if g.rebind:
        debugMsg(tag, 'Doing rebind op')
        return [RebindOp()]

    # Consider doing this work in a preprocessing pass
    for o in ops:
        debugMsg('appOp:detail', 'Operator', o)
        sharedPreconds = o.preconditions

        resultSets = powerset(o.results, includeEmpty = False)
        for resultSet in resultSets:
            debugMsg('appOp:result', 'result set', resultSet)
            preConds = mergeDicts([sharedPreconds] +\
                                   [ps for (r, ps) in resultSet])

            # List of sets of result fluents
            resultSetList = [r for (r, ps) in resultSet]
            results = squashSets(resultSetList)

            newOp = Operator(o.name, o.args, preConds,
                             #[(r, {}) for r in results]
                             [(results, {})],
                             o.functions, o.f, o.cost, o.prim,
                             o.sideEffects,
                             o.ignorableArgs,
                             o.argsToPrint,
                             o.specialRegress)
            bigBindingSet = getBindingsBetween(list(results), list(g.fluents),
                                               startState)
            bindingSet = []
            for b in bigBindingSet:
                boundRFs = [f.applyBindings(b) for f in results]
                # All results should be bound
                allBound = all([f.isGround() for f in boundRFs])
                # All results should be useful
                allUseful = all([any([rf.entails(gf, startState.details) \
                                      != False \
                                      for gf in g.fluents]) \
                                  for rf in boundRFs])
                # monotonic
                mono = any([(f.isGround() and \
                             startState.fluentValue(f) != f.getValue()) \
                                    for f in boundRFs])
                # check to see that this binding doesn't achieve any
                # additional results in the goal.  If it does, leave
                # it off the list because it doesn't have all the
                # necessary preconditions, and some other version
                # does.
                allBoundRfs = [f.applyBindings(b) for f in o.allResultFluents()]
            
                # Require these to be ground?  Or, definitely, not all
                # variables
                extraRfs = set(allBoundRfs).difference(set(boundRFs))
                # Asking this question backward.  We want to know whether there
                # are bindings that would make rf entail gf.
                dup = any([(rf.isPartiallyBound() and \
                            gf.entails(rf, startState) != False) \
                           for rf in extraRfs for gf in g.fluents])

                if allUseful and not dup and \
                    (mono or not monotonic or o in nonMonOps):
                    debugMsg('appOp:detail', 'adding binding', b, boundRFs)
                    bindingSet.append(b)
                elif not allUseful:
                    debugMsg('appOp:detail', 'all results not useful', b,
                             boundRFs)
                elif dup:
                    debugMsg('appOp:detail', 'some extra result in goal', b,
                             extraRfs)
                elif monotonic:
                    debugMsg('appOp:detail', 'nonmon so skipping binding', b,
                             boundRFs)
            for b in bindingSet:
                if b != False:
                    newOpBound = newOp.applyBindings(b, rename = True)
                    newOpBound.setAbstractionLevel(ancestors)
                    if not redundant(newOpBound, result):
                        result.add(newOpBound)
                        debugMsg('appOp:detail', 'added bound op', newOpBound)
                    else:
                        debugMsg('appOp:detail', 'redundant op', newOpBound)

    resultNames = [o.name for o in result]
    if len(result) == 0:
        print g
        debugMsg('appOp:number', ('h', inHeuristic, 'number', len(result)))
    debugMsg(tag, ('h', inHeuristic, 'number', len(result)), ('result', result))
    return result

def redundant(o1, opList):
    for o2 in opList:
        if opEquiv(o1, o2):
            return True
    return False

# Check to see if two operators are renamings of one another
def opEquiv(o1, o2):
    m1 = matchLists(o1.args, o2.args)
    m2 = matchLists(o2.args, o1.args)
    return m1 != None and m2 != None and \
        o1.name == o2.name and \
        len(m1.keys()) == len(set(m1.values())) and \
        len(m2.keys()) == len(set(m2.values()))

# Try to see if the unbound variables in the fluents can be bound in the
# details
def getGrounding(fluents, details):
    b = {}
    for f in fluents:
        if not f.isGround():
            g = f.getGrounding(details)
            if g != None:
                b.update(g)
    bFluents = [f.applyBindings(b) for f in fluents] if b != {} else fluents
    if all([bf.isGround() for bf in bFluents]):
        return b
    else:
        return None

def planBackwardAux(goal, startState, ops, ancestors, skeleton, monotonic,
                    lastOp, nonMonOps, heuristic, h, visitF, expandF,
                    prevExpandF, maxCost):
    return ucSearch.search(goal,
                           lambda subgoal: startState.satisfies(subgoal),
                           lambda g: applicableOps(g, ops,
                                                   startState,
                                                   ancestors, skeleton,
                                                   monotonic = monotonic,
                                                   lastOp = lastOp,
                                                   nonMonOps = nonMonOps),
                           lambda s, o: o.regress(s, startState, 
                                                  heuristic if h else h),
                           heuristic = heuristic, 
                           visitF = visitF,
                           expandF = expandF,
                           prevExpandF = prevExpandF,
                           multipleSuccessors = True,
                           maxNodes = 100000,
                           maxHDelta = maxHDelta,
                           hmax = hmax, 
                           greedy = plannerGreedy,
                           verbose = False,
                           maxCost = maxCost)        

def planBackward(startState, goal, ops, ancestors = [],
                 h = None, fileTag = None, skeleton = None, lastOp = None,
                 nonMonOps = []):

    skel = copy.copy(skeleton)
    goal.depth = 0
    
    if h:
        heuristic = lambda g: h(startState, g, ops, ancestors)
    else:
        heuristic = lambda g: 0

    hCacheReset() # flush heuristic values

    if fileTag:
        visitF = lambda s1, c1, h1, a, s2, c2, h2: \
                           visitTrace(f1, s1, c1, h1, a,
                                      s2, c2, h2, startState, heuristic)
        expandF = lambda n : expandTrace(f1, f2, n, False, startState,
                                         heuristic)
        prevExpandF = lambda n : expandTrace(f1, f2, n, True, startState,
                                             heuristic)
    else:
        (visitF, expandF, prevExpandF) = (None, None, None)

    if not hasattr(goal, 'planNum'):
        goal.planNum = 0

    (f1, f2) = None, None
    try:
        if fileTag:
            (f1, f2) = writeSearchPreamble(goal.planNum, fileTag)
        # Try monotonic first
        if glob.monotonicFirst: 
            (p, c) = planBackwardAux(goal, startState, ops, ancestors, skeleton,
                                     True, lastOp, nonMonOps, heuristic, h,
                                     visitF, expandF, prevExpandF, maxMonoCost)
            if p:
                if f1: writeSuccess(f1, f2, p)
                return p
            if f1:  writeSearchCoda(f1, f2)
            debugMsg('nonmon', 'Monotonic failed')
        # Now try non-monotonic
        if fileTag:
            (f1, f2) = writeSearchPreamble(goal.planNum, fileTag+'NonMon')
        hCacheReset() # flush heuristic values
        (p, c) = planBackwardAux(goal, startState, ops, ancestors, skeleton,
                                 False, lastOp, nonMonOps, heuristic, h,
                                 visitF, expandF, prevExpandF, float('inf'))
        if p and f1:
            writeSuccess(f1, f2, p)
        if p: return p

    finally:
        if f1:
            writeSearchCoda(f1, f2)
                           

############################################################################
#
# Binding stuff
#
############################################################################

# Bind at least one result
# Returns a list of possible bindings (maybe empty)

def getBindingsBetween(resultFs, goalFs, startState):
    if len(resultFs) == 0:
        debugMsg('gbb:detail', resultFs, [{}])
        return [{}]
    else:
        result = []
        rf = resultFs[0]
        debugMsg('gbb:detail', 'working on', rf)
        restAnswer = getBindingsBetween(resultFs[1:], goalFs, startState)
        debugMsg('gbb:detail', 'rest of answer', restAnswer)
        for b in restAnswer:
            matched = False
            for gf in goalFs:
                rfb = rf.applyBindings(b)
                newB = gf.entails(rfb, startState.details)
                if newB != False:
                    matched = True
                    debugMsg('gbb:detail', 'entails', b, gf.applyBindings(b),
                              ('newB', newB))
                    newB.update(b)
                    result.append(newB)
                    debugMsg('gbb:detail', 'appended updated newB', newB)
            if not matched:
                result.append(b)
        debugMsg('gbb', 'handled', rf, result)
        return result
    
############################################################################
#
# Printing utilities    
#
############################################################################

# For search trees
# red is color 1, unused

# clear
visitStyle = '" [shape=box, label="cost='
# blue
expandStyle = \
  '" [shape=box, style=filled, colorscheme=pastel16, color=2, label="cost='
# yellow
dupStyle = \
  '" [shape=box, style=filled, colorscheme=pastel16, color=6, label="cost='
# purple
rebindStyle = \
  '" [shape=box, style=filled, colorscheme=pastel16, color=4, label="cost='
# green
successStyle = \
  '" [shape=box, style=filled, colorscheme=pastel16, color=3, label="cost='

# Write and flush
def wf(f, s):
    f.write(s)
    f.flush()
    os.fsync(f.fileno())

dotSearchId = 0

def writeSearchPreamble(num, tag = ''):
    if local.dotSearch and (num != None or tag != '') and tag != 'nofile':
        f1 = open(local.dotSearch%(str(num) + tag, \
                                   '_'+timeString()), 'w')
        f2 = open(local.dotSearchX%(str(num) + tag, \
                                    '_'+timeString()), 'w')
        f1.write('digraph G {\n')
        f1.write('    ordering=out;\n')
        f1.write('    node [fontname=HelveticaBold];\n')
        f2.write('digraph G {\n')
        f2.write('    ordering=out;\n')
        f2.write('    node [fontname=HelveticaBold];\n')
        return (f1, f2)
    else:
        return (None, None)

def writeSuccess(f1, f2, p):
    for (a, s) in p:
        f1.write('    "'+s.uniqueStr()+\
                 '" [shape=box, style=filled, colorscheme=pastel16,color=3];\n')
        f2.write('    "'+s.uniqueStr()+\
                 '" [shape=box, style=filled, colorscheme=pastel16,color=3];\n')

def writeSearchCoda(f1, f2):
    if local.dotSearch:
        if f1 and not f1.closed:
            f1.write('}\n')
            f1.close()
        if f2 and not f2.closed:
            f2.write('}\n')
            f2.close()

# file, state, cost, heurstic
def writeSearchNode(f, s, c, h, styleStr, start = None, hFun = None):
    f.write('    "'+s.uniqueStr()+\
            styleStr+prettyString(c, True)+\
            ' h='+prettyString(h)+'\\n'+\
            s.prettyString(False, start, hFun)+'"];\n')

def getStyle(s, default = visitStyle):
    if s.rebind:
        return rebindStyle
    else:
        return default

def writeSearchArc(f, s1, s2, a):
    # use s2.operator instead of a, if it is not None,
    # because it has the bindings in it
    op = s2.operator or a
    
    f.write('    "'+s1.uniqueStr()+'" -> "'+s2.uniqueStr()+'"[label="'+\
            op.prettyString(False)+'"];\n')

def visitTrace(f, s1, c1, h1, a, s2, c2, h2, start = None, hFun = None):
    writeSearchNode(f, s1, c1, h1, getStyle(s1), start, hFun)
    writeSearchNode(f, s2, c2, h2, getStyle(s2), start, hFun)
    writeSearchArc(f, s1, s2, a)

def expandTrace(f1, f2, n, duplicate = False, start = None, hFun = None):
    writeSearchNode(f1, n.state, n.cost, n.heuristicCost,
                    dupStyle if duplicate else expandStyle, start, hFun)
    writeSearchNode(f2, n.state, n.cost, n.heuristicCost,
                    visitStyle, start, hFun)
    if n.parent:
        writeSearchNode(f2, n.parent.state, n.parent.cost,
                        n.parent.heuristicCost, expandStyle, start, hFun)
        writeSearchArc(f2, n.parent.state, n.state, n.action)

####   For HPN trees

def writePreamble(fileName):
    f = open(local.outDir + fileName + '_' + timeString()+'.dot', 'w') \
        if fileName else None
    if f:
        f.write('digraph G {\n')
        f.write('    ordering=out;\n')
        f.write('    node [fontname=HelveticaBold];\n')
    return f

def writeCoda(f):
    if f:
        f.write('}\n')
        f.close()

# For HPN trees
subtaskStyle = 'shape=box, style=filled, colorscheme=pastel16, color=4'
primitiveStyle = 'shape=box, style=filled, colorscheme=pastel16, color=3'
planGoalStyle = 'shape=box, style=filled, colorscheme=pastel16, color=2'
surpriseStyle = 'shape=box, style=filled, colorscheme=pastel16, color=5'
failureStyle = 'shape=box, style=filled, colorscheme=pastel16, color=1'
planStepArrowStyle = ''
refinementArrowStyle = 'style=dashed'
indent = '    '
arrow = ' -> '
eol = ';\n'  # terminate a line in a dot file
nl = '\\n'   # to put inside a label
def name(x): return '"' + x + '"'
def styleStr(x): return ' ['+x+']'

# Write and flush
def wf(f, s):
    f.write(s)
    f.flush()
    os.fsync(f.fileno())

dotSearchId = 0

def writeGoalNode(f, goal):
    global dotSearchId 
    goal.planNum = dotSearchId
    dotSearchId += 1    
    print 'Planning for goal', goal.planNum
    for fl in goal.fluents: print '    ', fl.prettyString()
    goalNodeName =  name(goal.goalName())    
    goalLabel = name('Goal '+str(goal.planNum)+nl+goal.prettyString(False,None))
    if f:
        wf(f, indent + goalNodeName + styleStr(planGoalStyle + ', label=' +\
                                               goalLabel) + eol)

def writeSurpriseNode(f, surpriseNodeName, prevIndex, currIndex):
    if f:
        nodeLabel = name('Surprise'+nl+\
                         'Upper step ' + str(currIndex-1))
                          # +nl+\'Got index ' + str(currIndex))
        wf(f, indent + surpriseNodeName + styleStr(surpriseStyle + ', label=' +\
                                               nodeLabel) + eol)

def writeFailureNode(f, nodeName, fluents):
    if f:
        g = State(fluents)
        nodeLabel = name('Failure.  Expected'+nl+\
                         g.prettyString(False, None))
        wf(f, indent + nodeName + styleStr(failureStyle + ', label=' +\
                                               nodeLabel) + eol)

def writeSubgoalRefinement(f, p, subgoal):
    writeGoalNode(f, subgoal)
    if f and p.steps[1][0] != top:  # if task 1 is top, this is a root
        subgoalNodeName =  name(subgoal.goalName())
        wf(f, indent + p.subtasksNodeName + arrow + subgoalNodeName + \
           styleStr(refinementArrowStyle) + eol)

surpriseCount = 0
def writeSurprise(f, p, prevIndex, currIndex):
    global surpriseCount
    surpriseCount += 1
    if not hasattr(p, 'subtasksNodeName'):
        p.subtasksNodeName = '"Top"'
    surpriseNodeName =  name(p.subtasksNodeName[1:-1]+':'+str(prevIndex)+':'\
                                 +str(currIndex)+':'+str(surpriseCount))
    writeSurpriseNode(f, surpriseNodeName, prevIndex, currIndex)
    if f and p.steps[1][0] != top:  # if task 1 is top, this is a root
        wf(f, indent + p.subtasksNodeName + arrow + surpriseNodeName + \
           styleStr(refinementArrowStyle) + eol)

failureCount = 0
def writeFailure(f, p, fluents):
    global failureCount
    failureCount += 1
    failNodeName =  name(p.subtasksNodeName[1:-1]+':'+str(surpriseCount))
    writeFailureNode(f, failNodeName, fluents)
    if f and p.steps[1][0] != top:  # if task 1 is top, this is a root
        wf(f, indent + p.subtasksNodeName + arrow + failNodeName + \
           styleStr(refinementArrowStyle) + eol)

def writeSubtasks(f, p, goal):
    tasks = [step[0] for step in p.steps[1:]]
    if f:
        goalNodeName =  name(goal.goalName())
        subtasksNodeName = name(','.join([t.uniqueStr() for t in tasks]))
        subtasksLabel = name('\\n'.join(\
         [str(i)+': '+t.prettyString(False) for (i, t) in enumerate(tasks)]))
        wf(f, indent + subtasksNodeName + \
              styleStr(subtaskStyle + ', label=' + subtasksLabel) + eol)
        wf(f, indent + goalNodeName + arrow + subtasksNodeName + \
              styleStr(planStepArrowStyle) + eol)
        p.subtasksNodeName = subtasksNodeName
        p.goalNodeName = goalNodeName
        for t in tasks:
            t.nodeName = subtasksNodeName

def writePrimRefinement(f, op):
    op.incrementNum()
    if f:
        tasksNodeName = op.nodeName
        primNodeName = name(op.uniqueStr()+'PRIM')
        primLabel = name(op.prettyString(False))
        wf(f, indent + primNodeName + \
           styleStr('label='+primLabel+primitiveStyle)+eol)
        wf(f, indent + tasksNodeName + arrow + primNodeName + \
           styleStr(refinementArrowStyle) + eol)

