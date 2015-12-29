import pdb

"""
Procedures and classes for doing uniform cost search, always with
dynamic programming.  Becomes A* if a heuristic is specified.

LPK: Changed to store heuristic value in node to minimize the number
of times the heuristic function is called.

"""

from heapq import heappush, heappop

import planGlobals as glob
from traceFile import debugMsg, debug, trAlways, tr

class SearchNode:
    """A node in a search tree"""
    def __init__(self, action, state, parent, actionCost, heuristicCost = 0):
        self.state = state
        self.action = action
        self.actionCost = actionCost
        self.heuristicCost = heuristicCost
        """Action that moves from C{parent} to C{state}"""
        self.parent = parent
        if self.parent:
            self.cost = self.parent.cost + actionCost
            """The cost of the path from the root to C{self.state}"""
        else:
            self.cost = actionCost
        
    def path(self):
        """@returns: list of C{(action, state)} pairs from root to this node"""
        if self.parent is None:
            return [(self.action, self.state)]
        else:
            return self.parent.path() + [(self.action, self.state)]

    def costs(self):
        """@returns: list of C{(action, state)} pairs from root to this node"""
        if self.parent is None:
            return [self.cost]
        else:
            return self.parent.costs() + [self.cost]

    def inPath(self, s):
        """@returns: C{True} if state C{s} is in the path from here to
        the root"""
        if s == self.state:
            return True
        elif self.parent is None:
            return False
        else:
            return self.parent.inPath(s)

    def __repr__(self):
        if self.parent is None:
            return str(self.state)
        else:
            return repr(self.parent) + \
                   "-"+str(self.action)+"->"+str(self.state)

    __str__ = __repr__

# greedy ranges from 0 to 1;  0 is UCS, 0.5 i A*, 1 is best first

def search(initialState, goalTest, actions, successor,
           heuristic = lambda s: 0, maxNodes = 10000,
           visitF = None, expandF = None, hmax = float('inf'),
           prevExpandF = None, checkExpandF = None,
           multipleSuccessors = False,
           greedy = 0.5,
           verbose = False, printFinal = True, maxHDelta = None,
           maxCost = float('inf'),
           fail = True,
           returnFirstGoal = False):
        """
        @param initialState: root of the search
        @param goalTest: function from state to Boolean
        @param actions: function from state to list of actions
        @param successor: function from state and action to next state and cost
        @param heuristic: function from state to estimated cost to reach a goal;
            defaults to a heuristic of 0, making this uniform cost search
        @param maxNodes: kill the search after it expands this many nodes
        @returns: path from initial state to a goal state as a list of
               (action, state) tuples and a list of path costs from start to
               each state in the path
        """
        
        somewhatVerbose = verbose
        verbose = False

        hVals = {}
        def getH(state):
            if not state in hVals:
                hv = heuristic(state)
                hVals[state] = hv
            return hVals[state]

        startNode = SearchNode(None, initialState, None, 0,
                               getH(initialState))
        if goalTest(initialState):
            return startNode.path(), [0]
        if startNode.heuristicCost >= hmax:
            trAlways('Root has infinite heuristic value', pause = True)
            return None, None
            
        agenda = []
        count = 1
        countExpanded = 0
        heappush(agenda, (0, count, startNode))
        expanded = set([])
        while not agenda == [] and maxNodes > count:
            if verbose:
                print "agenda: ", agenda
            (hc, _, n) = heappop(agenda)
            if n.state in expanded:
                if prevExpandF: prevExpandF(n)
                if verbose:
                    print  "previously expanded: ", n.cost, n.state
                    raw_input('okay?')
                continue
            expanded.add(n.state)
            countExpanded += 1

            if checkExpandF:            # check legality on demand
                n = checkExpandF(n)      # possibly modify the node or set to None
                if n is None: continue

            if expandF: expandF(n)
            if verbose: print  "expanding node: ", n.cost, n.state
            if goalTest(n.state):
                # We're done!
                if somewhatVerbose or verbose:
                    print 'Found goal state', n.state
                if somewhatVerbose or verbose or printFinal:
                    print count, 'nodes visited;', \
                          countExpanded, 'states expanded;', \
                          'solution cost:', n.cost
                if getH(n.state) > 0:
                    debugMsg('heuristic', 'positive value at goal state',
                             n.state, getH(n.state))
                return n.path(), n.costs()
            if n.cost > maxCost:
                if True: #somewhatVerbose or verbose:
                    print "Search failed: exceeded max cost ", n.cost
                return None, None

            if getH(n.state)== 0:
                debugMsg('heuristic', 'zero value at non-goal state', n.state)

            applicableActions = actions(n.state)

            if len(applicableActions) == 0 and verbose:
                raw_input('no children')
            if somewhatVerbose or verbose:
                print "   ", n.cost, ":   expanding: ",  n
                print "         ", len(applicableActions), 'actions'
            if countExpanded % 10000 == 0:
                print 'cost', n.cost, 'nodes', count
            successors = set()
            for a in applicableActions:
                succ = successor(n.state, a)
                if not succ: continue
                if not multipleSuccessors:
                    succ = [succ]
                if verbose: print '        ', len(succ), 'successors'
                for (newS, cost) in succ:
                    if newS in successors: continue
                    else: successors.add(newS)
                    if verbose or somewhatVerbose:
                        print '           ', cost, newS
                    hValue = getH(newS)
                    if newS in expanded:
                        if prevExpandF and visitF:
                            newN = SearchNode(a, newS, n, cost, hValue)
                            visitF(n.state, n.cost, n.heuristicCost, a,
                                   newS, newN.cost, hValue)
                            prevExpandF(newN)
                            if maxHDelta and n.heuristicCost - hValue > maxHDelta:
                                print 'current h =', n.heuristicCost, 'new h =', hValue
                                raw_input('H delta exceeded')
                            if verbose:
                                print  "previously expanded: ", \
                                      newN.cost, newN.state
                                raw_input('okay?')
                    else:
                        count += 1
                        tr('h', hValue)
                        if hValue >= hmax: continue
                        newN = SearchNode(a, newS, n, cost, hValue)
                        if visitF: visitF(n.state, n.cost, n.heuristicCost, 
                                          a, newS, newN.cost, hValue)
                        if maxHDelta and n.heuristicCost - hValue > maxHDelta:
                            print 'current h =', n.heuristicCost, \
                              'new h =', hValue
                            raw_input('H delta exceeded')
                        if returnFirstGoal and goalTest(newS):
                            return newN.path(), newN.costs()
                        heappush(agenda,
                                 ((1 - greedy) * newN.cost + \
                                  greedy * hValue, count, newN))

        if somewhatVerbose or verbose or count >= maxNodes:
            print "Search failed after visiting ", count, " states."
        return None, None

# This is a generator version of search... it has multiple explicit
# state goals and a verification function.  The heuristic is a
# function of state and a single goal, the generalization to multiple
# goals is min over goals.  Once a path to a goal is found, it is
# removed from the goal set.  Once paths to all goals are found, the
# search stops.
def searchGen(initialState, goalStates, actions, successor,
              heuristic = lambda s: 0,
              goalTest = lambda x: True,
              maxNodes = 10000,
              maxExpanded = 1000,
              visitF = None, expandF = None, hmax = 1.0e3,
              prevExpandF = None,
              multipleSuccessors = False,
              greedy = 0.5,
              goalKey = lambda x: x,
              goalCostFn = lambda x: 0,
              verbose = False, printFinal = True, maxHDelta = None,
              fail = False):

    somewhatVerbose = verbose
    verbose = False

    goalStates = goalStates[:]          # we will be modifying this.
    hVals = {}
    def getH(state):
        if not state in hVals:
            # As goalStates change, this becomes
            # innaccurate... probably not worth fixing.
            hVals[state] = min(heuristic(state , g) + goalCostFn(g) for g in goalStates) \
                           if goalStates else 0.
        return hVals[state]

    gk = goalKey(initialState)
    # !! Not clear that the heuristic should depend on goal key only
    startNode = SearchNode(None, initialState, None, 0, getH(gk))
    if gk in goalStates:
        goalStates.remove(gk)
        if goalTest(gk):
            yield startNode.path(), [goalCostFn(gk)]
    agenda = []
    count = 1
    countExpanded = 0
    heappush(agenda, (0, count, startNode))
    expanded = set([])
    while not agenda == [] and goalStates and maxNodes > count and maxExpanded > countExpanded:
        if verbose:
            print "agenda: ",
            for (h, _, n) in agenda: print '    ', h, n.state
        (hc, _, n) = heappop(agenda)

        if n.state in expanded:
            if prevExpandF: prevExpandF(n)
            if verbose:
                print  "previously expanded: ", n.cost, n.state
                raw_input('okay?')
            continue
        expanded.add(n.state)
        countExpanded += 1
        if expandF: expandF(n)
        if verbose: print  "expanding node: ", n.cost, n.state
        gk = goalKey(n.state)
        if gk in goalStates:
            goalStates.remove(gk)
            if goalTest(gk):
                # raw_input('Goal?')
                # We're done! At least for this invocation...
                if somewhatVerbose or verbose:
                    print 'Found goal state', n.state
                if somewhatVerbose or verbose or printFinal:
                    print count, 'nodes visited;', \
                          countExpanded, 'states expanded;', \
                          'solution cost:', n.cost
                if debug('heuristic') and  getH(gk) > 0:
                        debugMsg('heuristic', 'positive value at goal state',
                                 n.state, getH(goalKey(n.state)))
                yield (n.path(), n.costs() + [goalCostFn(gk)])
            else:
                if debug('heuristic') and getH(gk)== 0:
                    debugMsg('heuristic', 'zero value at non-goal state', n.state)

        applicableActions = actions(n.state)

        if len(applicableActions) == 0 and verbose:
            raw_input('no children')
        if somewhatVerbose or verbose:
            print "   ", n.cost, ":   expanding: ",  n
            print "         ", len(applicableActions), 'actions'
        if countExpanded % 1000 == 0:
            print 'cost', n.cost, 'nodes', count
        successors = set()
        for a in applicableActions:
            succ = successor(n.state, a)
            if not multipleSuccessors:
                succ = [succ]
            if verbose: print '        ', len(succ), 'successors'
            for (newS, cost) in succ:
                if newS in successors: continue
                else: successors.add(newS)
                if verbose or somewhatVerbose:
                    print '           ', cost, newS
                gk = goalKey(newS)
                if newS in expanded:
                    if prevExpandF and visitF:
                        hValue = getH(gk)
                        newN = SearchNode(a, newS, n, cost+goalCostFn(gk), hValue)
                        visitF(n.state, n.cost, n.heuristicCost, a,
                               newS, newN.cost, hValue)
                        prevExpandF(newN)
                        if maxHDelta and n.heuristicCost - hValue > maxHDelta:
                            print 'current h =', n.heuristicCost, 'new h =', hValue
                            raw_input('H delta exceeded')
                        if verbose:
                            print  "previously expanded: ", \
                                  newN.cost, newN.state
                            raw_input('okay?')
                else:
                    count += 1
                    hValue = getH(gk)
                    if verbose: print '        h', hValue
                    if hValue >= hmax: continue
                    newN = SearchNode(a, newS, n, cost, hValue)
                    if visitF: visitF(n.state, n.cost, n.heuristicCost, 
                                      a, newS, newN.cost, hValue)
                    if maxHDelta and n.heuristicCost - hValue > maxHDelta:
                        print 'current h =', n.heuristicCost, 'new h =', hValue
                        raw_input('H delta exceeded')
                    heappush(agenda,
                             ((1 - greedy) * newN.cost + \
                              greedy * hValue, count, newN))

    if somewhatVerbose or verbose or count >= maxNodes or countExpanded >= maxExpanded:
        print "Search failed after visiting ", count, " states."
        # raw_input('Too many nodes')
        if fail: pdb.set_trace()

    yield None, None

