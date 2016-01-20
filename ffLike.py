import pdb

"""
And-or search, breadth-first, branch-and-bound
Return a set of action instances and total cost
"""

import planGlobals as glob
from traceFile import debugMsg, debug, trAlways, tr
from miscUtil import squashSets, argminWithVal, argmin, prettyString, timeString
import local

maxVisitedNodes = 500
terminationPct = 0.05

class AONode:
    def __init__(self, state, lb = 0, ub = float('inf'), budget = float('inf'),
                 parent = None, children = None, cost = 0):
        self.state = state
        self.lb = lb # We know it will cost at least this much
        self.ub = ub # We know it will not cost more than this
        self.budget = budget # If it costs more than this, we have a better way
        self.parents = {parent} if parent is not None else set()
        self.children = None
        self.expanded = False
        self.cost = cost # only valid at and node
        self.ubActs = None
        self.lbActs = None
        assert self.lb <= self.ub
        self.root = False
        self.cyclic = False

    def __str__(self):
        start = str(self.state)+(': And' if self.nodeType == 'and' else ': Or')
        if self.unexpanded():
            mid = '(Unexpanded)'
        elif self.cyclic:
            mid = '(Cyclic)'
        else:
            mid = str(tuple([str(c.state) for c in self.children]))
            if self.nodeType == 'and':
                mid = mid + ' + ' + str(self.cost)
        end = ' [' + str(self.lb) + ', ' + str(self.ub) + ']'
        return start + mid + end

    def done(self):
        return self.ub == self.lb or self.budget < self.lb

    def unexpanded(self):
        return not self.expanded

    def boundsNotTight(self):
        diff = self.ub - self.lb
        return diff == float('inf') or float(diff) / self.ub > terminationPct

    def isAncestor(self, state, c = 0):
        if c >= 100:
            print '100 deep in isAncestor'
            print state
            raw_input('go?')
        if state == self.state:
            return True
        elif self.root:
            return False
        else:
            return any([p.isAncestor(state, c+1) for p in self.parents])

    __repr__ = __str__

    def __hash__(self):
        return hash(str(self.state))

    def writeAOTree(self, fp, closed):
        if self in closed: return
        style = andStyle if self.nodeType == 'and' else orStyle
        if self.budget < self.lb:
            style = prunedStyle
        if self.cyclic:
            style = cyclicStyle
        if self.unexpanded():
            style = unexpandedStyle
        costStr = prettyString((self.lb, self.ub))
        if self.nodeType == 'and':
            costStr = costStr + '\nOp cost = ' + prettyString(self.cost)
        writeHNode(fp, self, costStr, style)
        closed.add(self)
        if self.children != None:
            if len(self.children) > 0:
                bestUBChild, _ = argminWithVal(self.children, lambda x: x.ub)
            for c in self.children:
                c.writeAOTree(fp, closed)
                bold = self.nodeType == 'or' and c == bestUBChild
                # Hack for now to try to avoid horrible bottom node
                if c.children or c.unexpanded():
                    writeSearchArc(fp, self, c, bold = bold)

    def updateChildBudgets(self):
        if self.lb == self.ub or self.unexpanded(): return
        for c in self.children:
            ocb = c.budget
            c.updateBudget()
            if c.lb > c.budget:
                pass
            elif c.budget != ocb:
                c.updateChildBudgets()

    def updateBudget(self):
        self.budget = max([p.getBudget(self) for p in self.parents])

    def reEval(self):
        if self.children != None:
            for c in self.children: c.reEval()
        self.eval()

class AndNode(AONode):
    nodeType = 'and'

    def eval(self):
        if self.unexpanded():
            self.lbActs = {self}
            self.lb = self.cost
        else:
            childLBs = [c.lb for c in self.children]
            childLBActionSets = [c.lbActs for c in self.children \
                                 if not c.unexpanded()]
            if not float('inf') in childLBs:
                self.lbActs = squashSets(childLBActionSets)
                self.lbActs.add(self)
                self.lb = sum([a.cost for a in self.lbActs])
                assert self.lb < float('inf')
            else:
                self.lbActs = None
                self.lb = float('inf')
            childUBActionSets = [c.ubActs for c in self.children]
            if not None in childUBActionSets:
                self.ubActs = squashSets(childUBActionSets)
                self.ubActs.add(self)
                self.ub = sum([a.cost for a in self.ubActs])
            else:
                self.ubActs = None
                self.ub = float('inf')

        # We could be in trouble here!  Each child has lb < ub.  But
        # because we are taking union of action sets, it's possible
        # that the unions fail the following test.
        if self.ub < self.lb:
            self.lb = self.ub
            self.lbActs = self.ubActs
    
    def updateBounds(self):
        if self.done(): return
        (oldLb, oldUb) = (self.lb, self.ub)

        self.eval()

        if (oldLb != self.lb or oldUb != self.ub):
            for p in self.parents: p.updateBounds()
            tr('ffl', 'Updated bounds', (oldLb, oldUb), '->', (self.lb, self.ub))
        else:
            # If we got here, it's because a child changed.  Only pass
            # down as budget when this is not going to go up any higher
            if self.expanded and self.children is not None:
                self.updateChildBudgets()

        assert self.lb <= self.ub

    def getBudget(self, child):
        totalLb = sum([c.lb for c in self.children])
        if totalLb == float('inf'):
            return 0
        b = self.budget - self.cost - totalLb + child.lb

        if debug('ffl') and b < float('inf'):
            print 'Budget for', child.state, 'is', b
            print 'self.budget - self.cost - totalLb + child.lb'
            print '   ', self.budget, '-', self.cost, '-', totalLb, '+', child.lb
        return b


class OrNode(AONode):
    nodeType = 'or'

    def eval(self):
        if self.unexpanded():
            (self.lb, self.ub) = (0, float('inf'))
        elif self.children == []:
            (self.lb, self.ub) = (float('inf'), float('inf'))
        else:
            bestUBChild, minUB = argminWithVal(self.children, lambda x: x.ub)
            bestLBChild, minLB = argminWithVal(self.children, lambda x: x.lb)
            self.ub = minUB
            self.lb = minLB
            self.ubActs = bestUBChild.ubActs
            self.lbActs = bestLBChild.lbActs
        assert self.lb <= self.ub

    def updateBounds(self):
        if self.done(): return
        (oldLb, oldUb) = (self.lb, self.ub)

        self.eval()
        
        if (oldLb != self.lb or oldUb != self.ub):
            for p in self.parents: p.updateBounds()
            tr('ffl', 'Updated bounds', (oldLb, oldUb), '->', (self.lb, self.ub))
        else:
            # If we got here, it's because a child changed.  Only pass
            # down as budget when this is not going to go up any higher
            if self.children:
                self.updateChildBudgets()
        assert self.lb <= self.ub

    def getBudget(self, child):
        b = min([c.ub for c in self.children])
        if debug('ffl') and b < float('inf'):
            print 'Budget for', child.state, 'is', b
            print 'min([c.ub for c in self.children])'
            print '   min', [c.ub for c in self.children]
        return b

# Return top-level search node, with a set of actions and a value

# To really find the minimal action set, we would have to solve a more
# difficult optimization problem;

# Dictionary from (state, nodeType) to node
visited = {}

# Traverse tree looking for a node in the least-cost tree that is not
# done and that is on the agenda.  Do BFS traversal.
def getAWinner(topNode, agenda):
    aa = [topNode]
    while True:
        n = aa.pop(0)
        if n in agenda:
            agenda.remove(n)
            return n
        else:
            if n.done():
                pass
            elif n.nodeType == 'and':
                aa.extend(n.children)
            else:
                aa.append(argmin(n.children, lambda x: x.lb))
        if len(aa) == 0:
            return None
            #return agenda.pop(0)

def search(initialState, andSuccessors, orSuccessors, staticEval,
           writeFile = False, initNodeType = 'or'):

    def findOrMakeNode(s, parent):
        nodeType = 'and' if parent.nodeType == 'or' else 'or'
        key = (s, nodeType)
        if not key in visited:
            tr('ffl', 'New node', key)
            n = AndNode(s) if nodeType == 'and' else OrNode(s)
            visited[key] = n
            agenda.append(n)
            n.updateBounds()
            assert n.lb <= n.ub
        else:
            tr('ffl', 'Cache hit', key)
            n = visited[key]
            if n.lb < n.ub and not key in visitedThisTime:
                tr('ffl', 'Revisiting')
                agenda.append(n)

        visitedThisTime.add(key)

        # Check for cycles
        if any([p.isAncestor(s) for p in parent.parents]):
            # If parent is 'and', then it's infeasible
            if nodeType == 'or':
                parent.ub = parent.lb = float('inf')
                for p in parent.parents: p.updateBounds()
                parent.cyclic = True
                parent.lbActs = None
            return None
        
        n.parents.add(parent)
        n.budget = float('inf')
        parent.updateChildBudgets()
        return n

    try:
        initNode = AndNode(initialState) if initNodeType == 'and' \
                                               else OrNode(initialState)
        initNode.root = True
        initNode.cost = 0
        agenda = [initNode]         # List of nodes to be expanded
        # Dictionary from state to node
        visited[(initialState, initNodeType)] = initNode
        visitedThisTime = {(initialState, initNodeType)} # Set

        while agenda and initNode.boundsNotTight() and \
                                      len(visitedThisTime) < maxVisitedNodes:
            if len(visited) % 100 == 0: print 'v', len(visited)
            node = getAWinner(initNode, agenda)
            if node is None:
                writeAOTree(initNode)
                print 'heuristic ran out of winners but bounds open'
                break
            if node.done():  continue
            tr('ffl', 'Expanding', node.state)
            if node.nodeType == 'or':
                (c, childStates) = orSuccessors(node.state); print 'o',
            else:
                (c, childStates) = andSuccessors(node.state); print 'a',
                node.cost = c
            node.children = [n for n in \
                             [findOrMakeNode(s, node) for s in childStates] \
                             if n is not None]
            node.expanded = True
            node.updateBounds()
        if len(visitedThisTime) == maxVisitedNodes:
            writeAOTree(initNode)
            print 'heuristic ran out of nodes'
        if initNode.ub == float('inf'):
            writeAOTree(initNode)
            print 'infinite heuristic'
    finally:
        if writeFile: writeAOTree(initNode)
    return initNode

# Doesn't make sense to prune OR node

def writeAOTree(node):
    fp = openHFile()
    node.writeAOTree(fp, set())
    closeHFile(fp)

example1 = {'a' : (0, ['a1', 'a2', 'a3', 'a4']),
            'a1' : (14, []),
            'a2' : (1, ['b', 'c']),
            'a3' : (7, ['c', 'd']),
            'b'  : (0, ['b1']),
            'c'  : (0, ['c1']),
            'd'  : (0, ['d1', 'd2']),
            'b1' : (5, ['a']),
            'c1' : (2, []),
            'd1' : (4, ['b']),
            'd2' : (1, ['c']),
            'a4' : (1, ['e', 'f', 'g']),
            'e'  : (0, ['e1']),
            'e1' : (3, []),
            'f'  : (0, ['f1']),
            'f1' : (11, []),
            'g'  : (0, ['g1']),
            'g1' : (11, [])}
            
'''
c : c1 2
d : {d1, c1} 1 + 2 = 3
b : {b1, a3} 5 + 7 + 2 + 1 = 15
a : {a3, c1, d1} 7 + 2 + 1 = 10
'''        

def test1(target = 'a'):
    def succ(s, domain): return domain[s]
    
    top = search(target,
                  lambda s: succ(s, example1),
                  lambda s: succ(s, example1),
                  lambda s: float('inf'),
                  writeFile = True)

    print top
    print 'Action set:'
    for a in top.ubActs:
        print '   ', a.state, a.cost
    return top

def nStr(n):
    return (n.state.uniqueStr() if hasattr(n.state, 'uniqueStr') else \
                    str(n.state)) + n.nodeType
            
def writeHNode(f, n, c, styleStr):
    # TODO: fbch specific hack
    f.write('    "'+nStr(n) + styleStr +\
             prettyString(c, True) + '\\n'+prettyString(n.state, False)+'"];\n')

def writeSearchArc(f, n1, n2, op = None, bold = False):
    # TODO : LPK : fbch-specific hack
    if hasattr(n2.state, 'operator'): op = n2.state.operator
    penStr = 'penwidth=3,' if bold else ''
    opStr = prettyString(op, False) if op is not None else ''
    f.write('    "' + nStr(n1)+'" -> "'+ \
                      nStr(n2)+'"['+penStr+'label="'+ \
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

# blue
cyclicStyle = \
  '" [shape=box, style=filled, colorscheme=pastel16, color=2, label="Cycle'
# pink
initStyle = \
  '" [shape=box, style=filled, colorscheme=pastel16, color=8, label="Init cost='
# yellow
cacheStyle = \
 '" [shape=box, style=filled, colorscheme=pastel16, color=6, label="Cache cost='
l2CacheStyle = \
 '" [shape=box, style=filled, colorscheme=pastel16, color=6, label="L2 Cache cost='
# purple
andStyle = \
  '" [shape=box, style=filled, colorscheme=pastel16, color=4, label="'
# green
orStyle = \
  '" [shape=box, style=filled, colorscheme=pastel16, color=3, label="'
# orange
specialStyle = \
 '" [shape=box, style=filled, colorscheme=pastel16, color=5, label="Special cost='
# brown
leafStyle = \
  '" [shape=box, style=filled, colorscheme=pastel19, color=7, label="Leaf cost='
# red
prunedStyle = \
  '" [shape=box, style=filled, colorscheme=pastel19, color=1, label="Pruned '
# gray
unexpandedStyle = \
  '" [shape=box, style=filled, colorscheme=pastel19, color=9, label="Unexpanded '
# clear
bbStyle = \
  '" [shape=box, label="Pruned cost='

print 'Loaded ffLike.py'
