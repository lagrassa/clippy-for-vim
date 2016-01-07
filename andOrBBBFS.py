import pdb

"""
And-or search, breadth-first, branch-and-bound
"""

import planGlobals as glob
from traceFile import debugMsg, debug, trAlways, tr

class AONode:
    def __init__(self, state, lb = 0, ub = float('inf'), budget = float('inf'),
                 parent = None, children = None, cost = 0):
        self.state = state
        self.lb = lb # We know it will cost at least this much
        self.ub = ub # We know it will not cost more than this
        self.budget = budget # If it costs more than this, we have a better way
        self.parent = parent
        self.children = None
        self.cost = cost # only valid at and node
        self.pruned = False

    # Prune if lb > budget

    def __str__(self):
        start = str(self.state)+(': And' if self.nodeType == 'and' else ': Or')
        if self.pruned: start = 'XXX: ' + start
        if self.children == None:
            mid = '(Unexpanded)'
        else:
            mid = str(tuple([str(c.state) for c in self.children]))
            if self.nodeType == 'and':
                mid = mid + ' + ' + str(self.cost)
        end = ' [' + str(self.lb) + ', ' + str(self.ub) + ']'
        return start + mid + end

    def done(self):
        return self.pruned or self.ub == self.lb

    def unexpanded(self):
        return self.children == None

    def isAncestor(self, state):
        return state == self.state or \
               (self.parent and self.parent.isAncestor(state))

    __repr__ = __str__

    def __hash__(self):
        return str(self.state)

class AndNode(AONode):
    nodeType = 'and'
    def updateBounds(self):
        if self.done(): return
        (oldLb, oldUb) = (self.lb, self.ub)
        self.lb = self.cost + \
           (0 if self.unexpanded() else sum([c.lb for c in self.children]) )
        self.ub = self.cost + \
          (float('inf') if self.unexpanded() \
                    else sum([c.ub for c in self.children]))
        if (oldLb != self.lb or oldUb != self.ub) and self.parent:
            self.parent.updateBounds()
        else:
            # If we got here, it's because a child changed.  Only pass
            # down as budget when this is not going to go up any higher
            if self.children:
                self.updateChildBudgets()

    def updateChildBudgets(self):
        if self.done() or self.unexpanded(): return
        totalLb = sum([c.lb for c in self.children])
        print '  Rebudget', self, self.budget
        for c in self.children:
            ocb = c.budget
            c.budget = self.budget - self.cost - totalLb + c.lb
            if c.lb > c.budget:
                print 'XXX Pruned', c, c.budget
                c.pruned = True
                break
            elif c.budget != ocb:
                c.updateChildBudgets()

    def extractDerivation(self, depth = 0):
        print ' '*depth + str(self.state) + '(' + str(self.cost) + ')' + ' : '\
                 + str(self.lb)
        for c in self.children: c.extractDerivation(depth + 1)


class OrNode(AONode):
    nodeType = 'or'
    def updateBounds(self):
        if self.done(): return
        (oldLb, oldUb) = (self.lb, self.ub)
        if self.unexpanded():
            (self.lb, self.ub) = (0, float('inf'))
        elif self.children == []:
            (self.lb, self.ub) = (float('inf'), float('inf'))
        else:
            self.lb = min([c.lb for c in self.children])
            self.ub = min([c.ub for c in self.children]) 

        if (oldLb != self.lb or oldUb != self.ub) and self.parent:
            self.parent.updateBounds()
        else:
            # If we got here, it's because a child changed.  Only pass
            # down as budget when this is not going to go up any higher
            if self.children:
                self.updateChildBudgets()

    def updateChildBudgets(self):
        if self.done() or self.unexpanded(): return
        print '  Rebudget', self, self.budget
        minUb = min([c.ub for c in self.children])
        for c in self.children:
            if c.pruned: continue
            oldB = c.budget
            c.budget = minUb
            if c.lb > self.budget:
                # Prune this baby!
                print 'XXX Pruned', c, c.budget
                c.pruned = True
            elif c.budget != oldB:
                c.updateChildBudgets()

    def favoriteChild(self):
        # lowest upper bound
        ub, result = float('inf'), None
        for c in self.children:
            if c.ub < ub:
                ub, result = (c.ub, c)
        return result

    def extractDerivation(self, depth = 0):
        self.favoriteChild().extractDerivation(depth)

# Return top-level search node and a value
        
def search(initialState, andSuccessors, orSuccessors, staticEval):

    def findOrMakeNode(s, parent, andNode):
        if not s in visited:
            n = AndNode(s) if andNode else OrNode(s)
            n.parent = parent
            visited[s] = n
            if parent.isAncestor(s):
                n.ub = n.lb = float('inf')
            else:
                agenda.append(n)
            n.updateBounds()
        return visited[s]

    initNode = OrNode(initialState)
    # List of nodes to be expanded
    agenda = [initNode]
    # Dictionary from state to node
    visited = {initialState: initNode}
    while agenda and initNode.ub != initNode.lb:
        node = agenda.pop(0)
        print 'Expanding', node.state
        if node.pruned:
            continue
        if node.nodeType == 'or':
            (c, childStates) = orSuccessors(node.state)
        else:
            (c, childStates) = andSuccessors(node.state)
            node.cost = c
        node.children = [findOrMakeNode(s, node, node.nodeType == 'or') \
                             for s in childStates]
        node.updateBounds()

    return initNode


example1 = {'a' : (0, ['a1', 'a2', 'a3']),
            'a1' : (14, []),
            'a2' : (1, ['b', 'c']),
            'a3' : (7, ['c', 'd']),
            'b'  : (0, ['b1']),
            'c'  : (0, ['c1']),
            'd'  : (0, ['d1', 'd2']),
            'b1' : (5, ['a']),
            'c1' : (2, []),
            'd1' : (4, ['b']),
            'd2' : (1, ['c'])}

def test1():
    def succ(s, domain): return domain[s]
    
    top = search('a',
                  lambda s: succ(s, example1),
                  lambda s: succ(s, example1),
                  lambda s: float('inf'))

    top.extractDerivation()
    return top
                
            
# Make work if circular (path costs infinity)
