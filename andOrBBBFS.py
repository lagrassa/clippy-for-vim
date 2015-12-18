import pdb

"""
And-or search, breadth-first, branch-and-bound
"""

import planGlobals as glob
from traceFile import debugMsg, debug, trAlways, tr

class Node:
    def __init__(self, state, lb = 0, ub = float('inf'), budget = float('inf'),
                 parent = None, children = None, cost = 0):
        self.state = state
        self.lb = lb
        self.ub = ub
        self.budget = budget
        self.parent = parent
        self.children = [] if children is none else children
        self.cost = cost
        self.pruned = False

class AndNode(Node):
    def updateBounds(self):
        (oldLb, oldUB) = (self.lb, self.ub)
        self.lb = self.cost + sum([c.lb for c in self.children])
        self.ub = self.cost + sum([c.ub for c in self.children])
        if oldLb != self.lb or oldUb != self.ub:
            self.parent.updateBounds()
        if oldUb != self.ub:
            for c in self.children:
                self.children.updateBudget()

    def updateBudget(self):
        oldB = self.budget
        self.budget = self.parent.ub
        if self.lb > self.budget:
            # Prune this baby!
            self.pruned = True
        elif self.budget != oldB:
            self.
            

class OrNode:
    def updateBounds(self):
        (oldLb, oldUB) = (self.lb, self.ub)
        self.lb = min([c.lb for c in self.children])
        self.ub = min([c.ub for c in self.children])
        if oldLb != self.lb or oldUb != self.ub:
            self.parent.updateBounds()
        if oldUb != self.ub:
            for c in self.children:
                self.children.updateBudget()


    def updateBudget(self, budget):
        pass
        
