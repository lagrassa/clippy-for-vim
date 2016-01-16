import pdb
import dist
from operator import mul
import planGlobals as glob
from traceFile import debugMsg, debug

# Constrained Factored Filter
# - Set of variables, each of which is an independent filter
# - Set of constraints, each of which contains:
#     - list of variables
#     - potential function: maps an assignment of values of the vars to {0, 1)

class CFF:
    def __init__(self, variables, constraints):
        # variables is a dictionary from names to filters
        # constraints is a list of CFFConstraint
        self.varNames = variables.keys()
        self.variables = variables
        self.constraints = constraints

    def computeMarginals(self):
        self.marginals = dict([(name, self.computeMarginal(name)) \
                               for name in self.varNames])

    def computeMarginal(self, varName):
        # Assuming all variables are relevant (connected to varName by
        # some constraint)
        # But then it's super-loopy
        # Decide some links to break

        # One idea:
        # Define strength of constraint (based on prior) C to variable V
        # as the difference between p(V | ov) and the message from C to V
        # Score them all Delete weakest until we have a tree.  Don't
        # want minimum spanning tree, because we don't care if it's a
        # spanning tree (can delete whole variables).

        # First order:  just take constraints that mention varName
        prior = self.variables[varName]
        constraints = self.getFirstOrderTree(varName)
        vMsgs = [self.computeMsg(c, varName) for c in constraints]
        marg = dist.DDist(dict([(a, reduce(mul,[vMsg[a] for vMsg in vMsgs],1) *\
                                    prior.prob(a)) \
                            for a in prior.support()]))
        marg.normalize()

        debugMsg('cff', (varName, 'prior', prior), ('posterior', marg))
        
        return marg

    def getFirstOrderTree(self, varName):
        return [c for c in self.constraints if varName in c.variables]

    def computeMsg(self, c, vName):
        # message from constraint to variable
        # dictionary
        # sum out other vars of
        #     product of messages from other vars to this constraint

        return dict([(a, sum([c.fun(*assignment) * \
                       reduce(mul, [self.variables[nn].prob(nv) for \
                             (nn, nv) in zip(c.variables, assignment) \
                             if nn != vName]) for \
                     assignment in self.genAssignments(c.variables, vName,a)]))\
                for a in self.variables[vName].support()])

    def genAssignments(self, varNames, fixedName, fixedVal):
        if len(varNames) == 0:
            return [[]]
        else:
            rest = self.genAssignments(varNames[1:], fixedName, fixedVal)
            if varNames[0] == fixedName:
                return [[fixedVal] + r for r in rest]
            else:
                return [[v] + r for v in self.variables[varNames[0]].support()\
                                for r in rest]
        
class CFFConstraint:
    # 1 is good
    def __init__(self, variables, fun, funName = 'chi'):
        self.variables = variables
        self.fun = fun
        self.funName = funName

    def __str__(self):
        return self.funName+str(self.variables)

def test1():
    cff = CFF({'a' : dist.DDist({1 : .1, 2 : .7, 3 : .2}),
               'b' : dist.DDist({1 : .4, 2 : .4, 3 : .2})},
              [CFFConstraint(('a', 'b'),
                             lambda a, b: 0 if a == b else 1)])
    cff.computeMarginals()
    for i in cff.marginals.items():
        print i

def test2():
    cff = CFF({'a' : dist.DDist({1 : .1, 2 : .7, 3 : .2}),
               'b' : dist.DDist({1 : .4, 2 : .4, 3 : .2}),
               'occ1' : dist.DDist({True : .9, False : .1}),
               'occ2' : dist.DDist({False : .9, True : .1}),
               'occ3' : dist.DDist({True : .9, False : .1})},
               [CFFConstraint(('a', 'b'),
                              lambda a, b: 0 if a == b else 1),
                CFFConstraint(('a', 'occ1'),
                              lambda a, o: 0 if (a == 1 and not o) else 1),
                CFFConstraint(('a', 'occ2'),
                             lambda a, o: 0 if (a == 2 and not o) else 1),
                CFFConstraint(('a', 'occ3'),
                               lambda a, o: 0 if (a == 3 and not o) else 1),
                CFFConstraint(('b', 'occ1'),
                              lambda b, o: 0 if (b == 1 and not o) else 1),
                CFFConstraint(('b', 'occ2'),
                              lambda b, o: 0 if (b == 2 and not o) else 1),
                CFFConstraint(('b', 'occ3'),
                              lambda b, o: 0 if (b == 3 and not o) else 1)])
    cff.computeMarginals()
    for i in cff.marginals.items():
        print i

