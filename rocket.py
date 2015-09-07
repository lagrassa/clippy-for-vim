import copy
from fbch import Operator, Fluent, Function, simplifyCond, getMatchingFluents

class Launched(Fluent):
    predicate = 'Launched'
    def test(self, details):
        return details.launched

class CanLiftOff(Fluent):
    implicit = True
    conditional = True
    predicate = 'CanLiftOff'
    def test(self, details):
        (cond,) = self.args
        pbs = details.updateFromCond(cond)
        return pbs.fuel > pbs.cargoMass and pbs.fuel < 1000

class CargoMass(Fluent):
    predicate = 'CargoMass'
    def test(self, details):
        return self.cargoMass

class Contains(Fluent):
    predicate = 'Contains'
    def test(self, details):
        (obj,) = self.args
        return obj in details.cargo
    
class Fuel(Fluent):
    predicate = 'Fuel'
    def test(self, details):
        return details.fuel

class RocketBel:
    def __init__(self):
        self.cargoMass = 0
        self.cargo = []
        self.fuel = 0
        self.launched = False

    # Cargo mass dictionary
    cmd = {'exp1' : 100, 'exp2' : 200}
        
    def objectMass(self, o):
        return self.cmd[o]

    def updateFromCond(self, cond):
        pbs = copy.copy(self)
        for c in cond:
            if c.predicate == 'Fuel':
                pbs.fuel = c.value
            elif c.predicate == 'CargoMass':
                raw_input('not sure how to handle cargo mass in condition')
                pbs.cargoMass = c.value
            elif c.predicate == 'Contains':
                pbs.cargoMass += self.objectMass(c.args[0])
        return pbs

class PrevMass(Function):
    @staticmethod    
    def fun(args, goal, start):
        (obj, postMass) = args
        return [[postMass - start.objectMass(obj)]]

class AddFuelCond(Function):
    @staticmethod    
    def fun(args, goal, start):
        (postCond, f) = args
        cond = Fuel([], f)
        return [[simplifyCond(postCond, [cond])]]

class GenFuel(Function):
    @staticmethod
    # How much fuel to load?  
    def fun(args, goalFluents, start):
        (postCond,) = args
        # Don't be inconsistent with goal
        fb = getMatchingFluents(goalFluents, Fuel([], 'F'))
        # Could fail right away if postCond already has Fuel fluent in
        # it (this won't change anything.
        if len(fb) > 0:
            f = fb[0][1]['F']  # ugly;  get the binding for 'F'
            return [[f]]   # no other values allowed
        else:
            # simple generator; could also look in the goal and start
            # to guess how much to load
            return [[50], [100], [150], [500], [1000]]
    
canLiftOff = Operator('CanLiftOff', ['F', 'PreCond', 'PostCond'],
                       {0 : {Fuel([], 'F'),
                             CanLiftOff(['PreCond'], True)}},
                       [({CanLiftOff(['PostCond'], True)}, {})],
                       [GenFuel(['F'], ['PostCond']),
                        AddFuelCond(['PreCond'], ['PostCond', 'F'])])

blastOff = Operator('BlastOff', [],
                    {0 : {Launched([], False),
                          CanLiftOff([[]], True)}},
                    [({Launched([], True)}, {})])

loadFuel = Operator('LoadFuel', ['F'],
                    {0 : {Launched([], False)}},
                    [({Fuel([], 'F')}, {})])

loadCargo = Operator('LoadCargo', ['O', 'PreM', 'PostM'],
                     {0 : {Launched([], False)}},
                     [({CargoMass([], 'PostM')},
                       {0: {CargoMass([], 'PreM')}}),
                      ({Contains(['O'], True)}, {})],
                     [PrevMass(['PreM'], ['O', 'PostM'])])


######################################################################
#                         Planning
######################################################################                    
ops = [blastOff, loadFuel, loadCargo, canLiftOff]

def runTest(init, goal):
    p = planBackward(init, goal, ops, fileTag = 'rocket', maxNodes = 100)
    if p:
        planObj = makePlanObj(p, init)
        planObj.printIt(verbose = False)

def rt1():
    init = State([], RocketBel())
    goal = State([Contains(['exp1'], True)])
    runTest(init, goal)

def rt2():
    init = State([], RocketBel())
    goal = State([Launched([], True)])
    runTest(init, goal)

def rt3():
    init = State([], RocketBel())
    goal = State([Launched([], True),
                  Contains(['exp1'], True)])
    runTest(init, goal)
    
print 'rocket loaded'
