from fbch import Operator, Fluent

class Launched(Fluent):
    def test(self, details):
        return details.launched

class CanLiftOff(Fluent):
    implicit = True
    conditional = True
    def test(self, details):
        # fuel > cargo mass

class CargoMass(Fluent):
    def test(self, details):
        return self.cargoMass

class Fuel(Fluent):
    def test(self, details):
        return self.fuel
    

blastOff = Operator('BlastOff', [],
                    {0 : {Launched([], False),
                          CanLiftOff([[]], True)}},
                    [({Launched([], True}}, {})])
                    
                    
