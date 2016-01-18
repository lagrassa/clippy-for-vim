import math
import numpy as np
import windowManager3D as wm
from dist import GMU, MultivariateGaussianDistribution
from miscUtil import prettyString, diagToSq
from traceFile import tr, trAlways, debug
MVG = MultivariateGaussianDistribution

zeroObjectVarianceArray = [[0]*4]*4
hugeVarianceArray = [[100, 0, 0, 0],
                     [0, 100, 0, 0],
                     [0, 0, 100, 0],
                     [0, 0, 0, 100]]

identPoseTuple = (0.0, 0.0, 0.0, 0.0)

lostDist = GMU([(MVG(identPoseTuple, hugeVarianceArray, pose4 = True), 0.99)])

# Keep all the raw representation necessary for filtering
# After every belief update, it will have an instance variable
# pbs, which is the current PBS

# Eventually, the real state estimator will go here, and whenever it
# updates, it will generate a new pbs

class BeliefState:

    def __init__(self, pbs, domainProbs, awayRegion):
        objNames = pbs.objectBs.keys()
        self.pbs = pbs
        self.domainProbs = domainProbs
        self.awayRegion = awayRegion
        self.poseModeProbs = dict([(name , 0.99) \
                                   for name in objNames])
        # Share the poseModeProbs.
        self.pbs.poseModeProbs = self.poseModeProbs
        self.graspModeProb = {'left' : 0.99, 'right' : 0.99}
        self.relPoseVars = self.getRelPoseVars()

    def getRelPoseVars(self):
        # Take them out of the current pbs, going via the robot
        result = []
        objNames = self.pbs.objectBs.keys()
        for o1 in objNames:
            f1 = self.pbs.getPlacedObjBs()[o1].support.maxProbElt()
            var1 = self.pbs.getPlaceB(o1, f1).poseD.var
            for o2 in objNames:
                f2 = self.pbs.getPlacedObjBs()[o2].support.maxProbElt()
                var2 = self.pbs.getPlaceB(o1, f2).poseD.var
                result.append(((o1, o2),
                               tuple([a + b for (a, b) in zip(var1, var2)])))
        return dict(result)

    # Take the min of what we had and the current ones
    def updateRelPoseVars(self):
        # Can take out when done debugging!!
        self.pbs.draw(0.95, 'Belief')
        # Can take out when done debugging!!
        objNames = self.pbs.objectBs.keys()
        for o1 in objNames:
            f1 = self.pbs.getPlacedObjBs()[o1].support.mode()
            var1 = self.pbs.getPlaceB(o1, f1).poseD.var
            for o2 in objNames:
                old = self.relPoseVars[(o1, o2)]
                if o1 == o2:
                    rv = (0, 0, 0, 0)
                else:
                    f2 = self.pbs.getPlacedObjBs()[o2].support.mode()
                    var2 = self.pbs.getPlaceB(o1, f2).poseD.var
                    rv = tuple([a + b for (a, b) in zip(var1, var2)])
                if old == None:
                    # We just placed.
                    print 'New RPV', rv
                    print o1, var1
                    print o2, var2
                    raw_input('okay?')
                    self.relPoseVars[(o1, o2)] = rv
                else:
                    self.relPoseVars[(o1, o2)] = \
                              tuple([min(a, b) for (a, b) in zip(rv, old)])
                if o1 < o2:
                    print 'RPSD', o1, o2, \
                       prettyString(np.sqrt(self.relPoseVars[(o1, o2)][0]))

    def clearRelPoseVars(self, o):
        objNames = self.pbs.objectBs.keys()
        for otherO in objNames:
            if otherO == o: continue
            self.relPoseVars[(o, otherO)] = None
            self.relPoseVars[(otherO, o)] = None

    # Temporary hacks to keep all the types right
    def graspModeDist(self, obj, hand, face):
        if obj == 'none' or face == 'none':
            return GMU([(MVG(identPoseTuple, zeroObjectVarianceArray,
                             pose4 = True), 0.99)]) 
        else:
            if face == '*': face = None
            poseD = self.pbs.getGraspBForObj(obj, hand, face).poseD
            return GMU([(MVG(poseD.modeTuple(), diagToSq(poseD.var),
                             pose4 = True),
                        self.graspModeProb[hand])])

    def poseModeDist(self, obj, face):
        if obj == 'none' or face == 'none':
            return GMU([(MVG(identPoseTuple, zeroObjectVarianceArray,
                             pose4 = True), 0.99)])
        else:
            if face == '*': face = None
            poseD = self.pbs.getPlaceB(obj, face).poseD
            return GMU([(MVG(poseD.modeTuple(), diagToSq(poseD.var),
                             pose4 = True),
                         self.poseModeProbs[obj])])

    def draw(self, w = 'Belief'):
        s = '------------  Belief -------------\n'
        s += 'Conf:\n'
        for key in self.pbs.getConf().keys():
            s += '   ' + key + ' ' + prettyString(self.pbs.getConf()[key]) + '\n'
        gb = self.pbs.getGraspB
        gbl = gb('left')
        gbr = gb('right')
        s += 'Held Left: %s mode prob %s\n'%\
             (self.pbs.getHeld('left'), prettyString(self.graspModeProb['left']))
        s += '    Grasp type: %s\n'%(prettyString(gbl.grasp) if gbl else None)
        s += '    Grasp mean: %s\n'%(prettyString(gbl.poseD.meanTuple()) if (gbl and gbl.poseD) else None)
        s += '    Grasp stdev: %s\n'%(prettyStdev(gbl.poseD.varTuple())  if (gbl and gbl.poseD) else None)
        s += 'Held Right: %s mode prob %s\n'%\
             (self.pbs.getHeld('right'),prettyString(self.graspModeProb['right']))
        s += '    Grasp type: %s\n'%(prettyString(gbr.grasp) if gbr else None)
        s += '    Grasp mean: %s\n'%(prettyString(gbr.poseD.meanTuple()) if (gbr and gbr.poseD) else None)
        s += '    Grasp stdev: %s\n'%(prettyStdev(gbr.poseD.varTuple())  if (gbr and gbr.poseD) else None)
        s += 'Objects:\n'
        for (name, (fix, stuff)) in self.pbs.objectBs.items():
            s += name + '(fixed=%s)'%fix + '\n'
            s += '   prob: %s\n'%self.poseModeProbs[name]
            s += '   face: %s\n'%stuff.support
            s += '   pose: %s\n'%prettyString(stuff.poseD.meanTuple())
            s += '  stdev: %s\n'%prettyStdev(stuff.poseD.varTuple())
        s += '------------  Belief -------------\n'
        # wm.getWindow('Belief').capturing = True
        if debug('noTrace'):            # really always
            self.pbs.draw(0.9, w)
        trAlways(s, pause = False, draw=[(self.pbs, 0.9, w)], snap=[w])
        wm.getWindow('Belief').update()
        wm.getWindow('Belief').pause()
        # wm.getWindow('Belief').capturing = False


def prettyStdev(vt):
    return prettyString([math.sqrt(x) for x in vt])             

