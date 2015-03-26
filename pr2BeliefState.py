import math
import windowManager3D as wm
from dist import GMU, MultivariateGaussianDistribution, UniformDist, \
     DeltaDist, chiSqFromP
from miscUtil import prettyString
MVG = MultivariateGaussianDistribution

zeroObjectVarianceArray = [[0]*4]*4
identPoseTuple = (0.0, 0.0, 0.0, 0.0)

# Keep all the raw representation necessary for filtering
# After every belief update, it will have an instance variable
# pbs, which is the current PBS

# Eventually, the real state estimator will go here, and whenever it
# updates, it will generate a new pbs

class BeliefState:

    def __init__(self, pbs, domainProbs, awayRegion):
        self.pbs = pbs
        self.domainProbs = domainProbs
        self.awayRegion = awayRegion
        self.poseModeProbs = dict([(name , 1.0) \
               for name in (pbs.moveObjBs.keys() + pbs.fixObjBs.keys())])
        self.graspModeProb = {'left' : 1.0, 'right' : 1.0}

    # Temporary hacks to keep all the types right
    def graspModeDist(self, obj, hand, face):
        if obj == 'none' or face == 'none':
            return GMU([(MVG(identPoseTuple, zeroObjectVarianceArray), 1.0)])
        else:
            poseD = self.pbs.getGraspB(obj, hand, face).poseD
            return GMU([(MVG(poseD.mu.xyztTuple(), diagToSq(poseD.var)),
                        self.graspModeProb[hand])])

    def poseModeDist(self, obj, face):
        if obj == 'none' or face == 'none':
            return GMU([(MVG(identPoseTuple, zeroObjectVarianceArray), 1.0)])
        else:
            poseD = self.pbs.getPlaceB(obj, face).poseD
            return GMU([(MVG(poseD.mu.xyztTuple(), diagToSq(poseD.var)),
                         self.poseModeProbs[obj])])

    def draw(self, w = 'Belief'):
        print '------------  Belief -------------'
        print 'Conf:'
        for key in self.pbs.conf.keys():
            print '   ', key, prettyString(self.pbs.conf[key])
        gb = self.pbs.graspB
        gbl = gb['left']
        gbr = gb['right']
        print 'Held Left:', self.pbs.held['left'], \
                 'mode prob', prettyString(self.graspModeProb['left'])
        print '    Grasp type:', prettyString(gbl.grasp) \
               if gbl else None
        print '    Grasp mean:', prettyString(gbl.poseD.meanTuple()) \
               if (gbl and gbl.poseD) else None
        print '    Grasp stdev:', prettyStdev(gbl.poseD.varTuple()) \
               if (gbl and gbl.poseD) else None
        print 'Held Right:', self.pbs.held['right'], \
                 'mode prob', prettyString(self.graspModeProb['right'])
        print '    Grasp type:', prettyString(gbr.grasp) \
                         if gbr else None
        print '    Grasp mean:', prettyString(gbr.poseD.meanTuple()) \
                    if (gbr and gbr.poseD) else None
        print '    Grasp stdev:', prettyStdev(gbr.poseD.varTuple()) \
                    if (gbr and gbr.poseD) else None
        print 'Objects:'
        for (name, stuff) in self.pbs.moveObjBs.items() + \
                             self.pbs.fixObjBs.items():
            print name
            print '   prob:', self.poseModeProbs[name]
            print '   face:', stuff.support
            print '   pose:', prettyString(stuff.poseD.meanTuple())
            print ' stdev :', prettyStdev(stuff.poseD.varTuple())
        print '------------  Belief -------------'
        print self.pbs.draw(0.9, w)
        wm.getWindow(w).update()

def diagToSq(d):
    return [[(d[i] if i==j else 0.0) \
             for i in range(len(d))] for j in range(len(d))]

def prettyStdev(vt):
    return prettyString([math.sqrt(x) for x in vt])             

