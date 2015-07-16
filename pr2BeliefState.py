import math
import windowManager3D as wm
from dist import GMU, MultivariateGaussianDistribution
from miscUtil import prettyString
from traceFile import tr
MVG = MultivariateGaussianDistribution

zeroObjectVarianceArray = [[0]*4]*4
hugeVarianceArray = [[100, 0, 0, 0],
                     [0, 100, 0, 0],
                     [0, 0, 100, 0],
                     [0, 0, 0, 100]]

identPoseTuple = (0.0, 0.0, 0.0, 0.0)

lostDist = GMU([(MVG(identPoseTuple, hugeVarianceArray), 0.99)])

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
        self.poseModeProbs = dict([(name , 0.99) \
               for name in (pbs.moveObjBs.keys() + pbs.fixObjBs.keys())])
        self.graspModeProb = {'left' : 0.99, 'right' : 0.99}

    # Temporary hacks to keep all the types right
    def graspModeDist(self, obj, hand, face):
        if obj == 'none' or face == 'none':
            return GMU([(MVG(identPoseTuple, zeroObjectVarianceArray), 0.99)])
        else:
            if face == '*': face = None
            poseD = self.pbs.getGraspB(obj, hand, face).poseD
            return GMU([(MVG(poseD.mu.xyztTuple(), diagToSq(poseD.var)),
                        self.graspModeProb[hand])])

    def poseModeDist(self, obj, face):
        if obj == 'none' or face == 'none':
            return GMU([(MVG(identPoseTuple, zeroObjectVarianceArray), 0.99)])
        else:
            if face == '*': face = None
            poseD = self.pbs.getPlaceB(obj, face).poseD
            return GMU([(MVG(poseD.mu.xyztTuple(), diagToSq(poseD.var)),
                         self.poseModeProbs[obj])])

    '''
    def drawOld(self, w = 'Belief'):
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
    '''

    def draw(self, w = 'Belief'):
        s = '------------  Belief -------------\n'
        s += 'Conf:\n'
        for key in self.pbs.conf.keys():
            s += '   ' + key + ' ' + prettyString(self.pbs.conf[key]) + '\n'
        gb = self.pbs.graspB
        gbl = gb['left']
        gbr = gb['right']
        s += 'Held Left: %s mode prob %s\n'%\
             (self.pbs.held['left'], prettyString(self.graspModeProb['left']))
        s += '    Grasp type: %s\n'%(prettyString(gbl.grasp) if gbl else None)
        s += '    Grasp mean: %s\n'%(prettyString(gbl.poseD.meanTuple()) if (gbl and gbl.poseD) else None)
        s += '    Grasp stdev: %s\n'%(prettyStdev(gbl.poseD.varTuple())  if (gbl and gbl.poseD) else None)
        s += 'Held Right: %s mode prob %s\n'%\
             (self.pbs.held['right'],prettyString(self.graspModeProb['right']))
        s += '    Grasp type: %s\n'%(prettyString(gbr.grasp) if gbr else None)
        s += '    Grasp mean: %s\n'%(prettyString(gbr.poseD.meanTuple()) if (gbr and gbr.poseD) else None)
        s += '    Grasp stdev: %s\n'%(prettyStdev(gbr.poseD.varTuple())  if (gbr and gbr.poseD) else None)
        s += 'Objects:\n'
        for (name, stuff) in self.pbs.moveObjBs.items() + \
                             self.pbs.fixObjBs.items():
            s += name + '\n'
            s += '   prob: %s\n'%self.poseModeProbs[name]
            s += '   face: %s\n'%stuff.support
            s += '   pose: %s\n'%prettyString(stuff.poseD.meanTuple())
            s += '  stdev: %s\n'%prettyStdev(stuff.poseD.varTuple())
        s += '------------  Belief -------------\n'
        tr('B', s)
        tr('B', draw=[(self.pbs, 0.9, w)], snap=[w])
        wm.getWindow(w).update()

def diagToSq(d):
    return [[(d[i] if i==j else 0.0) \
             for i in range(len(d))] for j in range(len(d))]

def prettyStdev(vt):
    return prettyString([math.sqrt(x) for x in vt])             

