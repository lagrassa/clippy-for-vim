import sys
import pdb
import time
from time import sleep
import planGlobals as glob
from traceFile import debug, debugMsg
from pr2Util import removeDuplicateConfs
import hu
import windowManager3D as wm
from random import random

if glob.useMPL:
    sys.path.append("/Users/tlp/MacDocuments/Research/git/MCR/")
    from mpl.algorithm_runner import runAlgorithm

############################################################
#  TO DO:
#  Make sure that the nearest nodes in the tree are marked as not "inter"
#  so that we have all the key points in the trajectory if we ignore "inter" nodes.
############################################################

# For now, use configurations until they are added to the tree as Nodes.

maxStopNodeSteps = 10

class RRT:
    def __init__(self, pbs, prob, initConf, goalConf, allowedViol, moveChains):
        if debug('rrt'): print 'Setting up RRT'
        ic = initConf; gc = goalConf
        self.pbs = pbs
        self.prob = prob
        self.robot = pbs.getRobot()
        self.moveChains = moveChains
        self.allowedViol = allowedViol
        self.Ta = Tree(ic, pbs, prob, True, allowedViol, moveChains)
        if goalConf:
            self.Tb = Tree(gc, pbs, prob, False, allowedViol, moveChains)

    def randConf(self):
        return self.robot.randomConf(self.moveChains)

    def swapTrees(self):
        if self.Ta.size > self.Tb.size:
            self.Ta, self.Tb = self.Tb, self.Ta

    # the qx variables denote confs; the nx variable denote nodes
    def buildBiTree(self, K=1000):
        """Builds the RRT and returns either a pair of nodes (one in each tree)
        with a common configuration or FAILURE."""
        if debug('rrt'): print 'Building BiRRT'
        n_new = self.Ta.stopNode(self.Tb.root.conf, self.Ta.root)
        if eqChains(n_new.conf, self.Tb.root.conf, self.moveChains):
            if debug('rrt'): print 'Found direct path'
            na_new = self.Ta.addNode(n_new.conf, self.Ta.root)
            nb_new = self.Tb.addNode(n_new.conf, self.Tb.root)
            return (na_new, nb_new)
        for i in range(K):
            if debug('rrt'):
                if i % 100 == 0: print i
            q_rand = self.randConf()
            na_near = self.Ta.nearest(q_rand)
            # adjust continuous angle values
            q_rand = self.robot.normConf(q_rand, na_near.conf)
            na_new = self.Ta.stopNode(q_rand, na_near)
            if not na_new is na_near:
                nb_near = self.Tb.nearest(na_new.conf)
                nb_new = self.Tb.stopNode(na_new.conf, nb_near)
                if eqChains(na_new.conf, nb_new.conf, self.moveChains):
                    if debug('rrt'):
                        print 'BiRRT: Goal reached in' +\
                              ' %s iterations' %str(i)
                    return (na_new, nb_new)
            self.swapTrees()
        if debug('rrt'):
            print '\nBiRRT: Goal not reached in' + ' %s iterations\n' %str(K)
        return 'FAILURE'

    # the qx variables denote confs; the nx variable denote nodes
    def buildTree(self, goalTest, K=1000):
        """Builds the RRT and returns either a node or FAILURE."""
        if debug('rrt'): print 'Building RRT'
        if goalTest(self.Ta.root.conf):
            return self.Ta.root
        for i in range(K):
            if debug('rrt'):
                if i % 100 == 0: print i
            q_rand = self.randConf()
            na_near = self.Ta.nearest(q_rand)
            # adjust continuous angle values
            q_rand = self.robot.normConf(q_rand, na_near.conf)
            na_new = self.Ta.stopNode(q_rand, na_near, maxSteps = maxStopNodeSteps)
            if goalTest(na_new.conf):
                return na_new
        if debug('rrt'):
            print '\nRRT: Goal not reached in' + ' %s iterations\n' %str(K)
        return 'FAILURE'

    def safePath(self, qf, qi, display = False):
        q = self.Ta.stopNode(qf, Node(qi, None, None),
                             addNodes=False, display=display).conf
        if display:
            wm.getWindow('W').clear()
            self.pbs.draw(self.prob, 'W')
            qi.draw('W', 'cyan')
            q.draw('W', 'red')
            qf.draw('W', 'orange')
            raw_input('q==qf is %s'%str(q==qf))
        return eqChains(q, qf, self.moveChains)

    def tracePath(self, node):
        path = [node]; cur = node
        while cur.parent != None:
            cur = cur.parent
            path.append(cur)
        return path

    def findGoalPath(self, goalTest, K=None):
        node = self.buildTree(goalTest, K)
        if node is 'FAILURE': return 'FAILURE'
        path = self.tracePath(node)[::-1]
        goalValues = [goalTest(c.conf) for c in path]
        goalIndex = goalValues.index(True)
        return path[:goalIndex+1]       # include up to first True 
    
    def findPath(self, K=None):
        sharedNodes = self.buildBiTree(K)
        if sharedNodes is 'FAILURE': return 'FAILURE'
        pathA = self.tracePath(sharedNodes[0])
        pathB = self.tracePath(sharedNodes[1])
        if pathA[0].tree.init:
            return pathA[::-1] + pathB
        elif pathB[0].tree.init:
            return pathB[::-1] + pathA
        else:
            raise Exception, "Neither path is marked init"

def safeConf(conf, pbs, prob, allowedViol):
    viol = pbs.confViolations(conf, prob)
    ans =  viol \
           and viol.obstacles <= allowedViol.obstacles \
           and viol.shadows <= allowedViol.shadows \
           and all(viol.heldObstacles[h] <= allowedViol.heldObstacles[h] for h in (0,1)) \
           and all(viol.heldShadows[h] <= allowedViol.heldShadows[h] for h in (0,1))
    if debug('safeConf'):
        if not ans:
            pbs.draw(prob, 'W')
            conf.draw('W', 'blue')
            print 'viol', viol
            print 'allowedViol', allowedViol
            raw_input('safeConf')
    return ans

def eqChains(conf1, conf2, moveChains):
    return all([conf1.conf[c]==conf2.conf[c] for c in moveChains])

idnum = 0
class Node:
    def __init__(self, conf, parent, tree, inter=False):
        global idnum
        self.inter = inter
        self.id = idnum; idnum += 1
        self.conf = conf
        self.children = []
        self.parent = parent
        self.tree = tree
    def __str__(self):
        return 'Node:'+str(self.id)
    def __hash__(self):
        return self.id

class Tree:
    def __init__(self, conf, pbs, prob, init, allowedViol, moveChains):
        self.root = Node(conf, None, self)
        self.nodes = [Node(conf, None, self)]
        self.size = 0
        self.pbs = pbs
        self.prob = prob
        self.robot = pbs.getRobot()
        self.init = init
        self.allowedViol = allowedViol
        self.moveChains = moveChains

    def addNode(self, conf, parent, inter=False):
        n_new = Node(conf, parent, self, inter=inter)
        parent.children.append(n_new)
        self.nodes.append(n_new)
        self.size += 1
        return n_new

    def nearest(self, q):               # q is conf
        return hu.argmax(self.nodes, lambda v: -self.robot.distConf(q, v.conf))
    
    def stopNode(self, q_f, n_i,
                 stepSize = glob.rrtInterpolateStepSize,
                 addNodes = True,
                 maxSteps = 1000,
                 display = False):
        q_i = n_i.conf
        if eqChains(q_f, q_i, self.moveChains): return n_i
        step = 0
        while True:
            if maxSteps:
                if step >= maxSteps:
                    # if debug('rrt'): print 'Exceed maxSteps in rrt stopNodes'
                    return n_i
            step += 1
            q_new = self.robot.stepAlongLine(q_f, q_i, stepSize,
                                             forward = self.init,
                                             moveChains = self.moveChains)
            if safeConf(q_new, self.pbs, self.prob, self.allowedViol):
                # We may choose to add intermediate nodes to the tree or not.
                if addNodes:
                    n_new = self.addNode(q_new, n_i, inter=True);
                else:
                    n_new = Node(q_new, n_i, self, inter=True)
                if eqChains(q_f, q_new, self.moveChains):
                    n_new.inter = False
                    return n_new
                n_i = n_new
                q_i = n_i.conf
            else:                       # a collision
                n_i.inter = False
                return n_i

    def __str__(self):
        return 'TREE:['+str(len(self.size))+']'

def planRobotPath(pbs, prob, initConf, destConf, allowedViol, moveChains,
                  maxIter = None, failIter = None, safeCheck = True, inflate = False):
    startTime = time.time()
    if inflate:
        glob.ignoreShadowZ = False
    else:
        glob.ignoreShadowZ = True
    if allowedViol==None:
        v1 = pbs.confViolations(destConf, prob)
        v2 = pbs.confViolations(initConf, prob)
        if v1 and v2:
            allowedViol = v1.update(v2)
        else:
            glob.ignoreShadowZ = True
            return [], None
    if safeCheck:
        if not safeConf(initConf, pbs, prob, allowedViol):
            if debug('rrt'):
                print 'RRT: not safe enough at initial position... continuing'
            glob.ignoreShadowZ = True
            return [], None
        if not safeConf(destConf, pbs, prob, allowedViol):
            if debug('rrt'):
                print 'RRT: not safe enough at final position... continuing'
            glob.ignoreShadowZ = True
            return [], None
    inflated = pbsInflate(pbs, prob, initConf, destConf) if inflate else pbs

    if glob.useMPL:
        path = runMPL(inflated, prob, initConf, destConf, allowedViol, moveChains)
    else:
        nodes = 'FAILURE'
        failCount = -1                      # not really a failure the first time
        while nodes == 'FAILURE' and failCount < (failIter or glob.failRRTIter):
            rrt = RRT(inflated, prob, initConf, destConf, allowedViol, moveChains)
            nodes = rrt.findPath(K = maxIter or glob.maxRRTIter)
            failCount += 1
            if True:
                if failCount > 0: print '    RRT has failed', failCount, 'times'
        if failCount == (failIter or glob.failRRTIter):
            glob.ignoreShadowZ = True
            return [], None
        rrtTime = time.time() - startTime
        if debug('rrt'):
            print 'Found path in', rrtTime, 'secs'
        path = [c.conf for c in nodes]

    if debug('verifyRRTPath'):
        # verifyPath(pbs, prob, path, 'rrt:'+str(moveChains))
        verifyPath(pbs, prob, interpolatePath(path), allowedViol,
                   'interp rrt:'+str(moveChains))

    # Verify that only the moving chain is moved.
    #for chain in initConf.conf:
    #    if chain not in moveChains:
            # ## !! LPK needs to be approximately equal
            # eps = 1e-6
            # assert all([all([within(x, y, eps) \
            #             for (x, y) in zip(initConf.conf[chain], c.conf[chain])]) for \
            #               c in path])
            #assert all(initConf.conf[chain] == c.conf[chain] for c in path)

    glob.ignoreShadowZ = True
    return path, allowedViol

def planRobotPathSeq(pbs, prob, initConf, destConf, allowedViol,
                     maxIter = None, failIter = None, inflate = False):
    chains = [chain for chain in destConf.conf \
              if chain in initConf.conf \
              and max([abs(x-y) > 1.0e-6 for (x,y) in zip(initConf.conf[chain], destConf.conf[chain])])]
    return planRobotPath(pbs, prob, initConf, destConf, allowedViol, chains,
                  maxIter = maxIter, failIter = failIter, safeCheck = False, inflate = inflate)


def planRobotGoalPath(pbs, prob, initConf, goalTest, allowedViol, moveChains,
                      maxIter = None, failIter = None):
    startTime = time.time()
    if allowedViol==None:
        v = pbs.confViolations(initConf, prob)
        if v:
            allowedViol = v
        else:
            return [], None
    if not safeConf(initConf, pbs, prob, allowedViol):
        if debug('rrt'):
            print 'RRT: not safe enough at initial position... continuing'
        return [], None
    nodes = 'FAILURE'
    failCount = -1                      # not really a failure the first time
    while nodes == 'FAILURE' and failCount < (failIter or glob.failRRTIter):
        rrt = RRT(pbs, prob, initConf, None, allowedViol, moveChains)
        nodes = rrt.findGoalPath(goalTest, K = maxIter or glob.maxRRTIter)
        failCount += 1
        if debug('rrt') or failCount % 10 == 0:
            if failCount > 0:
                print 'RRT Failed', failCount, 'times in planRobotGoalPath'
    if failCount == (failIter or glob.failRRTIter):
        return [], None
    rrtTime = time.time() - startTime
    if debug('rrt'):
        print 'Found goal path in', rrtTime, 'secs'
    path = [c.conf for c in nodes]
    # Verify that only the moving chain is moved.
    for chain in initConf.conf:
        if chain not in moveChains:
            assert all(initConf.conf[chain] == c.conf[chain] for c in path)
    if debug('verifyRRTPath'):
        # verifyPath(pbs, prob, path, allowedViol, 'rrt:'+chain)
        verifyPath(pbs, prob, interpolatePath(path), allowedViol, 'interp rrt:'+chain)
    return path, allowedViol

def interpolate(q_f, q_i, stepSize=glob.rrtInterpolateStepSize, moveChains=None, maxSteps=300):
    return list(interpolateGen(q_f, q_i, stepSize=glob.rrtInterpolateStepSize,
                               moveChains=None, maxSteps=maxSteps))

def interpolateGen(q_f, q_i, stepSize=glob.rrtInterpolateStepSize, moveChains=None, maxSteps=300):
    robot = q_f.robot
    path = [q_i]
    q = q_i
    step = 0
    moveChains = moveChains or q_f.conf.keys()
    yield q_i
    while q != q_f:
        if step > maxSteps:
            raw_input('interpolate exceeded maxSteps')
        qn = robot.stepAlongLine(q_f, q, stepSize,
                                 moveChains = moveChains or q_f.conf.keys())
        if eqChains(q, qn, moveChains): break
        q = qn
        path.append(q)
        yield q
        step += 1
    if eqChains(path[-1], q_f, moveChains):
        path.pop()
    path.append(q_f)
    if len(path) > 1 and not(path[0] == q_i and path[-1] == q_f):
        raw_input('Path inconsistency')
    yield q_f

def verifyPath(pbs, p, path, allowedViol, msg='rrt'):
    shWorld = pbs.getShadowWorld(p)
    obst = shWorld.getObjectShapes()
    attached = shWorld.attached
    pbsDrawn = False
    allowed = allowedViol.allObstacles() + allowedViol.allShadows()
    win = wm.getWindow('W')
    for conf in path:
        robotShape = conf.placement(attached=attached)
        if debug('verifyRRTPath'):
            pbs.draw(p, 'W')
            pbsDrawn = True
            conf.draw('W')
            win.update()
            sleep(0.2)
        if any(o not in allowed and robotShape.collides(o) for o in obst):
            if not pbsDrawn:
                pbs.draw(p, 'W')
                pbsDrawn = True
            print msg, 'path',
            colliders = [o for o in obst if robotShape.collides(o)]
            robotShape.draw('W', 'red')
            for o in colliders: o.draw('W', 'red')
            print 'collision with', [o.name() for o in colliders]
            raw_input('Ok?')
    return True

def interpolatePath(path, stepSize = glob.rrtInterpolateStepSize):
    interpolated = []
    for i in range(1, len(path)):
        qf = path[i]
        qi = path[i-1]
        confs = interpolate(qf, qi, stepSize=stepSize)
        if debug('rrt'): print i, 'path segment has', len(confs), 'confs'
        interpolated.extend(confs)
    return removeDuplicateConfs(interpolated)

def pbsInflate(pbs, prob, initConf, goalConf):
    if not glob.useInflation: return pbs
    newBS = pbs.copy()
    newBS.conf = (False, initConf)
    inflatedVar = (0.05**2, 0.05**2, 0.05**2, 0.1**2)
    for obj in newBS.objectBs:
        fix, objB = newBS.objectBs[obj]
        newBS.updatePlaceB(objB.modifyPoseD(var=inflatedVar))
    newBS.internalCollisionCheck(dither=False, objChecks=False, factor=1.1)
    newBS.conf = (newBS.conf[0], goalConf)
    newBS.internalCollisionCheck(dither=False, objChecks=False, factor=1.1)
    newBS.draw(prob, 'W')
    wm.getWindow('W').update()
    # raw_input('Inflation')
    return newBS

##################################################
# Interface for Amruth's MCR code
##################################################

# Should really avoid the conversion to tuples

class MCRHelper():
    def __init__(self, pbs, prob, allowedViol, moveChains, conf):
        self.pbs = pbs
        self.prob = prob
        self.allowedViol = allowedViol
        self.moveChains = moveChains
        self.conf = conf
        self.stepSize = glob.rrtInterpolateStepSize

    # return back list of obstacles that are in collision when robot is at configuration q
    def collisionsAtQ(self, q):
        conf = confFromTuple(q, self.moveChains, self.conf)
        return confCollisions(conf, self.pbs, self.prob, self.allowedViol)

    # return a configuration represented as a list, can use the passed goal to do goal biasing in random sampling
    def sampleConfig(self, goal):
        return tupleFromConf(self.pbs.getRobot().randomConf(self.moveChains),
                             self.moveChains)

    # return a list of configurations (as defined above that exclude qFrom and qTo ie (qFrom... qTo) )
    def generateInBetweenConfigs(self, qFrom, qTo):
        confFrom = confFromTuple(qFrom, self.moveChains, self.conf)
        confTo = confFromTuple(qTo, self.moveChains, self.conf)
        for c in interpolate(confTo, confFrom):
            yield tupleFromConf(c, self.moveChains)

    # get configuration through linear scaling of vector `qFrom + scaleFactor * (qTo - qFrom)`
    def getBetweenConfigurationWithFactor(self, qFrom, qTo, scaleFactor):
        confFrom = confFromTuple(qFrom, self.moveChains, self.conf)
        confTo = confFromTuple(qTo, self.moveChains, self.conf)
        return tupleFromConf(self.conf.robot.stepAlongLine(confTo, confFrom,
                                                           scaleFactor, moveChains=self.moveChains),
                             self.moveChains)

    # scalar representation of the distance between these configurations
    def distance(self, q1, q2):
        conf1 = confFromTuple(q1, self.moveChains, self.conf)
        conf2 = confFromTuple(q2, self.moveChains, self.conf)
        return self.conf.robot.distConf(conf1, conf2)

    # need a way to get the weight of an obstacle (right now its obstacle.getWeight())

def runMPL(pbs, prob, initConf, destConf, allowedViol, moveChains):
    algorithms = ['mcr', 'rrt', 'birrt',
                  'ignore start and goal birrt', 'collision based rrt']
    helper = MCRHelper(pbs, prob, allowedViol, moveChains, initConf)
    init = tupleFromConf(initConf, moveChains)
    dest = tupleFromConf(destConf, moveChains)
    if len(init) == 0 or len(dest) == 0:
        return [initConf, destConf]
    path, cover = runAlgorithm(init, dest, helper, algorithms.index('birrt'))
    return [confFromTuple(c, moveChains, initConf) for c in path]

def tupleFromConf(conf, moveChains):
    values = []
    for chain in moveChains:
        values.extend(conf.conf[chain])
    return tuple(values)

def confFromTuple(tup, moveChains, conf):
    index = 0
    for chain in moveChains:
        n = len(conf.conf[chain])
        conf = conf.set(chain, list(tup[index:index+n]))
        index += n
    return conf

def confCollisions(conf, pbs, prob, allowedViol):
    viol = pbs.confViolations(conf, prob)
    ans =  viol \
           and viol.obstacles <= allowedViol.obstacles \
           and viol.shadows <= allowedViol.shadows \
           and all(viol.heldObstacles[h] <= allowedViol.heldObstacles[h] for h in (0,1)) \
           and all(viol.heldShadows[h] <= allowedViol.heldShadows[h] for h in (0,1))

    print 'allowed', allowedViol
    print viol
    pbs.draw(prob, 'W'); conf.placement().draw('W')
    raw_input('Next?')

    if ans:
        # No unallowed violations
        return []
    elif ans is None:
        return ['permanent']
    # Collect collisions
    collisions = []
    for o in viol.obstacles:
        if not o in allowedViol.obstacles: collisions.append(o.name())
    for o in viol.shadows:
        if not o in allowedViol.shadows: collisions.append(o.name())
    for h in (0,1):
        for o in viol.heldObstacles[h]:
            if not o in allowedViol.heldObstacles[h]: collisions.append(o.name())
        for o in viol.heldShadows[h]:
            if not o in allowedViol.heldShadows[h]: collisions.append(o.name())

    pbs.draw(prob, 'W'); conf.placement().draw('W')
    print 'collisions', collisions
    raw_input('Next?')
    
    return collisions
