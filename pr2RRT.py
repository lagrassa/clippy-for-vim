import math
import random
import copy
import time
import planGlobals as glob
from planGlobals import debug, debugMsg
import util
import windowManager3D as wm

############################################################
#  TO DO:
#  Make sure that the nearest nodes in the tree are marked as not "inter"
#  so that we have all the key points in the trajectory if we ignore "inter" nodes.
############################################################

# For now, use configurations until they are added to the tree as Nodes.

class RRT:
    def __init__(self, pbs, prob, initConf, goalConf, allowedViol, moveChains):
        if debug('rrt'): print 'Setting up RRT'
        self.pbs = pbs
        self.prob = prob
        self.robot = pbs.getRobot()
        self.moveChains = moveChains
        self.allowedViol = allowedViol
        self.Ta = Tree(initConf, pbs, prob, True, allowedViol, moveChains)
        if goalConf:
            self.Tb = Tree(goalConf, pbs, prob, False, allowedViol, moveChains)

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
        q_new = self.Ta.stopNode(self.Tb.root.conf, self.Ta.root)
        if q_new == self.Tb.root.conf:
            if debug('rrt'): print 'Found direct path'
            na_new = self.Ta.addNode(q_new, self.Ta.root)
            nb_new = self.Tb.addNode(q_new, self.Tb.root)
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
                if na_new.conf == nb_new.conf:
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
        for i in range(K):
            if debug('rrt'):
                if i % 100 == 0: print i
            q_rand = self.randConf()
            na_near = self.Ta.nearest(q_rand)
            # adjust continuous angle values
            q_rand = self.robot.normConf(q_rand, na_near.conf)
            na_new = self.Ta.stopNode(q_rand, na_near, maxSteps = 5)
            if goalTest(na_new.conf):
                return na_new
        if debug('rrt'):
            print '\nRRT: Goal not reached in' + ' %s iterations\n' %str(K)
        return 'FAILURE'

    def safePath(self, qf, qi, display = False):
        q = self.Ta.stopNode(qf, Node(qi, None, None),
                             addNodes=False, display=display).conf
        if verbose:
            wm.getWindow('W').clear()
            self.pbs.draw(self.prob, 'W')
            qi.draw('W', 'cyan')
            q.draw('W', 'red')
            qf.draw('W', 'orange')
            raw_input('q==qf is %s'%str(q==qf))
        return q == qf

    def tracePath(self, node):
        path = [node]; cur = node
        while cur.parent != None:
            cur = cur.parent
            path.append(cur)
        return path

    def findGoalPath(self, goalTest, K=None):
        node = self.buildTree(goalTest, K)
        if node is 'FAILURE': return 'FAILURE'
        return self.tracePath(node)[::-1]
    
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
    viol, _ = pbs.getRoadMap().confViolations(conf, pbs, prob)
    return viol \
           and viol.obstacles <= allowedViol.obstacles \
           and viol.shadows <= allowedViol.shadows

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
        return 'Node:'+str(i)
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
        return util.argmax(self.nodes, lambda v: -self.robot.distConf(q, v.conf))   
    
    def stopNode(self, q_f, n_i,
                 stepSize = glob.rrtInterpolateStepSize,
                 addNodes = True,
                 maxSteps = 1000,
                 display = False):
        q_i = n_i.conf
        if all([q_f[c] == q_i[c] for c in q_f.conf]): return n_i
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
                if all([q_f[c] == q_new[c] for c in q_f.conf]):
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
                  maxIter = None, failIter = None):
    startTime = time.time()
    if allowedViol==None:
        v1, _ = pbs.getRoadMap().confViolations(destConf, pbs, prob)
        v2, _ = pbs.getRoadMap().confViolations(initConf, pbs, prob)
        if v1 and v2:
            allowedViol = v1.update(v2)
        else:
            return None
    if not safeConf(initConf, pbs, prob, allowedViol):
        if debug('rrt'):
            print 'RRT: not safe enough at initial position... continuing'
        return []
    if not safeConf(destConf, pbs, prob, allowedViol):
        if debug('rrt'):
            print 'RRT: not safe enough at final position... continuing'
        return []
    nodes = 'FAILURE'
    failCount = -1                      # not really a failure the first time
    while nodes == 'FAILURE' and failCount < (failIter or glob.failRRTIter):
        rrt = RRT(pbs, prob, initConf, destConf, allowedViol, moveChains)
        nodes = rrt.findPath(K = maxIter or glob.maxRRTIter)
        failCount += 1
        if debug('rrt'):
            if failCount > 0: print 'Failed', failCount, 'times'
    if failCount == (failIter or glob.failRRTIter):
        return None
    rrtTime = time.time() - startTime
    if debug('rrt'):
        print 'Found path in', rrtTime, 'secs'
    return [c.conf for c in nodes]


def planRobotGoalPath(pbs, prob, initConf, goalTest, allowedViol, moveChains,
                      maxIter = None, failIter = None):
    startTime = time.time()
    if allowedViol==None:
        v, _ = pbs.getRoadMap().confViolations(initConf, pbs, prob)
        if v:
            allowedViol = v
        else:
            return None
    if not safeConf(initConf, pbs, prob, allowedViol):
        if debug('rrt'):
            print 'RRT: not safe enough at initial position... continuing'
        return []
    nodes = 'FAILURE'
    failCount = -1                      # not really a failure the first time
    while nodes == 'FAILURE' and failCount < (failIter or glob.failRRTIter):
        rrt = RRT(pbs, prob, initConf, None, allowedViol, moveChains)
        nodes = rrt.findGoalPath(goalTest, K = maxIter or glob.maxRRTIter)
        failCount += 1
        if debug('rrt'):
            if failCount > 0: print 'Failed', failCount, 'times'
    if failCount == (failIter or glob.failRRTIter):
        return None
    rrtTime = time.time() - startTime
    if debug('rrt'):
        print 'Found goal path in', rrtTime, 'secs'
    return [c.conf for c in nodes]

def interpolate(q_f, q_i, stepSize=0.25, moveChains=None, maxSteps=100):
    robot = q_f.robot
    path = [q_i]
    q = q_i
    step = 0
    while q != q_f:
        if step > maxSteps:
            raw_input('interpolate exceeded maxSteps')
        qn = robot.stepAlongLine(q_f, q, stepSize,
                                 moveChains = moveChains or q_f.keys())
        if q == qn: break
        q = qn
        path.append(q)
        step += 1
    path.append(q_f)
    return path
