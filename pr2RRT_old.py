import math
import random
import copy
import time
import planGlobals as glob
import util
import windowManager3D as wm

############################################################
#  TO DO:
#  Make sure that the nearest nodes in the tree are marked as not "inter"
#  so that we have all the key points in the trajectory if we ignore "inter" nodes.
############################################################

# The initConf is a complete robot configuration.  The configurations kept at
# the nodes are complete, but the sampling is done in the subspace defined by
# the gaolConfig argument, which may be partial.

# For now, use configurations until they are added to the tree as Nodes.

verbose = False

class RRT:
    def __init__(self, world, robot, initConf, goalConf):

        assert None, 'Needs updating to handle attached objects'

        if verbose: print 'Setting up RRT'
        self.world = world
        self.robot = robot
        self.moveChains = goalConf.keys()  # the chains we want to move
        self.Ta = Tree(initConf,
                       world, robot, init=True, moveChains = self.moveChains)
        self.Tb = Tree(robot.completeJointConf(goalConf, baseConf=initConf),
                       world, robot, init=False, moveChains = self.moveChains)

    def randConf(self):
        return self.robot.randomConf(self.moveChains)

    def swapTrees(self):
        if self.Ta.size > self.Tb.size:
            self.Ta, self.Tb = self.Tb, self.Ta

    # the qx variables denote confs; the nx variable denote nodes
    def build(self, K=1000):
        """Builds the RRT and returns either a pair of nodes (one in each tree
        with a common configuration or FAILURE."""
        if verbose: print 'Building RRT'
        q_new = self.Ta.stopNode(self.Tb.root.conf, self.Ta.root)
        if q_new == self.Tb.root.conf:
            print 'Found direct path'
            na_new = self.Ta.addNode(q_new, self.Ta.root)
            nb_new = self.Tb.addNode(q_new, self.Tb.root)
            return (na_new, nb_new)
        for i in range(K):
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
                    print 'RRT: Goal reached in' +\
                          ' %s iterations' %str(i)
                    return (na_new, nb_new)
            self.swapTrees()
        print '\nRRT: Goal not reached in' + ' %s iterations\n' %str(K)
        return 'FAILURE'

    def safePath(self, qf, qi, display = False):
        q = self.Ta.stopNode(qf, Node(qi, None, None),
                             addNodes=False, display=display).conf
        if verbose:
            wm.getWindow('W').clear()
            self.world.draw('W')
            self.robot.placement(qi, self.world)[0].draw('W', 'cyan')
            self.robot.placement(q, self.world)[0].draw('W', 'red')
            self.robot.placement(qf, self.world)[0].draw('W', 'orange')
            raw_input('q==qf is %s'%str(q==qf))
        return q == qf

    def smoothPath(self, path, nsteps = glob.smoothSteps):
        n = len(path)
        if verbose: print 'Path has %s points'%str(n), '... smoothing'
        smoothed = list(path)
        checked = set([])
        count = 0
        step = 0
        while count < nsteps:
            if verbose: print step, 
            i = random.randrange(n)
            j = random.randrange(n)
            if j < i: i, j = j, i 
            step += 1
            if verbose: print i, j, len(checked)
            if j-i < 2 or \
                (smoothed[j], smoothed[i]) in checked:
                count += 1
                continue
            else:
                checked.add((smoothed[j], smoothed[i]))
            if self.safePath(smoothed[j], smoothed[i]):
                count = 0
                smoothed[i+1:j] = []
                n = len(smoothed)
                if verbose: print 'Smoothed path length is', n
            else:
                count += 1
        return smoothed

    def tracePath(self, node):
        path = [node]; cur = node
        while cur.parent != None:
            cur = cur.parent
            path.append(cur)
        return path

    def findPath(self, K=None):
        sharedNodes = self.build(K)
        if sharedNodes is 'FAILURE': return 'FAILURE'
        pathA = self.tracePath(sharedNodes[0])
        pathB = self.tracePath(sharedNodes[1])
        if pathA[0].tree.init:
            return pathA[::-1] + pathB
        elif pathB[0].tree.init:
            return pathB[::-1] + pathA
        else:
            raise Exception, "Neither path is marked init"
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
    def __init__(self, conf, world, robot, init=False, moveChains=None):
        self.root = Node(conf, None, self)
        self.nodes = [Node(conf, None, self)]
        self.size = 0
        self.world = world
        self.robot = robot
        self.init = init
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
                 stepSize = glob.rrtStep,
                 addNodes = True,
                 maxSteps = 1000,
                 display = False):
        q_i = n_i.conf
        if all([q_f[c] == q_i[c] for c in q_f.conf]): return n_i
        step = 0
        while True:
            if maxSteps:
                if step >= maxSteps:
                    print 'Exceed maxSteps in rrt stopNodes'
                    return n_i
            step += 1
            q_new = self.robot.stepAlongLine(q_f, q_i, stepSize,
                                             forward = self.init,
                                             moveChains = self.moveChains)
            if self.robot.safeConf(q_new, self.world):
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

def planRobotPath(world, robot, initConf, destConf,
                  smooth = True, maxIter = None, failIter = None):
    startTime = time.time()
    if not robot.safeConf(initConf, world, showCollisions=True):
        print 'RRT: not safe at initial position... continuing'
        return []
    if not robot.safeConf(robot.completeJointConf(destConf, baseConf=initConf),
                          world, showCollisions=True):
        print 'RRT: not safe at final position... continuing'
        return []
    if glob.skipRRT:
        return [initConf,
                robot.completeJointConf(destConf, baseConf=initConf)]
    nodes = 'FAILURE'
    failCount = -1                      # not really a failure the first time
    while nodes == 'FAILURE' and failCount < (failIter or glob.failRRTIter):
        rrt = RRT(world, robot, initConf, destConf)
        nodes = rrt.findPath(K = maxIter or glob.maxRRTIter)
        failCount += 1
        if failCount > 0: print 'Failed', failCount, 'times'
    if failCount == (failIter or glob.failRRTIter):
        return None
    rrtTime = time.time() - startTime
    print 'Found path in', rrtTime, 'secs'
    # path = [c.conf for c in nodes if c.inter is False]
    path = [c.conf for c in nodes]
    if smooth:
        return rrt.smoothPath(path)
    else:
        return path

def interpolate(q_f, q_i, stepSize=0.5, moveChains=None):
    robot = q_f.robot
    path = [q_i]
    q = q_i
    while q != q_f:
        qn = robot.stepAlongLine(q_f, q, stepSize,
                                 moveChains = moveChains or q_f.keys())
        if q == qn: break
        q = qn
        path.append(q)
    path.append(q_f)
    return path
