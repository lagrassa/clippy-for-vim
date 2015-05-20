import math
import random
import copy
import time
import util
from scipy.spatial import cKDTree
import windowManager3D as wm
import numpy as np
import shapes
from ranges import *
from pr2Robot import CartConf
from objects import WorldState
from geom import bboxOverlap, bboxUnion
from transformations import quaternion_slerp
import ucSearchPQ as search
reload(search)
import pr2RRT as rrt
reload(rrt)
import fbch
import planGlobals as glob
from planGlobals import debugMsg, debug, debugDraw, debugOn
from miscUtil import prettyString

from pr2Util import Violations, NextColor, drawPath, NextColor, shadowWidths, Hashable

objCollisionCost = 10.0                    # !! was 2.0
shCollisionCost = 2.0

maxSearchNodes = 5000                   # 5000
maxExpandedNodes = 2000                  # 2000
searchGreedy = 0.75
searchOpt = 0.75                        # should be 0.5 ideally, but it's slow...

useVisited = True           # unjustified
minStep = 0.25              # !! normally 0.25 for joint interpolation

confReachViolGenBatch = 10

nodeHash = {}
node_idnum = 0
class Node(Hashable):
    def __init__(self, conf, cartConf=None, point=None):
        global node_idnum
        self.id = node_idnum; node_idnum += 1
        self.conf = conf
        self.cartConf = cartConf
        self.point = tuple(point or self.pointFromConf(self.conf)) # used for nearest neighbors
        self.key = False
    def cartConf(self):
        if not self.cartConf:
            self.cartConf = self.conf.cartConf()
        return self.cartConf
    def baseConf(self):
        return tuple(self.conf['pr2Base'])
    def pointFromConf(self, conf):
        if False:
            point = []
            for chain in ['pr2Base', 'pr2LeftArm', 'pr2RightArm']:
                if 'Gripper' in chain or 'Torso' in chain:
                    point.append(conf[chain][0])
                else:
                    point.extend(conf[chain])
        else:
            x,y,th=conf['pr2Base']
            point = (x,y, 0.1*math.cos(th), 0.1*math.sin(th))
        return point
    def pointFromCart(self, cart, alpha = 0.1):
        point = []
        for chain in ['pr2Base', 'pr2LeftArm', 'pr2RightArm']:
            if 'Gripper' in chain or 'Torso' in chain:
                point.append(cart[chain][0])
            else:
                pose = cart[chain] 
                pose_o = [pose.matrix[i,3] for i in range(3)]
                point.extend(pose_o)
                for j in [2]:           # could be 0,1,2
                    point.extend([alpha*pose.matrix[i,j] + pose_o[i] for i in range(3)])
        return point
    def draw(self, window, color = 'blue'):
        self.conf.draw(window, color=color)
    def desc(self):
        return self.conf                # used for hashing and equality
    def __str__(self):
        return 'Node:'+str(self.id)+prettyString(self.point)
    __repr__ = __str__

def makeNode(conf, cart=None):
    if not conf in nodeHash:
        nodeHash[conf] = Node(conf, cart)
    return nodeHash[conf]

edge_idnum = 0
class Edge(Hashable):
    def __init__(self, n1, n2):
        global edge_idnum
        self.id = edge_idnum; edge_idnum += 1
        self.ends = (n1, n2)
        self.nodes = []              # intermediate nodes
        # Cache for collisions on this edge.
        self.heldCollisions = {}     # grasp + h: {object : {True, False}}
        self.robotCollisions = {}    # h : {object : {True, False}}
        self.heldShapes = {}         # grasp + h :{node : shape}
        self.robotShapes = {}        # h : {node : shape}
        self.bbox = None
        # print 'Creating', self
    def draw(self, window, color = 'cyan'):
        for node in self.nodes:
            node.draw(window, color=color)
    def desc(self):
        return frozenset(self.ends)
    def __str__(self):
        return 'Edge:'+str(self.id)+"(Node:%d-Node:%d)"%tuple(n.id for n in self.ends)
    __repr__ = __str__

cluster_idnum = 0
class Cluster:
    def __init__(self, roadMap, nodes, params):
        global cluster_idnum
        self.roadMap = roadMap
        self.params = params
        self.id = cluster_idnum; cluster_idnum += 1
        self.baseConf = nodes[0].baseConf()
        x,y,th = self.baseConf
        self.point = (x,y,0.1*math.cos(th), 0.1*math.sin(th)) # used for nearest neighbors
        self.nodes = nodes
        self.reps = set([])             # representative nodes
        self.kdTree = KDTree(nodes, params['kdLeafSize'])
        self.kNearest = params['kNearest']
        self.nodeGraph= NodeGraph({}, {})

    def copy(self):
        cluster = Cluster(self.roadMap, self.nodes, self.params)
        cluster.kdTree = KDTree(self.nodes, self.params['kdLeafSize'])
        cluster.kNearest = self.params['kNearest']
        cluster.nodeGraph= self.nodeGraph.copy()
        return cluster
        
    def project(self, node):
        if tuple(node.conf['pr2Base']) != self.baseConf:
            newConf = node.conf.set('pr2Base', list(self.baseConf))
            return makeNode(newConf)
        else:
            return node
    def addRep(self, node):
        if node in self.reps:
            return node
        node = self.project(node)
        self.reps.add(node)
        self.addNode(node, proj = False)
        return node
    def addNode(self, node, proj = True):
        if proj:
            node = self.project(node)
        self.kdTree.addEntry(node)
        near = self.kdTree.nearest(node, self.params['kNearest'])
        for d, n in near:
            if d < np.inf:
                self.roadMap.addEdge(self.nodeGraph, node, n)        
    def draw(self, window, color='green'):
        for node in self.reps:
            node.draw(window, color=color)
    def desc(self):
        return tuple(self.baseConf)     # used for hash and equality
    def __str__(self):
        return 'Cluster:'+str(self.id)+prettyString(self.desc())
    __repr__ = __str__

class KDTree:
    def __init__(self, entries, kdLeafSize = 20):
        self.entries = entries
        points = [e.point for e in entries]
        self.points = np.array(points)  # array of point arrays
        self.kdTree = cKDTree(self.points, kdLeafSize)
        self.kdLeafSize = kdLeafSize
        self.size = len(points)
        self.newEntries = []
        self.newPoints = []             # list of point lists
        self.newSize = 0
        self.newKDTree = None
        self.entryTooClose = 0.001 # is there a rational way of picking this?

    def allEntries(self):
        return self.entries + self.newEntries

    def batchAddEntries(self, entries):
        if not entries: return
        last = len(entries) - 1
        for i, entry in enumerate(entries):
            self.addEntry(entry, merge = (i == last))   # merge on the last one

    def mergeKDTrees(self):
        if self.newSize == 0: return
        self.points = np.vstack((self.points,
                                 np.array(self.newPoints)))
        self.kdTree = cKDTree(self.points, self.kdLeafSize)
        self.entries.extend(self.newEntries)
        self.size += self.newSize
        self.newPoints = []
        self.newEntries = []
        self.newSize = 0
        self.newKDTree = None

    def findEntry(self, entry):
        (d, ne) = self.nearest(entry, 1)[0]
        if d <= self.entryTooClose:
            return ne
        
    def addEntry(self, entry, merge = False):
        point = entry.point
        if merge:
            ne = self.findEntry(entry)
            if ne:
                if debug('addEntry'):
                    print 'New', entry.point
                    print 'Old', ne.point
                    raw_input('Entry too close, d=%s'%d)
                return ne
        self.newPoints.append(point)
        self.newEntries.append(entry)
        self.newSize += 1
        self.newKDTree = None
        if merge and self.newSize > self.kdLeafSize:
            self.mergeKDTrees()
        return entry

    # Is it better just to make a new global tree when we do addConf
    def nearest(self, entry, k):
        merge = []
        dists, ids = self.kdTree.query(entry.point, k)
        if k == 1:
            dists = [dists]; ids = [ids]
        if self.newPoints:
            if not self.newKDTree:
                self.newKDTree = cKDTree(self.newPoints, self.kdLeafSize)
        else:
            return [(d, self.entries[i]) for d, i in zip(dists, ids) if d < np.inf]
        for (d, i) in zip(dists, ids):
            if d < np.inf: merge.append((d, i, False))
        assert self.newKDTree
        newDists, newIds = self.newKDTree.query(entry.point, k)
        if k == 1:
            newDists = [newDists]; newIds = [newIds]
        for (d, i) in zip(newDists, newIds):
            if d < np.inf: merge.append((d, i, True))
        merge.sort()
        return [(d, self.newEntries[i]) if new else (d, self.entries[i]) \
                for (d, i, new) in merge[:k]]

class NodeGraph:
    def __init__(self, edges, incidence):
        self.edges = edges
        self.incidence = incidence      # Node -> set of edges
    def copy(self):
        return NodeGraph(self.edges.copy(), self.incidence.copy())

def combineNodeGraphs(*graphs):
    edges = graphs[0].edges.copy()
    incidence = graphs[0].incidence.copy()
    for graph in graphs[1:]:
        edges.update(graph.edges)
        for v in graph.incidence:
            if v in incidence:
                incidence[v] = incidence[v].copy()
                incidence[v].update(graph.incidence[v])
            else:
                incidence[v] = graph.incidence[v]
    return NodeGraph(edges, incidence)
    
viol0 = Violations()
stats = {'newRob':0, 'newHeld':0, 'oldRob':0, 'oldHeld':0,
         'newTest':0, 'oldTest':0}
allMoveChains = ['pr2Base', 'pr2LeftArm', 'pr2RightArm'] # no grippers or head

# params: kNearest, kdLeafSize, cartesian, moveChains
# cartesian indicates whether to use cartesian interpolation
# The moveChains indicate which parts of the robot are moving.
class RoadMap:
    def __init__(self, homeConf, world, params):
        self.params = params
        self.world = world
        self.robot = world.robot
        self.homeConf = homeConf
        self.robotPlace = self.robot.placement(homeConf)[0]
        self.root = makeNode(homeConf)
        self.rootCluster = Cluster(self, [self.root], params)
        self.rootCluster.addRep(self.root) # !! add several alternatives?
        self.clustersByPoint = {self.rootCluster.point:self.rootCluster}
        self.kNearest = params['kNearest']
        # cluster kdTree
        self.kdTree = KDTree([self.rootCluster])
        # graph of rep nodes among the clusters 
        self.clusterGraph = NodeGraph({}, {})
        # Caches
        self.confReachCache = {}
        self.approachConfs = {}

    def nodes(self):
        return self.clusterGraph.incidence.keys()

    def drawClusters(self, color='yellow'):
        for entry in self.kdTree.entries + self.kdTree.newEntries:
            entry.draw('W', color=color)

    def confReachViol(self, targetConf, pbs, prob,
                      initViol=viol0, startConf = None,
                      optimize = False, moveBase = True):

        def displayAns(ans):
            if not debug('confReachViol'): return
            if ans:
                if (not fbch.inHeuristic or debug('drawInHeuristic')):
                    (viol, cost, edgePath) = ans
                    path = self.confPathFromEdgePath(edgePath)
                    drawPath(path, viol=viol, attached=attached)
                    debugMsg('confReachViol', ('->', (viol, cost, 'path len = %d'%len(path))))
            else:
                drawProblem(forceDraw=True)
                debugMsg('confReachViol', ('-> None'))

        def cacheAns(ans):
            if debug('confReachViolCache'):
                debugMsg('confReachViolCache',
                         ('targetConf', targetConf.conf),
                         ('initConf', initConf),
                         ('prob', prob),
                         ('moveObjBs', pbs.moveObjBs),
                         ('fixObjBs', pbs.fixObjBs),
                         ('held', (pbs.held['left'].mode(),
                                   pbs.held['right'].mode(),
                                   grasp('left'), grasp('right'))),
                         ('initViol', initViol),
                         ('avoidShadow', pbs.avoidShadow))
            if not (optimize or fbch.inHeuristic):
                key = (targetConf, initConf, moveBase)
                if not key in self.confReachCache:
                    self.confReachCache[key] = []
                self.confReachCache[key].append((pbs, prob,
                                                 ans if ans else (None, None, None)))

        def checkCache(key, type='full', loose=False):
            if fbch.inHeuristic or optimize: return 
            if key in self.confReachCache:
                if debug('confReachViolCache'): print 'confReachCache tentative hit'
                cacheValues = self.confReachCache[key]
                sortedCacheValues = sorted(cacheValues,
                                           key=lambda v: v[-1][0].weight() if v[-1][0] else 1000.)
                ans = bsEntails(pbs, prob, sortedCacheValues, loose=loose)
                if ans != None:
                    if debug('traceCRH'): print '    actual', type, 'cache hit',
                    if debug('confReachViolCache'):
                        debugMsg('confReachViolCache', 'confReachCache '+type+' actual hit')
                        print '    returning', ans
                    return ans
            else:
                self.confReachCache[key] = []
                if debug('confReachViolCache'): print 'confReachCache miss'

        def checkFullCache():
            return checkCache((targetConf, initConf, moveBase))

        def checkApproachCache():
            if fbch.inHeuristic: return # don't bother
            if targetConf in self.approachConfs:
                ans = checkCache((self.approachConfs[targetConf], initConf, moveBase),
                                 type='approach', loose=True)
                # !! This does not bother adding the final location to the path
                if not ans:
                    if debug('traceCRH'):
                        raw_input('    No cached value for approach')
                    return None
                cv = self.confViolations(targetConf, pbs, prob)[0]
                if cv != None:
                    (viol, cost, path) = ans
                    return (viol.update(cv), cost, path)
                else:
                    raw_input('Collision at pick/place conf')

        def confAns(ans, reverse=False):
            displayAns(ans)
            if ans and ans[0]:
                (viol, cost, edgePath) = ans
                path = self.confPathFromEdgePath(edgePath)
                if reverse: path.reverse()
                return (viol, cost, path)
            else:
                return (None, None, None)

        def drawProblem(forceDraw=False):
            if forceDraw or \
                   (debug('confReachViol') and \
                    (not fbch.inHeuristic  or debug('drawInHeuristic'))):
                pbs.draw(prob, 'W')
                initConf.draw('W', 'blue', attached=attached)
                targetConf.draw('W', 'pink', attached=attached)
                print 'startConf is blue; targetConf is pink'
                raw_input('confReachViol')

        initConf = startConf or self.homeConf
        initNode = makeNode(initConf)
        targetNode = makeNode(targetConf)
        attached = pbs.getShadowWorld(prob).attached
        if initViol == None:
            return confAns(None)
        cachedApproach = checkApproachCache()
        if cachedApproach:
            (v,c,p) = confAns(cachedApproach, reverse=True)
            return (v, c, p + [targetConf])
        cached = checkFullCache()
        if cached:
            return confAns(cached, reverse=True)

        targetCluster = self.addToCluster(targetNode, connect=False)
        startCluster = self.addToCluster(initNode)
        graph = combineNodeGraphs(self.clusterGraph,
                                  startCluster.nodeGraph,
                                  targetCluster.nodeGraph)
        # if not fbch.inHeuristic:
        #     print '    Graph nodes =', len(graph.incidence), 'graph edges', len(graph.edges)
        if debug('traceCRH'): print '    find path',
        # search back from target...
        ansGen = self.minViolPathGen(graph, targetNode, [initNode], pbs, prob,
                                     initViol=initViol, optimize=optimize, moveBase=moveBase)
        ans = next(ansGen, None)
        if (ans == None or ans[0] == None) and not moveBase:
            path, viol = rrt.planRobotPathSeq(pbs, prob, initConf, targetConf, None,
                                              maxIter=20, failIter=5)
            return (viol, 0, path) 
        cacheAns(ans)
        return confAns(ans, reverse=True)

    def addToCluster(self, node, rep=False, connect=True):
        if debug('addToCluster'): print 'Adding', node, 'to cluster'
        cluster = self.clustersByPoint.get(node.point, None)
        if cluster and not connect:
            cluster = cluster.copy()
        if debug('addToCluster'): print 'Existing cluster', cluster
        if not cluster:
            cluster = Cluster(self, [node], self.params)
            if debug('addToCluster'): print 'New cluster', cluster
            if connect:
                self.kdTree.addEntry(cluster, merge=False)
                self.clustersByPoint[cluster.point] = cluster
                # Generate reps for this cluster
                near = self.kdTree.nearest(cluster, self.params['kNearest'])
                for d, cl in near:
                    if d == np.inf: break
                    if not cl.reps: continue
                    for n in cl.reps:
                        n0 = cluster.addRep(n)   # project relevant chains
                        # Connect rep from other cluster to projection in this one
                        if debug('addToCluster'): print 'Adding rep edge to cluster', cl
                        self.addEdge(self.clusterGraph, n0, n)
        # connect the node to the cluster nodes (including reps)
        if rep and connect:
            cluster.addRep(node)
            near = self.kdTree.nearest(cluster, self.params['kNearest'])
            for d, cl in near:
                if d == np.inf: break
                n0 = cl.addRep(node)
                if debug('addToCluster'): print 'Adding new rep to cluster', cl
                self.addEdge(self.clusterGraph, node, n0)
        elif not connect:
            cluster.addNode(node)
            near = self.kdTree.nearest(cluster, self.params['kNearest'])
            for d, cl in near:
                if d == np.inf: break
                n0 = cl.project(node)
                self.addEdge(cluster.nodeGraph, node, n0)
                near = cl.kdTree.nearest(n0, self.params['kNearest'])
                for dn, n in near:
                    if dn < np.inf:
                        self.addEdge(cluster.nodeGraph, n0, n)        
        else:
            cluster.addNode(node)
        if debug('addToCluster'):
            print 'Cluster incidence'
            for n in cluster.nodeGraph.incidence:
                print '    ', n, cluster.nodeGraph.incidence[n]
        debugMsg('addToCluster', 'Continue?')
        return cluster

    def batchAddClusters(self, initConfs):
        startTime = time.time()
        print 'Start batchAddClusters'
        clusters = []
        for conf in initConfs:
            node = makeNode(conf)
            cluster = self.clustersByPoint.get(node.point, None)
            if not cluster:
                cluster = Cluster(self, [node], self.params)
                clusters.append(cluster)
                self.clustersByPoint[cluster.point] = cluster
            cluster.addRep(node)
        self.kdTree.batchAddEntries(clusters)
        for cluster in clusters:
            near = self.kdTree.nearest(cluster, self.params['kNearest'])
            for d, cl in near:
                if d == np.inf: break
                for n1 in cluster.reps:
                    n0 = cl.addRep(n1)
                    self.addEdge(self.clusterGraph, n0, n1)
                for n2 in cl.reps:
                    n0 = cluster.addRep(n2)
                    self.addEdge(self.clusterGraph, n0, n2)
        print 'End batchAddClusters, time=', time.time()-startTime

    def confReachViolGen(self, targetConfs, pbs, prob, initViol=viol0,
                         testFn = lambda x: True, goalCostFn = lambda x: 0,
                         startConf = None, draw=False):
        attached = pbs.getShadowWorld(prob).attached
        initConf = startConf or self.homeConf
        batchSize = confReachViolGenBatch
        batch = 0
        while True:
            # Collect the next batach of trialConfs
            batch += 1
            trialConfs = []
            count = 0
            for c in targetConfs:       # targetConfs is a generator
                if self.confViolations(c, pbs, prob)[0] != None:
                    count += 1
                    trialConfs.append(c)
                    if initConf == c and testFn(c):
                        ans = initViol or Violations(), 0, [initConf]
                        yield ans
                        return
                if count == batchSize: break
            if debug('confReachViolGen'):
                print '** Examining batch', batch, 'of', count, 'confs'
            if count == 0:              # no more confs
                if debug('confReachViolGen'):
                    print '** Finished the batches'
                break
            random.shuffle(trialConfs)
            if debug('confReachViolGen') and not fbch.inHeuristic:
                pbs.draw(prob, 'W')
                initConf.draw('W', 'blue', attached=attached)
                for trialConf in trialConfs:
                    trialConf.draw('W', 'pink', attached=attached)
                print 'startConf is blue; targetConfs are pink'
                debugMsg('confReachViolGen', 'Go?')
            
            # keep track of the original conf for the nodes
            nodeTestFn = lambda n: testFn(n.conf)
            goalNodeCostFn = lambda n: goalCostFn(n.conf)
            targetNodes = [makeNode(conf) for conf in trialConfs]
            targetClusters = [self.addToCluster(targetNode, connect=False) for targetNode in targetNodes]
            initNode = makeNode(initConf)
            startCluster = self.addToCluster(initNode)
            graph = combineNodeGraphs(*([self.clusterGraph, startCluster.nodeGraph] + \
                                        [cl.nodeGraph for cl in targetClusters]))
            gen = self.minViolPathGen(graph, initNode, targetNodes, pbs, prob,
                                      initViol=initViol or Violations(),
                                      testFn=nodeTestFn, goalCostFn=goalNodeCostFn, draw=draw)
            for ans in gen:
                if ans and ans[0] and ans[2]:
                    (viol, cost, edgePath) = ans
                    path = self.confPathFromEdgePath(edgePath)
                    if debug('confReachViolGen') and not fbch.inHeuristic:
                        drawPath(path, viol=viol, attached=attached)
                        newViol = self.checkPath(path, pbs, prob)
                        if newViol.weight() != viol.weight():
                            print 'viol', viol
                            print 'newViol', newViol
                            raw_input('checkPath failed')
                        debugMsg('confReachViolGen', ('->', (viol, cost, 'path len = %d'%len(path))))
                    yield (viol, cost, path)
                else:
                    if not fbch.inHeuristic:
                        debugMsg('confReachViolGen', ('->', ans))
                    break
        ans = None, 0, []
        debugMsg('confReachViolGen', ('->', ans))
        yield ans
        return

    def interpPose(self, pose_f, pose_i, minLength, ratio=0.5):
        if isinstance(pose_f, list):
            return [f*ratio + i*(1-ratio) for (f,i) in zip(pose_f, pose_i)], \
                   all([abs(f-i)<=minLength for (f,i) in zip(pose_f, pose_i)])
        else:
            pr = pose_f.point()*ratio + pose_i.point()*(1-ratio)
            qr = quaternion_slerp(pose_i.quat().matrix, pose_f.quat().matrix, ratio)
            return util.Transform(None, pr.matrix, qr), \
                   pose_f.near(pose_i, minLength, minLength)

    def cartInterpolators(self, n_f, n_i, minLength, depth=0):
        if depth > 10:
            raw_input('cartInterpolators depth > 10')
        if n_f.conf == n_i.conf: return [n_f]
        c_f = n_f.cartConf()
        c_i = n_i.cartConf()
        newVals = {}
        terminal = True
        for chain in self.moveChains: # make sure that we get every chain
            new, near = self.interpPose(c_f[chain], c_i[chain], minLength)
            if depth == 10 and not near:
                print depth, chain, near
                print 'c_f\n', c_f[chain]
                print 'c_i\n', c_i[chain]
                print 'new\n', new
                raw_input('Huh?')
            newVals[cartChainName(chain)] = new
            terminal = terminal and near
        if terminal: return []        # no chain needs splitting
        cart = n_i.cartConf().copy()
        cart.conf = newVals
        conf = self.robot.inverseKin(cart, conf=n_i.conf, complain=debug('cartInterpolators'))
        for chain in self.robot.chainNames: #  fill in
            if not chain in conf.conf:
                conf[chain] = n_i.conf[chain]
                cart[chain] = n_i.cartConf()[chain]
        if all([conf[chain] for chain in self.moveChains]):
            newNode =  makeNode(conf, cart)
            final = self.cartInterpolators(n_f, newNode, minLength, depth+1)
            if final != None:
                init = self.cartInterpolators(newNode, n_i, minLength, depth+1)
                if init != None:
                    final.append(newNode)
                    final.extend(init)
        else:
            # Switch to joint interpolation if cartesian interpolation fails!!
            final = []
            for c in rrt.interpolate(n_i.conf, n_f.conf,
                                     stepSize=0.5 if fbch.inHeuristic else 0.25,
                                     moveChains=self.moveChains):
                for chain in self.robot.chainNames: #  fill in
                    if not chain in c.conf:
                        c[chain] = n_i.conf[chain]
                        c[chain] = n_i.cartConf()[chain]
                final.append(makeNode(c))
        return final

    # Returns list of nodes that go from initial node to final node
    def cartLineSteps(self, node_f, node_i, minLength):
        interp = self.cartInterpolators(node_f, node_i, minLength)
        if interp is None:
            return
        elif node_i == node_f:
            node_f.key = True
            return [node_f]
        else:
            nodes = [node_i]
            nodes.extend(interp[::-1])
            nodes.append(node_f)
            for node in nodes: node.key = True
            return nodes

    def jointInterpolators(self, n_f, n_i, minLength, depth=0):
        if n_f.conf == n_i.conf: return [n_f]
        final = []
        for c in rrt.interpolate(n_i.conf, n_f.conf,
                                 minLength, moveChains=self.params['moveChains']):
            for chain in self.robot.chainNames: #  fill in
                if not chain in c.conf:
                    c[chain] = n_i.conf[chain]
            final.append(makeNode(c))
        return final

    # Returns list of nodes that go from initial node to final node
    def jointLineSteps(self, node_f, node_i, minLength):
        interp = self.jointInterpolators(node_f, node_i, minLength)
        if interp is None:
            return
        elif node_i == node_f:
            node_f.key = True
            return [node_f]
        else:
            nodes = [node_i]
            nodes.extend(interp[::-1])
            nodes.append(node_f)
            node_i.key = True; node_f.key = True
            return nodes

    def robotSelfCollide(self, shape, heldDict={}):
        if fbch.inHeuristic: return False
        # Very sparse checks...
        checkParts = {'pr2LeftArm': ['pr2RightArm', 'pr2RightGripper', 'right'],
                      'pr2LeftGripper': ['pr2RightArm', 'pr2RightGripper', 'right'],
                      'pr2RightArm': ['left'],
                      'pr2RightGripper': ['left']}
        parts = dict([(part.name(), part) for part in shape.parts()])
        for h in ('right', 'left'):
            if h in heldDict and heldDict[h]:
                parts[h] = heldDict[h]
        for p in checkParts:
            pShape = parts.get(p, None)
            if not pShape: continue
            for check in checkParts[p]:
                if not check in parts: continue
                checkShape = parts[check]
                if pShape.collides(checkShape):
                    return True
        if heldDict:
            heldParts = [x for x in heldDict.values() if x]
        else:
            heldParts = [parts[p] for p in parts if p[:3] != 'pr2']
        return self.heldSelfCollide(heldParts)

    def heldSelfCollide(self, shape):
        shapeParts = shape if isinstance(shape, (list, tuple)) else shape.parts()
        if len(shapeParts) < 2:
            return False
        elif len(shapeParts) == 2:
            return shapeParts[0].collides(shapeParts[1])
        else:
            assert None, 'There should be at most two parts in attached'
        return False

    def edgeCollide(self, rob, key, collisions, robotShapes, attached, \
                    nodes, allObstacles, permanent, coll, viol, noViol):
        ecoll = collisions.get(key, None)
        if ecoll is None:
            ecoll = {}
            collisions[key] = ecoll
        if 'robotSelfCollision' in ecoll or 'heldSelfCollision' in ecoll:
            return None
        rshapes = robotShapes.get(key, None)
        if rshapes is None:
            rshapes = {}       # robot shapes
            robotShapes[key] = rshapes
            for node in nodes:
                if rob:
                    stats['newRob'] += 1
                else:
                    stats['newHeld'] += 1
                    robShape, attachedParts = node.conf.placementAux(getShapes=[],
                                                                     attached=attached)
                    parts = [x for x in attachedParts.values() if x]
                    rshapes[node] = shapes.Shape(parts, None)
                    if self.heldSelfCollide(rshapes[node]):
                        ecoll['heldSelfCollision'] = True
                        return None
        else:
            stats['oldRob' if rob else 'oldHeld'] += 1

        vset = viol.obstacles.union(viol.shadows)
        obst = []             # relevant obstacles
        permanentObst = []
        for obstacle in allObstacles:
            if not obstacle.parts() or obstacle in vset: continue
            if obstacle in ecoll:       # already checked
                val = ecoll[obstacle]
                if val:
                    if obstacle.name() in permanent or noViol:
                        return None
                    else:
                        coll.add(obstacle)
            else:                       # check this obstacle
                if obstacle.name() in permanent:
                    permanentObst.append(obstacle)
                else:
                    obst.append(obstacle)
        val = self.checkCollisions(obst, nodes, rshapes, ecoll, permanentObst, noViol)
        if val is None:
            # collision with permanent object, permanency can
            # change, so don't cache.
            return None
        if val and debug('colliders:collision'):
            wm.getWindow('W').clear()
            for n in rshapes: rshapes[n].draw('W', 'green')
            for o in val: o.draw('W', 'red')
            raw_input([o.name() for o in val])
        for o in obst:
            ecoll[o] = o in val
        for o in val:
            coll.add(o)
        return coll

    # !! Do all the collision tests in-place, without making new placements from scratch.
    def checkCollisions(self, obstacles, nodes, rshapes, ecoll, permanentObst, noViol):
        if len(nodes) == 0 or (len(obstacles) == 0 and len(permanentObst) == 0):
            return set([])
        elif len(nodes) == 1:
            node = nodes[0]
            if not node in rshapes:
                # Let the placement method do any caching
                placement = node.conf.placement(getShapes=True)
                if self.robotSelfCollide(placement):
                    ecoll['robotSelfCollision'] = True
                    return None
            else:
                placement = rshapes[node]
            if any(o for o in permanentObst if placement.collides(o)):
                return None
            coll = set([o for o in obstacles if placement.collides(o)])
            return None if (coll and noViol) else coll
        else:
            mid = len(nodes)/2
            c2 = self.checkCollisions(obstacles, nodes[mid:mid+1], rshapes, ecoll, permanentObst, noViol)
            if c2 is None: return None
            elif c2:
                obstacles = [o for o in obstacles if o not in c2]
            c1 = self.checkCollisions(obstacles, nodes[0:mid], rshapes, ecoll, permanentObst, noViol)
            if c1 is None: return None
            elif c1:
                obstacles = [o for o in obstacles if o not in c1]
            c3 = self.checkCollisions(obstacles, nodes[mid+1:len(nodes)], rshapes, ecoll, permanentObst, noViol)
            if c3 is None or (c3 and noViol): return None
            return c1.union(c2).union(c3)

    def addEdge(self, graph, node_f, node_i):
        if node_f == node_i: return
        if graph.edges.get((node_f, node_i), None) or graph.edges.get((node_i, node_f), None): return
        if not validEdge(node_i, node_f): return
        edge = Edge(node_f, node_i)
        if self.params['cartesian']:
            edge.nodes = self.cartLineSteps(node_f, node_i,
                                            2*minStep if fbch.inHeuristic else minStep)
        else:
            edge.nodes = self.jointLineSteps(node_f, node_i,
                                             2*minStep if fbch.inHeuristic else minStep)
        graph.edges[(node_f, node_i)] = edge
        for node in (node_f, node_i):
            if node in graph.incidence:
                graph.incidence[node].add(edge)
            else:
                graph.incidence[node] = set([edge])
        return edge

    # We want edge to depend only on endpoints so we can cache the
    # interpolated confs.  The collisions depend on the robot variance
    # as well as the particular obstacles (and their varince).
    def colliders(self, edge, pbs, prob, viol, noViol=False):
        shWorld = pbs.getShadowWorld(prob)
        attached = shWorld.attached
        coll = set([])
        empty = {}
        allObstacles = shWorld.getObjectShapes()
        permanent = shWorld.fixedObjects # set of names
        # We don't want to confuse the robot model during heuristic
        # with the one for regular planning.
        coll = self.edgeCollide(True, fbch.inHeuristic, edge.robotCollisions, edge.robotShapes,
                                attached, edge.nodes, allObstacles, permanent, coll, viol, noViol)
        if coll is None:
            return coll
        key = tuple([pbs.graspB[h] for h in ['left', 'right']] + [fbch.inHeuristic])
        coll = self.edgeCollide(False, key, edge.heldCollisions, edge.heldShapes,
                                attached, edge.nodes, allObstacles, permanent, coll, viol, noViol)
        return coll

    def checkPath(self, path, pbs, prob):
        newViol = viol0
        for conf in path:
            newViol, _ = self.confViolations(conf, pbs, prob, initViol=newViol)
            if newViol is None: return None
        return newViol

    def checkNodePathTest(self, graph, nodePath, pbs, prob):
        actual = self.checkPath(self.confPathFromNodePath(graph, nodePath), pbs, prob)
        test = self.checkNodePathTest(graph, nodePath, pbs, prob)
        if not actual == test:
            print 'actual', actual
            print 'test', test
            raw_input('Go?')
        return actual

    def checkNodePath(self, graph, nodePath, pbs, prob):
        ecoll = set([])
        v = nodePath[0]
        for w in nodePath[1:]:
            edge = graph.edges.get((v, w), None) or \
                   graph.edges.get((w, v), None)
            assert edge
            c = self.colliders(edge, pbs, prob, viol0)
            if c is None: return None
            ecoll = ecoll.union(c)
            v = w
        if ecoll is None:
            return None
        elif ecoll:
            shWorld = pbs.getShadowWorld(prob)
            fixed = shWorld.fixedObjects
            obstacleSet = set([sh for sh in shWorld.getNonShadowShapes() \
                               if not sh.name() in fixed])
            shadowSet = set([sh for sh in shWorld.getShadowShapes() \
                             if not sh.name() in fixed])
            obst = ecoll.intersection(obstacleSet)
            shad = ecoll.intersection(shadowSet)
            return Violations(obst, shad)
        else:
            return viol0

    def checkEdgePath(self, edgePath, pbs, prob):
        if len(edgePath) == 1:
            edge, end = edgePath[0]
            return self.confViolations(edge.ends[end].conf, pbs, prob)
        ecoll = set([])
        for edge, end in edgePath[:-1]:
            c = self.colliders(edge, pbs, prob, viol0)
            if c is None: return None
            ecoll = ecoll.union(c)
        if ecoll is None:
            return None
        elif ecoll:
            shWorld = pbs.getShadowWorld(prob)
            fixed = shWorld.fixedObjects
            obstacleSet = set([sh for sh in shWorld.getNonShadowShapes() \
                               if not sh.name() in fixed])
            shadowSet = set([sh for sh in shWorld.getShadowShapes() \
                             if not sh.name() in fixed])
            obst = ecoll.intersection(obstacleSet)
            shad = ecoll.intersection(shadowSet)
            return Violations(obst, shad)
        else:
            return viol0

    def confViolations(self, conf, pbs, prob, initViol=viol0, ignoreAttached=False):
        if initViol is None:
            return None, (None, None)
        shWorld = pbs.getShadowWorld(prob)
        attached = None if ignoreAttached else shWorld.attached
        robotShape, attachedPartsDict = conf.placementAux(attached=attached)
        attachedParts = [x for x in attachedPartsDict.values() if x]
        if debug('confViolations'):
            robotShape.draw('W', 'purple')
            for part in attachedParts: part.draw('W', 'purple')
        if self.heldSelfCollide(attachedParts):
            return None, (False, True)
        elif self.robotSelfCollide(robotShape, attachedPartsDict):
            return None, (True, False)
        fixed = shWorld.fixedObjects
        obstacleSet = set([sh for sh in shWorld.getNonShadowShapes() \
                           if not sh.name() in fixed])
        shadowSet = set([sh for sh in shWorld.getShadowShapes() \
                           if not sh.name() in fixed])
        allObstacles = shWorld.getObjectShapes()
        wObjs = set([])
        for obstacle in allObstacles:
            if not obstacle.parts(): continue
            heldVal = any(obstacle.collides(p) for p in attachedParts) # check this first
            robotVal = obstacle.collides(robotShape)
            if not (robotVal or heldVal): continue
            if obstacle.name() in fixed:
                # collision with fixed object
                if debug('confViolations'):
                    obstacle.draw('W', 'red')
                    debugMsg('confViolations', ('Collision with permanent', obstacle.name()))
                return None, (robotVal, heldVal)
            else:
                wObjs.add(obstacle)
        obst = wObjs.intersection(obstacleSet)
        shad = wObjs.intersection(shadowSet)
        if debug('confViolations'):
            debugMsg('confViolations',
                     ('obstacles:', [o.name() for o in obst]),
                     ('shadows:', [o.name() for o in shad]))
        return initViol.update(Violations(obst, shad)), (False, False)

    def testEdge(self, edge, pbs, prob, viol=viol0, optimize=False):
        shWorld = pbs.getShadowWorld(prob)
        fixed = shWorld.fixedObjects
        staticObstSet = set([sh for sh in shWorld.getObjectShapes() \
                             if sh.name() in fixed])
        wObjs = self.colliders(edge, pbs, prob, viol, noViol=optimize)
        if wObjs is None or not wObjs.isdisjoint(staticObstSet):
            return None
        obstacleSet = set([sh for sh in shWorld.getNonShadowShapes() \
                           if not sh.name() in fixed])
        shadowSet = set([sh for sh in shWorld.getShadowShapes() \
                           if not sh.name() in fixed])
        obst = wObjs.intersection(obstacleSet)
        shad = wObjs.intersection(shadowSet)
        return viol.combine(obst, shad)        

    def minViolPathGen(self, graph, startNode, targetNodes, pbs, prob, initViol=viol0,
                       optimize = False, draw=False, moveBase = True,
                       testFn = lambda x: True, goalCostFn = lambda x: 0):

        def testConnection(edge, viol):
            wObjs = self.colliders(edge, pbs, prob, viol, noViol=optimize)
            if wObjs is None or not wObjs.isdisjoint(staticObstSet):
                return None
            obst = wObjs.intersection(obstacleSet)
            shad = wObjs.intersection(shadowSet)
            return viol.combine(obst, shad)

        def successors(s):
            (v, viol) = s
            successors = []
            if not graph.incidence.get(v, []):
                if debug('successors'):
                    print v
                    raw_input('No edges are incident')
            for edge in graph.incidence.get(v, []):
                (a, b)  = edge.ends
                w = a if a != v else b
                if not moveBase:
                    if a.baseConf() != b.baseConf():
                        continue

                # !! Wholly unjustified...
                if useVisited:
                    if w in visited: continue
                    else: visited.add(w)

                nviol = testConnection(edge, viol)
                if nviol is None:
                    if debug('successors'):
                        print edge
                        pbs.draw(prob, 'W')
                        edge.draw('W', 'orange')
                        raw_input('Collision on edge')
                    continue
                successors.append((w, nviol))                  

            if draw or debug('successors'):
                pbs.draw(prob, 'W')
                v.conf.draw('W', color='cyan', attached=attached)
                color = colorGen.next()
                print v, '->'
                for (n, _) in successors:
                    n.conf.draw('W', color=color, attached=attached)
                    print '    ', n
                wm.getWindow('W').update()
                debugMsg('successors', 'Go?')
                
            return successors

        visited = set([])
        shWorld = pbs.getShadowWorld(prob)

        if draw or debug('successors'):
            colorGen = NextColor(20, s=0.6)
            wm.getWindow('W').clear()
            shWorld.draw('W')

        fixed = shWorld.fixedObjects
        staticObstSet = set([sh for sh in shWorld.getObjectShapes() \
                             if sh.name() in fixed])
        obstacleSet = set([sh for sh in shWorld.getNonShadowShapes() \
                           if not sh.name() in fixed])
        shadowSet = set([sh for sh in shWorld.getShadowShapes() \
                           if not sh.name() in fixed])
        attached = shWorld.attached
        prev = set([startNode])
        targets = [(goalCostFn(tnode), tnode) for tnode in targetNodes]
        targets.sort()

        if fbch.inHeuristic:
            # Static tests at init and target.  Ignore attached as a weakening.
            cv = self.confViolations(startNode.conf, pbs, prob,
                                     initViol=initViol, ignoreAttached=True)[0]
            if cv == None: return
            for (c, targetNode) in targets:
                cvt = self.confViolations(targetNode.conf, pbs, prob,
                                          initViol=cv, ignoreAttached=True)[0]
                if cvt == None or not testFn(targetNode): continue
                edge = Edge(startNode, targetNode)
                ans = (cvt,
                       objCollisionCost*len(cvt.obstacles)+shCollisionCost*len(cvt.shadows),
                       [(edge, 0), (edge, 1)])
                yield ans
            return
        
        if targets:                     # some targets remaining
            # Each node is (conf node, obj collisions, shadow collisions)
            if debug('expand'):
                wm.getWindow('W').clear()
                pbs.draw(prob, 'W')
                startNode.conf.draw('W','blue')
                targets[0][1].conf.draw('W','pink')
            gen = search.searchGen((startNode, initViol),
                                   [x[1] for x in targets],
                                   successors,
                                   # compute incremental cost...
                                   lambda s, a: (a, pointDist(s[0].point, a[0].point) + \
                                                 objCollisionCost * len(a[1].obstacles - s[1].obstacles) + \
                                                 shCollisionCost * len(a[1].shadows - s[1].shadows)),
                                   goalTest = testFn,
                                   heuristic = lambda s,g: pointDist(s.point, g.point),
                                   goalKey = lambda x: x[0],
                                   goalCostFn = goalCostFn,
                                   maxNodes = maxSearchNodes, maxExpanded = maxExpandedNodes,
                                   maxHDelta = None,
                                   expandF = minViolPathDebugExpand if debug('expand') else None,
                                   visitF = minViolPathDebugVisit if debug('expand') else None,
                                   greedy = searchOpt if optimize else searchGreedy,
                                   printFinal = True,
                                   verbose = False)
            for ans in gen:
                (path, costs) = ans
                if not path:
                    if debug('minViolPath'):
                        pbs.draw(prob, 'W')
                        for (c, tnode) in targets:
                            tnode.conf.draw('W', 'red')
                    yield None
                    return
                if debug('minViolPath'):
                    pbs.draw(prob, 'W')
                    for (_, p) in path:
                        p[0].conf.draw('W', 'green')
                # an entry in path is (action, (conf, violation))
                (_, (_, finalViolation)) = path[-1] # last path entry
                if debug('minViolPath'):
                    for ((_, (node, viol)) , cost) in zip(path, costs):
                        node.conf.draw('W', 'blue')
                        raw_input('v=%s, cost=%f'%(viol.weight(), cost))
                nodePath = [node for (_, (node, _)) in path]
                yield finalViolation, costs[-1], self.edgePathFromNodePath(graph, nodePath)

    def nodePathFromEdgePath(self, edgePath):
        return [edge.ends[end] for (edge, end) in edgePath]

    def edgePathFromNodePath(self, graph, nodePath):
        edgePath = []
        if len(nodePath) == 1:          # an edge case... har, har...
            edge = iter(graph.incidence[nodePath[0]]).next()
            if edge.ends[0] == nodePath[0]:
                return [(edge, 0)]
            else:
                return [(edge, 1)]
        nextEntry = None
        for i in range(len(nodePath)-1):
            this = nodePath[i]
            next = nodePath[i+1]
            edge = graph.edges.get((this, next), None)
            if edge:
                nextEntry = (edge, 1)
                edgePath.append((edge, 0))
            else:
                edge = graph.edges.get((next, this), None)
                if edge:
                    nextEntry = (edge, 0)
                    edgePath.append((edge, 1))
        edgePath.append(nextEntry)
        return edgePath

    def confPathFromEdgePath(self, edgePath):
        # Each edgePath element is an end point of an edge
        if fbch.inHeuristic:
            return [e.ends[end].conf for (e, end) in edgePath]
        if not edgePath: return []
        edge1, end1 = edgePath[0]
        confPath = [edge1.ends[end1].conf]
        if len(edgePath) > 1:
            for (edge2, end2) in edgePath:
                nodes = edge1.nodes[1:]
                if end1 == 1:
                    nodes.reverse()
                confPath.extend([n.conf for n in nodes])
                edge1, end1 = edge2, end2
        return confPath

    def confPathFromNodePath(self, graph, nodePathIn):
        confPath = []
        if fbch.inHeuristic:
            return [n.conf for n in nodePathIn]
        nodePath = [node for node in nodePathIn if node.key]
        if len(nodePath) > 1:
            for i in range(len(nodePath)-1):
                edge = graph.edges.get((nodePath[i], nodePath[i+1]), None)
                if edge:
                    confs = [n.conf for n in edge.nodes[::-1]]
                else:
                    edge = graph.edges.get((nodePath[i+1], nodePath[i]), None)
                    if edge:
                        confs = [n.conf for n in edge.nodes]
                assert edge
                if confPath:
                    confPath.extend(confs[1:])
                else:
                    confPath.extend(confs)
        elif len(nodePath) == 1:
            confPath = [nodePath[0].conf]
        return confPath

    def safePath(self, qf, qi, pbs, prob):
        for conf in rrt.interpolate(qf, qi, stepSize=minStep):
            newViol, _ = self.confViolations(conf, pbs, prob, initViol=viol0)
            if newViol is None or newViol.weight() > 0.:
                return False
        return True

    def smoothPath(self, path, pbs, prob, verbose=False, nsteps = glob.smoothSteps):
        n = len(path)
        if n < 3: return path
        if verbose: print 'Path has %s points'%str(n), '... smoothing'
        smoothed = path[:]
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
            if self.safePath(smoothed[j], smoothed[i], pbs, prob):
                count = 0
                smoothed[i+1:j] = []
                n = len(smoothed)
                if verbose: print 'Smoothed path length is', n
            else:
                count += 1
        return smoothed

    def __str__(self):
        return 'RoadMap:['+str(self.size+self.newSize)+']'

def bsEntails(bs1, p1, cacheValues, loose=False):
    for cacheValue in cacheValues:
        (bs2, p2, ans) = cacheValue
        (viol2, cost2, path2) = ans
        if debug('confReachViolCache'):
            print 'cached v=', viol2.weight() if viol2 else viol2
        if path2 and viol2 and (loose or viol2.empty()):
            # when theres a "clear" path and bs2 is "bigger", then we can use it
            # if there is some collision with obstacle or shadow, then it doesn't follow.
            if bsBigger(bs2, p2, bs1, p1) :
                return ans
        elif not viol2:
            if bsBigger(bs1, p1, bs2, p2):
                return ans
        elif debug('confReachViolCache'):
            print 'viol2', viol2
            print 'path2', path2
            raw_input('Huh?')
    if debug('confReachViolCache'):
        print 'bsEntails returns False'
    return None

def bsBigger(bs1, p1, bs2, p2):
    (held1, grasp1, conf1, avoid1, fix1, move1) = bs1.items()
    (held2, grasp2, conf2, avoid2, fix2, move2) = bs2.items()
    if held1 != held2:
        if debug('confReachViolCache'):
            print 'held are not the same'
            print '    held1', held1
            print '    held2', held2
        return False
    if not (avoid1 == avoid2 or not avoid2):
        if debug('confReachViolCache'):
            print 'avoid1 is not superset of avoid2'
            print '    avoid1', avoid1
            print '    avoid2', avoid2
        return False
    if not placesBigger(fix1, p1, fix2, p2):
        if debug('confReachViolCache'):
            print 'fix1 is not superset of fix2'
            print '    fix1', fix1
            print '    fix2', fix2
        return False
    if not placesBigger(move1, p1, move2, p2):
        if debug('confReachViolCache'):
            print 'move1 is not superset of move2'
            print '    move1', move1
            print '    move2', move2
        return False
    if grasp1 != grasp2:
        gr1 = dict(list(grasp1))
        gr2 = dict(list(grasp2))
        if not graspBigger(gr1, p1, gr2, p2):
            if debug('confReachViolCache'):
                print 'grasp1 is not bigger than grasp2'
                print '    grasp1', grasp1
                print '    grasp2', grasp2
                debugMsg('confReachViolCache', 'Go?')
            return False
    if debug('confReachViolCache'): print 'bsBigger = True'
    return True

def placesBigger(places1, p1, places2, p2):
    if debug('confReachViolCache'):
        print 'placesBigger'
        print '    places1', places1
        print '    places2', places2
    if not set([name for name, place in places2]) <= set([name for name, place in places1]):
        if debug('confReachViolCache'): print 'names are not superset'
        return False
    dict2 = dict(list(places2))
    for (name1, placeB1) in places1:
        if not name1 in dict2: continue
        if not placeBigger(placeB1, p1, dict2[name1], p2):
            if debug('confReachViolCache'): print placeB1, 'not bigger than', dict2[name1]
            return False
    return True

def placeBigger(pl1, p1, pl2, p2):
    if debug('confReachViolCache'):
        print 'placeBigger'
        print '    pl1', pl1
        print '    pl2', pl2
    if pl1 == pl2: return True
    if pl1.obj != pl2.obj: return False
    if pl1.poseD.muTuple != pl2.poseD.muTuple: return False
    w1 = shadowWidths(pl1.poseD.var, pl1.delta, p1)
    w2 = shadowWidths(pl2.poseD.var, pl2.delta, p2)
    if not all([x1 >= x2 for x1, x2 in zip(w1, w2)]):
        if debug('confReachViolCache'): print w1, 'not bigger than', w2
        return False
    else:
        if debug('confReachViolCache'): print w1, 'is bigger than', w2
    return True

def graspBigger(gr1, p1, gr2, p2):
    for hand in ('left', 'right'):
        g1 = gr1[hand]
        g2 = gr2[hand]
        if g1 == g2: continue
        if g1 is None or g2 is None: continue
        if g1.obj != g2.obj: return False
        if g1.poseD.muTuple != g2.poseD.muTuple: return False
        w1 = shadowWidths(g1.poseD.var, g1.delta, p1)
        w2 = shadowWidths(g2.poseD.var, g2.delta, p2)
        if not all([x1 >= x2 for x1, x2 in zip(w1, w2)]):
            return False
    return True

def cartChainName(chainName):
    if 'Gripper' in chainName or 'Torso' in chainName:
        return chainName
    else:
        return chainName+'Frame'

def pointDist(p1, p2):
    return sum([(a-b)**2 for (a, b) in zip(p1, p2)])**0.5

def validEdge(node_i, node_f):
    (xi, yi, thi) = node_i.conf['pr2Base']
    (xf, yf, thf) = node_f.conf['pr2Base']
    if max(abs(xi-xf), abs(yi-yf)) < 0.1:
        # small enough displacement
        return True
    if abs(util.angleDiff(math.atan2(yf-yi, xf-xi), thi)) <= 0.75*math.pi:
        # Not strictly back, so the head can look at where it's going
        return True
    return False

r = 0.02
boxPoint = shapes.Shape([shapes.BoxAligned(np.array([(-2*r, -2*r, -r), (2*r, 2*r, r)]), None),
                         shapes.BoxAligned(np.array([(2*r, -r, -r), (3*r, r, r)]), None)], None)
def minViolPathDebugExpand(n):
    (node, _) = n.state
    # node.conf.draw('W')
    # raw_input('expand')
    (x,y,th) = node.conf['pr2Base']
    boxPoint.applyTrans(util.Pose(x,y,0,th)).draw('W')
    wm.getWindow('W').update()

def minViolPathDebugVisit(state, cost, heuristicCost, a, newState, newCost, hValue):
    (node, _) = newState
    (x,y,th) = node.conf['pr2Base']
    boxPoint.applyTrans(util.Pose(x,y,0,th)).draw('W', 'cyan')
    wm.getWindow('W').update()

def reachable(graph, node):
    reach = set([])
    agenda = [node]
    while agenda:
        n = agenda.pop()
        edges = graph.incidence[n]
        if e.ends[0] == n: other = e.ends[1]
        else: other = e.ends[0]
        if other in reach: continue
        reach.add(other)
        agenda.append(other)
    return reach

def pickByName(ob, ls):
    for x in ls:
        if ob.name() == x.name():
            return x
