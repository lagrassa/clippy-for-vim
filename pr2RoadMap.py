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
from heapq import heappush, heappop
from planUtil import Violations
from pr2Util import NextColor, drawPath, shadowWidths, Hashable, combineViols, shadowp

considerReUsingPaths = True

violationCosts = (10.0, 2.0, 10.0, 5.0)

# Don't try too hard, fall back to the RRT when we can't find a path quickly
maxSearchNodes = 1000                   # 5000
maxExpandedNodes = 400                  # 2000

searchGreedy = 0.75 # greedy, trust that the heuristic is good...
searchOpt = 0.5     # should be 0.5 ideally, but it's slow...

useVisited = True           # unjustified
minStep = 0.25              # !! normally 0.25 for joint interpolation

confReachViolGenBatch = 10

hands = (0, 1)                          # left, right
handName = ('left', 'right')

nodePointRotScale = 0.1

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
        self.hVal = None
        self.parent = None
        Hashable.__init__(self)
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
            point = (x,y, nodePointRotScale*math.cos(th), nodePointRotScale*math.sin(th))
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
    def __init__(self, n1, n2, interpolator):
        global edge_idnum
        self.id = edge_idnum; edge_idnum += 1
        self.ends = (n1, n2)
        self.interpolator = interpolator
        self.nodes = None  # intermediate nodes
        # Cache for collisions on this edge.
        # aColl: {object : {True, False}}
        # hColl {graspB: {object : {True, False}}}
        # hsColl {graspB: {object : {True, False}}}
        self.aColl = {}
        self.hColl = {'left':{}, 'right':{}}
        self.hsColl = {'left':{}, 'right':{}}
        self.bbox = None
        # print 'Creating', self
        Hashable.__init__(self)
    def draw(self, window, color = 'cyan'):
        for node in self.nodes:
            node.draw(window, color=color)
    def getNodes(self):
        if self.nodes is None:
            node_i, node_f = self.ends
            self.nodes = self.interpolator(node_f, node_i, minStep)
        return self.nodes
    def desc(self):
        if not self.descValue:
            self.descValue = frozenset(self.ends)
        return self.descValue
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
        self.point = (x,y,nodePointRotScale*math.cos(th), nodePointRotScale*math.sin(th)) # used for nearest neighbors
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

# This should be used for creating violation
# coll is a collision description
def makeViolations(shWorld, coll):
    def fixed(obst):
        obstNames = set([o.name() for o in obst])
        return obstNames.intersection(fixedNames)
    if coll is None: return None
    # These xColl don't satisfy the non-overlapping condition for
    # Violations; that will be enforced when we return.
    (aColl, hColl, hsColl) = coll
    fixedNames = set(shWorld.fixedObjects)
    # Detect irretrievable conditions.
    if fixed(aColl) \
       or any((shWorld.fixedHeld[handName[h]] and fixed(hColl[h])) for h in hands) \
       or any((shWorld.fixedGrasp[handName[h]] and fixed(hsColl[h])) for h in hands):
        return None
    shad = [o for o in aColl if shadowp(o)]
    obst = [o for o in aColl if not o in shad]
    return Violations(obst, shad, hColl, hsColl)

def violToColl(viol):
    aColl = list(viol.obstacles)+list(viol.shadows)
    for h in hands:
        for o in viol.heldObstacles[h]:
            if o in aColl:
                raw_input('Inconsistent viol')
        for o in viol.heldShadows[h]:
            if o in aColl:
                raw_input('Inconsistent viol')
    hColl = tuple([list(viol.heldObstacles[h]) for h in hands])
    hsColl = tuple([list(viol.heldShadows[h]) for h in hands])
    return (aColl, hColl, hsColl)

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

    # NB: Returns a path from targetConf to startConf
    def confReachViol(self, targetConf, pbs, prob,
                      initViol=viol0, startConf = None, reversePath = False,
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
                                   pbs.held['right'].mode())),
                         ('initViol', initViol))
            if not (optimize or fbch.inHeuristic):
                key = (targetConf, initConf, moveBase)
                if not key in self.confReachCache:
                    self.confReachCache[key] = []
                self.confReachCache[key].append((pbs, prob,
                                                 ans if ans else (None, None, None)))

        def checkCache(key, type='full', loose=False):
            if fbch.inHeuristic or optimize: return 
            if key in self.confReachCache:
                if debug('traceCRH'): print '    cache?',
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
                elif considerReUsingPaths:
                    initObst = set(endPtViol.allObstacles())
                    initShadows = set(endPtViol.allShadows())
                    for i, cacheValue in enumerate(sortedCacheValues):
                        if i > 0: break # don't try too many times - just once?
                        (_, _, ans) = cacheValue
                        if not ans[0]: return None
                        (_, _, edgePath) = ans
                        viol2 = self.checkEdgePath(edgePath, pbs, prob)
                        if viol2 and set(viol2.allObstacles()) <= initObst and set(viol2.allShadows()) <= initShadows:
                            if debug('traceCRH'): print '    reusing path'
                            (_, cost, path) = ans
                            ans = (viol2, cost, path) # use the new violations
                            if debug('confReachViolCache'):
                                debugMsg('confReachViolCache', 'confReachCache reusing path')
                                print '    returning', ans
                            return ans
                        else:
                            print i, 'bad path', 
            else:
                self.confReachCache[key] = []
                if debug('confReachViolCache'): print 'confReachCache miss'

        def checkFullCache():
            return checkCache((targetConf, initConf, moveBase))

        def confAns(ans, show=False):
            displayAns(ans)
            if ans and ans[0]:
                (viol, cost, edgePath) = ans
                viol = viol.update(initViol)
                viol = viol.update(endPtViol)
                path = self.confPathFromEdgePath(edgePath)
                if show and debug('showPath') and not fbch.inHeuristic:
                    print 'confAns'
                    showPath(pbs, prob, path)
                if debug('verifyPath'):
                    if not self.checkPath(path, pbs, prob):
                        raw_input('Failed checkPath')
                if not (path[0] == targetConf and path[-1] == initConf):
                    raw_input('Path inconsistency')
                if finalConf: path = [finalConf] + path
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


        ######### procedure starts ##########

        initConf = startConf or self.homeConf
        initNode = makeNode(initConf)
        attached = pbs.getShadowWorld(prob).attached

        cv1 = self.confViolations(targetConf, pbs, prob)
        cv2 = self.confViolations(initConf, pbs, prob)
        endPtViol = combineViols(cv1, cv2)
        if endPtViol == None:
            if debug('endPoint:collision'):
                pbs.draw(prob, 'W')
                initConf.draw('W', 'blue')
                targetConf.draw('W', 'pink')
                print '    collision at end point'
                raw_input('okay?')
            return confAns(None)
        finalConf = None
        if not fbch.inHeuristic and targetConf in self.approachConfs:
            finalConf = targetConf
            targetConf = self.approachConfs[targetConf]
            if debug('traceCRH'): print '    using approach conf'
        targetNode = makeNode(targetConf)
        cached = checkFullCache()
        if cached:
            return confAns(cached)

        targetCluster = self.addToCluster(targetNode, connect=False)
        startCluster = self.addToCluster(initNode)
        graph = combineNodeGraphs(self.clusterGraph,
                                  startCluster.nodeGraph,
                                  targetCluster.nodeGraph)
        # if not fbch.inHeuristic:
        #     print '    Graph nodes =', len(graph.incidence), 'graph edges', len(graph.edges)
        if debug('traceCRH'): print '    find path',
        # search back from target... if we will execute in reverse, it's a double negative.
        ansGen = self.minViolPathGen(graph, targetNode, [initNode], pbs, prob,
                                     optimize=optimize, moveBase=moveBase,
                                     reverse = (not reversePath),
                                     useStartH = True)
        ans = next(ansGen, None)
        
        if ans == None:
            # (ans[0] and not ans[0].empty() and ans[0].names() != initViol.names()):
            print '    trying RRT'
            path, viol = rrt.planRobotPathSeq(pbs, prob, targetConf, initConf, endPtViol,
                                              maxIter=50, failIter=10)
            if viol:
                viol = viol.update(initViol)
            else:
                debugMsg('confReachViol', 'RRT failed')
            if not viol:
                pass
            elif ans == None or \
                 (set(viol.allObstacles()) < set(ans[0].allObstacles()) and \
                  set(viol.allShadows()) < set(ans[0].allShadows())):
                # print 'original viol', ans if ans==None else ans[0]
                # print 'RRT viol', viol
                print '    returning RRT ans'
                if not( path[0] == targetConf and path[-1] == initConf):
                    raw_input('Path inconsistency')
                if finalConf: path = [finalConf] + path
                return (viol, 0, path)
        cacheAns(ans)
        return confAns(ans, show=True)

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
        clusters = [self.rootCluster]
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
                    self.addEdge(self.clusterGraph, n0, n1, strict=True)
                for n2 in cl.reps:
                    n0 = cluster.addRep(n2)
                    self.addEdge(self.clusterGraph, n0, n2, strict=True)
        # scanH(self.clusterGraph, makeNode(self.homeConf))
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
                if self.confViolations(c, pbs, prob) != None:
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
                                     stepSize=2*minStep if fbch.inHeuristic else minStep,
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
        for c in rrt.interpolate(n_f.conf, n_i.conf,
                                 minLength, moveChains=self.params['moveChains']):
            for chain in self.robot.chainNames: #  fill in
                if not chain in c.conf:
                    c[chain] = n_i.conf[chain]
            final.append(makeNode(c))
        if not(n_i == final[0] and n_f == final[-1]):
            raw_input('Path inconsistency')
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
            for n in interp:
                if not n == nodes[-1]: nodes.append(n)
            if not node_f == nodes[-1]: nodes.append(node_f)
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
        for h in handName:
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
        if not shape: return False
        shapeParts = shape if isinstance(shape, (list, tuple)) else shape.parts()
        if len(shapeParts) < 2:
            return False
        elif len(shapeParts) == 2:
            return shapeParts[0].collides(shapeParts[1])
        else:
            assert None, 'There should be at most two parts in attached'
        return False

    def addEdge(self, graph, node_f, node_i, strict=False):
        if node_f == node_i: return
        if graph.edges.get((node_f, node_i), None) or graph.edges.get((node_i, node_f), None): return
        if strict:
            base_f = node_f.conf['pr2Base']
            base_i = node_i.conf['pr2Base']
            # Change position or angle, but not both.
            if all([f != i for (f,i) in zip(base_f, base_i)]): return
        if self.params['cartesian']:
            edge = Edge(node_f, node_i, self.cartLineSteps)
        else:
            edge = Edge(node_f, node_i, self.jointLineSteps)
        graph.edges[(node_f, node_i)] = edge
        for node in (node_f, node_i):
            if node in graph.incidence:
                graph.incidence[node].add(edge)
            else:
                graph.incidence[node] = set([edge])
        return edge

    def confCollidersAux(self, rob, obst, sumColl, edgeColl, perm, draw):
        if obst in sumColl: return True # known collision
        if edgeColl:
            if edgeColl.get('heldSelfCollision', False):
                if draw: raw_input('selfCollision')
                return None
            coll = edgeColl.get(obst, None) # check outcome in cache
        else:
            coll = None                 # not known
        if coll is None:
            coll = rob.collides(obst)   # outcome
        if coll:                        # collision
            sumColl.append(obst)        # record in summary
            if edgeColl:
                edgeColl[obst] = True   # store in cache
            if perm:                    # permanent
                if draw:
                    obst.draw('W', 'magenta')
                    raw_input('Collision with perm = %s'%obst.name())
                return None # irremediable
            return True                 # collision
        else:
            return False                # no collision

    def confColliders(self, pbs, prob, conf, aColl, hColl, hsColl, 
                      edge=None, ignoreAttached=False, draw=False):
        def heldParts(obj):
            parts = obj.parts()
            assert len(parts) == 2
            if shadowp(parts[0]):
                return (parts[1], parts[0])
            else:
                return (parts[0], parts[1])
        shWorld = pbs.getShadowWorld(prob)
        attached = None if ignoreAttached else shWorld.attached
        robShape, attachedPartsDict = conf.placementAux(attached=attached)
        attachedParts = [x for x in attachedPartsDict.values() if x]
        permanentNames = set(shWorld.fixedObjects) # set of names
        if draw:
            robShape.draw('W', 'purple')
            for part in attachedParts: part.draw('W', 'purple')
        # The self collision can depend on grasps - how to handle caching?
        if self.robotSelfCollide(robShape, attachedPartsDict):
            if draw:
                raw_input('selfCollision')
            return None
        for obst in shWorld.getObjectShapes():
            perm = obst.name() in permanentNames
            eColl = edge.aColl if edge else None
            res = self.confCollidersAux(robShape, obst, aColl, eColl,
                                        perm, draw)
            if res is None: return None
            elif res: continue          # collision
            if not attached or not any(attached.values()): continue
            for h in hands:
                hand = handName[h]
                if not attached[hand] or obst in hColl[h] or obst in hsColl[h]:
                    continue
                held, heldSh = heldParts(attachedPartsDict[hand])
                # check hColl
                if edge and pbs.graspB[hand] not in edge.hColl[hand]:
                    edge.hColl[hand][pbs.graspB[hand]] = {}
                eColl = edge.hColl[hand][pbs.graspB[hand]] if edge else None
                res = self.confCollidersAux(held, obst, hColl[h], eColl,
                                            (perm and shWorld.fixedHeld[hand]), draw)
                if res is None: return None
                elif res: continue          # collision
                # Check hsColl
                if edge and pbs.graspB[hand] not in edge.hsColl[hand]:
                    edge.hsColl[hand][pbs.graspB[hand]] = {}
                eColl = edge.hsColl[hand][pbs.graspB[hand]] if edge else None
                res = self.confCollidersAux(heldSh, obst, hsColl[h], eColl,
                                            (perm and shWorld.fixedGrasp[hand]), draw)
                if res is None: return None
        return True

    # We want edge to depend only on endpoints so we can cache the
    # interpolated confs.  The collisions depend on the robot variance
    # as well as the particular obstacles (and their varince).
    def colliders(self, edge, pbs, prob, initViol):
        if edge.aColl.get('robotSelfCollision', False): return None
        shWorld = pbs.getShadowWorld(prob)
        # Start with empty viol so caching reflects only edge
        (aColl, hColl, hsColl) = violToColl(viol0)
        for node in edge.getNodes():
            # updates aColl, hColl, hsColl by side-effect
            if self.confColliders(pbs, prob, node.conf, aColl, hColl, hsColl,
                                  edge=edge) is None:
                return None
        # Cache in the edge
        attached = shWorld.attached
        for obst in shWorld.getObjectShapes():
            if obst in aColl:
                # if obst not in edge.aColl:
                #     print 'Adding obst to edge', obst.name(), edge
                if obst in edge.aColl: assert edge.aColl[obst] == True
                edge.aColl[obst] = True
            else:
                if obst in edge.aColl: assert edge.aColl[obst] == False
                edge.aColl[obst] = False
                
            # The following may not be justified if we had a collision
            # with a permanent object in aColl and therefore did not
            # test an obst.

            # for h in hands:
            #     hand = handName[h]
            #     if attached[hand]:
            #         if edge and pbs.graspB[hand] not in edge.hColl[hand]:
            #             edge.hColl[hand][pbs.graspB[hand]] = {}
            #         edge.hColl[hand][pbs.graspB[hand]][obst] = (obst in hColl[h])
            #         if pbs.graspB[hand] not in edge.hsColl[hand]:
            #             edge.hsColl[hand][pbs.graspB[hand]] = {}
            #         edge.hsColl[hand][pbs.graspB[hand]][obst] = (obst in hsColl[h])
        viol = initViol.update(makeViolations(shWorld, (aColl, hColl, hsColl)))

        if debug('verifyPath'):
            testViol = self.checkPath([n.conf for n in edge.getNodes()],
                                      pbs, prob)
            testViol = testViol.update(initViol)
            if testViol.names() != viol.names():
                print 'colliders', viol
                print 'check    ', testViol
                print 'edge     ', edge
                raw_input('verifyPath: Inconsistent viols')

        return viol

    def checkPath(self, path, pbs, prob):
        newViol = viol0
        for conf in path:
            newViol = self.confViolations(conf, pbs, prob, initViol=newViol)
            if newViol is None:
                print 'viol', newViol
                pbs.draw(prob, 'W'); conf.draw('W', 'magenta')
                return None
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
        shWorld = pbs.getShadowWorld(prob)
        v = nodePath[0]
        viol = viol0
        for w in nodePath[1:]:
            edge = graph.edges.get((v, w), None) or \
                   graph.edges.get((w, v), None)
            assert edge
            viol = self.colliders(edge, pbs, prob, viol)
            if viol is None: return None
            v = w
        return viol

    def checkEdgePath(self, edgePath, pbs, prob):
        shWorld = pbs.getShadowWorld(prob)
        if len(edgePath) == 1:
            edge, end = edgePath[0]
            return self.confViolations(edge.ends[end].conf, pbs, prob)
        viol = viol0
        for edge, end in edgePath[:-1]:
            viol = self.colliders(edge, pbs, prob, viol)
            if viol is None: return None
        return viol

    def confViolations(self, conf, pbs, prob, initViol=viol0, ignoreAttached=False):
        if initViol is None:
            return None, (None, None)
        shWorld = pbs.getShadowWorld(prob)
        (aColl, hColl, hsColl) = violToColl(initViol)
        if self.confColliders(pbs, prob, conf, aColl, hColl, hsColl,
                              ignoreAttached=ignoreAttached,
                              draw=debug('confViolations')) is None:
            return None
        return makeViolations(shWorld, (aColl, hColl, hsColl))

    def testEdge(self, edge, pbs, prob, viol=viol0):
        return self.colliders(edge, pbs, prob, viol)

    def minViolPathGen(self, graph, startNode, targetNodes, pbs, prob, initViol=viol0,
                       optimize = False, draw=False, moveBase = True, reverse = False,
                       useStartH=False, testFn = lambda x: True, goalCostFn = lambda x: 0):

        def testConnection(edge, viol):
            return self.colliders(edge, pbs, prob, viol)

        def successors(s):
            (v, viol) = s
            succ = []
            if not graph.incidence.get(v, []):
                if debug('successors'):
                    print v
                    raw_input('No edges are incident')
            for edge in graph.incidence.get(v, []):
                (a, b)  = edge.ends
                w = a if a != v else b
                # Check for valid edge
                ve = validEdge(w, v) if reverse else validEdge(v, w)
                # ve = True
                if debug('successors'):
                    if reverse:
                        print ve, edge, w.conf['pr2Base'], '->', v.conf['pr2Base']
                    else:
                        print ve, edge, v.conf['pr2Base'], '->', w.conf['pr2Base']
                if not ve: continue
                if not moveBase:
                    if a.baseConf() != b.baseConf():
                        continue

                # !! Wholly unjustified...
                if useVisited:
                    if w in visited: continue
                    else: visited.add(w)

                nviol = testConnection(edge, viol)

                if debug('verifyPath'):
                    vi = all(self.confViolations(n.conf, pbs, prob) for n in edge.getNodes())
                    if (nviol == None and vi) or (nviol != None and not vi):
                        raw_input('successors: bad edge')

                if nviol is None:
                    if debug('successors'):
                        pbs.draw(prob, 'W')
                        edge.draw('W', 'orange')
                        raw_input('Collision on edge %s'%edge)
                    continue
                else:
                    succ.append((w, nviol))                  

            if draw or debug('successors'):
                pbs.draw(prob, 'W')
                v.conf.draw('W', color='cyan', attached=attached)
                color = colorGen.next()
                print v, '->'
                for (n, _) in succ:
                    n.conf.draw('W', color=color, attached=attached)
                    print '    ', n
                wm.getWindow('W').update()
                debugMsg('successors', 'Go?')
                
            return succ

        visited = set([])
        shWorld = pbs.getShadowWorld(prob)

        if draw or debug('successors'):
            colorGen = NextColor(20, s=0.6)
            wm.getWindow('W').clear()
            shWorld.draw('W')

        attached = shWorld.attached
        prev = set([startNode])
        targets = [(goalCostFn(tnode), tnode) for tnode in targetNodes]
        targets.sort()

        if fbch.inHeuristic:
            # Static tests at init and target.  Ignore attached as a weakening.
            cv = self.confViolations(startNode.conf, pbs, prob,
                                     initViol=initViol, ignoreAttached=True)
            if cv == None: return
            for (c, targetNode) in targets:
                if not moveBase:
                    if startNode.baseConf() != targetNode.baseConf():
                        return
                cvt = self.confViolations(targetNode.conf, pbs, prob,
                                          initViol=cv, ignoreAttached=True)
                if cvt == None or not testFn(targetNode): continue
                edge = Edge(startNode, targetNode, self.jointLineSteps)
                ans = (cvt, cvt.weight(violationCosts),
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
            # precomputed heuristic is distance from home, so only useful in reverse
            gen = search.searchGen((startNode, initViol),
                                   [x[1] for x in targets],
                                   successors,
                                   # compute incremental cost...
                                   lambda s, a: (a, pointDist(s[0].point, a[0].point) + \
                                                 a[1].weight(violationCosts) - \
                                                 s[1].weight(violationCosts)),
                                   goalTest = testFn,
                                   heuristic = lambda s,g: (useStartH and s.hVal) or pointDist(s.point, g.point),
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
                edgePath = self.edgePathFromNodePath(graph, nodePath)

                # print 'startNode', startNode
                # print 'targetNodes', targetNodes
                # print 'nodePath', nodePath
                # print 'edgePath', edgePath
                # raw_input('Node path and edge path')
                
                yield finalViolation, costs[-1], edgePath

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
                assert edge.ends[0] == this
                edgePath.append((edge, 0))
            else:
                edge = graph.edges.get((next, this), None)
                if edge:
                    nextEntry = (edge, 0)
                    assert edge.ends[1] == this
                    edgePath.append((edge, 1))
                else:
                    assert False, 'Missing edge in path'
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
            for (edge2, end2) in edgePath[1:]:
                nodes = edge1.getNodes()
                if end1 == 1:
                    nodes = nodes[::-1]
                confPath.extend([n.conf for n in nodes[1:]])
                edge1, end1 = edge2, end2

        # for conf in confPath:
        #     conf.draw('W'); raw_input('confPathFromEdgePath')

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
                    confs = [n.conf for n in edge.getNodes()[::-1]]
                else:
                    edge = graph.edges.get((nodePath[i+1], nodePath[i]), None)
                    if edge:
                        confs = [n.conf for n in edge.getNodes()]
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
            newViol = self.confViolations(conf, pbs, prob, initViol=viol0)
            if newViol is None or newViol.weight() > 0.:
                if debug('smooth'): conf.draw('W', 'red')
                return False
            else:
                if debug('smooth'): conf.draw('W', 'green')
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
            if debug('smooth'):
                pbs.draw(prob, 'W')
                for k in range(i, j+1):
                    smoothed[k].draw('W', 'blue')
                raw_input('Testing')
            if self.safePath(smoothed[j], smoothed[i], pbs, prob):
                count = 0
                if debug('smooth'):
                    raw_input('Safe')
                    pbs.draw(prob, 'W')
                    for k in range(i+1)+range(j,len(smoothed)):
                        smoothed[k].draw('W', 'blue')
                    raw_input('remaining')
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
    return validEdgeTest(node_i.conf['pr2Base'], node_f.conf['pr2Base'])

def validEdgeTest(xyt_i, xyt_f):
    (xi, yi, thi) = xyt_i
    (xf, yf, thf) = xyt_f
    if max(abs(xi-xf), abs(yi-yf)) < 0.01:
        # small enough displacement
        return True
    # Not strictly back, so the head can look at where it's going
    return abs(util.angleDiff(math.atan2(yf-yi, xf-xi), thi)) <= 0.75*math.pi

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

def scanH(graph, start):
    agenda = []
    expanded = set([])
    heappush(agenda, (0., start, None))
    count = 0
    links = set([])
    verts = set([])
    while not agenda == []:
        (cost, v, parent) = heappop(agenda)
        if v in expanded: continue
        v.hVal = cost
        v.parent = parent
        expanded.add(v)
        count += 1
        if debug('scanH'):
            (xv,yv,thv) = v.conf['pr2Base']
        for edge in graph.incidence.get(v, []):
            (a, b)  = edge.ends
            w = a if a != v else b
            if debug('scanH'):
                vert = tuple(w.conf['pr2Base'])
                if w not in verts:
                    (x,y,th) = vert
                    boxPoint.applyTrans(util.Pose(x,y,0,th)).draw('W')
                    verts.add(vert)
                link = ((xv, yv), (x, y))
                if not( x == xv and y == yv) and link not in links:
                    r = 0.02
                    length = ((xv - x)**2 + (yv - y)**2)**0.5
                    ray = shapes.BoxAligned(np.array([(-r, -r, -r), (length, r, r)]), None)
                    pose = util.Pose(xv, yv, 0.0, math.atan2(y-yv, x-xv))
                    ray.applyTrans(pose).draw('W', color='cyan')
                    links.add(link)
            heappush(agenda, (cost + pointDist(v.point, w.point), w, v))
        wm.getWindow('W').update()
    print 'Scanned', count, 'nodes'
    
def showPath(pbs, p, path):
    for c in path:
        pbs.draw(p, 'W')
        c.draw('W')
        raw_input('Next?')
    raw_input('Path end')

    

            
