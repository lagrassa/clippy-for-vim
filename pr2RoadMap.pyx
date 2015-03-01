from cpython cimport bool
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
from planGlobals import debugMsg, debug, debugDraw

from pr2Util import Violations, NextColor, drawPath, NextColor

cdef float objCollisionCost = 1.0
cdef float shCollisionCost = 0.5
cdef int maxSearchNodes = 5000
cdef int maxExpandedNodes = 1500
cdef float searchGreedy = 0.75
cdef float minStep = 0.2
cdef float minStepHeuristic = 0.4
cdef float confReachViolGenBatch = 5

cdef int node_idnum = 0
cdef class Node:
    def __init__(self, conf, cartConf, point):
        global node_idnum
        self.id = node_idnum; node_idnum += 1
        self.conf = conf
        self.cartConf = cartConf
        self.point = point
    def __str__(self):
        return 'Node:'+str(self.id)+str(self.point)
    def __hash__(self):
        return hash(tuple(self.point))

cdef int edge_idnum = 0
cdef class Edge:
    def __init__(self, n1, n2):
        global edge_idnum
        self.id = edge_idnum; edge_idnum += 1
        self.nodes = []                   # intermediate nodes
        # Cache for collisions on this edge.
        self.heldCollisions = {}     # grasp + h: {object : {True, False}}
        self.robotCollisions = {}    # h : {object : {True, False}}
        self.heldShapes = {}         # grasp + h :{node : shape}
        self.robotShapes = {}        # h : {node : shape}
        self.bbox = None
    def __str__(self):
        return 'Edge:'+str(self.id)
    def __hash__(self):
        return self.id
 
noViol = Violations()
cdef dict stats = {'newRob':0, 'newHeld':0, 'oldRob':0, 'oldHeld':0,
                   'newTest':0, 'oldTest':0}

# The Tree is rooted at the home location.  The moveChains indicate
# which parts of the robot are moving.  
cdef class RoadMap:
    def __init__(self, homeConf, world,
                 cartesian = True,        # type of interpolation
                 kdLeafSize = 20,
                 kNearest = 20,
                 moveChains=['pr2Base', 'pr2LeftArm']):
        self.robot = world.robot
        self.cartesian = cartesian
        self.moveChains = moveChains
        self.kdLeafSize = kdLeafSize
        self.homeConf = homeConf
        self.robotShapes = {}
        save = fbch.inHeuristic
        fbch.inHeuristic = True         # partial robot
        self.robotShapes[fbch.inHeuristic] = homeConf.placement()
        fbch.inHeuristic = False        # full robot
        self.robotShapes[fbch.inHeuristic] = homeConf.placement()
        fbch.inHeuristic = save
        cart = self.robot.forwardKin(homeConf)
        self.root = Node(homeConf, cart, self.pointFromCart(cart))
        self.nodeTooClose = 0.001      # is there a rational way of picking this?
        # The points in the main kdTree
        self.nodes = [self.root]
        self.points = np.array([self.root.point]) # array of point arrays
        self.size = 1
        self.kdTree = cKDTree(self.points, kdLeafSize)
        self.kNearest = kNearest
        # The points waiting to be inserted into the main tree
        self.newNodes = []
        self.newPoints = []             # list of point lists
        self.newSize = 0
        self.newKDTree = None
        # ----
        # Caches
        self.edges = {}
        self.confReachCache = {}
        self.confReachCacheHits = 0
        self.confReachPathFails = 0
        self.confReachCacheTotal = 0

    cpdef batchAddNodes(self, list confs):
        if not confs: return
        cdef int last = len(confs) - 1
        cdef int i
        for i in xrange(len(confs)):
            conf = confs[i]
            self.addNode(conf, merge = (i == last))   # merge on the last one

    cpdef mergeKDTrees(self):
        if self.newSize == 0: return
        self.points = np.vstack((self.points,
                                 np.array(self.newPoints)))
        self.kdTree = cKDTree(self.points, self.kdLeafSize)
        self.nodes.extend(self.newNodes)
        self.size += self.newSize
        self.newPoints = []
        self.newNodes = []
        self.newSize = 0
        self.newKDTree = None
        
    cpdef pointFromCart(self, cart, alpha = 0.1):
        point = []
        for chain in ['pr2Base', 'pr2LeftArm', 'pr2RightArm']: # self.moveChains
            if 'Gripper' in chain or 'Torso' in chain:
                point.append(cart[chain][0])
            else:
                pose = cart[chain] 
                pose_o = [pose.matrix[i,3] for i in range(3)]
                point.extend(pose_o)
                #for j in range(2):            # x and y axes
                #    point.extend([alpha*pose.matrix[i,j] + pose_o[i] for i in range(3)])
        return point
            
    cpdef addNode(self, conf, merge = True):
        cart = self.robot.forwardKin(conf)
        point = self.pointFromCart(cart)
        n_new = Node(conf, cart, point)
        (d, nc) = self.nearest(n_new, 1)[0]
        if d <= self.nodeTooClose and conf == nc.conf:
            if debug('addNode'):
                print 'New', conf
                print 'Old', nc.conf
                raw_input('Node too close, d=%s'%d)
            return nc
        self.newPoints.append(point)
        self.newNodes.append(n_new)
        self.newSize += 1
        self.newKDTree = None
        if merge and self.newSize > self.kdLeafSize:
            self.mergeKDTrees()
        return n_new

    # Is it better just to make a new global tree when we do addConf
    cpdef nearest(self, Node node, int k):
        cdef:
            list merge = []
            float d
            int i
        dists, ids = self.kdTree.query(node.point, k)
        if k == 1:
            dists = [dists]; ids = [ids]
        if self.newPoints:
            if not self.newKDTree:
                self.newKDTree = cKDTree(self.newPoints, self.kdLeafSize)
        else:
            return [(d, self.nodes[i]) for d, i in zip(dists, ids)]
        for (d, i) in zip(dists, ids):
            if d < np.inf: merge.append((d, i, False))
        assert self.newKDTree
        newDists, newIds = self.newKDTree.query(node.point, k)
        if k == 1:
            newDists = [newDists]; newIds = [newIds]
        for (d, i) in zip(newDists, newIds):
            if d < np.inf: merge.append((d, i, True))
        merge.sort()
        return [(d, self.newNodes[i]) if new else (d, self.nodes[i]) \
                for (d, i, new) in merge[:k]]

    cpdef close(self, node_f, node_i, thr, cart):
        if cart:
            c_f = node_f.cartConf
            c_i = node_i.cartConf
            for chain in self.moveChains:
                dist = c_f[chain].near(c_i[chain], 0.1, 0.1)   # ELIMINATE CONSTANTS
                if not dist <= thr: return False
            return True
        else:
            return node_f.conf == node_i.conf

    cpdef drawNodes(self, color='yellow'):
        for node in self.nodes + self.newNodes:
            node.conf.draw('W', color=color)

    cpdef interpPose(self, pose_f, pose_i, minLength, ratio=0.5):
        if isinstance(pose_f, list):
            return [f*ratio + i*(1-ratio) for (f,i) in zip(pose_f, pose_i)], \
                   all([abs(f-i)<=minLength for (f,i) in zip(pose_f, pose_i)])
        else:
            pr = pose_f.point()*ratio + pose_i.point()*(1-ratio)
            qr = quaternion_slerp(pose_i.quat().matrix, pose_f.quat().matrix, ratio)
            return util.Transform(None, pr.matrix, qr), \
                   pose_f.near(pose_i, minLength, minLength)

    cpdef cartInterpolators(self, n_f, n_i, minLength, depth=0):
        if depth > 10:
            raw_input('cartInterpolators depth > 10')
        if n_f.conf == n_i.conf: return [n_f]
        c_f = n_f.cartConf
        c_i = n_i.cartConf
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
        cart = n_i.cartConf.copy()
        cart.conf = newVals
        conf = self.robot.inverseKin(cart, conf=n_i.conf, complain=debug('cartInterpolators'))
        for chain in self.robot.chainNames: #  fill in
            if not chain in conf.conf:
                conf[chain] = n_i.conf[chain]
                cart[chain] = n_i.cartConf[chain]
        if all([conf[chain] for chain in self.moveChains]):
            newNode =  Node(conf, cart, self.pointFromCart(cart))
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
                cart = self.robot.forwardKin(c)
                for chain in self.robot.chainNames: #  fill in
                    if not chain in c.conf:
                        c[chain] = n_i.conf[chain]
                        c[chain] = n_i.cartConf[chain]
                final.append(Node(c, cart, self.pointFromCart(cart)))
        return final

    # Should this fall back on joint interpolation when necessary?
    # Returns list of nodes that go from initial node to final node
    cpdef cartLineSteps(self, node_f, node_i, minLength):
        if fbch.inHeuristic:            # !! ?
            return [node_i, node_f]
        interp = self.cartInterpolators(node_f, node_i, minLength)
        if interp is None:
            return
        elif node_i == node_f:
            return [node_f]
        else:
            nodes = [node_i]
            nodes.extend(interp[::-1])
            nodes.append(node_f)
            return nodes

    cpdef robotSelfCollide(self, shape):
        if fbch.inHeuristic: return False
        # Very sparse checks...
        cdef dict checkParts = {'pr2LeftArm': ['pr2RightArm', 'pr2RightGripper'],
                                'pr2LeftGripper': ['pr2RightArm', 'pr2RightGripper'],
                                'pr2RightArm': [],
                                'pr2RightGripper': []}
        cdef dict parts = dict([(part.name(), part) for part in shape.parts()])
        for p in checkParts:
            pShape = parts[p]
            for check in checkParts[p]:
                if not check in parts: continue
                checkShape = parts[check]
                if pShape.collides(checkShape):
                    return True
        heldParts = [parts[p] for p in parts if p[:3] != 'pr2']
        return self.heldSelfCollide(heldParts)

    cpdef heldSelfCollide(self, shape):
        shapeParts = shape if isinstance(shape, (list, tuple)) else shape.parts()
        if len(shapeParts) < 2:
            return False
        elif len(shapeParts) == 2:
            return shapeParts[0].collides(shapeParts[1])
        else:
            assert None, 'There should be at most two parts in attached'
        return

    # The compiler does not like this as cpdef...
    def edgeCollide(self, bool rob, key, dict collisions, dict robotShapes, dict attached, \
                    list nodes, list allObstacles, set permanent, set coll):
        cdef dict ecoll = collisions.get(key, None)
        if ecoll is None:
            ecoll = {}
            collisions[key] = ecoll
        if 'robotSelfCollision' in ecoll or 'heldSelfCollision' in ecoll:
            return None
        cdef dict rshapes = robotShapes.get(key, None)
        if rshapes is None:
            rshapes = {}       # robot shapes
            robotShapes[key] = rshapes
            for node in nodes:
                if rob:
                    stats['newRob'] += 1
                    rshapes[node] = node.conf.placement()
                    if self.robotSelfCollide(rshapes[node]):
                        ecoll['robotSelfCollision'] = True
                        return None
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

        for obstacle in allObstacles:
            if not obstacle.parts: continue
            val = False
            if obstacle in ecoll:
                stats['oldTest'] += 1
                val = ecoll[obstacle]
            else:
                stats['newTest'] += 1
                val = any(obstacle.collides(rshapes.get(node)) \
                          for  node in nodes)
                ecoll[obstacle] = val
                if val and debug('colliders:collision'):
                    wm.getWindow('W').clear()
                    for n in rshapes: rshapes[n].draw('W', 'green')
                    obstacle.draw('W', 'red')
                    raw_input(obstacle.name())
            if val is None or (val and (obstacle.name() in permanent)):
                # collision with permanent object, permanency can
                # change, so don't cache.
                return None
            if val:               # a collision
                coll.add(obstacle) # previous collision
        return coll

    # We want edge to depend only on endpoints so we can cache the
    # interpolated confs.  The collisions depend on the robot variance
    # as well as the particular obstacles (and their varince).
    cpdef colliders(self, node_f, node_i, bState, prob, avoidShadow=[], attached=None):
        shWorld = bState.getShadowWorld(prob, avoidShadow)
        coll = set([])
        empty = {}
        edge = self.edges.get((node_f, node_i), empty) or \
               self.edges.get((node_i, node_f), empty)
        if not edge:
            edge = Edge(node_f, node_i)
            edge.nodes = self.cartLineSteps(node_f, node_i,
                                            minStepHeuristic if fbch.inHeuristic else minStep)
            self.edges[(node_f, node_i)] = edge
        allObstacles = shWorld.getObjectShapes()
        permanent = shWorld.fixedObjects # set of names
        # We don't want to confuse the robot model during heuristic
        # with the one for regular planning.
        coll = self.edgeCollide(True, fbch.inHeuristic, edge.robotCollisions, edge.robotShapes,
                                attached, edge.nodes, allObstacles, permanent, coll)
        if coll is None:
            return coll
        key = tuple([bState.graspB[h] for h in ['left', 'right']] + [fbch.inHeuristic])
        coll = self.edgeCollide(False, key, edge.heldCollisions, edge.heldShapes,
                                attached, edge.nodes, allObstacles, permanent, coll)
        return coll

    def confViolations(self, conf, bState, prob,
                       initViol=noViol,
                       avoidShadow=[], attached=None,
                       additionalObsts = []):
        shWorld = bState.getShadowWorld(prob, avoidShadow)
        robotShape, attachedParts = conf.placementAux(attached=attached)
        attachedParts = [x for x in attachedParts.values() if x]
        if debug('confViolations'):
            robotShape.draw('W', 'purple')
            for part in attachedParts: part.draw('W', 'purple')
        if self.heldSelfCollide(attachedParts):
            return None, (False, True)
        elif self.robotSelfCollide(robotShape):
            return None, (True, False)
        fixed = shWorld.fixedObjects
        obstacleSet = set([sh for sh in shWorld.getNonShadowShapes() \
                           if not sh.name() in fixed])
        shadowSet = set([sh for sh in shWorld.getShadowShapes() \
                           if not sh.name() in fixed])
        allObstacles = shWorld.getObjectShapes()
        wObjs = set([])
        for obstacle in allObstacles + additionalObsts:
            if not obstacle.parts: continue
            heldVal = any(obstacle.collides(p) for p in attachedParts) # check this first
            robotVal = obstacle.collides(robotShape)
            if not (robotVal or heldVal): continue
            if (obstacle.name() in fixed or obstacle in additionalObsts):
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
        return initViol.combine(obst, shad), (False, False)

    # !! Should do additional sampling to connect confs.
    def minViolPathGen(self, targetConfNodes, bState, prob, avoidShadow=[], startConf = None,
                       initViol=noViol, objCost=1.0, shCost=0.5, maxNodes=maxSearchNodes, 
                       testFn = lambda x: True,
                       attached = None, goalCostFn = lambda x: 0, draw=False, maxExpanded= maxExpandedNodes):

        def testConnection(v, w, viol):
            wObjs = self.colliders(v, w, bState, prob, attached=attached)
            if wObjs is None or not wObjs.isdisjoint(staticObstSet):
                return None
            obst = wObjs.intersection(obstacleSet)
            shad = wObjs.intersection(shadowSet)
            return viol.combine(obst, shad)

        def successors(s):
            (v, viol) = s
            successors = []
            nearNodes = self.nearest(v, self.kNearest)

            if draw:
                v.conf.draw('W', 'cyan')
                wm.getWindow('W').update()
                print v.cartConf['pr2LeftArm'].point()

            if draw or debug('successors'):
                color = colorGen.next()
                for (_, n) in nearNodes:
                    n.conf.draw('W', color)
                print [c for (c,_) in nearNodes]
                raw_input('successors')

            if debug('successors'):
                print v, '->'
            for (_, w) in nearNodes:
                if debug('successors'):
                    print '    ', w
                nviol = testConnection(v, w, viol)
                if nviol is None: continue
                successors.append((w, nviol))
            return successors

        def processPath(nodePath):
            confPath = []
            if len(nodePath) > 1:
                for i in range(len(nodePath)-1):
                    edge = self.edges.get((nodePath[i], nodePath[i+1]), None)
                    if edge:
                        confs = [n.conf for n in edge.nodes[::-1]]
                    else:
                        edge = self.edges.get((nodePath[i+1], nodePath[i]), None)
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

        shWorld = bState.getShadowWorld(prob, avoidShadow)

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
        attached = attached or shWorld.attached
        initConf = startConf or self.homeConf
        initConfNode = self.addNode(initConf)  # have it stored??
        targets = [(goalCostFn(tnode), tnode) for tnode in targetConfNodes]
        targets.sort()
        for (cost, tnode) in targets:
            finalViolation = testConnection(initConfNode, tnode, initViol)
            if finalViolation and finalViolation.empty():  # only if no collision
                targets.remove((cost, tnode))
                nodePath = [initConfNode, tnode]
                confPath = processPath(nodePath)
                # finalViolation, finalCost, path (list of confs)
                yield finalViolation, cost, confPath
        if targets:                     # some targets remaining
            # Each node is (conf node, obj collisions, shadow collisions)
            gen = search.searchGen((initConfNode, initViol),
                                   [x[1] for x in targets],
                                   successors,
                                   # compute incremental cost...
                                   lambda s, a: (a, pointDist(s[0].point, a[0].point) + \
                                                 objCost * len(a[1].obstacles - s[1].obstacles) + \
                                                 shCost * len(a[1].shadows - s[1].shadows)),
                                   goalTest = testFn,
                                   heuristic = lambda s,g: pointDist(s.point, g.point),
                                   goalKey = lambda x: x[0],
                                   goalCostFn = goalCostFn,
                                   maxNodes = maxNodes, maxExpanded = maxExpanded,
                                   maxHDelta = None,
                                   greedy = searchGreedy, printFinal = False)
            for ans in gen:
                (path, costs) = ans
                if not path:
                    if debug('minViolPath'):
                        bState.draw(prob, glob.debugWin)
                        for tnode in targets:
                            tnode.conf.draw('W', 'red')
                    yield None, 1e10, None
                    return
                if debug('minViolPath'):
                    self.bState.draw(prob, glob.debugWin)
                    for (_, p) in path:
                        p[0].conf.draw(glob.debugWin, 'green')
                # an entry in path is (action, (conf, violation))
                (_, (_, finalViolation)) = path[-1] # last path entry
                if debug('minViolPath'):
                    for ((_, (node, viol)) , cost) in zip(path, costs):
                        node.conf.draw('W', 'blue')
                        raw_input('v=%s, cost=%f'%(viol.weight(), cost))
                nodePath = [node for (_, (node, _)) in path]
                confPath = processPath(nodePath)
                # finalViolation, finalCost, path (list of confs)
                yield finalViolation, costs[-1], confPath

    # Cached version of the call to minViolPath
    def confReachViol(self, targetConf, bState, prob, initViol=noViol, avoidShadow = [],
                        objCost = objCollisionCost, shCost = shCollisionCost,
                        maxNodes = maxSearchNodes, startConf = None, attached = None):
        def grasp(hand):
            g = bState.graspB[hand]
            if g == None or g.obj == 'none':
                return None
            else:
                return g
        def exitWithAns(ans):
            self.confReachCache[key].append((bState, prob, avoidShadow, ans))
            if not ans:
                self.confReachPathFails += 1
            if ans and ans[0] and ans[2] and not fbch.inHeuristic:
                (viol, cost, path) = ans
                if debug('confReachViol'):
                    drawPath(path, viol=viol,
                             attached=bState.getShadowWorld(prob).attached)
                    debugMsg('confReachViol', ('->', (viol, cost, 'path len = %d'%len(path))))
            else:
                debugMsg('confReachViol', ('->', ans))
            return ans
            
        if fbch.inHeuristic:
            prob = 0.99*prob             # make slightly easier
        if attached == None:
            attached = bState.getShadowWorld(prob).attached
        # key = (targetConf, startConf, prob, bState, tuple(avoidShadow), initViol),
        key = (targetConf, startConf, initViol, fbch.inHeuristic)
        if debug('confReachViolCache'):
            debugMsg('confReachViolCache',
                     ('targetConf', targetConf.conf),
                     ('startConf', startConf.conf if startConf else None),
                     ('prob', prob),
                     ('moveObjBs', bState.moveObjBs),
                     ('fixObjBs', bState.fixObjBs),
                     ('held', (bState.held['left'].mode(),
                               bState.held['right'].mode(),
                               grasp('left'), grasp('right'))),
                     ('initViol', ([x.name() for x in initViol.obstacles],
                                   [x.name() for x in initViol.shadows]) if initViol else None),
                     ('avoidShadow', avoidShadow))
        self.confReachCacheTotal += 1
        if key in self.confReachCache:
            if debug('confReachViolCache'): print 'confReachCache tentative hit'
            cacheValue = self.confReachCache[key]
            ans = bsEntails(bState, prob, avoidShadow, cacheValue)
            if ans != None:
                if debug('confReachViolCache'):
                    raw_input('confReachCache actual hit')
                    print '    returning', ans
                return ans   # actual answer
        else:
            self.confReachCache[key] = []
            if debug('confReachViolCache'): print 'confReachCache miss'

        initConf = startConf or self.homeConf
        if debug('confReachViol') and \
            (not fbch.inHeuristic  or debug('drawInHeuristic')):
            bState.draw(prob, 'W')
            initConf.draw('W', 'blue', attached=attached)
            targetConf.draw('W', 'pink', attached=attached)
            print 'startConf is blue; targetConf is pink'
            raw_input('confReachViol - Go?')

        cv = self.confViolations(targetConf, bState, prob,
                                 avoidShadow=avoidShadow, attached=attached)[0]
        cv = self.confViolations(initConf, bState, prob, initViol=cv,
                                 avoidShadow=avoidShadow, attached=attached)[0]
        if cv is None:
            if debug('confReachViol'):
                print 'targetConf is unreachable'
            return exitWithAns((None, None, None))
        cvi = initViol.combine(cv.obstacles, cv.shadows)
        if initConf == targetConf:
            return exitWithAns((cvi, 0, [targetConf]))
        if fbch.inHeuristic:
            ans = cvi, objCost*len(cvi.obstacles)+shCost*len(cvi.shadows), [initConf, targetConf]
            return exitWithAns(ans)
        node = self.addNode(targetConf)
        gen = self.minViolPathGen([node], bState, prob, avoidShadow,
                                  startConf, initViol,
                                  objCost, shCost, maxNodes, attached=attached)
        return exitWithAns(gen.next())

    # Cached version of the call to minViolPath
    def confReachViolGen(self, targetConfs, bState, prob, initViol=noViol, avoidShadow = [],
                         objCost = objCollisionCost, shCost = shCollisionCost,
                         maxNodes = maxSearchNodes, testFn = lambda x: True, goalCostFn = lambda x: 0,
                         startConf = None, attached = None, draw=False):
        if fbch.inHeuristic:
            prob = 0.99*prob             # make slightly easier
        if attached == None:
            attached = bState.getShadowWorld(prob).attached
        initConf = startConf or self.homeConf
        batchSize = confReachViolGenBatch
        batch = 0
        while True:
            # Collect the next batach of trialConfs
            batch += 1
            trialConfs = []
            count = 0
            for c in targetConfs:       # targetConfs is a generator
                if self.confViolations(c, bState, prob, avoidShadow=avoidShadow, attached=attached)[0] != None:
                    count += 1
                    trialConfs.append(c)
                    if initConf == c:
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
                bState.draw(prob, 'W')
                initConf.draw('W', 'blue', attached=attached)
                for trialConf in trialConfs:
                    trialConf.draw('W', 'pink', attached=attached)
                print 'startConf is blue; targetConfs are pink'
                raw_input('confReachViolGen - Go?')
            
            # keep track of the original conf for the nodes
            nodeConf = dict([(self.addNode(tc), tc) for tc in trialConfs])
            origConf = dict([(nc.conf, c) for (nc, c) in nodeConf.items()])
            nodeTestFn = lambda n: testFn(nodeConf[n])
            goalNodeCostFn = lambda n: goalCostFn(nodeConf[n])
            gen = self.minViolPathGen(nodeConf.keys(),
                                      bState, prob, avoidShadow,
                                      initConf, initViol or Violations(),
                                      objCost, shCost, maxNodes, nodeTestFn, attached,
                                      goalCostFn=goalNodeCostFn, draw=draw)
            for ans in gen:
                if ans and ans[0] and ans[2]:
                    (viol, cost, path) = ans
                    pathOrig = path[:-1] + [origConf[path[-1]]]
                    if debug('confReachViolGen') and not fbch.inHeuristic:
                        drawPath(pathOrig, viol=viol,
                                 attached=bState.getShadowWorld(prob).attached)
                        debugMsg('confReachViolGen', ('->', (viol, cost, 'path len = %d'%len(pathOrig))))
                    yield (viol, cost, pathOrig)
                else:
                    if not fbch.inHeuristic:
                        debugMsg('confReachViolGen', ('->', ans))
                    break
        ans = None, 0, []
        debugMsg('confReachViolGen', ('->', ans))
        yield ans
        return

    def __str__(self):
        return 'RoadMap:['+str(self.size+self.newSize)+']'

cdef tuple bsEntails(bs1, p1, avoid1, cacheValues):
    for cacheValue in cacheValues:
        (bs2, p2, avoid2, ans) = cacheValue
        (viol2, cost2, path2) = ans
        if path2 and viol2 and viol2.empty():
            # when theres a "clear" path and bs2 is "bigger", then we can use it
            # if there is some collision with obstacle or shadow, then it doesn't follow.
            if bsBigger(bs2, bs1) and p2 >= p1 and (avoid1 == avoid2 or not avoid1):
                return ans
        elif not viol2:
            if bsBigger(bs1, bs2) and p2 <= p1 and (avoid1 == avoid2 or not avoid2):
                return ans
    if debug('confReachViolCache'):
        print 'bsEntails returns False'
    return None

cdef bool bsBigger(bs1, bs2):
    (held1, grasp1, conf1, fix1, move1) = bs1.items()
    (held2, grasp2, conf2, fix2, move2) = bs2.items()
    if held1 != held2:
        if debug('confReachViolCache'):
            print 'held are not the same'
            print '    held1', held1
            print '    held2', held2
        return False
    if not fix2 <= fix1:
        if debug('confReachViolCache'):
            print 'fix1 is not superset of fix2'
            print '    fix1', fix1
            print '    fix2', fix2
        return False
    if not move2 <= move1:
        if debug('confReachViolCache'):
            print 'move1 is not superset of move2'
            print '    move1', move1
            print '    move2', move2
        return False
    if grasp1 != grasp2:
        gr1 = dict(list(grasp1))
        gr2 = dict(list(grasp2))
        if not graspBigger(gr1, gr2):
            if debug('confReachViolCache'):
                print 'grasp1 is not bigger than grasp2'
                print '    grasp1', grasp1
                print '    grasp2', grasp2
                raw_input('Go?')
            return False
    if debug('confReachViolCache'): 'bsBigger = True'
    return True

cdef bool graspBigger(gr1, gr2):
    for hand in ('left', 'right'):
        g1 = gr1[hand]
        g2 = gr2[hand]
        if g1 == g2: continue
        if g1.obj != g2.obj: return False
        if g1.poseD.muTuple != g2.poseD.muTuple: return False
        w1 = [v+d for (v,d) in zip(g1.poseD.muTuple, g1.delta)]
        w2 = [v+d for (v,d) in zip(g2.poseD.muTuple, g2.delta)]
        if not all([x1 >= x2 for x1, x2 in zip(w1, w2)]):
            return False
    return True

cdef str cartChainName(str chainName):
    if 'Gripper' in chainName or 'Torso' in chainName:
        return chainName
    else:
        return chainName+'Frame'

def pointDist(p1, p2):
    return sum([(a-b)**2 for (a, b) in zip(p1, p2)])**0.5
