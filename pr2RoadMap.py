import math
import random
import copy
import time
import pdb
import hu
from scipy.spatial import cKDTree
import windowManager3D as wm
import numpy as np
import shapes
from ranges import *
from objects import WorldState
from geom import bboxOverlap, bboxUnion, bboxCenter
from transformations import quaternion_slerp
import pr2RRT as rrt
reload(rrt)
import planGlobals as glob
from traceFile import debugMsg, debug, trAlways
from miscUtil import prettyString
from heapq import heappush, heappop
from planUtil import Violations
from pr2Util import NextColor, drawPath, shadowWidths, Hashable, combineViols, shadowp, removeDuplicateConfs
from traceFile import tr
from pr2Robot import CartConf, compileObjectFrames, compileAttachedFrames
from gjk import chainCollides, confPlaceChains, confSelfCollide, chainBBoxes

considerReUsingPaths = True

violationCosts = (10.0, 2.0, 10.0, 5.0)

minGripperDistance = 0.25

# Don't try too hard, fall back to the RRT when we can't find a path quickly
maxSearchNodes = 1000                   # 1000
maxExpandedNodes = 400                  # 400

searchGreedy = 0.5 # 0.75 # greedy, trust that the heuristic is good...
searchOpt = 0.5     # should be 0.5 ideally, but it's slow...

useVisited = True           # unjustified
minStep = glob.rrtInterpolateStepSize # !! normally 0.25 for joint interpolation

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
        return tuple(self.conf.baseConf())
    def pointFromConf(self, conf):
        x,y,th=conf.baseConf()
        point = (x,y, nodePointRotScale*math.cos(th), nodePointRotScale*math.sin(th))
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

class KDTree:                           # the trivial version
    def __init__(self, entries, kdLeafSize = 20):
        self.entries = entries
        points = [e.point for e in entries]
        self.points = np.array(points)  # array of point arrays
        self.size = len(points)
        self.entryTooClose = 0.001**2   # we compute dist^2 in nearest

    def allEntries(self):
        return self.entries + self.newEntries

    def batchAddEntries(self, entries):
        if not entries: return
        for i, entry in enumerate(entries):
            self.addEntry(entry)

    def addEntry(self, entry):
        [(d,ne)] = self.nearest(entry, 1)
        if d == 0.0:
            return ne
        self.entries.append(entry)
        self.points = np.vstack([self.points, np.array([entry.point])])
        self.size += 1
        return entry

    def nearest(self, entry, k):
        point = entry.point
        dist_2 = np.sum((self.points - point)**2, axis=1)
        ind = np.argsort(dist_2)[:k]
        ans = [(dist_2[i], self.entries[i]) for i in ind]
        return ans

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
        self.kNearest = params['kNearest']
        self.nodeList = []
        # cluster kdTree
        # self.kdTree = KDTree([self.rootCluster])
        # Caches
        self.confReachCache = {}
        self.approachConfs = {}

    def nodes(self):
        return self.nodeList
    def batchAddNodes(self, confs):
        self.nodeList = [makeNode(conf) for conf in confs]

    # NB: Returns a path from targetConf to startConf
    def confReachViol(self, targetConf, pbs, prob,
                      initViol=viol0, startConf = None, reversePath = False,
                      optimize = False, moveBase = True):

        def displayAns(ans):
            if not debug('confReachViol'): return
            if ans:
                if (not glob.inHeuristic or debug('drawInHeuristic')):
                    (viol, cost, path) = ans
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
                         ('objectBs', pbs.objectBs),
                         ('held', (pbs.held['left'],
                                   pbs.held['right'])),
                         ('initViol', initViol))
            if glob.inHeuristic: return 
            key = (targetConf, initConf, moveBase)
            if not key in self.confReachCache:
                self.confReachCache[key] = []
            self.confReachCache[key].append((pbs, prob,
                                             ans if ans else (None, None, None)))

        def checkCache(key, type='full', loose=False):
            if glob.inHeuristic or glob.skipSearch: return 
            if key in self.confReachCache:
                if debug('traceCRH'): print '    cache?',
                cacheValues = self.confReachCache[key]
                sortedCacheValues = sorted(cacheValues,
                                           key=lambda v: v[-1][0].weight() if v[-1][0] else 1000.)
                ans = bsEntails(pbs, prob, sortedCacheValues, loose=loose)
                if ans != None:
                    if debug('traceCRH'): print '    actual', type, 'cache hit'
                    return ans
                elif considerReUsingPaths:
                    initObst = set(endPtViol.allObstacles())
                    initShadows = set(endPtViol.allShadows())
                    for i, cacheValue in enumerate(sortedCacheValues):
                        if i > 1: break # don't try too many times...
                        (_, _, ans) = cacheValue
                        if not ans[0]: return None
                        (_, _, cachedPath) = ans
                        viol2 = self.checkPath(cachedPath[1], pbs, prob)
                        if viol2 and set(viol2.allObstacles()) <= initObst and \
                               set(viol2.allShadows()) <= initShadows:
                            if debug('traceCRH'): print '    reusing path'
                            (_, cost, path) = ans
                            ans = (viol2, cost, path) # use the new violations
                            if debug('confReachViolCache'):
                                debugMsg('confReachViolCache', 'confReachCache reusing path')
                                print '    returning', ans
                            return ans
                    if finalConf:
                        if debug('traceCRH'): print 'Caching failed for approach'
            else:
                if debug('confReachViolCache'): print 'confReachCache miss'
                if finalConf:
                    if debug('traceCRH'): print 'Cache miss with approach'

        def checkFullCache():
            return checkCache((targetConf, initConf, moveBase))

        def confAns(ans, show=False):
            displayAns(ans)
            if ans and ans[0]:
                (viol, cost, cachedPath) = ans
                viol = viol.update(initViol)
                viol = viol.update(endPtViol)
                path = cachedPath[1]
                if show and debug('showPath') and not glob.inHeuristic:
                    print 'confAns'
                    showPath(pbs, prob, path)
                if debug('verifyPath'):
                    if not self.checkPath(path, pbs, prob):
                        raw_input('Failed checkPath')
                if len(path) > 1 and not (path[0] == targetConf and path[-1] == initConf):
                    raw_input('Path inconsistency')
                if finalConf: path = [finalConf] + path
                return (viol, cost, path)
            else:
                return (None, None, None)

        def drawProblem(forceDraw=False):
            if forceDraw or \
                   (debug('confReachViol') and \
                    (not glob.inHeuristic  or debug('drawInHeuristic'))):
                pbs.draw(prob, 'W')
                initConf.draw('W', 'blue', attached=attached)
                targetConf.draw('W', 'pink', attached=attached)
                print 'startConf is blue; targetConf is pink'
                raw_input('confReachViol')


        ######### procedure starts ##########

        if debug('traceCRH'): print 'Find path...'
        initConf = startConf or self.homeConf
        initNode = makeNode(initConf)
        attached = pbs.getShadowWorld(prob).attached

        finalConf = None
        if not glob.inHeuristic and targetConf in self.approachConfs:
            finalConf = targetConf
            targetConf = self.approachConfs[targetConf]
            if debug('traceCRH'): print '    using approach conf',

        cv1 = self.confViolations(targetConf, pbs, prob)
        cv2 = self.confViolations(initConf, pbs, prob)
        endPtViol = combineViols(cv1, cv2)
        if finalConf:
            cv3 = self.confViolations(finalConf, pbs, prob)
            endPtViol = combineViols(endPtViol, cv3)
        if endPtViol is None:
            if debug('endPoint:collision'):
                pbs.draw(prob, 'W')
                initConf.draw('W', 'blue')
                targetConf.draw('W', 'pink')
                print '    collision at end point'
                raw_input('okay?')
            return confAns(None)
        if glob.inHeuristic or (glob.skipSearch and not optimize):
            return (endPtViol, 0, (targetConf, initConf))

        cached = checkFullCache()
        if cached:
            return confAns(cached)
        ans = None

        startTime = time.time()
        chains = [chain for chain in initConf.conf \
                  if chain in targetConf.conf \
                  and max([abs(x-y) > 1.0e-6 for (x,y) in zip(initConf.conf[chain], targetConf.conf[chain])])]

        # if self.robot.baseChainName in chains and self.robot.headChainName in chains:
        #     chains.remove(self.robot.headChainName)

        for inflate in ((True, False) if optimize else (False,)):
            attempts = 1 if (not optimize or targetConf.baseConf() == initConf.baseConf()) else glob.rrtPlanAttempts
            bestDist = float('inf')
            bestPath = None
            bestViol = None
            for attempt in range(attempts):
                path, v = rrt.planRobotPath(pbs, prob, targetConf, initConf, endPtViol, chains,
                                            maxIter=glob.maxRRTIter,
                                            failIter=glob.failRRTIter,
                                            safeCheck = False,
                                            inflate=inflate)
                if v is None: break
                if attempts > 1:
                    # Do a quick smoothing step to make the base dist estimate more accurate.
                    path = self.smoothPath(path, pbs, prob,
                                           nsteps = glob.smoothSteps/4, npasses = 1.)
                    dist = basePathLength(path)
                    print 'RRT base path length', dist, (dist == bestDist)
                    if dist < bestDist:
                        bestPath = path
                        bestViol = v
                        bestDist = dist
                    elif dist == bestDist:
                        break           # this would only happen if direct path works
                else:
                    bestPath = path; bestViol = v
            path = bestPath; viol = bestViol
            if viol is not None: break
        if viol and optimize:
            path = self.smoothPath(path, pbs, prob)
        runningTime = time.time() - startTime
        trAlways('RRT time', runningTime)
        if viol:
            viol = viol.update(initViol)
        else:
            trAlways('RRT failed')
            print 'endPtViol', endPtViol
            for x in pbs.getShadowWorld(prob).objectShapes.values(): print x
            print 'targetBase', targetConf.baseConf()
            print 'initBase', initConf.baseConf()
            targetConf.draw('W', 'pink'); initConf.draw('W', 'blue')
            raw_input('RRT failed')

        if not viol:
            pass
        elif ans is None or \
             (set(viol.allObstacles()) < set(ans[0].allObstacles()) and \
              set(viol.allShadows()) < set(ans[0].allShadows())):
            # print 'original viol', ans if ans==None else ans[0]
            # print 'RRT viol', viol
            tr('CRH', '    returning RRT ans')
            if len(path) > 1 and not( path[0] == targetConf and path[-1] == initConf):
                raw_input('Path inconsistency')
            ans = (viol, 0, ('confs', path))

        cacheAns(ans)
        return confAns(ans, show=True)

    def robotSelfCollide(self, shape, heldDict):
        def partDistance(p1, p2):
            c1 = p1.origin().point()
            c2 = p2.origin().point()
            return c1.distance(c2)
        tag = 'robotSelfCollide'
        if glob.inHeuristic: return False
        # Very sparse checks...
        armChains = self.robot.armChainNames
        gripperChains = self.robot.gripperChainNames
        # 'left' and 'right' stand in for the held objects if any
        checkParts = {armChains['left'] : [armChains['right'], gripperChains['right'], 'right'],
                      gripperChains['left'] : [armChains['right'], gripperChains['right'], 'right'],
                      armChains['right']: ['left'],
                      gripperChains['right']: ['left']}
        parts = dict([(part.name(), part) for part in shape.parts()])
        if partDistance(parts[gripperChains['right']], parts[gripperChains['left']]) < minGripperDistance:
            if debug(tag): print tag, 'grippers are too close'
            return True
        for h in handName:              # fill in the held objects as 'left' or 'right'
            if h in heldDict and heldDict[h]:
                parts[h] = heldDict[h]
        for p in checkParts:
            pShape = parts.get(p, None)
            if not pShape: continue
            for check in checkParts[p]:
                if not check in parts: continue
                checkShape = parts[check]
                if pShape.collides(checkShape):
                    if debug(tag): print tag, p, 'collides with', check
                    return True
        heldParts = [x for x in heldDict.values() if x]
        return self.heldSelfCollide(heldParts)

    def heldSelfCollide(self, shape):
        tag = 'robotSelfCollide'
        if not shape: return False
        shapeParts = shape if isinstance(shape, (list, tuple)) else shape.parts()
        if len(shapeParts) < 2:
            return False
        elif len(shapeParts) == 2:
            coll = shapeParts[0].collides(shapeParts[1])
            if debug(tag) and coll:
                print tag, 'held parts collide'
            return coll
        else:
            assert None, 'There should be at most two parts in attached'
        return False
    def addEdge(self, graph, node_f, node_i, strict=False):
        if node_f == node_i: return
        if graph.edges.get((node_f, node_i), None) or graph.edges.get((node_i, node_f), None): return
        if strict:
            base_f = node_f.conf.baseConf()
            base_i = node_i.conf.baseConf()
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

    def confColliders(self, pbs, prob, conf, aColl, hColl, hsColl, 
                      edge=None, ignoreAttached=False, clearance=0.0, draw=False):

        aColl1 = aColl[:]
        hColl1 = (hColl[0][:], hColl[1][:])
        hsColl1 = (hsColl[0][:], hsColl[1][:])
        aColl2 = aColl[:]
        hColl2 = (hColl[0][:], hColl[1][:])
        hsColl2 = (hsColl[0][:], hsColl[1][:])
        
        if glob.useCC:
            ansCC, scCC = self.confCollidersCC(pbs, prob, conf, aColl, hColl, hsColl,
                                             edge, ignoreAttached, clearance, draw)
            if debug('testCC'):
                ans, sc = self.confCollidersPlace(pbs, prob, conf, aColl1, hColl1, hsColl1,
                                                  edge, ignoreAttached, draw)
                if ans != ansCC or (aColl, hColl, hsColl) != (aColl1, hColl1, hsColl1):

                    ansCC2, scCC2 = self.confCollidersCC(pbs, prob, conf, aColl2, hColl2, hsColl2,
                                                         edge, ignoreAttached, 0., draw)
                    if ansCC2 != ans:

                        print 'ansCC', ansCC, 'ansCC2', ansCC2, 'ans', ans
                        if not (scCC2 or sc):
                            pbs.draw(prob, 'W');
                            conf.draw('W', 'blue', attached=pbs.getShadowWorld(prob).attached)
                            pdb.set_trace()
                        else:
                            print 'Disagreement about self collision'

            return ansCC
        else:
            return self.confCollidersPlace(pbs, prob, conf, aColl, hColl, hsColl,
                                           edge, ignoreAttached, draw)

    # Checks individual collision: returns True if remediable
    # collision, False if no collision and None if irremediable.
    def confCollidersPlaceAux(self, rob, obst, sumColl, edgeColl, perm, draw):
        if obst in sumColl:
            # already encountered this, can't have been perm or we
            # would have stopped, so return True.
            return True
        # coll is True or False if already know, None if not known.
        # Yes, this is confusing...
        if edgeColl:
            if edgeColl.get('heldSelfCollision', False):
                if draw: raw_input('selfCollision')
                return None                 # known irremediable
            coll = edgeColl.get(obst, None) # check outcome in cache
        else:
            coll = None                 # not known
        if coll is None:                # not known, so check
            coll = rob.collides(obst)   # outcome
        if coll:                        # collision
            sumColl.append(obst)        # record in summary
            if edgeColl:
                edgeColl[obst] = True   # store in cache
            if perm:                    # permanent
                if draw:
                    obst.draw('W', 'magenta')
                    raw_input('Collision with perm = %s'%obst.name())
                return None             # irremediable collision
            return True                 # collision
        else:
            return False                # no collision

    # This updates the collisions in aColl, hColl and hsColl
    # (collisions with robot, held and heldShadow).  It returns None,
    # if an irremediable collision is found, otherwise return value is
    # not significant.
    def confCollidersPlace(self, pbs, prob, conf, aColl, hColl, hsColl, 
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
        # robShape, attachedPartsDict = conf.placementModAux(self.robotPlace, attached=attached)
        attachedParts = [x for x in attachedPartsDict.values() if x]
        permanentNames = set(shWorld.fixedObjects) # set of names
        if draw:
            robShape.draw('W', 'purple')
            for part in attachedParts: part.draw('W', 'purple')
        # The self collision can depend on grasps - how to handle caching?
        if self.robotSelfCollide(robShape, attachedPartsDict):
            if debug('robotSelfCollide'):
                conf.draw('W', 'red')
                raw_input('selfCollision')
            return None, True           # irremediable collision
        for obst in shWorld.getObjectShapes():
            perm = obst.name() in permanentNames
            eColl = edge.aColl if edge else None
            res = self.confCollidersPlaceAux(robShape, obst, aColl, eColl,
                                        perm, draw)
            if res is None: return None, False # irremediable
            elif res: continue          # collision with robot, go to next obj
            # Check for held collisions if not collision so far
            if not attached or not any(attached.values()): continue
            for h in hands:
                hand = handName[h]
                if not attached[hand] or obst in hColl[h] or obst in hsColl[h]:
                    continue
                held, heldSh = heldParts(attachedPartsDict[hand])
                # check hColl
                if edge and pbs.getGraspB(hand) not in edge.hColl[hand]:
                    edge.hColl[hand][pbs.getGraspB(hand)] = {}
                eColl = edge.hColl[hand][pbs.getGraspB(hand)] if edge else None
                res = self.confCollidersPlaceAux(held, obst, hColl[h], eColl,
                                            (perm and shWorld.fixedHeld[hand]), draw)
                if res is None: return None, False # irremediable
                elif res: continue          # collision, move to next obj
                # Check hsColl
                if edge and pbs.getGraspB(hand) not in edge.hsColl[hand]:
                    edge.hsColl[hand][pbs.getGraspB(hand)] = {}
                eColl = edge.hsColl[hand][pbs.getGraspB(hand)] if edge else None
                res = self.confCollidersPlaceAux(heldSh, obst, hsColl[h], eColl,
                                                 (perm and shWorld.fixedGrasp[hand]), draw)
                if res is None: return None, False # irremediable
        return True, False

    # Checks individual collision: returns True if remediable
    # collision, False if no collision and None if irremediable.
    def confCollidersAux(self, rob, rcc, obst, occ, sumColl, edgeColl, perm, clearance, draw):
        if obst in sumColl:
            # already encountered this, can't have been perm or we
            # would have stopped, so return True.
            return True
        # coll is True or False if already know, None if not known.
        # Yes, this is confusing...
        if edgeColl:
            if edgeColl.get('heldSelfCollision', False):
                if draw: raw_input('selfCollision')
                return None                 # known irremediable
            coll = edgeColl.get(obst, None) # check outcome in cache
        else:
            coll = None                 # not known
        if coll is None:                # not known, so check
            coll = self.checkCollision(rob, rcc, obst, occ, clearance) # outcome
        if coll:                        # collision
            sumColl.append(obst)        # record in summary
            if edgeColl:
                edgeColl[obst] = True   # store in cache
            if perm:                    # permanent
                if draw:
                    obst.draw('W', 'magenta')
                    raw_input('Collision with perm = %s'%obst.name())
                return None             # irremediable collision
            return True                 # collision
        else:
            return False                # no collision

    # This updates the collisions in aColl, hColl and hsColl
    # (collisions with robot, held and heldShadow).  It returns None,
    # if an irremediable collision is found, otherwise return value is
    # not significant.
    def confCollidersCC(self, pbs, prob, conf, aColl, hColl, hsColl, 
                        edge=None, ignoreAttached=False, clearance=0.0, draw=False):
        def heldSolidParts(obj):
            parts = obj.parts()
            assert len(parts) == 2
            if shadowp(parts[0]):
                return shapes.Shape([parts[1]], None)
            else:
                return shapes.Shape([parts[0]], None)
        def heldShadowParts(obj):
            parts = obj.parts()
            assert len(parts) == 2
            if shadowp(parts[0]):
                return shapes.Shape([parts[0]], None)
            else:
                return shapes.Shape([parts[1]], None)
        tag = 'confCollidersCC'
        shWorld = pbs.getShadowWorld(prob)
        robot = conf.robot
        attached = None if ignoreAttached else shWorld.attached
        assert conf.robot.compiledChains
        robCC = confPlaceChains(conf, robot.compiledChains)
        # Ignore big shadows when checking for self collision
        attCCd = tuple([compileAttachedFrames(conf.robot, attached, h, robCC[0], heldSolidParts) \
                        for h in handName])

        permanentNames = set(shWorld.fixedObjects) # set of names
        chainNames = [[robot.armChainNames['left']], [robot. gripperChainNames['left']],
                      [robot.armChainNames['right']], [robot. gripperChainNames['right']]]
        if debug(tag):
            pbs.draw(prob, 'W')
            conf.draw('W', 'purple', attached=attached)
        # The self collision can depend on grasps - how to handle caching?
        if confSelfCollide(robCC, attCCd, chainNames):
            if debug(tag):
                conf.draw('W', 'red', attached=attached)
                raw_input('selfCollision')
            return None, True           # irremediable collision

        for obst in shWorld.getObjectShapes():
            perm = obst.name() in permanentNames
            eColl = edge.aColl if edge else None
            oCC = compileObjectFrames(obst)
            res = self.confCollidersAux(None, robCC, obst, oCC, aColl, eColl,
                                        perm, clearance, draw)
            if debug(tag): print 'Robot obst', obst.name(), 'res', res
            if res is None: return None, False # irremediable
            elif res: continue          # collision with robot, go to next obj
            # Check for held collisions if not collision so far
            if not attached or not any(attached.values()): continue
            for h in hands:
                hand = handName[h]
                if not attached[hand] or obst in hColl[h] or obst in hsColl[h]:
                    continue
                # check hColl
                if edge and pbs.getGraspB(hand) not in edge.hColl[hand]:
                    edge.hColl[hand][pbs.getGraspB(hand)] = {}
                eColl = edge.hColl[hand][pbs.getGraspB(hand)] if edge else None
                attCC = compileAttachedFrames(conf.robot, attached, hand, robCC[0], heldSolidParts)
                res = self.confCollidersAux(None, attCC, obst, oCC, hColl[h], eColl,
                                            (perm and shWorld.fixedHeld[hand]), clearance, draw)
                if debug(tag): print 'Held obst', obst.name(), 'res', res
                if res is None: return None, False # irremediable
                elif res: continue          # collision, move to next obj
                # Check hsColl
                if edge and pbs.getGraspB(hand) not in edge.hsColl[hand]:
                    edge.hsColl[hand][pbs.getGraspB(hand)] = {}
                eColl = edge.hsColl[hand][pbs.getGraspB(hand)] if edge else None
                attCC = compileAttachedFrames(conf.robot, attached, hand, robCC[0], heldShadowParts)
                res = self.confCollidersAux(None, attCC, obst, oCC, hsColl[h], eColl,
                                            (perm and shWorld.fixedGrasp[hand]), clearance,  draw)
                if debug(tag): print 'HeldShadow obst', obst.name(), 'res', res
                if res is None: return None, False # irremediable
        if debug(tag): print 'returning True'
        return True, False

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
            #         if edge and pbs.getGraspB(hand) not in edge.hColl[hand]:
            #             edge.hColl[hand][pbs.getGraspB(hand)] = {}
            #         edge.hColl[hand][pbs.getGraspB(hand)][obst] = (obst in hColl[h])
            #         if pbs.getGraspB(hand) not in edge.hsColl[hand]:
            #             edge.hsColl[hand][pbs.getGraspB(hand)] = {}
            #         edge.hsColl[hand][pbs.getGraspB(hand)][obst] = (obst in hsColl[h])
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
                # pbs.draw(prob, 'W'); conf.draw('W', 'magenta')
                return None
        return newViol

    def confViolations(self, conf, pbs, prob, initViol=viol0,
                       ignoreAttached=False, clearance = 0.0):
        if initViol is None:
            return None, (None, None)
        shWorld = pbs.getShadowWorld(prob)
        (aColl, hColl, hsColl) = violToColl(initViol)
        if self.confColliders(pbs, prob, conf, aColl, hColl, hsColl,
                              ignoreAttached=ignoreAttached,
                              clearance=clearance,
                              draw=debug('confViolations')) is None:
            if debug('confViolations'):
                conf.draw('W')
                raw_input('confViolations -> None')
            return None
        return makeViolations(shWorld, (aColl, hColl, hsColl))

    def safePath(self, qf, qi, pbs, prob):
        for conf in rrt.interpolate(qf, qi, stepSize=minStep):
            newViol = self.confViolations(conf, pbs, prob, initViol=viol0)
            if newViol is None or newViol.weight() > 0.:
                if debug('smooth'): conf.draw('W', 'red')
                return False
            else:
                if debug('smooth'): conf.draw('W', 'green')
        return True

    def smoothPath(self, path, pbs, prob, verbose=False,
                   nsteps = glob.smoothSteps, npasses = glob.smoothPasses):
        verbose = verbose or debug('smooth')
        n = len(path)
        if n < 3: return path
        if verbose: print 'Path has %s points'%str(n), '... smoothing'
        input = removeDuplicateConfs(path)
        if len(input) < 3:
            return path
        checked = set([])
        outer = 0
        count = 0
        step = 0
        if verbose: print 'Smoothing...'
        while outer < npasses:
            if verbose:
                print '  Start smoothing pass', outer, 'len=', len(input), 'dist=', basePathLength(input)
            smoothed = []
            for p in input:
                if not smoothed or smoothed[-1] != p:
                    smoothed.append(p)
            n = len(smoothed)
            while count < nsteps and n > 2:
                if debug('smooth'): print 'step', step, ':', 
                if n < 1:
                    debugMsg('smooth', 'Path is empty!')
                    pdb.set_trace()
                    return removeDuplicateConfs(path)
                i = random.randrange(n)
                j = random.randrange(n)
                if j < i: i, j = j, i 
                step += 1
                if debug('smooth'): print i, j, len(checked)
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
                    debugMsg('smooth', 'Testing')
                if self.safePath(smoothed[j], smoothed[i], pbs, prob):
                    count = 0
                    if debug('smooth'):
                        debugMsg('smooth', 'Safe')
                        pbs.draw(prob, 'W')
                        for k in range(i+1)+range(j,len(smoothed)):
                            smoothed[k].draw('W', 'blue')
                        debugMsg('smooth', 'remaining')
                    smoothed[i+1:j] = []
                    n = len(smoothed)
                    if verbose: print 'Smoothed path length is', n
                else:
                    count += 1
            outer += 1
            if outer < npasses:
                count = 0
                if verbose: print 'Re-expanding path'
                input = removeDuplicateConfs(rrt.interpolatePath(removeDuplicateConfs(smoothed)))
        if verbose:
            print 'Final smooth path len =', len(smoothed), 'dist=', basePathLength(smoothed)

        ans = removeDuplicateConfs(smoothed)
        assert ans[0] == path[0] and ans[-1] == path[-1]
        return ans

    # does not use self... could be a static method
    def checkRobotCollision(self, conf, obj, clearance=0.0, attached=None, selectedChains=None):
        if glob.useCC:
            assert conf.robot.compiledChains
            rcc = confPlaceChains(conf, conf.robot.compiledChains)
            occ = compileObjectFrames(obj)
            ansCC = False
            if chainCollides(rcc, selectedChains, occ, None):
                ansCC = True
            if (not ansCC) and attached:
                for hand in ('left', 'right'):
                    if selectedChains is not None:
                        armChainName = conf.robot.armChainNames[hand]
                        if armChainName not in selectedChains: continue
                    acc = compileAttachedFrames(conf.robot, attached, hand, rcc[0])
                    if chainCollides(acc, None, occ, None, clearance):
                        ansCC = True
                        break
            if debug('testCC'):
                placement = conf.placement(attached=attached)
                # should be limited to the relevant chains
                ans = placement.collides(obj) 
                if ans != ansCC:
                    conf.draw('W'); obj.draw('W', 'blue')
                    for fr in rcc[0].values():
                        if fr.link: fr.draw('W', 'magenta')
                    for fr in occ[0].values():
                        if fr.link: fr.draw('W', 'cyan')
                    pdb.set_trace()
            return ansCC
        else:
            assert not selectedChains, 'Does not handle selectedChains'
            placement = conf.placement(attached=attached)
            return placement.collides(obj)

    def checkCollision(self, rob, rcc, obj, occ, clearance=0.0):
        if glob.useCC and rcc and occ:
            ansCC = chainCollides(rcc, None, occ, None, clearance)
            if rob and debug('testCC'):
                ans = rob.collides(obj)
                if ans != ansCC:
                    pdb.set_trace()
            return ansCC
        else:
            return rob.collides(obj)

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
    (held1, grasp1, conf1, base1, tconf1, avoid1, obj1) = bs1.items()
    (held2, grasp2, conf2, base2, tconf2, avoid2, obj2) = bs2.items()
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
    fix1 = frozenset([(o,pB) for (o, (fix, pB)) in obj1 if fix])
    fix2 = frozenset([(o,pB) for (o, (fix, pB)) in obj2 if fix])
    if not placesBigger(fix1, p1, fix2, p2):
        if debug('confReachViolCache'):
            print 'fix1 is not superset of fix2'
            print '    fix1', fix1
            print '    fix2', fix2
        return False
    move1 = frozenset([(o,pB) for (o, (fix, pB)) in obj1 if not fix])
    move2 = frozenset([(o,pB) for (o, (fix, pB)) in obj2 if not fix])
    if not placesBigger(move1, p1, move2, p2):
        if debug('confReachViolCache'):
            print 'move1 is not superset of move2'
            print '    move1', move1
            print '    move2', move2
        return False
    if grasp1 != grasp2:
        gr1 = dict([(o,gB) for (o, (fix, gB)) in grasp1])
        gr2 = dict([(o,gB) for (o, (fix, gB)) in grasp2])
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

r = 0.02
boxPoint = shapes.Shape([shapes.BoxAligned(np.array([(-2*r, -2*r, -r), (2*r, 2*r, r)]), None),
                         shapes.BoxAligned(np.array([(2*r, -r, -r), (3*r, r, r)]), None)], None)
def minViolPathDebugExpand(n):
    (node, _) = n.state
    # node.conf.draw('W')
    # raw_input('expand')
    (x,y,th) = node.conf.baseConf()
    boxPoint.applyTrans(hu.Pose(x,y,0,th)).draw('W')
    wm.getWindow('W').update()

def minViolPathDebugVisit(state, cost, heuristicCost, a, newState, newCost, hValue):
    (node, _) = newState
    (x,y,th) = node.conf.baseConf()
    boxPoint.applyTrans(hu.Pose(x,y,0,th)).draw('W', 'cyan')
    wm.getWindow('W').update()
    
def showPath(pbs, p, path, stop=True):
    attached = pbs.getShadowWorld(p).attached
    for c in path:
        pbs.draw(p, 'W')
        c.draw('W', attached=attached)
        raw_input('Next?')
    raw_input('Path end')

def basePathDistAndAngle(path):
    distSoFar = 0.0
    angleSoFar = 0.0
    for i in xrange(1, len(path)):
        prevXYT = path[i-1].baseConf()
        newXYT = path[i].baseConf()
        distSoFar += math.sqrt(sum([(prevXYT[i]-newXYT[i])**2 for i in (0,1)]))
        # approx pi => 1 meter
        angleSoFar += abs(hu.angleDiff(prevXYT[2],newXYT[2]))
    return distSoFar, angleSoFar

def basePathLength(path, angleEquiv = 0.33):
    dist, angle = basePathDistAndAngle(path)
    return dist + angleEquiv*angle

