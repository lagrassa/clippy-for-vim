import pdb
import time
import string

import windowManager3D as wm
from geom import bboxGrow
from objects import World
from miscUtil import timeString

import planGlobals as glob
reload(glob)
from planGlobals import useROS

import traceFile
reload(traceFile)
from traceFile import traceStart, traceEnd, tr, trAlways

import fbch
reload(fbch)
from fbch import State, planBackward, makePlanObj, HPN

import belief
reload(belief)
from belief import BBhAddBackBSet, B, Bd

import pr2Util
reload(pr2Util)
from pr2Util import *

from planUtil import ObjPlaceB

import dist
reload(dist)
from dist import DeltaDist, MultivariateGaussianDistribution
MVG = MultivariateGaussianDistribution

import pr2Robot
reload(pr2Robot)
from pr2Robot import makePr2Chains, makePr2ChainsShadow, PR2, JointConf, pr2Init

import pr2RoadMap
reload(pr2RoadMap)
from pr2RoadMap import RoadMap

import pr2BeliefState
reload(pr2BeliefState)
from pr2BeliefState import BeliefState

import pr2Fluents
reload(pr2Fluents)
from pr2Fluents import partition, In, Holding, Grasp, GraspFace, Pose, SupportFace

import pr2PlanBel
reload(pr2PlanBel)
from pr2PlanBel import BeliefContext, PBS, findSupportRegion

import pr2Visible
reload(pr2Visible)
pr2Visible.cache = {}

import pr2GenAux
reload(pr2GenAux)

import pr2Gen
reload(pr2Gen)

import pr2Push
reload(pr2Push)

import pr2Ops
reload(pr2Ops)
# lookAtHand
from pr2Ops import move, push, pick, place, lookAt, achCanReach, achCanReachNB,\
   achCanPickPlace, achCanPush, \
   hRegrasp, poseAchIn, moveNB, bLoc1, bLoc2

import pr2Sim
reload(pr2Sim)
from pr2Sim import RealWorld

import pr2ROS
reload(pr2ROS)
from pr2ROS import RobotEnv, pr2GoToConf

import testObjects
reload(testObjects)
from testObjects import *

import mathematica

######################################################################
# Clear caches
######################################################################

def clearCaches(details):
    pbs = details.pbs
    bc = pbs.beliefContext

    # invkin cache
    # robot.confCache
    
    pbs.getRoadMap().confReachCache.clear()
    pbs.getRoadMap().approachConfs.clear()
    for thing in bc.genCaches.values():
        thing.clear()
    pr2Visible.cache.clear()
    bc.pathObstCache.clear()
    bc.objectShadowCache.clear()
    pr2GenAux.graspConfGenCache.clear()
    bc.world.robot.cacheReset()
    pr2Visible.cache.clear()
    fbch.hCacheReset()
    pr2Fluents.pushPathCache.clear()
    pr2Push.pushGenCache.clear()

######################################################################
# Test Rig
######################################################################

# Counts unsatisfied fluent groups
# noinspection PyUnusedLocal
def hEasy(s, g, ops, ancestors):
    return g.easyH(s, defaultFluentCost = 1.5)

hDepth = 10

heuristicTime = 0.0

# Return a value and a set of action instances
def habbs(s, g, ops, ancestors):
    global heuristicTime
    startTime = time.time()
    feasibleOnly = debug('feasibleHeuristicOnly')
    hops = ops + [hRegrasp]
    val = BBhAddBackBSet(s, g, hops, ancestors, ddPartitionFn = partition,
                                maxK = hDepth, feasibleOnly = feasibleOnly)
    if val == 0:
        # Just in case the addBack heuristic thinks we're at 0 when
        # the goal is not yet satisfied.
        isSat = s.satisfies(g)
        if not isSat:
            easyVal = hEasy(s, g, ops, ancestors)
            tr('heuristic0', '*** returning easyVal', easyVal)
            return easyVal, set()
    heuristicTime += (time.time() - startTime)
    return val

from timeout import timeout, TimeoutError

# 5 min timeout for all tests
# noinspection PyUnusedLocal
@timeout(600)
def testFunc(n, skeleton=None, heuristic=habbs, hierarchical=True, easy=False, rip=True):
    eval('test%s(skeleton=skeleton, heuristic=heuristic, hierarchical=hierarchical, easy=easy, rip=rip)'%str(n))

def testRepeat(n, repeat=3, **args):
    for i in range(repeat):
        try:
            testFunc(n, **args)
        except TimeoutError:
            trAlways( '************** Timed out **************')

def testAll(indices, repeat=3, crashIsError=True, **args):
    pr2Sim.crashIsError = crashIsError
    for i in indices:
        if i == 0: continue
        testRepeat(i, repeat=repeat, **args)
    print testResults

testResults = {}

######################################################################
# Test Cases
######################################################################

def cl(window='W'):
    wm.getWindow(window).clear()

def capture(testFns, winName = 'W'):
    for testFn in testFns: 
        cl()
        wm.getWindow(winName).startCapture()
        testFn()
        wm.getWindow(winName).stopCapture()
        mathematica.mathFile(win.getWindow(winName).capture)
        raw_input('Next?')

workspace = ((-1.0, -2.5, 0.0), (3.0, 2.5, 2.0))
((wx0, wy0, _), (wx1, wy1, wdz)) = workspace
viewPort = [wx0, wx1, wy0, wy1, 0.0, wdz]

def testWorld(include = []):
    ((x0, y0, _), (x1, y1, dz)) = workspace
    w = 0.1
    wm.makeWindow('W', viewPort, 600)   # was 800
    if useROS: wm.makeWindow('MAP', viewPort)
    world = World()
    # The room
    world.workspace = np.array([(x0, y0, -w), (x1, y1, -0.0001)])
    floor = Sh([Ba(world.workspace)], name = 'floor')
    world.addObjectShape(floor)
    walls = Sh([hor((x0, x1), y0, dz, w),
                hor((x0, x1), y1, dz, w),
                ver(x0, (y0, y1), dz, w),
                ver(x1, (y0, y1), dz, w),
                ], name = 'walls')
    world.addObjectShape(walls)

    for obj in include:
        otype = world.getObjType(obj)
        (shape, spaces) = glob.constructor[otype](name=obj)
        world.addObjectShape(shape)
        for (reg, pose) in spaces:
            world.addObjectRegion(obj, reg.name(), reg, pose)

    # The planning robot is a bit fatter, except in the hands...  This
    # is to provide some added tolerance for modeling and execution
    # uncertainty.
    robot = PR2('MM', makePr2ChainsShadow('PR2', world.workspace, radiusVar=0.01))
    thinRobot = PR2('MM', makePr2Chains('PR2', world.workspace))
    # This affects randomConf and stepAlongLine, unless overriden
    for r in (robot, thinRobot):
        if glob.useRight:
            r.moveChainNames = ['pr2LeftArm', 'pr2LeftGripper', 'pr2Base',
                                'pr2RightArm', 'pr2RightGripper']
        else:
            r.moveChainNames = ['pr2LeftArm', 'pr2LeftGripper', 'pr2Base']
    # This uses the initial Conf as a reference for inverseKin, when no conf is given
    robot.nominalConf = JointConf(pr2Init, robot)
    world.setRobot(robot) # robot is in world and used for planning
    thinRobot.nominalConf = JointConf(pr2Init, robot) # used in simulator
    return world, thinRobot

standardVerticalConf = None
standardHorizontalConf = None
def makeConf(robot,x,y,th,g=0.07, vertical=False):
    global standardVerticalConf, standardHorizontalConf
    dx = dy = dz = 0
    dt = 0.0 if vertical else glob.torsoZ - 0.3
    if vertical and standardVerticalConf:
        c = standardVerticalConf.copy()
        c.conf['pr2Base'] = [x, y, th]            
        c.conf['pr2LeftGripper'] = [g]
        if glob.useRight:
            c.conf['pr2RightGripper'] = [g]
        return c
    elif (not vertical) and standardHorizontalConf:
        c = standardHorizontalConf.copy()
        c.conf['pr2Base'] = [x, y, th]            
        c.conf['pr2LeftGripper'] = [g]
        if glob.useRight:
            c.conf['pr2RightGripper'] = [g]
        return c
    else:
        c = JointConf(pr2Init.copy(), robot)
        c = c.set('pr2Base', [x, y, th])            
        c = c.set('pr2LeftGripper', [g])
        if glob.useRight:
            c = c.set('pr2RightGripper', [g])
        cart = c.cartConf()
        base = cart['pr2Base']
        if vertical:
            q = np.array([0.0, 0.7071067811865475, 0.0, 0.7071067811865475])
            h = hu.Transform(p=np.array([[a] for a in [ 0.4+dx, 0.3+dy,  1.1+dt, 1.]]), q=q)
            cart = cart.set('pr2LeftArm', base.compose(h))
            if glob.useRight:
                hr = hu.Transform(p=np.array([[a] for a in [ 0.4+dx, -(0.3+dy),  1.1+dt, 1.]]), q=q)
                cart = cart.set('pr2RightArm', base.compose(hr))
        else:
            # h = hu.Pose(0.3+dx,0.33+dy,0.9+dz+dt,math.pi/4)
            h = hu.Pose(0.3+dx,0.5+dy,0.9+dz+dt,math.pi/4)
            cart = cart.set('pr2LeftArm', base.compose(h))
            if glob.useRight:
                # hr = hu.Pose(0.3+dx,-(0.33+dy),0.9+dz+dt,-math.pi/4)
                hr = hu.Pose(0.3+dx,-(0.5+dy),0.9+dz+dt,-math.pi/4)
                cart = cart.set('pr2RightArm', base.compose(hr))
        c = robot.inverseKin(cart, conf=c)
        c.conf['pr2Head'] = [0., 0.]
        assert all(c.values())
        if vertical:
            standardVerticalConf = c
        else:
            standardHorizontalConf = c
        return c

initConfs = []

# gf is grasp index
def makeInitBel(bs, grasped, hand, gf):
    # Change pbs so obj B is in the left hand
    gm = (0, -0.025, 0, 0)
    gv = (1e-6,)*4    # very small
    gd = (1e-4,)*4
    bs.pbs.updateHeld(grasped, gf, PoseD(gm, gv), hand, gd)
    bs.pbs.excludeObjs([grasped])
    bs.pbs.reset()

def makeAttachedWorldFromPBS(pbs, realWorld, grasped, hand):
    attachedShape = pbs.getRobot().\
                    attachedObj(pbs.getShadowWorld(0.9), hand)
    shape = pbs.getWorld().\
            getObjectShapeAtOrigin(grasped).applyLoc(attachedShape.origin())
    realWorld.robot.attach(shape, realWorld, hand)
    robot = pbs.getRobot()
    cart = realWorld.robotConf.cartConf()
    handPose = cart[robot.armChainNames[hand]].compose(robot.toolOffsetX[hand])
    pose = shape.origin()
    realWorld.held[hand] = grasped
    realWorld.grasp[hand] = handPose.inverse().compose(pose)
    realWorld.delObjectState(grasped)
    realWorld.setRobotConf(realWorld.robotConf) # to compute a new robotPlace

class PlanTest:
    def __init__(self, name, exp,
                 multiplier = 6, var = 1.0e-5): # var was 10e-10
        self.name = name
        self.multiplier = multiplier
        self.objects = exp.fixPoses.keys() + exp.movePoses.keys()
        self.domainProbs = exp.domainProbs
        self.world, self.thinRobot = testWorld(include=self.objects)
        if not initConfs:
            ((x0, y0, _), (x1, y1, dz)) = workspace
            dx = x1 - x0; dy = y1 - y0
            count = 2*multiplier
            for x in range(count+1):
                for y in range(count+1):
                    # print (x0+x*dx/float(count), y0+y*dy/float(count))
                    for angle in [0, math.pi/2, -math.pi/2, math.pi]:
                        if glob.useHorizontal:
                            initConfs.append(
                            makeConf(self.world.robot,
                                     x0 + x*dx/float(count),
                                     y0 + y*dy/float(count), angle)),
                        if glob.useVertical:
                            initConfs.append(
                             makeConf(self.world.robot,
                                      x0+x*dx/float(count),
                                      y0+y*dy/float(count), angle, vertical=True))
        self.initConfs = initConfs
        ff = lambda o: self.world.getFaceFrames(o) if o in self.objects else []
        self.fix = {}
        for name in exp.fixPoses:
            oShape = self.world.getObjectShapeAtOrigin(name).applyLoc(exp.fixPoses[name])
            supFace = supportFaceIndex(oShape)
            oVar = exp.varDict[name] if (exp.varDict and name in exp.varDict) else exp.defaultVar
            self.fix[name] = ObjPlaceB(name, ff(name), DeltaDist(supFace),
                              exp.fixPoses[name], oVar, exp.defaultDelta)
        self.move = {}
        for name in exp.movePoses:
            oShape = self.world.getObjectShapeAtOrigin(name).applyLoc(exp.movePoses[name])
            supFace = supportFaceIndex(oShape)
            oVar = exp.varDict[name] if (exp.varDict and name in exp.varDict) else exp.defaultVar
            self.move[name] = ObjPlaceB(name, ff(name), DeltaDist(supFace),
                              exp.movePoses[name], oVar, exp.defaultDelta)
        self.operators = exp.operators
        wm.makeWindow('Belief', viewPort, 500)
        wm.makeWindow('World', viewPort, 500)

    def buildBelief(self, home=None, regions=frozenset([])):
        world = self.world
        belC = BeliefContext(world)
        pr2Home = home or makeConf(world.robot, 0.0, 0.0, 0.0)
        rm = RoadMap(pr2Home, world,
                     params={'kNearest':17, # May be too low
                             'kdLeafSize':100,
                             'cartesian': glob.useCartesian,
                             'moveChains':
                             ['pr2Base', 'pr2LeftArm', 'pr2RightArm'] if glob.useRight \
                             else ['pr2Base', 'pr2LeftArm']})
        rm.batchAddClusters(self.initConfs)
        belC.roadMap = rm
        pbs = PBS(belC, conf=pr2Home, fixObjBs = self.fix.copy(),
                  moveObjBs = self.move.copy(), regions = frozenset(regions),
                  domainProbs=self.domainProbs, useRight=glob.useRight) 
        pbs.draw(0.95, 'Belief')
        bs = BeliefState(pbs, self.domainProbs, 'table2Top')
        # TODO:  LPK Awful modularity
        bs.partitionFn = partition
        # noinspection PyAttributeOutsideInit
        self.bs = bs

    def run(self, goal, skeleton = None, hpn = True,
            home=None, regions = frozenset([]), hierarchical = False,
            heuristic = habbs,
            greedy = 0.75, simulateError = False,
            initBelief = None, initWorld=None,
            rip = False, alwaysReplan = False, **other):
        randomizedInitialPoses = rip
        global heuristicTime
        glob.inHeuristic = False
        if not hierarchical:
            glob.maxNodesHPN = 1000
            print 'Not hierarchical, setting glob.maxNodesHPN =', glob.maxNodesHPN
        else:
            glob.maxNodesHPN = glob.savedMaxNodesHPN
        if skeleton:
            fbch.dotSearchId = 0
            glob.debugOn = list(set(list(glob.debugOn) + list(glob.skeletonTags)))
            glob.pauseOn = list(set(list(glob.pauseOn) + list(glob.skeletonTags)))
        else:
            glob.debugOn = [x for x in glob.debugOn if x not in glob.skeletonTags]
            glob.pauseOn = [x for x in glob.pauseOn if x not in glob.skeletonTags]
        startTime = time.clock()
        fbch.flatPlan = not hierarchical
        fbch.plannerGreedy = greedy 
        pr2Sim.simulateError = simulateError
        for win in wm.windows:
            wm.getWindow(win).clear()
        self.buildBelief(home=home, regions = frozenset(regions))

        ###   Initialize the world
        world = self.bs.pbs.getWorld()
        if glob.useROS:
            # pass belief state so that we can do obs updates in prims.
            self.realWorld = RobotEnv(world, self.bs) 
            startConf = self.bs.pbs.conf.copy()
            # Move base to [0., 0., 0.]
            startConf.set('pr2Base', 3*[0.])
            result, cnfOut, _ = pr2GoToConf(startConf,'move')
            # Reset the internal coordinate frames
            result, cnfOut, _ = pr2GoToConf(cnfOut, 'reset')
            debugMsg('robotEnv', result, cnfOut)
        else:
            # noinspection PyAttributeOutsideInit
            self.realWorld = RealWorld(world, self.bs,
                                       self.domainProbs,
                                       robot = self.thinRobot) # simulator

            # TODO: !! Gross hack for debugging
            glob.realWorld = self.realWorld

            # LPK!! add collision checking
            heldLeft = self.bs.pbs.held['left'].mode()
            heldRight = self.bs.pbs.held['right'].mode()
            self.realWorld.setRobotConf(self.bs.pbs.conf)
            for obj in self.objects:
                if not obj in (heldLeft, heldRight):
                    pb = self.bs.pbs.getPlaceB(obj)
                    meanObjPose = pb.objFrame().pose()
                    if randomizedInitialPoses:
                        var = pb.poseD.variance()
                        objPose = getSupportedPose(self.realWorld, obj, meanObjPose, var)
                    else:
                        objPose = meanObjPose
                    self.realWorld.setObjectPose(obj, objPose)
            self.realWorld.setRobotConf(self.bs.pbs.conf)

        # Modify belief and world if these hooks are defined
        if initBelief: initBelief(self.bs)
        if not glob.useROS:
            if initWorld: initWorld(self.bs, self.realWorld)

        # Draw
        wm.getWindow('Belief').startCapture()
        wm.getWindow('World').startCapture()
        self.bs.pbs.draw(0.9, 'Belief')
        self.bs.pbs.draw(0.9, 'W')
        if not glob.useROS:
            self.realWorld.draw('World')
            for regName in self.bs.pbs.regions:
                self.realWorld.regionShapes[regName].draw('World', 'purple')
            if self.bs.pbs.regions and debug('regions'):
                raw_input('Regions')

        if not goal: return

        s = State([], details = self.bs)
        
        try:
            traceStart(self.name)
            print '**************', self.name,\
                     'Hierarchical' if hierarchical else '', '***************'
            if hpn:
                HPN(s,
                    goal,
                    self.operators,
                    self.realWorld,
                    hpnFileTag = self.name,
                    skeleton = skeleton,
                    h = heuristic,
                    verbose = False,
                    fileTag = self.name if not debug('noWriteSearch') else None,
                    #nonMonOps = ['Move', 'MoveNB', 'LookAt', 'Place'],
                    nonMonOps = ['Move', 'MoveNB', 'LookAt'],
                    maxNodes = glob.maxNodesHPN,
                    clearCaches = clearCaches,
                    alwaysReplan = alwaysReplan)
            else:
                p = planBackward(s,
                                 goal,
                                 self.operators,
                                 h = heuristic,
                                 fileTag = self.name if not debug('noWriteSearch') else None,
                                 nonMonOps = [move])
                if p:
                    makePlanObj(p, s).printIt(verbose = False)
                else:
                    print 'Planning failed'
            runTime = time.clock() - startTime
        finally:
            traceEnd()
        if (self.name, hierarchical) not in testResults:
            testResults[(self.name, hierarchical)] = [runTime]
        else:
            testResults[(self.name, hierarchical)].append(runTime)
        print '**************', self.name, \
                'Hierarchical' if hierarchical else '', \
                'Time =', runTime, '***************'
        print 'Heuristic time:', heuristicTime
        heuristicTime = 0.0
        # Remove the skeleton tags
        glob.debugOn = [x for x in glob.debugOn if x not in glob.skeletonTags]
        glob.pauseOn = [x for x in glob.pauseOn if x not in glob.skeletonTags]
        name = self.name+'_'+timeString()
        print 'Writing mathematica movies for', name
        mathematica.mathMovie(wm.getWindow('Belief').capture,
                              glob.movieDir+name+'_bel.m')
        mathematica.mathMovie(wm.getWindow('World').capture,
                              glob.movieDir+name+'.m')
        while not debug('noPlayback'):
            ans = raw_input('Playback? w = world, b = belief, q = no: ')
            if ans in ('w', 'W'):
                wm.getWindow('World').playback(delay=0.01)
            elif ans in ('b', 'B'):
                wm.getWindow('Belief').playback(delay=0.1)
            elif ans in ('q', 'Q'):
                print 'Done playback'
                return
            else:
                print 'Please enter w, b, or q'

def getSupportedPose(realWorld, obj, meanObjPose, variance):
    stDev = tuple([math.sqrt(v) for v in variance])
    if realWorld.world.getGraspDesc(obj): # graspable
        supported = False
        while not supported:
            objPose = meanObjPose.corruptGauss(0.0, stDev, noZ =True)
            shape = realWorld.world.getObjectShapeAtOrigin(obj).applyLoc(objPose)
            supported = findSupportRegion(shape, realWorld.regionShapes,
                                          strict=True, fail=False)
            # realWorld.draw('World'); shape.draw('World', 'magenta')
    else:
        objPose = meanObjPose.corruptGauss(0.0, stDev, noZ =True)
    return objPose

def checkLogLikelihood(pbs, objPose, meanObjPose, variance):
    stDev = tuple([math.sqrt(v) for v in variance()])
    d = MVG(np.mat(meanObjPose.xyztTuple()).T,
            makeDiag(variance))
    ll = float(d.logProb(np.mat(objPose.xyztTuple()).T))
    print 'Obj pose', obj
    print '  mean', meanObjPose
    print '  delta', [x-y for (x,y) in \
                      zip(meanObjPose.xyztTuple(),
                          objPose.xyztTuple())]
    print '  stdev', stDev
    print '  draw', objPose
    print '  log likelihood', ll
    objShape = pbs.getObjectShapeAtOrigin(obj)
    pbs.draw(0.95, 'Belief')
    objShape.applyLoc(objPose).draw('Belief', 'pink')
    raw_input('okay?')

######################################################################
# Test Cases
######################################################################

# No Z error in observations for now!  Address this eventually.
# Turned down pickVar until we can look at hand

# Made odo error smaller...

#variance for placement = 2 * obsvar + placeDelta

typicalErrProbs = DomainProbs(
            # stdev, constant, assuming we control it by tracking while moving
            # odoError = (0.04, 0.04, 1e-5, 0.1),
            # odoError = (0.02, 0.02, 1e-5, 0.05),
            odoError = (0.01, 0.01, 1e-5, 0.03),
            # odoError = (0.005, 0.005, 1e-5, 0.01),
            # variance in observations; diagonal for now
            obsVar = (0.005**2, 0.005**2, 1e-5 **2, 0.01**2),
            # big angle var from robot experience
            # obsVar = (0.005**2, 0.005**2,0.005**2, 0.15**2),
            # get type of object wrong
            obsTypeErrProb = 0.05,
            # fail to pick or place in the way characterized by the Gaussian
            pickFailProb = 0.02,
            placeFailProb = 0.02,
            pushFailProb = 0.1,
            # variance in grasp after picking
            pickVar = (0.001**2, 0.001**2, 1e-11, 0.002**2),
            # variance in pose after placing
            placeVar = (0.001**2, 0.001**2, 1e-11, 0.002**2),
            pushVar = (0.01**2, 0.01**2, 1e-11, 0.02**2),
            # pickTolerance
            pickTolerance = (0.025, 0.025, 0.025, 0.1),
            maxGraspVar = (0.0051**2, .0051**2, .005**2, .015**2),
            #maxPushVar = (0.001**2, .001**2, .0001**2, .002**2),
            #maxPushVar = (0.005**2, .005**2, .0001**2, .01**2),
            maxPushVar = (0.01**2, .01**2, .0001**2, .1**2),
            moveConfDelta = (0.001, 0.001, 1e-6, 0.002),
            #shadowDelta = (0.01, 0.01, 1.0e-8, 0.05),
            #shadowDelta = (0.001, 0.001, 1e-11, 0.002),

            # The soda box is getting too fat... TLP
            # shadowDelta = (0.004, 0.004, 1e-6, 0.008),
            shadowDelta = (0.001, 0.001, 1e-6, 0.002),

            # Use this for placing objects
            #placeDelta = (0.005, 0.005, 1.0e-4, 0.01),
            # graspDelta = (0.001, 0.001, 1.0e-4, 0.002))
            placeDelta = (0.01, 0.01, 1.0e-4, 0.02),
            graspDelta = (0.005, 0.005, 1.0e-4, 0.008))



# tinyErrProbs = DomainProbs(
#             # stdev, constant, assuming we control it by tracking while moving
#             odoError = (0.0001, 0.0001, 1e-11, 0.0001),
#             # variance in observations; diagonal for now
#             obsVar = (0.0001**2, 0.0001**2,0.0001**2, 0.0001**2),
#             # get type of object wrong
#             obsTypeErrProb = 0.0,
#             # fail to pick or place in the way characterized by the Gaussian
#             pickFailProb = 0.0,
#             placeFailProb = 0.0,
#             pushFailProb = 0.0,
#             # variance in grasp after picking
#             pickVar = (0.0001**2, 0.0001**2, 1e-11, 0.0001**2),
#             # variance in pose after placing
#             placeVar = (0.0001**2, 0.0001**2, 1e-11, 0.0001**2),
#             pushVar = (0.0001**2, 0.0001**2, 1e-11, 0.0001**2),
#             # pickTolerance
#             pickTolerance = (0.025, 0.025, 0.025, 0.05),
#             maxGraspVar = (0.005**2, .005**2, .005**2, .015**2),
#             maxPushVar = (0.01**2, .01**2, .01**2, .02**2),
#             # Use this for placing objects
#             placeDelta = (0.005, 0.005, 1.0e-4, 0.01),
#             graspDelta = (0.001, 0.001, 1.0e-4, 0.002))

allOperators = [move, push, lookAt, moveNB,
                achCanReach, achCanReachNB, achCanPickPlace, achCanPush,
                poseAchIn, bLoc1, bLoc2]

if not debug('disablePickPlace'):
    allOperators.extend([pick, place])

#lookAtHand  achCanSee
