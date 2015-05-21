import pdb
import math
import numpy as np
import time

import util
import windowManager3D as wm
import transformations as transf
from geom import bboxGrow
import shapes

import objects
from objects import WorldState, World

import planGlobals as glob
reload(glob)
from planGlobals import debug, debugMsg, useROS

import fbch
reload(fbch)
from fbch import State, planBackward, makePlanObj, HPN

import belief
reload(belief)
from belief import hAddBackBSetID, B, Bd

import pr2Util
reload(pr2Util)
from pr2Util import *

import dist
reload(dist)
from dist import DDist, DeltaDist, MultivariateGaussianDistribution, makeDiag
MVG = MultivariateGaussianDistribution

import pr2Robot
reload(pr2Robot)
from pr2Robot import makePr2Chains, PR2, JointConf, CartConf, pr2Init, \
     gripperTip

import pr2RoadMap
reload(pr2RoadMap)
from pr2RoadMap import RoadMap

import pr2BeliefState
reload(pr2BeliefState)
from pr2BeliefState import BeliefState

import pr2Fluents
reload(pr2Fluents)
from pr2Fluents import Conf, SupportFace, Pose, Holding, GraspFace, Grasp,\
     partition, In, CanPickPlace

import pr2PlanBel
reload(pr2PlanBel)
from pr2PlanBel import BeliefContext, PBS

import pr2Visible
reload(pr2Visible)
pr2Visible.cache = {}

import pr2GenAux
reload(pr2GenAux)

import pr2Gen
reload(pr2Gen)

import pr2Ops
reload(pr2Ops)
from pr2Ops import move, pick, place, lookAt, poseAchCanReach, poseAchCanSee,\
      lookAtHand, hRegrasp, poseAchCanPickPlace, graspAchCanPickPlace,\
      poseAchIn, moveNB

import pr2Sim
reload(pr2Sim)
from pr2Sim import RealWorld

import pr2ROS
reload(pr2ROS)
from pr2ROS import RobotEnv, pr2GoToConf, reactiveApproach, testReactive

writeSearch = True

######################################################################
# Right Arm??
######################################################################

useCartesian = False
useLookAtHand = False

# DEBUG
useRight = True
useVertical = True
useHorizontal = True

if useROS:
    useRight = False
    useVertical = True
    useHorizontal = True

######################################################################
# Test Rig
######################################################################

# Counts unsatisfied fluent groups
def hEasy(s, g, ops, ancestors):
    return g.easyH(s, defaultFluentCost = 1.5)

def habbs(s, g, ops, ancestors):
    hops = ops + [hRegrasp]
    val = hAddBackBSetID(s, g, hops, ancestors, ddPartitionFn = partition,
                         maxK = 20)
    if val == 0:
        # Just in case the addBack heuristic thinks we're at 0 when
        # the goal is not yet satisfied.
        isSat = s.satisfies(g)
        if not isSat: 
            print '*** habbs is 0 but goal not sat ***'
            if debug('heuristic'):
                for thing in g.fluents:
                    if not thing.isGround() or \
                      thing.valueInDetails(s.details) == False:
                        print thing
                raw_input('okay?')
            easyVal = hEasy(s, g, ops, ancestors)
            print '*** returning easyVal', easyVal, '***'
            return easyVal
    return val

from timeout import timeout, TimeoutError

# 5 min timeout for all tests
@timeout(600)
def testFunc(n, skeleton=None, heuristic=habbs, hierarchical=False, easy=False):
    eval('test%d(skeleton=skeleton, heuristic=heuristic, hierarchical=hierarchical, easy=easy)'%n)

def testRepeat(n, repeat=3, **args):
    for i in range(repeat):
        try:
            testFunc(n, **args)
        except TimeoutError:
            print '************** Timed out **************'

testResults = {}

def testAll(indices, repeat=3, crashIsError=True, **args):
    pr2Sim.crashIsError = crashIsError
    for i in indices:
        if i == 0: continue
        testRepeat(i, repeat=repeat, **args)
    print testResults

######################################################################
# Test Cases
######################################################################

def cl(window='W'):
    wm.getWindow(window).clear()

def Ba(bb, **prop): return shapes.BoxAligned(np.array(bb), None, **prop)
def Sh(args, **prop): return shapes.Shape(list(args), None, **prop)

workspace = ((-1.0, -2.5, 0.0), (3.0, 2.5, 2.0))
((x0, y0, _), (x1, y1, dz)) = workspace
viewPort = [x0, x1, y0, y1, 0, dz]

tZ = 0.68

def testWorld(include = ['objA', 'objB', 'objC'],
              draw = True):
    ((x0, y0, _), (x1, y1, dz)) = workspace
    w = 0.1
    wm.makeWindow('W', viewPort, 600)   # was 800
    if useROS: wm.makeWindow('MAP', viewPort)
    def hor((x0, x1), y, w):
        return Ba(np.array([(x0, y-w/2, 0), (x1, y+w/2.0, dz)]))
    def ver(x, (y0, y1), w, extendSingleSide=False):
        if not extendSingleSide:
            return Ba(np.array([(x-w/2., y0, 0), (x+w/2.0, y1, dz)]))
        return Ba(np.array([(x-w, y0, 0.), (x, y1, dz)]))
    def place((x0, x1), (y0, y1), (z0, z1)):
        return Ba(np.array([(x0, y0, z0), (x1, y1, z1)]))

    def placeSc((x0, x1), (y0, y1), (z0, z1)):
        return shapes.BoxScale(x1 - x0, y1 - y0, z1 - z0, util.Pose(0,0,-0.5*(z1-z0),0), 0.5)
    
    world = World()
    # The room
    world.workspace = np.array([(x0, y0, -w), (x1, y1, -0.0001)])
    floor = Sh([Ba(world.workspace)], name = 'floor')
    world.addObjectShape(floor)
    walls = Sh([hor((x0, x1), y0, w),
                hor((x0, x1), y1, w),
                ver(x0, (y0, y1), w),
                ver(x1, (y0, y1), w),
                ], name = 'walls')
    world.addObjectShape(walls)
    # Some tables
    # table1 = Sh([place((-0.603, 0.603), (-0.298, 0.298), (0.0, 0.67))], name = 'table1', color='brown')
    table1 = makeTable(0.603, 0.298, 0.67, name = 'table1', color='brown')
    if 'table1' in include: world.addObjectShape(table1)
    table2 = Sh([place((-0.603, 0.603), (-0.298, 0.298), (0.0, 0.67))], name = 'table2', color='brown')
    if 'table2' in include: world.addObjectShape(table2)
    table3 = Sh([place((-0.603, 0.603), (-0.125, 0.125), (0.0, 0.67))], name = 'table3', color='brown')
    if 'table3' in include: world.addObjectShape(table3)

    for i in range(1,4):
        name = 'table%d'%i
        if name in include:
            bbox = world.getObjectShapeAtOrigin(name).bbox()
            regName = name+'Top'
            print 'Region', regName, '\n', bbox
            world.addObjectRegion(name, regName, Sh([Ba(bbox)], name=regName),
                                  util.Pose(0,0,2*bbox[1,2],0))
            bboxLeft = np.empty_like(bbox); bboxLeft[:] = bbox
            bboxLeft[0][0] = 0.5*(bbox[0][0] + bbox[1][0]) + 0.2
            regName = name+'Left'
            print 'Region', regName, '\n', bboxLeft
            world.addObjectRegion(name, regName, Sh([Ba(bboxLeft)], name=regName),
                                  util.Pose(0,0,2*bbox[1,2],0))
            bboxRight = np.empty_like(bbox); bboxRight = bbox
            bboxRight[1][0] = 0.5*(bbox[0][0] + bbox[1][0]) - 0.2
            regName = name+'Right'
            print 'Region', regName, '\n', bbox
            world.addObjectRegion(name, regName, Sh([Ba(bboxRight)], name=regName),
                                  util.Pose(0,0,2*bbox[1,2],0))

    # Some handy regions on table 1
    bbox = world.getObjectShapeAtOrigin('table1').bbox()
    mfbbox = np.empty_like(bbox); mfbbox[:] = bbox
    mfbbox[0][0] = 0.4 * bbox[0][0] + 0.6 * bbox[1][0]
    mfbbox[1][0] = 0.6 * bbox[0][0] + 0.4 * bbox[1][0]
    mfbbox[1][1] = 0.5*(bbox[0][1] + bbox[1][1])
    world.addObjectRegion('table1', 'table1MidRear', 
                           Sh([Ba(mfbbox)], name='table1MidRear'),
                                  util.Pose(0,0,2*bbox[1,2],0))
    mrbbox = np.empty_like(bbox); mrbbox[:] = bbox
    mrbbox[0][0] = 0.4 * bbox[0][0] + 0.6 * bbox[1][0]
    mrbbox[1][0] = 0.6 * bbox[0][0] + 0.4 * bbox[1][0]
    mrbbox[0][1] = 0.5*(bbox[0][1] + bbox[1][1])
    world.addObjectRegion('table1', 'table1MidFront', 
                           Sh([Ba(mrbbox)], name='table1MidFront'),
                                  util.Pose(0,0,2*bbox[1,2],0))
    # Other permanent objects
    cupboard1 = Sh([place((-0.25, 0.25), (-0.05, 0.05), (0.0, 0.4))],
                     name = 'cupboardSide1', color='brown')
    cupboard2 = Sh([place((-0.25, 0.25), (-0.05, 0.06), (0.0, 0.4))],
                     name = 'cupboardSide2', color='brown')
    if 'cupboardSide1' in include:
        world.addObjectShape(cupboard1)
    if 'cupboardSide2' in include:
        world.addObjectShape(cupboard2)
    
    # Some objects to grasp
    manipulanda = [oname for oname in include if oname[0:3] == 'obj']

    colors = ['red', 'green', 'blue', 'cyan', 'purple', 'pink', 'orange']
    for i, objName in enumerate(manipulanda):
        thing = Sh([place((-0.0445, 0.0445), (-0.027, 0.027), (0.0, 0.1175))],
                   name = objName, color=colors[i%len(colors)])
        # thing.typeName = 'soda'  #!! HACK
        height = thing.bbox()[1,2]
        world.addObjectShape(thing)
        # The bbox has been centered
        extraHeight = 1.5*height+0.01
        bbox = bboxGrow(thing.bbox(), np.array([0.075, 0.075, extraHeight]))
        regName = objName+'Top'
        print 'Region', regName, '\n', bbox
        world.addObjectRegion(objName, regName, Sh([Ba(bbox)], name=regName),
                              util.Pose(0,0,2*(height)+extraHeight,0))

    world.graspDesc = {}
    gMat0 = np.array([(0.,1.,0.,0.),
                      (0.,0.,1.,-0.025),
                      (1.,0.,0.,0.),
                      (0.,0.,0.,1.)])
    gMat1 = np.array([(0.,-1.,0.,0.),
                      (0.,0.,-1.,0.025),
                      (1.,0.,0.,0.),
                      (0.,0.,0.,1.)])
    # from the top
    gMat2= np.array([(-1.,0.,0.,0.),
                     (0.,0.,-1.,0.025),
                     (0.,-1.,0.,0.),
                     (0.,0.,0.,1.)])
    gMat3= np.array([(1.,0.,0.,0.),     # closer
                     (0.,0.,1.,-0.025),
                     (0.,-1.,0.,0.),
                     (0.,0.,0.,1.)])
    for obj in manipulanda:
        world.graspDesc[obj] = []
        if useHorizontal:             # horizontal
            world.graspDesc[obj].extend([GDesc(obj, util.Transform(gMat0),
                                               0.05, 0.05, 0.025),
                                         GDesc(obj, util.Transform(gMat1),
                                               0.05, 0.05, 0.025)])
        if useVertical:    # vertical
            world.graspDesc[obj].extend([GDesc(obj, util.Transform(gMat3),
                                               0.05, 0.05, 0.025),
                                         GDesc(obj, util.Transform(gMat2),
                                               0.05, 0.05, 0.025)])

    def t(o):
        if o[0:3] == 'obj': return 'soda'
        if o[0:5] == 'table': return 'table'
        return 'unknown'

    world.objectTypes = dict([(o, t(o)) for o in include])
    world.symmetries = {'soda' : ({4 : 4}, {4 : [util.Pose(0.,0.,0.,0.),
                                                 util.Pose(0.,0.,0.,math.pi)]}),
                        'table' : ({4 : 4}, {4 : [util.Pose(0.,0.,0.,0.),
                                                 util.Pose(0.,0.,0.,math.pi)]})}

    robot = PR2('MM', makePr2Chains('PR2', world.workspace))
    # This affects randomConf and stepAlongLine, unless overriden
    if useRight:
        robot.moveChainNames = ['pr2LeftArm', 'pr2LeftGripper', 'pr2Base',
                                'pr2RightArm', 'pr2RightGripper']
    else:
        robot.moveChainNames = ['pr2LeftArm', 'pr2LeftGripper', 'pr2Base']
    # This uses the initial Conf as a reference for inverseKin, when no conf is given
    robot.nominalConf = JointConf(pr2Init, robot)
    world.setRobot(robot) # robot is in world

    return world

standardVerticalConf = None
standardHorizontalConf = None
def makeConf(robot,x,y,th,g=0.07, vertical=False):
    global standardVerticalConf, standardHorizontalConf
    dx = dy = dz = 0
    if vertical and standardVerticalConf:
        c = standardVerticalConf.copy()
        c.conf['pr2Base'] = [x, y, th]            
        c.conf['pr2LeftGripper'] = [g]
        if useRight:
            c.conf['pr2RightGripper'] = [g]
        return c
    elif standardHorizontalConf:
        c = standardHorizontalConf.copy()
        c.conf['pr2Base'] = [x, y, th]            
        c.conf['pr2LeftGripper'] = [g]
        if useRight:
            c.conf['pr2RightGripper'] = [g]
        return c
    else:
        c = JointConf(pr2Init.copy(), robot)
        c = c.set('pr2Base', [x, y, th])            
        c = c.set('pr2LeftGripper', [g])
        if useRight:
            c = c.set('pr2RightGripper', [g])
        cart = c.cartConf()
        base = cart['pr2Base']
        if vertical:
            q = np.array([0.0, 0.7071067811865475, 0.0, 0.7071067811865475])
            h = util.Transform(p=np.array([[a] for a in [ 0.4+dx, 0.3+dy,  1.1, 1.]]), q=q)
            cart = cart.set('pr2LeftArm', base.compose(h))
            if useRight:
                hr = util.Transform(p=np.array([[a] for a in [ 0.4+dx, -(0.3+dy),  1.1, 1.]]), q=q)
                cart = cart.set('pr2RightArm', base.compose(hr))
        else:
            h = util.Pose(0.3+dx,0.33+dy,0.9+dz,0.)
            cart = cart.set('pr2LeftArm', base.compose(h))
            if useRight:
                hr = util.Pose(0.3+dx,-(0.33+dy),0.9+dz,0.)
                cart = cart.set('pr2RightArm', base.compose(hr))
        c = robot.inverseKin(cart, conf=c)
        c.conf['pr2Head'] = [0., 0.]
        assert all(c.values())
        if vertical:
            standardVerticalConf = c
        else:
            standardHorizontalConf = c
        return c

def makeTable(dx, dy, dz, name, width = 0.1, color = 'orange'):
    legInset = 0.02
    legOver = 0.02
    return Sh([\
        Ba([(-dx, -dy, dz-width), (dx, dy, dz)],
           name=name+ 'top', color=color),
        Ba([(-dx,      -dy, 0.0),
            (-dx+width, dy, dz-width)],
           name=name+' leg 1', color=color),
        Ba([(dx-width, -dy, 0.0),
            (dx,       dy, dz-width)],
           name=name+' leg 2', color=color)
        ], name = name, color=color)

initConfs = []

class PlanTest:
    def __init__(self, name, domainProbs, operators,
                 objects = ['table1','objA'], fixPoses = {},
                 movePoses = {}, held = None, grasp = None,
                 multiplier = 8, var = 1.0e-10, varDict = None):
        self.name = name
        self.multiplier = multiplier
        self.objects = objects          # list of objects to consider
        self.domainProbs = domainProbs
        self.world = testWorld(include=self.objects)
        if not initConfs:
            startTime = time.time()
            print 'Creating initial confs ...',
            ((x0, y0, _), (x1, y1, dz)) = workspace
            dx = x1 - x0; dy = y1 - y0
            count = 2*multiplier
            for x in range(count+1):
                for y in range(count+1):
                    # print (x0+x*dx/float(count), y0+y*dy/float(count))
                    for angle in [0, math.pi/2, -math.pi/2, math.pi]:
                        if useHorizontal:
                            initConfs.append(\
                            makeConf(self.world.robot,
                                     x0 + x*dx/float(count),
                                     y0 + y*dy/float(count), angle)),
                        if useVertical:
                            initConfs.append(\
                             makeConf(self.world.robot,
                                      x0+x*dx/float(count),
                                      y0+y*dy/float(count), angle, vertical=True))
            print 'done in', time.time() - startTime
        self.initConfs = initConfs
        print 'Using', len(self.initConfs), 'initial confs'
        var4 = (var, var, 1e-10, var)
        del0 = (0.0, 0.0, 0.0, 0.0)
        ff = lambda o: self.world.getFaceFrames(o) if o in objects else []
        # The poses of the supporting face frames (the placement)
        fixObjPoses = {'table1':util.Pose(1.1, 0.0, 0.0, math.pi/2),
                       'table2': util.Pose(1.0, -0.75, 0.0, 0.0),
                       'table3': util.Pose(1.6,0.0,0.0,math.pi/2),
                       'cupboardSide1': util.Pose(1.1, -0.2, 0.6, 0.0),
                       'cupboardSide2': util.Pose(1.1, 0.2, 0.6, 0.0)}
        moveObjPoses = {'objA': util.Pose(1.1, 0.0, tZ, 0.0),
                        'objB': util.Pose(0.95, -0.4, tZ, 0.0),
                        'objC': util.Pose(-0.25, -1.2, tZ, 0.0),
                        'objD': util.Pose(0.95, -0.2, tZ, 0.0),
                        'objE': util.Pose(0.95, 0.0, tZ, 0.0),
                        'objF': util.Pose(0.95, 0.2, tZ, 0.0),
                        'objG': util.Pose(0.95, 0.4, tZ, 0.0),
                        'objH': util.Pose(0.95, 0.6, tZ, 0.0),
                        'objI': util.Pose(0.95, 0.8, tZ, 0.0)}
                   
        moveObjPoses.update(movePoses)           # input poses
        print 'updated', moveObjPoses
        fixObjPoses.update(fixPoses)           # input poses
        print 'updated', fixObjPoses
        self.fix = {}
        for name in fixObjPoses:
            if name in self.objects:
                oShape = self.world.getObjectShapeAtOrigin(name).applyLoc(fixObjPoses[name])
                supFace = supportFaceIndex(oShape)
                print 'supportFace', name, supFace
                oVar = varDict[name] if (varDict and name in varDict) else var4
                self.fix[name] = ObjPlaceB(name, ff(name), DeltaDist(supFace),
                                  fixObjPoses[name], oVar, del0)
        self.move = {}
        for name in moveObjPoses.keys():
            if name in self.objects:
                oShape = self.world.getObjectShapeAtOrigin(name).applyLoc(moveObjPoses[name])
                supFace = supportFaceIndex(oShape)
                print 'supportFace', name, supFace
                oVar = varDict[name] if (varDict and name in varDict) else var4
                self.move[name] = ObjPlaceB(name, ff(name), DeltaDist(supFace),
                                  moveObjPoses[name], oVar, del0)
        self.operators = operators
        wm.makeWindow('Belief', viewPort, 500)
        wm.makeWindow('World', viewPort, 500)

    def buildBelief(self, home=None, regions=frozenset([])):
        world = self.world
        belC = BeliefContext(world)
        pr2Home = home or makeConf(world.robot, 0.0, 0.0, 0.0)
        rm = RoadMap(pr2Home, world,
                     params={'kNearest':11,
                             'kdLeafSize':20,
                             'cartesian': useCartesian,
                             'moveChains':
                             ['pr2Base', 'pr2LeftArm', 'pr2RightArm'] if useRight \
                             else ['pr2Base', 'pr2LeftArm']})
        rm.batchAddClusters(self.initConfs)
        belC.roadMap = rm
        pbs = PBS(belC, conf=pr2Home, fixObjBs = self.fix.copy(), moveObjBs = self.move.copy(),
        regions = frozenset(regions), domainProbs=self.domainProbs, useRight=useRight) 
        pbs.draw(0.95, 'Belief')
        bs = BeliefState(pbs, self.domainProbs, 'table2Top')
        ### !!!!  LPK Awful modularity
        bs.partitionFn = partition
        self.bs = bs

    def run(self, goal, skeleton = None, hpn = True,
            home=None, regions = frozenset([]), hierarchical = False,
            heuristic = None,
            greedy = 0.75, simulateError = False,
            initBelief = None, initWorld=None,
            rip = False):
        randomizedInitialPoses = rip
        fbch.inHeuristic = False
        if skeleton: fbch.dotSearchId = 0
        startTime = time.clock()
        fbch.flatPlan = not hierarchical
        fbch.plannerGreedy = greedy  # somewhat greedy by default
        pr2Sim.simulateError = simulateError
        for win in wm.windows:
            wm.getWindow(win).clear()
        self.buildBelief(home=home, regions = set(regions))

        ###   Initialize the world
        world = self.bs.pbs.getWorld()
        if glob.useROS:
            # pass belief state so that we can do obs updates in prims.
            self.realWorld = RobotEnv(world, self.bs) 
            startConf = self.bs.pbs.conf.copy()
            # Move base to [0., 0., 0.]
            startConf.set('pr2Base', 3*[0.])
            result, cnfOut = pr2GoToConf(startConf,'move')
            # Reset the internal coordinate frames
            result, cnfOut = pr2GoToConf(cnfOut, 'reset')
            debugMsg('robotEnv', result, cnfOut)
        else:
            self.realWorld = RealWorld(world, self.bs,
                                       self.domainProbs) # simulator
            self.realWorld.setRobotConf(self.bs.pbs.conf)
            # LPK!! add collision checking
            heldLeft = self.bs.pbs.held['left'].mode()
            heldRight = self.bs.pbs.held['right'].mode()
            for obj in self.objects:
                if not obj in (heldLeft, heldRight):
                    pb = self.bs.pbs.getPlaceB(obj)
                    meanObjPose = pb.objFrame().pose()
                    if randomizedInitialPoses:
                        for i in range(1):   # increas for debugging
                            stDev = tuple([np.sqrt(v) for v in pb.poseD.variance()])
                            objPose = meanObjPose.corruptGauss(0.0, stDev, noZ =True)
                            # Check log likelihood
                        #     d = MVG(np.mat(meanObjPose.xyztTuple()).T,
                        #             makeDiag(pb.poseD.variance()))
                        #     ll = float(d.logProb(np.mat(objPose.xyztTuple()).T))
                        #     print 'Obj pose', obj
                        #     print '  mean', meanObjPose
                        #     print '  delta', [x-y for (x,y) in \
                        #                       zip(meanObjPose.xyztTuple(),
                        #                           objPose.xyztTuple())]
                        #     print '  stdev', stDev
                        #     print '  draw', objPose
                        #     print '  log likelihood', ll
                        #     objShape = self.bs.pbs.getObjectShapeAtOrigin(obj)
                        #     objShape.applyLoc(objPose).draw('Belief', 'pink')

                        # raw_input('okay?')
                        # self.bs.pbs.draw(0.95, 'Belief')
                    else:
                        objPose = meanObjPose
                    self.realWorld.setObjectPose(obj, objPose)

        # Modify belief and world if these hooks are defined
        if initBelief: initBelief(self.bs)
        if not glob.useROS:
            if initWorld: initWorld(self.bs, self.realWorld)

        # Draw
        self.bs.pbs.draw(0.9, 'Belief')
        self.bs.pbs.draw(0.9, 'W')
        if not glob.useROS:
            self.realWorld.draw('World')
            for regName in self.bs.pbs.regions:
                self.realWorld.regionShapes[regName].draw('World', 'purple')
            #if self.bs.pbs.regions: raw_input('Regions')

        s = State([], details = self.bs)

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
                fileTag = self.name if writeSearch else None,
                nonMonOps = ['Move', 'MoveNB', 'LookAt', 'Place'])
        else:
            p = planBackward(s,
                             goal,
                             self.operators,
                             h = heuristic,
                             fileTag = self.name if writeSearch else None,
                             nonMonOps = [move])
            if p:
                makePlanObj(p, s).printIt(verbose = False)
            else:
                print 'Planning failed'
        runTime = time.clock() - startTime
        if (self.name, hierarchical) not in testResults:
            testResults[(self.name, hierarchical)] = [runTime]
        else:
            testResults[(self.name, hierarchical)].append(runTime)
        print '**************', self.name, \
                'Hierarchical' if hierarchical else '', \
                'Time =', runTime, '***************'

######################################################################
# Test Cases
######################################################################

# No Z error in observations for now!  Address this eventually.
# Turned down pickVar until we can look at hand

typicalErrProbs = DomainProbs(\
            # stdev, constant, assuming we control it by tracking while moving
            odoError = (0.015, 0.015, 1e-11, 0.015),
            #odoError = (0.01, 0.01, 1e-11, 0.01),
            # variance in observations; diagonal for now
            obsVar = (0.01**2, 0.01**2,0.01**2, 0.01**2),
            # get type of object wrong
            obsTypeErrProb = 0.05,
            # fail to pick or place in the way characterized by the Gaussian
            pickFailProb = 0.02,
            placeFailProb = 0.02,
            # variance in grasp after picking
            # pickVar = (0.01**2, 0.01**2, 1e-11, 0.02**2),
            pickVar = (0.005**2, 0.005**2, 1e-11, 0.005**2),
            # variance in grasp after placing
            placeVar = (0.01**2, 0.01**2, 1e-11, 0.02**2),
            # pickTolerance
            #pickTolerance = (0.02, 0.02, 0.02, 0.02))
            # Too big?  Needs to be big to make the planner work unless
            # observations are a lot better
            pickTolerance = (0.05, 0.05, 0.05, 0.1))

smallErrProbs = DomainProbs(\
            # stdev, as a percentage of the motion magnitude
            odoError = (0.01, 0.01, 0.01, 0.01),
            # variance in observations; diagonal for now
            obsVar = (0.001**2, 0.001**2, 1e-6, 0.002**2),
            # get type of object wrong
            obsTypeErrProb = 0.02,
            # fail to pick or place in the way characterized by the Gaussian
            pickFailProb = 0.0,
            placeFailProb = 0.0,
            # variance in grasp after picking
            pickVar = (0.01**2, 0.01**2, 0.01**2, 0.01**2),
            # variance in grasp after placing
            placeVar = (0.01**2, 0.01**2, 0.01**2, 0.01**2),
            # pickTolerance
            pickTolerance = (0.02, 0.02, 0.02, 0.02))

tinyErrProbs = DomainProbs(\
            # stdev, as a percentage of the motion magnitude
            odoError = (0.0001, 0.0001, 0.0001, 0.0001),
            # variance in observations; diagonal for now
            obsVar = (0.00001**2, 0.00001**2, 1e-6, 0.00002**2),
            # get type of object wrong
            obsTypeErrProb = 0.0000001,
            # fail to pick or place in the way characterized by the Gaussian
            pickFailProb = 0.0,
            placeFailProb = 0.0,
            # variance in grasp after picking
            pickVar = (0.0001**2, 0.0001**2, 0.0001**2, 0.0001**2),
            # variance in grasp after placing
            placeVar = (0.0001**2, 0.0001**2, 0.0001**2, 0.0001**2),
            # pickTolerance
            pickTolerance = (0.02, 0.02, 0.02, 0.02))

allOperators = [move, pick, place, lookAt, poseAchCanReach,
                poseAchCanSee, poseAchCanPickPlace, poseAchIn, moveNB]
              #lookAtHand    #graspAchCanPickPlace


# Just move and look.  Can't do this because poseAchIn suggests moving
# the object.  This was intended to be a simple test that just needs
# to look at both objects without moving the base.
def testN1(hpn = True, skeleton = False, hierarchical = False, heuristic=habbs,
          easy = False, rip = False):

    glob.rebindPenalty = 700
    glob.monotonicFirst = False

    goalProb, errProbs = (0.5,smallErrProbs) if easy else (0.95,typicalErrProbs)

    varDict = {} if easy else {'table1': (0.07**2, 0.03**2, 1e-10, 0.2**2),
                               'objA': (0.1**2, 0.1**2, 1e-10, 0.3**2)} 
    front = util.Pose(1.1, 0.0, tZ, 0.0)
    table1Pose = util.Pose(1.3, 0.0, 0.0, math.pi/2)

    region = 'table1Top'
    goal = State([Bd([In(['objA', region]), True, goalProb], True)])

    t = PlanTest('testN1',  errProbs, allOperators,
                 objects=['table1', 'objA'],
                 fixPoses={'table1': table1Pose},
                 movePoses={'objA': front},
                 varDict = varDict)

    skel = [[poseAchIn,
             lookAt.applyBindings({'Obj' : 'table1'}), moveNB,
             lookAt.applyBindings({'Obj' : 'table1'}), moveNB,
             lookAt.applyBindings({'Obj' : 'objA'}), moveNB,
             lookAt.applyBindings({'Obj' : 'objA'}), move]]

    t.run(goal,
          hpn = hpn,
          skeleton = skel if skeleton else None,
          hierarchical = hierarchical,
          regions=[region],
          heuristic = heuristic,
          rip = rip
          )
    return t


# 1 table; move 1 object
# Assumption is that odometry error is kept in check during motions.
# Use domainProbs.odoError as the stdev of any object.

def test0(hpn = True, skeleton = False, hierarchical = False, heuristic=habbs,
          easy = False, rip = False):

    glob.rebindPenalty = 100
    glob.monotonicFirst = True

    goalProb, errProbs = (0.5,smallErrProbs) if easy else (0.95,typicalErrProbs)

    varDict = {} if easy else {'table1': (0.07**2, 0.03**2, 1e-10, 0.2**2),
                               'objA': (0.1**2, 0.1**2, 1e-10, 0.3**2)} 
    front = util.Pose(1.1, 0.0, tZ, 0.0)
    table1Pose = util.Pose(1.3, 0.0, 0.0, math.pi/2)

    region = 'table1Left'
    goal = State([Bd([In(['objA', region]), True, goalProb], True)])

    t = PlanTest('test0',  errProbs, allOperators,
                 objects=['table1', 'objA'],
                 fixPoses={'table1': table1Pose},
                 movePoses={'objA': front},
                 varDict = varDict)

    hskel = [[poseAchIn],
             [poseAchIn, place.applyBindings({'Obj' : 'objA'}),
              lookAt.applyBindings({'Obj' : 'table1'})], #1
             [lookAt.applyBindings({'Obj' : 'table1'}), moveNB], #2
             [place.applyBindings({'Obj' : 'objA'})], #3
             [lookAt.applyBindings({'Obj' : 'objA'})], #4
             [lookAt.applyBindings({'Obj' : 'objA'}), moveNB], #5
             [poseAchIn], #6
             [poseAchIn, lookAt.applyBindings({'Obj' : 'objA'}),
              place.applyBindings({'Obj' : 'objA'})], #7
             [place.applyBindings({'Obj' : 'objA'}),
              pick.applyBindings({'Obj' : 'objA'})], #8
             [pick.applyBindings({'Obj' : 'objA'})], #9
             [pick.applyBindings({'Obj' : 'objA'})], #10
             [pick.applyBindings({'Obj' : 'objA'}), move], #11
             [moveNB, lookAt.applyBindings({'Obj' : 'objA'}), move], #12
             [move]
            ]

    skel = [[poseAchIn,
             lookAt.applyBindings({'Obj' : 'table1'}), moveNB,
             lookAt.applyBindings({'Obj' : 'table1'}), moveNB,
             lookAt.applyBindings({'Obj' : 'objA'}), moveNB,
             lookAt.applyBindings({'Obj' : 'objA'}), move,
             place.applyBindings({'Obj': 'objA'}),
             poseAchCanPickPlace, moveNB,
             lookAt.applyBindings({'Obj' : 'table1'}),
             move,        # 13
             pick,
             poseAchCanPickPlace, moveNB,
             lookAt.applyBindings({'Obj' : 'objA'}), moveNB,
             lookAt.applyBindings({'Obj' : 'objA'}), moveNB,
             lookAt.applyBindings({'Obj' : 'table1'}), move,
             poseAchCanReach,
             # Skel should work in other order, but doesn't.  This
             # order doesn't always work, either.
             lookAt.applyBindings({'Obj' : 'table1'}), moveNB,
             lookAt.applyBindings({'Obj' : 'objA'}), moveNB],   # 27 total
            [poseAchIn,
             lookAt.applyBindings({'Obj' : 'objA'}), moveNB,
             lookAt.applyBindings({'Obj' : 'objA'}), moveNB,
             lookAt.applyBindings({'Obj' : 'table1'}), moveNB,
             lookAt.applyBindings({'Obj' : 'table1'}), move,
             place.applyBindings({'Obj': 'objA'}), move, 
             pick, moveNB,
             lookAt.applyBindings({'Obj' : 'objA'}), move,
             poseAchCanPickPlace,
             lookAt.applyBindings({'Obj' : 'table1'}), move],
            [poseAchIn,
             lookAt.applyBindings({'Obj' : 'objA'}), moveNB,
             lookAt.applyBindings({'Obj' : 'objA'}), moveNB,
             lookAt.applyBindings({'Obj' : 'table1'}), move,
             place.applyBindings({'Obj': 'objA'}), move, 
             pick, moveNB,
             lookAt.applyBindings({'Obj' : 'objA'}), move],
            [poseAchIn,
             lookAt.applyBindings({'Obj' : 'objA'}), moveNB,
             lookAt.applyBindings({'Obj' : 'objA'}), moveNB,
             lookAt.applyBindings({'Obj' : 'table1'}), move,
             place.applyBindings({'Obj': 'objA'}), move, 
             pick, moveNB]]
             
    actualSkel = easySkel if easy else (hskel if hierarchical else skel)

    t.run(goal,
          hpn = hpn,
          skeleton = actualSkel if skeleton else None,
          hierarchical = hierarchical,
          regions=[region],
          heuristic = heuristic,
          rip = rip
          )
    return t

# pick and place into region
def test1(hpn = True, skeleton = False, hierarchical = False, heuristic=habbs,
          easy = False, rip = False, multiplier=8):

    glob.rebindPenalty = 700
    glob.monotonicFirst = True

    goalProb, errProbs = (0.5,smallErrProbs) if easy else (0.95,typicalErrProbs)

    varDict = {} if easy else {'table1': (0.07**2, 0.03**2, 1e-10, 0.2**2),
                               'table2': (0.07**2, 0.03**2, 1e-10, 0.2**2),
                               'objA': (0.1**2, 0.1**2, 1e-10, 0.3**2)} 
    front = util.Pose(1.1, 0.0, tZ, 0.0)
    table2Pose = util.Pose(1.0, -1.00, 0.0, 0.0)
    table1Pose = util.Pose(1.3, 0.0, 0.0, math.pi/2)

    region = 'table2Left'
    goal = State([Bd([In(['objA', region]), True, goalProb], True)])

    t = PlanTest('test1',  errProbs, allOperators,
                 objects=['table1', 'objA', 'table2'],
                 fixPoses={'table1': table1Pose,
                           'table2': table2Pose},
                 movePoses={'objA': front},
                 varDict = varDict,
                 multiplier = multiplier)

    t.run(goal,
          hpn = hpn,
          skeleton = skel if skeleton else None,
          hierarchical = hierarchical,
          regions=[region],
          heuristic = heuristic,
          rip = rip
          )
    return t


# Weird initial conditions
def test2(hpn = True, skeleton = False, hierarchical = False, heuristic=habbs,
          easy = False, rip = False):

    global moreGD
    moreGD = True
    glob.rebindPenalty = 700
    glob.monotonicFirst = True

    goalProb, errProbs = (0.5,smallErrProbs) if easy else (0.95,typicalErrProbs)

    varDict = {} if easy else {'table1': (0.07**2, 0.03**2, 1e-10, 0.2**2),
                               'table2': (0.07**2, 0.03**2, 1e-10, 0.2**2),
                               'objA': (0.1**2, 0.1**2, 1e-10, 0.3**2)} 
    # varDict = {} if easy else {'table1': (0.03**2, 0.03**2, 1e-10, 0.05**2),
    #                            'table2': (0.03**2, 0.03**2, 1e-10, 0.05**2),
    #                            'objA': (0.05**2, 0.05**2, 1e-10, 0.05**2)} 
    front = util.Pose(1.1, 0.0, tZ, 0.0)
    table2Pose = util.Pose(1.0, -1.00, 0.0, 0.0)
    table1Pose = util.Pose(1.3, 0.0, 0.0, math.pi/2)

    front = util.Pose(1.211, -0.082, 0.680, 3.120)
    table1Pose = util.Pose(1.338, -0.148, 0.000, 1.562)

    region = 'table1Left'
    goal = State([Bd([In(['objA', region]), True, goalProb], True)])

    t = PlanTest('test2',  errProbs, allOperators,
                 objects=['table1', 'objA'],
                 fixPoses={'table1': table1Pose,
                           'table2': table2Pose},
                 movePoses={'objA': front},
                 varDict = varDict)

    moreGD = False

    skel = [[#lookAt.applyBindings({'Obj' : 'objA'}), move,
             place, move, pick, move,
             lookAt.applyBindings({'Obj' : 'objA'}), move,
             lookAt.applyBindings({'Obj' : 'objA'}), move,
             lookAt.applyBindings({'Obj' : 'table2'}), move,
             lookAt.applyBindings({'Obj' : 'table2'}), move]]

    skel = [[poseAchIn, place, move, pick, move, lookAt, move, lookAt, move]]

    hSkel = [[poseAchIn,
              lookAt.applyBindings({'Obj' : 'objA'}),
              place.applyBindings({'Obj' : 'objA'}),
              lookAt.applyBindings({'Obj' : 'table2'})],
              [lookAt.applyBindings({'Obj' : 'table2'}),
               move],
              [move],
              [place.applyBindings({'Obj' : 'objA'})]]
              
              
    
    t.run(goal,
          hpn = hpn,
          skeleton = hSkel if skeleton else None,
          hierarchical = hierarchical,
          regions=[region],
          heuristic = heuristic,
          rip = rip
          )
    return t

# Another weird initial condition
def test2a(hpn = True, skeleton = False, hierarchical = False, heuristic=habbs,
          easy = False, rip = False):

    glob.rebindPenalty = 700
    glob.monotonicFirst = True

    goalProb, errProbs = (0.5,smallErrProbs) if easy else (0.95,typicalErrProbs)

    varDict = {} if easy else {'table1': (0.07**2, 0.03**2, 1e-10, 0.2**2),
                            'table2': (0.010**2, 0.009**2, 0.0001**2, 0.010**2),
                               'objA': (0.1**2, 0.1**2, 1e-10, 0.3**2)} 
    front = util.Pose(1.1, 0.0, tZ, 0.0)
    table2Pose = util.Pose(0.999, -0.969, 0.000, 0.148)
    table1Pose = util.Pose(1.3, 0.0, 0.0, math.pi/2)

    region = 'table1Left'
    goal = State([Bd([In(['objA', region]), True, goalProb], True)])

    t = PlanTest('test2',  errProbs, allOperators,
                 objects=['table1', 'objA', 'table2'],
                 fixPoses={'table1': table1Pose,
                           'table2': table2Pose},
                 movePoses={'objA': front},
                 varDict = varDict)

    skel = [[#lookAt.applyBindings({'Obj' : 'objA'}), move,
             place, move, pick, move,
             lookAt.applyBindings({'Obj' : 'objA'}), move,
             lookAt.applyBindings({'Obj' : 'objA'}), move,
             lookAt.applyBindings({'Obj' : 'table2'}), move,
             lookAt.applyBindings({'Obj' : 'table2'}), move]]

    skel = [[poseAchIn, place, move, pick, move, lookAt, move, lookAt, move]]

    hSkel = [[poseAchIn,
              lookAt.applyBindings({'Obj' : 'objA'}),
              place.applyBindings({'Obj' : 'objA'}),
              lookAt.applyBindings({'Obj' : 'table2'})],
              [lookAt.applyBindings({'Obj' : 'table2'}),
               move],
              [move],
              [place.applyBindings({'Obj' : 'objA'})]]
              
              
    
    t.run(goal,
          hpn = hpn,
          skeleton = hSkel if skeleton else None,
          hierarchical = hierarchical,
          regions=[region],
          heuristic = heuristic,
          rip = rip
          )
    return t

# pick and place into region... one table, for robot.
def test3(hpn = True, skeleton = False, hierarchical = False, heuristic=habbs,
          easy = False, rip = False):

    glob.rebindPenalty = 700
    glob.monotonicFirst = True

    goalProb, errProbs = (0.5,smallErrProbs) if easy else (0.95,typicalErrProbs)

    varDict = {} if easy else {'table1': (0.07**2, 0.03**2, 1e-10, 0.2**2),
                               'objA': (0.1**2, 0.1**2, 1e-10, 0.3**2)} 
    front = util.Pose(1.1, 0.0, tZ, 0.0)
    table1Pose = util.Pose(1.3, 0.0, 0.0, math.pi/2)

    region = 'table1Left'
    goal = State([Bd([In(['objA', region]), True, goalProb], True)])

    skel = [[poseAchIn,
             lookAt.applyBindings({'Obj' : 'objA'}), move,
             lookAt.applyBindings({'Obj' : 'objA'}), move,
             place.applyBindings({'Obj' : 'objA'}), move,
             pick.applyBindings({'Obj' : 'objA'}), move,
             lookAt.applyBindings({'Obj' : 'objA'}), move,
             lookAt.applyBindings({'Obj' : 'table1'}), move,
             lookAt.applyBindings({'Obj' : 'objA'}), move,
             lookAt.applyBindings({'Obj' : 'table1'}), move]]

    t = PlanTest('test1',  errProbs, allOperators,
                 objects=['table1', 'objA'],
                 fixPoses={'table1': table1Pose},
                 movePoses={'objA': front},
                 varDict = varDict)

    t.run(goal,
          hpn = hpn,
          skeleton = skel if skeleton else None,
          hierarchical = hierarchical,
          regions=[region],
          heuristic = heuristic,
          rip = rip
          )
    return t

# pick and place into region... one table, for robot.
def test4(hpn = True, skeleton = False, hierarchical = False, heuristic=habbs,
          easy = False, rip = False):

    glob.rebindPenalty = 700
    glob.monotonicFirst = True

    goalProb, errProbs = (0.5,smallErrProbs) if easy else (0.95,typicalErrProbs)

    varDict = {} if easy else {'table1': (0.07**2, 0.03**2, 1e-10, 0.2**2),
                               'objA': (0.1**2, 0.1**2, 1e-10, 0.3**2),
                               'objB': (0.1**2, 0.1**2, 1e-10, 0.3**2)} 
    front = util.Pose(1.1, 0.0, tZ, 0.0)
    right = util.Pose(1.1, -0.4, tZ, 0.0)
    table1Pose = util.Pose(1.3, 0.0, 0.0, math.pi/2)

    region = 'table1Left'
    goal = State([Bd([In(['objA', region]), True, goalProb], True),
                  Bd([In(['objB', region]), True, goalProb], True)])

    t = PlanTest('test1',  errProbs, allOperators,
                 objects=['table1', 'objA', 'objB'],
                 fixPoses={'table1': table1Pose},
                 movePoses={'objA': right, 'objB':front},
                 varDict = varDict)

    skel = [[poseAchIn.applyBindings({'Obj1' : 'objA'}),
             poseAchIn.applyBindings({'Obj1' : 'objB'}),
             lookAt, lookAt, lookAt, lookAt]]

    t.run(goal,
          hpn = hpn,
          skeleton = skel if skeleton else None,
          hierarchical = hierarchical,
          regions=[region],
          heuristic = heuristic,
          rip = rip
          )
    return t

#  Swap!
def testSwap(hpn = True, skeleton = False, hierarchical = False,
           heuristic = habbs, easy = False, rip = False):


    # Seems to need this
    global useRight, useVertical
    useRight, useVertical = True, True

    glob.rebindPenalty = 150
    goalProb, errProbs = (0.4, tinyErrProbs) if easy else (0.95,typicalErrProbs)
    glob.monotonicFirst = True
    table2Pose = util.Pose(1.0, -1.2, 0.0, 0.0)
    
    front = util.Pose(0.95, 0.0, tZ, 0.0)
    # Put this back to make the problem harder
    #back = util.Pose(1.1, 0.0, tZ, 0.0)
    back = util.Pose(1.25, 0.0, tZ, 0.0)
    parking1 = util.Pose(0.95, 0.3, tZ, 0.0)
    parking2 = util.Pose(0.95, -0.3, tZ, 0.0)

    varDict = {} if easy else {'table1': (0.07**2, 0.03**2, 1e-10, 0.2**2),
                               'table2': (0.07**2, 0.03**2, 1e-10, 0.2**2),
                               'objA': (0.05**2,0.05**2, 1e-10,0.2**2),
                               'objB': (0.05**2,0.05**2, 1e-10,0.2**2)}


    t = PlanTest('testSwap',  errProbs, allOperators,
                 objects=['table1', 'table2', 'objA',
                          'objB'], #,'cupboardSide1', 'cupboardSide2'],
                 movePoses={'objA': back,
                            'objB': front},
                 fixPoses={'table2': table2Pose},
                 varDict = varDict)

    goal = State([Bd([In(['objB', 'table1MidRear']), True, goalProb], True),
                  Bd([In(['objA', 'table1MidFront']), True, goalProb], True)])

    goal1 = State([Bd([In(['objB', 'table2Top']), True, goalProb], True)])
    skel1 = [[poseAchIn, lookAt, move,
              place.applyBindings({'Hand' : 'right'}),
              move, pick, move, lookAt, move, lookAt, move]]

    goal2 = State([Bd([In(['objB', 'table2Top']), True, goalProb], True),
                   Bd([In(['objA', 'table2Top']), True, goalProb], True)])

    goal3 = State([Bd([In(['objB', 'table1MidRear']), True, goalProb], True)])

    skel3 = [[poseAchIn,
              lookAt.applyBindings({'Obj' : 'objB'}), move,
              lookAt.applyBindings({'Obj' : 'objB'}), move,
              place.applyBindings({'Hand' : 'right', 'Obj' : 'objB'}), move,
              pick, move,
              poseAchCanPickPlace,
              lookAt.applyBindings({'Obj' : 'objA'}), move,
              place.applyBindings({'Hand' : 'left', 'Obj' : 'objA'}), move,
              pick, move,
              lookAt.applyBindings({'Obj' : 'objA'}), move,
              lookAt.applyBindings({'Obj' : 'objA'}), move,
              lookAt.applyBindings({'Obj' : 'objB'}), move,
              lookAt.applyBindings({'Obj' : 'objB'}), move,
              lookAt.applyBindings({'Obj' : 'table1'}), move,
              poseAchCanPickPlace,
              lookAt.applyBindings({'Obj' : 'table1'}), move,
              lookAt.applyBindings({'Obj' : 'table2'}), move,
              poseAchCanPickPlace,
              lookAt.applyBindings({'Obj' : 'table2'}), move]]

    t.run(goal,
          hpn = hpn,
          skeleton = skel3 if skeleton else None,
          heuristic = heuristic,
          hierarchical = hierarchical,
          rip = rip,
          regions=['table1Top', 'table2Top', 'table1MidFront',
                   'table1MidRear']
          )

#  
def testFrotz(hpn = True, skeleton = False, hierarchical = False,
           heuristic = habbs, easy = False, rip = False):
    # Seems to need this
    global useRight, useVertical
    useRight, useVertical = True, True

    glob.rebindPenalty = 150
    goalProb, errProbs = (0.4, tinyErrProbs) if easy else (0.95,typicalErrProbs)
    glob.monotonicFirst = True
    table2Pose = util.Pose(1.0, -1.20, 0.0, 0.0)
    table1Pose = util.Pose(1.101, 0.009, 0.000, 1.569)
    
    front = util.Pose(0.947, 0.003, tZ, 0.001)
    back = util.Pose(1.25, 0.0, tZ, 0.0)

    varDict = {} if easy else {'table1': (0.007**2, 0.007**2, 1e-10, 0.007**2),
                               'table2': (0.07**2, 0.03**2, 1e-10, 0.2**2),
                               'objA': (0.05**2,0.05**2, 1e-10,0.2**2),
                               'objB': (0.007**2,0.007**2, 1e-10,0.007**2)}

    t = PlanTest('testHold',  errProbs, allOperators,
                 objects=['table1', 'objA', 'objB'],   # 'table2'
                           #'cupboardSide1', 'cupboardSide2'],
                 movePoses={'objA': back,
                            'objB': front},
                 fixPoses={'table2': table2Pose,
                           'table1': table1Pose},
                 varDict = varDict)

    obj = 'objB'
    hand = 'right'
    grasp = 0
    delta = (0.01,)*4

    # Make it possible to place B where A currently is.
    goal = State([Bd([CanPickPlace([
        JointConf({'pr2LeftGripper': [0.08], 'pr2RightArm': [-1.6833515167236328, 1.0618836879730225, -2.0999999046325684, -1.8750070333480835, 2.055434465408325, -0.49070966243743896, -1.6279093027114868], 'pr2Base': [1.8830066919326782, 0.7301344275474548, 2.356194496154785], 'pr2Torso': [0.3], 'pr2RightGripper': [0.07], 'pr2Head': [-2.0649600894148774, 5.504898941438661e-12], 'pr2LeftArm': [2.029942035675049, -0.029148640111088753, 1.7999999523162842, -1.050041675567627, 1.4296925067901611, -1.3872668743133545, 0.21086381375789642]}, t.world.robot),
JointConf({'pr2LeftGripper': [0.08], 'pr2RightArm': [-1.6833516359329224, 1.0618836879730225, -2.0999999046325684, -1.8750070333480835, 2.055434465408325, -0.4907096028327942, -1.6279094219207764], 'pr2Base': [1.8830066919326782, 0.7301344275474548, 2.356194496154785], 'pr2Torso': [0.3], 'pr2RightGripper': [0.07], 'pr2Head': [0.0, 0.0], 'pr2LeftArm': [2.0088162422180176, 0.1509757786989212, 1.7999999523162842, -0.9659018516540527, 1.5666221380233765, -1.2970997095108032, 0.24068012833595276]}, t.world.robot),
'left',
'objB',
(1.3166142702102661, 0.02232048660516739, 0.6800000071525574, 0.0),
(0.0005, 0.0005, 0.00040000001000000003, 0.0084),
(0.01, 0.01, 0.01, 0.03),
4,
2,
(0.0, -0.02500000037252903, 0.0, 0.0),
(0.0004, 0.0004, 0.0004, 0.008),
(0.005, 0.005, 0.005, 0.015),
'none',
0,
(0.0, 0.0, 0.0, 0.0),
(0.0, 0.0, 0.0, 0.0),
(0.001, 0.001, 0.001, 0.001),
[]]), True, 0.9], True)])

    # Place obj B where A currently is
    goal3 = State([Bd([In(['objB', 'table1MidRear']), True, goalProb], True)])

    skel3 = [[poseAchIn,
              lookAt.applyBindings({'Obj' : 'objB'}), move,
              lookAt.applyBindings({'Obj' : 'objB'}), move,
              place.applyBindings({'Obj' : 'objB'}), move,
              pick, move,
              poseAchCanPickPlace,
              lookAt.applyBindings({'Obj' : 'objA'}), move,
              place.applyBindings({'Obj' : 'objA'}), move,
              pick, move,
              lookAt.applyBindings({'Obj' : 'objA'}), move,
              lookAt.applyBindings({'Obj' : 'objA'}), move,
              lookAt.applyBindings({'Obj' : 'objB'}), move,
              lookAt.applyBindings({'Obj' : 'objB'}), move,
              lookAt.applyBindings({'Obj' : 'table1'}), move,
              poseAchCanPickPlace,
              lookAt.applyBindings({'Obj' : 'table1'}), move]]

    t.run(goal3,
          hpn = hpn,
          skeleton = skel3 if skeleton else None,
          heuristic = heuristic,
          hierarchical = hierarchical,
          rip = rip,
          regions=['table1Top']
          )

def testHold(hpn = True, skeleton = False, hierarchical = False,
           heuristic = habbs, easy = False, rip = False):
    # Seems to need this
    global useRight, useVertical
    useRight, useVertical = True, True

    glob.rebindPenalty = 150
    goalProb, errProbs = (0.4, tinyErrProbs) if easy else (0.95,typicalErrProbs)
    glob.monotonicFirst = True
    table2Pose = util.Pose(1.0, -1.20, 0.0, 0.0)
    
    front = util.Pose(0.95, 0.0, tZ, 0.0)
    # Put this back to make the problem harder
    #back = util.Pose(1.1, 0.0, tZ, 0.0)
    back = util.Pose(1.25, 0.0, tZ, 0.0)

    varDict = {} if easy else {'table1': (0.07**2, 0.03**2, 1e-10, 0.2**2),
                               'table2': (0.07**2, 0.03**2, 1e-10, 0.2**2),
                               'objA': (0.05**2,0.05**2, 1e-10,0.2**2),
                               'objB': (0.05**2,0.05**2, 1e-10,0.2**2)}

    t = PlanTest('testHold',  errProbs, allOperators,
                 objects=['table1', 'table2', 'objA', 'objB',
                          'cupboardSide1', 'cupboardSide2'],
                 movePoses={'objA': back,
                            'objB': front},
                 fixPoses={'table2': table2Pose},
                 varDict = varDict)

    obj = 'objB'
    hand = 'right'
    grasp = 0
    delta = (0.01,)*4

    goal = State([Bd([Holding([hand]), obj, goalProb], True),
                  Bd([GraspFace([obj, hand]), grasp, goalProb], True),
                  B([Grasp([obj, hand,  grasp]),
                     (0,-0.025,0,0), (0.01, 0.01, 0.01, 0.01), delta,
                     goalProb], True)])

    t.run(goal,
          hpn = hpn,
          skeleton = skel3 if skeleton else None,
          heuristic = heuristic,
          hierarchical = hierarchical,
          rip = rip,
          regions=['table1Top']
          )


######################################################################
#    Old tests    
'''
def test4(hpn = True, hierarchical = False, skeleton = False,
          heuristic = habbs, easy = False):

    goalProb, errProbs = (0.2,smallErrProbs) if easy else (0.95,typicalErrProbs)
    glob.rebindPenalty = 50


    t = PlanTest('test4',  errProbs, allOperators,
                 objects=['table1', 'objA', 'objB'])
    targetPose = (1.05, 0.25, tZ, 0.0)
    targetPoseB = (1.05, -0.2, tZ, 0.0)
    targetVar = (0.001, 0.001, 0.001, 0.005)
    targetDelta = (.02, .02, .02, .04)

    goal = State([Bd([In(['objA', 'table1MidFront']), True, goalProb], True),
                  Bd([In(['objB', 'table1MidRear']), True, goalProb], True)])

    t.run(goal,
          hpn = hpn,
          skeleton = skel if skeleton else None,
          hierarchical = hierarchical,
          regions=['table1Top'],
          heuristic = heuristic
          )

# Test placing in a region    
def test5(hpn = True, skeleton = False, heuristic=habbs, hierarchical = False,
          easy = False):

    goalProb, errProbs = (0.4,smallErrProbs) if easy else (0.95,typicalErrProbs)

    p1 = util.Pose(0.95, 0.0, tZ, 0.0)
    p2 = util.Pose(1.1, 0.0, tZ, 0.0)
    t = PlanTest('test5',  errProbs, allOperators,
                 objects=['table1', 'objA', 'table2'],
                 movePoses={'objA': p1,
                            'objB': p2})

    goal = State([Bd([In(['objA', 'table2Top']), True, goalProb], True)])

    t.run(goal,
          hpn = hpn,
          skeleton = [[place, move, pick, move]] if skeleton else None,
          regions=['table1Top', 'table2Top'],
          heuristic = heuristic
          )

# Test looking
def test6(hpn = True, skeleton=False, heuristic=habbs, hierarchical = False,
          easy = False, rip = False):

    goalProb, errProbs = (0.8,smallErrProbs) if easy else (0.95,typicalErrProbs)
        
    p2 = util.Pose(0.9, 0.0, tZ, 0.0)
    t = PlanTest('test6', errProbs, allOperators,
                 objects=['table1', 'objA'],
                 movePoses={'objA': p2},
                 varDict = {'table1': (0.1**2, 0.08**2, 0.000001, 0.1**2),
                            'objA': (0.075**2,0.075**2,0.000001,0.2**2)})

    goal = State([B([Pose(['objA', 4]), p2.xyztTuple(),
                     (0.001, 0.001, 0.001, 0.005),
                          (0.025,)*4, goalProb], True),
                  Bd([SupportFace(['objA']), 4, goalProb], True)])

    t.run(goal,
          hpn = hpn,
          skeleton = [[lookAt, move],
                      # need lookat to decrease prob
                      [lookAt,
                       move,
                       place.applyBindings({'Hand' : 'left'}),
                       move, pick, move,
                       poseAchCanPickPlace, lookAt, move, lookAt, move],
                      [lookAt,
                       move,
                       place.applyBindings({'Hand' : 'left'}),
                       move, pick, move, lookAt, move]] \
                      if skeleton else None,
          hierarchical = hierarchical,
          regions=['table1Top'],
          heuristic = heuristic,
          rip = rip
          )

# Test look, pick
def test7(hpn = True, flip=False, skeleton = False, heuristic=habbs,
          hierarchical = False, easy = False, gd = 0, rip = False):
    glob.rebindPenalty = 50
    goalProb, errProbs = (0.8,smallErrProbs) if easy else (0.99,typicalErrProbs)

    p1 = util.Pose(0.95, 0.0, tZ, 0.0)
    p2 = util.Pose(1.1, 0.0, tZ, 0.0)
    delta = (0.02, 0.02, 0.02, 0.05)

    t = PlanTest('test7',  errProbs, allOperators,
                 objects=['table1', 'objA', 'table2'],
                 movePoses={'objA': p2,
                            'objB': p1},
                 varDict = {'objA': (0.075**2,0.075**2, 1e-10,0.2**2)})
    targetPose = (1.05, 0.25, tZ, 0.0)

    goal = State([Bd([Holding(['left']), 'objA', goalProb], True),
                  Bd([GraspFace(['objA', 'left']), 0, goalProb], True),
                  B([Grasp(['objA', 'left', 0]),
                     (0,-0.025,0,0), (0.01, 0.01, 0.01, 0.01), delta,
                     goalProb], True)])
    homeConf = makeConf(t.world.robot, 0.0, 0.0, math.pi) if flip else None
    skel = [[lookAtHand, move, pick, move, lookAt, move]]*3
    t.run(goal,
          hpn = hpn,
          skeleton = skel if skeleton else None,
          heuristic = heuristic,
          hierarchical = hierarchical,
          regions=['table1Top'],
          home=homeConf,
          rip = rip
          )

# Regrasp!
def test8(hpn = True, skeleton=False, hierarchical = False, 
            heuristic = habbs, hand='left', flip = False, gd=1,
            easy = False):

    glob.rebindPenalty = 50
    goalProb, errProbs = (0.4, tinyErrProbs) if easy else (0.95,typicalErrProbs)
    #glob.monotonicFirst = False

    t = PlanTest('test8', errProbs, allOperators,
                 objects=['table1', 'table3', 'objA'])
    goalConf = makeConf(t.world.robot, 0.5, 1.0, 0.0)
    goal = State([Bd([Holding([hand]), 'objA', goalProb], True),
                  Bd([GraspFace(['objA', hand]), gd, goalProb], True),
                  B([Grasp(['objA', hand, gd]),
                     (0,-0.025,0,0), (0.001, 0.001, 0.001, 0.005),
                     (0.02,)*4, goalProb], True)])
    homeConf = makeConf(t.world.robot, 0.0, 0.0, math.pi) if flip else None
    goodSkel = [[pick,
                 move,
                 place.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
                 move,
                 lookAtHand,
                 move,
                 pick,
                 move]]
    t.run(goal,
          hpn = hpn,
          skeleton = goodSkel if skeleton else None,
          heuristic = heuristic, 
          hierarchical = hierarchical,
          home=homeConf,
          regions=['table1Top']
          )

def test9(hpn=True, skeleton = False, heuristic=habbs, hierarchical = False,
          easy = False, rip = False):
    glob.rebindPenalty = 50
    goalProb, errProbs = (0.4, tinyErrProbs) if easy else (0.95,typicalErrProbs)

    t = PlanTest('test9', errProbs, allOperators,
                 objects = ['table1'],
                 varDict = {'table1': (0.1**2, 0.05**2, 0.0000001, 0.1**2)},
                 # fixPoses={'table1': util.Pose(1.3, 0.0, 0.0, math.pi/2)}
                 )

    #goalConf = makeConf(t.world.robot, 1.1, 1.3, 0, 0.0)
    goalConf = makeConf(t.world.robot, 1.2, 1.4, 0, 0.0)
    confDeltas = (0.05, 0.05, 0.05, 0.05)
    goal = State([Conf([goalConf, confDeltas], True)])
    t.run(goal,
          greedy = .8,
          hpn = hpn,
          heuristic = heuristic,
          hierarchical = hierarchical,
          regions=['table1Top'],
          rip = rip,
          skeleton = [[move, poseAchCanReach,
                       lookAt, move]] if skeleton else None,
          )
    return t

def test10(hpn = True, skeleton = False, hierarchical = False, heuristic=habbs,
           easy = False, rip = False):
    # need to look at A to clear path to b
    glob.rebindPenalty = 50
    goalProb, errProbs = (0.4, tinyErrProbs) if easy else (0.99,typicalErrProbs)
    
    t = PlanTest('test10',  errProbs, allOperators,
                 objects=['table1', 'objA', 'objB'],
                 varDict = {'objA': (0.075**2,0.075**2, 1e-10,0.2**2)})
    targetPose = (1.05, 0.25, tZ, 0.0)
    targetPoseB = (1.05, -0.2, tZ, 0.0)
    targetVar = (0.01, 0.01, 0.01, 0.05) 

    goalConf = makeConf(t.world.robot, 0.5, 1.0, 0.0)
    confDeltas = (0.05, 0.05, 0.05, 0.05)

    goal = State([\
                  Bd([SupportFace(['objB']), 4, goalProb], True),
                  B([Pose(['objB', 4]),
                     targetPoseB, targetVar, (0.02,)*4,
                     goalProb], True)])
    t.run(goal,
          hpn = hpn,
          skeleton = [[place, move, pick, move, poseAchCanPickPlace,
                       lookAt, move]]*3 if skeleton else None,
          hierarchical = hierarchical,
          regions=['table1Top'],
          heuristic = heuristic,
          rip = rip
          )


def test11(hpn = True, skeleton = False, hierarchical = False,
           heuristic = habbs, easy = False, rip = False):
    glob.rebindPenalty = 50
    glob.monotonicFirst = hierarchical
    glob.monotonicFirst = True
    goalProb, errProbs = (0.4, tinyErrProbs) if easy else (0.99,typicalErrProbs)
    t = PlanTest('test11',  errProbs, allOperators,
                 objects=['table1', 'objA', 'objB'],
    # varDict = {} if easy else {'objA': (0.075**2, 0.1**2, 1e-10,0.2**2),
    #                            'objB': (0.075**2, 0.1**2, 1e-10,0.2**2),
    #                            'table1': (0.1**2, 0.03**2, 1e-10, 0.3**2)}
    varDict = {} if easy else {'objA': (0.075**2, 0.1**2, 1e-10,0.2**2),
                                'objB': (0.075**2, 0.1**2, 1e-10,0.2**2),
                                'table1': (0.05**2, 0.03**2, 1e-10, 0.2**2)})

    targetPose = (1.05, 0.25, tZ, 0.0)
    targetPoseB = (1.05, -0.25, tZ, 0.0)
    # targetVar = (0.002, 0.002, 0.002, 0.005)  make this work!
    targetVar = (0.0005, 0.0005, 0.0005, 0.001) 

    goal = State([\
                  Bd([SupportFace(['objA']), 4, goalProb], True),
                  B([Pose(['objA', 4]),
                     targetPose, targetVar, (0.02, 0.02, 0.02, 0.08),
                     goalProb], True),
                  Bd([SupportFace(['objB']), 4, goalProb], True),
                  B([Pose(['objB', 4]),
                     targetPoseB, targetVar, (0.02, 0.02, 0.02, 0.08),
                     goalProb], True)])

    skel = [[place.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
             move,
             place.applyBindings({'Obj' : 'objB', 'Hand' : 'right'}),
             move,
             lookAtHand.applyBindings({'Obj' : 'objB'}),
              move,
             pick.applyBindings({'Obj' : 'objB', 'Hand' : 'right'}),
             move,
             lookAt.applyBindings({'Obj' : 'objB'}),
             move,
             lookAtHand.applyBindings({'Obj' : 'objA'}),
             move,
             pick.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
             poseAchCanPickPlace,
             move,
             lookAt.applyBindings({'Obj' : 'objB'}),
             move,
             lookAt.applyBindings({'Obj' : 'objA'}),
             move]]*5


    hardSkel = [[lookAt.applyBindings({'Obj' : 'objA'}),
                move,
             place.applyBindings({'Obj' : 'objA', 'Hand' : 'right'}),
             move,
             pick.applyBindings({'Obj' : 'objA', 'Hand' : 'right'}),
             move,
             poseAchCanPickPlace,
             lookAt.applyBindings({'Obj' : 'objB'}),
             move,
             place.applyBindings({'Obj' : 'objB', 'Hand' : 'right'}),
             move,
             pick.applyBindings({'Obj' : 'objB', 'Hand' : 'right'}),
             move,
             lookAt.applyBindings({'Obj' : 'objA'}),
             move,
             lookAt.applyBindings({'Obj' : 'objB'}),
             move,
             lookAt.applyBindings({'Obj' : 'objA'}),
             move,
             lookAt.applyBindings({'Obj' : 'objB'}),
             move]]*5

    hierSkel = [[lookAt.applyBindings({'Obj' : 'objB'}),
                 lookAt.applyBindings({'Obj' : 'objA'}),
                 place.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
                 place.applyBindings({'Obj' : 'objB', 'Hand' : 'left'})],
                [place.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
                 poseAchCanPickPlace.applyBindings({'Obj' : 'objB'})],
                [poseAchCanPickPlace.applyBindings({'Obj' : 'objB'}),
                 lookAt.applyBindings({'Obj' : 'objA'}),
                 move,
                 poseAchCanPickPlace.applyBindings({'Obj' : 'objB'}),
                 lookAt.applyBindings({'Obj' : 'table1'}),
                 move]]
                 
    t.run(goal,
          hpn = hpn,
          skeleton = hierSkel if skeleton else None,
          hierarchical = hierarchical,
          #regions=['table1Top'],
          heuristic = heuristic,
          rip = rip
          )
    
def test12(hpn = True, skeleton = False, hierarchical = False,
           heuristic = habbs, easy = False, rip = False):
    glob.rebindPenalty = 50
    goalProb, errProbs = (0.4, tinyErrProbs) if easy else (0.99,typicalErrProbs)

    t = PlanTest('test12',  errProbs, allOperators,
                 objects=['table1', 'table2',
                          'objA', 'objB', 'objD',
                          'objE', 'objF', 'objG'])
    targetPose = (1.05, 0.25, tZ, 0.0)
    targetPoseB = (1.05, -0.2, tZ, 0.0)
    targetVar = (0.001, 0.001, 0.001, 0.005) 

    goalConf = makeConf(t.world.robot, 0.5, 1.0, 0.0)
    confDeltas = (0.05, 0.05, 0.05, 0.05)
    hand = 'left'
    gd = 0
    goal = State([Bd([Holding([hand]), 'objA', goalProb], True),
                  Bd([GraspFace(['objA', hand]), 0, goalProb], True),
                  B([Grasp(['objA', hand, gd]),
                     (0,-0.025,0,0), (0.001, 0.001, 0.001, 0.001),
                     (0.02,)*4, goalProb], True)])

    easySkel = [[pick,
                       move,
                       poseAchCanPickPlace,
                       place.applyBindings({'Hand' : 'left'}),
                       move,
                       pick,
                       move,
                       ]]

    hardSkel = [[lookAtHand, move, pick,
                       move,
                       poseAchCanPickPlace,
                       lookAt,
                       move,
                       place.applyBindings({'Hand' : 'left'}),
                       move,
                       pick,
                       move,
                       ]]
        
    t.run(goal,
          hpn = hpn,
          skeleton = hardSkel if skeleton else None,
          hierarchical = hierarchical,
          # regions=['table2Top'],
          regions=['table2Top', 'table1Top'],
          rip = rip,
          heuristic = heuristic
          )

def test13(hpn = True, skeleton = False, hierarchical = False, heuristic=habbs,
           easy = False, rip = False):
    glob.rebindPenalty = 50
    goalProb, errProbs = (0.4, tinyErrProbs) if easy else (0.99,typicalErrProbs)
    t = PlanTest('test13',  errProbs, allOperators,
                 objects=['table1', 'table2', 'objA', 'objB', 'objD',
                          'objE', 'objF', 'objG'])
    targetPose = (1.05, 0.25, tZ, 0.0)
    targetPoseB = (1.05, -0.2, tZ, 0.0)
    targetVar = (0.001, 0.001, 0.001, 0.005)  # should be this

    if skeleton and not hierarchical:
        raw_input('skeleton wrong for this problem')

    hskel = [[place],  #0
             [place, poseAchCanReach, poseAchCanReach, poseAchCanReach], #1
             [poseAchCanReach, lookAt, place], #2
             [place], #3
             [place, pick], #4
             [pick], #5
             [pick, move], #6
             [move], #7
             [place, pick], #8
             [pick, poseAchCanReach, poseAchCanReach, poseAchCanReach], #9
             [poseAchCanReach, lookAt, place], #10
             [place], #11
             [place, pick], #12
             [pick], #13
             [pick, move], #14
             [move], #15
             [place.applyBindings({'Obj' : 'objE'}),
              move], #16
             [move], #17
             [pick.applyBindings({'Obj' : 'objA'}),   # 18
              move,
              place.applyBindings({'Obj' : 'objF'}), move],
             [move], 
             [place],
             [move], 
             [place.applyBindings({'Hand' : 'left', 'Obj' : 'objA'})
              ],  # a, left
             [pick, move],
             [move],
             [place, move],
             [move]]

    goal = State([\
                  Bd([SupportFace(['objA']), 4, goalProb], True),
                  B([Pose(['objA', 4]),
                     targetPose, targetVar, (0.02,)*4,
                     goalProb], True)])
    t.run(goal,
          hpn = hpn,
          skeleton = hskel if skeleton else None,
          hierarchical = hierarchical,
          regions=['table1Top', 'table2Top'],
          rip = rip,
          heuristic = heuristic
          )
    
def test14(hpn = True, skeleton = False, hierarchical = False, heuristic=habbs,
           easy = False, rip = False):
    # Move A so we can look at B
    # Example isn't really constructed right

    glob.rebindPenalty = 50
    goalProb, errProbs = (0.4, tinyErrProbs) if easy else (0.99,typicalErrProbs)

    p1 = util.Pose(0.9, 0.0, tZ, 0.0)
    p2 = util.Pose(1.3, 0.0, tZ, 0.0)
    t = PlanTest('test14', errProbs, allOperators,
                 objects=['table1', 'objA', 'objB', 'table2',
                          'cupboardSide1', 'cupboardSide2'],
                 movePoses={'objA': p1,
                            'objB': p2},
                 varDict = {'objA': (0.05**2,0.05**2, 1e-10,0.2**2),
                            'objB': (0.05**2,0.05**2, 1e-10,0.2**2)})

    goal = State([ Bd([SupportFace(['objB']), 4, goalProb], True),
                   B([Pose(['objB', 4]), p2.xyztTuple(),
                     (0.00001, 0.00001, 0.00001, 0.0001),
                          (0.02,)*4, goalProb], True)])

    t.run(goal,
          hpn = hpn,
          skeleton = [[lookAt, poseAchCanSee, move,
                       place, move, pick, move]] \
                             if skeleton else None,
          hierarchical = hierarchical,
          regions=['table1Top'],
          heuristic = heuristic,
          rip = rip
          )


# Test look at hand
def test15(hpn = True, skeleton=False, hand='left', flip = False, gd = 0,
           heuristic=habbs, hierarchical=False, easy = False):

    glob.rebindPenalty = 50
    goalProb, errProbs = (0.4, tinyErrProbs) if easy else (0.99,typicalErrProbs)

    t = PlanTest('test15', errProbs, allOperators,
                 objects=['table1', 'objA'])
    goalConf = makeConf(t.world.robot, 0.5, 1.0, 0.0)
    confDeltas = (0.05, 0.05, 0.05, 0.05)
    goal = State([Bd([Holding([hand]), 'objA', .6], True),
                  Bd([GraspFace(['objA', hand]), gd, .6], True),
                  B([Grasp(['objA', hand, gd]),
                     (0,-0.025,0,0), (0.0001, 0.0001, 0.0001, 0.0001),
                     (0.02,)*4, 0.6], True),
                  Conf([goalConf, confDeltas], True)])
    homeConf = makeConf(t.world.robot, 0.0, 0.0, math.pi) \
                         if flip else None
    t.run(goal,
          hpn = hpn,
          skeleton = [[move, lookAtHand, move, pick, move]] \
          if skeleton else None,
          heuristic = heuristic,
          regions=['table1Top'],
          home=homeConf
          )

# pick and place with more noise 
def test16(hpn = True, skeleton = False, hierarchical = False,
           heuristic = habbs, easy = False, rip = False):
    glob.rebindPenalty = 50
    goalProb, errProbs = (0.4, tinyErrProbs) if easy else (0.99,typicalErrProbs)

    t = PlanTest('test16',  errProbs, allOperators,
                 objects=['table1', 'objA', 'objB'],
                 varDict = {'objA': (0.05**2,0.05**2, 1e-10,0.2**2),
                            'objB': (0.05**2,0.05**2, 1e-10,0.2**2)})

    targetPose = (1.05, 0.25, tZ, 0.0)
    targetVar = (0.0001, 0.0001, 0.0001, 0.0005)
    goal = State([Bd([SupportFace(['objA']), 4, .5], True),
                  B([Pose(['objA', 4]),
                     targetPose, targetVar, (0.05,)*4,
                     goalProb], True)])
    t.run(goal,
          hpn = hpn,
          skeleton = [[lookAt, move, place, move,
                       pick, move, lookAt]]*5 \
                       if skeleton else None,
          hierarchical = hierarchical,
          regions=['table1Top'],
          rip = rip,
          heuristic = heuristic
          )

# Do first with tiny error probs, and then gradually relax.

# A sequence of tests, working up to swap.
# 17.  A in back, B in parking -> B in back   (move out of the way)
# 18.  A in parking, B in parking -> A in front, B in back  (ordering)
# 19.  A in back, B in parking -> A in front, B in back (combination)
# 20.  A in front, B in parking -> A in front, B in back (combination, non-mon)
# 21.  A in back, B in front -> A in front, B in back (whole enchilada)

def test17(hpn = True, skeleton = False, hierarchical = False,
           heuristic = habbs, easy = False, rip = False):
    glob.rebindPenalty = 50
    goalProb, errProbs = (0.4, tinyErrProbs) if easy else (0.99,typicalErrProbs)

    front = util.Pose(0.95, 0.0, tZ, 0.0)
    back = util.Pose(1.1, 0.0, tZ, 0.0)
    parking = util.Pose(0.95, 0.3, tZ, 0.0)
    t = PlanTest('test17',  errProbs, allOperators,
                 objects=['table1', 'objA', 'objB'],
                 movePoses={'objA': back,
                            'objB': parking},
                 varDict = {'objA': (0.05**2,0.05**2, 1e-10,0.2**2),
                            'objB': (0.05**2,0.05**2, 1e-10,0.2**2)})

    skel = [[place.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
             move,
             pick.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
             move,
             poseAchCanPickPlace,
             place.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
             move,
             lookAtHand,
             move,
             pick.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
             move]]*5

    # Small var
    targetVar = (0.0001, 0.0001, 0.0001, 0.0005)
    targetDelta = (0.02, 0.02, 0.02, 0.05)
    
    goal = State([Bd([SupportFace(['objB']), 4, goalProb], True),
                  B([Pose(['objB', 4]),
                     back.xyztTuple(), targetVar, targetDelta,
                     goalProb], True)])

    t.run(goal,
          hpn = hpn,
          skeleton = skel if skeleton else None,
          heuristic = heuristic,
          hierarchical = hierarchical,
          rip = rip,
          regions=['table1Top']
          )


# 18.  A in parking, B in parking -> A in front, B in back  (ordering)
# Trivial with skeleton
def test18(hpn = True, skeleton = False, hierarchical = False,
           heuristic = habbs, easy = False, rip = False):

    glob.rebindPenalty = 50
    goalProb, errProbs = (0.4, tinyErrProbs) if easy else (0.99,typicalErrProbs)

    front = util.Pose(0.95, 0.0, tZ, 0.0)
    back = util.Pose(1.1, 0.0, tZ, 0.0)
    parking1 = util.Pose(0.95, 0.3, tZ, 0.0)
    parking2 = util.Pose(0.95, -0.3, tZ, 0.0)
    t = PlanTest('test18',  errProbs, allOperators,
                 objects=['table1', 'objA', 'objB'],
                 fixPoses={'objA': parking1,
                            'objB': parking2},
                 varDict = {'objA': (0.05**2,0.05**2, 1e-10,0.2**2),
                            'objB': (0.05**2,0.05**2, 1e-10,0.2**2)})

    skel = [[place.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
             move,
             pick.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
             move,
             place.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
             move,
             pick.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
             move]]*5

    # Small var
    targetVar = (0.0001, 0.0001, 0.0001, 0.0005)
    targetDelta = (0.02, 0.02, 0.02, 0.05)
    
    goal = State([Bd([SupportFace(['objA']), 4, goalProb], True),
                  B([Pose(['objA', 4]),
                     front.xyztTuple(), targetVar, targetDelta,
                     goalProb], True),
                  Bd([SupportFace(['objB']), 4, goalProb], True),
                  B([Pose(['objB', 4]),
                     back.xyztTuple(), targetVar, targetDelta,
                     goalProb], True)])

    t.run(goal,
          hpn = hpn,
          skeleton = skel if skeleton else None,
          hierarchical = hierarchical,
          heuristic = heuristic,
          regions=['table1Top'],
          rip = rip
          )

# 19.  A in back, B in parking -> A in front, B in back (combination)    
def test19(hpn = True, skeleton = False, hierarchical = False,
           heuristic = habbs, easy = False, rip = False):

    glob.rebindPenalty = 50
    goalProb, errProbs = (0.4, tinyErrProbs) if easy else (0.99,typicalErrProbs)

    front = util.Pose(0.95, 0.0, tZ, 0.0)
    back = util.Pose(1.1, 0.0, tZ, 0.0)
    parking1 = util.Pose(0.95, 0.3, tZ, 0.0)
    parking2 = util.Pose(0.95, -0.3, tZ, 0.0)
    t = PlanTest('test19',  errProbs, allOperators,
                 objects=['table1', 'objA', 'objB'],
                 movePoses={'objA': back,
                            'objB': parking2},
                 varDict = {'objA': (0.05**2,0.05**2, 1e-10,0.2**2),
                            'objB': (0.05**2,0.05**2, 1e-10,0.2**2)})

    skel = [[place.applyBindings({'Obj' : 'objA', 'Hand' : 'right'}),
             move,
             place.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
             move,
             pick.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
             move,
             pick.applyBindings({'Obj' : 'objA', 'Hand' : 'right'}),
             move]]*5

    # Small var
    targetVar = (0.0001, 0.0001, 0.0001, 0.0005)
    targetDelta = (0.02, 0.02, 0.02, 0.05)
    
    goal = State([Bd([SupportFace(['objA']), 4, goalProb], True),
                  B([Pose(['objA', 4]),
                     front.xyztTuple(), targetVar, targetDelta,
                     goalProb], True),
                  Bd([SupportFace(['objB']), 4, goalProb], True),
                  B([Pose(['objB', 4]),
                     back.xyztTuple(), targetVar, targetDelta,
                     goalProb], True)])

    t.run(goal,
          hpn = hpn,
          skeleton = skel if skeleton else None,
          hierarchical = hierarchical,
          heuristic = heuristic,
          regions=['table1Top'],
          rip = rip
          )


def test19a(hpn = True, skeleton = False, hierarchical = False,
           heuristic = habbs, easy = False):
    glob.rebindPenalty = 50
    goalProb, errProbs = (0.4, tinyErrProbs) if easy else (0.99,typicalErrProbs)
    glob.monotonicFirst = True

    front = util.Pose(0.95, 0.0, tZ, 0.0)
    back = util.Pose(1.1, 0.0, tZ, 0.0)
    parking1 = util.Pose(0.95, 0.3, tZ, 0.0)
    parking2 = util.Pose(0.95, -0.3, tZ, 0.0)
    parkingBad = util.Pose(1.183, 0.222, tZ0, 1.571)
    t = PlanTest('test19a',  errProbs, allOperators,
                 objects=['table1', 'objA', 'objB'],
                 movePoses={'objA': parkingBad,
                            'objB': back})

    # Small var
    targetVar = (0.0001, 0.0001, 0.0001, 0.0005)
    targetDelta = (0.02, 0.02, 0.02, 0.05)

    easyGoal = State([Bd([SupportFace(['objA']), 4, goalProb], True),
                  B([Pose(['objA', 4]),
                     (0.45, -0.4, tZ, 0.0), targetVar, targetDelta,
                     goalProb], True)])
    
    goal = State([Bd([SupportFace(['objA']), 4, goalProb], True),
                  B([Pose(['objA', 4]),
                     front.xyztTuple(), targetVar, targetDelta,
                     goalProb], True),
                  Bd([SupportFace(['objB']), 4, goalProb], True),
                  B([Pose(['objB', 4]),
                     back.xyztTuple(), targetVar, targetDelta,
                     goalProb], True)])

    flatSkeleton = [[place.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
                     move,
                     pick.applyBindings({'Obj' : 'objA'}),
                     move,
                     place.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
                     move,
                     lookAtHand.applyBindings({'Hand' : 'left'}),
                     move, 
                     pick.applyBindings({'Obj' : 'objB'}),
                     move,                                              
                     place.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
                     move,
                     lookAtHand.applyBindings({'Hand' : 'left'}),
                     move, 
                     pick.applyBindings({'Obj' : 'objA'}),
                     move,
                     place.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
                     move,
                     lookAtHand.applyBindings({'Hand' : 'left'}),
                     move, 
                     pick.applyBindings({'Obj' : 'objB'}),
                     move]]


    easySkeleton = [[place.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
                     move,
                     pick.applyBindings({'Obj' : 'objA'}),
                     move, 
                     poseAchCanPickPlace,
                     place.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
                     move,
                     lookAtHand.applyBindings({'Hand' : 'left'}),
                     move, 
                     pick.applyBindings({'Obj' : 'objB'}),
                     move]]

    t.run(goal,  #goal
          hpn = hpn,
          skeleton = easySkeleton if skeleton else None, #flatSkeleton,
          hierarchical = hierarchical,
          heuristic = heuristic,
          regions=['table1Top']
          )

# 20.  Swap!
def test20(hpn = True, skeleton = False, hierarchical = False,
           heuristic = habbs, easy = False, rip = False):

    glob.rebindPenalty = 50
    goalProb, errProbs = (0.4, tinyErrProbs) if easy else (0.99,typicalErrProbs)
    glob.monotonicFirst = True


    front = util.Pose(0.95, 0.0, tZ, 0.0)
    back = util.Pose(1.1, 0.0, tZ, 0.0)
    parking1 = util.Pose(0.95, 0.3, tZ, 0.0)
    parking2 = util.Pose(0.95, -0.3, tZ, 0.0)
    t = PlanTest('test20',  errProbs, allOperators,
                 objects=['table1', 'objA', 'objB'],
                 movePoses={'objA': back,
                            'objB': front},
                 varDict = {'objA': (0.05**2,0.05**2, 1e-10,0.2**2),
                            'objB': (0.05**2,0.05**2, 1e-10,0.2**2)})

    # This just gets us down the first left expansion
    hierSkel = [[place, place], #0
                [place, poseAchCanPickPlace], #1
                [poseAchCanPickPlace, place], #2
                [place, pick], #3
                [lookAtHand.applyBindings({'Hand' : 'left', 'Obj':'objA'}),
                 pick,
                 poseAchCanPickPlace], #4
                [poseAchCanPickPlace, lookAt, place], #5
                [place], #6
                [place, lookAtHand, pick],#7
                [pick],  #8
                [pick, move], #9
                [move]] #10


    swapSkel = [[place.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
                 move,
                 pick.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
                 move,
                 place.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
                 move,
                 pick.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
                 move,
                 lookAt.applyBindings({'Obj' : 'objA'}),
                 move,
                 place.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
                 move,
                 pick.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
                 move,
                 place.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
                 move,
                 pick.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
                 move]]*5

    # Small var
    targetVar = (0.0001, 0.0001, 0.0001, 0.0005)
    targetDelta = (0.02, 0.02, 0.02, 0.05)
    
    goal = State([Bd([SupportFace(['objA']), 4, goalProb], True),
                  B([Pose(['objA', 4]),
                     front.xyztTuple(), targetVar, targetDelta,
                     goalProb], True),
                  Bd([SupportFace(['objB']), 4, goalProb], True),
                  B([Pose(['objB', 4]),
                     back.xyztTuple(), targetVar, targetDelta,
                     goalProb], True)])

    t.run(goal,
          hpn = hpn,
          skeleton = hierSkel if skeleton else None,
          heuristic = heuristic,
          hierarchical = hierarchical,
          rip = rip,
          regions=['table1Top']
          )

# 20a.  A situation we encounter if we serialize badly.  A is in
# front, B is in the hand.
def test20a(hpn = True, skeleton = False, hierarchical = False,
           heuristic = habbs, easy = False, rip = False):

    glob.rebindPenalty = 50
    goalProb, errProbs = (0.4, tinyErrProbs) if easy else (0.99,typicalErrProbs)
    glob.monotonicFirst = False

    front = util.Pose(0.95, 0.0, tZ, 0.0)
    back = util.Pose(1.15, 0.0, tZ, 0.0)

    t = PlanTest('test20a',  errProbs, allOperators,
                 objects=['table1', 'objA', 'objB'],
                 movePoses={'objA': front,
                            'objB': back},
                 varDict = {'objA': (.01**2, .01**2, .001**2, .01**2)})

    flatSkel = [[lookAt.applyBindings({'Obj' : 'objA'}),
                move,
                place.applyBindings({'Obj' : 'objA', 'Hand' : 'right'}),
                move,
                lookAt.applyBindings({'Obj' : 'objB'}),
                move,                
                place.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
                move,
                pick.applyBindings({'Obj' : 'objA', 'Hand' : 'right'}),
                move,
                lookAt.applyBindings({'Obj' : 'objA'}),
                move]]

    flatSkelRegrasp = [[lookAt.applyBindings({'Obj' : 'objB'}),
                move,
                place.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
                move,
                pick.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
                move,
                lookAt.applyBindings({'Obj' : 'objB'}),
                move,                
                place.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
                move]]

    grasped = 'objB'
    hand = 'left'
    def initBel(bs):
        # Change pbs so obj B is in the hand
        gm = (0, -0.025, 0, 0)
        gv = (0.014**2, 0.014**2, 0.0001**2, 0.022**2)
        gd = (1e-4,)*4
        gf = 0
        bs.pbs.updateHeld(grasped, gf, PoseD(gm, gv), hand, gd)
        bs.pbs.excludeObjs([grasped])
        bs.pbs.shadowWorld = None # force recompute

    def initWorld(bs, realWorld):
        attachedShape = bs.pbs.getRobot().\
                         attachedObj(bs.pbs.getShadowWorld(0.9), hand)
        shape = bs.pbs.getWorld().\
               getObjectShapeAtOrigin(grasped).applyLoc(attachedShape.origin())
        realWorld.robot.attach(shape, realWorld, hand)
        robot = bs.pbs.getRobot()
        cart = realWorld.robotConf.cartConf()
        handPose = cart[robot.armChainNames[hand]].compose(gripperTip)
        pose = shape.origin()
        realWorld.held[hand] = grasped
        realWorld.grasp[hand] = handPose.inverse().compose(pose)
        realWorld.delObjectState(grasped)    

    # Small var
    targetVar = (0.0001, 0.0001, 0.0001, 0.0005)
    targetDelta = (0.02, 0.02, 0.02, 0.05)
    
    goal = State([Bd([SupportFace(['objA']), 4, goalProb], True),
                  B([Pose(['objA', 4]),
                     front.xyztTuple(), targetVar, targetDelta,
                     goalProb], True),
                  Bd([SupportFace(['objB']), 4, goalProb], True),
                  B([Pose(['objB', 4]),
                     back.xyztTuple(), targetVar, targetDelta,
                     goalProb], True)])

    t.run(goal,
          hpn = hpn,
          skeleton = flatSkelRegrasp if skeleton else None,
          heuristic = heuristic,
          hierarchical = hierarchical,
          rip = rip,
          regions=['table1Top'],
          initBelief = initBel,
          initWorld = initWorld
          )

    
# stack objects?
def testStack(hpn = True, skeleton = False, hierarchical = False,
           heuristic = habbs):
    p1 = util.Pose(0.95, 0.0, tZ, 0.0)
    p2 = util.Pose(1.1, 0.0, tZ, 0.0)
    p3 = util.Pose(0.95, 0.2, tZ, 0.0)
    t = PlanTest('test18',  smallErrProbs, allOperators,
                 objects=['table1', 'objA', 'objB', 'objC'],
                 movePoses={'objA': p1,
                            'objB': p2,
                            'objC': p3})

    skel = [[place.applyBindings({'Obj' : 'objC', 'Hand' : 'left'}),
             move,
             lookAtHand.applyBindings({'Obj' : 'objC'}),
             move,
             pick.applyBindings({'Obj' : 'objC', 'Hand' : 'left'}),
             move,
             place.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
             move,
             lookAtHand.applyBindings({'Obj' : 'objA'}),
             move,
             pick.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
             move]]*5

    goal = State([Bd([In(['objA', 'objBTop']), True, .4], True),
                  # Bd([In(['objC', 'objATop']), True, .4], True)
                  ])

    t.run(goal,
          hpn = hpn,
          skeleton = skel if skeleton else None,
          heuristic = heuristic,
          hierarchical = hierarchical,
          regions=['objATop', 'objBTop', 'table1Top']
          )

# Empty hand
def test21(hpn = True, skeleton = False, hierarchical = False,
           heuristic = habbs, easy=False):
    p1 = util.Pose(0.95, 0.0, tZ, 0.0)
    p2 = util.Pose(0.95, 0.4, tZ, 0.0)

    t = PlanTest('test21',  smallErrProbs, allOperators,
                 objects=['table1', 'objA', 'objB'],
                 movePoses={'objA': p1,
                            'objB': p2})

    # Init var
    initGraspVar = (1e-4,)*4    # very small
    # initGraspVar = (0.01, 0.01, 0.01, 0.05)    # very big

    # Small var
    targetVar = (0.0001, 0.0001, 0.0001, 0.0005)
    targetDelta = (0.001, 0.001, 0.001, 0.005)
    # Increase this
    goalProb = 0.4
    # Need to empty the hand in order to achieve this
    goal3 = State([Bd([SupportFace(['objA']), 4, goalProb], True),
                  B([Pose(['objA', 4]),
                     p2.xyztTuple(), targetVar, targetDelta,
                     goalProb], True)])
    # Just empty the hand.  No problem.
    goal1 = State([Bd([Holding(['left']), 'none', goalProb], True)])
    # Empty the hand.  Other goal condition is already true.
    goal2 = State([Bd([Holding(['left']), 'none', goalProb], True),
                   Bd([SupportFace(['objA']), 4, goalProb], True),
                   B([Pose(['objA', 4]),
                     p1.xyztTuple(), targetVar, targetDelta,
                     goalProb], True)])

    grasped = 'objB'
    hand = 'left'
    def initBel(bs):
        # Change pbs so obj B is in the hand
        gm = (0, -0.025, 0, 0)
        gv = initGraspVar
        gd = (1e-4,)*4
        gf = 0
        bs.pbs.updateHeld(grasped, gf, PoseD(gm, gv), hand, gd)
        bs.pbs.excludeObjs([grasped])
        bs.pbs.shadowWorld = None # force recompute

    def initWorld(bs, realWorld):
        attachedShape = bs.pbs.getRobot().attachedObj(bs.pbs.getShadowWorld(0.9), hand)
        shape = bs.pbs.getWorld().getObjectShapeAtOrigin(grasped).applyLoc(attachedShape.origin())
        realWorld.robot.attach(shape, realWorld, hand)
        robot = bs.pbs.getRobot()
        cart = realWorld.robotConf.cartConf
        handPose = cart[robot.armChainNames[hand]].compose(gripperTip)
        pose = shape.origin()
        realWorld.held[hand] = grasped
        realWorld.grasp[hand] = handPose.inverse().compose(pose)
        realWorld.delObjectState(grasped)    

    skeleton1 = [[place, move, lookAtHand, move, lookAtHand,
                  move]]
    t.run(goal1,
          hpn = hpn,
          skeleton = skeleton1 if skeleton else None,
          heuristic = heuristic,
          regions = ['table1Top'],
          initBelief = initBel,
          initWorld = initWorld
          )

# Need to verify that hand is empty
def test22(hpn = True, skeleton = False, hierarchical = False,
           heuristic = habbs, easy=False):
    p1 = util.Pose(0.95, 0.0, tZ, 0.0)
    p2 = util.Pose(0.95, 0.4, tZ, 0.0)

    t = PlanTest('test22',  smallErrProbs, allOperators,
                 objects=['table1', 'objA', 'objB'],
                 movePoses={'objA': p1,
                            'objB': p2})

    # Small var
    targetVar = (0.0001, 0.0001, 0.0001, 0.0005)
    # Bigger for testing
    targetVar = (0.001, 0.001, 0.001, 0.005)
    targetDelta = (0.001, 0.001, 0.001, 0.005)
    # Increase this
    goalProb = 0.4

    goal1 = State([Bd([Holding(['left']), 'none', .97], True)])


    # Need to be very sure
    goal3 = State([Bd([Holding(['left']), 'objA', .97], True),
                   Bd([GraspFace(['objA', 'left']), 0, .97], True),
                   B([Grasp(['objA', 'left', 0]),
                     (0,-0.025,0,0), (0.001, 0.001, 0.001, 0.001), (0.001,)*4,
                     .97], True)])

    # Need to verify empty hand in order to achieve this
    goal2 = State([Bd([SupportFace(['objA']), 4, goalProb], True),
                  B([Pose(['objA', 4]),
                     p2.xyztTuple(), targetVar, targetDelta,
                     goalProb], True)])

    def initBel(bs):
        # Change pbs so we are unsure whether both hands are empty
        bs.pbs.held['left'] = dist.DDist({'none' : .6, 'objA' : .3, 'objB': .1})
        bs.pbs.held['right'] = dist.DDist({'none' : .6, 'objA' : .1, 'objB': .3})
        bs.pbs.shadowWorld = None # force recompute
        bs.pbs.draw(0.9, 'W')
        bs.pbs.draw(0.9, 'Belief')

    skeleton1 = [[lookAtHand.applyBindings({'Hand' : 'left'}),
                  move,
                  lookAtHand.applyBindings({'Hand' : 'left'}),
                  move]]
    skeleton2 = [[lookAt, move, place, move, pick, move, lookAtHand, move]]

    t.run(goal2,
          hpn = hpn,
          skeleton = skeleton1 if skeleton else None,
          heuristic = heuristic,
          initBelief = initBel,
          regions = ['table1Top']
          )
'''

def prof(test, n=50):
    import cProfile
    import pstats
    cProfile.run(test, 'prof')
    p = pstats.Stats('prof')
    p.sort_stats('cumulative').print_stats(n)
    # p.sort_stats('cumulative').print_callers(n)


# Evaluate on details and a fluent to flush the caches and evaluate
def firstAid(details, fluent = None):
    glob.debugOn.extend(['confReachViol', 'confViolations'])

    pbs = details.pbs
    bc = pbs.beliefContext

    pbs.getRoadMap().confReachCache.clear()
    bc.pathObstCache.clear()
    bc.objectShadowCache.clear()
    for c in bc.genCaches.values():
        c.clear()

    pr2GenAux.graspConfGenCache.clear()

    # bc.world.robot.cacheReset()

    pr2Visible.cache.clear()

    belief.hCacheReset()
    
    if fluent:
        return fluent.valueInDetails(details)

# Get false fluents
def ff(g, details):
    return [thing for thing in g.fluents if thing.isGround() \
            and thing.valueInDetails(details) == False]


def testReact():
    t = PlanTest('testReact', typicalErrProbs, allOperators, multiplier = 1)
    startConf = makeConf(t.world.robot, 0.0, 0.0, 0.0, dx=0.1, dz=0.1)
    result, cnfOut = pr2GoToConf(startConf, 'move')
    result, cnfOut = pr2GoToConf(startConf, 'look')
    # Reset the internal coordinate frames
    result, cnfOut = pr2GoToConf(startConf, 'reset')
    glob.debugOn.append('invkin')
    testReactive(startConf)

def gripOpen(conf, hand, width=0.08):
    return conf.set(conf.robot.gripperChainNames[hand], [width])

def testOpen(hand='left'):
    t = PlanTest('testReact', typicalErrProbs, allOperators, multiplier = 1)
    startConf = makeConf(t.world.robot, 0.0, 0.0, 0.0, dx=0.1, dz=0.1)[0]
    result, cnfOut = pr2GoToConf(gripOpen(startConf, hand), 'open')    

