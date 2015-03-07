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
from dist import DDist, DeltaDist

import pr2Robot
reload(pr2Robot)
from pr2Robot import makePr2Chains, PR2, JointConf, CartConf, pr2Init, \
     gripperTip

import pr2RoadMap
reload(pr2RoadMap)
from pr2RoadMap import RoadMap

import pr2Fluents
reload(pr2Fluents)
from pr2Fluents import Conf, SupportFace, Pose, Holding, GraspFace, Grasp,\
     partition, In

import pr2PlanBel
reload(pr2PlanBel)
from pr2PlanBel import BeliefState, BeliefContext, PBS

import pr2Gen
reload(pr2Gen)

import pr2Ops
reload(pr2Ops)
from pr2Ops import move, pick, place, lookAt, poseAchCanReach, poseAchCanSee,\
      lookAtHand, hRegrasp

import pr2Sim
reload(pr2Sim)
from pr2Sim import RealWorld


writeSearch = True

######################################################################
# Right Arm??
######################################################################

useRight = True

######################################################################
# Test Rig
######################################################################

def habbs(s, g, ops, ancestors):
    hops = ops + [hRegrasp]
    val = hAddBackBSetID(s, g, hops, ancestors, ddPartitionFn = partition,
                         maxK = 20)
    return val

from timeout import timeout, TimeoutError

# 5 min timeout for all tests
@timeout(300)
def testFunc(n, skeleton=None, heuristic=habbs, hierarchical=False):
    eval('test%d(skeleton=skeleton, heuristic=heuristic, hierarchical=hierarchical)'%n)

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
        testRepeat(i, repeat=repeat, **args)
    print testResults

######################################################################
# Test Cases
######################################################################

def cl(window='W'):
    wm.getWindow(window).clear()

def Ba(bb, **prop): return shapes.BoxAligned(np.array(bb), None, **prop)
def Sh(args, **prop): return shapes.Shape(list(args), None, **prop)

viewPort =  [-1.75, 1.75, -1.75, 1.75, 0, 2]
moreGD = False
def testWorld(include = ['objA', 'objB', 'objC'],
              draw = True):
    x0=-1.75; x1=1.75; y0=-1.75; y1=1.75; dz=1.80; w = 0.1
    wm.makeWindow('W', viewPort, 800)
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
    table1 = Sh([place((-0.5, 0.5), (-0.25, 0.25), (0.0, 0.6))], name = 'table1', color='brown')
    if 'table1' in include: world.addObjectShape(table1)
    table2 = Sh([place((-0.5, 0.5), (-0.25, 0.25), (0.0, 0.6))], name = 'table2', color='brown')
    if 'table2' in include: world.addObjectShape(table2)
    table3 = Sh([place((-0.5, 0.5), (-0.125, 0.125), (0.0, 0.6))], name = 'table3', color='brown')
    if 'table3' in include: world.addObjectShape(table3)

    for i in range(1,4):
        name = 'table%d'%i
        if name in include:
            bbox = world.getObjectShapeAtOrigin(name).bbox()
            regName = name+'Top'
            print 'Region', regName, '\n', bbox
            world.addObjectRegion(name, regName, Sh([Ba(bbox)], name=regName),
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
        thing = Sh([placeSc((-0.05, 0.05), (-0.025, 0.025), (0.0, 0.1))],
                   name = objName, color=colors[i%len(colors)])
        world.addObjectShape(thing)
        bbox = bboxGrow(thing.bbox(), np.array([0.075, 0.075, 0.02]))
        regName = objName+'Top'
        print 'Region', regName, '\n', bbox
        world.addObjectRegion(objName, regName, Sh([Ba(bbox)], name=regName),
                              util.Pose(0,0,bbox[1,2]+0.02,0))

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
    gMat2= np.array([(1.,0.,0.,0.),
                     (0.,0.,1.,-0.025),
                     (0.,-1.,0.,0.),
                     (0.,0.,0.,1.)])
    for obj in manipulanda:
        world.graspDesc[obj] = [GDesc(obj, util.Transform(gMat0),
                                      0.05, 0.05, 0.025),
                                # Needs more general confs for grasping
                                # GDesc(obj, util.Transform(gMat2),
                                #      0.05, 0.05, 0.025)
                                ]
    if moreGD:
        for obj in manipulanda:
            world.graspDesc[obj].extend([GDesc(obj, util.Transform(gMat1),
                                               0.05, 0.05, 0.025)])

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

def makeConf(robot,x,y,th,g=0.07,up=True):
    c = JointConf(pr2Init.copy(), robot)
    c = c.set('pr2Base', [x, y, th])
    c = c.set('pr2LeftGripper', [g])
    if useRight:
        c = c.set('pr2RightGripper', [g])
    if up:
        cart = robot.forwardKin(c)
        base = cart['pr2Base']
        # Used to be 0.2, 0.33, 0.75 or 0.2, 0.5, 0.8
        hand = base.compose(util.Transform(transf.translation_matrix([0.2,0.33,0.75])))
        cart = cart.set('pr2LeftArm', hand)
        if useRight:
            hand = base.compose(util.Transform(transf.translation_matrix([0.2,-0.33,0.75])))
            cart = cart.set('pr2RightArm', hand)
        c = robot.inverseKin(cart, conf=c)
    return c

class PlanTest:
    def __init__(self, name, domainProbs,
                 objects = ['table1','objA'],
                 movePoses = {}, held = None, grasp = None,
                 multiplier = 8, var = 1.0e-10, randomizeObjects = False,
                 varDict = None):
        self.name = name
        self.size = 1.75                 # half size of workspace
        self.multiplier = multiplier
        self.objects = objects          # list of objects to consider
        self.domainProbs = domainProbs
        self.world = testWorld(include=self.objects)
        self.initConfs = [makeConf(self.world.robot,
                                   x*self.size/float(multiplier),
                                   y*self.size/float(multiplier), 0) \
                              for x in range(-multiplier, (multiplier+1)) \
                              for y in range(-multiplier, (multiplier+1))] + \
                              [makeConf(self.world.robot,
                                        x*self.size/float(multiplier),
                                        y*self.size/float(multiplier), math.pi/2) \
                               for x in range(-multiplier/2, (multiplier/2)+1) \
                               for y in range(-multiplier/2, (multiplier/2)+1)] + \
                              [makeConf(self.world.robot,
                                        x*self.size/float(multiplier),
                                        y*self.size/float(multiplier), -math.pi/2) \
                               for x in range(-multiplier/2, (multiplier/2)+1) \
                               for y in range(-multiplier/2, (multiplier/2)+1)] + \
                               [makeConf(self.world.robot,
                                        x*self.size/float(multiplier),
                                        y*self.size/float(multiplier), math.pi) \
                               for x in range(-multiplier/2, (multiplier/2)+1) \
                               for y in range(-multiplier/2, (multiplier/2)+1)]
        print 'Creating', len(self.initConfs), 'initial confs'
        var4 = (var, var, 0.0, var)
        del0 = (0.0, 0.0, 0.0, 0.0)
        ff = lambda o: self.world.getFaceFrames(o) if o in objects else []
        # The poses of the supporting face frames (the placement)
        fixObjPoses = {'table1':util.Pose(0.6, 0.0, 0.0, math.pi/2),
                       'table2': util.Pose(0.5, -0.75, 0.0, 0.0),
                       'table3': util.Pose(1.1,0.0,0.0,math.pi/2),
                       'cupboardSide1': util.Pose(0.6, -0.2, 0.6, 0.0),
                       'cupboardSide2': util.Pose(0.6, 0.2, 0.6, 0.0)}
        moveObjPoses = {'objA': util.Pose(0.6, 0.0, 0.61, 0.0),
                        'objB': util.Pose(0.45, -0.4, 0.61, 0.0),
                        'objC': util.Pose(-0.75, -1.2, 0.81, 0.0),
                        'objD': util.Pose(0.45, -0.2, 0.61, 0.0),
                        'objE': util.Pose(0.45, 0.0, 0.61, 0.0),
                        'objF': util.Pose(0.45, 0.2, 0.61, 0.0),
                        'objG': util.Pose(0.45, 0.4, 0.61, 0.0),
                        'objH': util.Pose(0.45, 0.6, 0.61, 0.0),
                        'objI': util.Pose(0.45, 0.8, 0.61, 0.0)}
                   
        moveObjPoses.update(movePoses)           # input poses
        print 'updated', moveObjPoses
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
        self.randomizeObjects = randomizeObjects
        self.operators = {'move': move, 'pick': pick, 'place': place,
                          'lookAt': lookAt, 'poseAchCanReach' : poseAchCanReach,
                          'poseAchCanSee' : poseAchCanSee,
                          'lookAtHand' : lookAtHand}
        wm.makeWindow('Belief', viewPort, 500)
        wm.makeWindow('World', viewPort, 500)

    def buildBelief(self, home=None, regions=[]):
        world = self.world
        belC = BeliefContext(world)
        pr2Home = home or makeConf(world.robot, -0.5, 0.0, 0.0, up=True)
        rm = RoadMap(pr2Home, world, kNearest = 20,
                     moveChains = \
                     ['pr2Base', 'pr2LeftGripper', 'pr2LeftArm', 'pr2RightGripper', 'pr2RightArm'] if useRight \
                     else ['pr2Base', 'pr2LeftGripper', 'pr2LeftArm'],)
        rm.batchAddNodes(self.initConfs)
        belC.roadMap = rm
        pbs = PBS(belC, conf=pr2Home, fixObjBs = self.fix.copy(), moveObjBs = self.move.copy(),
        regions = regions, domainProbs=self.domainProbs) 
        pbs.draw(0.9, 'Belief')
        bs = BeliefState()
        bs.pbs = pbs
        bs.domainProbs = self.domainProbs
        bs.awayRegion = 'table2Top'
        self.bs = bs

    def run(self, goal, skeleton = None, operators = None, hpn = True,
            home=None, regions = [], hierarchical = False, heuristic = None,
            greedy = 0.7, simulateError = False):
        fbch.inHeuristic = False
        if skeleton:
            fbch.dotSearchId = 0    # should make skeletons work without reload
        startTime = time.clock()
        fbch.flatPlan = not hierarchical
        fbch.plannerGreedy = greedy  # somewhat greedy by default
        pr2Sim.simulateError = simulateError
        for win in wm.windows:
            wm.getWindow(win).clear()
        self.buildBelief(home=home, regions=regions)
        self.bs.pbs.draw(0.9, 'W')
        self.bs.pbs.draw(0.9, 'Belief')
        # Initialize simulator
        self.realWorld = RealWorld(self.bs.pbs.getWorld(),
                                   self.domainProbs) # simulator
        self.realWorld.setRobotConf(self.bs.pbs.conf)
        for obj in self.objects:
            self.realWorld.setObjectPose(obj, self.bs.pbs.getPlaceB(obj).objFrame())
        if self.bs.pbs.held['left'].mode() != 'none':
            self.realWorld.held['left'] = self.bs.pbs.held['left'].mode()
            print 'Need to figure out grasp!'
            assert False
        self.realWorld.draw('World')
        s = State([], details = self.bs)
        print '**************', self.name,\
                 'Hierarchical' if hierarchical else '', '***************'
        if hpn:
            if skeleton:
                skel = [[(self.operators[o] if type(o) == str else o) \
                          for o in stuff] for stuff in skeleton]
            HPN(s,
                goal,
                [self.operators[o] for o in operators],
                self.realWorld,
                hpnFileTag = self.name,
                skeleton = skel if skeleton else None,
                h = heuristic,
                verbose = False,
                fileTag = self.name if writeSearch else None)
        else:
            p = planBackward(s,
                             goal,
                             [self.operators[o] for o in operators],
                             h = heuristic,
                             fileTag = self.name if writeSearch else None)
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

typicalErrProbs = DomainProbs(\
            # stdev, as a percentage of the motion magnitude
            odoError = (0.05, 0.05, 0.05, 0.05),
            # variance in observations; diagonal for now
            obsVar = (0.01**2, 0.01**2, 1e-12, 0.01**2),
            # variance in grasp after picking
            pickVar = (0.02**2, 0.02**2, 0.02**2, 0.04**2),
            # variance in grasp after placing
            placeVar = (0.02**2, 0.02**2, 0.02**2, 0.02**2),
            # pickTolerance
            pickTolerance = (0.02, 0.02, 0.02, 0.02))

smallErrProbs = DomainProbs(\
            # stdev, as a percentage of the motion magnitude
            odoError = (0.01, 0.01, 0.01, 0.01),
            # variance in observations; diagonal for now
            obsVar = (0.001**2, 0.001**2, 1e-12, 0.002**2),
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
            obsVar = (0.00001**2, 0.00001**2, 1e-12, 0.00002**2),
            # variance in grasp after picking
            pickVar = (0.0001**2, 0.0001**2, 0.0001**2, 0.0001**2),
            # variance in grasp after placing
            placeVar = (0.0001**2, 0.0001**2, 0.0001**2, 0.0001**2),
            # pickTolerance
            pickTolerance = (0.02, 0.02, 0.02, 0.02))

# Try to make a plan!     Just move
def test1(hpn=True, skeleton=False, heuristic=habbs, hierarchical=False):
    t = PlanTest('test1', typicalErrProbs)
    goalConf = makeConf(t.world.robot, 0.0, 1.0, 0.0, up=True)
    confDeltas = (0.05, 0.05, 0.05, 0.05)
    goal = State([Conf([goalConf, confDeltas], True)])
    t.run(goal,
          hpn = hpn,
          operators=['move', 'pick'],
          heuristic=heuristic
          )

# Pick something up! and move
def test2(hpn = True, skeleton=False, hand='left', flip = False, gd = 0,
          heuristic=habbs, hierarchical=False):
    global moreGD
    if gd != 0: moreGD = True           # hack!
    t = PlanTest('test2', smallErrProbs, objects=['table1', 'objA'])
    goalConf = makeConf(t.world.robot, 0.0, 1.0, 0.0, up=True)
    confDeltas = (0.05, 0.05, 0.05, 0.05)
    goal = State([Bd([Holding([hand]), 'objA', .6], True),
                  Bd([GraspFace(['objA', hand]), gd, .6], True),
                  B([Grasp(['objA', hand, gd]),
                     (0,-0.025,0,0), (0.001, 0.001, 0.001, 0.001),
                     (0.001,)*4, 0.6], True),
                  Conf([goalConf, confDeltas], True)])
    homeConf = makeConf(t.world.robot, -0.5, 0.0, math.pi, up=True) \
                         if flip else None
    t.run(goal,
          hpn = hpn,
          skeleton = [['move', 'pick', 'move']] \
          if skeleton else None,
          heuristic = heuristic,
          operators=['move', 'pick', 'place', 'lookAt', 'poseAchCanReach',
                     'poseAchCanSee', 'lookAtHand'],
          home=homeConf
          )
    return t

# pick and place
def test3(hpn = True, skeleton = False, hierarchical = False, heuristic=habbs):
    t = PlanTest('test3',  smallErrProbs,
                 objects=['table1', 'objA'])
    targetPose = (0.55, 0.25, 0.61, 0.0)
    goalProb = 0.5
    # large target var is no problem
    targetVar = (0.01, 0.01, 0.01, 0.05)
    goal = State([Bd([SupportFace(['objA']), 4, .5], True),
                  B([Pose(['objA', 4]),
                     targetPose, targetVar, (0.001,)*4,
                     0.5], True),
                  Bd([Holding(['left']), 'none', 0.5], True)])

    skel = [[place.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
                 move,
                 pick.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
                 move]]
    t.run(goal,
          hpn = hpn,
          skeleton = skel if skeleton else None,
          hierarchical = hierarchical,
          operators=['move', 'pick', 'place', 'lookAt', 'poseAchCanReach',
                     'poseAchCanSee', 'lookAtHand'],
          heuristic = heuristic
          )

def test4(hpn = True, hierarchical = False, skeleton = False,
          heuristic = habbs):
    t = PlanTest('test4',  smallErrProbs,
                 objects=['table1', 'objA', 'objB'])
    targetPose = (0.55, 0.25, 0.61, 0.0)
    targetPoseB = (0.55, -0.2, 0.61, 0.0)
    targetVar = (0.001, 0.001, 0.001, 0.005) 

    goalConf = makeConf(t.world.robot, 0.0, 1.0, 0.0, up=True)
    confDeltas = (0.05, 0.05, 0.05, 0.05)

    goalProb = 0.2  # ridiculous

    goal = State([Bd([SupportFace(['objA']), 4, goalProb], True),
                  B([Pose(['objA', 4]),
                     targetPose, targetVar, (0.001,)*4,
                     goalProb], True),
                  Bd([SupportFace(['objB']), 4, goalProb], True),
                  B([Pose(['objB', 4]),
                     targetPoseB, targetVar, (0.001,)*4,
                     goalProb], True)])

    skel = [[place.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
                 move,
                 pick.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
                 move,
                 place.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
                 move,
                 pick.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
                 move]]

    t.run(goal,
          hpn = hpn,
          operators=['move', 'pick', 'place', 'lookAt', 'poseAchCanReach', 'poseAchCanSee', 'lookAtHand'],
          skeleton = skel if skeleton else None,
          hierarchical = hierarchical,
          heuristic = heuristic
          )

# Test placing in a region    
def test5(hpn = True, skeleton = False, heuristic=habbs, hierarchical = False):
    p1 = util.Pose(0.45, 0.0, 0.61, 0.0)
    p2 = util.Pose(0.6, 0.0, 0.61, 0.0)
    t = PlanTest('test5',  smallErrProbs, 
                 objects=['table1', 'objA', 'table2'],
                 movePoses={'objA': p1,
                            'objB': p2})

    goal = State([Bd([In(['objA', 'table2Top']), True, .4], True)])

    t.run(goal,
          hpn = hpn,
          skeleton = [['place', 'move', 'pick', 'move']] if skeleton else None,
          operators=['move', 'pick', 'place', 'lookAt', 'poseAchCanReach',
                     'poseAchCanSee', 'lookAtHand'],
          heuristic = heuristic
          )

# Test looking
def test6(hpn = True, skeleton=False, heuristic=habbs, hierarchical = False):
    p1 = util.Pose(0.45, 0.0, 0.61, 0.0)
    p2 = util.Pose(0.6, 0.0, 0.61, 0.0)
    t = PlanTest('test6', smallErrProbs,
                 objects=['table1', 'objA', 'table2'],
                 movePoses={'objA': p1,
                            'objB': p2},
                 varDict = {'objA': (0.01*2,)*4})

    goalProb = 0.8

    goal = State([B([Pose(['objA', 4]), p1.xyztTuple(),
                     (0.0001, 0.0001, 0.0001, 0.001),
                          (0.01,)*4, goalProb], True),
                  Bd([SupportFace(['objA']), 4, goalProb], True)])
    t.run(goal,
          hpn = hpn,
          skeleton = [['lookAt', 'move']] if skeleton else None,
          operators=['move', 'pick', 'place', 'lookAt', 'poseAchCanReach',
                     'poseAchCanSee', 'lookAtHand'],
          hierarchical = hierarchical,
          heuristic = heuristic,
          )

# Test look, pick, place
def test7(hpn = True, flip=False, skeleton = False, heuristic=habbs,
          hierarchical = False):
    p1 = util.Pose(0.45, 0.0, 0.61, 0.0)
    p2 = util.Pose(0.6, 0.0, 0.61, 0.0)
    t = PlanTest('test7',  smallErrProbs, # !! should be typical
                 objects=['table1', 'objA', 'table2'],
                 movePoses={'objA': p1,
                            'objB': p2},
                 varDict = {'objA': (0.01*2,)*4})

    targetPose = (0.55, 0.25, 0.61, 0.0)
    goalProb = 0.8

    goal = State([Bd([Holding(['left']), 'objA', goalProb], True),
                  Bd([GraspFace(['objA', 'left']), 0, goalProb], True),
                  B([Grasp(['objA', 'left', 0]),
                     (0,-0.025,0,0), (0.001, 0.001, 0.001, 0.001), (0.001,)*4,
                     goalProb], True)])
    homeConf = makeConf(t.world.robot, -0.5, 0.0, math.pi, up=True) if flip else None
    skel = [['pick', 'move', 'lookAt', 'move']]*3
    t.run(goal,
          hpn = hpn,
          skeleton = skel if skeleton else None,
          operators=['move', 'pick', 'place', 'lookAt', 'poseAchCanReach',
                     'poseAchCanSee', 'lookAtHand'],
          heuristic = heuristic,
          hierarchical = hierarchical,
          home=homeConf
          )

# Regrasp!
def test8(hpn = True, skeleton=False, hierarchical = False, 
            heuristic = habbs, hand='left', flip = False, gd=1):
    global moreGD
    if gd != 0: moreGD = True
    t = PlanTest('test8', tinyErrProbs, objects=['table1', 'table3', 'objA'])
    goalConf = makeConf(t.world.robot, 0.0, 1.0, 0.0, up=True)
    confDeltas = (0.05, 0.05, 0.05, 0.05)
    targetProb = 0.4
    goal = State([Bd([Holding([hand]), 'objA', targetProb], True),
                  Bd([GraspFace(['objA', hand]), gd, targetProb], True),
                  B([Grasp(['objA', hand, gd]),
                     (0,-0.025,0,0), (0.005, 0.005, 0.005, 0.05),
                     (0.001,)*4, targetProb], True)])
                  #Conf([goalConf, confDeltas], True)])
    homeConf = makeConf(t.world.robot, -0.5, 0.0, math.pi, up=True) if flip else None
    goodSkel = [['pick',
                 'move',
                  place.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
                 'move',
                 'lookAtHand',
                 'move',
                 'pick',
                 'move']]
    t.run(goal,
          hpn = hpn,
          skeleton = goodSkel if skeleton else None,
          heuristic = heuristic, 
          operators=['move', 'pick', 'place', 'lookAt', 'poseAchCanReach',
                     'poseAchCanSee', 'lookAtHand'],
          hierarchical = hierarchical,
          home=homeConf,
          regions=['table1Top']
          )

def test9(hpn=True, skeleton = False, heuristic=habbs, hierarchical = False):
    t = PlanTest('test9', typicalErrProbs,
                 objects = ['table1'],
                 varDict = {'table1': (0.01*2,)*4})

    goalConf = makeConf(t.world.robot, 0.0, 1.0, 0.0, up=True)
    confDeltas = (0.05, 0.05, 0.05, 0.05)
    goal = State([Conf([goalConf, confDeltas], True)])
    t.run(goal,
          hpn = hpn,
          heuristic = heuristic,
          hierarchical = hierarchical,
          regions=['table1Top'],
          operators=['move', 'pick', 'place', 'lookAt', 'poseAchCanReach',
                     'poseAchCanSee', 'lookAtHand'],
          skeleton = [['move', 'poseAchCanReach',
                       'lookAt']] if skeleton else None,
          )

def test10(hpn = True, skeleton = False, hierarchical = False, heuristic=habbs):
    # need to look at A to clear path to b
    
    t = PlanTest('test10',  smallErrProbs,
                 objects=['table1', 'objA', 'objB'],
                 varDict = {'objA': (0.01*2,)*4})
    targetPose = (0.55, 0.25, 0.61, 0.0)
    targetPoseB = (0.55, -0.2, 0.61, 0.0)
    targetVar = (0.01, 0.01, 0.01, 0.05) 

    goalConf = makeConf(t.world.robot, 0.0, 1.0, 0.0, up=True)
    confDeltas = (0.05, 0.05, 0.05, 0.05)

    goalProb = 0.4

    goal = State([\
                  Bd([SupportFace(['objB']), 4, goalProb], True),
                  B([Pose(['objB', 4]),
                     targetPoseB, targetVar, (0.001,)*4,
                     goalProb], True)])
    t.run(goal,
          hpn = hpn,
          operators=['move', 'pick', 'place', 'lookAt', 'poseAchCanReach',
                     'poseAchCanSee', 'lookAtHand'],
          skeleton = [['place', 'move', 'pick', 'move', 'poseAchCanReach',
                       'lookAt', 'move']]*3 if skeleton else None,
          hierarchical = hierarchical,
          regions=['table1Top'],
          heuristic = heuristic
          )


def test11(hpn = True, skeleton = False, hierarchical = False,
           heuristic = habbs):
    t = PlanTest('test11',  smallErrProbs,
                 objects=['table1', 'objA', 'objB'],
                 varDict = {'objA': (0.01*2,)*4,
                            'objB': (0.01*2,)*4})
    targetPose = (0.55, 0.25, 0.61, 0.0)
    targetPoseB = (0.55, -0.2, 0.61, 0.0)
    targetVar = (0.001, 0.001, 0.001, 0.005) 

    goalConf = makeConf(t.world.robot, 0.0, 1.0, 0.0, up=True)
    confDeltas = (0.05, 0.05, 0.05, 0.05)

    goalProb = 0.7

    goal = State([\
                  Bd([SupportFace(['objA']), 4, goalProb], True),
                  B([Pose(['objA', 4]),
                     targetPose, targetVar, (0.001,)*4,
                     goalProb], True),
                  Bd([SupportFace(['objB']), 4, goalProb], True),
                  B([Pose(['objB', 4]),
                     targetPoseB, targetVar, (0.001,)*4,
                     goalProb], True)])

    glob.monotonicFirst = False
    glob.rebindPenalty = 200

    skel = [[place.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
             'move',
             place.applyBindings({'Obj' : 'objA', 'Hand' : 'right'}),
             'move',
             lookAtHand.applyBindings({'Obj' : 'objA'}),
             'move',
             pick.applyBindings({'Obj' : 'objA', 'Hand' : 'right'}),
             'move',
             lookAtHand.applyBindings({'Obj' : 'objB'}),
             'move',
             pick.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
             'move',
             lookAt.applyBindings({'Obj' : 'objB'}),
             'move',
              lookAt.applyBindings({'Obj' : 'objA'}),
              'move']]*5

    skel = [[place.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
             'move',
             place.applyBindings({'Obj' : 'objB', 'Hand' : 'right'}),
             'move',
             lookAtHand.applyBindings({'Obj' : 'objB'}),
             'move',
             pick.applyBindings({'Obj' : 'objB', 'Hand' : 'right'}),
             'move',
             lookAt.applyBindings({'Obj' : 'objB'}),
             'move',
             lookAtHand.applyBindings({'Obj' : 'objA'}),
             'move',
             pick.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
             'poseAchCanReach',
             'move',
              lookAt.applyBindings({'Obj' : 'objB'}),
             'move',
              lookAt.applyBindings({'Obj' : 'objA'}),
              'move']]*5

                              
    t.run(goal,
          hpn = hpn,
          operators=['move', 'pick', 'place', 'lookAt', 'poseAchCanReach',
                     'poseAchCanSee', 'lookAtHand'],
          skeleton = skel if skeleton else None,
          hierarchical = hierarchical,
          regions=['table1Top'],
          heuristic = heuristic
          )
    
def test12(hpn = True, skeleton = False, hierarchical = False,
           heuristic = habbs):
    t = PlanTest('test12',  smallErrProbs,
                 objects=['table1', 'table2',
                          'objA', 'objB', 'objD',
                          'objE', 'objF', 'objG'])
    targetPose = (0.55, 0.25, 0.61, 0.0)
    targetPoseB = (0.55, -0.2, 0.61, 0.0)
    targetVar = (0.001, 0.001, 0.001, 0.005) 

    goalConf = makeConf(t.world.robot, 0.0, 1.0, 0.0, up=True)
    confDeltas = (0.05, 0.05, 0.05, 0.05)

    goalProb = 0.1

    hand = 'left'
    gd = 0
    goal = State([Bd([Holding([hand]), 'objA', goalProb], True),
                  Bd([GraspFace(['objA', hand]), 0, goalProb], True),
                  B([Grasp(['objA', hand, gd]),
                     (0,-0.025,0,0), (0.001, 0.001, 0.001, 0.001),
                     (0.001,)*4, goalProb], True)])
    t.run(goal,
          hpn = hpn,
          operators=['move', 'pick', 'place', 'lookAt', 'poseAchCanReach',
                     'poseAchCanSee', 'lookAtHand'],
          skeleton = [['pick',
                       'move',
                       'poseAchCanReach',
                       'lookAt',  # E
                       'move',
                       place.applyBindings({'Hand' : 'left'}),
                       'move',
                       'pick',
                       'move',
                       'poseAchCanReach',
                       'lookAt',  
                       'move',
                       place.applyBindings({'Hand' : 'left'}),
                       'move',
                       'pick',
                       'move',
                       ]] if skeleton else None,
          hierarchical = hierarchical,
          # regions=['table2Top'],
          regions=['table2Top', 'table1Top'],
          heuristic = heuristic
          )

def test13(hpn = True, skeleton = False, hierarchical = False, heuristic=habbs):
    t = PlanTest('test13',  smallErrProbs,
                 objects=['table1', 'table2', 'objA', 'objB', 'objD',
                          'objE', 'objF', 'objG'])
    targetPose = (0.55, 0.25, 0.61, 0.0)
    targetPoseB = (0.55, -0.2, 0.61, 0.0)
    targetVar = (0.001, 0.001, 0.001, 0.005)  # should be this

    goalConf = makeConf(t.world.robot, 0.0, 1.0, 0.0, up=True)
    confDeltas = (0.05, 0.05, 0.05, 0.05)

    goalProb = 0.1

    if skeleton:
        raw_input('skeleton wrong for this problem')

    goal = State([\
                  Bd([SupportFace(['objA']), 4, goalProb], True),
                  B([Pose(['objA', 4]),
                     targetPose, targetVar, (0.001,)*4,
                     goalProb], True)])
    t.run(goal,
          hpn = hpn,
          operators=['move', 'pick', 'place', 'lookAt', 'poseAchCanReach',
                     'poseAchCanSee', 'lookAtHand'],
          skeleton = [['place', 'move', 'pick', 'move', 'poseAchCanReach',
                       'place', 'move', 'pick', 'move']] if skeleton else None,
          hierarchical = hierarchical,
          regions=['table1Top', 'table2Top'],
          heuristic = heuristic
          )
    
def test14(hpn = True, skeleton = False, hierarchical = False, heuristic=habbs):
    # Move A so we can look at B
    # Example isn't really constructed right
    p1 = util.Pose(0.4, 0.0, 0.61, 0.0)
    p2 = util.Pose(0.8, 0.0, 0.61, 0.0)
    t = PlanTest('test14', smallErrProbs,
                 objects=['table1', 'objA', 'objB', 'table2',
                          'cupboardSide1', 'cupboardSide2'],
                 movePoses={'objA': p1,
                            'objB': p2},
                 varDict = {'objA': (0.0001*2,)*4,
                            'objB': (0.005*2,)*4})

    goalProb = 0.8

    goal = State([ Bd([SupportFace(['objB']), 4, goalProb], True),
                   B([Pose(['objB', 4]), p2.xyztTuple(),
                     (0.0001, 0.0001, 0.0001, 0.001),
                          (0.01,)*4, goalProb], True)])

    t.run(goal,
          hpn = hpn,
          operators=['move', 'pick', 'place', 'lookAt', 'poseAchCanReach',
                     'poseAchCanSee', 'lookAtHand'],
          skeleton = [['lookAt', 'poseAchCanSee', 'move',
                       'place', 'move', 'pick', 'move']] \
                             if skeleton else None,
          hierarchical = hierarchical,
          regions=['table1Top'],
          heuristic = heuristic
          )


# Test look at hand
def test15(hpn = True, skeleton=False, hand='left', flip = False, gd = 0,
           heuristic=habbs):
    global moreGD
    if gd != 0: moreGD = True           # hack!
    t = PlanTest('test15', typicalErrProbs, objects=['table1', 'objA'])
    goalConf = makeConf(t.world.robot, 0.0, 1.0, 0.0, up=True)
    confDeltas = (0.05, 0.05, 0.05, 0.05)
    goal = State([Bd([Holding([hand]), 'objA', .6], True),
                  Bd([GraspFace(['objA', hand]), gd, .6], True),
                  B([Grasp(['objA', hand, gd]),
                     (0,-0.025,0,0), (0.0001, 0.0001, 0.0001, 0.0001),
                     (0.001,)*4, 0.6], True),
                  Conf([goalConf, confDeltas], True)])
    homeConf = makeConf(t.world.robot, -0.5, 0.0, math.pi, up=True) \
                         if flip else None
    t.run(goal,
          hpn = hpn,
          skeleton = [['move', 'lookAtHand', 'move', 'pick', 'move']] \
          if skeleton else None,
          heuristic = heuristic,
          operators=['move', 'pick', 'place', 'lookAt', 'poseAchCanReach',
                     'poseAchCanSee', 'lookAtHand'],
          home=homeConf
          )

# pick and place with more noise 
def test16(hpn = True, skeleton = False, hierarchical = False,
           heuristic = habbs):
    t = PlanTest('test16',  typicalErrProbs,
                 objects=['table1', 'objA', 'objB'],
                 varDict = {'objA': (0.005*2,)*4},
                 #varDict = {'objA': (0.01*2,)*4}
                 )

    targetPose = (0.55, 0.25, 0.61, 0.0)
    goalProb = 0.9
    targetVar = (0.0001, 0.0001, 0.0001, 0.0005)
    goal = State([Bd([SupportFace(['objA']), 4, .5], True),
                  B([Pose(['objA', 4]),
                     targetPose, targetVar, (0.05,)*4,
                     goalProb], True)])
    t.run(goal,
          hpn = hpn,
          skeleton = [['lookAt', 'move', 'place', 'move',
                       'pick', 'move', 'lookAt']]*5 \
                       if skeleton else None,
          operators=['move', 'pick', 'place', 'lookAt', 'poseAchCanReach',
                     'poseAchCanSee', 'lookAtHand'],
          hierarchical = hierarchical,
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
           heuristic = habbs):
    front = util.Pose(0.45, 0.0, 0.61, 0.0)
    back = util.Pose(0.6, 0.0, 0.61, 0.0)
    parking = util.Pose(0.45, 0.3, 0.61, 0.0)
    t = PlanTest('test17',  tinyErrProbs, 
                 objects=['table1', 'objA', 'objB'],
                 movePoses={'objA': back,
                            'objB': parking})

    #glob.monotonicFirst = False
    glob.rebindPenalty = 200

    skel = [[place.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
             'move',
             pick.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
             'move',
             'poseAchCanReach',
             place.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
             'move',
             'lookAtHand',
             'move',
             pick.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
             'move']]*5

    # Small var
    targetVar = (0.0001, 0.0001, 0.0001, 0.0005)
    targetDelta = (0.001, 0.001, 0.001, 0.005)
    # Increase this
    goalProb = 0.2
    
    goal = State([Bd([SupportFace(['objB']), 4, goalProb], True),
                  B([Pose(['objB', 4]),
                     back.xyztTuple(), targetVar, targetDelta,
                     goalProb], True)])

    t.run(goal,
          hpn = hpn,
          skeleton = skel if skeleton else None,
          operators=['move', 'pick', 'place', 'lookAt', 'poseAchCanReach',
                     'poseAchCanSee', 'lookAtHand'],
          heuristic = heuristic,
          hierarchical = hierarchical,
          regions=['table1Top']
          )


# 18.  A in parking, B in parking -> A in front, B in back  (ordering)
# Trivial with skeleton
def test18(hpn = True, skeleton = False, hierarchical = False,
           heuristic = habbs):
    front = util.Pose(0.45, 0.0, 0.61, 0.0)
    back = util.Pose(0.6, 0.0, 0.61, 0.0)
    parking1 = util.Pose(0.45, 0.3, 0.61, 0.0)
    parking2 = util.Pose(0.45, -0.3, 0.61, 0.0)
    t = PlanTest('test18',  tinyErrProbs, 
                 objects=['table1', 'objA', 'objB'],
                 movePoses={'objA': parking1,
                            'objB': parking2})

    #glob.monotonicFirst = False
    glob.rebindPenalty = 200

    skel = [[place.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
             'move',
             pick.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
             'move',
             place.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
             'move',
             pick.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
             'move']]*5

    # Small var
    targetVar = (0.0001, 0.0001, 0.0001, 0.0005)
    targetDelta = (0.001, 0.001, 0.001, 0.005)
    # Increase this
    goalProb = 0.2
    
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
          operators=['move', 'pick', 'place', 'lookAt', 'poseAchCanReach',
                     'poseAchCanSee', 'lookAtHand'],
          hierarchical = hierarchical,
          heuristic = heuristic,
          regions=['table1Top']
          )

# 19.  A in back, B in parking -> A in front, B in back (combination)    
def test19(hpn = True, skeleton = False, hierarchical = False,
           heuristic = habbs):
    front = util.Pose(0.45, 0.0, 0.61, 0.0)
    back = util.Pose(0.6, 0.0, 0.61, 0.0)
    parking1 = util.Pose(0.45, 0.3, 0.61, 0.0)
    parking2 = util.Pose(0.45, -0.3, 0.61, 0.0)
    t = PlanTest('test19',  tinyErrProbs, 
                 objects=['table1', 'objA', 'objB'],
                 movePoses={'objA': back,
                            'objB': parking2})

    #glob.monotonicFirst = False
    glob.rebindPenalty = 200

    skel = [[place.applyBindings({'Obj' : 'objA', 'Hand' : 'right'}),
             'move',
             place.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
             'move',
             pick.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
             'move',
             pick.applyBindings({'Obj' : 'objA', 'Hand' : 'right'}),
             'move']]*5

    # Small var
    targetVar = (0.0001, 0.0001, 0.0001, 0.0005)
    targetDelta = (0.001, 0.001, 0.001, 0.005)
    # Increase this
    goalProb = 0.2
    
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
          operators=['move', 'pick', 'place', 'lookAt', 'poseAchCanReach',
                     'poseAchCanSee', 'lookAtHand'],
          hierarchical = hierarchical,
          heuristic = heuristic,
          regions=['table1Top']
          )

# 20.  Swap!
def test20(hpn = True, skeleton = False, hierarchical = False,
           heuristic = habbs):
    front = util.Pose(0.45, 0.0, 0.61, 0.0)
    back = util.Pose(0.6, 0.0, 0.61, 0.0)
    parking1 = util.Pose(0.45, 0.3, 0.61, 0.0)
    parking2 = util.Pose(0.45, -0.3, 0.61, 0.0)
    t = PlanTest('test20',  tinyErrProbs, 
                 objects=['table1', 'objA', 'objB'],
                 movePoses={'objA': back,
                            'objB': front})

    glob.monotonicFirst = False
    glob.rebindPenalty = 200

    skel = [[place.applyBindings({'Obj' : 'objA', 'Hand' : 'right'}),
             'move',
             place.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
             'move',
             pick.applyBindings({'Obj' : 'objA', 'Hand' : 'right'}),
             'move',
             pick.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
             'move']]*5

    swapSkel = [[place.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
                 'move',
                 pick.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
                 'move',
                 place.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
                 'move',
                 pick.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
                 'move',
                 lookAt.applyBindings({'Obj' : 'objA'}),
                 'move',
                 place.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
                 'move',
                 pick.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
                 'move',
                 place.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
                 'move',
                 pick.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
                 'move']]*5

    # Small var
    targetVar = (0.0001, 0.0001, 0.0001, 0.0005)
    targetDelta = (0.001, 0.001, 0.001, 0.005)
    # Increase this
    goalProb = 0.2
    
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
          skeleton = swapSkel if skeleton else None,
          operators=['move', 'pick', 'place', 'lookAt', 'poseAchCanReach',
                     'poseAchCanSee', 'lookAtHand'],
          heuristic = heuristic,
          hierarchical = hierarchical,
          regions=['table1Top']
          )

# stack objects?
def testStack(hpn = True, skeleton = False, hierarchical = False,
           heuristic = habbs):
    p1 = util.Pose(0.45, 0.0, 0.61, 0.0)
    p2 = util.Pose(0.6, 0.0, 0.61, 0.0)
    p3 = util.Pose(0.45, 0.2, 0.61, 0.0)
    t = PlanTest('test18',  smallErrProbs, 
                 objects=['table1', 'objA', 'objB', 'objC'],
                 movePoses={'objA': p1,
                            'objB': p2,
                            'objC': p3})

    skel = [[place.applyBindings({'Obj' : 'objC', 'Hand' : 'left'}),
             'move',
             lookAtHand.applyBindings({'Obj' : 'objC'}),
             'move',
             pick.applyBindings({'Obj' : 'objC', 'Hand' : 'left'}),
             'move',
             place.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
             'move',
             lookAtHand.applyBindings({'Obj' : 'objA'}),
             'move',
             pick.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
             'move']]*5

    goal = State([Bd([In(['objA', 'objBTop']), True, .4], True),
                  Bd([In(['objC', 'objATop']), True, .4], True)])

    t.run(goal,
          hpn = hpn,
          skeleton = skel if skeleton else None,
          operators=['move', 'pick', 'place', 'lookAt', 'poseAchCanReach',
                     'poseAchCanSee', 'lookAtHand'],
          heuristic = heuristic,
          hierarchical = hierarchical,
          regions=['objATop', 'objBTop']
          )

# Empty hand
def test21(hpn = True, skeleton = False, hierarchical = False,
           heuristic = habbs):
    p1 = util.Pose(0.45, 0.0, 0.61, 0.0)
    p2 = util.Pose(0.45, 0.4, 0.61, 0.0)

    t = PlanTest('test21',  smallErrProbs, 
                 objects=['table1', 'objA', 'objB'],
                 movePoses={'objA': p1,
                            'objB': p2})

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

    fbch.inHeuristic = False
    if skeleton:
       fbch.dotSearchId = 0    # should make skeletons work without reload
    fbch.flatPlan = not hierarchical
    fbch.plannerGreedy = 0.7
    pr2Sim.simulateError = False
    for win in wm.windows:
       wm.getWindow(win).clear()
    t.buildBelief(home=None, regions=['table1Top'])

    # Change pbs so obj B is in the hand
    o = 'objB'
    h = 'left'
    gm = (0, -0.025, 0, 0)
    gv = (1e-4,)*4
    gd = (1e-4,)*4
    gf = 0
    t.bs.pbs.updateHeld(o, gf, PoseD(gm, gv), h, gd)
    t.bs.pbs.excludeObjs([o])
    t.bs.pbs.shadowWorld = None # force recompute
    t.bs.pbs.draw(0.9, 'W')
    t.bs.pbs.draw(0.9, 'Belief')
    
    # Initialize simulator
    t.realWorld = RealWorld(t.bs.pbs.getWorld(),
                                   t.domainProbs) # simulator
    t.realWorld.setRobotConf(t.bs.pbs.conf)
    for obj in t.objects:
        t.realWorld.setObjectPose(obj, t.bs.pbs.getPlaceB(obj).objFrame())

    attachedShape = t.bs.pbs.getRobot().attachedObj(t.bs.pbs.getShadowWorld(0.9), 'left')
    shape = t.bs.pbs.getWorld().getObjectShapeAtOrigin(o).applyLoc(attachedShape.origin())
    t.realWorld.robot.attach(shape, t.realWorld, h)
    robot = t.bs.pbs.getRobot()
    cart = robot.forwardKin(t.realWorld.robotConf)
    handPose = cart[robot.armChainNames['left']].compose(gripperTip)
    pose = shape.origin()
    t.realWorld.held['left'] = o
    t.realWorld.grasp['left'] = handPose.inverse().compose(pose)
    t.realWorld.delObjectState(o)    

    t.realWorld.draw('World')
    s = State([], details = t.bs)

    operators=['move', 'pick', 'place', 'lookAt', 'poseAchCanReach',
                     'poseAchCanSee', 'lookAtHand']
    skeleton = None #[[place, move]]

    HPN(s, goal2, 
         [t.operators[o] for o in operators],
         t.realWorld,
         hpnFileTag = t.name,
         skeleton = skeleton if skeleton else None,
         h = heuristic,
         verbose = False,
         fileTag = t.name if writeSearch else None)

def prof(test):
    import cProfile
    import pstats
    cProfile.run(test, 'prof')
    p = pstats.Stats('prof')
    p.sort_stats('cumulative').print_stats(50)
    p.sort_stats('cumulative').print_callers(50)
