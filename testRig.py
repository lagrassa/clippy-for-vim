import time
import string

import windowManager3D as wm
from geom import bboxGrow
from objects import World

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
from belief import hAddBackBSet, BBhAddBackBSet, B, Bd

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
# lookAtHand
from pr2Ops import move, push, pick, place, lookAt, achCanReach, \
   achCanPickPlace, achCanPush, \
   hRegrasp, poseAchIn, moveNB, bLoc1, bLoc2, bLoc3

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

writeSearch = True

######################################################################
# Right Arm??
######################################################################

useCartesian = False
useLookAtHand = False

# DEBUG
useRight = True                        # DEBUG
useVertical = True                     # DEBUG
useHorizontal = True

if useROS:
    useRight = True
    useVertical = True
    useHorizontal = True

######################################################################
# Test Rig
######################################################################

# Counts unsatisfied fluent groups
# noinspection PyUnusedLocal
def hEasy(s, g, ops, ancestors):
    return g.easyH(s, defaultFluentCost = 1.5)

hDepth = 10

heuristicTime = 0.0

def habbs(s, g, ops, ancestors):
    global heuristicTime
    startTime = time.time()
    hops = ops + [hRegrasp]
    h = BBhAddBackBSet if debug('useBBinH') else hAddBackBSet
    val = h(s, g, hops, ancestors, ddPartitionFn = partition,
                         maxK = hDepth)
    if val == 0:
        # Just in case the addBack heuristic thinks we're at 0 when
        # the goal is not yet satisfied.
        isSat = s.satisfies(g)
        if not isSat:
            fbch.hCacheReset()
            newVal = h(s, g, hops, ancestors,
                                        ddPartitionFn = partition,
                                        maxK = hDepth)
            if newVal >= 0:
                tr('heuristic0', 'Heuristic cache inconsistent.'
                   'Leslie should fix this!')
                return newVal
            tr('heuristic0', 'habbs is 0 but goal not sat',
               [thing for thing in g.fluents if not thing.isGround() or \
                                            s.fluentValue(thing)==False])
            easyVal = hEasy(s, g, ops, ancestors)
            tr('heuristic0', '*** returning easyVal', easyVal)
            return easyVal
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


def testWorld(include = ('objA', 'objB', 'objC')):
    ((x0, y0, _), (x1, y1, dz)) = workspace
    w = 0.1
    wm.makeWindow('W', viewPort, 600)   # was 800
    if useROS: wm.makeWindow('MAP', viewPort)
    # noinspection PyShadowingNames
    def hor((x0, x1), y, w):
        return Ba([(x0, y-w/2, 0), (x1, y+w/2.0, dz)])

    # noinspection PyShadowingNames
    def ver(x, (y0, y1), w, extendSingleSide=False):
        if not extendSingleSide:
            return Ba([(x-w/2., y0, 0), (x+w/2.0, y1, dz)])
        return Ba([(x-w, y0, 0.), (x, y1, dz)])

    # noinspection PyShadowingNames
    def place((x0, x1), (y0, y1), (z0, z1)):
        return Ba([(x0, y0, z0), (x1, y1, z1)])

    # noinspection PyShadowingNames,PyUnusedLocal
    def placeSc((x0, x1), (y0, y1), (z0, z1)):
        return shapes.BoxScale(x1 - x0, y1 - y0, z1 - z0,
                               hu.Pose(0.0, 00.0,-0.5*(z1-z0), 0.0), 0.5)
    
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
    table1 = makeLegTable(name = 'table1', color='brown')
    if 'table1' in include: world.addObjectShape(table1)
    table2 = makeSolidTable(name='table2', color='brown')
    if 'table2' in include: world.addObjectShape(table2)
    table3 = makeSolidTable(name='table3', color='brown')
    if 'table3' in include: world.addObjectShape(table3)

    for i in range(1,4):
        name = 'table%d'%i
        if name in include:
            bbox = world.getObjectShapeAtOrigin(name).bbox()
            regName = name+'Top'
            tr('rig', ('Region', regName), bbox)
            world.addObjectRegion(name, regName, Sh([Ba(bbox)], name=regName),
                                  hu.Pose(0.0,0.0,2.0*bbox[1,2],0.0))
            bboxLeft = np.empty_like(bbox); bboxLeft[:] = bbox
            bboxLeft[0][0] = 0.5*(bbox[0][0] + bbox[1][0]) + 0.2
            regName = name+'Left'
            tr('rig', ('Region', regName), bboxLeft)
            world.addObjectRegion(name, regName, Sh([Ba(bboxLeft)], name=regName),
                                  hu.Pose(0.0,0.0,2.0*bbox[1,2],0.0))
            bboxLeftFront = np.empty_like(bbox); bboxLeftFront[:] = bbox
            bboxLeftFront[0][0] = 0.5*(bbox[0][0] + bbox[1][0]) + 0.2
            bboxLeftFront[0][1] = bboxLeftFront[1][1] - 0.2
            regName = name+'LeftFront'
            tr('rig', ('Region', regName), bboxLeftFront)
            world.addObjectRegion(name, regName, Sh([Ba(bboxLeftFront)], name=regName),
                                  hu.Pose(0.0,0.0,2.0*bbox[1,2],0.0))
            #bboxRight = np.empty_like(bbox)
            # TODO: This is what was here.  Could it be right?  Aliasing?
            bboxRight = bbox
            bboxRight[1][0] = 0.5*(bbox[0][0] + bbox[1][0]) - 0.2
            regName = name+'Right'
            tr('rig', ('Region', regName), bboxRight)
            world.addObjectRegion(name, regName, Sh([Ba(bboxRight)], name=regName),
                                  hu.Pose(0.0,0.0,2.0*bbox[1,2],0.0))

    # Some handy regions on table 1
    if 'table1' in include:
        bbox = world.getObjectShapeAtOrigin('table1').bbox()
        mfbbox = np.empty_like(bbox); mfbbox[:] = bbox
        mfbbox[0][0] = 0.4 * bbox[0][0] + 0.6 * bbox[1][0]
        mfbbox[1][0] = 0.6 * bbox[0][0] + 0.4 * bbox[1][0]
        mfbbox[1][1] = 0.5*(bbox[0][1] + bbox[1][1])
        world.addObjectRegion('table1', 'table1MidRear', 
                               Sh([Ba(mfbbox)], name='table1MidRear'),
                                      hu.Pose(0.0,0.0,2.0*bbox[1,2],0.0))
        mrbbox = np.empty_like(bbox); mrbbox[:] = bbox
        mrbbox[0][0] = 0.4 * bbox[0][0] + 0.6 * bbox[1][0]
        mrbbox[1][0] = 0.6 * bbox[0][0] + 0.4 * bbox[1][0]
        mrbbox[0][1] = 0.5*(bbox[0][1] + bbox[1][1])
        world.addObjectRegion('table1', 'table1MidFront', 
                               Sh([Ba(mrbbox)], name='table1MidFront'),
                                      hu.Pose(0.0,0.0,2.0*bbox[1,2],0.0))
    # Other permanent objects
    cupboard1 = Sh([place((-0.25, 0.25), (-0.05, 0.05), (0.0, 0.4))],
                     name = 'cupboardSide1', color='brown')
    cupboard2 = Sh([place((-0.25, 0.25), (-0.05, 0.06), (0.0, 0.4))],
                     name = 'cupboardSide2', color='brown')
    if 'cupboardSide1' in include: world.addObjectShape(cupboard1)
    if 'cupboardSide2' in include: world.addObjectShape(cupboard2)
    (coolShelves, aboveCoolShelves) = makeCoolShelves(name='coolShelves')
    if 'coolShelves' in include: world.addObjectShape(coolShelves)
    for (reg, pose) in aboveCoolShelves:
        world.addObjectRegion('coolShelves', reg.name(), reg, pose)

    # Some objects to grasp
    manipulanda = [oname for oname in include if oname[0:3] in ('obj', 'big')]

    colors = ['red', 'green', 'blue', 'cyan', 'purple', 'pink', 'orange']
    for i, objName in enumerate(manipulanda):
        if objName[0:3] == 'obj':
            thing = makeSoda(name = objName, color=colors[i%len(colors)])
        else:
            thing = makeBig(name = objName, color=colors[i%len(colors)])
        height = thing.bbox()[1,2]
        world.addObjectShape(thing)
        # The bbox has been centered
        extraHeight = 1.5*height+0.01
        bbox = bboxGrow(thing.bbox(), np.array([0.075, 0.075, extraHeight]))
        regName = objName+'Top'
        tr('rig', ('Region', regName), bbox)
        world.addObjectRegion(objName, regName, Sh([Ba(bbox)], name=regName),
                              hu.Pose(0.0,0.0,2.0* height +extraHeight,0.0))

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
        # TODO: derive these grasps from the shape of the object(s)
        if useHorizontal:             # horizontal
            world.graspDesc[obj].extend([GDesc(obj, hu.Transform(gMat0),
                                               0.05, 0.05, 0.025),
                                         GDesc(obj, hu.Transform(gMat1),
                                               0.05, 0.05, 0.025)])
        if useVertical:    # vertical
            world.graspDesc[obj].extend([GDesc(obj, hu.Transform(gMat3),
                                               0.05, 0.05, 0.025),
                                         GDesc(obj, hu.Transform(gMat2),
                                               0.05, 0.05, 0.025)])

    # noinspection PyShadowingNames
    def t(o):
        if o[0:3] == 'obj': return 'soda'
        if o[0:3] == 'big': return 'big'
        if o[0:5] == 'table': return 'table'
        if o[0:7] == 'shelves': return 'shelves'
        if o[0:4] == 'cool': return 'coolShelves'
        return 'unknown'

    world.objectTypes = dict([(o, t(o)) for o in include])
    world.symmetries = {'soda' : ({4 : 4}, {4 : [hu.Pose(0.,0.,0.,0.),
                                                 hu.Pose(0.,0.,0.,math.pi)]}),
                        'big' : ({4 : 4}, {4 : [hu.Pose(0.,0.,0.,0.),
                                                 hu.Pose(0.,0.,0.,math.pi)]}),
                        'table' : ({4 : 4}, {4 : [hu.Pose(0.,0.,0.,0.),
                                                 hu.Pose(0.,0.,0.,math.pi)]}),
                        'shelves' : ({4 : 4}, {4 : [hu.Pose(0.,0.,0.,0.)]}),
                        'coolShelves' : ({4 : 4}, {4 : [hu.Pose(0.,0.,0.,0.)]})}

    # Used by ICP - not in use now.
    # world.typePointClouds = {t:shapes.readOff('meshes/%s_points.off'%t, name=t) \
    #                          for t in set(world.objectTypes.values())}

    # The planning robot is a bit fatter, except in the hands...  This
    # is to provide some added tolerance for modeling and execution
    # uncertainty.
    robot = PR2('MM', makePr2ChainsShadow('PR2', world.workspace, radiusVar=0.01))
    thinRobot = PR2('MM', makePr2Chains('PR2', world.workspace))
    # This affects randomConf and stepAlongLine, unless overriden
    for r in (robot, thinRobot):
        if useRight:
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
        if useRight:
            c.conf['pr2RightGripper'] = [g]
        return c
    elif (not vertical) and standardHorizontalConf:
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
            h = hu.Transform(p=np.array([[a] for a in [ 0.4+dx, 0.3+dy,  1.1+dt, 1.]]), q=q)
            cart = cart.set('pr2LeftArm', base.compose(h))
            if useRight:
                hr = hu.Transform(p=np.array([[a] for a in [ 0.4+dx, -(0.3+dy),  1.1+dt, 1.]]), q=q)
                cart = cart.set('pr2RightArm', base.compose(hr))
        else:
            # h = hu.Pose(0.3+dx,0.33+dy,0.9+dz+dt,math.pi/4)
            h = hu.Pose(0.3+dx,0.5+dy,0.9+dz+dt,math.pi/4)
            cart = cart.set('pr2LeftArm', base.compose(h))
            if useRight:
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

def makeTable(dx, dy, dz, name, width = 0.1, color = 'orange'):
    #legInset = 0.02
    #legOver = 0.02
    return Sh([
        Ba([(-dx, -dy, dz-width), (dx, dy, dz)],
           name=name+ 'top', color=color),
        Ba([(-dx,      -dy, 0.0),
            (-dx+width, dy, dz-width)],
           name=name+' leg 1', color=color),
        Ba([(dx-width, -dy, 0.0),
            (dx,       dy, dz-width)],
           name=name+' leg 2', color=color)
        ], name = name, color=color)

eps = 0.01
epsz = 0.02
shelfDepth = 0.3
shelfWidth = 0.02
# makeShelves(shelfDepth/2.0, 0.305, 0.45, width=0.02, nshelf=2)

def makeShelves(dx=shelfDepth/2.0, dy=0.305, dz=0.45,
                width = shelfWidth, nshelf = 2,
                name='shelves', color='brown'):
    sidePrims = [
        Ba([(-dx, -dy-width, 0), (dx, -dy, dz)],
           name=name+'_side_A', color=color),
        Ba([(-dx, dy, 0), (dx, dy+width, dz)],
           name=name+'_side_B', color=color),
        Ba([(dx, -dy, 0), (dx+width, dy, dz)],
           name=name+'_backside', color=color),
        ]
    shelfSpaces = []
    shelfRungs = []
    for i in xrange(nshelf+1):
        frac = i/float(nshelf)
        bot = dz*frac
        #top = dz*frac+width
        shelf = Ba([(-dx, -dy-width, bot),
                    (dx, dy+width, bot+width)],
                   color=color,
                   name=name+'_shelf_'+string.ascii_uppercase[i])
        shelfRungs.append(shelf)
        spaceName = name+'_space_'+str(i+1)
        space = Ba([(-dx+eps, -dy-width+eps, eps),
                    (dx-eps, dy+width-eps, (dz/nshelf) - width - eps)],
                   color='green', name=spaceName)
        space = Sh([space], name=spaceName, color='green')
        shelfSpaces.append((space, hu.Pose(0.0,0.0,bot+eps-(dz/2.0),0.0)))
    shelves = Sh(sidePrims + shelfRungs, name = name+'Body', color=color)
    return shelves, shelfSpaces

def makeTableShelves(dx=shelfDepth/2.0, dy=0.305, dz=0.45,
                width = shelfWidth, nshelf = 2,
                name='tableShelves', color='brown'):
    coolerPose = hu.Pose(0.0, 0.0, tZ, -math.pi/2.0)
    shelvesPose = hu.Pose(0.0, 0.0, tZ+coolerZ, -math.pi/2.0)
    tH = 0.67                           # table height
    cooler = Sh([Ba([(-0.12, -0.165, 0), (0.12, 0.165, coolerZ)],
                    name='cooler', color=color)],
                name='cooler', color=color)
    table = makeTable(0.603, 0.298, tH, name = 'table1', color=color)
    shelves, shelfSpaces = makeShelves(dx, dy, dz, width, nshelf, name='tableShelves', color=color)
    offset = shelvesPose.compose(hu.Pose(0.,0.,-(tH/2+coolerZ/2),0.))
    shelfSpaces = [(s,pose.compose(offset)) for (s, pose) in shelfSpaces]
    obj = Sh( shelves.applyTrans(shelvesPose).parts() \
              + cooler.applyTrans(coolerPose).parts() \
              + table.parts(),
              name=name, color=color)
    return obj, shelfSpaces

def makeCoolShelves(dx=shelfDepth/2.0, dy=0.305, dz=0.45,
                      width = shelfWidth, nshelf = 2,
                      name='coolShelves', color='brown'):
    coolerPose = hu.Pose(0.0, 0.0, 0.0, -math.pi/2.0)
    shelvesPose = hu.Pose(0.0, 0.0, coolerZ, -math.pi/2.0)
    cooler = Sh([Ba([(-0.12, -0.165, 0), (0.12, 0.165, coolerZ)],
                    name='cooler', color=color)],
                name='cooler', color=color)
    shelves, shelfSpaces = makeShelves(dx, dy, dz, width, nshelf, name='coolShelves', color=color)
    offset = shelvesPose.compose(hu.Pose(0.,0.,-(coolerZ/2),0.))
    shelfSpaces = [(s,pose.compose(offset)) for (s, pose) in shelfSpaces]
    obj = Sh( shelves.applyTrans(shelvesPose).parts() \
              + cooler.applyTrans(coolerPose).parts(),
              name=name, color=color)
    return obj, shelfSpaces

initConfs = []

class PlanTest:
    def __init__(self, name, domainProbs, operators,
                 objects = ('table1','objA'), fixPoses = None,
                 movePoses = None, held = None, grasp = None,
                 multiplier = 6, var = 1.0e-5, varDict = None):   # var was 10e-10
        self.grasp = grasp
        self.held = held
        self.name = name
        self.multiplier = multiplier
        self.objects = objects          # list of objects to consider
        self.domainProbs = domainProbs
        self.world, self.thinRobot = testWorld(include=self.objects)
        if not initConfs:
            ((x0, y0, _), (x1, y1, dz)) = workspace
            dx = x1 - x0; dy = y1 - y0
            count = 2*multiplier
            for x in range(count+1):
                for y in range(count+1):
                    # print (x0+x*dx/float(count), y0+y*dy/float(count))
                    for angle in [0, math.pi/2, -math.pi/2, math.pi]:
                        if useHorizontal:
                            initConfs.append(
                            makeConf(self.world.robot,
                                     x0 + x*dx/float(count),
                                     y0 + y*dy/float(count), angle)),
                        if useVertical:
                            initConfs.append(
                             makeConf(self.world.robot,
                                      x0+x*dx/float(count),
                                      y0+y*dy/float(count), angle, vertical=True))
        self.initConfs = initConfs
        var4 = (var, var, 1e-10, var)
        del0 = (0.0, 0.0, 0.0, 0.0)
        #del02 = (0.02, 0.02, 0.2, 0.04)
        del05 = (0.05, 0.05, 0.5, 0.05)
        # Make this bigger to keep the robot from coming right up
        # against obstacles

        ff = lambda o: self.world.getFaceFrames(o) if o in objects else []
        # The poses of the supporting face frames (the placement)
        fixObjPoses = {'table1':hu.Pose(1.1, 0.0, 0.0, math.pi/2.0),
                       'tableShelves':hu.Pose(1.1, 0.0, 0.0, math.pi/2.0),
                       'coolShelves':hu.Pose(1.1, 0.0, tZ, math.pi/2.0),
                       'table2': hu.Pose(1.0, -0.75, 0.0, 0.0),
                       'table3': hu.Pose(1.6,0.0,0.0,math.pi/2.0),
                       'cupboardSide1': hu.Pose(1.1, -0.2, 0.6, 0.0),
                       'cupboardSide2': hu.Pose(1.1, 0.2, 0.6, 0.0),
                       'cooler': hu.Pose(1.1, 0.0, tZ, 0.)}
        moveObjPoses = {'objA': hu.Pose(1.1, 0.0, tZ, 0.0),
                        'objB': hu.Pose(0.95, -0.4, tZ, 0.0),
                        'objC': hu.Pose(0.45, -1.2, tZ, 0.0),
                        'objD': hu.Pose(0.95, -0.2, tZ, 0.0),
                        'objE': hu.Pose(0.95, 0.0, tZ, 0.0),
                        'objF': hu.Pose(0.95, 0.2, tZ, 0.0),
                        'objG': hu.Pose(0.95, 0.4, tZ, 0.0),
                        'objH': hu.Pose(0.95, 0.6, tZ, 0.0),
                        'objI': hu.Pose(0.95, 0.8, tZ, 0.0)}
        if movePoses is not None:
            moveObjPoses.update(movePoses)           # input poses
        #print 'updated', moveObjPoses
        if fixPoses is not None:
            fixObjPoses.update(fixPoses)           # input poses
        #print 'updated', fixObjPoses
        self.fix = {}
        for name in fixObjPoses:
            if name in self.objects:
                oShape = self.world.getObjectShapeAtOrigin(name).applyLoc(fixObjPoses[name])
                supFace = supportFaceIndex(oShape)
                #print 'supportFace', name, supFace
                oVar = varDict[name] if (varDict and name in varDict) else var4
                self.fix[name] = ObjPlaceB(name, ff(name), DeltaDist(supFace),
                                  fixObjPoses[name], oVar, del0)
        self.move = {}
        for name in moveObjPoses.keys():
            if name in self.objects:
                oShape = self.world.getObjectShapeAtOrigin(name).applyLoc(moveObjPoses[name])
                supFace = supportFaceIndex(oShape)
                #print 'supportFace', name, supFace
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
                     params={'kNearest':17, # May be too low
                             'kdLeafSize':100,
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
        # TODO:  LPK Awful modularity
        bs.partitionFn = partition
        # noinspection PyAttributeOutsideInit
        self.bs = bs

    def run(self, goal, skeleton = None, hpn = True,
            home=None, regions = frozenset([]), hierarchical = False,
            heuristic = None,
            greedy = 0.75, simulateError = False,
            initBelief = None, initWorld=None,
            rip = False):
        randomizedInitialPoses = rip
        global heuristicTime
        glob.inHeuristic = False
        if skeleton: fbch.dotSearchId = 0
        startTime = time.clock()
        fbch.flatPlan = not hierarchical
        fbch.plannerGreedy = greedy  # somewhat greedy by default
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

            self.realWorld.setRobotConf(self.bs.pbs.conf)
            # LPK!! add collision checking
            heldLeft = self.bs.pbs.held['left'].mode()
            heldRight = self.bs.pbs.held['right'].mode()
            for obj in self.objects:
                if not obj in (heldLeft, heldRight):
                    pb = self.bs.pbs.getPlaceB(obj)
                    meanObjPose = pb.objFrame().pose()
                    if randomizedInitialPoses:
                        stDev = tuple([math.sqrt(v) for v in pb.poseD.variance()])
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
                    fileTag = self.name if writeSearch else None,
                    nonMonOps = ['Move', 'MoveNB', 'LookAt', 'Place'],
                    maxNodes = 100)
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

######################################################################
# Test Cases
######################################################################

# No Z error in observations for now!  Address this eventually.
# Turned down pickVar until we can look at hand

# Made odo error smaller...

typicalErrProbs = DomainProbs(
            # stdev, constant, assuming we control it by tracking while moving
            odoError = (0.008, 0.008, 1e-5, 0.015),
            # variance in observations; diagonal for now
            obsVar = (0.005**2, 0.005**2, 1e-5 **2, 0.01**2),
            # big angle var from robot experience
            # obsVar = (0.005**2, 0.005**2,0.005**2, 0.15**2),
            # get type of object wrong
            obsTypeErrProb = 0.05,
            # fail to pick or place in the way characterized by the Gaussian
            pickFailProb = 0.02,
            placeFailProb = 0.02,
            pushFailProb = 0.02,
            # variance in grasp after picking
            pickVar = (0.001**2, 0.001**2, 1e-11, 0.002**2),
            # variance in pose after placing
            placeVar = (0.001**2, 0.001**2, 1e-11, 0.002**2),
            pushVar = (0.01**2, 0.01**2, 1e-11, 0.02**2),
            # pickTolerance
            pickTolerance = (0.025, 0.025, 0.025, 0.1),
            maxGraspVar = (0.0051**2, .0051**2, .005**2, .015**2),
            moveConfDelta = (0.001, 0.001, 1e-6, 0.002),
            #shadowDelta = (0.01, 0.01, 1.0e-8, 0.05),
            #shadowDelta = (0.001, 0.001, 1e-11, 0.002),
            shadowDelta = (0.004, 0.004, 1e-6, 0.008),
            # Use this for placing objects
            # placeDelta = (0.005, 0.005, 1.0e-4, 0.01),
            # graspDelta = (0.001, 0.001, 1.0e-4, 0.002))
            placeDelta = (0.01, 0.01, 1.0e-4, 0.05),
            graspDelta = (0.005, 0.005, 1.0e-4, 0.008))

tinyErrProbs = DomainProbs(
            # stdev, constant, assuming we control it by tracking while moving
            odoError = (0.0001, 0.0001, 1e-11, 0.0001),
            # variance in observations; diagonal for now
            obsVar = (0.0001**2, 0.0001**2,0.0001**2, 0.0001**2),
            # get type of object wrong
            obsTypeErrProb = 0.0,
            # fail to pick or place in the way characterized by the Gaussian
            pickFailProb = 0.0,
            placeFailProb = 0.0,
            pushFailProb = 0.0,
            # variance in grasp after picking
            pickVar = (0.0001**2, 0.0001**2, 1e-11, 0.0001**2),
            # variance in pose after placing
            placeVar = (0.0001**2, 0.0001**2, 1e-11, 0.0001**2),
            pushVar = (0.0001**2, 0.0001**2, 1e-11, 0.0001**2),
            # pickTolerance
            pickTolerance = (0.025, 0.025, 0.025, 0.05),
            maxGraspVar = (0.005**2, .005**2, .005**2, .015**2),
            # Use this for placing objects
            placeDelta = (0.005, 0.005, 1.0e-4, 0.01),
            graspDelta = (0.001, 0.001, 1.0e-4, 0.002))

allOperators = [move, # pick, place,
                push,
                lookAt, moveNB,
                achCanReach, achCanPickPlace, poseAchIn,
                bLoc1, bLoc2, bLoc3]
              #lookAtHand    #graspAchCanPickPlace #dropAchCanPickPlace achCanSee
