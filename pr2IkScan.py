import pdb
import math
import util
from ranges import realRange, angleRange
import transformations as transf
import windowManager3D as wm
import numpy as np
import shapes
import objects
from objects import WorldState, World
import pr2Robot
reload(pr2Robot)
from pr2Robot import makePr2Chains, PR2, JointConf, CartConf, pr2Init, \
     gripperToolOffset
import planGlobals as glob


############################################################
# IK
############################################################

# Build cartesian workspace "roadmap".  Starting from some seed cartesian
# configuration, generate a set of cartesian neighbors, connect to those that
# have a kinematic solution that is sufficiently close to the base
# configuration.

# There's a symmetry around the shoulder rotation (about z), we could try to
# avoid exploring cartesian configurations that are obtained by rotation about
# that axis.

class ScanNode:
    def __init__(self, index, angleConfs, wristFrames, node):
        self.index = index
        self.parent = node
        self.angleConfs = angleConfs
        self.wristFrames = wristFrames
        self.children = []
        self.childrenByDir = 6*[None]
    def pose(self, i):
        return node.wristFrames[i].pose()

# 4 DOF search, exploits the commutativity.

xyz_incr = 0.1                          # too big
theta_divisor = 3                        # too big
theta_incr = math.pi/theta_divisor
PI2 = 2*math.pi
Ident = util.Transform(np.eye(4))            # identity transform

def displacement(index):
    return util.Pose(index[0]*xyz_incr, index[1]*xyz_incr, index[2]*xyz_incr,
                     index[3]*theta_incr)

indexOffsets = [(-1,0,0,0),(1,0,0,0),
                (0,-1,0,0),(0,1,0,0),
                (0,0,-1,0),(0,0,1,0)]

indexOffsetsNoZ = [(-1,0,0,0),(1,0,0,0),
                   (0,-1,0,0),(0,1,0,0)]

# Do the displacements in base coordinates.
def scanCartesianKin(robot, wstate, initWrist, chainName,
                     offsets = indexOffsets,
                     rotations = [Ident],
                     display = False):
    def angleCost(aL, bL):
        return sum([abs(util.angleDiff(a,b)) for (a,b) in zip(aL, bL)])

    def scanAngles(i):
        if display:
            wm.getWindow('Lab').clear()
        baseConf = None
        angleConfs = nrots*[None]
        wristFrames = nrots*[None]
        count = 0
        for r, rotation in enumerate(rotations):
            h = displacement(i).compose(handTrans).compose(rotation).compose(handRot).compose(toolOffsetInv)
            c = init.set(chainFrameName, h)
            j = robot.inverseKin(c, wstate,
                                 complain = False, fail = False)
            if j.get(chainName):
                place = robot.placement(j)[0]
                collide = [x for x in place.parts() if x.name()==gripName][0].collides(baseObj)
                if collide:
                    if display: print 'Collision'
                else:
                    count += 1
                    angleConfs[r] = j
                    wristFrames[r] = h
                    if not baseConf: baseConf = j
            else:
                continue
            if display:
                place.draw('Lab', 'cyan')
                p.applyTrans(c.get(chainFrameName)).draw('Lab')
        print 'rotations=', count
        return baseConf, angleConfs, wristFrames

    chainFrameName = chainName+'Frame'
    gripName = chainName[:-3]+'Gripper' # remove Arm from chainName
    init = robot.forwardKin(robot.nominalConf).copy()
    init.conf[chainFrameName] = initWrist
    initHand = initWrist.compose(gripperToolOffset)
    nodeConf = robot.inverseKin(init, wstate, complain = False, fail = False)
    assert nodeConf[chainName]
    baseObj = robot.placement(nodeConf)[0].parts()[0]
    assert baseObj.name()=='pr2Base'
    startNode = ScanNode((0,0,0,0), [], [], None)
    agenda = [startNode]
    done = {}
    p = shapes.BoxAligned(np.array([(-0.02,-0.01,-0.01),(0.02,0.01,0.01)]), None)
    toolOffsetInv = gripperToolOffset.inverse()
    handTrans = util.Transform(transf.translation_matrix(initHand.point().xyzTuple()))
    handRot = handTrans.inverse().compose(initHand)
    nrots = len(rotations)
    while agenda:
        node = agenda.pop(0)
        if node.index in done or abs(node.index[3]) >= theta_divisor:
            continue
        else: done[node.index] = node
        if not node.angleConfs:
            baseConf, angleConfs, wristFrames = scanAngles(node.index)
            node.angleConfs = angleConfs
            node.wristFrames = wristFrames
        nodeConf = [x for x in node.angleConfs if x][0]
        wstate.setRobotConf(nodeConf) # robot is in wstate
        for k, off in enumerate(offsets):                      # x, y, z offsets
            i = tuple([a+b for (a,b) in zip(node.index, off)]) # new index
            if i in done:
                node.children.append(done[i])
                node.childrenByDir[k] = done[i]
                continue
            baseConf, angleConfs, wristFrames = scanAngles(i)
            if baseConf:
                if display:
                    robot.placement(baseConf)[0].draw('Lab', 'red')
                    print i
                    wm.getWindow('Lab').update()
                    # raw_input('Next')
                newNode = ScanNode(i, angleConfs, wristFrames, node)
                node.children.append(newNode)
                node.childrenByDir[k] = newNode
                agenda.append(newNode) # breadth-first
    return startNode

def handRotations(downRange, roundRange, stepSize):
    downLo, downHi = downRange
    roundLo, roundHi = roundRange
    rots = []
    nDown = int(math.floor((downHi - downLo)/stepSize))+1
    nRound = int(math.floor((roundHi - roundLo)/stepSize))+1
    for d in range(nDown):
        downAngle = downLo + d * stepSize
        for r in range(nRound):
            roundAngle = roundLo + r * stepSize
            rot = np.dot(transf.rotation_matrix(downAngle, (0,1,0)),
                         transf.rotation_matrix(roundAngle, (0,0,1)))
            rots.append(util.Transform(rot))
    return rots

# handRotations((0, math.pi/2),(-math.pi, 0), math.pi/8)

def zRotations(n=8):
    return [util.Pose(0,0,0,PI2*k/n) for k in range(n)]

def scanNodeDistances(sn, chainName, world):
    robot = world.robot
    n = len(sn.angleConfs)
    dists = np.zeros((n,n))
    arm = 'l' if chainName=='pr2LeftArm' else 'r'
    for i in range(n):
        if not sn.angleConfs[i]: continue
        T = robot.forwardKin(sn.angleConfs[i], world)[chainName+'Frame'].compose(gripperToolOffset)
        for j in range(i+1, n):
            if not sn.angleConfs[j]: continue
            path = interpolate(sn.angleConfs[j], sn.angleConfs[i], world, chainName)
            dist = 0.
            for q in path:
                T1 = robot.forwardKin(q, world)[chainName+'Frame'].compose(gripperToolOffset)
                dist = max(dist, T.point().distance(T1.point()))
            dists[i,j] = dists[j,i] = dist
    return dists

def interpolate(q_f, q_i, world, chainName, stepSize=0.5):
    robot = world.robot
    path = [q_i]
    q = q_i
    while q != q_f:
        qn = robot.stepAlongLine(q_f, q, stepSize, moveChains=[chainName])
        if q == qn: break
        q = qn
        path.append(q)
    path.append(q_f)
    return path

def showMove(q_f, q_i, world, chainName, window='W'):
    path = interpolate(q_f, q_i, world, chainName)
    print 'Path length', len(path)
    colors = ['red','orange','gold', 'green', 'cyan', 'blue','purple','black']
    nc = len(colors)
    for i, q in enumerate(path):
        world.robot.placement(q, world)[0].draw(window, colors[i%nc])

def testScan(chainName, rotations=[Ident], indices=indexOffsets, display=False):
    colors = ['red','orange','gold', 'green', 'cyan', 'blue','purple','black']
    if display:
        viewPort =  [-1.75, 1.75, -1.75, 1.75, 0, 2]
        wm.makeWindow('Lab', viewPort)
    done = set([])
    def scanShow(p, node, robot, wstate, name):
        if node.index in done: return
        # print node.index
        done.add(node.index)
        for i, conf in enumerate(node.angleConfs):
            if not conf: continue
            tr = robot.forwardKin(conf, wstate).get(name).compose(gripperToolOffset)
            p.applyTrans(tr).draw(window, colors[i%len(colors)])
        for c in node.children: scanShow(p, c, robot, wstate, name)
    world = World()
    world.workspace = np.array([[-1.,-1.,0.],[1.,1.,0.]])
    robot = PR2('MM', makePr2Chains('PR2', world.workspace))
    robot.nominalConf = JointConf(pr2Init.copy(), robot)
    world.setRobot(robot)

    basePose = Ident
    window = 'Lab' if world else 'W'
    # To get a vertical hand (pointing down)
    off = util.Transform(transf.rotation_matrix(math.pi/2, (0, 1, 0)))
    # To get a horizontal hand
    # off = Ident
    ltarget = 0.5, 0.501, glob.torsoZ+0.550, 0.000
    rtarget = 0.5, -0.501, glob.torsoZ+0.550, 0.000
    poseL = util.Pose(*ltarget).compose(off)
    poseR = util.Pose(*rtarget).compose(off)
    print poseL.compose(gripperToolOffset)
    cart = CartConf({'pr2BaseFrame': basePose,
                     'pr2LeftArmFrame': basePose.compose(poseL),
                     'pr2RightArmFrame': basePose.compose(poseR)}, robot)
    conf = robot.inverseKin(cart, conf=robot.nominalConf, complain=True)
    if None in conf.values():
        raise Exception, 'Kinematics failed'
    cl(window)
    robot.nominalConf = conf.copy()
    robot.placement(conf)[0].draw(window, 'green')
    wm.getWindow(window).update()
    raw_input('Go?')
    wstate = WorldState(world)
    wstate.setRobotConf(robot.nominalConf)

    node = scanCartesianKin(robot, wstate,
                            cart[chainName],
                            chainName, indices,
                            rotations=rotations, display = display)

    raw_input('Finished scanning')
    p = shapes.BoxAligned(np.array([(-0.01,-0.01,-0.05),(0.01,0.01,0.0)]), None)
    
    scanShow(p, node, robot, wstate, chainName+'Frame')

    return wstate, node

def cl(window='W'):
    wm.getWindow(window).clear()


# Should specify the size of the neighborhood in the node graph that must agree
# for a nugget.  This version assumes the neighborhood size is 1.

def collectConfs(wstate, node, minLen = 6, display = False):
    visited=set([])
    return collectLoop(wstate, node, visited, minLen, display)

def collectLoop(wstate, node, visited, minLen, display):
    if node in visited: return []
    pt= shapes.BoxAligned(np.array([(-0.01,-0.01,-0.05),(0.01,0.01,0.0)]), None)
    visited.add(node)
    nuggets = []
    if len(node.children) >= minLen:
        nodes = [node] + node.children
        nConfs = len(node.angleConfs)
        for i in xrange(nConfs):
            if all([c.angleConfs[i] and c.angleConfs[(i-1)%nConfs] and c.angleConfs[(i+1)%nConfs] \
                    for c in nodes]) \
                    and len(nodeRun(node, indexOffsets[4])) >= 3 \
                    and len(nodeRun(node, indexOffsets[5])) >= 3:
                print 'Node', node.index, len(node.children)
                print '   nugget', i
                if display:
                    cl()
                    showMove(node.angleConfs[(i-1)%nConfs],
                             node.angleConfs[(i+1)%nConfs],
                             wstate, 'pr2LeftArm')
                    raw_input('Next?')
                    cl()
                    node.angleConfs[i].draw('W')
                    pt.applyTrans(node.wristFrames[i]).draw('W', 'red')
                    raw_input('Wrist?')
                # This is the wrist frame!
                nuggets.append(node.wristFrames[i])
                
    for n in node.children:
        nuggets.extend(collectLoop(wstate, n, visited, minLen, display))
    return nuggets

def nodeRun(node, offset):
    targetIndex = tuple([x+y for x,y in zip(node.index, offset)])
    for c in node.children:
        if c.index == targetIndex:
            return [c] + nodeRun(c, offset)
    else:
        return []

# wstate, node = testScan('pr2LeftArm', zRotations(), display=True)
# foo = collectConfs(wstate, node, display=False)
# This returns a list of (quat, point) -- 2 lists of 4 entries each

