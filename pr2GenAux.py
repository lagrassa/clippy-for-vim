import numpy as np
import math
import random
import util
import copy
import windowManager3D as wm
import shapes
from planGlobals import debugMsg, debugDraw, debug, pause, torsoZ
from miscUtil import argmax, isGround
from dist import UniformDist
from pr2Robot import CartConf, gripperFaceFrame
from pr2Util import PoseD, ObjGraspB, ObjPlaceB, Violations, shadowName, objectName
from fbch import getMatchingFluents
from belief import Bd, B
from pr2Fluents import CanReachHome, canReachHome, In, Pose
from transformations import rotation_matrix
from cspace import xyCI, CI

Ident = util.Transform(np.eye(4))            # identity transform

################
# Basic tests for pick and place
################

deltaThreshold = (0.01, 0.01, 0.01, 0.02)
def legalGrasp(bState, conf, hand, objGrasp, objPlace):
    # !! This should check for kinematic feasibility over a range of poses.
    of = objectGraspFrame(bState, objGrasp, objPlace)
    rf = robotGraspFrame(bState, conf, hand)
    result = of.withinDelta(rf, deltaThreshold)
    return result

def objectGraspFrame(bState, objGrasp, objPlace):
    # Find the robot wrist frame corresponding to the grasp at the placement
    objFrame = objPlace.objFrame()
    graspDesc = objGrasp.graspDesc[objGrasp.grasp.mode()]
    faceFrame = graspDesc.frame.compose(objGrasp.poseD.mode())
    centerFrame = faceFrame.compose(util.Pose(0,0,graspDesc.dz,0))
    graspFrame = objFrame.compose(centerFrame)
    # !! Rotates wrist frame to grasp face frame - defined in pr2Robot
    gT = gripperFaceFrame
    wristFrame = graspFrame.compose(gT.inverse())
    assert wristFrame.pose()

    if debug('objectGraspFrame'):
        print 'objGrasp', objGrasp
        print 'objPlace', objPlace
        print 'objFrame\n', objFrame.matrix
        print 'grasp faceFrame\n', faceFrame.matrix
        print 'centerFrame\n', centerFrame.matrix
        print 'graspFrame\n', graspFrame.matrix
        print 'object wristFrame\n', wristFrame.matrix

    return wristFrame

def robotGraspFrame(bState, conf, hand):
    robot = bState.getRobot()
    _, frames = robot.placement(conf, getShapes=[])
    wristFrame = frames[robot.wristFrameNames[hand]]
    if debug('robotGraspFrame'):
        print 'robot wristFrame\n', wristFrame
    return wristFrame

###################
# !! Need implementations of InPickApproach and InPlaceDeproach

def inPickApproach(*x): return True
def InPlaceDeproach(*x): return True

###################

# Pick conditions
# pick1. move home->pre with obj at pick pose
# pick2. move home->pick without obj
# pick3. move home->pick with obj in hand 

# Place conditions are equivalent
# place1. move home->pre with obj in hand
# place1. move home->place with hand empty
# place2. move home->pre with obj at place pose

def canPickPlaceTest(bState, preConf, pickConf, hand, objGrasp, objPlace, p):
    args = (preConf, pickConf, hand, objGrasp, objPlace, p, bState)
    if debug('canPickPlaceTest'):
        print zip(('preConf', 'pickConf', 'hand', 'objGrasp', 'objPlace', 'p', 'bState'),
                  args)
    if not legalGrasp(bState, pickConf, hand, objGrasp, objPlace):
        if debug('canPickPlaceTest'):
            print 'Grasp is not legal'
        return None
    # if preConf and not inPickApproach(bState, preConf, pickConf, hand, objGrasp, objPlace):
    #     if debug('canPickPlaceTest'):
    #         print 'Not a proper approach')
    #     return None
    violations = Violations()           # cumulative
    # 1.  Can move from home to pre holding nothing with object placed at pose
    if preConf:
        bState1 = bState.copy().updatePermObjPose(objPlace).updateHeldBel(None, hand)
        if debug('canPickPlaceTest'): print 'H->App, obj=pose (condition 1)'
        path, violations = canReachHome(bState1, preConf, p, violations)
        if not path:
            debugMsg('canPickPlaceTest', 'Failed H->App, obj=pose (condition 1)')
            return None
    # 2 - Can move from home to pre holding the object
    obj = objGrasp.obj
    bState2 = bState.copy().excludeObjs([obj]).updateHeldBel(objGrasp, hand)
    if debug('canPickPlaceTest'): print 'H->App, obj=held (condition 2)'
    path, violations = canReachHome(bState2, preConf, p, violations)
    if not path:
        debugMsg('canPickPlaceTest', 'Failed H->App, obj=held (condition 2)')
        return None
    # 3.  Can move from home to pick while obj is placed with zero variance
    oB = objPlace.modifyPoseD(var=4*(0.0,)) # ignore uncertainty
    oB.delta = 4*(0.0,)
    bState3 = bState.copy().updatePermObjPose(oB).updateHeldBel(None, hand)
    if debug('canPickPlaceTest'): print 'H->Target, obj placed (0 var) (condition 3)'
    path, violations = canReachHome(bState3, pickConf, p, violations)
    if not path:
        debugMsg('canPickPlaceTest', 'Failed H->Target (condition 3)')
        return None
    # 4.  Can move from home to pick while holding obj with zero grasp variance
    gB = objGrasp.modifyPoseD(var=4*(0.0,)) # ignore uncertainty
    gB.delta = 4*(0.0,)
    bState3 = bState.copy().excludeObjs([obj]).updateHeldBel(gB, hand)
    if debug('canPickPlaceTest'): print 'H->Target, holding obj (0 var) (condition 4)'
    path, violations = canReachHome(bState3, pickConf, p, violations)
    if not path:
        debugMsg('canPickPlaceTest', 'Failed H->Target (condition 4)')
        return None
    debugMsg('canPickPlaceTest', ('->', violations))
    return violations

################
## GENERATORS
################

memoizerBufferN = 5
class Memoizer:
    def __init__(self, name, generator, values = None, bufN = memoizerBufferN):
        self.name = name
        self.generator = generator               # shared
        self.values = values if values else [] # shared
        self.bufN = bufN                       # shared
        self.done = set([])             # not shared
    def __iter__(self):
        return self
    def copy(self):
        # shares the generator and values list, only index differs.
        new = Memoizer(self.name, self.generator, self.values, self.bufN)
        return new
    def next(self):
        dif = len(self.values) - len(self.done)
        # Fill up the buffer, if possible
        if dif < self.bufN:
            for i in range(self.bufN - dif):
                try:
                    val = self.generator.next()
                    self.values.append(val)
                    if val[1].weight() < 1.0: break
                except StopIteration:
                    break
        if len(self.values) > len(self.done):
            elegible = set(range(len(self.values))) - self.done
            # Find min weight index among elegible
            nextI = argmax(list(elegible), lambda i: -self.values[i][1].weight())
            self.done.add(nextI)
            chosen = self.values[nextI]
            debugMsg('Memoizer',
                     self.name,
                     ('weights', [self.values[i][1].weight() for i in elegible]),
                     ('chosen', chosen[1].weight()))
            # if chosen[1].weight() > 5:
            #    raw_input('Big weight - Ok?')
            return chosen
        else:
            raise StopIteration

# This needs generalization
approachBackoff = 0.10
zBackoff = approachBackoff
def findApproachConf(bState, obj, placeB, conf, hand, prob):
    robot = bState.getRobot()
    cart = conf.cartConf()
    wristFrame = cart[robot.armChainNames[hand]]
    wristFrameBack = wristFrame.compose(\
             util.Pose(-approachBackoff,0.,zBackoff,0.))
    cartBack = cart.set(robot.armChainNames[hand], wristFrameBack)
    confBack = robot.inverseKin(cartBack, conf = conf)
    if not None in confBack.values():
        return confBack
    else:
        return None

def potentialGraspConfGen(bState, placeB, graspB, conf, hand, prob, nMax=None):
    def ground(pose):
        params = list(pose.xyztTuple())
        params[2] = 0.0
        return util.Pose(*params)
    if conf:
        yield conf
        return
    robot = bState.getRobot()
    rm = bState.getRoadMap()
    wrist = objectGraspFrame(bState, graspB, placeB)
    if debug('potentialGraspConfs'):
        print 'wrist', wrist
        bState.draw(prob, 'W')
    count = 0
    tried = 0
    for basePose in robot.nuggetPoses[hand]:
        if nMax and count >= nMax: break
        tried += 1
        cart = CartConf({'pr2BaseFrame': ground(wrist.compose(basePose).pose()),
                         'pr2Torso':[torsoZ]}, robot)
        if hand == 'left':
            cart.conf['pr2LeftArmFrame'] = wrist 
            cart.conf['pr2LeftGripper'] = [0.08] # !! pick better value
        else:
            cart.conf['pr2RightArmFrame'] = wrist 
            cart.conf['pr2RightGripper'] = [0.08]
        conf = robot.inverseKin(cart, complain=debug('potentialGraspConfs'))
        if hand == 'left':
            conf.conf['pr2RightArm'] = bState.conf['pr2RightArm']
            conf.conf['pr2RightGripper'] = bState.conf['pr2RightGripper']
        else:
            conf.conf['pr2LeftArm'] = bState.conf['pr2LeftArm']
            conf.conf['pr2LeftGripper'] = bState.conf['pr2LeftGripper']
        if not None in conf.values():
            viol, _ = rm.confViolations(conf, bState, prob) # don't include attached...
            if viol and findApproachConf(bState, placeB.obj, placeB, conf, hand, prob):
                if debug('potentialGraspConfs'):
                    conf.draw('W','green')
                    debugMsg('potentialGraspConfs', ('->', conf.conf))
                count += 1
                yield conf
            else:
                if debug('potentialGraspConfs'): conf.draw('W','red')
        elif debug('potentialGraspConfs'):
                print conf.conf
                conf.conf['pr2LeftArm'] = bState.getShadowWorld(prob).robotConf['pr2LeftArm']
                conf.conf['pr2RightArm'] = bState.getShadowWorld(prob).robotConf['pr2RightArm']
                if not None in conf.values(): conf.draw('W','red')
    if debug('potentialGraspConfs'):
        print 'Tried', tried, 'Found', count, 'potential grasp confs'
    return

def potentialLookConfGen(rm, shape, maxDist):
    def testPoseInv(basePoseInv):
        relVerts = np.dot(basePoseInv.matrix, shape.vertices())
        dots = np.dot(visionPlanes, relVerts)
        return not np.any(np.any(dots < 0, axis=0))
    centerPoint = util.Point(np.resize(np.hstack([shape.center(), [1]]), (4,1)))
    # visionPlanes = np.array([[1.,1.,0.,0.], [-1.,1.,0.,0.]])
    visionPlanes = np.array([[1.,0.,0.,0.]])
    for node in rm.nodes:
        basePose = node.cartConf['pr2Base']
        dist = centerPoint.distance(basePose.point())
        if dist > maxDist:
            continue
        inv = basePose.inverse()
        if not testPoseInv(inv):
            # Rotate the base to face the center of the object
            center = inv.applyToPoint(centerPoint)
            angle = math.atan2(center.matrix[0,0], center.matrix[1,0])
            rotBasePose = basePose.compose(util.Pose(0,0,0,-angle))
            par = rotBasePose.pose().xyztTuple()
            rotConf = node.conf.set('pr2Base', (par[0], par[1], par[3]))
            if debug('potentialLookConfs'):
                print 'basePose', node.conf['pr2Base']
                print 'center', center
                print 'rotBasePose', rotConf['pr2Base']
            if testPoseInv(rotBasePose.inverse()):
                yield rotConf
        else:
            yield node.conf
            if debug('potentialLookConfs'):
                print 'node.conf', node.conf['pr2Base']
    return

def otherHand(hand):
    return 'left' if hand == 'right' else 'right'

ang = -math.pi/2
rotL = util.Transform(rotation_matrix(-math.pi/4, (1,0,0)))
def trL(p): return p.compose(rotL)
rotR = util.Transform(rotation_matrix(math.pi/4, (1,0,0)))
def trR(p): return p.compose(rotR)
lookPoses = {'left': [trL(x) for x in [util.Pose(0.5,0.08,1.0, ang),
                                       util.Pose(0.5,0.18,1.0, ang)]],
             'right': [trR(x) for x in [util.Pose(0.5,-0.08,1.0, -ang),
                                        util.Pose(0.5,-0.18,1.0, -ang)]]}
def potentialLookHandConfGen(bState, prob, hand):
    shWorld = bState.getShadowWorld(prob)
    robot = bState.conf.robot
    curCartConf = bState.conf.cartConf()
    chain = robot.armChainNames[hand]
    baseFrame = curCartConf['pr2Base']
    for pose in lookPoses[hand]:
        target = baseFrame.compose(pose)
        cartConf = curCartConf.set(chain, target)
        conf = robot.inverseKin(cartConf, conf=bState.conf)
        if all(v for v in conf.conf.values()):
            if debug('potentialLookHandConfs'):
                print 'lookPose\n', pose.matrix
                print 'target\n', target.matrix
                print 'conf', conf.conf
                print 'cart\n', cartConf[chain].matrix
            yield conf
    return

################
## SUPPORT FUNCTIONS
################

# returns lists of (poseB, graspB, conf, canfAppr)

# !! THIS IS NOT FINISHED!

def candidatePlaceH(bState, inCondsRev, graspB, reachObsts, hand, prob):

    assert False

    # REVERSE THE INCONDS -- because regression is in opposite order
    inConds = inCondsRev[::-1]
    debugMsg('candidatePGCC', ('inConds - reversed', inConds))
    objs = [obj for (obj,_,_,_) in inConds]
    objPB = [opb for (_,_,opb,_,_) in inConds]
    shWorld = bState.getShadowWorld(prob)
    # Shadows (at origin) for objects to be placed.
    ## !! Maybe give it a little cushion
    objShadows = [shWorld.world.getObjectShapesAtOrigin(o.name()) for o in shWorld.getShadowShapes()]
    newBS = bState.copy()
    newBS.excludeObjs(objs)
    shWorld = newBS.getShadowWorld(prob)
    poseObsts = [sh for sh in shWorld.getShadowShapes() if not sh.name() in shWorld.fixedObjects]
    fixedObst = [sh for sh in shWorld.getShadowShapes() if sh.name() in shWorld.fixedObjects]
    # inCond is (obj, regShape, placeB, prob)
    allObsts = [fixedObst] + poseObsts + reachObsts
    robObsts = [fixedObst] + poseObsts
    regShapes = [regShape for (_,regShape,_,_) in inConds]
    robot = shWorld.robot
    # 1. Find plausible grasps -- could skip and just use legalGrasps
    # 2. Find combinations of poses and grasps

################
## Drawing
################
    
def drawPoseConf(bState, placeB, conf, confAppr, prob, win, color = None):
    ws = bState.getShadowWorld(prob)
    ws.world.getObjectShapeAtOrigin(placeB.obj).applyLoc(placeB.objFrame()).draw(win, color=color)
    conf.draw(win, color=color)

def drawObjAndShadow(bState, placeB, prob, win, color = None):
    ws = bState.getShadowWorld(prob)
    obj = placeB.obj
    ws.world.getObjectShapeAtOrigin(obj).applyLoc(placeB.objFrame()).draw(win, color=color)
    if shadowName(obj) in ws.world.objects:
        ws.world.getObjectShapeAtOrigin(shadowName(obj)).applyLoc(placeB.objFrame()).draw(win, color=color)

################
## Looking at goal fluents
################
            
def getGoalInConds(goalConds, X=[]):
    # returns a list of [(obj, regShape, prob)]
    fbs = getMatchingFluents(goalConds,
                             Bd([In(['Obj', 'Reg']), 'Val', 'P'], True))
    return [(b['Obj'], b['Reg'], b['P']) \
            for (f, b) in fbs if isGround(b.values())]

def pathShape(path, prob, bState, name):
    assert isinstance(path, (list, tuple))
    attached = bState.getShadowWorld(prob).attached
    return shapes.Shape([c.placement(attached=attached) for c in path], None, name=name)

def pathObst(cs, lgb, rgb, cd, p, bState, name):
    newBS = bState.copy()
    newBS = newBS.updateFromGoalPoses(cd) if cd else newBS
    newBS.updateHeldBel(lgb, 'left')
    newBS.updateHeldBel(rgb, 'right')
    key = (cs, newBS, p)
    if key in bState.beliefContext.pathObstCache:
        return bState.beliefContext.pathObstCache[key]
    path,  viol = canReachHome(newBS, cs, p, Violations())
    if debug('pathObst'):
        newBS.draw(p, 'W')
        cs.draw('W', 'red', attached=newBS.getShadowWorld(p).attached)
        print 'lgb', lgb
        print 'rgb', rgb
        print 'condition', cd
    if not path:
        if debug('pathObst'):
            print 'pathObst', 'failed to find path to conf in red', (cs, p, newBS)
        ans = None
    else:
        ans = pathShape(path, p, newBS, name)
    bState.beliefContext.pathObstCache[key] = ans
    return ans

def getReachObsts(goalConds, bState):
    fbs = getMatchingFluents(goalConds,
                             Bd([CanReachHome(['C', 'H',
                                               'LO', 'LF', 'LGM', 'LGV', 'LGD',
                                               'RO', 'RF', 'RGM', 'RGV', 'RGD',
                                               'Cond']),
                                  True, 'P'], True))
    world = bState.getWorld()
    obsts = []
    index = 0
    for (f, b) in fbs:
        if not isGround(b.values()): continue
        if debug('getReachObsts'):
            print 'GRO', f
        ignoreObjects = set([])
        lo = b['LO']; ro = b['RO']
        if lo != 'none': ignoreObjects.add(lo)
        if ro != 'none' : ignoreObjects.add(ro)
        gB1 = ObjGraspB(lo, world.getGraspDesc(lo), b['LF'],
                        PoseD(b['LGM'], b['LGV']), delta=b['LGD'])
        gB2 = ObjGraspB(ro, world.getGraspDesc(ro), b['RF'],
                        PoseD(b['RGM'], b['RGV']), delta=b['RGD'])
        if b['H'] == 'left':
            gBL = gB1; gBR = gB2
        else:
            gBL = gB2; gBR = gB1
        obst = pathObst(b['C'], gBL, gBR, b['Cond'], b['P'], bState, name= 'reachObst%d'%index)
        index += 1
        if not obst:
            debugMsg('getReachObsts', ('path fail', f, b.values()))
            return None
        # Look at Poses in conditions; they are exceptions
        pfbs = getMatchingFluents(b['Cond'],
                                  B([Pose(['Obj', 'Face']), 'Mu', 'Var', 'Delta', 'P'], True))
        for (pf, pb) in pfbs:
            if isGround(pb.values()):
                ignoreObjects.add(pb['Obj'])
        obsts.append((ignoreObjects, obst))
    debugMsg('getReachObsts', ('->', len(obsts), 'obsts'))
    return obsts

def bboxRandomDraw(bb):
    pt = tuple([random.uniform(bb[0,i], bb[1,i]) for i in xrange(3)])
    return pt[:2]+(bb[0,2],)

# find bbox for CI_1(2), that is, displacements of bb1 that place it
# inside bb2.  Assumes that bb1 has origin at 0,0.
def bboxInterior(bb1, bb2):
    for j in xrange(3):
        di1 = bb1[1,j] - bb1[0,j]
        di2 = bb2[1,j] - bb2[0,j]
        if di1 > di2: return None
    return np.array([[bb2[i,j] - bb1[i,j] for j in range(3)] for i in range(2)])

# !! Should pick relevant orientations... or more samples.
angleDist = UniformDist([-math.pi/2. -math.pi/4., 0.0, math.pi/4, math.pi/2])
def potentialRegionPoseGen(bState, obj, placeB, prob, regShapes, reachObsts, maxPoses = 30):
    def genPose(rs, angle):
        for i in xrange(5):
            (x,y,z) = bboxRandomDraw(bI.bbox())
            # Support pose, we assume that sh is on support face
            pose = util.Pose(x,y,z + clearance, angle)
            sh = shRot.applyTrans(pose)
            if debug('potentialRegionPoseGen'):
                sh.draw('W', 'brown')
                wm.getWindow('W').update()
            if all([rs.containsPt(p) for p in sh.vertices().T]) and \
               all(not sh.collides(obst) for (ig, obst) in reachObsts if obj not in ig):
                return pose
    clearance = 0.01
    for rs in regShapes: rs.draw('W', 'purple')
    ff = placeB.faceFrames[placeB.support.mode()]
    objShadow = bState.objShadow(obj, True, prob, placeB, ff)
    count = 0
    for rs in regShapes:
        for angle in (angleDist.draw() for i in xrange(maxPoses)):
            shRot =  objShadow.applyTrans(util.Pose(0,0,0,angle))
            bI = CI(objShadow.prim(), rs.prim())
            if bI == None:
                if debug('potentialRegionPoseGen'):
                    print 'bI is None'
                continue
            elif debug('potentialRegionPoseGen'):
                bI.draw('W', 'cyan')
                debugMsg('potentialRegionPoseGen', 'Region in cyan')
            pose = genPose(rs, angle)
            if pose:
                count += 1
                yield pose
    if debug('potentialRegionPoseGen'):
        print 'Returned', count, 'for regions', [r.name() for r in regShapes]
    return

def baseDist(c1, c2):
    (x1,y1,_) = c1['pr2Base']
    (x2,y2,_) = c2['pr2Base']
    return ((x2-x1)**2 + (y2-y1)**2)**0.5

    
