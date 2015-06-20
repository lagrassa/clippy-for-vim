import numpy as np
import random
import math
import pdb
import copy
import time
import transformations as transf

import pointClouds as pc
import planGlobals as glob
from planGlobals import debug, debugMsg
import windowManager3D as wm
from pr2Util import shadowWidths, supportFaceIndex, bigAngleWarn, objectName
from pr2Visible import lookAtConf, findSupportTable, visible
import pr2Robot
reload(pr2Robot)
from pr2Robot import cartInterpolators, JointConf, CartConf
from pr2Ops import lookAtBProgress

import util
import tables
reload(tables)

twoPi = 2.0*math.pi

if glob.useROS:
    import rospy
    import roslib
    roslib.load_manifest('hpn_redux')
    import hpn_redux.msg
    import hpn_redux.srv
    from hpn_redux.srv import *
    import kinematics_msgs.msg
    import kinematics_msgs.srv
    from std_msgs.msg import Header
    from arm_navigation_msgs.msg import RobotState
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from cardboard.srv import *
    import geometry_msgs.msg as gm
    # rospy.init_node('hpn'); print 'Starting ROS'

## Communicate with ROS controller to move the robot, just unpack the configuration.

# operation could be:
# reset - treat current location as origin
# resetForce - recalibrate touch sensors
# move - move to configuration
# closeGuarded - use ROS guarded grip
# grab - use ROS grab
# close - as hard as you can
# open - to commanded width
# look - move head to look at target

headTurn = util.Transform(transf.rotation_matrix(-math.pi/2, (0,1,0)))

def pr2GoToConf(cnfIn,                  # could be partial...
                operation,              # a string
                arm = 'both',
                speedFactor = glob.speedFactor):
    if not glob.useROS: return None, None
    rospy.wait_for_service('pr2_goto_configuration')
    try:
        gotoConf = rospy.ServiceProxy('pr2_goto_configuration',
                                      hpn_redux.srv.GoToConf)
        conf = hpn_redux.msg.Conf()
        conf.arm = arm
        conf.base = map(float, cnfIn.get('pr2Base', []))
        conf.torso = map(float, cnfIn.get('pr2Torso', [])) 
        conf.left_joints = map(float, cnfIn.get('pr2LeftArm', []))
        conf.right_joints = map(float, cnfIn.get('pr2RightArm', []))

        if operation == 'grab':
            assert arm != 'both'

        if operation == 'open':
            conf.left_grip = map(float, cnfIn.get('pr2LeftGripper', []))
            conf.left_grip[0] = max(0., min(0.8, conf.left_grip[0]))
            conf.right_grip = map(float, cnfIn.get('pr2RightGripper', []))
            conf.right_grip[0] = max(0., min(0.8, conf.right_grip[0]))
            operation = 'move'
        else:
            conf.left_grip = []
            conf.right_grip = []
        if operation == 'look':
            assert cnfIn.get('pr2Head', None)
            gaze = gazeCoords(cnfIn)
            conf.head = map(float, gaze) # a look point relative to robot
            raw_input('Looking at %s'%gaze)
            operation = 'move'
        else:
            conf.head = []

        if debug('pr2GoToConf'): print operation, conf
        
        resp = gotoConf(operation, conf, speedFactor)

        if debug('pr2GoToConf'): print 'response', resp
        c = resp.resultConf

        cnfOut = {}
        cnfOut['pr2Base'] = c.base
        cnfOut['pr2Torso']  = c.torso
        cnfOut['pr2LeftArm'] = c.left_joints
        cnfOut['pr2LeftGripper'] = [max(0., min(0.8, c.left_grip[0]))]
        cnfOut['pr2RightArm'] = c.right_joints
        cnfOut['pr2RightGripper'] = [max(0., min(0.8, c.right_grip[0]))]
        cnfOut['pr2Head'] = cnfIn.get('pr2Head', [0.,0.])
        # cnfOut['pr2Head'] = [0., 0.]

        print 'cnfOut[ pr2Head]=', cnfOut['pr2Head']

        if cnfIn:
            cnfOut = cnfIn.robot.normConf(JointConf(cnfOut, cnfIn.robot), cnfIn)
            cnfOut = enforceLimits(cnfOut)
            return resp.result, cnfOut
        else:
            return resp.result, cnfOut  # !!

    except rospy.ServiceException, e:
        print "Service call failed: %s"%e
        raw_input('Continue?')
        return None, None               # should we pretend it worked?

def gazeCoords(cnfIn):
    cnfInCart = cnfIn.cartConf()
    head = cnfInCart['pr2Head']
    headTurned = cnfInCart['pr2Head'].compose(headTurn)
    # Transform relative to robot base
    headTrans = cnfInCart['pr2Base'].inverse().compose(head)
    gaze = headTrans.applyToPoint(util.Point(np.array([0.,0.,1.,1.]).reshape(4,1)))
    confHead = gaze.matrix.reshape(4).tolist()[:3]
    confHead[2] = confHead[2] - 0.2     # brute force correction
    # if confHead[0] < 0:
    #     if debug('pr2GoToConf'):  print 'Dont look back!'
    #     confHead[0] = -conf.head[0]
    # if confHead[2] > 1.5:
    #     if debug('pr2GoToConf'): print 'Dont look up!'
    #     confHead[2] = 1.0
    if debug('pr2GoToConf'):
        print confHead
    return confHead

def enforceLimits(conf):
    outConfDict = {}
    for chain in conf.conf.keys():
        if not chain in ('pr2LeftArm', 'pr2LeftGripper', 'pr2RightArm', 'pr2RightGripper'):
            outConfDict[chain] = conf.conf[chain]
        else:
            limits = conf.robot.chains.chainsByName[chain].limits()
            outConfDict[chain] = [max(lo+0.001, min(hi-0.001, v)) \
                                  for (v, (lo, hi)) in zip(conf.conf[chain], limits)]
    return JointConf(outConfDict, conf.robot)

maxOpenLoopDist = 2.0                   # How far to move between looks

# The key interface spec...
# obs = env.executePrim(op, params)
class RobotEnv:                         # plug compatible with RealWorld (simulator)
    def __init__(self, world, bs, **args):
        self.world = world
        self.bs = bs                    # belief state for updates in prims

    # dispatch on the operators...
    def executePrim(self, op, params = None):
        def endExec(obs):
            print 'Executed', op.name, 'got obs', obs
            return obs
        if op.name == 'Move':
            return endExec(self.executeMove(op, params))
        if op.name == 'MoveNB':
            return endExec(self.executeMove(op, params, noBase=True))
        elif op.name == 'LookAtHand':
            return endExec(self.executeLookAtHand(op, params))
        elif op.name == 'LookAt':
            return endExec(self.executeLookAt(op, params))
        elif op.name == 'Pick':
            return endExec(self.executePick(op, params))
        elif op.name == 'Place':
            return endExec(self.executePlace(op, params))
        else:
            raise Exception, 'Unknown operator: '+str(op)

    def getObjShapes(self):
        shWorld = self.bs.pbs.getShadowWorld(0.95)
        held = shWorld.held.values()
        return [shWorld.objectShapes[obj] \
                for obj in shWorld.objectShapes if not obj in held]

    def executePath(self, path, placeBs):
        shWorld = self.bs.pbs.getShadowWorld(0.95)
        objShapes = shWorld.getObjectShapes()

        if debug('robotEnv'):
            for conf in path:
                conf.draw('W', 'blue')
        debugMsg('robotEnv', 'executePath')

        distSoFar = 0
        prevXYT = path[0]['pr2Base']
        for (i, conf) in enumerate(path):
            debugMsg('robotEnvCareful', '    conf[%d]'%i)
            newXYT = conf['pr2Base']
            result, outConf = pr2GoToConf(conf, 'move')

            # !! Do some looking and update the belief state.
            distSoFar += math.sqrt(sum([(prevXYT[i]-newXYT[i])**2 for i in (0,1)]))
            # approx pi => 1 meter
            distSoFar += 0.33*abs(util.angleDiff(prevXYT[2],newXYT[2]))
            print 'distSoFar', distSoFar
            # Check whether we should look
            args = 12*[None]
            if distSoFar >= maxOpenLoopDist:
                distSoFar = 0           #  reset
                obj = next(self.visibleShapes(conf, objShapes), None)
                if obj:
                    lookConf = lookAtConf(conf, obj)
                    if lookConf:
                        obs = self.doLook(lookConf, placeBs)
                        if obs:
                            args[1] = lookConf
                            lookAtBProgress(self.bs, args, obs)
                        else:
                            raw_input('No observation')
                    else:
                        raw_input('No lookConf for %s'%obj.name())
                else:
                    raw_input('No visible object')
            prevXYT = newXYT

        return None

    def executeMove(self, op, params, noBase=False):
        if noBase:
            # !! This should not move the base...use a better test  LPK
            startConf = op.args[0]
            targetConf = op.args[1]
            assert targetConf['pr2Base'] == \
                   startConf if isinstance(startConf, list) else startConf['pr2Base']
            raw_input('Should check noBase against actual robot conf')

        if params:
            path, interpolated, placeBs = params
            debugMsg('robotEnv', 'executeMove: path len = ', len(path))
            obs = self.executePath(path, placeBs)
        else:
            print op
            raw_input('No path given')
            obs = None
        return obs

    def visibleShapes(self, conf, objShapes):
        def rem(l,x): return [y for y in l if y != x]
        prob = 0.95
        world = self.bs.pbs.getWorld()
        shWorld = self.bs.pbs.getShadowWorld(prob)
        rob = self.bs.pbs.getRobot().placement(conf, attached=shWorld.attached)[0]
        solids = [s for s in objShapes if 'shadow' not in s.name()]
        obst =  solids + [rob]
        for s in solids:
            if visible(shWorld, conf, s, rem(obst,s), prob,
                       moveHead=True, fixed=rem(obst,s))[0]:
                yield s

    def executeLookAtHand(self, op, params):
        pass
    
    def executeLookAt(self, op, params):
        lookConf = op.args[1]

        if params:
            placeBs = params
        else:
            print op
            raw_input('No object distributions given')
            return None

        return self.doLook(lookConf, placeBs)

    def doLook(self, lookConf, placeBs):
        def lookAtTable(basePose, placeB):
            debugMsg('robotEnv', 'Get cloud?')
            scan = getPointCloud(basePose)
            ans = tables.getTableDetections(self.world, [placeB], scan)
            if ans:
                score, table = ans[0]
                if debug('robotEnv'):
                    print 'score=', score, 'table=', table.name()
                    table.draw('W', 'red')
                    raw_input(table.name())
                return table
            else:
                print 'No table found'
                return None

        shWorld = self.bs.pbs.getShadowWorld(0.95)
        objShapes = shWorld.getObjectShapes()
        visShapes = list(self.visibleShapes(lookConf, objShapes))
        visTables = [shape.name() for shape in visShapes \
                     if 'table' in shape.name() and 'shadow' not in shape.name()]
        obs = []

        debugMsg('robotEnv', 'executeLookAt', lookConf.conf)
        result, outConf = pr2GoToConf(lookConf, 'look')
        outConfCart = lookConf.robot.forwardKin(outConf)
        if visTables:
            assert len(visTables) == 1
            tableName = visTables[0]
            print 'Looking at table', tableName
            # Table is in world coordinates
            table = lookAtTable(outConfCart['pr2Base'], placeBs[tableName])
            if not table: return []
            basePose = outConfCart['pr2Base']
            tableRob = table.applyTrans(basePose.inverse())
            trueFace = supportFaceIndex(table)
            tablePose = getSupportPose(table, trueFace)
            obs.append((self.world.getObjType(tableName), trueFace, tablePose))
        else:
            raw_input('No tables visible... returning null obs')
            return []
        targets = []
        for shape in visShapes:
            if 'table' in shape.name(): continue
            targetObj = shape.name()
            supportTableB = findSupportTable(targetObj, self.world, placeBs)
            assert supportTableB
            if not supportTableB.obj == tableName:
                print 'Skipping obj', targetObj, 'not supported by', tableName
                continue
            placeB = placeBs[supportTableB.obj]
            targets.append((targetObj, placeBs[targetObj]))

        if targets:
            print 'Fix height of table'
            surfacePoly = makeROSPolygon(tableRob, zPlane=0.67) # from perceived table
            ans = getObjDetections(self.world,
                                   dict(targets),
                                   outConf, # the lookConf actually achieved
                                   [surfacePoly])
            for (score, objType, objPlaceRobot) in ans:
                if not objPlaceRobot:
                    continue
                trueFace = supportFaceIndex(objPlaceRobot)
                objPlace = objPlaceRobot.applyTrans(outConfCart['pr2Base'])
                pose = getSupportPose(objPlace, trueFace)
                obs.append((self.world.getObjType(objPlace.name()),
                            trueFace, pose))
                if debug('robotEnv'):
                    print 'Obs', objType, objPlace.name(), 'score=', score,
                    print 'face=', trueFace, 'pose=', pose
                    objPlace.draw('W', 'cyan')
                    raw_input(objType)
        if debug('robotEnv') and not obs:
            raw_input('Got no observations for %s'%([shape.name() for shape in visShapes]))
        return obs

    def lookObjShape(self, placeB):
        shape = self.world.getObjectShapeAtOrigin(placeB.obj)
        return shape.applyLoc(placeB.objFrame())

    def executePick(self, op, params):
        (obj, hand, pickConf, approachConf) = \
                 (op.args[0], op.args[1], op.args[11], op.args[9])

        if params:
            placeBs = params
        else:
            print op
            raw_input('No object distributions given')
            return None

        gripper = 'pr2LeftGripper' if hand=='left' else 'pr2RightGripper'

        debugMsg('robotEnv', 'executePick - open')
        result, outConf = pr2GoToConf(approachConf, 'move')
        result, outConf = pr2GoToConf(lookAtConf(approachConf, self.lookObjShape(placeBs[obj])),
                                      'look')
        
        debugMsg('robotEnv', 'executePick - move to pickConf')
        reactiveApproach(approachConf, pickConf, 0.06, hand)

        debugMsg('robotEnv', 'executePick - close')
        result, outConf = pr2GoToConf(pickConf, 'close', arm=hand[0]) # 'l' or 'r'
        g = confGrip(outConf, hand)
        gripConf = gripOpen(outConf, hand, g-0.03)
        # close then grab
        result, outConf = pr2GoToConf(gripConf, 'open')
        result, outConf = pr2GoToConf(gripConf, 'grab', arm=hand[0])
        if debug('robotEnv'):
            raw_input('Closed?')

        debugMsg('robotEnv', 'executePick - move to approachConf')
        result, outConf = pr2GoToConf(approachConf, 'move')

        return None

    def executePlace(self, op, params):
        (hand, placeConf, approachConf) = \
               (op.args[1], op.args[-6], op.args[-8])

        debugMsg('robotEnv', 'executePlace - move to approachConf')
        result, outConf = pr2GoToConf(approachConf, 'move')
        
        debugMsg('robotEnv', 'executePlace - move to placeConf')
        result, outConf = pr2GoToConf(placeConf, 'move')

        debugMsg('robotEnv', 'executePlace - open')
        placeConf = gripOpen(outConf, hand, 0.08) # keep height
        result, outConf = pr2GoToConf(placeConf, 'open')

        debugMsg('robotEnv', 'executePlace - move to approachConf')
        result, outConf = pr2GoToConf(approachConf, 'move')
        
        return None

def getObjDetections(world, obsTargets, robotConf, surfacePolys, maxFitness = 3):
    targetPoses = dict([(placeB.obj, placeB.poseD.mode()) \
                        for placeB in obsTargets.values()])
    baseX, baseY, baseTh = robotConf['pr2Base']
    robotFrame = util.Pose(baseX, baseY, 0.0, baseTh)
    robotFrameInv = robotFrame.inverse()
    targetPosesRobot = dict([(obj, robotFrameInv.compose(p)) \
                             for (obj, p) in targetPoses.items()])
    shWidths = [shadowWidths(pB.poseD.var, pB.delta, 0.99)[:3] \
                     for pB in obsTargets.values()]
    minDist = 1.0
    targetDistsXY = [max(w[:2] + [minDist]) for w in shWidths]
    targetDistsZ = [max(w[2:3] + [minDist]) for w in shWidths]
    if debug('robotEnv'):
        print 'Original target poses', targetPoses
        print 'Robot relative target poses', targetPosesRobot
        print 'targetDistsXY', targetDistsXY
        print 'targetDistsZ', targetDistsZ

    rospy.wait_for_service('detect_models')
    detect = rospy.ServiceProxy('detect_models', DetectModels)
    targetModels = [world.getObjType(o) for o in targetPoses]
    targetObjForType = {}
    for o in targetPoses:               # reverse index
        targetObjForType[world.getObjType(o)] = o
    if debug('robotEnv'):
        print 'calling perception with', targetModels
    reqD = DetectModelsRequest(timeout = rospy.Duration(15),
                               surface_polygons = surfacePolys,
                               models = targetModels,
                               initial_poses = [makeROSPose3D(targetPosesRobot[o]) for o in targetPosesRobot],
                               max_dists_xy = targetDistsXY,
                               max_dists_z = targetDistsZ)
    reqD.header.frame_id = '/base_footprint'
    reqD.header.stamp = rospy.Time.now()
    state = None
    try:
        state = detect(reqD)
        if debug('robotEnv'):
            print 'perception state', state
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e
    assert state
    detections = []
    for obj in state.scene.objects:
        if obj.fitness_score < maxFitness: # fitness is in std dev, so high is bad
            trans = TransformFromROSMsg(obj.pose.position, obj.pose.orientation)

            # Pick a representative object of the type and use its shape ??
            objType = obj.name
            objShape = world.getObjectShapeAtOrigin(targetObjForType[objType]).applyLoc(trans)

            detections.append((obj.fitness_score, objType, objShape))
    detections.sort()                   # lower is better
    return detections

def getPointCloud(basePose, resolution = glob.cloudPointsResolution):
    rospy.wait_for_service('point_cloud')
    time.sleep(3)
    print 'Asking for cloud points'
    try:
        getCloudPts = rospy.ServiceProxy('point_cloud', PointCloud)
        reqC = PointCloudRequest(resolution = resolution/5)
        reqC.header.frame_id = '/base_footprint'
        reqC.header.stamp = rospy.Time.now()
        response = None
        response = getCloudPts(reqC)
        print 'Got cloud points:', len(response.cloud.points)
        trans = response.cloud.eye.transform
        headTrans = TransformFromROSMsg(trans.translation, trans.rotation)
        eye = headTrans.point()
        print 'headTrans\n', headTrans.matrix
        print 'eye', eye
        points = np.zeros((4, len(response.cloud.points)+1), dtype=np.float64)
        points[:,0] = eye.matrix[:,0]
        for i, p in enumerate(response.cloud.points):
            points[:, i+1] = (p.x, p.y, p.z, 1.0)
        scan = pc.Scan(headTrans, None,
                       verts=points, name='ROS')
        scan = scan.applyTrans(basePose)
        scan.draw('W', 'red')
        raw_input('Scan')
        return scan
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e
        raw_input('Continue?')
        return []

def makeROSPolygon(obj, zPlane=None):
    points = []
    verts = obj.xyPrim().vertices()
    zhi = obj.zRange()[1]
    for p in range(verts.shape[1]):
        (x, y, z) = verts[0:3,p]
        if abs(z - zhi) < 0.001:
            points.append(gm.Point(x, y, zPlane or z))
    return gm.Polygon(points)

def makeROSPose2D(pose):
    pose = pose.pose()
    return gm.Pose2D(pose.x, pose.y, pose.theta)

def makeROSPose3D(pose):
    (x,y,z,w) = pose.quat().matrix
    (px, py, pz) = pose.matrix[0:3,3]
    rpose = gm.Pose()
    rpose.position.x = px
    rpose.position.y = py
    rpose.position.z = pz
    rpose.orientation.x = x
    rpose.orientation.y = y
    rpose.orientation.z = z
    rpose.orientation.w = w
    return rpose

def TransformFromROSMsg(pos, quat):
    pos = np.array([[x] for x in [pos.x, pos.y, pos.z, 1.0]])
    quat = np.array([quat.x, quat.y, quat.z, quat.w])
    return util.Transform(p=pos, q=quat)

def transformPoly(poly, x, y, z, theta):
    def pt(coords):
        return gm.Point(coords[0], coords[1], coords[2])
    tr = numpy.dot(transf.translation_matrix([x, y, z]),
                   transf.rotation_matrix(theta, (0,0,1)))
    points = [pt(numpy.dot(tr, [p.x, p.y, p.z, 1.0])) for p in poly.points]
    return gm.Polygon(points)

def wellLocalized(pB):
    widths = shadowWidths(pB.poseD.var, pB.delta, 0.95)
    print pB.obj, 'well localized?', widths, '<', 4*(0.03,)
    return(all(x<=y for (x,y) in zip(widths, 4*(0.03,))))

def getSupportPose(shape, supportFace):
    pose = shape.origin().compose(shape.faceFrames()[supportFace])
    print 'origin frame\n', shape.origin().matrix
    print 'supportPose\n', pose.matrix
    return pose

# Given an approach conf and a target conf, interpolate cartesion
# poses and check for contacts.  We assume that the approach and
# target have the same hand orientations, and only the origin is
# displaced.

# Mnemonic access indeces
obsConf, obsGrip, obsTrigger, obsContacts = range(4)

# x direction is approach direction.
xoffset = 0.05
yoffset = 0.01

def pr2GoToConfNB(conf, op, arm='both', speedFactor=glob.speedFactor):
    c = conf.conf.copy()
    if 'pr2Base' in c: del c['pr2Base']
    return pr2GoToConf(conf, op, arm=arm, speedFactor=speedFactor)

def reactiveApproach(startConf, targetConf, gripDes, hand, tries = 10):
    print '***closing'
    startConfClose = gripOpen(startConf, hand, 0.01)
    pr2GoToConfNB(startConfClose, 'move')
    pr2GoToConfNB(startConfClose, 'open')
    targetConfClose = gripOpen(targetConf, hand, 0.01)
    (obs, traj) = tryGrasp(startConfClose, targetConfClose, hand)
    print 'obs after tryGrasp', obs
    curConf = obs[obsConf]
    print '***Contact'
    backConf = displaceHand(curConf, hand, dx=-xoffset, dz=0.01, nearTo=startConf)
    result, nConf = pr2GoToConfNB(backConf, 'move')
    backConf = displaceHand(backConf, hand, zFrom=targetConf, nearTo=startConf)
    result, nConf = pr2GoToConfNB(backConf, 'move')
    backConf = gripOpen(backConf, hand)
    result, nConf = pr2GoToConfNB(backConf, 'open')
    print 'backConf', handTrans(nConf, hand).point(), result
    target = displaceHand(curConf, hand, dx=1.1*xoffset, zFrom=targetConf, nearTo=startConf)
    return reactiveApproachLoop(backConf, target, gripDes, hand,
                                maxTarget=target)

def reactiveApproachLoop(startConf, targetConf, gripDes, hand, maxTarget,
                         ystep = 0.04, tries = 10):
    spaces = (10-tries)*' '
    if tries == 0:
        print spaces+'reactiveApproach failed'
        return None
    startConf = gripOpen(startConf, hand)
    targetConf = gripOpen(targetConf, hand)
    (obs, traj) = tryGrasp(startConf, targetConf, hand)
    print spaces+'obs after tryGrasp', obs
    curConf = obs[obsConf]
    if reactBoth(obs):
        if abs(obs[obsGrip] - gripDes) < 0.02:
            print spaces+'***holding'
            return obs
        else:
            print spaces+'***opening', obs[obsGrip], 'did not match', gripDes
            closeConf = gripOpen(curConf, hand)
            pr2GoToConfNB(closeConf, 'open')
    if reactLeft(obs):
        print spaces+'***reactLeft'
        backConf = displaceHand(curConf, hand, dx=-xoffset, nearTo=startConf)
        result, nConf = pr2GoToConfNB(backConf, 'move')
        print spaces+'backConf', handTrans(nConf, hand).point(), result
        return reactiveApproachLoop(backConf, 
                                    displaceHand(curConf, hand,
                                                 dx=1.1*xoffset, dy=ystep,
                                                 maxTarget=maxTarget, nearTo=startConf),
                                    gripDes, hand, maxTarget,
                                    ystep = max(0.01, ystep/2), tries=tries-1)
    else:                           # default, just to do something...
        print spaces+'***reactRight'
        backConf = displaceHand(curConf, hand, dx=-xoffset, nearTo=startConf)
        result, nConf = pr2GoToConfNB(backConf, 'move')
        print spaces+'backConf', handTrans(nConf, hand).point(), result
        return reactiveApproachLoop(backConf, 
                                    displaceHand(curConf, hand,
                                                 dx=1.1*xoffset, dy=-ystep,
                                                 maxTarget=maxTarget, nearTo=startConf),
                                    gripDes, hand, maxTarget,
                                    ystep = max(0.01, ystep/2), tries=tries-1)

def displaceHand(conf, hand, dx=0.0, dy=0.0, dz=0.0,
                 zFrom=None, maxTarget=None, nearTo=None):

    print 'displaceHand pr2Head', conf['pr2Head']

    cart = conf.cartConf()
    handFrameName = conf.robot.armChainNames[hand]
    trans = cart[handFrameName]
    if zFrom:
        diff = trans.inverse().compose(zFrom.cartConf()[handFrameName])
        dz = diff.matrix[2,3]
        print 'trans\n', trans.matrix
        print 'zFrom\n', zFrom.cartConf()[handFrameName].matrix
        print 'dz', dz
        raw_input('Ok?')
    if maxTarget:
        diff = trans.inverse().compose(maxTarget.cartConf()[handFrameName])
        max_dx = diff.matrix[0,3]
        print 'displaceHand', 'dx', dx, 'max_dx', max_dx
        dx = max(0., min(dx, max_dx)) # don't go past maxTrans
    nTrans = trans.compose(util.Pose(dx, dy, dz, 0.0))
    nCart = cart.set(handFrameName, nTrans)
    if debug('invkin'):
        if nearTo: nearTo.prettyPrint('nearTo conf:')
    nConf = conf.robot.inverseKin(nCart, conf=(nearTo or conf)) # use conf to resolve
    nConf.prettyPrint('displaceHand Conf:')
    if nConf.conf[handFrameName]:
        return nConf
    else:
        print 'displaceHand: failed kinematics'
        return conf
def reactBoth(obs):
    return obs[obsContacts][1] and obs[obsContacts][3] 
def reactRight(obs):
    return obs[obsTrigger] in ('R_tip', 'R_pad') \
           or obs[obsContacts][2] or obs[obsContacts][3]
def reactLeft(obs):
    return obs[obsTrigger] in ('L_tip', 'L_pad') \
           or obs[obsContacts][0] or obs[obsContacts][1]
def gripOpen(conf, hand, width=0.08):
    return conf.set(conf.robot.gripperChainNames[hand], [width])
def confGrip(conf, hand):
    return conf.get(conf.robot.gripperChainNames[hand])[0]
def handTrans(conf, hand):
    cart = conf.cartConf()
    handFrameName = conf.robot.armChainNames[hand]
    return cart[handFrameName]

# Could use 2 to not do interpolation.
cartInterpolationSteps = 6

def tryGrasp(approachConf, graspConf, hand, stepSize = 0.05,
             maxSteps = cartInterpolationSteps, verbose = False):
    def parseContacts(result):
        if result == 'LR_pad':
            contacts = [False, True, False, True]
        elif result == 'L_pad':
            contacts = [False, True, False, False]
        elif result == 'R_pad':
            contacts = [False, False, False, True]
        elif result == 'LR_tip':
            contacts = [True, False, True, False]
        elif result == 'L_tip':
            contacts = [True, False, False, False]
        elif result == 'R_tip':
            contacts = [False, False, True, False]
        elif result == 'none':
            contacts = 4*[False]
        else:
            raw_input('Unexpected contact result')
            contacts = 4*[False]
        return contacts
    def close():
        result = compliantClose(curConf, hand, 0.005)
        return parseContacts(result)
    print 'tryGrasp'
    print '    from', handTrans(approachConf, hand).point()
    print '      to', handTrans(graspConf, hand).point()
    result, curConf = pr2GoToConfNB(approachConf, 'move')
    resuly, curConf = pr2GoToConfNB(approachConf, 'resetForce', arm=hand[0])
    moveChains = [approachConf.robot.armChainNames[hand]+'Frame']
    path = cartInterpolators(graspConf, approachConf, stepSize)[::-1]
    # path = [approachConf, graspConf]
    if len(path) > maxSteps:
        inc = len(path)/(maxSteps - 1)
        ind = 0
        npath = [path[0]]
        while len(npath) < (maxSteps - 1):
            ind += inc
            npath.append(path[int(ind)])
        npath.append(path[-1])
        path = npath
    if not path:
        print 'No interpolation path'
        contacts = 4*[False]
    for i, p in enumerate(path):
        print i,  handTrans(p, hand).point()
    if debug('robotEnv'):
        raw_input('Go?')
    prevConf = None
    for i, conf in enumerate(path):
        conf.prettyPrint('tryGrasp conf %d'%i)
        if prevConf:
            bigAngleWarn(prevConf, conf)
        prevConf = conf
        result, curConf = pr2GoToConfNB(conf, 'moveGuarded', speedFactor=0.1)
        print 'tryGrasp result', result, handTrans(curConf, hand).point()
        if result in ('LR_tip', 'L_tip', 'R_tip',
                      'LR_pad', 'L_pad', 'R_pad'):
            contacts = parseContacts(result)
            break
        elif result == 'Acc':
            contacts = 4*[False]
            # break
            continue            # ignore Acc
        elif result == 'goal':
            contacts = 4*[False]
            continue
        else:
            raw_input('Unknown compliantClose result = %s'%result)
            contacts = 4*[False]
    if result in ('goal', 'Acc'):
        contacts = close()
    obs = (curConf, curConf[conf.robot.gripperChainNames[hand]][0],
           result, contacts)
    return obs, (approachConf, curConf)

def compliantClose(conf, hand, step = 0.01, n = 1):
    if n > 5:
        (result, cnfOut) = pr2GoToConfNB(conf, 'close', arm=hand[0])
        return result
    print 'compliantClose step=', step
    result, curConf = pr2GoToConfNB(conf, 'closeGuarded', arm=hand[0])
    print 'compliantClose result', result, handTrans(curConf, hand).point()
    # could displace to find contact with the other finger
    # instead of repeatedly closing.
    if result == 'LR_pad':
        (result, cnfOut) = pr2GoToConfNB(conf, 'close', arm=hand[0], speedFactor=0.1)
        return result
    elif result in ('L_pad', 'R_pad'):
        off = step if result == 'L_pad' else -step
        nConf = displaceHand(curConf, hand, dy=off, nearTo=conf)
        pr2GoToConfNB(nConf, 'move', speedFactor=0.1)      # should this be guarded?
        return compliantClose(nConf, hand, step=0.9*step, n = n+1)
    elif result == 'none':
        return result
    else:
        raw_input('Bad result in compliantClose: %s'%str(result))
        return result

def testReactive(startConf,
                 offset = (glob.approachBackoff, 0.0, -glob.approachPerpBackoff),
                 grip=0.06):
    hand = 'left'
    (dx,dy,dz) = offset
    targetConf = displaceHand(startConf, hand, dx=dx, dy=dy, dz=dz, nearTo=startConf)
    obs = reactiveApproach(startConf, targetConf, grip, hand)
    curConf = obs[obsConf]
    g = confGrip(curConf, hand)
    print 'grip after reactive', g
    gripConf = gripOpen(curConf, hand, 0.04)
    result, outConf = pr2GoToConfNB(gripConf, 'open')
    g = confGrip(outConf, hand)
    print 'grip after tightening', g
    raw_input('Done?')

