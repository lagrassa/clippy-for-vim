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
from pr2Util import shadowWidths

import util
import tables
reload(tables)

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

## Communicate with ROS controller to move the robot, just unpack the configuration.

# operation could be:
# reset - treat current location as origin
# resetForce - recalibrate touch sensors
# move - move to configuration
# closeGuarded - use ROS guarded grip
# grab - use ROS grab
# close - as hard as you can

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
        conf.left_grip = map(float, cnfIn.get('pr2LeftGripper', []))
        conf.right_joints = map(float, cnfIn.get('pr2RightArm', []))
        conf.right_grip = map(float, cnfIn.get('pr2RightGripper', []))
        conf.head = map(float, cnfIn.get('pr2Head', []))
        if conf.head:
            conf.head = [1.0, 0.0, 0.6]

        print operation, conf
        
        resp = gotoConf(operation, conf, speedFactor)

        print 'response', resp
        c = resp.resultConf

        cnfOut = {}
        cnfOut['pr2Base'] = c.base
        cnfOut['pr2Torso']  = c.torso
        cnfOut['pr2LeftArm'] = c.left_joints
        cnfOut['pr2LeftGripper'] = c.left_grip
        cnfOut['pr2RightArm'] = c.right_joints
        cnfOut['pr2RightGripper'] = c.right_grip
        cnfOut['pr2Head'] = c.head
        return resp.result, cnfOut

    except rospy.ServiceException, e:
        print "Service call failed: %s"%e
        raw_input('Continue?')
        return None, None               # should we pretend it worked?

# The key interface spec...
# obs = env.executePrim(op, params)
class RobotEnv:                         # plug compatible with RealWorld (simulator)
    def __init__(self, world, **args):
        self.world = world

    # dispatch on the operators...
    def executePrim(self, op, params = None):
        def endExec(obs):
            print 'Executed', op.name, 'got obs', obs
            debugMsg('executePrim')
            return obs
        if op.name == 'Move':
            return endExec(self.executeMove(op, params))
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

    def executePath(self, path):
        debugMsg('robotEnv', 'executePath')
        for (i, conf) in enumerate(path):
            debugMsg('robotEnv', '    conf[%d]'%i)
            result, outConf = pr2GoToConf(conf, 'move')
        return 'none'                   # !! check for collision?

    def executeMove(self, op, params):
        if params:
            path = params
            debugMsg('robotEnv', 'executeMove: path len = ', len(path))
            obs = self.executePath(path)
        else:
            print op
            raw_input('No path given')
            obs = 'none'
        return obs

    def executeLookAtHand(self, op, params):
        pass
    
    def executeLookAt(self, op, params):
        def lookAtTable(tableName):
            debugMsg('robotEnv', 'Get cloud?')
            scan = getPointCloud()
            score, table = tables.getTableDetections(self.world, [tableName], scan)[0]
            if debug('robotEnv'):
                table.draw('W', 'red')
                raw_input(table.name())
            return table

        (targetObj, lookConf) = \
                    (op.args[0], op.args[1])
        if params:
            placeBs = params
        else:
            print op
            raw_input('No object distributions given')
            return 'none'

        debugMsg('robotEnv', 'executeLookAt', targetObj, lookConf.conf)
        result, outConf = pr2GoToConf(lookConf, 'move')
        if 'table' in targetObj:
            table = lookAtTable(targetObj)
            trueFace = supportFaceIndex(table)
            return (targetObj, trueFace, table.origin())
        elif targetObj in placeBs:
            supportTable = findSupportTable(targetObj, self.world, placeBs)
            assert supportTable
            placeB = placeBs[supportTable]
            if not wellLocalized(placeB):
                table = lookAtTable(supportTable)
            else:
                table = world.getObjectShapeAtOrigin(supportTable).applyLoc(placeB.objFrame())
            surfacePolyPoly = makeROSPolygon(table)
            score, objPlace = getObjDetections(self.world,
                                               {targetObj: placeBs[targetObj]},
                                               outConf, # the lookConf actually achieved
                                               [surfacePoly])[0]
            if debug('robotEnv'):
                objPlace.draw('W', 'red')
                raw_input(objPlace.name())

            trueFace = supportFaceIndex(objPlace)
            return (objPlace, trueFace, objPlace.origin())
        else:
            raw_input('Unknown object: %s'%targetObj)
            return 'none'

    def executePick(self, op, params):
        (hand, pickConf, approachConf) = \
               (op.args[1], op.args[17], op.args[15])
        gripper = 'pr2LeftGripper' if hand=='left' else 'pr2RightGripper'

        debugMsg('robotEnv', 'executePick - open')
        result, outConf = pr2GoToConf({gripper: 0.08}, 'move')
        
        debugMsg('robotEnv', 'executePick - move to pickConf')
        result, outConf = pr2GoToConf(pickConf, 'move')

        debugMsg('robotEnv', 'executePick - close')
        result, outConf = pr2GoToConf({}, 'close', arm=hand[0]) # 'l' or 'r'

        debugMsg('robotEnv', 'executePick - move to approachConf')
        result, outConf = pr2GoToConf(approachConf, 'move')

        return 'none'

    def executePlace(self, op, params):
        (hand, placeConf, approachConf) = \
               (op.args[1], op.args[20], op.args[18])
        gripper = 'pr2LeftGripper' if hand=='left' else 'pr2RightGripper'

        debugMsg('robotEnv', 'executePick - move to pickConf')
        result, outConf = pr2GoToConf(pickConf, 'move')

        debugMsg('robotEnv', 'executePick - open')
        result, outConf = pr2GoToConf({gripper: 0.08}, 'move')

        debugMsg('robotEnv', 'executePick - move to approachConf')
        result, outConf = pr2GoToConf(approachConf, 'move')
        
        return 'none'

def getObjDetections(world, obsTargets, robotConf, surfacePolys, maxFitness = 3):
    targetPoses = dict([(placeB.obj, placeB.poseD.mode()) \
                        for placeB in obsTargets.values()])
    robotFrame = robotConf['pr2Base']
    robotFrameInv = robotFrame.inverse()
    targetPosesRobot = dict([(obj, robotFrameInv.compose(p)) \
                             for (obj, p) in targetPoses.items()])
    shWidths = [shadowWidths(pB.poseD.var, pB.delta, 0.99)[:3] \
                     for pB in obsTargets.values()]
    targetDistsXY = [max(w[:2] + [0.05]) for w in shWidths]
    targetDistsZ = [max(w[2:3] + [0.05]) for w in shWidths]
    if debug('robotEnv'):
        print 'Original target poses', targetPoses
        print 'Robot relative target poses', targetPosesRobot
        print 'targetDistsXY', targetDistsXY
        print 'targetDistsZ', targetDistsZ

    rospy.wait_for_service('detect_models')
    detect = rospy.ServiceProxy('detect_models', DetectModels)
    # targetModels = [o.typeName for o in targetPoses if o.typeName]
    targetModels = ['soda']
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
        if obj.fitness_score < maxFitness:
            objShape = world.getObjectShapeAtOrigin(obj.name).applyLoc(obj.pose)
            detections.append((obj.fitness_score, objShape))
    detections.sort()                   # lower is better
    return detections

def getPointCloud(resolution = glob.cloudPointsResolution):
    rospy.wait_for_service('point_cloud')
    time.sleep(3)
    print 'Asking for cloud points'
    try:
        getCloudPts = rospy.ServiceProxy('point_cloud', PointCloud)
        reqC = PointCloudRequest(resolution = resolution)
        reqC.header.frame_id = '/base_footprint'
        reqC.header.stamp = rospy.Time.now()
        response = None
        response = getCloudPts(reqC)
        print 'Got cloud points:', len(response.cloud.points)
        headTrans = TransformFromROSMsg(response.cloud.eye)
        eye = headTrans.point()
        print 'eye', eye
        points = np.zeros((4, len(response.cloud.points)+1), dtype=np.float64)
        points[:,0] = eye.matrix
        for i, pt in enumerate(response.cloud.points):
            points[:, i+1] = (p.x, p.y, p.z, 1.0)
        return pc.Scan(headTrans, None,
                       verts=points, name='ROS')
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e
        raw_input('Continue?')
        return []

def makeROSPolygon(obj, dz=0):
    points = []
    verts = obj.xyPrim().verts()
    zhi = obj.zRange()[1]
    for p in verts.shape[1]:
        (x, y, z) = verts[0:3,p]
        if abs(z - zhi) < 0.001:
            points.append(gm.Point(x, y, z))
    return gm.Polygon(points)

def makeROSPose2D(pose):
    pose = pose.pose()
    return gm.Pose2D(pose.x, pose.y, pose.theta)

def makeROSPose3D(pose):
    T = pose.matrix
    (x,y,z,w) = T.quat().matrix
    (px, py, pz) = T.matrix[0:3,3]
    rpose = gm.Pose()
    rpose.position.x = px
    rpose.position.y = py
    rpose.position.z = pz
    rpose.orientation.x = x
    rpose.orientation.y = y
    rpose.orientation.z = z
    rpose.orientation.w = w
    return rpose

def TransformFromROSMsg(msg):
    pos = msg.transform.translation
    quat = msg.transform.rotation
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
