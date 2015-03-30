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
from pr2Util import shadowWidths, supportFaceIndex
from miscUtil import argmax
from pr2Robot2 import JointConf

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
    # rospy.init_node('hpn'); print 'Starting ROS'

## Communicate with ROS controller to move the robot, just unpack the configuration.

# operation could be:
# reset - treat current location as origin
# resetForce - recalibrate touch sensors
# move - move to configuration
# closeGuarded - use ROS guarded grip
# grab - use ROS grab
# close - as hard as you can

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
        conf.left_grip = map(float, cnfIn.get('pr2LeftGripper', []))
        conf.right_joints = map(float, cnfIn.get('pr2RightArm', []))
        conf.right_grip = map(float, cnfIn.get('pr2RightGripper', []))
        conf.head = map(float, cnfIn.get('pr2Head', []))
        if conf.head:
            cnfInCart = cnfIn.cartConf()
            head = cnfInCart['pr2Head']
            headTurned = cnfInCart['pr2Head'].compose(headTurn)
            # Transform relative to robot base
            headTrans = cnfInCart['pr2Base'].inverse().compose(head)
            gaze = headTrans.applyToPoint(util.Point(np.array([0.,0.,1.,1.]).reshape(4,1)))
            conf.head = gaze.matrix.reshape(4).tolist()[:3]
            if conf.head[0] < 0:
                print 'Dont look back!'
                conf.head[0] = -conf.head[0]
            if conf.head[2] > 1.5:
                print 'Dont look up!'
                conf.head[2] = 1.0
            print conf.head

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
        cnfOut['pr2Head'] = c.head or cnfIn.get('pr2Head', [0.,0.])
        if cnfIn:
            return resp.result, JointConf(cnfOut, cnfIn.robot)
        else:
            return resp.result, cnfOut  # !!

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
        if debug('robotEnv'):
            for conf in path:
                conf.draw('W', 'blue')
        debugMsg('robotEnv', 'executePath')
        for (i, conf) in enumerate(path):
            debugMsg('robotEnvCareful', '    conf[%d]'%i)
            result, outConf = pr2GoToConf(conf, 'move')
        return None

    def executeMove(self, op, params):
        if params:
            path = params
            debugMsg('robotEnv', 'executeMove: path len = ', len(path))
            obs = self.executePath(path)
        else:
            print op
            raw_input('No path given')
            obs = None
        return obs

    def executeLookAtHand(self, op, params):
        pass
    
    def executeLookAt(self, op, params):
        def lookAtTable(tableName):
            debugMsg('robotEnv', 'Get cloud?')
            scan = getPointCloud()
            ans = tables.getTableDetections(self.world, [tableName], scan)
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

        (targetObj, lookConf) = \
                    (op.args[0], op.args[1])
        if params:
            placeBs = params
        else:
            print op
            raw_input('No object distributions given')
            return None

        debugMsg('robotEnv', 'executeLookAt', targetObj, lookConf.conf)
        result, outConf = pr2GoToConf(lookConf, 'move')
        outConfCart = lookConf.robot.forwardKin(outConf)
        if 'table' in targetObj:
            table = lookAtTable(targetObj)
            if not table: return None
            trueFace = supportFaceIndex(table)
            tablePoseRobot = getSupportPose(table, trueFace)
            tablePose = outConfCart['pr2Base'].compose(tablePoseRobot)
            #!! needs generalizing
            return ('table', trueFace, tablePose)
        elif targetObj in placeBs:
            supportTable = findSupportTable(targetObj, self.world, placeBs)
            assert supportTable
            placeB = placeBs[supportTable]
            if not wellLocalized(placeB):
                tableRobot = lookAtTable(supportTable)
                table = tableRobot.applyTrans(outConfCart['pr2Base'])
                if not table: return None
            else:
                table = world.getObjectShapeAtOrigin(supportTable).applyLoc(placeB.objFrame())
            surfacePoly = makeROSPolygon(table)
            ans = getObjDetections(self.world,
                                   {targetObj: placeBs[targetObj]},
                                   outConf, # the lookConf actually achieved
                                   [surfacePoly])
            if ans:
                # This is in robot coords
                score, objPlaceRobot = ans[0]
                if not objPlaceRobot:
                    print 'No detections'
                    raw_input()
                    return None
            trueFace = supportFaceIndex(objPlaceRobot)
            objPlace = objPlaceRobot.applyTrans(outConfCart['pr2Base'])
            if debug('robotEnv'):
                objPlace.draw('W', 'red')
                raw_input(objPlace.name())
            return (objPlace.name(), trueFace, getSupportPose(objPlace, trueFace))
        else:
            raw_input('Unknown object: %s'%targetObj)
            return None

    def executePick(self, op, params):
        (hand, pickConf, approachConf) = \
               (op.args[1], op.args[17], op.args[15])
        gripper = 'pr2LeftGripper' if hand=='left' else 'pr2RightGripper'

        debugMsg('robotEnv', 'executePick - open')
        conf = approachConf.copy()
        conf.conf[gripper] = 0.08
        result, outConf = pr2GoToConf(conf, 'move')
        
        debugMsg('robotEnv', 'executePick - move to pickConf')
        result, outConf = pr2GoToConf(pickConf, 'move')

        debugMsg('robotEnv', 'executePick - close')
        result, outConf = pr2GoToConf(pickConf, 'close', arm=hand[0]) # 'l' or 'r'

        debugMsg('robotEnv', 'executePick - move to approachConf')
        result, outConf = pr2GoToConf(approachConf, 'move')

        return None

    def executePlace(self, op, params):
        (hand, placeConf, approachConf) = \
               (op.args[1], op.args[20], op.args[18])
        gripper = 'pr2LeftGripper' if hand=='left' else 'pr2RightGripper'

        debugMsg('robotEnv', 'executePlace - move to placeConf')
        result, outConf = pr2GoToConf(placeConf, 'move')

        debugMsg('robotEnv', 'executePlace - open')
        conf = placeConf.copy()
        conf.conf[gripper] = 0.08
        result, outConf = pr2GoToConf(conf, 'move')

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
            #!! HACK
            name = obsTargets.keys()[0]
            trans = TransformFromROSMsg(obj.pose.position, obj.pose.orientation)
            objShape = world.getObjectShapeAtOrigin(name).applyLoc(trans)
            detections.append((obj.fitness_score, objShape))
    detections.sort()                   # lower is better
    return detections

def getPointCloud(resolution = glob.cloudPointsResolution):
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
        print 'eye', eye
        points = np.zeros((4, len(response.cloud.points)+1), dtype=np.float64)
        points[:,0] = eye.matrix[:,0]
        for i, p in enumerate(response.cloud.points):
            points[:, i+1] = (p.x, p.y, p.z, 1.0)
        scan = pc.Scan(headTrans, None,
                       verts=points, name='ROS')
        scan.draw('W', 'red')
        return scan
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e
        raw_input('Continue?')
        return []

def makeROSPolygon(obj, dz=0):
    points = []
    verts = obj.xyPrim().vertices()
    zhi = obj.zRange()[1]
    for p in range(verts.shape[1]):
        (x, y, z) = verts[0:3,p]
        if abs(z - zhi) < 0.001:
            points.append(gm.Point(x, y, z))
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
    return(all(x<=y for (x,y) in zip(shadowWidths(pB.poseD.var, pB.delta, 0.99),
                                     4*(0.02,))))

def findSupportTable(targetObj, world, placeBs):
    tableBs = [pB for pB in placeBs.values() if 'table' in pB.obj]
    print 'tablesBs', tableBs
    tableCenters = [pB.poseD.mode().point() for pB in tableBs]
    targetB = placeBs[targetObj]
    assert targetB
    targetCenter = targetB.poseD.mode().point()
    bestCenter = argmax(tableCenters, lambda c: -targetCenter.distance(c))
    ind = tableCenters.index(bestCenter)
    return tableBs[ind].obj

def getSupportPose(shape, supportFace):
    pose = shape.origin().compose(shape.faceFrames()[supportFace])
    print 'origin frame\n', shape.origin().matrix
    print 'supportPose\n', pose.matrix
    return pose

