import numpy as np
import random
import math
import pdb
import copy
import time
import transformations as transf

import planGlobals as glob
import windowManager3D as wm

import util
import tables as tab
reload(tab)

if glob.useROS:
    import rospy
    import pr2_hpn.msg
    import pr2_hpn.srv
    from pr2_hpn.srv import *
    import roslib
    roslib.load_manifest('pr2_hpn')
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
                speedFactor = glob.speedFactor):
    if not glob.useROS: return None, None
    rospy.wait_for_service('pr2_goto_configuration')
    try:
        gotoConf = rospy.ServiceProxy('pr2_goto_configuration',
                                      pr2_hpn.srv.GoToConf)
        conf = pr2_hpn.msg.Conf()
        conf.base = map(float, cnfIn.get('pr2Base', []))
        conf.torso = map(float, cnfIn.get('pr2Torso', [])) 
        conf.left_joints = map(float, cnfIn.get('pr2LeftArm', []))
        conf.left_grip = map(float, cnfIn.get('pr2LeftGripper', []))
        conf.right_joints = map(float, cnfIn.get('pr2RightArm', []))
        conf.rght_grip = map(float, cnfIn.get('pr2RightGripper', []))
        # conf.head = map(float, cnfIn.get('pr2Head', []))
        conf.head = []

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
    def __init__(self, world, probs):
        pass
    # dispatch on the operators...
    def executePrim(self, op, params):
        pass
    def executePath(self, path):
        pass
    def executeMove(self, op, params):
        pass
    def executeLookAtHand(self, op, params):
        pass
    def executeLookAt(self, op, params):
        pass
    def executePick(self, op, params):
        pass
    def executePlace(self, op, params):
        pass

class PointCloud:
    def __init__(self, matrix, eye):
        self.pointMatrix = matrix
        self.eye = eye

# obsTargets is dict: {name:ObjPlaceB or None, ...}
def getTables(world, obsTargets, pointCloud):
    tables = []
    exclude = []
    # A zone of interest
    zone = shapes.BoxAligned(np.array([(0, -2, 0), (3, 2, 1.5)]), None)
    for objName in obsTargets:
        if 'table' in objName:
            placeB = obsTargets[objName]
            if not placeB or pointCloud.eye.applyToPoint(table.origin().point()).x > 0):
                startTime = time.time()
                tableShape = world.getObjectShapeAtOrigin(objName)
                score, detection = tab.bestTable(zone, tableShape,
                                                 pointCloud, exclude,
                                                 angles = tab.anglesList(30),
                                                 zthr = 0.05,
                                                 debug=debug('getTables'))
                print 'Table detection', detection, 'with score', score
                print 'Running time for table detections =',  time.time() - startTime
                if detection:
                    tables.append((score, detection))
                    exclude.append(detection)
                    detection.draw('MAP', 'blue')
                    debugMsg('getTables', 'Detection for table=%s'%objName)
    return tables

def getPointCloud(resolution):
    rospy.wait_for_service('point_cloud')
    debugMsg('getPointCloud', 'Got Cloud?')
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
        points = np.zeros((4, len(response.cloud.points)), dtype=np.float64)
        for i, pt in enumerate(response.cloud.points):
            points[:, i] = (p.x, p.y, p.z, 1.0)
        return PointCloud(points, TransformFromROSMsg(response.cloud.eye))
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
