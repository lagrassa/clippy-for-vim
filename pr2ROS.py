import numpy as np
import random
import math
import pdb
import copy
import time
import transformations as transf
import util
import planGlobals as glob
import windowManager3D as wm

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
    
def getCloudPoints(resolution):
    rospy.wait_for_service('point_cloud')
    #raw_input('Got Cloud?')
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
        return ([util.Point(p.x, p.y, p.z) for p in response.cloud.points],
                response.cloud.eye)
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e
        raw_input('Continue?')
        return []

def makeROSPolygon(obj, dz=0):
    points = []
    if not 'side' in obj.getBaseName():
        verts = obj.prims()[0].verts()
        z = obj.zRange()[1]
        # Either pick out the top z verts or move them to that z
        points = [gm.Point(v.x, v.y, v.z) for v in verts \
                  if abs(z - v.z) < 0.001] \
                  or \
                  [gm.Point(v.x, v.y, z+dz) for v in verts]
    else:
        verts = obj.prims()[0].allVerts()
        verts = thinOutVerts(verts, len(verts)/2)
        points = [gm.Point(v.x, v.y, v.z+dz) for v in verts]
    return gm.Polygon(points)

def thinOutVerts(verts, size):
    # print 'thinOut', verts, size
    if len(verts) <= size:
        return verts
    closestIndex = 0
    closestDist = 1e10
    for i in xrange(len(verts)-1):
        for j in xrange(i+1, len(verts)):
            dist = verts[i].distance(verts[j])
            if dist < closestDist:
                closestIndex = j
                closestDist = dist
    return thinOutVerts(verts[:closestIndex]+verts[closestIndex+1:],
                        size)

def makeROSPose2D(pose):
    pose = pose.pose()
    return gm.Pose2D(pose.x, pose.y, pose.theta)

def makeROSPose3D(pose):
    T = pose.matrix
    r = transf.quaternion_from_matrix(T)
    o = (T[0][3],T[1][3],T[2][3])
    ((px,py,pz),(x,y,z,w)) = (o, r)
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
    posList = [pos.x, pos.y, pos.z]
    quatList = [quat.x, quat.y, quat.z, quat.w]
    trans = transf.translation_matrix(posList)
    rot = transf.quaternion_matrix(quatList)
    return util.Transform(numpy.dot(trans, rot))

def transformPoly(poly, x, y, z, theta):
    def pt(coords):
        return gm.Point(coords[0], coords[1], coords[2])
    tr = numpy.dot(transf.translation_matrix([x, y, z]),
                   transf.rotation_matrix(theta, (0,0,1)))
    points = [pt(numpy.dot(tr, [p.x, p.y, p.z, 1])) for p in poly.points]
    return gm.Polygon(points)
