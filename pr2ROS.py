import numpy as np
import random
import math
import pdb
import copy
import time
import transformations as transf
import util
import globVars as glob

if glob.useROS:
    import rospy
    import pr2_hpn.msg
    import pr2_hpn.srv
    from pr2_hpn.srv import *
    import kinematics_msgs.msg
    import kinematics_msgs.srv
    from std_msgs.msg import Header
    from arm_navigation_msgs.msg import RobotState
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

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

