#!/usr/bin/env python
import roslib
roslib.load_manifest('hpn_redux')
import rospy
import threading
import tf
import tf.transformations as transf
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose2D
import actionlib
from base_trajectory_action.msg import BaseTrajectoryAction, BaseTrajectoryGoal
from actionlib import SimpleActionClient
from actionlib_msgs.msg import *
from pr2_controllers_msgs.msg import *
from pr2_gripper_sensor_msgs.msg import *
from sensor_msgs.msg import JointState
from arm_navigation_msgs.msg import RobotState
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory
from hpn_redux.srv import *
from hpn_redux.msg import *
from sensor_msgs.msg import JointState
import pr2_hand_sensor_server
import math
import numpy
# from arm_navigation_msgs.srv import FilterJointTrajectory, FilterJointTrajectoryRequest

class Simple_Controller():
    base_rot_vel = 0.40
    base_rot_thr = 0.01
    base_trans_vel = 0.20
    base_trans_thr = 0.02
    torso_vel = 1.0
    def __init__(self, verbose = False):
        self.verbose = verbose
        self.grabbing = False
        self.pressureThreshold = 100
        self.max_gripper_effort = 50.0  # 50=gentle, -1=max_effort
        rospy.init_node('my_controller_server')

        self.lock = threading.Lock()
        self.bThread = threading.Thread(target=self.base_listener)
        self.bThread.start()
        print 'Listening for base pose messages'
        self.base_pose = None
        self.base_pose_count = 0
	
        self.tf_listener = tf.TransformListener()
        self.rate = rospy.Rate(30) # 30hz
	self.baseClient = SimpleActionClient("base_trajectory_action", BaseTrajectoryAction)
        self.baseClient.wait_for_server()
        self.jointClient = rospy.ServiceProxy('hpn_joints', GetJoints)
        self.jointClient.wait_for_service()
        self.torsoClient = SimpleActionClient('torso_controller/position_joint_action',
                                                        SingleJointPositionAction)
        self.torsoClient.wait_for_server()
        self.headClient = SimpleActionClient('/head_traj_controller/point_head_action',
                                                       PointHeadAction)
        self.headClient.wait_for_server()

        jointNames = ["_shoulder_pan_joint",
                      "_shoulder_lift_joint",
                      "_upper_arm_roll_joint",
                      "_elbow_flex_joint",
                      "_forearm_roll_joint",
                      "_wrist_flex_joint",
                      "_wrist_roll_joint"]
        self.armClient = {}
        self.jointNames = {}
        self.gripClient = {}
        self.pressure_listener = {}
        self.gripper_action_client = {}
        self.gripper_find_contact_action_client = {}
        self.gripper_grab_action_client = {}
        self.gripper_event_detector_action_client = {}
#        self.filter_trajectory_service = rospy.ServiceProxy\
#            ('/trajectory_filter_unnormalizer/filter_trajectory', FilterJointTrajectory)
#        rospy.loginfo('Waiting for trajectory filter service')
#        self.filter_trajectory_service.wait_for_service()
        SA = SimpleActionClient #  for shorter lines
        for arm in ('l', 'r'):
            self.pressure_listener[arm] = pr2_hand_sensor_server.FingertipSensorListener(arm, self.pressureThreshold)
            
            joint_trajectory_action_name = arm+'_arm_controller/joint_trajectory_action'
            gripper_action_name = arm+'_gripper_sensor_controller/gripper_action'
            gripper_find_contact_action_name = arm+'_gripper_sensor_controller/find_contact'
            gripper_grab_action_name = arm+'_gripper_sensor_controller/grab'
            gripper_event_detector_action_name = arm+'_gripper_sensor_controller/event_detector'
            
            self.armClient[arm] = SA(joint_trajectory_action_name, JointTrajectoryAction)
            self.jointNames[arm]=[arm+x for x in jointNames]

            self.gripper_action_client[arm] = SA(gripper_action_name, Pr2GripperCommandAction)
            self.gripper_find_contact_action_client[arm] = SA(gripper_find_contact_action_name, \
                                                           PR2GripperFindContactAction)
            self.gripper_grab_action_client[arm] = SA(gripper_grab_action_name, \
                                                   PR2GripperGrabAction)
            self.gripper_event_detector_action_client[arm] = SA(gripper_event_detector_action_name, \
                                                             PR2GripperEventDetectorAction)
            self.wait_for_action_server(self.armClient[arm], joint_trajectory_action_name)
            self.wait_for_action_server(self.gripper_action_client[arm], gripper_action_name)
            self.wait_for_action_server(self.gripper_find_contact_action_client[arm], \
                                       gripper_find_contact_action_name)
            self.wait_for_action_server(self.gripper_grab_action_client[arm], gripper_grab_action_name)
            self.wait_for_action_server(self.gripper_event_detector_action_client[arm],
                                        gripper_event_detector_action_name)

        self.initTransform = numpy.dot(transf.translation_matrix((0., 0., 0.)),
                                       transf.rotation_matrix(0., (0.,0.,1.)))
        self.initTransformInverse = transf.inverse_matrix(self.initTransform)


        # start services(s)
        rospy.Service('pr2_goto_configuration', GoToConf, self.handleGoToConf)
        rospy.loginfo("Ready to control the robot.")

    #thread function: listen for Pose2D messages
    def base_listener(self):
        rospy.Subscriber('pose2D', Pose2D, self.base_callback)
        rospy.spin()

    #callback function: when a pose2D message arrives, save the pose
    def base_callback(self, msg):
        self.lock.acquire()
        self.base_pose = msg
        self.base_pose_count += 1
        self.lock.release()
        if self.base_pose_count%100 == 0:
            rospy.loginfo("Base pose: %s"%str(msg))

    def handleGoToConf(self, req):
        rospy.loginfo("GoToConf: %s"%str(req))
        success = True
        op = req.operation

        self.speedFactor = req.speed
        arm = req.inputConf.arm         # 'l', 'r', 'both'
        # all represented as lists, so empty list is meaningful
        base = req.inputConf.base       # [x,y,theta]
        torso = req.inputConf.torso     # [height]
        left_joints = req.inputConf.left_joints   # list of 7 joint values
        right_joints = req.inputConf.right_joints # list of 7 joint values
        left_grip = req.inputConf.left_grip       # [width]
        right_grip = req.inputConf.right_grip     # [width]
        head = req.inputConf.head       # [...]
        
        if op == 'reset':
            if base:                    # reset so base is at desired
                if self.verbose:
                    print 'Desired base conf:', base
                    print 'Resetting from:'
                    print self.initTransform
                baseNow = self.getBaseTransform()
                baseNowInverse = transf.inverse_matrix(baseNow)
                trans = transf.translation_matrix((base[0], base[1], 0.0))
                rot = transf.rotation_matrix(base[2], (0,0,1))
                desired = numpy.dot(trans, rot)
                desiredInverse = transf.inverse_matrix(desired)
                if self.verbose:
                    print 'Desired matrix pose:'
                    print desired
                self.initTransform = numpy.dot(baseNow, desiredInverse)
                self.initTransformInverse = transf.inverse_matrix(self.initTransform)
                if self.verbose:
                    print 'Resetting init to:'
                    print self.initTransform
                    print 'Current pose:'
                    print self.getBasePose()
            else:                       # reset to base is at origin
                self.initTransform = self.getBaseTransform()
            self.initTransformInverse = transf.inverse_matrix(self.initTransform)
            source, result = 'all', 'reset'
        elif op == 'resetForce':
            self.pressure_listener[arm].set_thresholds()
            source, result = 'all', 'resetForce'
        elif op == 'move':
            source, result = self.moveRobotToConf(base, torso, head,
                                                  right_joints, right_grip,
                                                  left_joints, left_grip)
        elif op == 'moveGuarded':
            source, result = self.moveRobotToConf(base, torso, head,
                                                  right_joints, right_grip,
                                                  left_joints, left_grip,
						  guarded = True)
        elif op == 'closeGuarded':
            source, result = self.moveGripperToContact(arm, 'either')
        elif op == 'close':
            source, result = self.moveGripperToContact(arm, 'both')
        elif op == 'grab':
            source, result = self.start_gripper_grab(arm)
        else:
            raise Exception, 'Unknown op=%s'%str(op)

        return GoToConfResponse(result=result, source=source,
                                resultConf=self.reportConf('Final conf'))

    def getBaseTransform(self):
        now = rospy.Time()
        self.tf_listener.waitForTransform('/base_link','/odom_combined', now,
                                          rospy.Duration(60))
        pos, quat = self.tf_listener.lookupTransform('/odom_combined','/base_link', now)
        trans = transf.translation_matrix(pos)
        rot = transf.quaternion_matrix(quat)
        return numpy.dot(trans, rot)

    def getBaseTransformNew(self, count=0):
        if self.base_pose:
            bp = self.base_pose
            return numpy.dot(transf.translation_matrix((bp.x, bp.y, 0.)),
                             transf.rotation_matrix(bp.theta, (0.,0.,1.)))
        elif count < 100:
            self.rate.sleep()
            return self.getBaseTransform(count=count+1)
        else:
            raise Exception, 'No base pose information'

    def getBasePose(self):
        now = self.getBaseTransform()
        rel = numpy.dot(self.initTransformInverse, now)
        ans = [rel[0,3], rel[1,3], transf.euler_from_matrix(rel)[2]]
        return ans

    def getBaseZ(self):
        z = self.getBaseTransform()[2,3]
        return z
    
    def moveBaseToPose(self, pose, path = []):
        rospy.loginfo('moving base to '+str(pose))
	goal = BaseTrajectoryGoal()
    	goal.world_frame = "odom_combined"
    	goal.robot_frame = "base_footprint"
    	goal.linear_velocity = self.base_trans_vel
    	goal.angular_velocity = self.base_rot_vel
    	goal.angular_error = self.base_rot_thr
    	goal.linear_error = self.base_trans_thr
	goal.trajectory = []
        if path:
	    goal.trajectory = [Pose2D(x=x, y=y, theta=a) for (x,y,a) in path]
	if pose:
	    (x, y, a) = pose
	    goal.trajectory.append(Pose2D(x=x, y=y, theta=a))
    	rospy.loginfo('Sending goal')
    	self.baseClient.send_goal_and_wait(goal)
        return 'goal'

    def moveTorsoToHeight(self, z):
        rospy.loginfo('moving torso to '+str(z))
        goal = SingleJointPositionGoal(position = z, max_velocity = self.torso_vel)
        self.torsoClient.send_goal(goal)
        finished_within_time = self.torsoClient.wait_for_result(rospy.Duration(60))
        if not finished_within_time:
            rospy.logerr('Torso move did not finish on time')
            self.torsoClient.cancel_goal()
            return 'timeout'
        state = self.torsoClient.get_state()
        if state == GoalStatus.SUCCEEDED:
            rospy.loginfo("Success on torso move")
            return 'goal'
        else:
            rospy.logerr('Torso status is: %s'%str(state))
            return 'fail'

    def pointHeadAt(self, point):  # point is a 3 list, relative to robot base
        rospy.loginfo('point head at '+str(point))
        # map into base coordinates, if point is in absoltute coords
        # (cx, cy, ca) = self.getBasePose()
        # if self.verbose:
        #     print 'base pose', cx, cy, ca
        # sa = math.sin(ca)
        # ca = math.cos(ca)
        # dx = (point[0] - cx)
        # dy = (point[1] - cy)
        # dx_rel =  ca*dx + sa*dy
        # dy_rel = -sa*dx + ca*dy
        # dz_rel = point[2] - self.getBaseZ()
        # newPoint = [dx_rel, dy_rel, dz_rel]
        # rospy.loginfo('newPoint head at '+str(newPoint))
        goal = PointHeadGoal()
        goal.target.header.frame_id = 'base_link' # point is rel base
        goal.target.point.x = point[0]
        goal.target.point.y = point[1]
        goal.target.point.z = point[2]
        goal.pointing_frame = 'head_tilt_link'
        goal.pointing_axis.x = 1.0
        goal.pointing_axis.y = 0.0
        goal.pointing_axis.z = 0.0
        goal.min_duration = rospy.Duration(1.0)
        self.headClient.send_goal(goal)
        finished_within_time = self.headClient.wait_for_result(rospy.Duration(10))
        if not finished_within_time:
            rospy.logerr('Head move did not finish on time')
            self.headClient.cancel_goal()
            result = 'timeout'
        state = self.headClient.get_state()
        if state == GoalStatus.SUCCEEDED:
            rospy.loginfo("Success on head motion")
            result = 'goal'
        else:
            rospy.logerr("Head status is: %s"%str(state))
            result = 'fail'
        return result

    def moveArmToConf(self, arm, conf, vel = 0.25, guarded = False, path = None):
        rospy.loginfo('moving arm '+str(arm)+' to '+str(conf))
        rospy.loginfo('moving arm '+str(arm)+' via '+str(path))
        goal = JointTrajectoryGoal()
        if path and path.points:
#            reqT = FilterJointTrajectoryRequest()
#            reqT.trajectory = path
#            reqT.allowed_time = rospy.Duration(trajectoryTime(path.points,
#                                                              vel*self.speedFactor))
#            rospy.loginfo('filter_trajectory input time ='+str(reqT.allowed_time))
#            reqT.allowed_time = rospy.Duration(5.0)
#            reqT.start_state = RobotState()
#            reqT.start_state.joint_state = JointState()
#            joints = self.jointClient(arm)
#            reqT.start_state.joint_state.name = joints.name
#            reqT.start_state.joint_state.position = joints.position
#            rospy.loginfo('calling filter trajectory ')
#            resT = self.filter_trajectory_service(reqT)
#            rospy.loginfo('returned from filter_trajectory')
#            for point in resT.trajectory.points:
#                rospy.loginfo('filtered point positions:'+str(point.positions))
#                rospy.loginfo('filtered point velocities:'+str(point.velocities))
#            goal.trajectory = resT.trajectory

            (goal.trajectory, max_time) = setVelocities(path, vel*self.speedFactor)
        else:
            # single final conf, stop at the end
            jointvelocities=[0]*7
            jointaccelerations=[0]*7
            angles = self.get_arm_angles(arm)
            max_time = max([max(0.01, abs(fixAnglePlusMinusPi(y - x))/(vel*self.speedFactor)) \
                            for (x,y) in zip(conf, angles)])
            max_time = max(0.0, min(5.0, max_time))
            time_for_motion=rospy.Duration(max_time)
            rospy.loginfo('max_time='+str(time_for_motion)+ ', '+str(max_time))
            point = JointTrajectoryPoint(conf, jointvelocities,
                                         jointaccelerations, time_for_motion)
            goal.trajectory.joint_names = self.jointNames[arm]
            goal.trajectory.points = [point,]
        if guarded: 
            self.start_gripper_event_detector(arm)
        rospy.loginfo('Sending goal')
        goal.trajectory.header.stamp = rospy.get_rostime()
        self.armClient[arm].send_goal(goal)
        rospy.loginfo('returned from sending goal')
        trigger = self.waitForArmEvent(arm, max_time, guarded)
        rospy.loginfo('state='+str(self.armClient[arm].get_state()))
        return trigger

    def waitForArmEvent(self, arm, action_time, guarded):
        start_time = rospy.get_rostime()
        trigger = False
        while not rospy.is_shutdown():
            if guarded:
                state = self.get_gripper_event_detector_state(arm)
                (ltip, lpad, rtip, rpad) = self.check_all_contacts(arm)                
                if (ltip and rtip): trigger = 'LR_tip'
                elif ltip: trigger = 'L_tip'
                elif rtip: trigger = 'R_tip'
                elif (lpad and rpad): trigger = 'LR_pad'
                elif lpad: trigger = 'L_pad'
                elif rpad: trigger = 'R_pad'
                elif state not in [GoalStatus.ACTIVE, GoalStatus.PENDING]:
                    trigger = 'Acc'
                    
                if trigger:
                    self.armClient[arm].cancel_goal()
                    self.moveArmToConf(arm, self.get_arm_angles(arm), guarded = False) # freeze the arm
                    rospy.loginfo("guarded move saw event: "+trigger+", stopping the arm")
                    return trigger
            state = self.armClient[arm].get_state()
            if state == GoalStatus.SUCCEEDED: 
                trigger = 'goal'
                return trigger
            if rospy.get_rostime() - start_time > rospy.Duration(action_time + 10):
                rospy.logerr("guarded move timed out, stopping the arm")
                self.armClient[arm].cancel_goal()
                self.moveArmToConf(arm, self.get_arm_angles(arm), guarded = False) # freeze the arm
                trigger = 'timeout'
                return trigger
        return trigger

    def describeGripperResult(self, left, right):
        if left and right: return 'LR_pad'
        elif left: return 'L_pad'
        elif right: return 'R_pad'
        else: return 'none'

    def waitForGripperEvent(self, arm, action_time, condition):
        start_time = rospy.get_rostime()
        while not rospy.is_shutdown():
            (left, right) = self.check_closing_contacts(arm)
            if (condition == 'left' and left) or \
               (condition == 'right' and right) or \
               (condition == 'either' and (left or right)) or \
               (condition == 'both' and (left and right)):
                self.gripper_action_client[arm].cancel_goal()
                self.moveGripperToWidth(arm, self.gripper_opening(arm)) # freeze the gripper
                print condition, left, right
                rospy.loginfo(arm+' grip condition achieved')
                return self.describeGripperResult(left, right)
            state = self.gripper_action_client[arm].get_state()
            if state == GoalStatus.SUCCEEDED:
                rospy.loginfo(arm+' gripper move reached end')
                return self.describeGripperResult(left, right)
            if rospy.get_rostime() - start_time > rospy.Duration(action_time + 5):
                self.gripper_action_client[arm].cancel_goal()
                self.moveGripperToWidth(arm, self.gripper_opening(arm)) # freeze the gripper
                rospy.logerr("gripper move timed out")
                return self.describeGripperResult(left, right)

    def moveGripperToWidth(self, arm, width):
        if self.verbose:
            print 'moving gripper to', width
        self.clearGrab(arm)
        goal = Pr2GripperCommandGoal()
        goal.command.position = width
        goal.command.max_effort = self.max_gripper_effort
        self.gripper_action_client[arm].send_goal(goal)
        finished_within_time = self.gripper_action_client[arm].wait_for_result(rospy.Duration(5))
        if not finished_within_time:
            rospy.logerr('Gripper move did not finish on time')
            self.gripper_action_client[arm].cancel_goal()
            return 'timeout'
        state = self.gripper_action_client[arm].get_state()
        if state == GoalStatus.SUCCEEDED:
            rospy.loginfo('Success on '+ arm+' gripper move')
            return 'goal'
        else:
            rospy.logerr('Gripper status is %s'%str(state))
            return 'none'

    # tell the gripper to close until contact (on one or both finger pads)
    # contacts_desired is "both", "left", "right", or "either"
    def moveGripperToContactEvent(self, arm, contacts_desired, zero_fingertips = 0, timeout = 12.):
        if self.verbose:
            print 'moving gripper to contact:', contacts_desired
        self.clearGrab(arm)
        source = 'l_grip' if arm == 'l' else 'r_grip'
        goal = PR2GripperFindContactGoal()
        contacts_desired_dict = {"both":goal.command.BOTH, "left":goal.command.LEFT,
                                 "right":goal.command.RIGHT, "either":goal.command.EITHER}
        goal.command.contact_conditions = contacts_desired_dict[contacts_desired]
        print 'goal condition =', goal.command.contact_conditions
        goal.command.zero_fingertip_sensors = zero_fingertips
        rospy.loginfo("Sending find contact goal")
        self.gripper_find_contact_action_client[arm].send_goal(goal)
        # Wait for the state to reach the desired state
        finished_on_time = self.gripper_find_contact_action_client[arm].wait_for_result(rospy.Duration(timeout))
        if not finished_on_time:
            self.gripper_find_contact_action_client[arm].cancel_goal()
            rospy.logerr("Gripper didn't see the desired number of contacts while closing in time")
            return source, 'timeout'
        state = self.gripper_find_contact_action_client[arm].get_state()
        if state == GoalStatus.SUCCEEDED:
            rospy.loginfo("Find contact goal succeeded")
            return source, self.describeGripperResult(result.data.left_fingertip_pad_contact,
                                                      result.data.right_fingertip_pad_contact)
        rospy.loginfo("Find contact goal failed: status=%s"%str(state))
        return source, 'fail'

    def clearGrab(self, arm):
        if not self.grabbing: return True
        state = self.gripper_grab_action_client[arm].get_state()
        if state == GoalStatus.ACTIVE:
            rospy.loginfo('Clearing active grab!')
            self.gripper_grab_action_client[arm].cancel_goal()
            self.grabbing = False

    def moveGripperToContact(self, arm, contacts_desired, zero_fingertips = 0, timeout = 5.):
        if self.verbose:
            print 'moving gripper to contact:', contacts_desired
        self.clearGrab(arm)
        source = 'l_grip' if arm == 'l' else 'r_grip'
        goal = Pr2GripperCommandGoal()
        goal.command.position = 0.02
        goal.command.max_effort = self.max_gripper_effort
        self.gripper_action_client[arm].send_goal(goal)
        return source, self.waitForGripperEvent(arm, timeout, contacts_desired)

    def get_arm_angles(self, arm):
        joints = self.jointClient(arm)
        indices = [joints.name.index(name) for name in self.jointNames[arm]]
        return [joints.position[i] for i in indices]

    def gripper_opening(self, arm):
        joints = self.jointClient(arm)
        return joints.position[joints.name.index(arm+'_gripper_joint')]

    def reportConf(self, message, arm='both'):
        if self.verbose:
            print message
            print 'base', zip(['x', 'y', 'theta'], self.getBasePose())
        joints = self.jointClient('l')
        torso = joints.position[joints.name.index('torso_lift_joint')]
        if self.verbose:
            print ('torso_lift_joint', torso)

        left_joints = self.get_arm_angles('l')
        right_joints = self.get_arm_angles('r')
        left_grip = self.gripper_opening('l')
        right_grip = self.gripper_opening('r')

        conf = Conf(arm    = 'both',
                    base   = self.getBasePose(),
                    torso  = [torso],
                    left_joints = left_joints,
                    right_joints = right_joints,
                    left_grip   = [left_grip],
                    right_grip   = [right_grip])
        return conf

    def moveRobotToConf(self, base, torso, head,
                        r_conf, r_grip, 
                        l_conf, l_grip,
                        guarded = False, armPath = None, basePath = []):
        self.reportConf('start configuration')
        if r_grip:
            result = self.moveGripperToWidth('r', r_grip[0])
            if not result == 'goal': return 'r_grip', result
        if l_grip:
            result = self.moveGripperToWidth('l', l_grip[0])
            if not result == 'goal': return 'l_grip', result
        if torso:
            result = self.moveTorsoToHeight(torso[0])
            if not result == 'goal': return 'torso', result
        if head:
            result = self.pointHeadAt(head)
            if not result == 'goal': return 'head', result
        rospy.loginfo('base '+str(base))
        if base:
            result = self.moveBaseToPose(base, path=basePath)
            if not result == 'goal': return 'base', result
        if r_conf:
            result = self.moveArmToConf('r', r_conf, guarded=guarded, path=armPath)
            if not result == 'goal': return 'r_arm', result
        if l_conf:
            result = self.moveArmToConf('l', l_conf, guarded=guarded, path=armPath)
            if not result == 'goal': return 'l_arm', result
        return 'all', 'goal'

    ## Adapted from Kaijen Hsiao's reactive_grasping

    # opens and closes the gripper to determine how hard to grasp the
    # object, then starts up slip servo mode
    def start_gripper_grab(self,  arm, hardness_gain = 0.03, timeout = 10.):
        source = 'l_grip' if arm == 'l' else 'r_grip'
        goal = PR2GripperGrabGoal()
        goal.command.hardness_gain = hardness_gain
        rospy.loginfo("starting slip controller grab")
        self.gripper_grab_action_client[arm].send_goal(goal)
        self.grabbing = True
        finished_on_time = self.gripper_grab_action_client[arm].wait_for_result(rospy.Duration(timeout))
        if not finished_on_time:
            self.gripper_grab_action_client[arm].cancel_goal()
            self.grabbing = False
            rospy.logerr("Gripper grab timed out")
            return source, 'timeout'
        state = self.gripper_grab_action_client[arm].get_state()
        if state == GoalStatus.SUCCEEDED:
            self.grabbing = False
            return source, 'goal'
        self.gripper_grab_action_client[arm].cancel_goal()
        self.grabbing = False
        return source, 'fail'

    # start up gripper event detector
    def start_gripper_event_detector(self, arm):
        goal = PR2GripperEventDetectorGoal()
        # use either acceleration or tip touch as a contact condition
        # goal.command.trigger_conditions = goal.command.FINGER_SIDE_IMPACT_OR_ACC
        # use only acceleration
        goal.command.trigger_conditions = goal.command.ACC
        goal.command.acceleration_trigger_magnitude = 4  #3.25 contact acceleration used to trigger 
        rospy.loginfo("Starting gripper event detector")
        self.gripper_event_detector_action_client[arm].send_goal(goal)

    # get the state from the gripper event detector
    def get_gripper_event_detector_state(self, arm):
        return self.gripper_event_detector_action_client[arm].get_state()

    # check for tip/side/back contact
    def check_guarded_move_contacts(self, arm):
        left_touching = right_touching = 0
        # regions are (tip, plus_z_side, neg_z_side, front, back)
        (l_regions_touching, r_regions_touching) = self.pressure_listener[arm].regions_touching()
        if any(l_regions_touching[0:3]) or l_regions_touching[4]:
            left_touching = 1
            rospy.loginfo("Saw left fingertip tip or side:"+str(l_regions_touching))
        if any(r_regions_touching[0:3] or r_regions_touching[4]):
            right_touching = 1
            rospy.loginfo("Saw right fingertip tip or side:"+str(r_regions_touching))
        return (left_touching, right_touching)

    # check for all contact
    def check_all_contacts(self, arm):
        left_pad_touching = right_pad_touching = 0
        left_tip_touching = right_tip_touching = 0
        # regions are (tip, plus_z_side, neg_z_side, front, back)
        (l_regions_touching, r_regions_touching) = self.pressure_listener[arm].regions_touching()
        if any(l_regions_touching[0:3]) or l_regions_touching[4]:
            left_tip_touching = 1
            rospy.loginfo("Saw left fingertip tip or side:"+str(l_regions_touching))
        if any(r_regions_touching[0:3] or r_regions_touching[4]):
            right_tip_touching = 1
            rospy.loginfo("Saw right fingertip tip or side:"+str(r_regions_touching))
        if l_regions_touching[3]:
            left_pad_touching = 1
            rospy.loginfo("Saw left pad:"+str(l_regions_touching))
        if r_regions_touching[3]:
            right_pad_touching = 1
            rospy.loginfo("Saw right pad:"+str(r_regions_touching))
        return (left_tip_touching, left_pad_touching, right_tip_touching, right_pad_touching)

    # check for pad contact
    def check_closing_contacts(self, arm):
        left_touching = right_touching = 0
        # regions are (tip, plus_z_side, neg_z_side, front, back)
        (l_regions_touching, r_regions_touching) = self.pressure_listener[arm].regions_touching()
        if l_regions_touching[3]:
            left_touching = 1
            rospy.loginfo("Saw left pad:"+str(l_regions_touching))
        if r_regions_touching[3]:
            right_touching = 1
            rospy.loginfo("Saw right pad:"+str(r_regions_touching))
        return (left_touching, right_touching)

    # wait for an action server to be ready
    def wait_for_action_server(self, client, name):
        while not rospy.is_shutdown():  
            rospy.loginfo("Waiting for %s to be there"%name)
            if client.wait_for_server(rospy.Duration(5.0)):
                break
        rospy.loginfo("%s found"%name)  

    # wait for a service to be ready
    def wait_for_service(self, name):
        while not rospy.is_shutdown():  
            rospy.loginfo("Waiting for %s to be there"%name)
            try:
                rospy.wait_for_service(name, 5.0)
            except rospy.ROSException:
                continue
            break
        rospy.loginfo("%s found"%name)  

def fixAnglePlusMinusPi(a):
    """
    A is an angle in radians;  return an equivalent angle between plus
    and minus pi
    """
    if a > math.pi:
        return fixAnglePlusMinusPi(a - 2.0* math.pi)
    elif a < -math.pi:
        return fixAnglePlusMinusPi(a + 2.0*math.pi)
    else:
        return a

def trajectoryTime(points, vel):
    total = 0.0
    for i in range(1,len(points)):
        total += max([abs(fixAnglePlusMinusPi(y - x)) \
                      for (x,y) in zip(points[i-1].positions, points[i].positions)])
    return min(max(total/vel, 0.01), 5.0)

def setVelocities(traj, vel):
    points = traj.points
    zero = [0.0 for p in points[0].positions]
    for p in points: p.accelerations = zero
    points[0].velocities = zero
    total = 0.0
    for i in range(1, len(points)):
        deltas = [fixAnglePlusMinusPi(y - x) \
                  for (x,y) in zip(points[i-1].positions, points[i].positions)]
        delta = max(map(abs, deltas))
        total += max(delta/vel, 0.01)
        point = points[i]
        point.time_from_start = rospy.Duration(total)
        point.velocities = [d/delta * vel for d in deltas]
        rospy.loginfo('hacked point positions:'+str(point.positions))
        rospy.loginfo('hacked point velocities:'+str(point.velocities))
        rospy.loginfo('hacked trajectory time (sec):'+str(total))
        rospy.loginfo('hacked point time:'+str(point.time_from_start))
    points[-1].velocities = zero
    return (traj, total)

###########################
if __name__ == '__main__':
    s = Simple_Controller()
    rospy.spin()



