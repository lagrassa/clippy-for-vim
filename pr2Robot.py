import pdb
import random
import math
import hu
import copy
import itertools
import transformations as transf
from transformations import quaternion_slerp
import numpy as np
from collections import deque
from collections import OrderedDict

import shapes
import cspace

import objects
from objects import *

from miscUtil import prettyString
import fbch

import planGlobals as glob
from traceFile import debugMsg, debug

from pr2IkPoses import ikTrans          # base poses for IK

from gjk import frameBBoxRad, chainBBoxes
from pr2InvKin import armInvKin
from geom import vertsBBox
import windowManager3D as wm

# # We need this for inverse kinematics
# from ctypes import *
# if glob.LINUX:
#     ik = CDLL(glob.libkinDir+"libkin.so.1.1")
# else:
#     ik = CDLL(glob.libkinDir+"libkin.dylib")
# ik.fkLeft.restype = None
# ik.fkRight.restype = None

Ident = hu.Transform(np.eye(4, dtype=np.float64)) # identity transform

def vec(str):
    return [float(x) for x in str.split()]

def transProd(lt):
    return hu.Transform(reduce(np.dot, lt))

# Force sensor offsets from Gustavo's URDF

# <origin rpy="0 -1.5707963267948966 0" xyz="0.0356 0 0"/>
# <origin rpy="0 0 1.0477302122478542" xyz="0 0 0"/>
# <origin rpy="0 1.5707963267948966 -1.2217304763960306" xyz="0 0 0"/>

r_forceSensorOffset = transProd([transf.translation_matrix(vec("0.0356 0 0")),
                                 transf.euler_matrix(*vec("0 -1.5707963267948966 0"), axes='sxyz'),
                                 transf.euler_matrix(*vec("0 0 1.0477302122478542"), axes='sxyz'),
                                 transf.euler_matrix(*vec("0 1.5707963267948966 -1.2217304763960306"), axes='sxyz')])
l_forceSensorOffset = Ident

# 'beige' for movies, but that doesn't show up against white background of 2D
# Python simulator.
pr2Color = 'gold'

pr2_torso_joints = ['base_footprint_joint',
                    'torso_lift_joint']

pr2_arm_joints = ['_shoulder_pan_joint',
                  '_shoulder_lift_joint',
                  '_upper_arm_roll_joint',
                  '_elbow_flex_joint',
                  '_forearm_roll_joint',
                  '_wrist_flex_joint',
                  '_wrist_roll_joint']

def armJointNames(arm, joint_names=pr2_arm_joints):
    return [arm+j for j in joint_names]

pr2_head_joints = ['head_pan_joint',
                   'head_tilt_joint',
                   'head_plate_frame_joint',
                   'head_mount_joint',
                   'head_mount_kinect_ir_joint',
                   'head_mount_kinect_ir_optical_frame_joint'
                   ]

def Ba(bb, **prop): return shapes.BoxAligned(np.array(bb), Ident, **prop)
def Sh(*args, **prop): return shapes.Shape(list(args), Ident, **prop)
dx = glob.baseGrowthX; dy = glob.baseGrowthY

# Small base and torso
pr2BaseLink = Sh(\
    Ba([(-0.33, -0.33, 0.0), (0.33, 0.33, 0.33)], name='base'),
    Ba([(-0.33, -0.33, 0.33), (0.0, 0.33, 1.0)], name='torso')
    )

# More conservative (bigger) base and torso
pr2BaseLinkGrown = Sh(\
    Ba([(-0.33-dx, -0.33-dy, 0.0), (0.33+dx, 0.33+dy, 0.33)], name='baseGrown'),
    Ba([(-0.33-dx, -0.33-dy, 0.0), (0.0, 0.33+dy, 1.5)], name='torsoGrown')
    )

# Connects to base, depends on torso height
pr2TorsoLinks = [\
    None,
    Sh(Ba([(-0.1, -0.1, 0.1), (0.1, 0.1, 0.3)], name='neck'))
    ]

# Connectz to torso, depens on head angles
pr2HeadLinks = [\
    None,
    # Sh(Ba([(0, -0.01, -0.01), (0.1, 0.01, 0.01)], name='sensorX'),
    #    Ba([(-0.01, 0, -0.01), (0.01, 0.1, 0.01)], name='sensorY'),
    #    Ba([(-0.01, -0.01, 0), (0.01, 0.01, 0.1)], name='sensorZ')),
    None,
    Sh(Ba([(-0.2, -0.1, -0.05), (0.1, 0.1, 0.05)], name='head')),
    None,
    None,
    Sh(Ba([(-0.1, -0.025, -0.05), (0.1, 0.025, 0.05)], name='kinect'))
       # a "beam" from the center of the kinect, along Z axis
       # Ba([(-0.01, -0.01, 0), (0.01, 0.01, 2)], name='sensorBeam')
    ]

def pr2ArmLinks(arm):
    angle = math.pi/2 if arm=='r' else -math.pi/2
    pose = hu.Transform(transf.rotation_matrix(angle, (1,0,0)))
    links = [\
        Sh(Ba([(-0.12, -0.12, -0.5), (0.24, 0.12, 0.1)], name='shoulder')),
        Sh(Ba([(0.12, -0.06, -0.08), (0.47, 0.06, 0.08)], name='upperArm')),
        None,
        Sh(Ba([(0.07, -0.06, -0.055), (0.18, 0.06, 0.03)], name='foreArm1'),
           Ba([(0.18, -0.06, -0.03), (0.36, 0.06, 0.03)], name='foreArm2')).applyTrans(pose),
        None, 
        None,
        None]
    return links

params = {'fingerLength' : 0.06,
          'fingerWidth' :  0.04,
          'fingerThick' :  0.02,
          'palmLength' : 0.09,
          'palmWidth' : 0.175,
          'palmThick' : 0.05,
          'gripperOffset': 0.04,
          'gripMax' :      0.08,
          'zRange': (0.0, 1.5),
          'zRangeHand': (0.35, 2.0),
          }

def pr2GripperLinks():
    palm_dx = params['palmLength']
    palm_dy = params['palmWidth']
    palm_dz = params['palmThick']
    fing_dx = params['fingerLength']
    fing_dy = params['fingerWidth']
    fing_dz = params['fingerThick']
    return [Sh(shapes.Box(palm_dx, palm_dy, palm_dz, Ident, name='palm')),
            Sh(shapes.Box(fing_dx, fing_dy, fing_dz, Ident, name='finger1')),
            Sh(shapes.Box(fing_dx, fing_dy, fing_dz, Ident, name='finger2'))]

def pr2GripperJoints(arm):
    o = params['gripperOffset']
    palm_dx = params['palmLength']
    fing_dx = params['fingerLength']
    fing_dy = params['fingerWidth']
    fing_dz = params['fingerThick']
    return [Rigid(arm+'_palm',
                  (r_forceSensorOffset if arm == 'r' else l_forceSensorOffset) * \
                  hu.Transform(transf.translation_matrix([o + palm_dx/2.,0.0,0.0])),
                  None, None),
            Prismatic(arm+'_finger1',
                      hu.Transform(transf.translation_matrix([palm_dx/2+fing_dx/2.,fing_dy/2.,0.0])),
                      (0.0, params['gripMax']/2.), (0.,1.,0)),
            Prismatic(arm+'_finger2',
                      hu.Transform(transf.translation_matrix([0.0,-fing_dy,0.0])),
                      (0.0, 0.08), (0.,-1.,0))]

# Transforms from wrist frame to hand frame
# This leaves X axis unchanged
left_gripperToolOffsetX = hu.Pose(0.18,0.0,0.0,0.0)
right_gripperToolOffsetX = hu.Transform(np.dot(r_forceSensorOffset.matrix,
                                                 left_gripperToolOffsetX.matrix))                          
# This rotates around the Y axis... so Z' points along X an X' points along -Z
left_gripperToolOffsetZ = hu.Transform(np.dot(left_gripperToolOffsetX.matrix,
                                               transf.rotation_matrix(math.pi/2,(0,1,0))))
right_gripperToolOffsetZ = hu.Transform(np.dot(right_gripperToolOffsetX.matrix,
                                                 transf.rotation_matrix(math.pi/2,(0,1,0))))

# Rotates wrist to grasp face frame
gFaceFrame = hu.Transform(np.array([(0.,1.,0.,0.18),
                                      (0.,0.,1.,0.),
                                      (1.,0.,0.,0.),
                                      (0.,0.,0.,1.)]))
gripperFaceFrame = {'left': gFaceFrame, 'right': r_forceSensorOffset*gFaceFrame}

# This behaves like a dictionary, except that it doesn't support side effects.
class JointConf:
    def __init__(self, conf, robot):
        global confIdnum
        self.conf = conf
        self.robot = robot
        self.items = None
        self.strStored = {True:None, False:None}
    def copy(self):
        return JointConf(self.conf.copy(), self.robot)
    def values(self):
        return self.conf.values()
    def keys(self):
        return self.conf.keys()
    def get(self, name, default = None):
        if name in self.conf:
            return self.conf[name]
        else:
            return default
    def set(self, name, value):
        assert value is None or isinstance(value, (list, tuple))
        c = self.copy()
        c.conf[name] = value
        return c
    def minimalConf(self, hand):
        armChainName = self.robot.armChainNames[hand]
        return (tuple(self.baseConf()), tuple(self.conf[armChainName]))
    # Abstract interface...
    def basePose(self):
        base = self.conf['pr2Base']
        return hu.Pose(base[0], base[1], 0.0, base[2])
    def setBaseConf(self, baseConf):
        assert isinstance(baseConf, (tuple, list)) and len(baseConf) == 3
        return self.set('pr2Base', list(baseConf))
    def baseConf(self):                 # (x, y, theta)
        return self.conf['pr2Base']
    def cartConf(self):
        return self.robot.forwardKin(self)
    def armShape(self, h, attached=None):
        return self.robot.armShape(self, h, attached)
    def handWorkspace(self):
        tz = self.conf['pr2Torso'][0]
        bb = ((0.5, -0.25, tz+0.2),(0.75, 0.25, tz+0.4)) # low z, so view cone extends
        return shapes.BoxAligned(np.array(bb), Ident).applyTrans(self.basePose())
    def placement(self, attached=None, getShapes=True):
        return self.robot.placement(self, attached=attached, getShapes=getShapes)[0]
    def placementMod(self, place, attached=None):
        return self.robot.placementMod(self, place, attached=attached)[0]    
    def placementAux(self, attached=None, getShapes=True):
        place, attachedParts, trans = self.robot.placementAux(self, attached=attached,
                                                              getShapes=getShapes)
        return place, attachedParts
    def placementModAux(self, place, attached=None, getShapes=True):
        place, attachedParts, trans = self.robot.placementModAux(self, place,
                                                                 attached=attached,
                                                                 getShapes=getShapes)
        return place, attachedParts
    def draw(self, win, color='black', attached=None):
        self.placement(attached=attached).draw(win, color)
    def prettyString(self, eq = True):
        if not eq:
            # If we don't need to maintain equality, just print the base
            if self.strStored[eq] is None:
                self.strStored[eq] = 'JointConf('+prettyString(self.conf['pr2Base'], eq)+')'
        else:
            if self.strStored[eq] is None:
                self.strStored[eq] = 'JointConf('+prettyString(self.conf, eq)+')'
        return self.strStored[eq]
    def prettyPrint(self, msg='Conf:'):
        print msg
        for key in sorted(self.conf.keys()):
            print '   ', key, prettyString(self.conf[key])
    def confItems(self):
        if not self.items:
            self.items = frozenset([(chain, tuple(self.conf[chain])) for chain in self.conf])
        return self.items
    def __str__(self):
        return 'JointConf('+str(self.conf)+')'
    def ss(self):
        return 'J%s'%(prettyString(self.conf['pr2Base']))
    def __getitem__(self, name):
        return self.conf[name]
    def __hash__(self):
        return hash(self.confItems())
    def __eq__(self, other):
        if not hasattr(other, 'conf'): return False
        return self.conf == other.conf
    def __ne__(self, other):
        if not hasattr(other, 'conf'): return True
        return not self.conf == other.conf

class CartConf(JointConf):
    def __init__(self, conf, robot):
        self.conf = conf
        self.robot = robot
        self.items = None

    def frameName(self, name):
        if 'Frame' in name or 'Gripper' in name or 'Torso' in name:
            return name
        else:
            return name + 'Frame'
    def get(self, name, default = None):
        name = self.frameName(name)
        if name in self.conf:
            return self.conf[self.frameName(name)]
        else:
            return default
    def set(self, name, value):
        c = self.copy()
        name = self.frameName(name)
        if 'Gripper' in name or 'Torso' in name:
            if isinstance(value, list):
                c.conf[name] = value
            else:
                c.conf[name] = [value]
        else:
            assert value is None or isinstance(value, hu.Transform)
            c.conf[name] = value
        return c
    def basePose(self):
        return self.conf['pr2Base'].pose()
    def confItems(self):
        if not self.items:
            vals = []
            for chain in self.conf:
                val = self.conf[chain]
                if isinstance(val, list):
                    vals.append(tuple(val))
                elif isinstance(val, hu.Transform):
                    vals.append(repr(val))
                else:
                    vals.append(val)
            self.items = frozenset(vals)
        return self.items
    def prettyPrint(self, msg='Cart Conf:'):
        print msg
        for key in sorted(self.conf.keys()):
            if isinstance(self.conf[key], hu.Transform):
                print '   ', key, '\n', self.conf[key].matrix
            else:
                print '   ', key, prettyString(self.conf[key])
    def copy(self):
        return CartConf(self.conf.copy(), self.robot)
    def __getitem__(self, name):
        return self.conf[self.frameName(name)]

rightStowAngles = [-2.1, 1.29, 0.000, -0.15, 0.000, -0.100, 0.000]
leftStowAngles = [2.1, 1.29, 0.000, -0.15, 0.000, -0.100, 0.000]
# This is a joint configuartion, specifying the joint angles for all the chains.
pr2Init = {'pr2Base':[0.0,0.0,0.0],
           'pr2Torso':[glob.torsoZ],
           'pr2LeftArm': leftStowAngles,
           'pr2LeftGripper': [0.02],
           'pr2RightArm': rightStowAngles,
           'pr2RightGripper': [0.02],
           'pr2Head': [0.0, 0.0]}
# In a cartesian configuration, we specify frames for base, left and right
# hands, and head.

pr2Chains = {}
def makePr2Chains(name, workspaceBounds, new=True):
    global pr2Chains
    if not new and name in pr2Chains:
         return pr2Chains[name]
    # Chain for base
    baseChain = Planar('pr2Base', 'root', pr2BaseLink, workspaceBounds)
    # Chain for torso
    torsoChain = Chain('pr2Torso', 'pr2Base_theta',
                       getUrdfJoints(pr2_torso_joints),
                       pr2TorsoLinks)
    # Chain for left arm
    leftArmChain = Chain('pr2LeftArm', 'torso_lift_joint',
                         getUrdfJoints(armJointNames('l', pr2_arm_joints)),
                         pr2ArmLinks('l'))
    # Chain for left gripper
    leftGripperChain = GripperChain('pr2LeftGripper', 'l_wrist_roll_joint',
                                    pr2GripperJoints('l'),
                                    pr2GripperLinks())
    # Chain for right arm
    rightArmChain = Chain('pr2RightArm', 'torso_lift_joint',
                         getUrdfJoints(armJointNames('r', pr2_arm_joints)),
                         pr2ArmLinks('r'))
    # Chain for right gripper
    rightGripperChain = GripperChain('pr2RightGripper', 'r_wrist_roll_joint',
                                     pr2GripperJoints('r'),
                                     pr2GripperLinks())
    # Chain for head
    headChain = Chain('pr2Head', 'torso_lift_joint',
                      getUrdfJoints(pr2_head_joints),
                      pr2HeadLinks)
    pr2Chains[name] = MultiChain(name,
                           [baseChain, torsoChain, leftArmChain, leftGripperChain,
                            rightArmChain, rightGripperChain, headChain])
    return pr2Chains[name]

# The radius is baseCovariance radius, angle in baseCovariance angle,
# reachPct is percentage of maximum reach of arm.
def makePr2ChainsShadow(name, workspaceBounds, radiusVar=0.0, angleVar=0.0, reachPct=1.0):
    sqrt2 = 2.0**0.5
    gr = radiusVar + sqrt2*0.33*angleVar
    pr2BaseLinkGrown = Sh(\
        Ba([(-0.33-gr, -0.33-gr, 0.0), (0.33+gr, 0.33+gr, 0.33)], name='baseGrown'),
        Ba([(-0.33-gr, -0.33-gr, 0.0), (0.0, 0.33+gr, 1.0)], name='torsoGrown')
        )
    def pr2ArmLinksGrown(arm):
        angle = math.pi/2 if arm=='r' else -math.pi/2
        pose = hu.Transform(transf.rotation_matrix(angle, (1,0,0)))
        gr1 = radiusVar + 0.43*reachPct*angleVar
        gr2 = radiusVar + 0.76*reachPct*angleVar
        gr3 = radiusVar + 1.05*reachPct*angleVar
        #print 'gr1', gr1, 'gr2', gr2, 'gr3', gr3
        # raw_input('arm growth factors')
        links = [\
            Sh(Ba([(-0.12-gr1, -0.12-gr1, -0.5), (0.24+gr1, 0.12+gr1, 0.1)], name='shoulder')),
            Sh(Ba([(0.12-gr2, -0.06-gr2, -0.08-gr2), (0.47+gr2, 0.06+gr2, 0.08+gr2)], name='upperArm')),
            None,
            Sh(Ba([(0.07-gr3, -0.06-gr3, -0.055-gr3), (0.18+gr3, 0.06+gr3, 0.03+gr3)], name='foreArm1'),
               Ba([(0.18-gr3, -0.06-gr3, -0.03-gr3), (0.36+gr3, 0.06+gr3, 0.03+gr3)], name='foreArm2')).applyTrans(pose),
            None, 
            None,
            None]
        return links
    def pr2GripperLinksGrown():
        palm_dx = params['palmLength']
        palm_dy = params['palmWidth']
        palm_dz = params['palmThick']
        fing_dx = params['fingerLength']
        fing_dy = params['fingerWidth']
        fing_dz = params['fingerThick']
        gr = radiusVar + 1.2*reachPct*angleVar
        return [Sh(shapes.Box(palm_dx+2*gr, palm_dy+2*gr, palm_dz+2*gr, Ident, name='palm')),
                Sh(shapes.Box(fing_dx+2*gr, fing_dy+2*gr, fing_dz+2*gr, Ident, name='finger1')),
                Sh(shapes.Box(fing_dx+2*gr, fing_dy+2*gr, fing_dz+2*gr, Ident, name='finger2'))]

    # Chain for base
    baseChain = Planar('pr2Base', 'root', pr2BaseLinkGrown, workspaceBounds)
    # Chain for torso
    torsoChain = Chain('pr2Torso', 'pr2Base_theta',
                       getUrdfJoints(pr2_torso_joints),
                       pr2TorsoLinks)     # unchanged
    # Chain for left arm
    leftArmChain = Chain('pr2LeftArm', 'torso_lift_joint',
                         getUrdfJoints(armJointNames('l', pr2_arm_joints)),
                         pr2ArmLinksGrown('l'))
    # Chain for left gripper
    leftGripperChain = GripperChain('pr2LeftGripper', 'l_wrist_roll_joint',
                                    pr2GripperJoints('l'),
                                    pr2GripperLinks()) # NB
    # Chain for right arm
    rightArmChain = Chain('pr2RightArm', 'torso_lift_joint',
                         getUrdfJoints(armJointNames('r', pr2_arm_joints)),
                         pr2ArmLinksGrown('r'))
    # Chain for right gripper
    rightGripperChain = GripperChain('pr2RightGripper', 'r_wrist_roll_joint',
                                     pr2GripperJoints('r'),
                                     pr2GripperLinks()) # NB
    # Chain for head
    headChain = Chain('pr2Head', 'torso_lift_joint',
                      getUrdfJoints(pr2_head_joints),
                      pr2HeadLinks)       # unchanged
    return MultiChain(name,
                      [baseChain, torsoChain, leftArmChain, leftGripperChain,
                       rightArmChain, rightGripperChain, headChain])

# These don't handle the rotation correctly -- what's the general form ??
def fliph(pose):
    params = list(pose.pose().xyztTuple())
    params[1] = -params[1]
    return hu.Pose(*params)

def flipv(pose):
    m = pose.matrix.copy()
    m[1,3] = -m[1,3]
    return hu.Transform(m)

def ground(pose):
    params = list(pose.xyztTuple())
    params[2] = 0.0
    return hu.Pose(*params)

# fkCount, fkCache, placeCount, placeCache
confCacheStats = [0, 0, 0, 0]

# Controls size of confCache - bigger cache leads to faster motion
# planning, but makes Python bigger, which can lead to swapping.
maxConfCacheSize = 150*10**3
# print '*** pr2Robot.maxConfCacheSize', maxConfCacheSize, '***'

# (additions, deletions)
confCacheUpdateStats = [0, 0]
def printStats():
    print 'maxConfCacheSize', maxConfCacheSize
    print 'confCacheStats = (fkCount, fkCache, placeCount, placeCache)\n', confCacheStats
    print 'confCacheUpdateStats = (additions, deletions)\n', confCacheUpdateStats

# This basically implements a Chain type interface, execpt for the wstate
# arguments to the methods.
robotIdCount = 0
class PR2:
    def __init__(self, name, chains, color = pr2Color):
        self.chains = chains
        self.color = color
        self.name = name
        # These are (focal, height, width, length, n)
        self.scanner = (0.3, 0.2, 0.2, 5, 30) # Kinect
        # These names encode the "order of actuation" used in interpolation
        self.chainNames = ['pr2LeftGripper', 'pr2RightGripper',
                           'pr2Torso', 'pr2Base',
                           'pr2LeftArm', 'pr2RightArm', 'pr2Head']
        self.moveChainNames = ['pr2LeftArm', 'pr2Base']
        self.armChainNames = {'left':'pr2LeftArm', 'right':'pr2RightArm'}
        self.gripperChainNames = {'left':'pr2LeftGripper', 'right':'pr2RightGripper'}
        self.wristFrameNames = {'left':'l_wrist_roll_joint', 'right':'r_wrist_roll_joint'}
        self.baseChainName = 'pr2Base'
        # This has the X axis pointing along fingers
        self.toolOffsetX = {'left': left_gripperToolOffsetX, 'right': right_gripperToolOffsetX}
        # This has the Z axis pointing along fingers (more traditional, as in ikFast)
        self.toolOffsetZ = {'left': left_gripperToolOffsetZ, 'right': right_gripperToolOffsetZ}
        self.nominalConf = None
        horizontalTrans, verticalTrans = ikTrans(level=2) # include more horizontal confs
        self.horizontalTrans = {'left': [p.inverse() for p in horizontalTrans],
                                'right': [fliph(p).inverse() for p in horizontalTrans]}
        self.verticalTrans = {'left': [p.inverse() for p in verticalTrans],
                                'right': [flipv(p).inverse() for p in verticalTrans]}
        self.confCache = {}
        self.confCacheKeys = deque([])  # in order of arrival
        self.compiledChains = compileChainFrames(self)
        if debug('PR2'): print 'New PR2!'
        global robotIdCount
        self.robotId = robotIdCount
        robotIdCount += 1

    def cacheReset(self):
        self.confCache = {}
        self.confCacheKeys = deque([])  # in order of arrival

    # The base transforms take into account any twist in the tool offset
    def potentialBasePosesGen(self, wrist, hand, n=None, complain=True):
        gripper = wrist*(left_gripperToolOffsetX if hand=='left' else right_gripperToolOffsetX)
        xAxisZ = gripper.matrix[2,0]
        if abs(xAxisZ) < 0.01:
            trs = self.horizontalTrans[hand]
        elif abs(xAxisZ + 1.0) < 0.01:
            trs = self.verticalTrans[hand]
        else:
            if complain:
                print 'gripper=\n', gripper.matrix
                raw_input('Illegal gripper trans for base pose')
            return
        for i, tr in enumerate(trs):
            if n and i > n: return
            # use largish zthr to compensate for twist in force sensor
            ans = wrist.compose(tr).pose(zthr = 0.1, fail=False) 
            if ans is None:
                if complain:
                    print 'gripper=\n', gripper.matrix
                    raw_input('Illegal gripper trans for base pose')
                return
            yield ground(ans)

    def fingerSupportFrame(self, hand, width):
        # The old way...
        # Origin is on the inside surface of the finger (at the far tip).
        # The -0.18 is from finger tip to the wrist  -- if using wrist frame
        # mat = np.dot(transf.euler_matrix(-math.pi/2, math.pi/2, 0.0, 'ryxz'),
        #              transf.translation_matrix([0.0, -0.18, -width/2]))

        # y points along finger approach, z points in closing direction
        # offset aligns with the grasp face.
        # This is global gripperFaceFrame offset to center object
        gripperFaceFrame_dy = hu.Transform(np.array([(0.,1.,0.,0.18),
                                                       (0.,0.,1.,-width/2),
                                                       (1.,0.,0.,0.),
                                                       (0.,0.,0.,1.)]))
        if hand == 'right':
            gripperFaceFrame_dy = r_forceSensorOffset * gripperFaceFrame_dy
        return gripperFaceFrame_dy

    def limits(self, chainNames = None):
        return itertools.chain(*[self.chains.chainsByName[name].limits()\
                                 for name in chainNames])

    def randomConf(self, moveChains=None):
        conf = JointConf({}, self)
        for chainName in (moveChains or self.moveChainNames):
            conf = conf.set(chainName,
                            self.chains.chainsByName[chainName].randomValues())
        return conf

    def baseShape(self, c):
        parts = dict([(o.name(), o) for o in c.placement().parts()])
        return parts['pr2Base']

    # This is useful to get the base shape when we don't yet have a conf
    def baseLinkShape(self, basePose=None):
        if basePose:
            return pr2BaseLink.applyTrans(basePose)
        else:
            return pr2BaseLink

    def armShape(self, c, hand, attached):
        parts = dict([(o.name(), o) for o in c.placement(attached=attached).parts()])
        armShapes = [parts[self.armChainNames[hand]],
                     parts[self.gripperChainNames[hand]]]
        if attached[hand]:
            armShapes.append(parts[attached[hand].name()])
        return shapes.Shape(armShapes, None)

    # attach the object (at its current pose) to gripper (at current conf)
    def attach(self, objectPlace, wstate, hand='left'):
        conf = wstate.robotConf
        cartConf = self.forwardKin(conf)
        frame = cartConf[self.armChainNames[hand]]
        obj = objectPlace.applyTrans(frame.inverse())
        wstate.attached[hand] = obj

    # attach object to gripper links, object expressed relative to wrist
    def attachRel(self, objectPlace, wstate, hand='left'):
        wstate.attached[hand] = objectPlace

    # detach and return the object (at its current pose) from gripper (at current conf)
    def detach(self, wstate, hand='left'):
        obj = wstate.attached[hand]
        if obj:
            conf = wstate.robotConf
            cartConf = self.forwardKin(conf)
            frame = cartConf[self.armChainNames[hand]]
            wstate.attached[hand] = None
            return obj.applyTrans(frame)
        else:
            raw_input('Attempt to detach, but no object is attached')

    # Just detach the object, don't return it.
    def detachRel(self, wstate, hand='left'):
        obj = wstate.attached[hand]
        if obj:
            wstate.attached[hand] = None
        else:
            assert None, 'Attempt to detach, but no object is attached'

    def attachedObj(self, wstate, hand='left'):
        obj = wstate.attached[hand]
        if obj:
            conf = wstate.robotConf
            cartConf = self.forwardKin(conf)
            frame = cartConf[self.armChainNames[hand]]
            return obj.applyTrans(frame)
        else:
            return None

    def confFromBaseAndWrist(self, basePose, hand, wrist,
                             defaultConf, counts=None):
        cart = CartConf({'pr2BaseFrame': basePose,
                         'pr2Torso':[glob.torsoZ]}, self)
        if hand == 'left':
            cart.conf['pr2LeftArmFrame'] = wrist 
            cart.conf['pr2LeftGripper'] = [0.08] # !! pick better value
        else:
            cart.conf['pr2RightArmFrame'] = wrist 
            cart.conf['pr2RightGripper'] = [0.08]
        # Check inverse kinematics
        conf = self.inverseKin(cart)
        if None in conf.values():
            debugMsg('potentialGraspConfsLose', 'invkin failure')
            if counts: counts[0] += 1       # kin failure
            return
        # Copy the other arm from pbs
        if hand == 'left':
            conf.conf['pr2RightArm'] = defaultConf['pr2RightArm']
            conf.conf['pr2RightGripper'] = [0.08]
        else:
            conf.conf['pr2LeftArm'] = defaultConf['pr2LeftArm']
            conf.conf['pr2LeftGripper'] = [0.08]
        return conf

    def placement(self, conf, wstate=None, getShapes=True, attached=None):
        place, attachedParts, trans = self.placementAux(conf, wstate, getShapes, attached)
        if attached and getShapes:
            return shapes.Shape(place.parts() + [x for x in attachedParts.values() if x],
                                place.origin(),
                                name=place.name()), trans
        else:
            return place, trans

    def placementMod(self, conf, place, wstate=None, getShapes=True, attached=None):
        place, attachedParts, trans = self.placementModAux(conf, place, wstate, getShapes, attached)
        if attached and getShapes:
            return shapes.Shape(place.parts() + [x for x in attachedParts.values() if x],
                                place.origin(),
                                name=place.name()), trans
        else:
            return place, trans

    def updateConfCache(self, key, value):
        while len(self.confCacheKeys) > maxConfCacheSize:
            confCacheUpdateStats[1] += 1
            oldKey = self.confCacheKeys.popleft()
            del(self.confCache[oldKey])
        confCacheUpdateStats[0] += 1
        self.confCacheKeys.append(key)
        self.confCache[key] = value

    def placementAux(self, conf, wstate=None, getShapes=True, attached=None):
        # The placement is relative to the state in some world (provides the base frame)
        # Returns a Shape object and a dictionary of frames for each sub-chain.
        frame = wstate.getFrame(self.chains.baseFname) if wstate else Ident
        shapeChains = getShapes
        key = (conf, frame, True if getShapes==True else tuple(getShapes))
        confCacheStats[0 if not getShapes else 2] += 1
        # confCache = (fkCount, fkCache, placeCount, placeCache)
        if key in self.confCache:
            confCacheStats[1 if not getShapes else 3] += 1
            place, trans = self.confCache[key]
        else:
            place, trans = self.chains.placement(frame, conf, getShapes=shapeChains)
            self.updateConfCache(key, (place, trans))
        attachedParts = {'left':None, 'right':None}
        if attached:
            for hand in ('left', 'right'):
                if attached[hand]:
                    attachedParts[hand] = attached[hand].applyTrans(trans[self.wristFrameNames[hand]])
        return place, attachedParts, trans

    def placementModAux(self, conf, place, wstate=None,
                          getShapes=True, attached=None):
        # The placement is relative to the state in some world (provides the base frame)
        # Returns a Shape object and a dictionary of frames for each sub-chain.
        frame = wstate.getFrame(self.chains.baseFname) if wstate else Ident
        shapeChains = getShapes
        place, trans = self.chains.placementMod(frame, conf, place, getShapes=shapeChains)
        attachedParts = {'left':None, 'right':None}
        if attached:
            for hand in ('left', 'right'):
                if attached[hand]:
                    attachedParts[hand] = attached[hand].applyTrans(trans[self.wristFrameNames[hand]])
        return place, attachedParts, trans

    def safeConf(self, conf, wstate, showCollisions = False):
        robotPlace, frames = self.placement(conf, wstate=wstate, attached=wstate.attached)
        # Check gripper collision with base
        base = next((x for x in robotPlace.parts() if x.name() == 'pr2Base'), None)
        for gname in ['pr2LeftGripper', 'pr2RightGripper']:
            gripper = next((x for x in robotPlace.parts() if x.name() == gname), None)
            if gripper and gripper.collides(base):
                if showCollisions:
                    print 'Collision of', gname, 'with base'
                return False
        # Check collisions with other objects
        for name, objPlace in wstate.objectPlaces.items():
            if not wstate.objectProps.get(name, {}).get('solid', True): continue
            if robotPlace.collides(objPlace):
                if showCollisions:
                    print 'Collision with', objPlace.name()
                    objPlace.draw(glob.stderrWindow)
                return False
        return True

    def obstacleCollisions(self, conf, static, transient,
                             wstate=None, showCollisions = False):
        robotPlace, frames = self.placement(conf, wstate=wstate, attached=wstate.attached)
        # Check gripper collision with base
        base = next((x for x in robotPlace.parts() if x.name() == 'pr2Base'), None)
        for gname in ['pr2LeftGripper', 'pr2RightGripper']:
            gripper = next((x for x in robotPlace.parts() if x.name() == gname), None)
            if gripper and gripper.collides(base):
                if showCollisions:
                    print 'Collision of', gname, 'with base'
                return False
        # Check for static collisions
        for objPlace in static:
            if robotPlace.collides(objPlace):
                if showCollisions:
                    print 'Collision with', objPlace.name()
                    objPlace.draw(glob.stderrWindow)
                return False
        # Check collisions with other objects
        collisions = []
        for objPlace in transient:
            if robotPlace.collides(objPlace):
                if showCollisions:
                    print 'Collision with', objPlace.name()
                    objPlace.draw(glob.stderrWindow)
                collisions.append(objPlace)
        return collisions
    
    def completeJointConf(self, conf, wstate=None, baseConf=None):
        assert wstate or baseConf
        if not baseConf:
            baseConf = wstate.objectConfs[self.chains.name]
        cfg = conf.copy()
        for cname in self.chainNames:
            if not cname in conf.keys():
                # use current chain confs as default.
                cfg = cfg.set(cname, baseConf[cname])
        return cfg

    def stepAlongLine(self, q_f, q_i, stepSize, forward = True, moveChains = None):
        moveChains = moveChains or self.moveChainNames
        q = q_i.copy()
        # Reverse the order of chains when working on the "from the goal" tree.
        for chainName in self.chainNames if forward else self.chainNames[::-1]:
            if not chainName in moveChains or \
               q_f[chainName] == q_i[chainName]: continue
            jv = self.chains.chainsByName[chainName].stepAlongLine(list(q_f[chainName]),
                                                                   list(q_i[chainName]),
                                                                   stepSize)
            return q.set(chainName, jv) # only move one chain at a time...
        return q_i

    def distConf(self, q1, q2):
        total = 0.
        for chainName in self.chainNames:
            if chainName in q1.conf and chainName in q2.conf:
                total += self.chains.chainsByName[chainName].dist(q1[chainName], q2[chainName])
        return total

    # "normalize" the angles...
    def normConf(self, target, source):
        cByN = self.chains.chainsByName
        for chainName in self.chainNames:
            if not chainName in target.conf or not chainName in source.conf: continue
            if target[chainName] == source[chainName]: continue
            target = target.set(chainName, cByN[chainName].normalize(target[chainName],
                                                                     source[chainName]))
        return target

    # Note that the "ArmFrame" is the wrist frame.
    def forwardKin(self, conf, wstate=None, complain = False, fail = False):
        shapes, frames = self.placement(conf, wstate=wstate, getShapes=[])
        return CartConf(\
            {'pr2BaseFrame': frames[self.chains.chainsByName['pr2Base'].joints[-1].name],
             'pr2LeftArmFrame':
             frames[self.chains.chainsByName['pr2LeftArm'].joints[-1].name],
             'pr2RightArmFrame':
             frames[self.chains.chainsByName['pr2RightArm'].joints[-1].name],
             'pr2HeadFrame': frames[self.chains.chainsByName['pr2Head'].joints[-1].name],
             'pr2LeftGripper': conf['pr2LeftGripper'],
             'pr2RightGripper': conf['pr2RightGripper'],
             'pr2Torso': conf['pr2Torso']},
            self)

    def inverseKin(self, cart, wstate=None,
                   conf = None, returnAll = False,
                   collisionAware = False, complain = False, fail = False):
        """Map from cartesian configuration (wrists and head) to joint
        configuration."""
        assert conf or wstate or self.nominalConf
        if collisionAware:
            assert wstate
        if conf is None:
            conf = wstate.robotConf if wstate else self.nominalConf
        torsoChain = self.chains.chainsByName['pr2Torso']
        baseChain = self.chains.chainsByName['pr2Base']
        # if wstate:
        #    assert wstate.frames[torsoChain.baseFname] == 'pr2BaseFrame'
        baseTrans = cart.get('pr2Base')
        baseAngles = baseChain.inverseKin(wstate.frames['root'] if wstate else Ident,
                                          baseTrans)
        if not baseAngles:
            if complain: print 'Failed invkin for base'
            if fail: raise Exception, 'Failed invkin for base'
            conf = conf.set('pr2Base', None)
        else:
            conf = conf.set('pr2Base', list(baseAngles))
        # First, pick a torso value, since that affects hands and head.
        if 'pr2Torso' in cart.conf:
            torsoZ = cart['pr2Torso'][0]
        else:
            torsoZ = torsoInvKin(self.chains, baseTrans,
                                 cart.get('pr2LeftArm', cart.get('pr2RightArm')),
                                 wstate,
                                 collisionAware=collisionAware)
        conf = conf.set('pr2Torso', [torsoZ])
        torsoTrans = self.chains.chainsByName['pr2Torso'].forwardKin(baseTrans, [torsoZ])
        # Solve for arms
        if 'pr2LeftArmFrame' in cart.conf:
            leftArmAngles = armInvKin(self.chains,
                                      'l', torsoTrans,
                                      cart['pr2LeftArm'],
                                      # if a nominal conf is available use as reference
                                      conf or self.nominalConf,
                                      wstate, returnAll = returnAll,
                                      collisionAware=collisionAware)
            if not leftArmAngles:
                if complain:
                    raw_input('Failed invkin for left arm')
                if fail: raise Exception, 'Failed invkin for left arm'
                conf = conf.set('pr2LeftArm', None)
            else:
                conf = conf.set('pr2LeftArm', leftArmAngles)
        if 'pr2RightArmFrame' in cart.conf:
            rightArmAngles = armInvKin(self.chains,
                                       'r', torsoTrans,
                                       cart['pr2RightArm'],
                                       # if a nominal conf is available use as reference
                                       conf or self.nominalConf,
                                       wstate, returnAll = returnAll,
                                       collisionAware=collisionAware)
            if not rightArmAngles:
                if complain:
                    raw_input('Failed invkin for right arm')
                if fail: raise Exception, 'Failed invkin for right arm'
                conf = conf.set('pr2RightArm', None)
            else:
                conf = conf.set('pr2RightArm', rightArmAngles)
        if 'pr2HeadFrame' in cart.conf:
            headAngles = headInvKin(self.chains,
                                    torsoTrans, cart['pr2Head'], wstate,
                                    collisionAware=collisionAware)
            if not headAngles:
                if complain: print 'Failed invkin for head'
                if fail: raise Exception, 'Failed invkin for head'
                conf = conf.set('pr2Head', None)
            else:
                conf = conf.set('pr2Head', headAngles)
        if 'pr2LeftGripper' in cart.conf:
            g = cart.conf['pr2LeftGripper']
            conf = conf.set('pr2LeftGripper', g if isinstance(g, (list,tuple)) else [g])
        if 'pr2RightGripper' in cart.conf:
            g = cart.conf['pr2RightGripper']
            conf = conf.set('pr2RightGripper', g if isinstance(g, (list,tuple)) else [g])
        return conf

########################################
# Inverse kinematics support functions
########################################

headInvKinCacheStats = [0,0]
headInvKinCache = {}

def headInvKin(chains, torso, targetFrame, wstate,
                 collisionAware=False, allowedViewError = 1e-5):
    headChain = chains.chainsByName['pr2Head']
    limits = headChain.limits()
    # Displacement from movable joints to sensor
    headSensorOffsetY = reduce(np.dot, [j.trans for j in headChain.joints[2:-1]]).matrix
    headSensorOffsetZ = reduce(np.dot, [j.trans for j in headChain.joints[1:-1]]).matrix

    headRotationFrameZ = np.dot(torso, headChain.joints[0].trans)
    # Target point relative to torso
    relFramePoint = torso.inverse().compose(targetFrame).point()

    key = relFramePoint
    headInvKinCacheStats[0] += 1
    if key in headInvKinCache:
        headInvKinCacheStats[1] += 1
        return headInvKinCache[key]

    if debug('pr2Head'):
        print 'target frame to\n', targetFrame.point().matrix
        print 'relFramePoint\n', relFramePoint.matrix

    if abs(relFramePoint.matrix[0,0]) < 0.1:
        # If the frame is just the head frame, then displace it.
        targetFrame = targetFrame.compose(hu.Pose(0.,0., 0.2, 0.0))
        if debug('pr2Head'):
            print 'displacing head frame to\n', targetFrame.point().matrix

    targetZ = headRotationFrameZ.inverse().applyToPoint(targetFrame.point()).matrix
    if debug('pr2Head'):
        print 'targetZ\n', targetZ
    angles1 = tangentSol(targetZ[0,0], targetZ[1,0], headSensorOffsetZ[0,3], headSensorOffsetZ[1,3])    
    angles1 += list(limits[0])
    
    best = None
    bestScore = 1.0e10
    bestError = None
    # print 'zero\n', headChain.forwardKin(torso, (0, 0)).matrix
    for a1 in angles1:
        headRotationFrameZ = np.dot(torso, headChain.joints[0].transform(a1))
        headRotationFrameY = np.dot(headRotationFrameZ, headChain.joints[1].trans)
        targetY = headRotationFrameY.inverse().applyToPoint(targetFrame.point()).matrix
        angles2 = [-x for x in tangentSol(targetY[0,0], targetY[2,0],
                                          headSensorOffsetY[0,3], headSensorOffsetY[2,3])]
        angles2 += list(limits[1])

        for a2 in angles2:
            if headChain.valid([a1, a2]):
                headTrans = headChain.forwardKin(torso, (a1, a2))
                sensorCoords = headTrans.inverse().applyToPoint(targetFrame.point()).matrix
                if sensorCoords[2] < 0.: continue
                score = math.sqrt(sensorCoords[0]**2 + sensorCoords[1]**2)
                # print 'score', score, 'sensorCoords', sensorCoords
                if score < bestScore:
                    best = [a1, a2]
                    bestScore = score
                    bestError = sensorCoords

    debugMsg('pr2Head',
             ('bestScore', bestScore, 'best', best), bestError)
        
    ans = best if bestScore <= allowedViewError else None
    headInvKinCache[key] = ans
    return ans

def tangentSolOld(x, y, x0, y0):
    # print 'X', (x,y), 'X0', (x0, y0)
    alpha = math.atan2(y,x)
    theta0 = math.atan2(y0,x0)
    r = math.sqrt(x0*x0 + y0*y0)
    d = math.sqrt(x*x + y*y)
    theta1 = math.pi/2
    ac = math.acos((r/d)*math.cos(theta0-theta1))
    # print 'alpha', alpha, 'theta0', theta0, 'ac', ac
    values =  [alpha + theta1 + ac,
               alpha + theta1 - ac,
               alpha - theta1 + ac,
               alpha - theta1 - ac]
    keep = []
    for angle in values:
        v = (x - r*math.cos(angle + theta0),
             y - r*math.sin(angle + theta0))
        vangle = math.atan2(v[1], v[0])
        # print 'angle', angle, 'atan', vangle
        if hu.angleDiff(angle, vangle) < 0.001:
            keep.append(vangle)
    # This generally returns two answers, but they may not be within limits.
    # print 'keep', keep
    return keep

# x0,y0 is sensor point (rotation is at origin)
# x,y is target point
# The sensor is pointing along the x axis when angle is zero!
# At desired sensor location we have a triangle (origin, sensor, target)
# l is distance from sensor to target, found from law of cosines.
# alpha is angle target-origin-sensor, also from law of cosines
def tangentSol(x, y, x0, y0):
    def quad(a,b,c):
        disc = b*b - 4*a*c
        if disc < 0: return []
        discr = disc**0.5
        return [(-b + discr)/(2*a), (-b - discr)/(2*a)]
        
    ph= math.atan2(y,x)
    d = math.sqrt(x*x + y*y)
    th0 = math.atan2(y0,x0)
    r = math.sqrt(x0*x0 + y0*y0)
    # c1 is cos of angle between sensor direction and line to x0,y0
    c1 = math.cos(math.pi - th0)
    # vals for l
    lvals = quad(1.0, -2*r*c1, r*r-d*d)
    # print 'lvals', lvals
    if not lvals: return []
    # vals for alpha
    avals = [math.acos(max(-1.0, min(1.0, (l*l - r*r - d*d)/(-2*r*d)))) for l in lvals]
    # print 'avals', avals
    # angs are candidate rotation angles
    angs = []
    for alpha in avals:
        angs.extend([ph - alpha - th0, ph + alpha - th0])
    angs = [hu.fixAnglePlusMinusPi(a) for a in angs]
    # print 'angs', angs
    ans = []
    for ang in angs:
        # check each angle, (x1,y1) is sensor location
        x1 = r*math.cos(th0 + ang)
        y1 = r*math.sin(th0 + ang)
        # distance sensor-target
        l = math.sqrt((x1-x)**2 + (y1-y)**2)
        # the sensor direction rotates by ang, check that it points at target
        ex = abs(x - (x1 + l*math.cos(ang)))
        ey = abs(y - (y1 + l*math.sin(ang)))
        # print 'ang', ang, 'ex', ex, 'ey', ey
        # keep the ones with low error
        if ex < 0.001 and ey < 0.001:
            ans.append(ang)
    return ans

def torsoInvKin(chains, base, target, wstate, collisionAware = False):
    # Should pick a good torso value to place the hand at target, prefering
    # not to change the current value if possible.
    return glob.torsoZ

###################
# Interpolators
##################

def interpPose(pose_f, pose_i, minLength, ratio=0.5):
    if isinstance(pose_f, (tuple, list)):
        return [f*ratio + i*(1-ratio) for (f,i) in zip(pose_f, pose_i)], \
               all([abs(f-i)<=minLength for (f,i) in zip(pose_f, pose_i)])
    else:
        pr = pose_f.point()*ratio + pose_i.point()*(1-ratio)
        qr = quaternion_slerp(pose_i.quat().matrix, pose_f.quat().matrix, ratio)
        return hu.Transform(None, pr.matrix, qr), \
               pose_f.near(pose_i, minLength, minLength)

def cartInterpolators(conf_f, conf_i, minLength):
    c_f = conf_f.cartConf()
    c_i = conf_i.cartConf()
    return cartInterpolatorsAux(c_f, c_i, conf_i, minLength)

def cartInterpolatorsAux(c_f, c_i, conf_i, minLength, depth=0):
    if depth > 10:
        raw_input('cartInterpolators depth > 10')
    robot = conf_i.robot
    if c_f == c_i: 
        conf = robot.inverseKin(c_f, conf=conf_i)
        conf['pr2Head'] = conf_i['pr2Head']
        return [conf]
    newVals = {}
    terminal = True
    for chain in c_i.conf:
        new, near = interpPose(c_f.conf[chain], c_i.conf[chain], minLength)
        newVals[chain] = new
        terminal = terminal and near
    if terminal: return []        # no chain needs splitting
    cart = CartConf(newVals, c_f.robot)
    conf = robot.inverseKin(cart, conf=conf_i)
    conf.conf['pr2Head'] = conf_i['pr2Head']
    if all([conf.conf.values()]):
        final = cartInterpolatorsAux(c_f, cart, conf, minLength, depth+1)
        if final != None:
            init = cartInterpolatorsAux(cart, c_i, conf_i, minLength, depth+1)
            if init != None:
                final.append(conf)
                final.extend(init)
    return final

###################
# Chain Frames
##################
inf = float('inf')
class ChainFrame:
    def __init__(self, base=None, joint=None, qi=None, link=None,
                 linkVerts=None, frame=None, bbox = None):
        self.base = base
        self.joint = joint
        self.qi = qi
        self.link = link
        self.linkVerts = linkVerts
        self.frame = frame
        # Radius squared for the link
        self.radius = vertsRadius(linkVerts) if linkVerts else None
        self.bbox = bbox

    def draw(self, win='W', color='magenta'):
        window = wm.getWindow(win)
        for v in self.linkVerts:
            verts = np.vstack([v.T, np.ones(v.shape[0])])
            window.draw(np.dot(self.frame, verts), color=color)
    def __str__(self):
        if self.joint: name=joint.name
        elif self.link: name=link.name()
        else: name=self.base
        return 'ChainFrame(%s)'%name

def vertsRadius(linkVerts):
    radSq = 0.0
    for verts in linkVerts:
        for i in xrange(verts.shape[0]):
            radSq = max(radSq, verts[i,0]*verts[i,0] + verts[i,1]*verts[i,1] +  verts[i,2]*verts[i,2] )
    return math.sqrt(radSq)

def linkVerts(link, rel=False):
    verts = []
    # print 'link origin\n', link.origin().matrix
    for prim in link.toPrims():
        if rel:
            # Attached objects are already relative to link frame
            off = prim.origin()
            verts.append(np.ascontiguousarray(np.dot(off.matrix,
                                                     prim.basePrim.baseVerts)[:3,:].T,
                                              dtype=np.double))
        else:
            off = link.origin().inverse().compose(prim.origin())
            verts.append(np.ascontiguousarray(np.dot(off.matrix,
                                                     prim.basePrim.baseVerts)[:3,:].T,
                                              dtype=np.double))
    return verts

# Compiles a Multi-Chain, like PR2
# frames is a dictionary of frameName : [base, joint...]
# framesList is a sequential list of frameNames, order matters
# chainNames is dictionary chainName : list of frame names in chain
# frameChain is dictionary frameName : chainName it belongs to
# chain
def compileChainFrames(robot):
    allChains = OrderedDict()
    frameChain = {}
    frames = {'root' : ChainFrame(frame=Ident.matrix)}
    framesList = []
    qi = 0
    for chain in robot.chains.chainsInOrder:
        base = chain.baseFname
        assert len(chain.joints) == len(chain.links)
        chainFrames = []
        for joint, link in zip(chain.joints, chain.links):
            framesList.append(joint.name)
            chainFrames.append(joint.name)
            frameChain[joint.name] = chain.name # reverse index
            if isinstance(joint, Rigid):
                index = None
            else:
                index = qi
                qi += 1
            if link and link.parts():
                frames[joint.name] = ChainFrame(base, joint, index, link, linkVerts(link),
                                                bbox=maxBBox)
            else:
                frames[joint.name] = ChainFrame(base, joint, index)
            base = joint.name
        allChains[chain.name] = chainFrames

    return frames, framesList, allChains, frameChain

inf = float('inf')
maxBBox = np.array(((-inf,-inf,-inf),(inf,inf,inf)))

att_framesList = ['attached']
att_allChains = {'left': OrderedDict([('pr2LeftArm', ['attached'])]),
                 'right': OrderedDict([('pr2RightArm', ['attached'])])}
att_frameChain = {'left': {'attached': 'pr2LeftArm'},
                 'right': {'attached': 'pr2RightArm'}}
 
attachedFramesCache = {}
attachedFramesCacheStats = [0,0]
def compileAttachedFrames(robot, attached, hand, robotFrames, pred=None):
    if (not attached) or (not attached[hand]): return None
    attachedFramesCacheStats[0] += 1
    wristFrame = robot.wristFrameNames[hand]
    attFrame = robotFrames[wristFrame].frame
    attFrame.flags.writeable = False
    key = (frozenset(attached.items()), hand, attFrame.data)
    if key in attachedFramesCache:
        attachedFramesCacheStats[1] += 1
        return attachedFramesCache[key]
    contents = attached[hand]
    if pred:
        contents = pred(contents)
    entry = ChainFrame(base=wristFrame, link=contents,
                       linkVerts=linkVerts(contents, rel=True),
                       frame=attFrame)
    entry.bbox = frameBBoxRad(entry)
    frames = {'attached': entry}
    ans = frames, att_framesList, att_allChains[hand], att_frameChain[hand]
    attachedFramesCache[key] = ans
    return ans

objectFramesCache = {}
objectFramesCacheStats = [0,0]
def compileObjectFrames(objShape):
    objectFramesCacheStats[0] += 1
    if objShape in objectFramesCache:
        objectFramesCacheStats[1] += 1
        return objectFramesCache[objShape]
    allChains = OrderedDict()
    frameChain = {}
    obj = objShape.name()
    origin = np.ascontiguousarray(objShape.origin().matrix, dtype=np.double)
    frames = {obj : ChainFrame(link=objShape, linkVerts=linkVerts(objShape),
                               frame=origin, bbox=objShape.bbox())}
    framesList = [obj]
    allChains[obj] = [obj]
    frameChain[obj] = obj
    ans = frames, framesList, allChains, frameChain
    objectFramesCache[objShape] = ans
    return ans

#########

def gripperPlace(conf, hand, wrist, robotPlace=None):
    confWrist = conf.cartConf()[conf.robot.armChainNames[hand]]
    if not robotPlace:
        robotPlace = conf.placement()
    gripperChain = conf.robot.gripperChainNames[hand]
    shape = (part for part in robotPlace.parts() if part.name() == gripperChain).next()
    return shape.applyTrans(confWrist.inverse()).applyTrans(wrist)
