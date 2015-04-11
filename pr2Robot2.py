import pdb
import math
import util
import copy
import itertools
import transformations as transf
from transformations import quaternion_slerp
import numpy as np

import shapes
import cspace

import objects
from objects import *

from miscUtil import prettyString
import fbch

import planGlobals as glob
from planGlobals import debugMsg, debugDraw, debug, pause

from pr2IkPoses import ikTrans          # base poses for IK

from pr2InvKin import armInvKin

# # We need this for inverse kinematics
# from ctypes import *
# if glob.LINUX:
#     ik = CDLL(glob.libkinDir+"libkin.so.1.1")
# else:
#     ik = CDLL(glob.libkinDir+"libkin.dylib")
# ik.fkLeft.restype = None
# ik.fkRight.restype = None

Ident = util.Transform(np.eye(4))            # identity transform

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

def armJointNames(arm, joint_names):
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
    pose = util.Transform(transf.rotation_matrix(angle, (1,0,0)))
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
                  util.Transform(transf.translation_matrix([o + palm_dx/2.,0.0,0.0])),
                  None, None),
            Prismatic(arm+'_finger1',
                      util.Transform(transf.translation_matrix([palm_dx/2+fing_dx/2.,fing_dy/2.,0.0])),
                      (0.0, params['gripMax']/2.), (0.,1.,0)),
            Prismatic(arm+'_finger2',
                      util.Transform(transf.translation_matrix([0.0,-fing_dy,0.0])),
                      (0.0, 0.08), (0.,-1.,0))]

gripperTip = util.Pose(0.18,0.0,0.0,0.0)
gripperToolOffset = util.Transform(np.dot(gripperTip.matrix,
                                          transf.rotation_matrix(math.pi/2,(0,1,0))))
# Rotates wrist to grasp face frame
gripperFaceFrame = util.Transform(np.array([(0.,1.,0.,0.18),
                                            (0.,0.,1.,0.0),
                                            (1.,0.,0.,0.),
                                            (0.,0.,0.,1.)]))

# This behaves like a dictionary, except that it doesn't support side effects.
class JointConf:
    def __init__(self, conf, robot):
        global confIdnum
        self.conf = conf
        self.robot = robot
        self.items = None
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
    def cartConf(self):
        return self.robot.forwardKin(self)
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
            return 'JointConf('+prettyString(self.conf['pr2Base'], eq)+')'
        else:
            return 'JointConf('+prettyString(self.conf, eq)+')'
    def prettyPrint(self, msg='Conf:'):
        print msg
        for key in self.conf.keys():
            print '   ', key, prettyString(self.conf[key])
    def confItems(self):
        if not self.items:
            self.items = frozenset([(chain, tuple(self.conf[chain])) for chain in self.conf])
        return self.items
    def __str__(self):
        return 'JointConf('+str(self.conf)+')'
    def __getitem__(self, name):
        return self.conf[name]
    def __hash__(self):
        return hash(self.confItems())
    def __eq__(self, other):
        if not (other and isinstance(other, JointConf)):
            return False
        else:
            return self.conf == other.conf
    def __neq__(self, other):
        return not self == other

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
            assert value is None or isinstance(value, util.Transform)
            c.conf[name] = value
        return c
    def confItems(self):
        if not self.items:
            vals = []
            for chain in self.conf:
                val = self.conf[chain]
                if isinstance(val, list):
                    vals.append(tuple(val))
                elif isinstance(val, util.Transform):
                    vals.append(repr(val))
                else:
                    vals.append(val)
            self.items = frozenset(vals)
        return self.items
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
        pose = util.Transform(transf.rotation_matrix(angle, (1,0,0)))
        gr1 = radiusVar + 0.43*reachPct*angleVar
        gr2 = radiusVar + 0.76*reachPct*angleVar
        gr3 = radiusVar + 1.05*reachPct*angleVar
        print 'gr1', gr1, 'gr2', gr2, 'gr3', gr3
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
                                    pr2GripperLinksGrown())
    # Chain for right arm
    rightArmChain = Chain('pr2RightArm', 'torso_lift_joint',
                         getUrdfJoints(armJointNames('r', pr2_arm_joints)),
                         pr2ArmLinksGrown('r'))
    # Chain for right gripper
    rightGripperChain = GripperChain('pr2RightGripper', 'r_wrist_roll_joint',
                                     pr2GripperJoints('r'),
                                     pr2GripperLinksGrown())
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
    return util.Pose(*params)

def flipv(pose):
    m = pose.matrix.copy()
    m[1,3] = -m[1,3]
    return util.Transform(m)

def ground(pose):
    params = list(pose.xyztTuple())
    params[2] = 0.0
    return util.Pose(*params)

confCacheStats = [0, 0, 0, 0]
kinCacheStats = [0, 0]

# This basically implements a Chain type interface, execpt for the wstate
# arguments to the methods.
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
        self.gripperTip = gripperTip
        self.nominalConf = None
        horizontalTrans, verticalTrans = ikTrans()
        self.horizontalTrans = {'left': [p.inverse() for p in horizontalTrans],
                                'right': [fliph(p).inverse() for p in horizontalTrans]}
        self.verticalTrans = {'left': [p.inverse() for p in verticalTrans],
                                'right': [flipv(p).inverse() for p in verticalTrans]}
        self.confCache = {}
        self.kinCache = {}
        if debug('PR2'): print 'New PR2!'
        return

    def potentialBasePosesGen(self, wrist, hand):
        xAxisZ = wrist.matrix[2,0]
        if abs(xAxisZ) < 0.01:
            trs = self.horizontalTrans[hand]
        elif abs(xAxisZ + 1.0) < 0.01:
            trs = self.verticalTrans[hand]
        else:
            print 'wrist=\n', wrist.matrix
            raw_input('Illegal wrist trans for base pose')
        for tr in trs:
            ans = wrist.compose(tr).pose(fail=False)
            if ans is None:
                print 'wrist=\n', wrist.matrix
                raw_input('Illegal wrist trans for base pose')
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
        gripperFaceFrame_dy = util.Transform(np.array([(0.,1.,0.,0.18),
                                                       (0.,0.,1.,-width/2),
                                                       (1.,0.,0.,0.),
                                                       (0.,0.,0.,1.)]))
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
            self.confCache[key] = (place, trans)
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
            jv = self.chains.chainsByName[chainName].stepAlongLine(q_f[chainName],
                                                                   q_i[chainName],
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
        if conf == None:
            conf = wstate.robotConf if wstate else self.nominalConf
        # This doesn't seem to pay off
        # key = (cart, conf, returnAll, collisionAware)
        # val = self.kinCache.get(key, None)
        # kinCacheStats[0] += 1
        # if val != None:
        #     kinCacheStats[1] += 1
        #     return val
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
            conf = conf.set('pr2Base', baseAngles)
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
                                      # The kinematics has a tool offset built in
                                      gripperToolOffset,
                                      # if a nominal conf is available use as reference
                                      self.nominalConf or conf,
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
                                       # The kinematics has a tool offset built in
                                       gripperToolOffset,
                                       # if a nominal conf is available use as reference
                                       self.nominalConf or conf,
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
        # self.kinCache[key] = conf
        return conf

########################################
# Inverse kinematics support functions
########################################

def headInvKin(chains, torso, targetFrame, wstate,
                 collisionAware=False, allowedViewError = 1e-5):
    headChain = chains.chainsByName['pr2Head']
    limits = headChain.limits()
    headSensorOffsetY = reduce(np.dot, [j.trans for j in headChain.joints[2:-1]]).matrix
    headSensorOffsetZ = reduce(np.dot, [j.trans for j in headChain.joints[1:-1]]).matrix

    headRotationFrameZ = np.dot(torso, headChain.joints[0].trans)

    relFramePoint = torso.inverse().compose(targetFrame).point()
    if abs(relFramePoint.matrix[0,0]) < 0.1:
        # If the frame is just the head frame, then displace it.
        targetFrame = targetFrame.compose(util.Pose(0.,0., 0.1, 0.0))

    targetZ = headRotationFrameZ.inverse().applyToPoint(targetFrame.point()).matrix
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
    # print 'bestScore', bestScore, 'best', best, 'error\n', bestError
    return best if bestScore <= allowedViewError else None

def tangentSol(x, y, x0, y0):
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
        if util.angleDiff(angle, vangle) < 0.001:
            keep.append(vangle)
    # This generally returns two answers, but they may not be within limits.
    # print 'keep', keep
    return keep

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
        return util.Transform(None, pr.matrix, qr), \
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
