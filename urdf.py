
import numpy as np
import math
import xml.etree.ElementTree as ET
import transformations as transf  # This is the widely available

def getUrdfJoints(jointNames,
                 file = "pr2.urdf"):
   robotTree = ET.parse(file)
   joints = [j for j in robotTree.findall('joint') if not j.find('origin') is None]
   result = []
   for name in jointNames:
       for j in joints:
           if j.attrib['name'] == name:
               result.append(jointFromUrdf(j))
   return result

def vec(str):
   return [float(x) for x in str.split()]

def jointFromUrdf(joint):
   name = joint.attrib['name']
   jtype = joint.attrib['type']
   origin = joint.find('origin')
   trn = transf.translation_matrix(vec(origin.attrib['xyz']))
   rot = transf.euler_matrix(*vec(origin.attrib['rpy']), axes='sxyz')
   limit = joint.find('safety_controller')
   axis = joint.find('axis')
   if not (axis is None):
       axis = vec(axis.attrib['xyz'])
   if jtype == 'continuous':
       limits = (-math.pi, math.pi)
   elif jtype == 'fixed':
       limits = None
   else:
       assert limit is not None and limit.attrib['soft_lower_limit'] is not None
       limits = (float(limit.attrib['soft_lower_limit']),
                 float(limit.attrib['soft_upper_limit']))
   print name, origin.attrib
   print name, jtype, limits, axis, '\n', trn, '\n', rot
   return Joint.subClasses[jtype](name, np.dot(trn, rot), limits, axis)

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

class Joint:
   subClasses = {}
   def __init__(self, name, trans, limits, axis):
       self.name = name
       self.trans = trans
       self.limits = limits
       self.axis = axis
       self.normalized = None

class Prismatic(Joint):
   def matrix(self, val):
       v = val
       vec = [q*v for q in self.axis]
       tr = np.array([[1., 0., 0., vec[0]],
                      [0., 1., 0., vec[1]],
                      [0., 0., 1., vec[2]],
                      [0., 0., 0., 1.]])
       return np.dot(self.trans, tr)
   def __repr__(self):
       return 'Joint:(%s, %s)'%('Prismatic', self.name)
   __str__ = __repr__
Joint.subClasses['prismatic'] = Prismatic

class Revolute(Joint):
   def matrix (self, val):
       v = val
       cv = math.cos(v); sv = math.sin(v)
       if self.axis[2] == 1.0:
           rot = np.array([[cv, -sv, 0., 0.],
                           [sv,  cv, 0., 0.],
                           [0.,  0., 1., 0.],
                           [0.,  0., 0., 1.]],
                          dtype=np.float64)
       elif self.axis[1] == 1.0:
           rot = np.array([[cv,  0., sv, 0.],
                           [0.,  1., 0., 0.],
                           [-sv,  0., cv, 0.],
                           [0.,  0., 0., 1.]],
                          dtype=np.float64)
       elif self.axis[0] == 1.0:
           rot = np.array([[1.,  0., 0., 0.],
                           [0., cv, -sv, 0.],
                           [0., sv,  cv, 0.],
                           [0.,  0., 0., 1.]],
                          dtype=np.float64)
       else:
           if debug('placement'): print 'general axis', self.axis
           rot = transf.rotation_matrix(val, self.axis)
       return np.dot(self.trans, rot)
   def __repr__(self):
       return 'Joint:(%s, %s)'%('Revolute', self.name)
   __str__ = __repr__
Joint.subClasses['revolute'] = Revolute
Joint.subClasses['continuous'] = Revolute

class General(Joint):
   def matrix(self, val):
       return val.matrix
   def __repr__(self):
       return 'Joint:(%s, %s)'%('General', self.name)
   __str__ = __repr__
Joint.subClasses['general'] = General

class Rigid(Joint):
   def matrix(self, val=None):
       return self.trans
   def __repr__(self):
       return 'Joint:(%s, %s)'%('Rigid', self.name)
   __str__ = __repr__
Joint.subClasses['fixed'] = Rigid


# Test cases

print getUrdfJoints(pr2_head_joints)
print getUrdfJoints(armJointNames('l', pr2_arm_joints))
