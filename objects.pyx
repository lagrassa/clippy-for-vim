import transformations as transf
import copy
import math
import random 
import numpy as np
import xml.etree.ElementTree as ET

from hu cimport Ident, Transform, angleDiff, fixAnglePlusMinusPi
from shapes cimport Shape
from planGlobals import  mergeShadows
from traceFile import debug, debugMsg
from geom import bboxUnion

PI2 = 2*math.pi

############################################################
# Objecs
############################################################

class World:
    def __init__(self):
        self.world = self                 # so x.world works..
        self.objects = {}
        self.regions = {}
        self.robot = None
        self.workspace = None
        # The frames are relative to the origin
        self.graspDesc = None               # dict {obj : [GDesc,...]}
        self.supportFrames = None           # dict {obj : [frame,..]}
        # Mapping from strings to strings:  obj name to obj type name
        self.objectTypes = {}
        # Mapping from object types to symmetries.  If obj type not
        # listed, assume no symmetries.  A symmetry entry has two
        # parts: a mapping from faces to a canonical face; mapping
        # from canonical faces to a set of 4D transforms
        self.symmetries = {}
        # pointClouds (4xn numpy arrays) for each type.
        self.typePointClouds = {}

    def getObjType(self, obj):
        return self.objectTypes[obj]
    def getSymmetries(self, objType):
        if objType in self.symmetries:
            return self.symmetries[objType]
        else:
            return ({}, {})

    def copy(self):
        cw = copy.copy(self)
        cw.objects = self.objects.copy()
        return cw
        
    def addObject(self, obj):             # obj is Chain or MultiChain
        if isinstance(obj, Chain):
            obj = MultiChain(obj.name, [obj])
        self.objects[obj.name] = obj

    def addObjectShape(self, objShape):
        self.addObject(Movable(objShape.name(), 'root', objShape))

    def delObject(self, name):
        del self.objects[name]

    def addObjectRegion(self, objName, regName, regShape, regTr):
        if objName in self.regions:
            self.regions[objName].append((regName, regShape, regTr))
        else:
            self.regions[objName] = [(regName, regShape, regTr)]

    def setRobot(self, robot):
        self.robot = robot

    def getObjectShapeAtOrigin(self, objName):
        def simplify(shape, depth=0):
            if depth >= 10:
                print 'Simplify loop:', objName, shape, shape.parts()
                raw_input('Huh')
                return shape
            if shape.name() == objName and \
               len(shape.parts()) == 1 and  \
               shape.parts()[0].name() == objName:
                return simplify(shape.parts()[0], depth=depth+1)
            else:
                return shape
        obj = self.objects[objName]
        chains = obj.chainsInOrder
        assert isinstance(chains[0], Movable)
        assert all([isinstance(chain, Permanent) for chain in chains[1:]])
        conf = dict([[objName, [Ident]]] +\
                    [[chain.name, []] for chain in chains[1:]])
        shape = obj.placement(Ident, conf)[0]
        if mergeShadows:
            return next((part for part in shape.parts() if part.name() == objName), None)
        else:
            # Make sure that we remove unnesessary nestings
            return simplify(shape)

    def getGraspDesc(self, obj):
        if obj == 'none':
            obj = self.graspDesc.keys()[0]
        return self.graspDesc[obj]
    def getFaceFrames(self, obj):
        return self.getObjectShapeAtOrigin(obj).faceFrames()
        
    def __str__(self):
        return 'World(%s)'%({'objects':self.objects})
    __repr__ = __str__

class WorldState:
    def __init__(self, world, robot=None):
        self.robot = robot or world.robot
        self.world = world        # This is a state of this world
        # Below is "state"
        self.objectConfs = {}     # object name -> object conf
        self.objectProps = {}     # object name -> property dictionary
        self.robotConf = None
        # Below is some derived information from this state
        self.frames = {'root': Ident}  # frame name -> Transform
        self.objectShapes = {}              # keep track of placements
        self.regionShapes = {}
        self.robotPlace = None
        self.fixedObjects = set([])           # immovable object names
        self.fixedHeld = {'left':False, 'right':False}  # is held obj fixed?
        self.fixedGrasp = {'left':False, 'right':False}  # is grasp fixed?
        self.held = {'left':None, 'right':None}   # object names
        self.grasp = {'left':None, 'right':None}  # transforms
        self.attached = {'left':None, 'right':None} # object shapes

    def getObjectShapeAtOrigin(self, objName):
        return self.world.getObjectShapeAtOrigin(objName)
    def getObjectShapes(self):
        return self.objectShapes.values()
    def getShadowShapes(self):
        return [shape for shape in self.getObjectShapes() \
                if shape.name()[-7:] == '_shadow']
    def getNonShadowShapes(self):
        return [shape for shape in self.getObjectShapes() \
                if shape.name()[-7:] != '_shadow']
                        
    def copy(self):
        cws = WorldState(self.world)
        cws.objectConfs = self.objectConfs.copy()
        cws.objectProps = self.objectProps.copy()
        cws.frames = self.frames.copy()
        cws.objectShapes = self.objectShapes.copy()
        cws.regionShapes = self.regionShapes.copy()
        cws.robotConf = self.robotConf
        cws.robotPlace = self.robotPlace
        cws.held = self.held.copy()
        cws.grasp = self.grasp.copy()
        cws.attached = self.attached.copy()
        return cws

    def setObjectConf(self, objName, conf):
        obj = self.world.objects[objName]
        if not isinstance(conf, dict):
            conf = {objName:conf}
        # Update the state of the world
        self.objectConfs[objName] = conf
        for chain in obj.chainsInOrder:
            chainPlace, chainTrans = chain.placement(self.frames[chain.baseFname],
                                                     conf[chain.name])
            for part in chainPlace.parts():
                self.objectShapes[part.name()] = part

            if objName in self.world.regions:
                for (regName, regShape, regTr) in self.world.regions[objName]:
                    tr = self.objectShapes[objName].origin().compose(regTr)
                    self.regionShapes[regName] = regShape.applyTrans(tr)
                    if debug('addRegion'):
                        print 'obj origin\n', self.objectShapes[objName].origin().matrix
                        print 'regTr\n', regTr.matrix
                        print 'tr\n', tr.matrix
                        print 'obj bb\n', self.objectShapes[objName].bbox()
                        print 'reg bb\n', self.regionShapes[regName].bbox()
                        self.regionShapes[regName].draw('W')
                        debugMsg('addRegion', 'adding %s'%regName)

            for fname, tr in zip(chain.fnames, chainTrans):
                self.frames[fname] = tr

    # Should there be a getObjectConf?

    def setObjectPose(self, objName, objPose):
        self.setObjectConf(objName, [objPose])

    def getObjectPose(self, objName):
        assert objName in self.objectConfs, 'Unknown object in world'
        confs = self.objectConfs[objName][objName]
        return confs[0] if confs else Ident

    def delObjectState(self, name):
        del self.objectConfs[name]
        del self.objectShapes[name]
        del self.frames[name]

    def setRobotConf(self, conf):
        robot = self.robot
        # Update the state of the world
        self.robotConf = conf
        # Include the frames in the world
        placement, frames = robot.placement(conf, attached = self.attached)
        self.robotPlace = placement
        for (fname, tr) in frames.items():
            self.frames[fname] = tr

    def getFrame(self, fname):
        return self.frames[fname]

    def getChainConf(self, cname):
        return self.chainConfs[cname]

    def gripperAtOrigin(self, hand):
        if not self.robotPlace: return None
        robot = self.robot
        gripperChain = robot.gripperChainNames[hand]
        wristFrame = robot.wristFrameNames[hand]
        gripper=next((x for x in self.robotPlace.parts() if x.name()==gripperChain), None)
        # The origin here is the wrist.  
        origin = self.frames[wristFrame].compose(self.robot.gripperTip).pose()
        return gripper.applyTrans(origin.inverse())

    def gripper(self, hand):
        if not self.robotPlace: return None
        robot = self.robot
        gripperChain = robot.gripperChainNames[hand]
        gripper = next((x for x in self.robotPlace.parts() if x.name()==gripperChain), None)
        return gripper

    def base(self):
        if not self.robotPlace: return None
        robot = self.robot
        baseChain = robot.baseChainName
        base = next((x for x in self.robotPlace.parts() if x.name()==baseChain), None)
        return base

    def draw(self, window, excluded = [],
             drawRobot = True, objectColors = {}, objectOpacities = {}):
        colors = ['red','orange','gold', 'green','blue','purple']
        objects = self.getShadowShapes() + self.getNonShadowShapes()
        for place in objects:
            name = place.name()
            if name in excluded: continue
            place.draw(window,
                       color=objectColors.get(place.name(),
                                              place.properties.get('color', None) \
                                              or random.choice(colors)),
                       opacity=objectOpacities.get(place.name(), 1.0))
        if drawRobot and self.robotPlace:
            self.robotPlace.draw(window,
                                 color=objectColors.get('robot', 'gold'),
                                 opacity=objectOpacities.get('robot', 1.0))

    def __str__(self):
        return 'WorldState(%s)'%(self.objectConfs)
    __repr__ = __str__
                
# A multi-chain object is basically a dictionary that maps part names
# into Chain instances.

cdef class MultiChain:
    def __init__(self, name, chains):
        self.type = 'MultiChain'
        self.chainsInOrder = sorted(chains, chainCmp)
        self.chainsByName = dict([(c.name, c) for c in chains])
        self.fnames = [j for chain in chains for j in chain.fnames]
        nonLocalBaseFnames = [chain.baseFname for chain in chains \
                              if not chain.baseFname in self.fnames]
        # There should only be one "free" base fname among the chains.
        assert len(nonLocalBaseFnames) == 1
        self.baseFname = nonLocalBaseFnames[0]
        self.name = name

    cpdef placement(self, base, conf, getShapes = True):
        """Returns a shape object given a dictionary of name:jointValues"""
        cfg = conf.copy()
        chainShapes = []
        frames = {self.baseFname : base}
        # pr = True if (getShapes and getShapes != True) else False
        pr = False
        for chain in self.chainsInOrder:
            if getShapes is True:
                chainShape = getShapes
            else:
                chainShape = chain.name in getShapes
            (sha, trs) = chain.placement(frames[chain.baseFname], cfg[chain.name],
                                         getShapes=chainShape)
            if pr:
                print ' **Placement**', chain.name, cfg[chain.name], bool(sha)
            if sha:
                chainShapes.append(sha)
            for joint, tr in zip(chain.joints, trs):
                frames[joint.name] = tr
        return Shape(chainShapes, None, name = self.name) if getShapes else None, frames

    cpdef placementMod(self, base, conf, place, getShapes = True):
        if len(self.chainsInOrder) != len(place.parts()):
            print 'chains', len(self.chainsInOrder), self.chainsInOrder
            print 'place.parts()', len(place.parts()), place.parts()
            raw_input('Mismatch of chains and parts')
        cfg = conf.copy()
        frames = {self.baseFname : base}
        pr = False
        for chain, part in zip(self.chainsInOrder, place.parts()):
            if getShapes is True:
                chainShape = getShapes
            else:
                chainShape = chain.name in getShapes
            (_, trs) = chain.placementMod(frames[chain.baseFname], cfg[chain.name], part)
            if pr:
                print ' **Placement**', chain.name, cfg[chain.name]
            for joint, tr in zip(chain.joints, trs):
                frames[joint.name] = tr
        place.thingBBox = bboxUnion([x.bbox() for x in place.parts()])
        return place, frames

    def __str__(self):
        return 'MultiChain:%s'%self.name
    __repr__ = __str__

cpdef int chainCmp(c1, c2):
    if c1.baseFname in c2.fnames:       # c1 needs to be later: c1 > c2
        return 1
    elif c2.baseFname in c1.fnames:     # c1 needs to be earlier: c1 < c2
        return -1
    return 0

############################################################
# Chains
############################################################

# A Chain is a list of links, a list of joints and a specified base frame name.

# Note that a regular "movable object" has a single link and a single
# (general) joint, parameterized by a Transform.  Some examples:
# tr = prismatic joint; rev = revolute joint; gen = general joint; fix = fixed joint
# movable: <base> <gen> <link>
# permanent: <base> <fix> <link>
# planar: <base> <tr: x> <empty> <tr: y> <empty> <rev: theta> <link>
# XYZT (cup or pan): <base> <tr: x> <empty> <tr: y> <empty> <tr: z> <empty> <rev: theta> <link>
# door: <base> <fix> <empty> <rev: theta> <link>

cdef class Chain:
    def __init__(self, name, baseFname, joints, links):
        self.name = name
        self.baseFname = baseFname
        self.joints = joints
        self.links = links
        self.fnames = [j.name for j in joints]
        self.chainsInOrder = [self]
        self.chainsByName = {name : self}
        self.jointLimits = None
        self.movingJoints = [joint for joint in self.joints\
                             if not isinstance(joint, Rigid)]
    # jointValues is a list of individual joint values
    cpdef frameTransforms(self, base, jointValues):
        """Returns all the frames (for each joint) given jointValues."""
        j = 0
        frames = [base]                 # list of Transforms
        for joint in self.joints:
            if isinstance(joint, Rigid):
                frames.append(frames[-1].compose(joint.transform()))
            else:
                val = jointValues[j]; j += 1
                if joint.valid(val):
                    frames.append(frames[-1].compose(joint.transform(val)))
                else:
                    print 'Invalid joint value for', joint, val
                    return None
        assert j == len(jointValues)    # we used them all
        return frames[1:]               # not the base

    cpdef limits(self):
        "Returns the limits for the relevant joints."
        if not self.jointLimits:
            self.jointLimits = [joint.limits for joint in self.movingJoints]
        return self.jointLimits

    cpdef randomValues(self):
        return [lo + random.random()*(hi-lo) for (lo, hi) in self.limits()]

    cpdef bool valid(self, list jointValues):
        cdef double jv
        cdef Joint joint
        assert len(jointValues) == len(self.movingJoints)
        for (joint, jv) in zip(self.movingJoints, jointValues):
            if not joint.valid(jv): return False
        return True

    cpdef placement(self, base, jointValues, getShapes=True):
        """Returns shape object for chain at the given jointValues and a list of
        the frames for each link."""
        trs = self.frameTransforms(base, jointValues)
        if trs:
            if getShapes:
                if debug('mod'):
                    print 'Non mod', self.name
                parts = []
                for p, tr in zip(self.links, trs):
                    if p:
                        parts.append(p.applyLoc(tr))
                        if debug('mod'):
                            print p.name(), '\n', parts[-1].vertices()
                shape = Shape(parts,
                              None,   # origin
                              name = self.name)
                return shape, trs
            else:
                return None, trs
        else:
            print 'Placement failed for', self, jointValues

    cpdef placementMod(self, base, jointValues, place):
        """Returns shape object for chain at the given jointValues and a list of
        the frames for each link."""
        trs = self.frameTransforms(base, jointValues)
        if trs:
            if debug('mod'): print 'Mod', self.name, place.name()
            index = 0
            parts = place.parts()
            for (p, tr) in zip(self.links, trs):
                if not isinstance(tr, hu.Transform):
                    raw_input('Foo')
                if p:
                    pM = parts[index]
                    index += 1
                    p.applyLocMod(tr, pM)
                    if debug('mod'):
                        print p.name(), '\n', pM.vertices()
            place.thingBBox = bboxUnion([x.bbox() for x in place.parts()])
            return None, trs
        else:
            print 'Placement failed for', self, jointValues

    cpdef forwardKin(self, base, jointValues):
        """Returns the targetValue for given jointValues or None if illegal."""
        tr = self.frameTransforms(base, jointValues)
        if tr: return tr[-1]
    
    # targetValue is a Transform
    cpdef targetPlacement(self, base, targetValue):
        """Returns shape object for chain that places last joint frame."""
        ik = self.inverseKin(base, targetValue)
        if ik is None: return None
        return self.placement(base, ik)

    cpdef stepAlongLine(self, jvf, jvi, stepSize):
        assert len(jvf) == len(jvi) == len(self.movingJoints)
        assert self.valid(jvf)
        indices = range(len(jvi))
        diffs = [self.movingJoints[i].diff(jvf[i], jvi[i]) for i in indices]
        length = sum([d**2 for d in diffs])**0.5
        if length == 0. or stepSize/length >= 1.: return jvf
        vals = [diffs[i]*(stepSize/length) + jvi[i] for i in indices]
        return vals

    cpdef interpolate(self, jvf, jvi, ratio, stepSize):
        assert len(jvf) == len(jvi) == len(self.movingJoints)
        assert self.valid(jvf)
        indices = range(len(jvi))
        diffs = [self.movingJoints[i].diff(jvf[i], jvi[i]) for i in indices]
        length = sum([d**2 for d in diffs])**0.5
        vals = [diffs[i]*ratio + jvi[i] for i in indices]
        return vals, length <= stepSize
    
    cpdef dist(self, jvf, jvi):
        assert len(jvf) == len(jvi) == len(self.movingJoints)
        indices = range(len(jvi))
        diffs = [self.movingJoints[i].diff(jvf[i], jvi[i]) for i in indices]
        return math.sqrt(sum([d**2 for d in diffs]))

    cpdef normalize(self, jvf, jvi):
        assert len(jvf) == len(jvi) == len(self.movingJoints)
        indices = range(len(jvi))
        diffs = [self.movingJoints[i].diff(jvf[i], jvi[i]) for i in indices]
        new = [jvi[i] + diffs[i] for i in indices]
        assert self.valid(new)
        return new

    cpdef inverseKin(self, base, targetValue):
        """Returns valid jointValues for given targetValue or None if illegal.
        There is no systematic process for inverseKin for arbitrary chains.
        This has to be provided by the subclass."""
        return None

    def __str__(self):
        return 'Chain:%s'%self.name
    __repr__ = __str__
    
cdef class Movable(Chain):
    def __init__(self, name, baseFname, shape):
        Chain.__init__(self, name, baseFname,
                       [General(name, None, None, None)],
                       [shape])

    cpdef inverseKin(self, base, tr):
        ibase = base.compose(self.joints[0].trans)
        return ibase.inverse().compose(tr)

cdef class Planar(Chain):
    def __init__(self, name, baseFname, shape, bbox):
        Chain.__init__(self, name, baseFname,
                       [Prismatic(name+'_x', Ident,
                                  (bbox[0,0], bbox[1,0]), (1.,0.,0.)),
                        Prismatic(name+'_y', Ident,
                                  (bbox[0,1], bbox[1,1]), (0.,1.,0.)),
                        Revolute(name+'_theta', Ident,
                                 (-math.pi, math.pi), (0.,0.,1.))],
                        [Shape([], None),
                         Shape([], None), shape])

    cpdef inverseKin(self, base, tr):
        ibase = base.compose(self.joints[0].trans)
        tr = ibase.inverse().compose(tr)
        # Is it a (nearly) planar transform? Rot about z, no z offset
        pose = tr.pose(fail=False)
        if pose and abs(tr.matrix[2,3]) < 0.001:
            params = [pose.x, pose.y, pose.theta]
            for (j, x) in zip(self.joints, params):
                if not j.valid(x): return None
            return params

cdef class XYZT(Chain):
    def __init__(self, name, baseFname, shape, bbox):
        Chain.__init__(self, name, baseFname,
                       [Prismatic(name+'_x', Ident,
                                  (bbox[0,0], bbox[1,0]), (1.,0.,0.)),
                        Prismatic(name+'_y', Ident,
                                  (bbox[0,1], bbox[1,1]), (0.,1.,0.)),
                        Prismatic(name+'_z', Ident,
                                  (bbox[0,2], bbox[1,2]), (0.,0.,1.)),
                        Revolute(name+'_theta', Ident,
                                 (-math.pi, math.pi), (0.,0.,1.))],
                        [Shape([], None),
                         Shape([], None),
                         Shape([], None), shape])        

    cpdef inverseKin(self, base, tr):
        ibase = base.compose(self.joints[0].trans)
        tr = ibase.inverse().compose(tr)
        # Is it a pose transform? Rot about z
        pose = tr.pose(fail=False)
        if pose and abs(tr.matrix[2,3]) < 0.001:
            params = [pose.x, pose.y, pose.z, pose.theta]
            for (j, x) in zip(self.joints, params):
                if not j.valid(x): return None
            return params

cdef class Permanent(Chain):
    def __init__(self, name, baseFname, shape, locTr):
        Chain.__init__(self, name, baseFname,
                       [Rigid(name+'_loc', locTr, None, None)],
                       [shape])

    cpdef inverseKin(self, base, target):
        return []                       # no motion

cdef class RevoluteDoor(Chain):
    def __init__(self, name, baseFname, shape, locTr, angles):
        Chain.__init__(self, name, baseFname,
                       [Rigid(name, locTr, None, None),
                        Revolute(name+'_theta', Ident, angles, (0.,0.,1.))],
                       [Shape([], None), shape])

    cpdef inverseKin(self, base, tr):
        ibase = base.compose(self.joints[0].trans)
        tr = ibase.inverse().compose(tr)
        # Is it a pure rotation about z transform?
        pose = tr.pose(fail=False)
        if pose and all([abs(tr.matrix[i,3]) < 0.001 for i in range(3)]):
            if self.joints[1].valid(pose.theta):
                return [None, pose.theta]

# This is a weird one..
cdef class GripperChain(Chain):
    cpdef frameTransforms(self, base, jointValues):
        width = jointValues[-1]
        return Chain.frameTransforms(self, base, [0.5*width, width])
    cpdef limits(self):
        return Chain.limits(self)[1:]
    cpdef bool valid(self, list jointValues):
        width = jointValues[-1]
        return Chain.valid(self, [0.5*width, width])
    cpdef placement(self, base, jointValues, getShapes=True):
        width = jointValues[-1]
        return Chain.placement(self, base, [0.5*width, width], getShapes=getShapes)
    cpdef placementMod(self, base, jointValues, place):
        width = jointValues[-1]
        return Chain.placementMod(self, base, [0.5*width, width], place)
    cpdef stepAlongLine(self, jvf, jvi, stepSize):
        assert len(jvf) == len(jvi) == 1
        assert self.valid(jvf)
        diff = jvf[-1] - jvi[-1]
        length = diff*diff
        if length == 0. or stepSize/length >= 1.: return jvf
        return [diff*(stepSize/length) + jvi[-1]]
    cpdef interpolate(self, jvf, jvi, ratio, stepSize):
        assert len(jvf) == len(jvi) == 1
        assert self.valid(jvf)
        diff = jvf[-1] - jvi[-1]
        length = (diff*diff)**0.5
        return [diff*ratio + jvi[-1]], length <= stepSize
    cpdef dist(self, jvf, jvi):
        assert len(jvf) == len(jvi) == 1
        diff = jvf[-1] - jvi[-1]
        return math.sqrt(diff*diff)
    cpdef normalize(self, jvf, jvi):
        return jvf
    cpdef forwardKin(self, base, jointValues):
        width = jointValues[-1]
        return Chain.forwardKin(self, base, [0.5*width, width])
    cpdef targetPlacement(self, base, targetValue):
        raise Exception, 'Not implemented'
    cpdef inverseKin(self, base, targetValue):
        raise Exception, 'Not implemented'

############################################################
# Joints
############################################################

cdef class Joint:
    subClasses = {}
    def __init__(self, name, trans, limits, axis):
        self.name = name
        self.trans = trans
        self.limits = limits
        self.axis = axis
        self.normalized = None

cdef class Prismatic(Joint):
    cpdef transform(self, val):
        return Transform(np.dot(self.trans.matrix,
                                     transf.translation_matrix([q*val for q in self.axis])))
    cpdef bool valid(self, double val):
        cdef double lo, hi
        (lo, hi) = self.limits
        return lo-0.0001 <= val <= hi+0.0001
    cpdef diff(self, a, b):
        return a - b
    def __repr__(self):
        return 'Joint:(%s, %s)'%('Prismatic', self.name)
    __str__ = __repr__
Joint.subClasses['prismatic'] = Prismatic
    
cdef class Revolute(Joint):
    # cpdef transform(self, val):
    #     return Transform(np.dot(self.trans.matrix,
    #                                      transf.rotation_matrix(val, self.axis)))
    cpdef transform(self, val):
        cv = math.cos(val); sv = math.sin(val)
        if self.axis == [0.0, 0.0, 1.0]:
            rot = np.array([[cv, -sv, 0., 0.],
                            [sv,  cv, 0., 0.],
                            [0.,  0., 1., 0.],
                            [0.,  0., 0., 1.]],
                           dtype=np.float64)
        elif self.axis == [0.0, 1.0, 0.0]:
            rot = np.array([[cv,  0., sv, 0.],
                            [0.,  1., 0., 0.],
                            [-sv,  0., cv, 0.],
                            [0.,  0., 0., 1.]],
                           dtype=np.float64)
        elif self.axis == [1.0, 0.0, 0.0]:
            rot = np.array([[1.,  0., 0., 0.],
                            [0., cv, -sv, 0.],
                            [0., sv,  cv, 0.],
                            [0.,  0., 0., 1.]],
                           dtype=np.float64)
        else:
            rot = transf.rotation_matrix(val, self.axis)
        return Transform(np.dot(self.trans.matrix, rot))
                                 
    cpdef bool valid(self, double val):
        cdef double lo, hi, vw
        if not self.normalized:
            self.normalized = normalizedAngleLimits(self.limits)
        for (lo, hi) in self.normalized:
            if val > math.pi or val < -math.pi:
                vw = (val+math.pi)%PI2
                if vw < 0: vw += PI2
                vw = vw - math.pi
            else:
                vw = val
            # vw = fixAnglePlusMinusPi(val)
            # if abs(vw1 - vw) > 0.0001:
            #    print val, 'vw1', vw1, 'vw', vw
            if (lo-0.001 <= vw <= hi+0.001):
                return True
        return False
    cpdef diff(self, a, b):
        if self.limits[0] == -math.pi and self.limits[1] == math.pi:
            return angleDiff(a, b)
        else:
            return a - b
    def __repr__(self):
        return 'Joint:(%s, %s)'%('Revolute', self.name)
    __str__ = __repr__
Joint.subClasses['revolute'] = Revolute
Joint.subClasses['continuous'] = Revolute

cdef list normalizedAngleLimits(tuple limits):
    (lo, hi) = limits
    if lo >= -math.pi and hi <= math.pi:
        return [limits]
    elif lo < -math.pi and hi <= math.pi:
        return [(-math.pi, hi), (fixAnglePlusMinusPi(lo), math.pi)]
    elif lo >= -math.pi and hi > math.pi:
        return [(-math.pi, fixAnglePlusMinusPi(hi)), (lo, math.pi)]
    else:
        raise Exception, 'Bad angle range'

cdef class General(Joint):
    cpdef transform(self, val):
        return Transform(val.matrix)
    cpdef bool valid(self, val):
        return True
    cpdef diff(self, a, b):
        return a.inverse().compose(b)
    def __repr__(self):
        return 'Joint:(%s, %s)'%('General', self.name)
    __str__ = __repr__
Joint.subClasses['general'] = General

cdef class Rigid(Joint):
    cpdef transform(self, val=None):
        return self.trans
    cpdef bool valid(self, val=None):
        return True
    cpdef diff(self, a, b):
        raise Exception, 'Rigid joint does not have diff'
    def __repr__(self):
        return 'Joint:(%s, %s)'%('Rigid', self.name)
    __str__ = __repr__
Joint.subClasses['fixed'] = Rigid

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
    
cdef vec(str):
    return [float(x) for x in str.split()]

def jointFromUrdf(joint):
    name = joint.attrib['name']
    jtype = joint.attrib['type']
    origin = joint.find('origin')
    trn = transf.translation_matrix(vec(origin.attrib['xyz']))
    rot = transf.euler_matrix(*vec(origin.attrib['rpy']), axes='sxyz')
    limit = joint.find('safety_controller')
    axis = joint.find('axis')
    if not axis is None:
        axis = vec(axis.attrib['xyz'])
    if jtype == 'continuous':
        limits = (-math.pi, math.pi)
    elif jtype == 'fixed':
        limits = None
    else:
        assert limit is not None and limit.attrib['soft_lower_limit'] is not None
        limits = (float(limit.attrib['soft_lower_limit']),
                  float(limit.attrib['soft_upper_limit']))
    return Joint.subClasses[jtype](name, Transform(np.dot(trn, rot)), limits, axis)


