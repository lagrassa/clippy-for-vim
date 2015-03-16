import pdb
import numpy as np
import math
import windowManager3D as wm
import copy
import util
import shapes
from miscUtil import isGround
from dist import UniformDist,  DeltaDist, chiSqFromP
from objects import World, WorldState
from pr2Robot import PR2, pr2Init, makePr2Chains
from planGlobals import debugMsg, debugDraw, debug, pause
from pr2Fluents import Holding, GraspFace, Grasp, Conf, Pose
from pr2Util import ObjGraspB, ObjPlaceB, shadowName
from fbch import getMatchingFluents, inHeuristic
from belief import B, Bd

Ident = util.Transform(np.eye(4))            # identity transform

########   Make fake belief update have lower variance on pick

################################################################
## Beliefs about the world
## This should be independent of how confs and poses are represented
################################################################


# This class does not keep state except for caches shared by all
# instances of PBS
# Also, fixed properties like object sizes, etc.
class BeliefContext:
    def __init__(self, world, confSamples = [], roadMap = None):
        # world does not hold state either, it has robot, shapes, etc
        self.world = world
        self.roadMap = roadMap
        # caching...
        self.objectShadowCache = {}
        self.pathObstCache = {}
        # Initialize generator caches
        self.genCaches = {}
        for gen in ['pickGen', 'placeGen', 'placeInGen', 'lookGen', 'clearGen', 'getShadowWorld']:
            self.genCaches[gen] = {}
    def __repr__(self):
        return "Belief(%s)"%(str(self.world))
    __str__ = __repr__

# Representation of belief state mostly used for planning
# Marginal distributions over object poses, relative to the robot
cdef class PBS:
    def __init__(self, beliefContext, held=None, conf=None,
                 graspB=None, fixObjBs=None, moveObjBs=None, regions=[], domainProbs=None):
        self.beliefContext = beliefContext
        self.conf = conf or None
        self.held = held or \
          {'left': DeltaDist('none'), 'right': DeltaDist('none')}
        # Keep graspB only for mode of held - value is ObjGraspB instance
        self.graspB = graspB or {'left':None, 'right':None}
        self.fixObjBs = fixObjBs or {}            # {obj: objPlaceB}
        self.moveObjBs = moveObjBs or {}          # {obj: objPlaceB}
        self.regions = regions
        self.pbs = self
        # cache
        self.shadowWorld = None                   # cached obstacles
        self.shadowProb = None                    # shadow probability
        self.avoidShadow = None                   # shadows to avoid, in cached obstacles
        self.heuristic = None                     # the robot shape depends on heuristic
        self.domainProbs = domainProbs

    cpdef getWorld(self):
        return self.beliefContext.world
    cpdef getRobot(self):
        return self.beliefContext.world.robot
    cpdef getRoadMap(self):
        return self.beliefContext.roadMap
    cpdef getObjectShapeAtOrigin(self, obj):
        return self.beliefContext.world.getObjectShapeAtOrigin(obj)

    cpdef awayRegions(self):
        return [r for r in self.regions if r[:5] == 'table']

    cpdef getPlaceB(self, obj, face = None, default = True):
        placeB = self.fixObjBs.get(obj, None) or self.moveObjBs.get(obj, None)
        if placeB and (face is None or face == placeB.support.mode()):
            return placeB
        elif default:
            return self.defaultPlaceB(obj)
        else:
            return None
    cpdef defaultPlaceB(self, obj):
        world = self.getWorld()
        fr = world.getFaceFrames(obj)
        return ObjPlaceB(obj, fr, UniformDist(range(len(fr))), Ident,
                         4*(100.0,))
    
    cpdef getGraspB(self, obj, hand, face = None, default = True):
        if obj == self.held[hand].mode():
            graspB = self.graspB[hand]
            if face is None or face == graspB.grasp.mode():
                return graspB
            elif default:
                return self.defaultGraspB(obj)
            else:
                return None
        elif default:
            return self.defaultGraspB(obj)
        else:
            return None
    cpdef defaultGraspB(self, obj):
        desc = self.getWorld().getGraspDesc(obj)
        return ObjGraspB(obj, desc, UniformDist(range(len(desc))), Ident, 4*(100.0,))

    cpdef getPlacedObjBs(self):
        objectBs = {}
        objectBs.update(self.fixObjBs)
        objectBs.update(self.moveObjBs)
        for hand in ('left', 'right'):
            heldObj = self.held[hand].mode()
            if heldObj != 'none' and heldObj in objectBs:
                del objectBs[heldObj]
        return objectBs

    cpdef getHeld(self, hand):
        return self.held[hand]

    cpdef copy(self):
        return PBS(self.beliefContext, self.held.copy(), self.conf.copy(),
                   self.graspB.copy(), self.fixObjBs.copy(), self.moveObjBs.copy(),
                   self.regions, self.domainProbs)

    def objectsInPBS(self):
        objects = []
        for held in self.held.values():
            if held and held.mode() != 'none':
                objects.append(held.mode())
        objects.extend(self.fixObjBs.keys())
        objects.extend(self.moveObjBs.keys())
        return set(objects)

    cpdef updateFromAllPoses(self, goalConds, updateHeld=True, updateConf=True):
        initialObjects = self.objectsInPBS()
        world = self.getWorld()
        if updateHeld:
            (self.held, self.graspB) = \
                        getHeldAndGraspBel(goalConds, world.getGraspDesc,
                                           self.held, self.graspB)
            for gB in self.graspB.values():
                if gB: self.excludeObjs([gB.obj])
        if updateConf:
            self.conf = getConf(goalConds, self.conf)
        self.fixObjBs = getAllPoseBels(goalConds, world.getFaceFrames,
                                       self.getPlacedObjBs())
        self.moveObjBs = {}
        self.shadowWorld = None
        finalObjects = self.objectsInPBS()
        if debug('conservation') and initialObjects != finalObjects:
            print 'Failure of conservation'
            print '    initial', sorted(list(initialObjects))
            print '    final', sorted(list(finalObjects))
        return self
    
    cpdef updateFromGoalPoses(self, goalConds, updateHeld=True, updateConf=True):
        initialObjects = self.objectsInPBS()
        world = self.getWorld()
        if updateHeld:
            (self.held, self.graspB) = \
                        getHeldAndGraspBel(goalConds, world.getGraspDesc,
                                           self.held, self.graspB)
            for gB in self.graspB.values():
                if gB: self.excludeObjs([gB.obj])
        if updateConf:
            self.conf = getConf(goalConds, self.conf)
        world = self.getWorld()
        self.fixObjBs.update(getGoalPoseBels(goalConds, world.getFaceFrames))
        self.moveObjBs = dict([(o, p) for (o, p) \
                               in self.getPlacedObjBs().iteritems() \
                               if o not in self.fixObjBs])
        self.shadowWorld = None
        finalObjects = self.objectsInPBS()
        if debug('conservation') and initialObjects != finalObjects:
            print 'Failure of conservation'
            print '    initial', sorted(list(initialObjects))
            print '    final', sorted(list(finalObjects))
        return self

    cpdef updateHeld(self, obj, face, graspD, hand, delta = None):
        desc = self.getWorld().getGraspDesc(obj) \
          if obj != 'none' else []
        og = ObjGraspB(obj, desc, DeltaDist(face), graspD, delta = delta)
        self.held[hand] = DeltaDist(obj) # !! Is this rigt??
        self.graspB[hand] = og
        self.excludeObjs([obj])
        self.shadowWorld = None
        return self

    cpdef updateHeldBel(self, graspB, hand):
        self.graspB[hand] = graspB
        if graspB is None:
            self.held[hand] = DeltaDist('none')
        else:
            self.held[hand] = DeltaDist(graspB.obj) # !! Is this rigt??
        if graspB:
            self.excludeObjs([graspB.obj])
        self.shadowWorld = None
        return self
    
    cpdef updateConf(self, c):
        self.conf = c
        if self.shadowWorld:
            self.shadowWorld.setRobotConf(c)
        return self

    cpdef updatePermObjPose(self, objPlace):
        obj = objPlace.obj
        if obj in self.moveObjBs:
            del self.moveObjBs[obj]
        for hand in ('left', 'right'):
            if self.held[hand].mode() == objPlace.obj:
                self.updateHeldBel(None, hand)
        self.fixObjBs[obj] = objPlace
        self.shadowWorld = None
        return self

    cpdef updateObjB(self, objPlace):
        obj = objPlace.obj
        if obj in self.moveObjBs:
            self.moveObjBs[obj] = objPlace
        elif obj in self.fixObjBs:
            self.fixObjBs[obj] = objPlace
        else:
            assert None
        self.shadowWorld = None

    cpdef excludeObjs(self, objs):
        for obj in objs:
            if obj in self.fixObjBs: del self.fixObjBs[obj]
            if obj in self.moveObjBs: del self.moveObjBs[obj]
        self.shadowWorld = None
        return self

    cpdef extendFixObjBs(self, objBs, objShapes):
        if not objBs: return self       # no change
        # To extend objects we need to copy beliefContext and change world.
        bc = copy.copy(self.beliefContext)
        bc.world = self.getWorld().copy()
        bs = self.copy()
        bs.beliefContext = bc
        for obj, objShape in zip(objBs, objShapes):
            bc.world.addObjectShape(objShape)
            bs.excludeObjs([obj])
            bs.fixObjBs[obj] = objBs[obj]
        return bs
    
    cpdef getShadowWorld(self, prob, avoidShadow = []):
        if self.shadowWorld and self.shadowProb == prob \
           and self.heuristic == inHeuristic \
           and set(avoidShadow) == set(self.avoidShadow):
            # print 'same shadowWorld'
            return self.shadowWorld
        else:
            cache = self.beliefContext.genCaches['getShadowWorld']
            key = (self.items(), inHeuristic)
            if key in cache:
                (self.shadowWorld, self.shadowProb, self.avoidShadow, self.heuristic) = cache[key]
                # print 'cached shadowWorld'
                return self.shadowWorld
        # print 'new shadowWorld'
        # The world holds objects, but not poses or shapes
        w = self.getWorld().copy()
        # the shadow world is a WorldState.  Cache it.
        self.shadowWorld = WorldState(w)
        self.shadowProb = prob
        self.avoidShadow = avoidShadow
        self.heuristic = inHeuristic
        # Add the objects and shadows
        sw = self.shadowWorld
        for (obj, objB) in self.getPlacedObjBs().iteritems():
            # The pose in the world is for the origin frame.
            objPose = objB.objFrame()
            # object is already in world, add it to sw
            sw.setObjectPose(obj, objPose)
            faceFrame = objB.faceFrames[objB.support.mode()]
            # Shadow relative to Identity pose
            shadow = self.objShadow(obj, True, prob, objB, faceFrame)
            if debug('getShadowWorld'):
                print 'objB', objB
                print obj, 'origin\n', sw.objectShapes[obj].origin()
                print obj, 'shadow\n', shadow.bbox()
                shadow.draw('W', 'brown')
                print obj, 'shadow origin\n', shadow.origin().matrix
                print obj, 'support pose\n', objB.poseD.mode().matrix
                print obj, 'origin pose\n', objB.objFrame().matrix
                raw_input('Shadow for %s'%obj)
            w.addObjectShape(shadow)
            sw.setObjectPose(shadow.name(), objPose)
            if obj in self.fixObjBs and self.domainProbs and \
                   all([x <= y for (x,y) in zip(objB.poseD.var, self.domainProbs.obsVarTuple)]):
                sw.fixedObjects.add(shadow.name())
            if obj in avoidShadow:      # can't collide with these shadows
                sw.fixedObjects.add(shadow.name())
            if  obj in self.fixObjBs:   # can't collide with these objects
                sw.fixedObjects.add(obj)
        # Add robot
        # !! Fix this
        sw.robot = PR2('MM', makePr2Chains('PR2', w.workspace, new=False))
        robot = sw.robot
        robot.nominalConf = w.robot.nominalConf
        # robot.nominalConf = nominalConf
        # Add shadow for held object
        for hand in ('left', 'right'):
            heldObj = self.held[hand].mode()
            if heldObj != 'none':
                assert self.graspB[hand] and self.graspB[hand].obj == heldObj
                graspIndex = self.graspB[hand].grasp.mode()
                graspDesc = self.graspB[hand].graspDesc[graspIndex]
                # faceFrame = graspDesc.frame.compose(self.graspB[hand].poseD.mode().inverse())
                faceFrame = graspDesc.frame.compose(self.graspB[hand].poseD.mode().inverse())
                shadow = self.objShadow(heldObj, True, prob, self.graspB[hand], faceFrame)
                fingerFrame = robot.fingerSupportFrame(hand, graspDesc.dz*2)
                graspShadow = shadow.applyTrans(fingerFrame)
                robot.attachRel(graspShadow, sw, hand)
                if debug('getShadowWorldGrasp'):
                    print 'faceFrame\n', faceFrame.matrix
                    print 'shadow\n', shadow.bbox()
                    print 'fingerFrame\n', fingerFrame.matrix
                    print 'graspShadow\n', graspShadow.bbox()
                    graspShadow.draw('W', 'red')
                    print 'attached origin\n', sw.attached[hand].origin().matrix
                    raw_input('Attach?')
                sw.held[hand] = heldObj
        sw.setRobotConf(self.conf)
        cache[key] = (sw, prob, avoidShadow, inHeuristic)
        return sw

    # Shadow over POSE variation.  Should only do finite number of poseVar/poseDelta values.
    cpdef objShadow(self, obj, shName, prob, poseBel, faceFrame):
        shape = self.getObjectShapeAtOrigin(obj)
        key = (shape, shName, prob, poseBel, faceFrame)
        shadow = self.beliefContext.objectShadowCache.get(key, None)
        if shadow:
            return shadow
        name = shadowName(shape) if shName else shape.name()
        # Origin * Support = Pose => Origin = Pose * Support^-1
        frame = faceFrame.inverse()     # pose is indentity
        sh = shape.applyLoc(frame)      # the shape with the specified support
        shadow = makeShadow(sh, prob, poseBel, name=name)
        self.beliefContext.objectShadowCache[key] = shadow
        debugMsg('objShadow', key, ('->', shadow.bbox()))
        return shadow

    cpdef draw(self, p = 0.9, win = 'W', clear=True):
        if clear: wm.getWindow(win).clear()
        if self.shadowWorld:            # don't recompute
            self.shadowWorld.draw(win)
        else:
            self.getShadowWorld(p).draw(win)

    cpdef items(self):
        return (frozenset(self.held.items()),
                frozenset(self.graspB.items()),
                self.conf,
                frozenset(self.fixObjBs.items()),
                frozenset(self.moveObjBs.items()))

    def __richcmp__(self, other, int op):
        if not (other and isinstance(other, PBS)):
            return True if op == 3 else False
        if op == 2:
            ans = self.items() == other.items()
        elif op == 3:
            ans = self.items() != other.items()
        else:
            ans = False
        return ans
    def __hash__(self):
        return hash(self.items())
    def __repr__(self):
        return 'BS:'+str(self.items())
    __str__ = __repr__

####################
# Shadow computation
####################

cpdef list shadowWidths(tuple variance, tuple delta, float probability):
    numStdDevs =  math.sqrt(chiSqFromP(1-probability, 2))
    assert all([v >= 0 for v in variance])
    return [numStdDevs*(v**0.5)+d for (v,d) in zip(variance, delta)]

cpdef sigmaPoses(float prob, poseD, poseDelta):
    cdef list widths = shadowWidths(poseD.variance(), poseDelta, prob)
    if debug('getShadowWorld'):
        print 'shadowWidths', widths
    cdef int n = len(widths)
    cdef list offsets = []
    (wx, wy, _, wt) = widths
    offsets.append([-wx, 0, 0, -wt])
    offsets.append([-wx, 0, 0, wt])
    offsets.append([wx, 0, 0, -wt])
    offsets.append([wx, 0, 0, wt])
    offsets.append([0, -wy, 0, -wt])
    offsets.append([0, -wy, 0, wt])
    offsets.append([0, wy, 0, -wt])
    offsets.append([0, wy, 0, wt])
    cdef list poses = []
    # poseTuple = poseD.mode().xyztTuple()
    for offset in offsets:
        # offPoseTuple = [c+o for c,o in zip(poseTuple, offset)]
        offPoseTuple = offset
        poses.append(util.Pose(*offPoseTuple))
    return poses

cpdef makeShadow(shape, prob, bel, name=None):
    shParts = []
    poses = sigmaPoses(prob, bel.poseD, bel.delta)
    if debug('getShadowWorld'):
        print 'sigma poses for', shape.name()
        wm.getWindow('W').clear()
    color = shape.properties.get('color', 'gray')

    for part in shape.parts():
        shp = []
        for pose in poses:
            shp.append(part.applyTrans(pose))
            if debug('getShadowWorld'):
                shp[-1].draw('W', 'cyan')
        # Note xyPrim !!
        shParts.append(shapes.Shape(shp, shape.origin(),
                                    name=part.name(), color=color).xyPrim())
        if debug('getShadowWorld'):
            shParts[-1].draw('W', 'brown')
            raw_input('Next part?')
    if len(shParts) == 1:
        if name:
            shParts[0].properties['name'] = name
        if debug('getShadowWorld'):
            print 'shadow name', shParts[0].name()
            raw_input('1 part shadow, Ok?')
        return shParts[0]
    else:
        if debug('getShadowWorld'):
            raw_input('multiple part shadow, Ok?')
        return shapes.Shape(shParts, shape.origin(),
                            name=name or shape.name(),
                            color=color)

cpdef LEQ(x, y):
    return all([x1 <= y1 for (x1, y1) in zip(x,y)])

####################
# Examining the fluents
####################

# Returns dictionary of obj poses, extracted from goalConds.
cpdef getGoalPoseBels(goalConds, getFaceFrames):
    if not goalConds: return {}
    fbs = getMatchingFluents(goalConds,
                            B([Pose(['Obj', 'Face']), 'Mu', 'Var', 'Delta', 'P'], True))

    ans = dict([(b['Obj'], ObjPlaceB(b['Obj'],
                                      getFaceFrames(b['Obj']), # !! ??
                                      DeltaDist(b['Face']),
                                      util.Pose(* b['Mu']),
                                      b['Var'], b['Delta'])) \
                 for (f, b) in fbs if isGround(b.values())])
    return ans

# Overrides is a list of fluents
# Returns dictionary of objects in belief state, overriden as appropriate.
cpdef getAllPoseBels(overrides, getFaceFrames, curr):
    if not overrides: return curr
    objects = curr.copy()
    for (o, p) in getGoalPoseBels(overrides, getFaceFrames).iteritems():
        objects[o] = p
    return objects

cpdef getConf(overrides, curr):
    if not overrides: return curr
    fbs = [(f, b) for (f, b) \
           in getMatchingFluents(overrides,
                                 Conf(['Mu', 'Delta'], True)) if f.isGround()]
    assert len(fbs) <= 1, 'Too many Conf fluents in conditions'
    conf = curr
    if len(fbs) == 1:
        ((f, b),) = fbs
        if isGround(b.values()):
            conf = b['Mu']
    return conf

cpdef getHeldAndGraspBel(overrides, getGraspDesc, currHeld, currGrasp):
    if not overrides: return (currHeld, currGrasp)

    # Figure out what we're holding
    hbs = getMatchingFluents(overrides, Bd([Holding(['H']), 'Obj', 'P'], True))
    held = currHeld.copy()
    objects = {}
    for (f, b) in hbs:
        if isGround(b.values()):  
            hand = b['H']
            assert hand in ('left', 'right') and not hand in objects
            held[hand] = DeltaDist(b['Obj'])   # !! is this right?
            objects[hand] = b['Obj']

    # Grasp faces
    gfbs = getMatchingFluents(overrides,
                            Bd([GraspFace(['Obj', 'H']), 'Face', 'P'], True))
    faces = {}
    for (f, b) in gfbs:
        if isGround(b.values()):
            (hand, obj, face) = (b['H'], b['Obj'], b['Face'])
            if obj == 'none' or (not hand in objects) or obj != objects[hand]:
                continue
            assert hand in ('left', 'right') and not hand in faces
            faces[hand] = face

    # Grasps
    gbs = getMatchingFluents(overrides,
            B([Grasp(['Obj', 'H', 'Face']), 'Mu', 'Var', 'Delta', 'P'], True))
    grasps = {}
    graspB = currGrasp.copy()
    for (f, b) in gbs:
        if  isGround(b.values()):
            (hand, obj, face, mu, var, delta) = \
              (b['H'], b['Obj'], b['Face'], b['Mu'], b['Var'], b['Delta'])
            if obj == 'none' or (not hand in objects) or (not hand in faces) or \
                   obj != objects[hand] or face != faces[hand]:
                continue
            assert hand in ('left', 'right') and not hand in grasps
            if obj != 'none':
                graspB[hand] = ObjGraspB(obj, getGraspDesc(obj),
                                         DeltaDist(face), util.Pose(*mu), var, delta)

    return (held, graspB)
