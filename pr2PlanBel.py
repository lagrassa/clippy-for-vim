import pdb
import numpy as np
import math
import windowManager3D as wm
import copy
import util
import shapes
from miscUtil import isGround
from dist import UniformDist,  DeltaDist
from objects import World, WorldState
from pr2Robot import PR2, pr2Init, makePr2Chains
from planGlobals import debugMsg, debugDraw, debug, pause
from pr2Fluents import Holding, GraspFace, Grasp, Conf, Pose
from pr2Util import ObjGraspB, ObjPlaceB, shadowName, shadowWidths, objectName
import fbch
from fbch import getMatchingFluents
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
        for gen in ['pickGen', 'placeGen', 'placeInGen', 'lookGen', 'clearGen',
                    'easyGraspGen', 'getShadowWorld']:
            self.genCaches[gen] = {}
    def __repr__(self):
        return "Belief(%s)"%(str(self.world))
    __str__ = __repr__

# Representation of belief state mostly used for planning
# Marginal distributions over object poses, relative to the robot
class PBS:
    def __init__(self, beliefContext, held=None, conf=None,
                 graspB=None, fixObjBs=None, moveObjBs=None, regions=[],
                 domainProbs=None, useRight=True, avoidShadow=[]):
        self.beliefContext = beliefContext
        self.conf = conf or None
        self.held = held or \
          {'left': DeltaDist('none'), 'right': DeltaDist('none')}
        # Keep graspB only for mode of held - value is ObjGraspB instance
        self.graspB = graspB or {'left':None, 'right':None}
        self.fixObjBs = fixObjBs or {}            # {obj: objPlaceB}
        self.moveObjBs = moveObjBs or {}          # {obj: objPlaceB}
        self.fixGrasp = {'left':False, 'right':False} # the graspB can be changed
        self.fixHeld = {'left':False, 'right':False} # whether the held can be changed
        self.regions = regions
        self.pbs = self
        self.useRight = useRight
        self.avoidShadow = avoidShadow  # shadows to avoid
        self.domainProbs = domainProbs
        # cache
        self.shadowWorld = None                   # cached obstacles
        self.shadowProb = None

    def reset(self):
        self.shadowWorld = None
        self.shadowProb = None

    def internalCollisionCheck(self):
        ws = self.getShadowWorld(0.0)   # minimal shadow
        # First check the robot for hard collisions
        confViols = self.beliefContext.roadMap.confViolations(self.conf,
                                                              self, 0.)
        if confViols == None or confViols.obstacles or \
          confViols.heldObstacles[0] or confViols.heldObstacles[1]:
            raise Exception, 'Collision with robot: '+name
        # Now for shadow collisions;  reduce the shadow if necessary
        confViols = self.beliefContext.roadMap.confViolations(self.conf,
                                                          self, .98)
        shadows = confViols.allShadows()
        while shadows:
            if shadows:
                print 'Robot collides with shadows', shadows
                self.draw(0.98, 'W')
                raw_input('Try to fix?')
                # Divide variance in half.  Very crude.  Should find the
                # max variance that does not result in a shadow colliion.
                for sh in shadows:
                    obj = objectName(sh)
                    pB = self.getPlaceB(obj)
                    var = pB.poseD.variance()
                    print 'oldVar', var
                    pB.poseD.var = tuple(v/2.0 for v in var)
                    print 'newVar', pB.poseD.variance()
                self.shadowWorld = None
            confViols = self.beliefContext.roadMap.confViolations(self.conf,
                                                          self, .98)
            shadows = confViols.allShadows()

        # Finally, look for object-object collisions
        objShapes = [o for o in ws.getObjectShapes() \
                     if 'shadow' not in o.name()]
        n = len(objShapes)
        for index in range(n):
            shape = objShapes[index]
            if debugMsg('collisionCheck'):
                shape.draw('W', 'black')
            for index2 in range(index+1, n):
                shape2 = objShapes[index2]
                # Ignore collisions between fixed objects
                if shape.collides(shape2) and \
                  not (shape.name() in self.fixObjBs and \
                       shape2.name() in self.fixObjBs):
                    for shape in objShapes: shape.draw('W', 'black')
                    shape2.draw('W', 'magenta')
                    raise Exception, 'Object-Object collision: '+ \
                                     shape.name()+' - '+shape2.name()
        debugMsg('collisionCheck')

    def getWorld(self):
        return self.beliefContext.world
    def getRobot(self):
        return self.beliefContext.world.robot
    def getRoadMap(self):
        return self.beliefContext.roadMap
    def getObjectShapeAtOrigin(self, obj):
        return self.beliefContext.world.getObjectShapeAtOrigin(obj)

    def awayRegions(self):
        return frozenset([r for r in self.regions if r[:5] == 'table'])

    def getPlaceB(self, obj, face = None, default = True):
        placeB = self.fixObjBs.get(obj, None) or self.moveObjBs.get(obj, None)
        if placeB and (face is None or face == placeB.support.mode()):
            return placeB
        elif default:
            return self.defaultPlaceB(obj)
        else:
            return None
    def defaultPlaceB(self, obj):
        world = self.getWorld()
        fr = world.getFaceFrames(obj)
        return ObjPlaceB(obj, fr, UniformDist(range(len(fr))), Ident,
                         4*(100.0,))
    
    def getGraspB(self, obj, hand, face = None, default = True):
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
    def defaultGraspB(self, obj):
        desc = self.getWorld().getGraspDesc(obj)
        return ObjGraspB(obj, desc, UniformDist(range(len(desc))), Ident, 4*(100.0,))

    def getPlacedObjBs(self):
        objectBs = {}
        objectBs.update(self.fixObjBs)
        objectBs.update(self.moveObjBs)
        for hand in ('left', 'right'):
            heldObj = self.held[hand].mode()
            if heldObj != 'none' and heldObj in objectBs:
                del objectBs[heldObj]
        return objectBs

    def getHeld(self, hand):
        return self.held[hand]

    def copy(self):
        return PBS(self.beliefContext, self.held.copy(), self.conf.copy(),
                   self.graspB.copy(), self.fixObjBs.copy(), self.moveObjBs.copy(),
                   self.regions, self.domainProbs, self.useRight, self.avoidShadow[:])

    def objectsInPBS(self):
        objects = []
        for held in self.held.values():
            if held and held.mode() != 'none':
                objects.append(held.mode())
        objects.extend(self.fixObjBs.keys())
        objects.extend(self.moveObjBs.keys())
        return set(objects)

    def updateAvoidShadow(self, avoidShadow):
        self.avoidShadow = avoidShadow
        return self

    def addAvoidShadow(self, avoidShadow):
        for s in avoidShadow:
            if s not in self.avoidShadow:
                self.avoidShadow = self.avoidShadow + [s]
        return self

    # Makes all objects permanent
    def updateFromAllPoses(self, goalConds,
                           updateHeld=True, updateConf=True, permShadows=False):
        initialObjects = self.objectsInPBS()
        world = self.getWorld()
        if updateHeld:
            (held, graspB) = \
                   getHeldAndGraspBel(goalConds, world.getGraspDesc)
            for h in ('left', 'right'):
                if held[h]:
                    self.fixHeld[h] = True
                    self.held[h] = held[h]
                if graspB[h]:
                    self.fixGrasp[h] = True
                    self.graspB[h] = graspB[h]
            for gB in self.graspB.values():
                if gB: self.excludeObjs([gB.obj])
        if updateConf:
            self.conf = getConf(goalConds, self.conf)
        self.fixObjBs = getAllPoseBels(goalConds, world.getFaceFrames,
                                       self.getPlacedObjBs())
        self.moveObjBs = {}
        # The shadows of Pose(obj) in the cond are also permanent
        if permShadows:
            self.updateAvoidShadow(getPoseObjs(goalConds))
        self.reset()
        finalObjects = self.objectsInPBS()
        if debug('conservation') and initialObjects != finalObjects:
            print 'Failure of conservation'
            print '    initial', sorted(list(initialObjects))
            print '    final', sorted(list(finalObjects))
            raw_input('conservation')
        return self
    
    # Makes objects mentioned in the goal permanent
    def updateFromGoalPoses(self, goalConds,
                            updateHeld=True, updateConf=True, permShadows=False):
        initialObjects = self.objectsInPBS()
        world = self.getWorld()
        if updateHeld:
            (held, graspB) = \
                   getHeldAndGraspBel(goalConds, world.getGraspDesc)
            for h in ('left', 'right'):
                if held[h]:
                    self.fixHeld[h] = True
                    self.held[h] = held[h]
                if graspB[h]:
                    self.fixGrasp[h] = True
                    self.graspB[h] = graspB[h]
            for gB in self.graspB.values():
                if gB: self.excludeObjs([gB.obj])
        if updateConf:
            self.conf = getConf(goalConds, self.conf)
        world = self.getWorld()
        self.fixObjBs.update(getGoalPoseBels(goalConds, world.getFaceFrames))
        self.moveObjBs = dict([(o, p) for (o, p) \
                               in self.getPlacedObjBs().iteritems() \
                               if o not in self.fixObjBs])
        # The shadows of Pose(obj) in the cond are also permanent
        if permShadows:
            self.updateAvoidShadow(getPoseObjs(goalConds))
        self.reset()
        finalObjects = self.objectsInPBS()
        if debug('conservation') and initialObjects != finalObjects:
            print 'Failure of conservation'
            print '    initial', sorted(list(initialObjects))
            print '    final', sorted(list(finalObjects))
            raw_input('conservation')
        return self

    def updateHeld(self, obj, face, graspD, hand, delta = None):
        desc = self.getWorld().getGraspDesc(obj) \
          if obj != 'none' else []
        og = ObjGraspB(obj, desc, DeltaDist(face), graspD, delta = delta)
        self.held[hand] = DeltaDist(obj) # !! Is this rigt??
        self.graspB[hand] = og
        self.excludeObjs([obj])
        self.reset()
        return self

    def updateHeldBel(self, graspB, hand):
        self.graspB[hand] = graspB
        if graspB is None:
            self.held[hand] = DeltaDist('none')
        else:
            self.held[hand] = DeltaDist(graspB.obj) # !! Is this rigt??
        if graspB:
            self.excludeObjs([graspB.obj])
        self.reset()
        return self
    
    def updateConf(self, c):
        self.conf = c
        if self.shadowWorld:
            self.shadowWorld.setRobotConf(c)
        return self

    def updatePermObjPose(self, objPlace):
        obj = objPlace.obj
        if obj in self.moveObjBs:
            del self.moveObjBs[obj]
        for hand in ('left', 'right'):
            if self.held[hand].mode() == objPlace.obj:
                self.updateHeldBel(None, hand)
        self.fixObjBs[obj] = objPlace
        self.reset()
        return self

    def updateObjB(self, objPlace):
        obj = objPlace.obj
        if obj in self.moveObjBs:
            self.moveObjBs[obj] = objPlace
        elif obj in self.fixObjBs:
            self.fixObjBs[obj] = objPlace
        else:
            assert None
        self.reset()

    def excludeObjs(self, objs):
        for obj in objs:
            if obj in self.fixObjBs: del self.fixObjBs[obj]
            if obj in self.moveObjBs: del self.moveObjBs[obj]
        self.reset()
        return self

    def extendFixObjBs(self, objBs, objShapes):
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

    # arg can also be a graspB since they share relevant methods
    def shadowPair(self, objB, faceFrame, prob):
        def shLE(w1, w2):
            return all([w1[i] <= w2[i] for i in (0,1,3)])
        obj = objB.obj
        graspable = obj in self.getWorld().graspDesc
        objBMinDelta = (0.002, 0.002, 0.0, 0.004)
         # 2 looks
        objBMinVarGrasp = tuple([x**2/2*x for x in self.domainProbs.obsVarTuple])
        objBMinVarStatic = tuple([x**2 for x in self.domainProbs.odoError])
        objBMinProb = 0.95
        # The irreducible shadow
        objBMinVar = objBMinVarGrasp if graspable else objBMinVarStatic
        objBMin = objB.modifyPoseD(var=objBMinVar)
        objBMin.delta = objBMinDelta

        shWidth = shadowWidths(objB.poseD.var, objB.delta, prob)
        minShWidth = shadowWidths(objBMinVar, objBMinDelta, objBMinProb)
        if shLE(shWidth, minShWidth):
            # If the "min" shadow is wider than actual shadow, make them equal
            objB = objBMin
            prob = objBMinProb

        if debug('shadowWidths'):
            print 'obj', obj, 'graspable', graspable, 'objB == objBMin', objB == objBMin
            print 'shWidth', shWidth
            print 'minShWidth', minShWidth

        # Shadows relative to Identity pose
        shadow = self.objShadow(obj, shadowName(obj), prob, objB, faceFrame)
        shadowMin = self.objShadow(obj, obj, objBMinProb, objBMin, faceFrame) # use obj name

        if debug('getShadowWorld'):
            print 'objB', objB
            shadow.draw('W', 'gray')
            print 'objBMin', objBMin
            shadowMin.draw('W', 'brown')
            print 'max shadow widths', shWidth
            print 'min shadow widths', minShWidth
            # print obj, 'origin\n', sw.objectShapes[obj].origin()
            print obj, 'shadow\n', shadow.bbox()
            print obj, 'shadow origin\n', shadow.origin().matrix
            print obj, 'support pose\n', objB.poseD.mode().matrix
            print obj, 'origin pose\n', objB.objFrame().matrix
            raw_input('Shadows for %s (brown in inner, gray is outer)'%obj)

        return shadowMin, shadow
    
    def getShadowWorld(self, prob):
        if self.shadowWorld and self.shadowProb == prob:
            return self.shadowWorld
        else:
            cache = self.beliefContext.genCaches['getShadowWorld']
            key = (self.items(), prob)
            if key in cache:
                ans = cache.get(key, None)
                if ans != None:
                    self.shadowWorld = ans
                    self.shadowProb = prob
                    # print 'cached shadowWorld'
                    return self.shadowWorld
        # The world holds objects, but not poses or shapes
        w = self.getWorld().copy()
        # the shadow world is a WorldState.  Cache it.
        self.shadowWorld = WorldState(w)
        self.shadowProb = prob
        # Add the objects and shadows
        sw = self.shadowWorld
        for (obj, objB) in self.getPlacedObjBs().iteritems():
            # The pose in the world is for the origin frame.
            objPose = objB.objFrame()
            faceFrame = objB.faceFrames[objB.support.mode()]
            shadowMin, shadow = self.shadowPair(objB, faceFrame, prob)

            w.addObjectShape(shadow)
            w.addObjectShape(shadowMin)
            sw.setObjectPose(shadow.name(), objPose)
            sw.setObjectPose(shadowMin.name(), objPose)

            if obj in self.avoidShadow:      # can't collide with these shadows
                sw.fixedObjects.add(shadow.name())
            if  obj in self.fixObjBs:   # can't collide with these objects
                sw.fixedObjects.add(shadowMin.name())
        # Add robot
        sw.robot = self.getRobot()
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
                # Transform from object origin to grasp surface
                # The graspDesc frame is relative to object origin
                # The graspB pose encodes finger tip relative to graspDesc frame
                faceFrame = graspDesc.frame.compose(self.graspB[hand].poseD.mode())
                # Create shadow pair and attach both to robot
                shadowMin, shadow = self.shadowPair(self.graspB[hand], faceFrame, prob)     
                # graspShadow is expressed relative to wrist and attached to arm
                # fingerFrame should map shodow (supported at graspDesc) into wrist frame
                fingerFrame = robot.fingerSupportFrame(hand, graspDesc.dz*2)
                graspShadow = shadow.applyTrans(fingerFrame)
                graspShadowMin = shadowMin.applyTrans(fingerFrame)
                # shadowMin will stand in for object
                heldShape = shapes.Shape([graspShadowMin, graspShadow],
                                         graspShadowMin.origin(),
                                         name = heldObj,
                                         color = graspShadowMin.properties.get('color', 'black'))
                robot.attachRel(heldShape, sw, hand)
                sw.held[hand] = heldObj
                if debug('getShadowWorldGrasp') and not fbch.inHeuristic:
                    print 'faceFrame\n', faceFrame.matrix
                    print 'shadow\n', shadow.bbox()
                    print 'fingerFrame\n', fingerFrame.matrix
                    print 'graspShadow\n', graspShadow.bbox()
                    graspShadow.draw('W', 'red')
                    print 'attached origin\n', sw.attached[hand].origin().matrix
                    raw_input('Attach?')
        sw.fixedHeld = self.fixHeld
        sw.fixedGrasp = self.fixGrasp
        sw.setRobotConf(self.conf)
        if debug('getShadowWorldGrasp') and not fbch.inHeuristic:
            sw.draw('W')
        cache[key] = sw
        return sw

    # Shadow over POSE variation.  Should only do finite number of poseVar/poseDelta values.
    def objShadow(self, obj, shName, prob, poseBel, faceFrame):
        shape = self.getObjectShapeAtOrigin(obj)
        color = shape.properties.get('color', None) or \
                (shape.parts() and [s.properties.get('color', None) for s in shape.parts()][0]) or \
                'gray'
        if shName == True or 'shadow' in shName:
            color = 'gray'
        key = (shape, shName, prob, poseBel, faceFrame)
        shadow = self.beliefContext.objectShadowCache.get(key, None)
        if shadow:
            return shadow
        # Origin * Support = Pose => Origin = Pose * Support^-1
        frame = faceFrame.inverse()     # pose is indentity
        sh = shape.applyLoc(frame)      # the shape with the specified support
        shadow = makeShadow(sh, prob, poseBel, name=shName, color=color)
        self.beliefContext.objectShadowCache[key] = shadow
        debugMsg('objShadow', key, ('->', shadow.bbox()))
        return shadow

    def draw(self, p = 0.9, win = 'W', clear=True):
        if clear: wm.getWindow(win).clear()
        self.getShadowWorld(p).draw(win)

    def items(self):
        return (frozenset(self.held.items()),
                frozenset(self.graspB.items()),
                self.conf, frozenset(self.avoidShadow),
                frozenset(self.fixObjBs.items()),
                frozenset(self.moveObjBs.items()))
    def __eq__(self, other):
        return hasattr(other, 'items') and self.items() == other.items()
    def __neq__(self, other):
        return not self == other
    def __hash__(self):
        return hash(self.items())
    def __repr__(self):
        return 'BS:'+str(self.items())
    __str__ = __repr__

####################
# Shadow computation
####################

def sigmaPoses(prob, poseD, poseDelta):
    interpStep = math.pi/4
    def interpAngle(lo, hi):
        if hi - lo <= interpStep:
            return [lo, hi]
        else:
            return interpAngle(lo, 0.5*(lo+hi))[:-1] + \
                   interpAngle(0.5*(lo+hi), hi)
    widths = shadowWidths(poseD.variance(), poseDelta, prob)
    n = len(widths)
    offsets = []
    (wx, wy, _, wt) = widths
    angles = interpAngle(-wt, wt)
    if debug('getShadowWorld'):
        print 'shadowWidths', widths
        print 'angles', angles
    for a in angles: offsets.append([-wx, 0, 0, a])
    for a in angles: offsets.append([wx, 0, 0, a])
    for a in angles: offsets.append([0, -wy, 0, a])
    for a in angles: offsets.append([0, wy, 0, a])
    poses = []
    # poseTuple = poseD.mode().xyztTuple()
    for offset in offsets:
        # offPoseTuple = [c+o for c,o in zip(poseTuple, offset)]
        offPoseTuple = offset
        poses.append(util.Pose(*offPoseTuple))
    return poses

def makeShadow(shape, prob, bel, name=None, color='gray'):
    shParts = []
    poses = sigmaPoses(prob, bel.poseD, bel.delta)
    if debug('getShadowWorld'):
        print 'sigma poses for', shape.name()
    shColor = shape.properties.get('color', color)

    for part in shape.parts():
        wm.getWindow('W').clear()
        shp = []
        for pose in poses:
            shp.append(part.applyTrans(pose))
            if debug('getShadowWorld'):
                shp[-1].draw('W', 'cyan')
        # Note xyPrim !!
        shParts.append(shapes.Shape(shp, shape.origin(),
                                    name=part.name(), color=shColor).xyPrim())
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
                            color=shColor)

def LEQ(x, y):
    return all([x1 <= y1 for (x1, y1) in zip(x,y)])

####################
# Examining the fluents
####################

# Returns dictionary of obj poses, extracted from goalConds.
def getGoalPoseBels(goalConds, getFaceFrames):
    if not goalConds: return {}
    fbs = getMatchingFluents(goalConds,
                            B([Pose(['Obj', 'Face']), 'Mu', 'Var', 'Delta', 'P'], True))

    ans = dict([(b['Obj'], ObjPlaceB(b['Obj'],
                                      getFaceFrames(b['Obj']), # !! ??
                                      DeltaDist(b['Face']),
                                      util.Pose(* b['Mu']),
                                      b['Var'], b['Delta'])) \
                 for (f, b) in fbs if \
                      (isGround(b.values()) and not ('*' in b.values()))])
    return ans

# Overrides is a list of fluents
# Returns dictionary of objects in belief state, overriden as appropriate.
def getAllPoseBels(overrides, getFaceFrames, curr):
    if not overrides: return curr
    objects = curr.copy()
    for (o, p) in getGoalPoseBels(overrides, getFaceFrames).iteritems():
        objects[o] = p
    return objects

def getConf(overrides, curr):
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

def getHeldAndGraspBel(overrides, getGraspDesc):
    held = {'left':None, 'right':None}
    graspB = {'left':None, 'right':None}
    if not overrides: return (held, graspB)

    # Figure out what we're holding
    hbs = getMatchingFluents(overrides, Bd([Holding(['H']), 'Obj', 'P'], True))
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

def getPoseObjs(goalConds):
    pfbs = fbch.getMatchingFluents(goalConds,
                                   B([Pose(['Obj', 'Face']), 'Mu', 'Var', 'Delta', 'P'], True))
    objs = []
    for (pf, pb) in pfbs:
        if isGround(pb.values()):
            objs.append(pb['Obj'])
    return objs

