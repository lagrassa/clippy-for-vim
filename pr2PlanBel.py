import pdb
import numpy as np
import math
import windowManager3D as wm
import copy
import hu
import shapes
import random
from miscUtil import isGround
from dist import UniformDist,  DeltaDist
from objects import World, WorldState
from traceFile import debugMsg, debug
import planGlobals as glob
from pr2Fluents import Holding, GraspFace, Grasp, Conf, Pose, \
     BaseConf, CanReachNB
from planUtil import ObjGraspB, ObjPlaceB
from pr2Util import shadowName, shadowWidths, objectName, supportFaceIndex, \
     PoseD, inside, permanent, pushable, graspable
from fbch import getMatchingFluents
from belief import B, Bd, Cond
from traceFile import tr, trAlways
from transformations import rotation_matrix
from geom import bboxVolume
from miscUtil import isVar

Ident = hu.Transform(np.eye(4))            # identity transform

########   Make fake belief update have lower variance on pick

# total count, current, cache hits, cache misses, new
shadowWorldStats = [0, 0, 0, 0, 0]

objShadowStats = [0, 0]

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
                    'easyGraspGen', 'getShadowWorld', 'pushPath',
                    'pushGen', 'graspConfGen', 'pathObst', 'objectShadow',
                    'confReach']:
            self.genCaches[gen] = {}
    def __repr__(self):
        return "Belief(%s)"%(str(self.world))
    __str__ = __repr__

# Representation of belief state mostly used for planning
# Marginal distributions over object poses, relative to the robot
class PBS:
    def __init__(self, beliefContext, held=None, conf=None,
                 graspB=None, objectBs=None, regions=[],
                 domainProbs=None, avoidShadow=[], base=None,
                 targetConf=None, conditions=set([]), poseModeProbs=None):
        self.beliefContext = beliefContext
        # The components of the state
        # The conf of the robot (fixed?, conf)
        self.conf = conf or (False, None)
        # Required base (x, y, th) or None
        self.base = base                # the required base
        self.targetConf = targetConf
        # The held object in each hand (fixed?, object)
        self.held = held or {'left': (False, 'none'), 'right': (False, 'none')}
        # The grasp parameters (fixed?, PoseD)
        self.graspB = graspB or {'left':(False, None), 'right': (False, None)}
        # The object poses, {obj: (fixed?, objPlaceB)}
        self.objectBs = objectBs or {}
        # Indicate which object shadows are fixed
        self.avoidShadow = avoidShadow  # shadows to avoid
        # List of fluents that must be feasible
        self.conditions = conditions
        # Cache
        self.shadowWorld = None         # cached shadow world
        self.shadowProb = None

        # Additional info
        self.regions = regions
        self.useRight = glob.useRight
        self.pbs = self                 # hack... to impersonate a belief
        self.domainProbs = domainProbs
        self.poseModeProbs = poseModeProbs

    # Access beliefContext
    def genCache(self, tag):
        return self.beliefContext.genCaches[tag]
    def getWorld(self):
        return self.beliefContext.world
    def getRobot(self):
        return self.beliefContext.world.robot
    def getRoadMap(self):
        return self.beliefContext.roadMap
    def getObjectShapeAtOrigin(self, obj):
        return self.beliefContext.world.getObjectShapeAtOrigin(obj)
    def confViolations(self, conf, prob, **keys):
        return self.beliefContext.roadMap.confViolations(conf, self, prob, **keys)

    # Accessors
    def fixObjBs(self):
        return [objB for (fix, objB) in self.objectBs.values() if fix]
    def moveObjBs(self):
        return [objB for (fix, objB) in self.objectBs.values() if not fix]
    def getConf(self):
        return self.conf[1]             # ignore the fix
    def awayRegions(self):
        # This is highly domain specific
        return frozenset([r for r in self.regions if r[:5] == 'table'])
    def defaultGraspB(self, obj):
        desc = self.getWorld().getGraspDesc(obj)
        return ObjGraspB(obj, desc, UniformDist(range(len(desc))), None, Ident, 4*(100.0,))
    def getGraspBForObj(self, obj, hand, face = None, default = True):
        if obj == self.getHeld(hand):
            graspB = self.graspB[hand][1]
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
    def defaultPlaceB(self, obj):
        world = self.getWorld()
        fr = world.getFaceFrames(obj)
        return ObjPlaceB(obj, fr, UniformDist(range(len(fr))), Ident,
                         4*(100.0,))
    def getPlaceB(self, obj, face = None, default = True):
        (fix, placeB) = self.objectBs.get(obj, (False, None))
        if placeB and (face is None or face == placeB.support.mode()):
            return placeB
        elif default:
            return self.defaultPlaceB(obj)
        else:
            return None
    def getPlacedObjBs(self):
        objectBs = {o:pB for (o, (fix,pB)) in self.objectBs.iteritems()}
        for hand in ('left', 'right'):
            heldObj = self.getHeld(hand)
            if heldObj in objectBs:
                del objectBs[heldObj]
        return objectBs
    def getHeld(self, hand):
        return self.held[hand][1]
    def getBase(self):
        return self.base[1] if self.base else None
    def getTargetConf(self):
        return self.targetConf[1] if self.targetConf else None
    def getGraspB(self, hand):
        return self.graspB[hand][1]
    def objectsInPBS(self):
        objects = []
        for (fix, held) in self.held.values():
            if held and held != 'none':
                objects.append(held)
        objects.extend(self.objectBs.keys())
        return set(objects)

    # These return new PBS
    def conditioned(self, goalConds, cond):
        newBS = self.copy()
        newBS = newBS.updateFromConds(goalConds)
        if cond:
            newBS = newBS.updateFromConds(cond, permShadows=True)
        newBS.conditions.update([fluent for fluent in goalConds \
                                 if fluent.isConditional()])
        return newBS

    def copy(self):
        def copyVal(x): return (x[0], x[1].copy())
        return PBS(self.beliefContext, self.held.copy(), copyVal(self.conf),
                   self.graspB.copy(), self.objectBs.copy(),
                   self.regions, self.domainProbs, self.avoidShadow[:],
                   self.base, self.targetConf, self.conditions.copy(),
                   self.poseModeProbs)
    def feasible(self):
        # Check that all the conditions are feasible; applied to Bd
        for fl in self.conditions:
            if not fl.feasiblePBS(self): return False
        return True

    # Updates
    def updateAvoidShadow(self, avoidShadow):
        self.avoidShadow = list(avoidShadow)
        self.reset()
        return self
    def addAvoidShadow(self, avoidShadow):
        for s in avoidShadow:
            if s not in self.avoidShadow:
                self.avoidShadow = self.avoidShadow + [s]
        self.reset()
        return self
    def updateGraspB(self, hand, gB):
        obj = gB.obj
        (_, held) = self.held[hand]
        (fix, _) = self.graspB[hand]
        if obj == held:
            self.graspB[hand] = (fix, gB)
        else:
            assert None, 'Object does not match grasp in updateGraspB'
        self.reset()
    def updatePlaceB(self, pB):
        obj = pB.obj
        entry = self.objectBs.get(obj, None)
        if entry:
            self.objectBs[obj] = (entry[0], pB)
        elif obj == self.getHeld('left'):
            self.updateHeld('none', None, None, 'left', None)
            self.objectBs[obj] = (False, pB)
        elif obj == self.getHeld('right'):
            self.updateHeld('none', None, None, 'right', None)
            self.objectBs[obj] = (False, pB)
        else:
            assert None, 'Unknown obj in updatePlaceB'
        self.reset()
    def reset(self):
        self.shadowWorld = None
        self.shadowProb = None
    def updateHeld(self, obj, face, graspD, hand, delta = None):
        desc = self.getWorld().getGraspDesc(obj) \
               if obj != 'none' else []
        og = ObjGraspB(obj, desc, DeltaDist(face), None, graspD, delta = delta)
        self.held[hand] = (False, obj)
        self.graspB[hand] = (False, og)
        self.excludeObjs([obj])
        self.reset()
        return self
    def updateHeldBel(self, graspB, hand):
        self.graspB[hand] = (False, graspB)
        if graspB is None:
            self.held[hand] = (False, 'none')
        else:
            self.held[hand] = (False, graspB.obj)
        if graspB:
            self.excludeObjs([graspB.obj])
        self.reset()
        return self
    def updateConf(self, c):
        (fixedConf, oldConf) = self.conf
        self.conf = (fixedConf, c)
        # The shadowWorld conf
        if self.shadowWorld:
            self.shadowWorld.setRobotConf(c, fixed=fixedConf)
        return self
    def updatePermObjBel(self, objPlace):
        obj = objPlace.obj
        for hand in ('left', 'right'):
            if self.getHeld(hand) == obj:
                self.updateHeldBel(None, hand)
        self.objectBs[obj] = (True, objPlace) # make it permanent
        self.reset()
        return self
    def excludeObjs(self, objs):
        for obj in objs:
            if obj in self.objectBs:
                del self.objectBs[obj]
        self.reset()
        return self

    def updateFromConds(self, goalConds, permShadows=False):
        world = self.getWorld()
        initialObjects = self.objectsInPBS()
        goalPoseBels = getGoalPoseBels(goalConds, world.getFaceFrames)
        (held, graspB) = \
               getHeldAndGraspBel(goalConds, world.getGraspDesc)
        for h in ('left', 'right'):
            if held[h] is not None:
                self.held[h] = (True, held[h])
            if graspB[h] is not None:
                self.graspB[h] = (True, graspB[h])
        for gB in graspB.values():
            if gB is not None: self.excludeObjs([gB.obj])
        for h in ('left', 'right'):
            if held[h] in goalPoseBels:
                # print 'Held object in pose conditions, removing from hand'
                self.updateHeldBel(None, h)
        goalConf = getGoalConf(goalConds)
        if goalConf is not None:
            self.conf = (True, goalConf)
        base = getGoalBase(goalConds)
        if base is not None:
            self.base = (True, base)
        targetConf = getGoalTargetConf(goalConds)
        if targetConf is not None:
            self.targetConf = (True, targetConf)
        self.objectBs.update({o:(True, pB) \
                                  for (o, pB) in goalPoseBels.iteritems()})
        # The shadows of Pose(obj) in the cond are also permanent
        if permShadows:
            self.updateAvoidShadow(goalPoseBels.keys())
        self.reset()
        finalObjects = self.objectsInPBS()
        if initialObjects != finalObjects:
            tr('conservation',
               ('    initial', sorted(list(initialObjects))),
               ('    final', sorted(list(finalObjects))))
        return self

    # Modify world to eliminate collisions and keep support

    def ditherRobotOutOfCollision(self, p):
        count = 0
        confViols = self.confViolations(self.getConf(), p)
        while count < 100 and (confViols is None or confViols.obstacles or \
          confViols.heldObstacles[0] or confViols.heldObstacles[1]):
            count += 1
            if debug('dither'):
                self.draw(p, 'W')
                raw_input('go?')
            base = self.getConf().baseConf()
            # Should consider motions in both positive and negative directions
            # It won't wonder away too far... TLP
            newBase = tuple([b + (random.random() - 0.5) * 0.01 * count\
                              for b in base])
            newConf = self.getConf().setBaseConf(newBase)
            self.updateConf(newConf)
            confViols = self.confViolations(self.getConf(), p)
        if count == 100:
            raise Exception, 'Failed to move robot out of collision'

    def findSupportRegion(self, prob, shape, strict=False, fail=True):
        # This calls the procedure defined at the end of the file
        return findSupportRegion(shape, self.getShadowWorld(prob).regionShapes,
                                 strict=strict, fail=fail)

    def findSupportRegionForObj(self, prob, obj, strict=False, fail=True):
        placeB = self.getPlaceB(obj, default=False)
        if not placeB: return
        shape = placeB.shape(self.getWorld())
        return findSupportRegion(shape, self.getShadowWorld(prob).regionShapes,
                                 strict=strict, fail=fail)

    def ditherObjectToSupport(self, obj, p):
        ditherCount = 200
        def delta(x, c = 1): return x + (random.random() - 0.5)*0.01*c
        count = 0
        world = self.getWorld()
        pB = self.getPlaceB(obj)
        shape = pB.shape(world)
        supported = self.findSupportRegion(p, shape,
                                           strict=True, fail=False)
        while count < ditherCount and not supported:
            count += 1
            pB = self.getPlaceB(obj)
            poseList = list(pB.poseD.modeTuple())
            if debug('dither'):
                self.draw(p, 'W')
                print obj, poseList
                raw_input('go?')
            for i in (0,1):
                poseList[i] = delta(poseList[i], count)
            newPose = hu.Pose(*poseList)
            self.updatePlaceB(pB.modifyPoseD(newPose))
            self.reset()
            shape = pB.shape(world)
            supported = self.findSupportRegion(p, shape,
                                               strict=True, fail=False)
        if count == 100:
            raise Exception, 'Failed to move object to support'

    def internalCollisionCheck(self, dither=True, objChecks=True, factor=2.0):
        shProb = 0.1
        ws = self.getShadowWorld(shProb)   # minimal shadow
        # First check the robot for hard collisions.  Increase this to
        # give some boundary
        confViols = self.confViolations(self.getConf(), shProb)
        if dither and \
               confViols is None or confViols.obstacles or \
               confViols.heldObstacles[0] or confViols.heldObstacles[1]:
            tr('dither', 'Robot in collision.  Will try to fix.',
                     draw=[(self, 0.0, 'W')], snap=['W'])
            self.ditherRobotOutOfCollision(shProb)
            self.reset()
            confViols = self.confViolations(self.getConf(), shProb)

        # Now, see if the shadow of the object in the hand is colliding.
        # If so, reduce it.
        shProb = 0.98
        confViols = self.confViolations(self.getConf(), shProb)
        for h in (0, 1):
            colls = confViols.heldShadows[h]
            hand = ('left', 'right')[h]
            count = 0
            while colls:
                count += 1
                tr('dither',
                   'Shadow of obj in hand.  Will try to fix', hand, colls,
                    draw = [(self, shProb, 'W')], snap = ['W'])
                # Divide variance in half.  Very crude.  Should find the
                # max variance that does not result in a shadow colliion.
                if count > 50:
                    assert None, 'Could not reduce grasp shadow after 10 attempts'
                gB = self.getGraspBForObj(self.getHeld(hand), hand)
                var = gB.poseD.variance()
                newVar = tuple(v/factor for v in var)
                self.updateGraspB(hand, gB.modifyPoseD(var=newVar))
                self.reset()
                confViols = self.confViolations(self.getConf(), shProb)
                colls = confViols.heldShadows[h]
            
        # Now for shadow collisions;  reduce the shadow if necessary
        shadows = confViols.allShadows()
        count = 0
        while shadows:
            count += 1
            tr('dither',
                   'Robot collides with shadows.  Will try to fix', shadows,
                    draw = [(self, shProb, 'W')], snap = ['W'])
            # Divide variance in half.  Very crude.  Should find the
            # max variance that does not result in a shadow colliion.
            if count > 100:
                assert None, 'Could not reduce shadow after 50 attempts'
            for sh in shadows:
                obj = objectName(sh)
                pB = self.getPlaceB(obj)
                var = pB.poseD.variance()
                newVar = tuple(v/factor for v in var)
                self.updatePlaceB(pB.modifyPoseD(var=newVar))
            self.reset()
            confViols = self.confViolations(self.getConf(), shProb)
            shadows = confViols.allShadows()

        if not objChecks: return

        # Finally, look for object-object collisions
        objShapes = [o for o in ws.getObjectShapes() \
                     if 'shadow' not in o.name()]
        n = len(objShapes)
        fixed = [o.obj for o in self.fixObjBs()]
        for index in range(n):
            shape = objShapes[index]
            if debugMsg('collisionCheck'):
                shape.draw('W', 'black')
            for index2 in range(index+1, n):
                shape2 = objShapes[index2]
                # Ignore collisions between fixed objects
                if shape.collides(shape2) and \
                  not (shape.name() in fixed and \
                       shape2.name() in fixed):
                    for shapeXX in objShapes: shapeXX.draw('W', 'black')
                    shape2.draw('W', 'magenta')
                    raise Exception, 'Object-Object collision: '+ \
                                     shape.name()+' - '+shape2.name()

        # Check objects are supported
        for (fix, pB) in self.objectBs.values():
            if fix: continue
            shape = pB.shape(self.getWorld())
            supported = self.findSupportRegion(0.99, shape,
                                              strict=True, fail=False)
            tr('dither', pB.obj, 'supported=', supported)
            if not supported:
                tr('dither', 'Object not supported.  Will try to fix.',
                   draw=[(self, 0.0, 'W')], snap=['W'])
                self.ditherObjectToSupport(pB.obj, 0.99)
                self.reset()
        debugMsg('collisionCheck')
    
    # Creating shadow world
    
    # arg can also be a graspB since they share relevant methods
    def shadowPair(self, objB, faceFrame, prob):
        def shLE(w1, w2):
            return all([w1[i] <= w2[i] for i in (0,1,3)])
        obj = objB.obj
        # We set these to zero for canPickPlaceTest.
        cpp = sum(objB.poseD.var) == 0.0 and sum(objB.delta) == 0.0
        if cpp:
            objBMinDelta = objB.delta
        else:
            objBMinDelta = self.domainProbs.shadowDelta
        objBMinProb = 0.95
        # The irreducible shadow
        objBMinVar = self.domainProbs.objBMinVar(obj, objB.poseD.var if cpp else None)
        objBMin = objB.modifyPoseD(var=objBMinVar)
        objBMin.delta = objBMinDelta

        shWidth = shadowWidths(objB.poseD.var, objB.delta, prob)
        minShWidth = shadowWidths(objBMinVar, objBMinDelta, objBMinProb)
        if shLE(shWidth, minShWidth):
            # If the "min" shadow is wider than actual shadow, make them equal
            objB = objBMin
            prob = objBMinProb

        if debug('shadowWidths'):
            print 'obj', obj, 'movable', movable, 'objB == objBMin', objB == objBMin
            print 'shWidth', shWidth
            print 'minShWidth', minShWidth

        # Shadows relative to Identity pose
        shadow = self.objShadow(obj, shadowName(obj), prob, objB, faceFrame)
        shadowMin = self.objShadow(obj, obj, objBMinProb, objBMin, faceFrame) # use obj name

        if debug('getShadowWorld'):
            print 'objB', objB
            shadow.draw('W', 'gray')
            print 'objBMin', objBMin
            shadowMin.draw('W', 'orange')
            print 'max shadow widths', shWidth
            print 'min shadow widths', minShWidth
            # print obj, 'origin\n', sw.objectShapes[obj].origin()
            print obj, 'shadow\n', shadow.bbox()
            print obj, 'shadow origin\n', shadow.origin().matrix
            print obj, 'support pose\n', objB.poseD.mode().matrix
            print obj, 'origin pose\n', objB.objFrame().matrix
            raw_input('Shadows for %s (orange in inner, gray is outer)'%obj)

        return shadowMin, shadow
    
    def getShadowWorld(self, prob):
        # total count, current, cache hits, cache misses, new
        shadowWorldStats[0] += 1
        if self.shadowWorld and self.shadowProb == prob:
            shadowWorldStats[1] += 1
            return self.shadowWorld
        else:
            cache = self.beliefContext.genCaches['getShadowWorld']
            # key = (self.items(), prob)
            key = self.items() + (glob.ignoreShadowZ,) # !
            if key in cache:
                ans = cache.get(key, None)
                if ans != None:
                    shadowWorldStats[2] += 1 # cache hit
                    if prob in ans:
                        self.shadowWorld = ans[prob]
                        self.shadowProb = prob
                        return self.shadowWorld
                else:
                    print 'shadowWorld cache inconsistent'
            else:
                shadowWorldStats[3] += 1 # cache miss
                cache[key] = {}     # dict of prob->sw
        shadowWorldStats[4] += 1    # new world...
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
            if self.objectBs.get(obj, (False, None))[0]:   # can't collide with these objects
                sw.fixedObjects.add(shadowMin.name())
        # Add robot
        sw.robot = self.getRobot()
        robot = sw.robot
        robot.nominalConf = w.robot.nominalConf
        # robot.nominalConf = nominalConf
        # Add shadow for held object
        for hand in ('left', 'right'):
            heldObj = self.getHeld(hand)
            if heldObj != 'none':
                graspB = self.getGraspB(hand)
                assert graspB and graspB.obj == heldObj
                graspIndex = graspB.grasp.mode()
                graspDesc = graspB.graspDesc[graspIndex]
                # Transform from object origin to grasp surface
                # The graspDesc frame is relative to object origin
                # The graspB pose encodes finger tip relative to graspDesc frame
                faceFrame = graspDesc.frame.compose(graspB.poseD.mode())
                if graspIndex < 0:      # push grasp
                    support = graspB.support
                    supportFrame = w.getFaceFrames(heldObj)[support]
                    # Create shadow pair and attach both to robot
                    shadowMin, shadow = self.shadowPair(graspB,
                                                        supportFrame, prob)

                    shadow = shadow.applyTrans(supportFrame)
                    shadowMin = shadowMin.applyTrans(supportFrame)

                    if debug('getShadowWorldGrasp'):
                        shadow.draw('W', 'gray'); shadowMin.draw('W', 'red')
                    # graspShadow is expressed relative to wrist and attached to arm
                    # fingerFrame maps from wrist to center of fingers (second arg is 0.)
                    fingerFrame = robot.fingerSupportFrame(hand, 0.0)
                    heldFrame = fingerFrame.compose(faceFrame.inverse())
                    graspShadow = shadow.applyTrans(heldFrame)
                    graspShadowMin = shadowMin.applyTrans(heldFrame)
                    if debug('getShadowWorldGrasp'):
                        cart = self.getConf().cartConf()
                        wrist = cart[robot.armChainNames[hand]]
                        graspShadowMin.applyTrans(wrist).draw('W', 'red')
                        graspShadow.applyTrans(wrist).draw('W', 'gray')
                        raw_input('Grasped shadow')
                else:  # normal grasp
                    # Create shadow pair and attach both to robot
                    shadowMin, shadow = self.shadowPair(self.getGraspB(hand), faceFrame, prob)
                    # shadow is expressed in face frame, now we need
                    # to express it relative to wrist.
                    # fingerFrame maps from wrist to inner face of finger
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
                if debug('getShadowWorldGrasp') and not glob.inHeuristic:
                    print 'faceFrame\n', faceFrame.matrix
                    print 'shadow\n', shadow.bbox()
                    print 'fingerFrame\n', fingerFrame.matrix
                    print 'graspShadow\n', graspShadow.bbox()
                    graspShadow.draw('W', 'red')
                    print 'attached origin\n', sw.attached[hand].origin().matrix
                    raw_input('Attach?')
        sw.fixedHeld = {h:fix for (h, (fix,v)) in self.held.iteritems()}
        sw.fixedGrasp = {h:fix for (h, (fix,v)) in self.graspB.iteritems()}
        sw.setRobotConf(self.getConf())
        if debug('getShadowWorldGrasp') and not glob.inHeuristic:
            sw.draw('W')
        # cache[key] = sw
        cache[key][prob] = sw
        return sw

    # Shadow over POSE variation.  The variance is in absolute coordinates, so
    # it needs to be applied to the object already placed.  But, this returns
    # the resulting shadow at the object origin.  So, we need to transform it
    # there and back again.
    # TODO: Should only do finite number of poseVar/poseDelta values.
    def objShadow(self, obj, shName, prob, poseBel, faceFrame):
        shape = self.getObjectShapeAtOrigin(obj)
        color = shape.properties.get('color', None) or \
                (shape.parts() and [s.properties.get('color', None) for s in shape.parts()][0]) or \
                'gray'
        assert isinstance(shName, str)
        if 'shadow' in shName:
            color = 'gray'
        poseVar = poseBel.poseD.variance()
        poseDelta = poseBel.delta
        objShadowStats[0] += 1
        key = (shape, shName, prob, poseBel, faceFrame, glob.ignoreShadowZ)
        shadow = self.beliefContext.objectShadowCache.get(key, None)
        if shadow:
            objShadowStats[1] += 1
            return shadow
        # Origin * Support = Pose => Origin = Pose * Support^-1
        frame = faceFrame.inverse()     # pose is identity
        sh = shape.applyLoc(frame)      # the shape with the specified support
        trans = poseBel.poseD.mode()
        if trans:
            # Now, we need to rotate it as in poseBel
            rotAngle = trans.pose().theta
            sh = sh.applyTrans(hu.Pose(0., 0., 0., rotAngle))
            shadow = makeShadowOrigin(sh, prob, poseVar, poseDelta, name=shName, color=color)
            # Then, rotate it back
            shadow = shadow.applyTrans(hu.Pose(0., 0., 0., -rotAngle))
        else:
            shadow = makeShadowOrigin(sh, prob, poseVar, poseDelta, name=shName, color=color)
        self.beliefContext.objectShadowCache[key] = shadow
        if debug('getShadowWorld'):
            shadow.draw('W', 'red')
        return shadow

    # Miscellaneous
    def draw(self, p = 0.9, win = 'W', clear=True, drawRobot=True):
        if clear: wm.getWindow(win).clear()
        self.getShadowWorld(p).draw(win, drawRobot=drawRobot)

    def items(self):
        return (frozenset(self.held.items()),
                frozenset(self.graspB.items()),
                self.conf, self.base, self.targetConf,
                frozenset(self.avoidShadow),
                frozenset(self.objectBs.items()))
    def __eq__(self, other):
        return hasattr(other, 'items') and self.items() == other.items()
    def __ne__(self, other):
        return not self == other
    def __hash__(self):
        return hash(self.items())
    def __repr__(self):
        return 'BS:'+str(self.items())
    __str__ = __repr__

####################
# Shadow computation
####################

"""
Fear (= uncertainty) and loathing (= slack) in belief space:

Consider a constraint on the belief state.  The constraints are on
the values of mu and sigma.  We want to constrain them so that,
*for all the distributions inside the constraint on mu and sigma*,
the mu, sigma, p shadow is inside the region.  Assume we know how to
make a mu, sigma, p shadow.  That's under the assumption we know mu
and sigma.

But, we want to come up with a BV condition that will ensure the
object is inside the region wp > p, for all mu, sigma within a box
constraint.  The deltas are a box constraint on mu The sigma in the
fluent is really an upper bound on sigmas, but we don't have to think
about that.  So, the thing that has to fit in the region is the shadow
shape for sigma, p convolved with the detla box.  The critical thing
is that it has to work for any value of mu in the box
"""

def sigmaPoses(prob, poseD, poseDelta):
    interpStep = math.pi/16
    def interpAngle(lo, hi):
        if hi - lo <= interpStep:
            return [lo, hi]
        else:
            return interpAngle(lo, 0.5*(lo+hi))[:-1] + \
                   interpAngle(0.5*(lo+hi), hi)
    widths = shadowWidths(poseD.variance(), poseDelta, prob)
    n = len(widths)
    offsets = []
    (wx, wy, wz, wt) = widths
    if glob.ignoreShadowZ:
        wz = 0.
    angles = interpAngle(-wt, wt)
    if debug('getShadowWorld'):
        print 'shadowWidths', widths
        print 'angles', angles
        pdb.set_trace()
    for dx in (-wx, 0., wx):
        for dy in (-wy, 0., wy):
            if dx or dy:
                for dz in (-wz, wz):
                    for a in angles: offsets.append([dx, dy, dz, a])
    poses = []
    for offset in offsets:
        offPoseTuple = offset
        poses.append(hu.Pose(*offPoseTuple))
    return poses

shadowOpacity = 0.2
def makeShadow(shape, prob, bel, name=None, color='gray'):
    shParts = []
    poses = sigmaPoses(prob, bel.poseD, bel.delta)
    if debug('getShadowWorld'):
        print 'sigma poses for', shape.name()
    shColor = shape.properties.get('color', color)

    for part in shape.parts():
        if debug('getShadowWorld'):
            wm.getWindow('W').clear()
        shp = []
        for pose in poses:
            shp.append(part.applyTrans(pose))
            if debug('getShadowWorld'):
                shp[-1].draw('W', 'cyan')
        # Note xyPrim !!
        shParts.append(shapes.Shape(shp, shape.origin(),
                                    name=part.name(),
                                    color=shColor,
                                    opacity=shadowOpacity).xyPrim())
        shParts[-1].properties['opacity']=shadowOpacity
        if debug('getShadowWorld'):
            shParts[-1].draw('W', 'orange')
            raw_input('Next part?')
    if debug('getShadowWorld'):
        raw_input('multiple part shadow, Ok?')
    final = shapes.Shape(shParts, shape.origin(),
                         name=name or shape.name(),
                         color=shColor, opacity=shadowOpacity)
    if debug('getShadowWorld'):
        shape.draw('W', 'blue')
        final.draw('W', 'pink')
        raw_input('input shape (blue), final shadow (pink), Ok?')
    return final

def sigmaPosesOrigin(prob, poseVar, poseDelta):
    interpStep = math.pi/16
    def interpAngle(lo, hi):
        if hi - lo <= interpStep:
            return [lo, hi]
        else:
            return interpAngle(lo, 0.5*(lo+hi))[:-1] + \
                   interpAngle(0.5*(lo+hi), hi)
    widths = shadowWidths(poseVar, poseDelta, prob)
    n = len(widths)
    offsets = []
    (wx, wy, wz, wt) = widths
    if glob.ignoreShadowZ:
        wz = 0.
    angles = interpAngle(-wt, wt)
    if debug('getShadowWorld'):
        print 'shadowWidths', widths
        print 'angles', angles
    for a in angles: offsets.append([-wx, 0, -wz, a])
    for a in angles: offsets.append([wx, 0, wz, a])
    for a in angles: offsets.append([0, -wy, -wz, a])
    for a in angles: offsets.append([0, wy, wz, a])
    poses = []
    for offset in offsets:
        offPoseTuple = offset
        poses.append(hu.Pose(*offPoseTuple))
    return poses

def makeShadowOrigin(shape, prob, var, delta, name=None, color='gray'):
    shParts = []
    poses = sigmaPosesOrigin(prob, var, delta)
    if debug('getShadowWorld'):
        print 'sigma poses for', shape.name()
    shColor = shape.properties.get('color', color)

    for part in shape.parts():
        if debug('getShadowWorld'):
            wm.getWindow('W').clear()
        shp = []
        for pose in poses:
            shp.append(part.applyTrans(pose))
            if debug('getShadowWorld'):
                shp[-1].draw('W', 'cyan')
        # Note xyPrim !!
        shParts.append(shapes.Shape(shp, shape.origin(),
                                    name=part.name(),
                                    color=shColor,
                                    opacity=shadowOpacity).xyPrim())
        shParts[-1].properties['opacity']=shadowOpacity
        if debug('getShadowWorld'):
            shParts[-1].draw('W', 'orange')
            raw_input('Next part?')
    if debug('getShadowWorld'):
        raw_input('multiple part shadow, Ok?')
    final = shapes.Shape(shParts, shape.origin(),
                         name=name or shape.name(),
                         color=shColor, opacity=shadowOpacity)
    if debug('getShadowWorld'):
        shape.draw('W', 'blue')
        final.draw('W', 'pink')
        raw_input('input shape (blue), final shadow (pink), Ok?')
    return final

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

    fbs2 = getMatchingFluents(goalConds,
                B([Cond([Pose(['Obj', 'Face']), 'OPose', 'OVal']),
                                           'Mu', 'Var', 'Delta', 'P'], True))

    ans = dict([(b['Obj'], ObjPlaceB(b['Obj'],
                                     getFaceFrames(b['Obj']), # !! ??
                                     DeltaDist(b['Face']),
                                     hu.Pose(* b['Mu']),
                                     b['Var'], b['Delta'])) \
                 for (f, b) in fbs + fbs2 if \
                      (isGround(b.values()) and not ('*' in b.values()))])
    return ans

# Return None if there is no Conf requirement; otherwise return conf
def getGoalConf(goalConds):
    if not goalConds: return None
    fbs = [(f, b) for (f, b) \
           in getMatchingFluents(goalConds,
                                 Conf(['Mu', 'Delta'], True)) if f.isGround()]
    assert len(fbs) <= 1, 'Too many Conf fluents in conditions'
    conf = None
    if len(fbs) == 1:
        ((f, b),) = fbs
        if isGround(b.values()):
            conf = b['Mu']
    return conf

# Return None if there is no BaseConf requirement; otherwise return
# base pose tuple
def getGoalBase(goalConds):
    fbs = getMatchingFluents(goalConds,
                             BaseConf(['B', 'D'], True))
    assert len(fbs) <= 1, 'Too many BaseConf fluents in conditions'
    result = None
    for (f, b) in fbs:
        base = b['B']
        if not isVar(base):
            assert result is None, 'More than one Base fluent'
            result = tuple(base)
    return result

def getGoalTargetConf(goalConds):
    fbs_conf = [(f, b) for (f, b) \
                in getMatchingFluents(goalConds,
                                      Conf(['Mu', 'Delta'], True))]
    fbs_crnb = getMatchingFluents(goalConds,
                                  Bd([CanReachNB(['Start', 'End', 'Cond']),
                                      True, 'P'], True))
    if not (fbs_conf and fbs_crnb): return None
    # Goal has a conf specified and at least one CanReachNB
    conf = None
    for (fconf, bconf) in fbs_conf:
        for (fcrnb, bcrnb) in fbs_crnb:
            confVar = fconf.args[0]
            # Target conf is a var
            if isVar(confVar) and fconf.args[0] == fcrnb.args[0].args[0]:
                # Find the place we are trying to reach from here.
                conf = fcrnb.args[0].args[1]
    return conf

def getHeldAndGraspBel(goalConds, getGraspDesc):
    held = {'left':None, 'right':None}
    graspB = {'left':None, 'right':None}
    if not goalConds: return (held, graspB)
    # Figure out what we're holding
    hbs = getMatchingFluents(goalConds, Bd([Holding(['H']), 'Obj', 'P'], True))
    objects = {}
    for (f, b) in hbs:
        if isGround(b.values()):  
            hand = b['H']
            assert hand in ('left', 'right') and not hand in objects, \
                   'Multiple inconsistent Holding in conds'
            held[hand] = b['Obj']
            objects[hand] = b['Obj']

    # Grasp faces
    gfbs = getMatchingFluents(goalConds,
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
    gbs = getMatchingFluents(goalConds,
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
            if obj != 'none' and mu != '*':
                graspB[hand] = ObjGraspB(obj, getGraspDesc(obj),
                                         DeltaDist(face), None, hu.Pose(*mu), var, delta)

    return (held, graspB)

# If strict is True then ?? (it is passed into the inside test)
# If fail is True, then raise exception if this test fails
def findSupportRegion(shape, regions, strict=False, fail=True):
    tag = 'findSupportRegion'
    bbox = shape.bbox()
    bestRegShape = None
    bestVol = 0.
    if debug(tag): print 'find support region for', shape
    for regShape in regions.values():
        if debug(tag): print 'considering', regShape
        regBB = regShape.bbox()
        if inside(shape, regShape, strict=strict):
            # TODO: Have global support(region, shape) test?
            if debug(tag): print 'shape in region'
            if 0 <= bbox[0,2] - regBB[0,2] <= 0.02:
                if debug(tag): print 'bottom z is close enough'
                vol = bboxVolume(regBB)
                if vol > bestVol:
                    bestRegShape = regShape
                    bestVol = vol
                    if debug(tag): print 'bestRegShape', regShape
    if not bestRegShape:
        print 'Could not find supporting region for %s'%shape.name()
        shape.draw('W', 'magenta')
        for regShape in regions.values():
            regShape.draw('W', 'cyan')
        if fail:
            print 'Gonna fail!'
            raise Exception, 'Unsupported object'
    return bestRegShape
