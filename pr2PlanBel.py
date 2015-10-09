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
#from pr2Robot import PR2, pr2Init, makePr2Chains
from traceFile import debugMsg, debug
import planGlobals as glob
from pr2Fluents import Holding, GraspFace, Grasp, Conf, Pose
from planUtil import ObjGraspB, ObjPlaceB
from pr2Util import shadowName, shadowWidths, objectName, supportFaceIndex, PoseD, inside, permanent, pushable, graspable
#import fbch
from fbch import getMatchingFluents
from belief import B, Bd
from traceFile import tr, trAlways
from transformations import rotation_matrix
from geom import bboxVolume

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
                 graspB=None, fixObjBs=None, moveObjBs=None, regions=[],
                 domainProbs=None, useRight=True, avoidShadow=[],
                 fixGrasp=None, fixHeld=None):
        self.beliefContext = beliefContext
        self.conf = conf or None
        self.held = held or \
          {'left': DeltaDist('none'), 'right': DeltaDist('none')}
        # Keep graspB only for mode of held - value is ObjGraspB instance
        self.graspB = graspB or {'left':None, 'right':None}
        self.fixObjBs = fixObjBs or {}            # {obj: objPlaceB}
        self.moveObjBs = moveObjBs or {}          # {obj: objPlaceB}
        self.fixGrasp = fixGrasp or {'left':False, 'right':False} # the graspB can be changed
        self.fixHeld = fixHeld or {'left':False, 'right':False} # whether the held can be changed
        self.regions = regions
        self.pbs = self
        self.useRight = glob.useRight
        self.avoidShadow = avoidShadow  # shadows to avoid
        self.domainProbs = domainProbs
        # cache
        self.shadowWorld = None                   # cached obstacles
        self.shadowProb = None

    def reset(self):
        self.shadowWorld = None
        self.shadowProb = None

    def ditherRobotOutOfCollision(self, p):
        count = 0
        rm = self.beliefContext.roadMap
        confViols = rm.confViolations(self.conf, self, p)
        while count < 100 and (confViols is None or confViols.obstacles or \
          confViols.heldObstacles[0] or confViols.heldObstacles[1]):
            count += 1
            if debug('dither'):
                self.draw(p, 'W')
                raw_input('go?')
            base = self.conf['pr2Base']
            # Should consider motions in both positive and negative directions
            # It won't wonder away too far... TLP
            newBase = tuple([b + (random.random() - 0.5) * 0.01 * count\
                              for b in base])
            newConf = self.conf.set('pr2Base', newBase)
            self.updateConf(newConf)
            confViols = rm.confViolations(self.conf, self, p)
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
        rm = self.beliefContext.roadMap
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
            self.resetPlaceB(pB.modifyPoseD(newPose))
            self.reset()
            shape = pB.shape(world)
            supported = self.findSupportRegion(p, shape,
                                               strict=True, fail=False)
        if count == 100:
            raise Exception, 'Failed to move object to support'

    def internalCollisionCheck(self, dither=True, objChecks=True, factor=2.0):
        ws = self.getShadowWorld(0.0)   # minimal shadow
        rm = self.beliefContext.roadMap
        # First check the robot for hard collisions.  Increase this to
        # give some boundary
        shProb = 0.1
        confViols = rm.confViolations(self.conf, self, shProb)
        if dither and \
               confViols is None or confViols.obstacles or \
               confViols.heldObstacles[0] or confViols.heldObstacles[1]:
            tr('dither', 'Robot in collision.  Will try to fix.',
                     draw=[(self, 0.0, 'W')], snap=['W'])
            self.ditherRobotOutOfCollision(shProb)
            self.reset()
            confViols = rm.confViolations(self.conf, self, shProb)

        # Now, see if the shadow of the object in the hand is colliding.
        # If so, reduce it.
        shProb = 0.98
        confViols = rm.confViolations(self.conf, self, shProb)
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
                gB = self.getGraspB(self.held[hand].mode(), hand)
                var = gB.poseD.variance()
                newVar = tuple(v/factor for v in var)
                self.resetGraspB(obj, hand, gB.modifyPoseD(var=newVar))
                self.reset()
                confViols = rm.confViolations(self.conf, self, shProb)
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
            if count > 50:
                assert None, 'Could not reduce shadow after 50 attempts'
            for sh in shadows:
                obj = objectName(sh)
                pB = self.getPlaceB(obj)
                var = pB.poseD.variance()
                newVar = tuple(v/factor for v in var)
                self.resetPlaceB(pB.modifyPoseD(var=newVar))
            self.reset()
            confViols = rm.confViolations(self.conf, self, shProb)
            shadows = confViols.allShadows()

        if not objChecks: return

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
                    for shapeXX in objShapes: shapeXX.draw('W', 'black')
                    shape2.draw('W', 'magenta')
                    raise Exception, 'Object-Object collision: '+ \
                                     shape.name()+' - '+shape2.name()

        # Check objects are supported
        for pB in self.moveObjBs.values():
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

    def resetPlaceB(self, pB):
        obj = pB.obj
        if self.fixObjBs.get(obj, None):
            self.fixObjBs[obj] = pB
        elif self.moveObjBs.get(obj, None):
            self.moveObjBs[obj] = pB
        elif obj == self.held['left'].mode():
            self.updateHeld('none', None, None, 'left', None)
            self.moveObjBs[obj] = pB
        elif obj == self.held['right'].mode():
            self.updateHeld('none', None, None, 'right', None)
            self.moveObjBs[obj] = pB
        else:
            assert None, 'Unknown obj in resetPlaceB'
        self.reset()

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
    def resetGraspB(self, obj, hand, gB):
        if obj == self.held[hand].mode():
            self.graspB[hand] = gB
        else:
            assert None, 'Object does not match grasp in resetGraspB'
    def defaultGraspB(self, obj):
        desc = self.getWorld().getGraspDesc(obj)
        return ObjGraspB(obj, desc, UniformDist(range(len(desc))), None, Ident, 4*(100.0,))

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
                   self.regions, self.domainProbs, self.useRight, self.avoidShadow[:],
                   self.fixGrasp.copy(), self.fixHeld.copy())

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

    def conditioned(self, goalConds, cond = None, permShadows = False):
        newBS = self.copy()
        newBS = newBS.updateFromGoalPoses(goalConds)
        if cond is not None:
            newBS = newBS.updateFromGoalPoses(cond, permShadows=permShadows)
        return newBS
    
    # Makes objects mentioned in the goal permanent,
    # Side-effects self
    def updateFromGoalPoses(self, goalConds,
                            updateHeld=True, updateConf=True, permShadows=False):
        world = self.getWorld()
        initialObjects = self.objectsInPBS()
        goalPoseBels = getGoalPoseBels(goalConds, world.getFaceFrames)
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
            for h in ('left', 'right'):
                if self.held[h] in goalPoseBels:
                    # print 'Held object in pose conditions, removing from hand'
                    self.updateHeldBel(None, h)
        if updateConf:
            self.conf = getConf(goalConds, self.conf)
        world = self.getWorld()
        self.fixObjBs.update(goalPoseBels)
        self.moveObjBs = dict([(o, p) for (o, p) \
                               in self.getPlacedObjBs().iteritems() \
                               if o not in self.fixObjBs])
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

    def updateHeld(self, obj, face, graspD, hand, delta = None):
        desc = self.getWorld().getGraspDesc(obj) \
          if obj != 'none' else []
        og = ObjGraspB(obj, desc, DeltaDist(face), None, graspD, delta = delta)
        self.held[hand] = DeltaDist(obj) # !! Is this rigt??
        self.graspB[hand] = og
        self.excludeObjs([obj])
        self.reset()
        return self

    def updateHeldBel(self, graspB, hand):
        self.graspB[hand] = graspB
        if graspB is None:
            self.held[hand] = DeltaDist('none')
            self.fixHeld[hand] = False
            self.fixGrasp[hand] = False
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

    # Side effects the belief about this object in this world
    # def updateObjB(self, objPlace):
    #     obj = objPlace.obj
    #     if obj in self.moveObjBs:
    #         self.moveObjBs[obj] = objPlace
    #     elif obj in self.fixObjBs:
    #         self.fixObjBs[obj] = objPlace
    #     else:
    #         # Must be in the hand...
    #         pass
    #     self.reset()

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
            key = self.items()
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
                if graspIndex < 0:      # push grasp
                    support = self.graspB[hand].support
                    supportFrame = w.getFaceFrames(heldObj)[support]
                    # Create shadow pair and attach both to robot
                    shadowMin, shadow = self.shadowPair(self.graspB[hand],
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
                        cart = self.conf.cartConf()
                        wrist = cart[robot.armChainNames[hand]]
                        graspShadowMin.applyTrans(wrist).draw('W', 'red')
                        graspShadow.applyTrans(wrist).draw('W', 'gray')
                        raw_input('Grasped shadow')
                else:  # normal grasp
                    # Create shadow pair and attach both to robot
                    shadowMin, shadow = self.shadowPair(self.graspB[hand], faceFrame, prob)
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
        sw.fixedHeld = self.fixHeld
        sw.fixedGrasp = self.fixGrasp
        sw.setRobotConf(self.conf)
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
        key = (shape, shName, prob, poseBel, faceFrame)
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

    def draw(self, p = 0.9, win = 'W', clear=True, drawRobot=True):
        if clear: wm.getWindow(win).clear()
        self.getShadowWorld(p).draw(win, drawRobot=drawRobot)

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
    (wx, wy, _, wt) = widths
    angles = interpAngle(-wt, wt)
    if debug('getShadowWorld'):
        print 'shadowWidths', widths
        print 'angles', angles
        pdb.set_trace()
    for a in angles: offsets.append([-wx, 0, 0, a])
    for a in angles: offsets.append([wx, 0, 0, a])
    for a in angles: offsets.append([0, -wy, 0, a])
    for a in angles: offsets.append([0, wy, 0, a])
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

    ans = dict([(b['Obj'], ObjPlaceB(b['Obj'],
                                     getFaceFrames(b['Obj']), # !! ??
                                     DeltaDist(b['Face']),
                                     hu.Pose(* b['Mu']),
                                     b['Var'], b['Delta'])) \
                 for (f, b) in fbs if \
                      (isGround(b.values()) and not ('*' in b.values()))])
    return ans

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
            assert hand in ('left', 'right') and not hand in objects, \
                   'Multiple inconsistent Holding in conds'
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
