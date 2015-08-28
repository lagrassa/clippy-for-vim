import pdb
import math
import numpy as np
import objects
import random
from objects import WorldState
import windowManager3D as wm
import pr2Util
from pr2Util import supportFaceIndex, DomainProbs, bigAngleWarn, bboxGridCoords
from dist import DDist, DeltaDist, MultivariateGaussianDistribution
MVG = MultivariateGaussianDistribution
import hu
import copy
from traceFile import debug, debugMsg
import planGlobals as glob
from pr2Robot import gripperFaceFrame
from pr2Visible import visible, lookAtConf
from time import sleep
from pr2Ops import lookAtBProgress
from pr2RoadMap import validEdgeTest
from traceFile import tr, snap
import locate
reload(locate)
import pr2RRT as rrt

# debug tables
import pointClouds as pc
import tables
reload(tables)

######################################################################
#
# Simulator
#
######################################################################

def argN (v, vl, args):
    return args[vl.position(v)]

crashIsError = False

simulateError = False

animateSleep = 0.01

maxOpenLoopDist = 2.0

simOdoErrorRate = 0.0                   # was 0.02

pickSuccessDist = 0.1  # pretty big for now

laserScanParams = (0.3, 0.2, 0.1, 3., 50)

class RealWorld(WorldState):
    def __init__(self, world, bs, probs, robot = None):
        # probs is an instance of DomainProbs
        WorldState.__init__(self, world, robot)
        self.bs = bs
        self.domainProbs = probs
        wm.getWindow('World').startCapture()

    # dispatch on the operators...
    def executePrim(self, op, params = None):
        def endExec(obs):
            self.draw('World')
            tr('sim', 'Executed %s got obs= %s'%(op.name, obs),
               snap=['World'])
            wm.getWindow('World').pause()
            wm.getWindow('World').capturing = False
            return obs
        wm.getWindow('World').capturing = True
        if op.name == 'Move':
            return endExec(self.executeMove(op, params))
        if op.name == 'MoveNB':
            return endExec(self.executeMove(op, params, noBase = True))
        elif op.name == 'LookAtHand':
            return endExec(self.executeLookAtHand(op, params))
        elif op.name == 'LookAt':
            return endExec(self.executeLookAt(op, params))
        elif op.name == 'Pick':
            return endExec(self.executePick(op, params))
        elif op.name == 'Place':
            return endExec(self.executePlace(op, params))
        elif op.name == 'Push':
            return endExec(self.executePush(op, params))
        else:
            raise Exception, 'Unknown operator: '+str(op)

    # Be sure there are no collisions.  If so, stop early.
    def doPath(self, path, interpolated=None, action=None, ignoreCrash=False):
        def getObjShapes():
            held = self.held.values()
            return [self.objectShapes[obj] \
                    for obj in self.objectShapes if not obj in held]
        obs = None
        path = interpolated or path
        distSoFar = 0.0
        angleSoFar = 0.0
        backSteps = []
        pathTraveled = []
        prevXYT = self.robotConf.conf['pr2Base']
        for (i, conf) in enumerate(path):
            # !! Add noise to conf
            if action:
                action(path, i) # do optional action
            else:
                self.setRobotConf(conf)
            pathTraveled.append(conf)
            if debug('animate'):
                self.draw('World')
                sleep(animateSleep)
            else:
                self.robotPlace.draw('World', 'pink')
            wm.getWindow('World').pause()
            cart = conf.cartConf()
            leftPos = np.array(cart['pr2LeftArm'].point().matrix.T[0:3]).tolist()[0][:-1]
            rightPos = np.array(cart['pr2RightArm'].point().matrix.T[0:3]).tolist()[0][:-1]
            tr('sim',
               'base', conf['pr2Base'], 'left', leftPos, 'right', rightPos)
            if debug('sim'):
                print 'left\n', cart['pr2LeftArm'].matrix
                print 'right\n', cart['pr2RightArm'].matrix
            if ignoreCrash:
                pass
            else:
                for obst in getObjShapes():
                    if self.robotPlace.collides(obst):
                        obs ='crash'
                        tr('sim', 'Crash! with '+obst.name())
                        pdb.set_trace()
                        raw_input('Crash! with '+obst.name())
                        if crashIsError:
                            raise Exception, 'Crash'
                        break
                    else:
                        # Add noise here and make relative motions
                        pass
            if obs == 'crash':
                # Back up to previous conf
                c = path[i-1]
                self.setRobotConf(c)  # LPK: was conf
                self.robotPlace.draw('World', 'orange')
                cart = conf.cartConf()
                leftPos = np.array(cart['pr2LeftArm'].point().matrix.T[0:3])
                rightPos = np.array(cart['pr2RightArm'].point().matrix.T[0:3])
                tr('sim',
                   ('base', conf['pr2Base'], 'left', leftPos, 'right', rightPos))
                break
            newXYT = self.robotConf.conf['pr2Base']
            if debug('backwards') and not validEdgeTest(prevXYT, newXYT):
                backSteps.append((prevXYT, newXYT))
            # Integrate the displacement
            distSoFar += math.sqrt(sum([(prevXYT[i]-newXYT[i])**2 for i in (0,1)]))
            # approx pi => 1 meter
            angleSoFar += abs(hu.angleDiff(prevXYT[2],newXYT[2]))
            # Check whether we should look
            args = 14*[None]
            if distSoFar + 0.33*angleSoFar >= maxOpenLoopDist:
                distSoFar = 0.0           #  reset
                angleSoFar = 0.0
                # raw_input('Exceeded max distance')
                print 'Exceeded max distance - exiting'
                return self.robotConf, (distSoFar, angleSoFar)
                obj = self.visibleObj(objShapes)
                if obj:
                    lookConf = lookAtConf(self.robotConf, obj)
                    if lookConf:
                        obs = self.doLook(lookConf)
                        if obs:
                            args[1] = lookConf
                            lookAtBProgress(self.bs, args, obs)
                        else:
                            tr('sim', 'No observation')
                    else:
                        tr('sim', 'No lookConf for %s'%obj.name())
                else:
                    tr('sim', 'No visible object')
            noisyXYT = [c + 2 * (random.random() - 0.5) * c * simOdoErrorRate \
                                 for c in newXYT]
            #prevXYT = newXYT
            prevXYT = noisyXYT
        if debug('backwards') and backSteps:
            print 'Backward steps:'
            for prev, next in backSteps:
                print prev, '->', next
            raw_input('Backwards')
        wm.getWindow('World').update()
        tr('sim', 'Admire the path', snap=['World'])
        return self.robotConf, (distSoFar, angleSoFar)

    def visibleObj(self, objShapes):
        def rem(l,x): return [y for y in l if y != x]
        prob = 0.95
        world = self.world
        shWorld = self.bs.pbs.getShadowWorld(prob)
        rob = world.robot.placement(self.robotConf, attached=shWorld.attached)[0]
        fixed = [s.name() for s in objShapes] + [rob.name()]
        immovable = [s for s in objShapes if not world.getGraspdesc(s)]
        movable = [s for s in objShapes if world.getGraspdesc(s)]
        for s in immovable + movable:
            if visible(shWorld, self.robotConf, s, rem(objShapes,s)+[rob], prob,
                       moveHead=True, fixed=fixed)[0]:
                return s

    def executeMove(self, op, params, noBase = False):
        vl = \
           ['CStart', 'CEnd', 'DEnd',
            'LObj', 'LFace', 'LGraspMu', 'LGraspVar', 'LGraspDelta',
            'RObj', 'RFace', 'RGraspMu', 'RGraspVar', 'RGraspDelta',
            'P1', 'P2', 'PCR']
        if noBase:
            startConf = op.args[0]
            targetConf = op.args[1]
            if max([abs(a-b) for (a,b) \
                    in zip(self.robotConf.conf['pr2Base'], targetConf.conf['pr2Base'])]) > 1.0e-6:
                print '****** The base moved in MoveNB, probably to an earlier collision fix'

        if params:
            path, interpolated, _  = params
            tr('sim', 'path len = %d'%(len(path)))
            if not path:
                raw_input('No path!!')
            obs = self.doPath(path, interpolated)
        else:
            print op
            raw_input('No path given')
            obs = None
        return obs

    def executeLookAtHand(self, op, params):
        targetObj = op.args[0]
        hand = op.args[1]
        lookConf = op.args[2]
        nominalGD = op.args[3]
        nominalGPoseTuple = op.args[4]
        self.setRobotConf(lookConf)
        tr('sim', 'LookAtHand configuration', draw=[(self.robotPlace, 'World', 'orchid')])
        _, attachedParts = self.robotConf.placementAux(self.attached,
                                                       getShapes=[])
        shapeInHand = attachedParts[hand]
        if shapeInHand:
            gdIndex, graspTuple = graspFaceIndexAndPose(self.robotConf,
                                                        hand,
                                                        shapeInHand,
                                     self.world.getGraspDesc(targetObj))
            obstacles = [s for s in self.getObjectShapes() \
                         if s.name() != targetObj ] + [self.robotPlace]
            vis, _ = visible(self, self.robotConf, shapeInHand,
                             obstacles, 0.75, moveHead=False, fixed=[self.robotPlace.name()])
            if not vis:
                tr('sim', 'Object %s is not visible'%targetObj)
                return 'none'
            else:
                tr('sim', 'Object %s is visible'%targetObj)
            return (targetObj, gdIndex, graspTuple)
        else:
            # TLP! Please fix.  Should check that the hand is actually
            # visible
            return 'none'

    def doLook(self, lookConf):
        self.setRobotConf(lookConf)
        tr('sim', 'LookAt configuration', draw=[(self.robotPlace, 'World', 'orchid')])
        obs = []

        if debug('useLocate'):
            scan = pc.simulatedScan(lookConf, laserScanParams,
                                    self.getNonShadowShapes()+ [self.robotPlace])
            scan.draw('W', 'cyan')

        for shape in self.getObjectShapes():
            curObj = shape.name()
            objType = self.world.getObjType(curObj)
            if debug('useLocate'):
                placeB = self.bs.pbs.getPlaceB(curObj)
                (score, trans, obsShape) =\
                        locate.getObjectDetections(lookConf, placeB, self.bs.pbs, scan)
                if score != None:
                    obsPose = trans.pose()
                    tr('sim', 'Object %s is visible at %s'%(curObj, obsPose.xyztTuple()))
                    print 'placeB.poseD.mode()', placeB.poseD.mode()
                    print 'truePose', self.getObjectPose(curObj)
                    print ' obsPose', obsPose
                    obsShape.draw('World', 'cyan')
                    raw_input('Go?')
                    obs.append((objType, supportFaceIndex(obsShape), obsPose))
                else:
                    tr('sim', 'Object %s is not visible'%curObj)
            else:
                obstacles = [s for s in self.getObjectShapes() if \
                             s.name() != curObj ]  + [self.robotPlace]
                deb = 'visible' in glob.debugOn
                if (not deb) and debug('visibleEx'): glob.debugOn.append('visible')
                vis, _ = visible(self, self.robotConf,
                                 self.objectShapes[curObj],
                                 obstacles, 0.75, moveHead=False,
                                 fixed=[self.robotPlace.name()])
                if not deb and debug('visibleEx'): glob.debugOn.remove('visible')
                if not vis:
                    tr('sim', 'Object %s is not visible'%curObj)
                    continue
                else:
                    tr('sim', 'Object %s is visible'%curObj)
                truePose = self.getObjectPose(curObj).pose()
                # Have to get the resting face.  And add noise.
                trueFace = supportFaceIndex(self.objectShapes[curObj])
                tr('sim', 'Observed face=%s, pose=%s'%(trueFace, truePose.xyztTuple()))
                ff = self.objectShapes[curObj].faceFrames()[trueFace]
                obsMissProb = self.domainProbs.obsTypeErrProb
                miss = DDist({True: obsMissProb, False:1-obsMissProb}).draw()
                if miss:
                    tr('sim', 'Missed observation')
                    continue
                else:
                    obsVar = self.domainProbs.obsVar
                    obsPose = hu.Pose(*MVG(truePose.xyztTuple(), obsVar).draw())
                    obsPlace = obsPose.compose(ff).pose().xyztTuple()
                    obs.append((objType, trueFace, hu.Pose(*obsPlace)))
        tr('sim', 'Observation', obs)
        if not obs:
            debugMsg('sim', 'Null observation')
        return obs

    def executeLookAt(self, op, params):
        # targetObj = op.args[0]
        lookConf = op.args[1]

        return self.doLook(lookConf)

    def executePick(self, op, params):
        # Execute the pick prim, starting at c1, aiming for c2.
        # Have to characterize the likelihood of grasping O and
        # the variance of the grasp, then draw appropriately.
        vl = \
           ['Obj', 'Hand', 'PoseFace', 'Pose', 'PoseDelta',
            'GraspFace', 'GraspMu', 'GraspVar', 'GraspDelta',
            'PreConf', 'ConfDelta', 'PickConf', 'RealGraspVar', 'PoseVar',
            'P1', 'PR1', 'PR2', 'PR3']
        failProb = self.domainProbs.pickFailProb
        success = DDist({True : 1 - failProb, False : failProb}).draw()

        # Try to execute pick
        (hand, pickConf, approachConf) = \
                 (op.args[1], op.args[11], op.args[9])
        bigAngleWarn(approachConf, pickConf)
        self.setRobotConf(pickConf)
        self.robotPlace.draw('World', 'orchid')
        oDist = None
        o = None
        robot = self.robot
        cart = self.robotConf.cartConf()
        tool = robot.toolOffsetX[hand]
        handPose = cart[robot.armChainNames[hand]].compose(robot.toolOffsetX[hand])
        # Find closest object
        for oname in self.objectConfs:
            pose = self.getObjectPose(oname)
            od = handPose.distance(pose)
            # print oname, pose, od
            if not oDist or od < oDist:
                (oDist, o, opose) = (od, oname, pose)
        if oDist < pickSuccessDist:
            if success:
                self.held[hand] = o
                grasp = handPose.inverse().compose(opose)
                if simulateError:
                    noisyGrasp = grasp.pose().corruptGauss(0.0,
                                              self.domainProbs.pickStdev)
                    self.grasp[hand] = noisyGrasp
                else:
                    self.grasp[hand] = grasp
                # !! Add noise to grasp
                robot.attach(self.objectShapes[o], self, hand)
                self.delObjectState(o)
                tr('sim', ('picked', self.held[hand], self.grasp[hand]))
                self.setRobotConf(self.robotConf)
                self.robotPlace.draw('World', 'black')
                self.setRobotConf(approachConf)
                self.robotPlace.draw('World', 'orchid')
                #print 'retracted'
            else:
                # Failed to pick.  Move the object a little.
                newObjPose = opose.pose().corruptGauss(0.0,
                                           self.domainProbs.placeStdev)
                self.setObjectPose(o, newObjPose)
        else:
            tr('sim', ('Tried to pick but missed', o, oDist, pickSuccessDist))
        return None

    def executePlace(self, op, params):
        failProb = self.domainProbs.placeFailProb
        success = DDist({True : 1 - failProb, False : failProb}).draw()
        if success:
            # Execute the place prim, starting at c1, aiming for c2.
            # Every kind of horrible, putting these indices here..
            hand = op.args[1]
            placeConf = op.args[-6]
            approachConf = op.args[-8]
            bigAngleWarn(approachConf, placeConf)
            self.setRobotConf(placeConf)
            self.robotPlace.draw('World', 'orchid')            
            if not self.attached[hand]:
                raw_input('No object is attached')
                tr('sim', 'No object is attached')
            else:
                tr('sim', 'Object is attached')
            robot = self.robot
            detached = robot.detach(self, hand)
            self.setRobotConf(self.robotConf)
            obj = self.held[hand]
            # assert detached and obj == detached.name()
            if detached:
                cart = self.robotConf.cartConf()
                handPose = cart[robot.armChainNames[hand]].\
                  compose(robot.toolOffsetX[hand])
                objPose = handPose.compose(self.grasp[hand]).pose()
                if simulateError:
                    actualObjPose = objPose.corruptGauss(0.0,
                                                self.domainProbs.placeStdev)
                else:
                    actualObjPose = objPose
                self.setObjectPose(self.held[hand], actualObjPose)
                self.grasp[hand] = None
                self.held[hand] = None
                # print 'placed', obj, actualObjPose
            self.setRobotConf(approachConf)
            self.robotPlace.draw('World', 'orchid')
            # print 'retracted'
        return None

    def pushObject(self, obj, c1, c2, hand, deltaPose):
        place = c2.placement()
        shape = self.objectShapes[obj]
        if not place.collides(shape):   # obj at current pose
            return
        # There is contact, step the object along
        deltaPose = objectDisplacement(shape, c1, c2, hand, deltaPose)
        self.setObjectPose(obj, deltaPose.compose(self.getObjectPose(obj)))
        self.setRobotConf(c2)           # move robot and objectShapes update
        shape = self.objectShapes[obj]
        if self.robotPlace.collides(shape):
            print 'Push left object in collision'
            pbd.set_trace()
        print 'Touching', obj, 'in push, moved it to', self.getObjectPose(obj).pose()

    def executePush(self, op, params, noBase = True):
        # TODO: compute the cartConfs once.
        def moveObjRigid(path, i):
            # There's a lot of noise in the interpolation, so use the
            # whole path to get displacement.  Could do it once.
            w1 = path[0].cartConf()[robot.armChainNames[hand]]
            w2 = path[-1].cartConf()[robot.armChainNames[hand]]
            delta = w2.compose(w1.inverse()).pose(0.1) # from w1 to w2
            mag = (delta.x**2 + delta.y**2 + delta.z**2)**0.5
            deltaPose = hu.Pose(0.005*(delta.x/mag), 0.005*(delta.y/mag), 0.0, 0.0)
            if i > 0:
                place = path[i].placement()
                while place.collides(self.objectShapes[obj]):
                    self.setObjectPose(obj, deltaPose.compose(self.getObjectPose(obj)))
                    print i, 'Touching', obj, 'in push, moved it to', self.getObjectPose(obj).pose()

        def moveObjSim(path, i):
            if i > 0:
                self.pushObject(obj, path[i-1], path[i], hand, deltaPose)

        failProb = self.domainProbs.pushFailProb
        success = DDist({True : 1 - failProb, False : failProb}).draw()
        if success:
            # Execute the push prim
            if params:
                path, interpolated, _  = params
                tr('sim', 'path len = %d'%(len(path)))
                if not path:
                    raw_input('No path!!')
                obj = op.args[0]
                hand = op.args[1]
                robot = path[0].robot
                w1 = path[0].cartConf()[robot.armChainNames[hand]]
                w2 = path[-1].cartConf()[robot.armChainNames[hand]]
                delta = w2.compose(w1.inverse()).pose(0.1) # from w1 to w2
                mag = (delta.x**2 + delta.y**2 + delta.z**2)**0.5
                deltaPose = hu.Pose(0.005*(delta.x/mag), 0.005*(delta.y/mag), 0.005*(delta.z/mag), 0.0)
                moveFn = moveObjSim if debug('pushSim') else moveObjRigid
                obs = self.doPath(path, interpolated, action=moveFn)
                print 'Forward push path obs', obs
                obs = self.doPath(path[::-1], interpolated[::-1])
                print 'Reverse push path obs', obs
            else:
                print op
                raw_input('No path given')
                obs = None
            return obs
        else:
            print 'Random push path failure'

    def copy(self):
        return copy.copy(self)

    def draw(self, win):
        wm.getWindow(win).clear()
        WorldState.draw(self, win)
        wm.getWindow(win).update()

def graspFaceIndexAndPose(conf, hand, shape, graspDescs):
    robot = conf.robot
    wristFrame = conf.cartConf()[robot.armChainNames[hand]]
    fingerFrame = wristFrame.compose(gripperFaceFrame[hand])
    objFrame = shape.origin()
    for g in range(len(graspDescs)):
        graspDesc = graspDescs[g]
        faceFrame = graspDesc.frame
        centerFrame = faceFrame.compose(hu.Pose(0,0,graspDesc.dz,0))
        graspFrame = objFrame.compose(centerFrame)
        candidatePose = graspFrame.inverse().compose(fingerFrame).pose(fail=False)
        if candidatePose:
            params = candidatePose.xyztTuple()
            if abs(params[-1]) < math.pi/8:
                print 'Grasp face', g, 'Grasp offset', params
                return (g, params)
    raw_input('Could not find graspFace')
    return (0, (0,0,0,0))

from scipy.optimize import fmin, fmin_powell

def objectDisplacement(shape, c1, c2, hand, deltaPose):
    def sumDisplacements(delta):
        penalty = 0.
        pose = hu.Pose(delta[0], delta[1], 0.0, delta[2])
        nshape = shape.applyTrans(pose).toPrims()[0]
        if gripperShape.collides(nshape):
            # penalty = 100. * penetrationDist(nshape, gripperVerts)
            penalty = 10.
        disp = np.dot(pose.matrix, supportPoints)
        dist_2 = (disp - supportPoints)**2
        sumDisp = np.sum(np.sqrt(np.sum(dist_2, axis=0)))
        # print delta, '=> sumDisp', sumDisp, 'penalty', penalty
        return sumDisp + penalty
        
    parts = dict([(part.name(), part) for part in c2.placement().parts()])
    gripperName = c2.robot.gripperChainNames[hand]
    gripperShape = parts[gripperName]
    gripperVerts = allVerts(gripperShape)

    if gripperShape.collides(shape):
        nshape = shape.applyTrans(deltaPose)
        delta = deltaPose
        while gripperShape.collides(nshape):
            delta = delta.compose(delta).pose(0.1)
            nshape = shape.applyTrans(delta)
        supportPoints = shapeSupportPoints(shape)
        #  initial = np.array([delta.x, delta.y, delta.theta])
        initial = np.zeros(3)
        final, finalVal = bruteForceMin(sumDisplacements, initial)
        return hu.Pose(final[0], final[1], 0.0, final[2])
    else:
        return hu.Pose(0.0, 0.0, 0.0, 0.0)
    
def allVerts(shape):
    verts = shape.toPrims()[0].vertices()
    for p in shape.toPrims()[1:]:
        verts = np.hstack([verts, p.vertices()])
    return verts

tiny = 1.0e-6
def penetrationDist(prim, verts):
    planes = prim.planes()
    maxPen = 0.0
    for i in xrange(verts.shape[1]):
        dists = np.dot(planes, verts[:,i].reshape(4,1))
        if np.all(dists <= tiny):
            maxPen = max(maxPen, -np.min(dists))
    return maxPen

def shapeSupportPoints(shape):
    shape0 = shape.toPrims()[0]
    # Raise z a bit, to make sure that it is inside
    z = shape0.bbox()[0,2] + 0.01
    points = bboxGridCoords(shape0.bbox(), z=z, res = 0.02)
    planes = shape0.planes()
    inside = []
    for p in points:
        if np.all(np.dot(planes, p.reshape(4,1)) <= tiny):
            inside.append(p)
    return np.vstack(inside).T

def bruteForceMin(f, init):

    def fun(x):
        return fmin(f, x, full_output=True, disp=False)[:2]

    (minX, minVal) = fun(init)
    print 'initial', minX, '->', minVal
    for x in [-0.02, 0.02]:
        for y in [-0.02, 0.02]:
            for th in [-0.02, -0.01, 0.0, 0.01, 0.02]:
                X = np.array([x, y, th])
                newX, val = fun(X)
                if val < minVal:
                    minVal = val
                    minX = newX
    print 'final', minX, '->', minVal
    return minX, minVal
