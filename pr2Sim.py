import pdb
import math
import numpy as np
import objects
import random
from objects import WorldState
import windowManager3D as wm
import pr2Util
from pr2Util import supportFaceIndex, DomainProbs, bigAngleWarn, bboxGridCoords, permanent
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

animateSleep = 0.02

simOdoErrorRate = 0.0                   # was 0.02

pickSuccessDist = 0.1  # pretty big for now

class RealWorld(WorldState):
    def __init__(self, world, bs, probs, robot = None):
        # probs is an instance of DomainProbs
        WorldState.__init__(self, world, robot)
        self.bs = bs
        self.domainProbs = probs
        # wm.getWindow('World').startCapture()

    # dispatch on the operators...
    def executePrim(self, op, params = None):
        def endExec(obs):
            self.draw('World')
            tr('sim', 'Executed %s got obs= %s'%(op.name, obs),
               snap=['World'])
            wm.getWindow('World').pause()
            wm.getWindow('Belief').pause()
            # wm.getWindow('World').capturing = False
            return obs
        # wm.getWindow('World').capturing = True
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
        odoError = self.domainProbs.odoError
        prevXYT = self.robotConf.baseConf()
        prevConf = self.robotConf
        leftChainName = self.robotConf.robot.armChainNames['left']
        rightChainName = self.robotConf.robot.armChainNames['right']

        if max([abs(a-b) for (a,b) in zip(self.robotConf.basePose().xyztTuple(),
                                          path[0].basePose().xyztTuple())]) > 0.01:
            print 'current base pose', self.robotConf.basePose()
            print 'path[0] base pose', path[0].basePose()
            if debug('sim'):
                raw_input('Inconsistency in path and simulation')
        
        for (pathIndex, conf) in enumerate(path):
            originalConf = conf
            newXYT = conf.baseConf()
            prevBasePose = path[max(0, pathIndex-1)].basePose()
            newBasePose = path[pathIndex].basePose()
            # Compute the nominal displacement along the original path
            disp = prevBasePose.inverse().compose(newBasePose).pose()
            # Variance based on magnitude of original displacement
            dispXY = (disp.x**2 + disp.y**2)**0.5
            dispAngle = disp.theta
            odoVar = ((dispXY *  odoError[0])**2,
                      (dispXY *  odoError[1])**2,
                      0.0,
                      (dispAngle * odoError[3])**2)
            # Draw the noise with zero mean and this variance
            baseOff = hu.Pose(*MVG((0.,0.,0.,0.), np.diag(odoVar),
                                   pose4 = True).draw())
            # Apply the noise to the nominal displacement
            dispNoisy = baseOff.compose(disp)
            # Apply the noisy displacement to the "actual" robot location
            bc = self.robotConf.basePose().compose(dispNoisy).pose().xyztTuple()
            # This is the new conf to move to
            conf = conf.setBaseConf(tuple([bc[i] for i in (0,1,3)]))
            if debug('sim'):
                print 'Initial base conf', newXYT
                print 'draw', baseOff.xyztTuple()
                print '+++', dispNoisy.pose().xyztTuple()
                print '--> modified base conf', conf.baseConf()
            if action:
                action(prevConf, conf) # do optional action
                prevConf = conf
            else:
                self.setRobotConf(conf)
            pathTraveled.append(conf)
            if debug('animate'):
                self.draw('World');
                # This is the original commanded conf, draw to see accumulated error
                self.bs.pbs.draw(0.95, 'Belief', drawRobot=False)
                self.robotPlace.draw('Belief', 'gold')
                sleep(animateSleep)
            # else:
            #     self.robotPlace.draw('World', 'pink')
            #     self.robotPlace.draw('Belief', 'pink')
            wm.getWindow('World').pause()
            wm.getWindow('Belief').pause()
            cart = conf.cartConf()
            leftPos = np.array(cart[leftChainName].point().matrix.T[0:3]).tolist()[0][:-1]
            rightPos = np.array(cart[rightChainName].point().matrix.T[0:3]).tolist()[0][:-1]
            tr('sim',
               'base', conf.baseConf(), 'left', leftPos, 'right', rightPos)
            if debug('sim'):
                print 'left\n', cart[leftChainName].matrix
                print 'right\n', cart[rightChainName].matrix
            if ignoreCrash:
                pass
            else:
                for obst in getObjShapes():
                    if self.robotPlace.collides(obst):
                        obs ='crash'
                        tr('sim', 'Crash! with '+obst.name())
                        if originalConf.placement().collides(obst):
                            print 'original conf collides - bad path?'
                        else:
                            print 'original conf does not collide - simulated base error'
                        raw_input('Crash! with '+obst.name())
                        if crashIsError:
                            raise Exception, 'Crash'
                        break
                    else:
                        # Add noise here and make relative motions
                        pass
            if obs == 'crash':
                # Back up to previous conf
                print 'This is supposed to back up to previous step and stop... does it work?'
                pdb.set_trace()
                c = path[pathIndex-1]
                path = path[:pathIndex]         # cut off the rest of the path
                self.setRobotConf(c)  # LPK: was conf
                self.robotPlace.draw('World', 'orange')
                self.robotPlace.draw('Belief', 'orange')
                cart = c.cartConf()
                leftPos = np.array(cart[leftChainName].point().matrix.T[0:3])
                rightPos = np.array(cart[rightChainName].point().matrix.T[0:3])
                tr('sim',
                   ('base', c.baseConf(), 'left', leftPos, 'right', rightPos))
                break
            # Integrate the displacement
            distSoFar += math.sqrt(sum([(prevXYT[i]-newXYT[i])**2 for i in (0,1)]))
            # approx pi => 1 meter
            angleSoFar += abs(hu.angleDiff(prevXYT[2],newXYT[2]))
            # Check whether we should look
            args = 14*[None]
            if distSoFar + 0.33*angleSoFar >= glob.maxOpenLoopDist:
                print 'Exceeded max distance - exiting'
                return self.robotConf, (distSoFar, angleSoFar)
            prevXYT = newXYT
        if debug('backwards') and backSteps:
            print 'Backward steps:'
            for prev, next in backSteps:
                print prev, '->', next
            raw_input('Backwards')
        wm.getWindow('World').update()
        wm.getWindow('Belief').update()
        tr('sim', 'Admire the path', snap=['World'])

        # What should the observation be?
        # actual + noise?
        # commanded + noise?
        # commanded?  -- This is the hack we use on actual robot

        print 'sim robot base'
        print '    commanded:', path[-1].baseConf()
        print '       actual:', self.robotConf.baseConf()

        return path[-1], (distSoFar, angleSoFar)

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
            vis, occl = visible(shWorld, self.robotConf, s, rem(objShapes,s)+[rob], prob,
                                moveHead=True, fixed=fixed)
            if vis and len(occl) == 0:
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
                    in zip(self.robotConf.baseConf(), targetConf.baseConf())]) > 1.0e-6:
                print '****** MoveNB base pose does not match actual pose. '

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
        robotName = self.robotConf.robot.name
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
            vis, occl = visible(self, self.robotConf, shapeInHand,
                             obstacles, 0.75, moveHead=False, fixed=[self.robotPlace.name()])
            if not vis:
                if debug('sim'): print 'visible returned', vis, occl
                tr('sim', 'Object %s is not visible'%targetObj)
                return 'none'
            elif len(occl) > 0:
                if debug('sim'): print 'visible returned', vis, occl
                # This condition is implemented in canView.  It might
                # be very difficult to move hand out of the way of big
                # permanent objects.
                if occl == [robotName] and permanent(targetObj):
                    tr('sim', 'Permanent object %s is visible in spite of robot'%targetObj)
                else:
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
            scan = pc.simulatedScan(lookConf, glob.laserScanParams,
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
                vis, occl = visible(self, self.robotConf,
                                    self.objectShapes[curObj],
                                    obstacles, 0.75, moveHead=False,
                                    fixed=[self.robotPlace])
                if not deb and debug('visibleEx'): glob.debugOn.remove('visible')
                if not vis or len(occl) > 0:
                    tr('sim', 'Object %s is not visible'%curObj)
                    continue
                else:
                    tr('sim', 'Object %s is visible'%curObj)
                truePose = self.getObjectPose(curObj).pose()
                # Have to get the resting face.  And add noise.
                trueFace = supportFaceIndex(self.objectShapes[curObj])
                tr('sim', 'Observed face=%s, pose=%s'%(trueFace, truePose.xyztTuple()))
                ff = self.objectShapes[curObj].faceFrames()[trueFace]
                failProb = self.domainProbs.obsTypeErrProb
                if debug('simulateFaiure'):
                    success = DDist({True : 1 - failProb, False : failProb}).draw()
                    if not success:
                        print '*** Simulated look failure ***'
                else:
                    success = True
                if not success:
                    tr('sim', 'Missed observation')
                    continue
                else:
                    obsVar = self.domainProbs.obsVar
                    obsPose = hu.Pose(*MVG(truePose.xyztTuple(), obsVar,
                                           True).draw())
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
        if debug('simulateFaiure'):
            success = DDist({True : 1 - failProb, False : failProb}).draw()
            if not success:
                print '*** Simulated pick failure ***'
        else:
            success = True

        # Try to execute pick
        (hand, pickConf, approachConf) = \
                 (op.args[1], op.args[11], op.args[9])
        bigAngleWarn(approachConf, pickConf)
        self.setRobotConf(pickConf)
        # self.robotPlace.draw('World', 'orchid')
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
                # self.robotPlace.draw('World', 'orchid')
                # print 'retracted'
                obs = 'success'
            else:
                # Failed to pick.  Move the object a little.
                newObjPose = opose.pose().corruptGauss(0.0,
                                           self.domainProbs.placeStdev)
                self.setObjectPose(o, newObjPose)
                obs = 'failure'
        else:
            tr('sim', ('Tried to pick but missed', o, oDist, pickSuccessDist))
            obs = 'failure'
        return obs

    def executePlace(self, op, params):
        failProb = self.domainProbs.placeFailProb
        if debug('simulateFaiure'):
            success = DDist({True : 1 - failProb, False : failProb}).draw()
            if not success:
                print '*** Simulated place failure ***'
        else:
            success = True
        if success:
            # Execute the place prim, starting at c1, aiming for c2.
            # Every kind of horrible, putting these indices here..
            hand = op.args[1]
            placeConf = op.args[-6]
            approachConf = op.args[-8]
            bigAngleWarn(approachConf, placeConf)
            self.setRobotConf(placeConf)
            # self.robotPlace.draw('World', 'orchid')            
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
            # self.robotPlace.draw('World', 'orchid')
            # print 'retracted'
        return None

    def pushObjectSim(self, obj, c1, c2, hand, deltaPose):
        print 'In push:', obj, 'before push at', self.getObjectPose(obj).pose()
        place = c2.placement()
        shape = self.objectShapes[obj]
        if not place.collides(shape):   # obj at current pose
            print 'No contact with', obj
            self.setRobotConf(c2)           # move robot and objectShapes update
            return
        # There is contact, step the object along
        deltaPose = objectDisplacement(shape, c1, c2, hand, deltaPose)
        self.setObjectPose(obj, deltaPose.compose(self.getObjectPose(obj)))
        shape = self.objectShapes[obj]
        self.setRobotConf(c2)     # move robot and objectShapes update
        if self.robotPlace.collides(shape):
            print 'Push left object in collision'
            pdb.set_trace()
        print 'Touching', obj, 'in push, moved to', self.getObjectPose(obj).pose()

    def pushObjectRigid(self, obj, c1, c2, hand, deltaPose):
        place = c2.placement()
        shape = self.objectShapes[obj]
        if not place.collides(shape):   # obj at current pose
            print 'No contact with', obj
            self.setRobotConf(c2)           # move robot and objectShapes update
            return
        # There is contact, step the object along
        for steps in range(20):
            newPose = deltaPose.compose(self.getObjectPose(obj))
            self.setObjectPose(obj, newPose)
            shape = self.objectShapes[obj]
            self.setRobotConf(c2)     # move robot and objectShapes update
            if not self.robotPlace.collides(shape):
                break
        if self.robotPlace.collides(shape):
            print 'Push left object in collision'
            pdb.set_trace()
        print 'Touching', obj, 'in push, moved to', self.getObjectPose(obj).pose()

    def executePush(self, op, params, noBase = True):
        def moveObjSim(prevConf, conf):
            self.pushObjectSim(obj, prevConf, conf, hand, deltaPose)
        def moveObjRigid(prevConf, conf):
            # w1 = prevConf.cartConf()[chain]
            # w2 = conf.cartConf()[chain]
            # delta2 = w2.compose(w1.inverse()).pose(0.1) # from w1 to w2
            # deltaPose2 = hu.Pose(delta2.x+deltaPose.x, delta2.y+deltaPose.y, 0.0, 0.0)
            # print 'deltaPose2', deltaPose2
            self.pushObjectRigid(obj, prevConf, conf, hand, deltaPose)
        failProb = self.domainProbs.pushFailProb
        if debug('simulateFaiure'):
            success = DDist({True : 1 - failProb, False : failProb}).draw()
            if not success:
                print '*** Simulated push failure ***'
        else:
            success = True
        if success:
            # Execute the push prim
            if params:
                path, revPath, _  = params
                tr('sim', 'path len = %d'%(len(path)))
                if not path:
                    raw_input('No path!!')
                if len(path) < 4:
                    raw_input('Short path!!')
                obj = op.args[0]
                hand = op.args[1]
                robot = path[0].robot
                chain = robot.armChainNames[hand]
                w1 = path[0].cartConf()[chain]
                w2 = path[-1].cartConf()[chain]
                delta = w2.compose(w1.inverse()).pose(0.1) # from w1 to w2
                mag = (delta.x**2 + delta.y**2 + delta.z**2)**0.5
                deltaPose = hu.Pose(0.005*(delta.x/mag), 0.005*(delta.y/mag), 0.005*(delta.z/mag), 0.0)
                moveFn = moveObjSim if debug('pushSim') else moveObjRigid
                obs = self.doPath(path, path, action=moveFn)
                print 'Forward push path obs', obs
                obs = self.doPath(revPath, revPath)
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
        # nshape = shape.applyTrans(deltaPose)
        # delta = deltaPose
        # while gripperShape.collides(nshape):
        #     delta = delta.compose(delta).pose(0.1)
        #     nshape = shape.applyTrans(delta)
        supportPoints = shapeSupportPoints(shape)
        initial = np.zeros(3)
        final, finalVal = bruteForceMin(sumDisplacements, initial)
        displacement = math.sqrt(final[0]**2 + final[1]**2)
        if displacement > 0.05:
            print 'Displacement for the object while pushing', displacement
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
    for x in [-0.05, -0.02, 0.02, 0.05]:
        for y in [-0.05, -0.02, 0.02, 0.05]:
            for th in [-0.02, -0.01, 0.0, 0.01, 0.02]:
                X = np.array([x, y, th])
                newX, val = fun(X)
                if val < minVal:
                    minVal = val
                    minX = newX
    print 'final', minX, '->', minVal
    if minVal == 10.0:
        pdb.set_trace()
    return minX, minVal

print 'Loaded pr2Sim.py'

"""
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
"""
