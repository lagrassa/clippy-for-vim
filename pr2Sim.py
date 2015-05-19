import math
import numpy as np
import objects
from objects import WorldState
import windowManager3D as wm
import pr2Util
from pr2Util import supportFaceIndex, DomainProbs, bigAngleWarn
from dist import DDist, DeltaDist, MultivariateGaussianDistribution
MVG = MultivariateGaussianDistribution
import util
from planGlobals import debugMsg, debug, debugOn
from pr2Robot import gripperTip, gripperFaceFrame
from pr2Visible import visible, lookAtConf
from time import sleep
from pr2Ops import lookAtBProgress

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

animate = False
animateSleep = 0.2

maxOpenLoopDist = 1.0

pickSuccessDist = 0.1  # pretty big for now
class RealWorld(WorldState):
    def __init__(self, world, bs, probs):
        # probs is an instance of DomainProbs
        WorldState.__init__(self, world)
        self.bs = bs
        self.domainProbs = probs

    # dispatch on the operators...
    def executePrim(self, op, params = None):
        def endExec(obs):
            self.draw('World')
            print 'Executed', op.name, 'got obs', obs
            debugMsg('executePrim')
            return obs
        if op.name == 'Move' or op.name == 'MoveNB':
            return endExec(self.executeMove(op, params))
        elif op.name == 'LookAtHand':
            return endExec(self.executeLookAtHand(op, params))
        elif op.name == 'LookAt':
            return endExec(self.executeLookAt(op, params))
        elif op.name == 'Pick':
            return endExec(self.executePick(op, params))
        elif op.name == 'Place':
            return endExec(self.executePlace(op, params))
        else:
            raise Exception, 'Unknown operator: '+str(op)

    # Be sure there are no collisions.  If so, stop early.
    def doPath(self, path, interpolated=None):
        def getObjShapes():
            held = self.held.values()
            return [self.objectShapes[obj] \
                    for obj in self.objectShapes if not obj in held]
        objShapes = getObjShapes()
        obs = None
        path = interpolated or path
        distSoFar = 0
        prevXYT = self.robotConf.conf['pr2Base']
        for (i, conf) in enumerate(path):
            # !! Add noise to conf
            self.setRobotConf(conf)
            if animate:
                self.draw('World')
                sleep(animateSleep)
            else:
                self.robotPlace.draw('World', 'orchid')
            cart = conf.cartConf()
            leftPos = np.array(cart['pr2LeftArm'].point().matrix.T[0:3])
            rightPos = np.array(cart['pr2RightArm'].point().matrix.T[0:3])
            debugMsg('path', 
                     ('base', conf['pr2Base'], 'left', leftPos, 'right', rightPos))
            for obst in objShapes:
                if self.robotPlace.collides(obst):
                    obs ='crash'
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
                self.setRobotConf(conf)
                self.robotPlace.draw('World', 'orange')
                cart = conf.cartConf()
                leftPos = np.array(cart['pr2LeftArm'].point().matrix.T[0:3])
                rightPos = np.array(cart['pr2RightArm'].point().matrix.T[0:3])
                debugMsg('path',
                    ('base', conf['pr2Base'], 'left', leftPos,'right',rightPos))
                break
            newXYT = self.robotConf.conf['pr2Base']
            # Integrate the displacement
            distSoFar += math.sqrt(sum([(prevXYT[i]-newXYT[i])**2 for i in (0,1)]))
            # approx pi => 1 meter
            distSoFar += 0.33*abs(util.angleDiff(prevXYT[2],newXYT[2]))
            print 'distSoFar', distSoFar
            # Check whether we should look
            args = 11*[None]
            if distSoFar >= maxOpenLoopDist:
                obj = self.visibleObj(objShapes)
                lookConf = lookAtConf(self.robotConf, obj)
                obs = self.doLook(lookConf)
                if obs == []:
                    raw_input('No observation')
                args[1] = lookConf
                lookAtBProgress(self.bs, args, obs)
                distSoFar = 0           #  reset
            prevXYT = newXYT

        wm.getWindow('World').update()
        debugMsg('doPath', 'Admire the path')
        return obs

    def visibleObj(self, objShapes):
        def rem(l,x): return [y for y in l if y != x]
        prob = 0.95
        world = self.bs.pbs.getWorld()
        shWorld = self.bs.pbs.getShadowWorld(prob)
        rob = self.bs.pbs.getRobot().placement(self.robotConf, attached=shWorld.attached)[0]
        fixed = [s.name() for s in objShapes] + [rob.name()]
        immovable = [s for s in objShapes if s not in world.graspDesc]
        movable = [s for s in objShapes if s in world.graspDesc]
        for s in immovable + movable:
            if visible(shWorld, self.robotConf, s, rem(objShapes,s)+[rob], prob,
                       moveHead=True, fixed=fixed):
                return s

    def executeMove(self, op, params):
        vl = \
           ['CStart', 'CEnd', 'DEnd',
            'LObj', 'LFace', 'LGraspMu', 'LGraspVar', 'LGraspDelta',
            'RObj', 'RFace', 'RGraspMu', 'RGraspVar', 'RGraspDelta',
            'P1', 'P2', 'PCR']
        if params:
            path, interpolated  = params
            debugMsg('path', 'path len = ', len(path))
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
        self.robotPlace.draw('World', 'orchid')
        debugMsg('sim', 'LookAt configuration')
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
                print 'Object', targetObj, 'is not visible'
                return 'none'
            else:
                print 'Object', targetObj, 'is visible'
            return (targetObj, gdIndex, graspTuple)
        else:
            # TLP! Please fix.  Should check that the hand is actually
            # visible
            return 'none'

    def doLook(self, lookConf):
        self.setRobotConf(lookConf)

        self.robotPlace.draw('World', 'orchid')
        debugMsg('sim', 'LookAt configuration')
        # objType = self.world.getObjType(targetObj)
        obs = []
        for shape in self.getObjectShapes():
            curObj = shape.name()
            # if self.world.getObjType(curObj) != objType:
            #     continue
            obstacles = [s for s in self.getObjectShapes() if \
                         s.name() != curObj ]  + [self.robotPlace]

            deb = 'visible' in debugOn
            if (not deb) and debug('visibleEx'): debugOn.append('visible')
            vis, _ = visible(self, self.robotConf,
                             self.objectShapes[curObj],
                             obstacles, 0.75, moveHead=False,
                             fixed=[self.robotPlace.name()])
            if not deb and debug('visibleEx'): debugOn.remove('visible')
            if not vis:
                print 'Object', curObj, 'is not visible'
                continue
            else:
                print 'Object', curObj, 'is visible'
            #raw_input('Visible')

            truePose = self.getObjectPose(curObj)
            # Have to get the resting face.  And add noise.
            trueFace = supportFaceIndex(self.objectShapes[curObj])
            print 'observed Face', trueFace
            ff = self.objectShapes[curObj].faceFrames()[trueFace]
            obsMissProb = self.domainProbs.obsTypeErrProb
            miss = DDist({True: obsMissProb, False:1-obsMissProb}).draw()
            if miss:
                raw_input('Missed observation')
                continue
            else:
                obsVar = self.domainProbs.obsVar
                obsPose = util.Pose(*MVG(truePose.xyztTuple(), obsVar).draw())
                obsPlace = obsPose.compose(ff).pose().xyztTuple()
                objType = self.world.getObjType(curObj)
                obs.append((objType, trueFace, util.Pose(*obsPlace)))
        print 'Observation', obs
        return obs

    def executeLookAt(self, op, params):
        # targetObj = op.args[0]
        lookConf = op.args[1]
        # !! This should not move the base...
        assert lookConf.conf['pr2Base'] == self.robotConf.conf['pr2Base']

        return self.doLook(lookConf)

    def executePick(self, op, params):
        # Execute the pick prim, starting at c1, aiming for c2.
        # Have to characterize the likelihood of grasping O and
        # the variance of the grasp, then draw appropriately.
        vl = \
           ['Obj', 'Hand', 'OtherHand', 'PoseFace', 'Pose', 'PoseDelta',
            'RObj', 'RFace', 'RGraspMu', 'RGraspVar', 'RGraspDelta',
            'GraspFace', 'GraspMu', 'GraspVar', 'GraspDelta',
            'PreConf', 'ConfDelta', 'PickConf',
            'P1', 'P2', 'P3', 'P4', 'PR1', 'PR2', 'PR3']
        failProb = self.domainProbs.pickFailProb
        success = DDist({True : 1 - failProb, False : failProb}).draw()

        # Try to execute pick
        (hand, pickConf, approachConf) = \
                 (op.args[1], op.args[17], op.args[15])
        bigAngleWarn(approachConf, pickConf)
        self.setRobotConf(pickConf)
        self.robotPlace.draw('World', 'orchid')
        oDist = None
        o = None
        robot = self.robot
        cart = self.robotConf.cartConf()
        handPose = cart[robot.armChainNames[hand]].compose(gripperTip)
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
                print 'picked', self.held[hand], self.grasp[hand]
                self.setRobotConf(self.robotConf)
                self.robotPlace.draw('World', 'black')
                self.setRobotConf(approachConf)
                self.robotPlace.draw('World', 'orchid')
                print 'retracted'
            else:
                # Failed to pick.  Move the object a little.
                newObjPose = opose.pose().corruptGauss(0.0,
                                           self.domainProbs.placeStdev)
                self.setObjectPose(o, newObjPose)
        else:
            print 'tried to pick but missed', o, oDist, pickSuccessDist
        return None

    def executePlace(self, op, params):
        failProb = self.domainProbs.placeFailProb
        success = DDist({True : 1 - failProb, False : failProb}).draw()
        if success:
            # Execute the place prim, starting at c1, aiming for c2.
            # Every kind of horrible, putting these indices here..
            hand = op.args[1]
            placeConf = op.args[-8]
            approachConf = op.args[-10]
            bigAngleWarn(approachConf, placeConf)
            self.setRobotConf(placeConf)
            self.robotPlace.draw('World', 'orchid')            
            if not self.attached[hand]:
                raw_input('No object is attached')
                debugMsg('sim', 'No object is attached')
            else:
                debugMsg('sim', 'Object is attached')
            robot = self.robot
            detached = robot.detach(self, hand)
            self.setRobotConf(self.robotConf)
            obj = self.held[hand]
            #assert detached and obj == detached.name()
            if detached:
                cart = self.robotConf.cartConf()
                handPose = cart[robot.armChainNames[hand]].\
                  compose(gripperTip)
                objPose = handPose.compose(self.grasp[hand]).pose()
                if simulateError:
                    actualObjPose = objPose.corruptGauss(0.0,
                                                self.domainProbs.placeStdev)
                else:
                    actualObjPose = objPose
                self.setObjectPose(self.held[hand], actualObjPose)
                self.grasp[hand] = None
                self.held[hand] = None
                print 'placed', obj, actualObjPose
            self.setRobotConf(approachConf)
            self.robotPlace.draw('World', 'orchid')
            print 'retracted'
        return None

    def copy(self):
        return copy.copy(self)

    def draw(self, win):
        # print 'Robot', self.robotConf
        # print 'Objects', self.objectConfs
        # print 'Held', self.held
        # print 'Grasp', self.grasp
        wm.getWindow(win).clear()
        WorldState.draw(self, win)
        wm.getWindow(win).update()


def graspFaceIndexAndPose(conf, hand, shape, graspDescs):
    robot = conf.robot
    wristFrame = conf.cartConf()[robot.armChainNames[hand]]
    fingerFrame = wristFrame.compose(gripperFaceFrame)
    objFrame = shape.origin()
    for g in range(len(graspDescs)):
        graspDesc = graspDescs[g]
        faceFrame = graspDesc.frame
        centerFrame = faceFrame.compose(util.Pose(0,0,graspDesc.dz,0))
        graspFrame = objFrame.compose(centerFrame)
        candidatePose = graspFrame.inverse().compose(fingerFrame).pose(fail=False)
        if candidatePose:
            params = candidatePose.xyztTuple()
            if abs(params[-1]) < math.pi/8:
                print 'Grasp face', g, 'Grasp offset', params
                return (g, params)
    raw_input('Could not find graspFace')
    return (0, (0,0,0,0))
