import math
import numpy as np
import objects
from objects import WorldState
from pr2Robot2 import gripperTip
import windowManager3D as wm
import pr2Util
from pr2Util import supportFaceIndex, DomainProbs
from dist import DDist, DeltaDist, MultivariateGaussianDistribution
MVG = MultivariateGaussianDistribution
import util
from planGlobals import debugMsg, debug
from pr2Robot import gripperTip, gripperFaceFrame
from pr2Visible import visible
from time import sleep


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

pickSuccessDist = 0.1  # pretty big for now
class RealWorld(WorldState):
    def __init__(self, world, probs):
        # probs is an instance of DomainProbs
        WorldState.__init__(self, world)
        self.domainProbs = probs

    def getObjShapes(self):
        held = self.held.values()
        return [self.objectShapes[obj] \
                for obj in self.objectShapes if not obj in held]

    # Be sure there are no collisions.  If so, stop early.
    def executePath(self, path):
        objShapes = self.getObjShapes()
        obs = None
        for (i, conf) in enumerate(path):
            # !! Add noise to conf
            self.setRobotConf(conf)
            if animate:
                self.draw('World'); sleep(0.2)
            else:
                self.robotPlace.draw('World', 'orchid')
            cart = conf.robot.forwardKin(conf)
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
                cart = conf.robot.forwardKin(conf)
                leftPos = np.array(cart['pr2LeftArm'].point().matrix.T[0:3])
                rightPos = np.array(cart['pr2RightArm'].point().matrix.T[0:3])
                debugMsg('path',
                    ('base', conf['pr2Base'], 'left', leftPos,'right',rightPos))
                break
        wm.getWindow('World').update()
        debugMsg('executePath', 'Admire the path')
        return obs

    def executePrim(self, op, params = None):
        def endExec(obs):
            self.draw('World')
            print 'Executed', op.name, 'got obs', obs
            debugMsg('executePrim')
            return obs
        # Side effect
        obs = None
        if op.name == 'Move':
            vl = \
               ['CStart', 'CEnd', 'DEnd',
                'LObj', 'LFace', 'LGraspMu', 'LGraspVar', 'LGraspDelta',
                'RObj', 'RFace', 'RGraspMu', 'RGraspVar', 'RGraspDelta',
                'P1', 'P2', 'PCR']
            if params:
                path = params
                debugMsg('path', 'path len = ', len(path))
                if not path:
                    raw_input('No path!!')
                obs = self.executePath(path)
            else:
                print op
                raw_input('No path given')
                
        elif op.name == 'LookAtHand':
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
            objInHand = shapeInHand.name() if shapeInHand else 'none'
            if shapeInHand:
                gdIndex, graspTuple = graspFaceIndexAndPose(self.robotConf,
                                                            hand,
                                                            shapeInHand,
                                         self.world.getGraspDesc(targetObj))
                obstacles = [s for s in self.getObjectShapes() \
                             if s.name() != targetObj ] + [self.robotPlace]
                vis, _ = visible(self, self.robotConf, shapeInHand,
                                 obstacles, 0.75, moveHead=False)
                if not vis:
                    print 'Object', targetObj, 'is not visible'
                    return endExec(None)
                else:
                    print 'Object', targetObj, 'is visible'
                return endExec((targetObj, gdIndex, graspTuple))
            else:
                # TLP! Please fix.  Should check that the hand is actually
                # visible
                return endExec(None)

        elif op.name == 'LookAt':
            targetObj = op.args[0]
            lookConf = op.args[1]
            self.setRobotConf(lookConf)
            self.robotPlace.draw('World', 'orchid')
            debugMsg('sim', 'LookAt configuration')
            obstacles = [s for s in self.getObjectShapes() if \
                         s.name() != targetObj ]
            
            if not targetObj in self.objectShapes:
                vis = False
            else:
                vis, _ = visible(self, self.robotConf,
                                 self.objectShapes[targetObj],
                             obstacles, 0.75)
            if not vis:
                print 'Object', targetObj, 'is not visible'
                return endExec(None)
            else:
                print 'Object', targetObj, 'is visible'
            truePose = self.getObjectPose(targetObj)
            # Have to get the resting face.  And add noise.
            ff = self.objectShapes[targetObj].faceFrames()
            trueFace = supportFaceIndex(self.objectShapes[targetObj])
            print 'observed Face', trueFace
            truePlace = truePose.compose(ff[trueFace]).pose().xyztTuple()
            obsMissProb = self.domainProbs.obsTypeErrProb
            miss = DDist({True: obsMissProb, False:1-obsMissProb}).draw()
            if miss:
                raw_input('Missed observation')
                return endExec(None)
            else:
                obsVar = self.domainProbs.obsVar
                obsPlace = MVG(truePlace, obsVar).draw()
                objType = self.getObjType(targetObj)
                print 'Obj type', targetObj, objType
                raw_input('okay?')

                # debugging
                oShape = self.world.getObjectShapeAtOrigin(targetObj)
                oShape.applyLoc(util.Pose(*obsPlace)).draw('World', 'black')
                print 'True place', truePlace
                print 'Obs place', obsPlace
                print 'Delta',\
                          [abs(x - y) for (x, y) in zip(truePlace, obsPlace)]
                print 'Stdev',[np.sqrt(v) for v in self.domainProbs.obsVarTuple]
                #raw_input('Obs okay?')
                return endExec((objType, trueFace, util.Pose(*obsPlace)))


        elif op.name == 'Pick':
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
            self.setRobotConf(pickConf)
            self.robotPlace.draw('World', 'orchid')
            oDist = None
            o = None
            robot = self.robot
            cart = robot.forwardKin(self.robotConf)
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
            return endExec(None)

        elif op.name == 'Place':
            failProb = self.domainProbs.placeFailProb
            success = DDist({True : 1 - failProb, False : failProb}).draw()
            if success:
            # Execute the place prim, starting at c1, aiming for c2.
                hand = op.args[1]
                placeConf = op.args[20]
                approachConf = op.args[18]
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
                    cart = robot.forwardKin(self.robotConf)
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
            obs = None
        else:
            raise Exception, 'Unknown operator: '+str(op)
        return endExec(obs)

    def copy(self):
        return copy.copy(self)

    def draw(self, win):
        print 'Robot', self.robotConf
        print 'Objects', self.objectConfs
        print 'Held', self.held
        print 'Grasp', self.grasp
        wm.getWindow(win).clear()
        WorldState.draw(self, win)
        wm.getWindow(win).update()


def graspFaceIndexAndPose(conf, hand, shape, graspDescs):
    robot = conf.robot
    wristFrame = robot.forwardKin(conf)[robot.armChainNames[hand]]
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
