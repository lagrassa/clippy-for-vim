import numpy as np
import objects
from objects import WorldState
from pr2Robot import gripperTip
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

obsMissProb = 0.05

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
                return endExec('none')

        elif op.name == 'LookAt':
            targetObj = op.args[0]
            lookConf = op.args[1]
            self.setRobotConf(lookConf)
            self.robotPlace.draw('World', 'orchid')
            debugMsg('sim', 'LookAt configuration')
            obstacles = [s for s in self.getObjectShapes() if s.name() != targetObj ]
            vis, _ = visible(self, self.robotConf, self.objectShapes[targetObj],
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
            miss = DDist({True:obsMissProb, False:1-obsMissProb}).draw()
            if miss:
                return endExec(None)
            else:
                obsVar = self.domainProbs.obsVar
                obsPlace = MVG(truePlace, obsVar).draw()

                # debugging
                oShape = self.world.getObjectShapeAtOrigin(targetObj)
                oShape.applyLoc(util.Pose(*obsPlace)).draw('World', 'black')
                print 'True place', truePlace
                print 'Obs place (in black)', obsPlace
                print 'Delta',\
                          [abs(x - y) for (x, y) in zip(truePlace, obsPlace)]
                print 'Stdev',[np.sqrt(v) for v in self.domainProbs.obsVarTuple]
                #raw_input('Obs okay?')
                return endExec((targetObj, trueFace, util.Pose(*obsPlace)))


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
            hand = op.args[1]
            pickConf = op.args[17]
            approachConf = op.args[15]
            self.setRobotConf(pickConf)
            self.robotPlace.draw('World', 'orchid')
            # if params:
            #     path = params
            #     print 'path length = ', len(path)
            #     if not path:
            #         raw_input('No path!!')
            #     obs = self.executePath(path)
            #     if obs == 'crash':
            #         return endExec(obs)
            # else:
            #     print op
            #     raw_input('No path given')
            # Find obj nearest to "hand" (should use a better test)
            oDist = None
            o = None
            robot = self.robot
            cart = robot.forwardKin(self.robotConf)
            handPose = cart[robot.armChainNames[hand]].compose(gripperTip)
            # !! Add noise to handPose
            for oname in self.objectConfs:
                pose = self.getObjectPose(oname)
                od = handPose.distance(pose)
                # print oname, pose, od
                if not oDist or od < oDist:
                    oDist = od
                    o = oname
                    opose = pose
            if oDist < pickSuccessDist:
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
                # obs = self.executePath(path[::-1])
                obs = None
                print 'retracted'
            else:
                print 'tried to pick but missed', o, oDist, pickSuccessDist
                return endExec(obs)
        elif op.name == 'Place':
            # Execute the place prim, starting at c1, aiming for c2.
            hand = op.args[1]
            placeConf = op.args[20]
            approachConf = op.args[18]
            self.setRobotConf(placeConf)
            self.robotPlace.draw('World', 'orchid')            
            if not self.attached[hand]:
                debugMsg('sim', 'No object is attached')
            else:
                debugMsg('sim', 'Object is attached')
            # if params:
            #     path = params
            #     print 'path length = ', len(path)
            #     if not path:
            #         raw_input('No path!!')
            #     obs = self.executePath(path)
            #     if obs == 'crash':
            #         return endExec(obs)
            # else:
            #     print op
            #     raw_input('No path given')
            robot = self.robot
            if not self.attached[hand]:
                raw_input('No object is attached')
            detached = robot.detach(self, hand)
            self.setRobotConf(self.robotConf)
            obj = self.held[hand]
            assert detached and obj == detached.name()
            cart = robot.forwardKin(self.robotConf)

            handPose = cart[robot.armChainNames[hand]].compose(gripperTip)
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
            # obs = self.executePath(path[::-1])
            obs = None
            print 'retracted'
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
        candidateFrame = graspFrame.inverse().compose(fingerFrame)
        if candidateFrame.pose(fail=False):
            print 'Grasp face', g, 'Grasp offset', candidateFrame.pose().xyztTuple()
            return (g, candidateFrame.pose().xyztTuple())
    raw_input('Could not find graspFace')
    return (0, (0,0,0,0))
