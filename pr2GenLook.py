import hu
import numpy as np
import math
from geom import bboxCenter
from transformations import rotation_matrix
from traceFile import tr, debug, debugMsg

def potentialLookConfGen(pbs, prob, shape, maxDist):
    def testPoseInv(basePoseInv):
        bb = shape.applyTrans(basePoseInv).bbox()
        return bb[0][0] > 0 and bb[1][0] > 0

    centerPoint = hu.Point(np.resize(np.hstack([bboxCenter(shape.bbox()), [1]]), (4,1)))
    tested = set([])
    rm = pbs.getRoadMap()
    for node in rm.nodes():             # !!
        nodeBase = tuple(node.conf['pr2Base'])
        if nodeBase in tested:
            continue
        else:
            tested.add(nodeBase)
        x,y,th = nodeBase
        basePose = hu.Pose(x,y,0,th)
        dist = centerPoint.distanceXY(basePose.point())
        if dist > maxDist:
            continue
        inv = basePose.inverse()
        if not testPoseInv(inv):
            # Rotate the base to face the center of the object
            center = inv.applyToPoint(centerPoint)
            angle = math.atan2(center.matrix[0,0], center.matrix[1,0])
            rotBasePose = basePose.compose(hu.Pose(0,0,0,-angle))
            par = rotBasePose.pose().xyztTuple()
            rotConf = node.conf.set('pr2Base', (par[0], par[1], par[3]))
            if debug('potentialLookConfs'):
                print 'basePose', node.conf['pr2Base']
                print 'center', center
                print 'rotBasePose', rotConf['pr2Base']
            if testPoseInv(rotBasePose.inverse()):
                if rm.confViolations(rotConf, pbs, prob):
                    yield rotConf
        else:
            if debug('potentialLookConfs'):
                node.conf.draw('W')
                print 'node.conf', node.conf['pr2Base']
                raw_input('potential look conf')
            if rm.confViolations(node.conf, pbs, prob):
                yield node.conf
    return

ang = -math.pi/2
rotL = hu.Transform(rotation_matrix(-math.pi/4, (1,0,0)))
def trL(p): return p.compose(rotL)
rotR = hu.Transform(rotation_matrix(math.pi/4, (1,0,0)))
def trR(p): return p.compose(rotR)
lookPoses = {'left': [trL(x) for x in [hu.Pose(0.4, 0.35, 1.0, ang),
                                       hu.Pose(0.4, 0.25, 1.0, ang),
                                       hu.Pose(0.5, 0.08, 1.0, ang),
                                       hu.Pose(0.5, 0.18, 1.0, ang)]],
             'right': [trR(x) for x in [hu.Pose(0.4, -0.35, 0.9, -ang),
                                        hu.Pose(0.4, -0.25, 0.9, -ang),
                                        hu.Pose(0.5, -0.08, 1.0, -ang),
                                        hu.Pose(0.5, -0.18, 1.0, -ang)]]}
def potentialLookHandConfGen(pbs, prob, hand):
    shWorld = pbs.getShadowWorld(prob)
    robot = pbs.getConf().robot
    curCartConf = pbs.getConf().cartConf()
    chain = robot.armChainNames[hand]
    baseFrame = curCartConf['pr2Base']
    for pose in lookPoses[hand]:
        if debug('potentialLookHandConfs'):
            print 'potentialLookHandConfs trying:\n', pose
        target = baseFrame.compose(pose)
        cartConf = curCartConf.set(chain, target)
        conf = robot.inverseKin(cartConf, conf=pbs.getConf())
        if all(v for v in conf.conf.values()):
            if debug('potentialLookHandConfs'):
                conf.draw('W', 'blue')
                print 'lookPose\n', pose.matrix
                print 'target\n', target.matrix
                print 'conf', conf.conf
                print 'cart\n', cartConf[chain].matrix
                raw_input('potentialLookHandConfs')
            yield conf
    return
