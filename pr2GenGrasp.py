import hu
import planGlobals as glob
from planGlobals import torsoZ
from pr2Util import Memoizer, objectGraspFrame
from planUtil import Violations
from pr2GenUtils import sortedHyps, baseDist, inflatedBS, collisionMargin
from pr2Robot import CartConf, gripperPlace
from traceFile import tr, debug, debugMsg

approachConfCacheStats = [0,0]
graspConfGenCache = {}
graspConfGenCacheStats = [0,0]
graspConfs = set([])
graspConfStats = [0,0]

graspConfClear = 0.0

# Generate grasp
# Requirements: grasp is collision free
# Preferences:  grasp conf close to current conf

def potentialGraspConfGen(pbs, placeB, graspB, conf, hand, base, prob,
                          nMax=100, findApproach=True):
    tag = 'potentialGraspConfs'
    grasp = graspB.grasp.mode()

    pbs = pbs.copy().excludeObjs([graspB.obj])
    pbs = inflatedBS(pbs, prob)         # ??

    # When the grasp is -1 (a push), we need the full grasp spec.
    graspBCacheVal = graspB if grasp == -1 else grasp
    key = (pbs, placeB, graspBCacheVal, conf, hand, tuple(base) if base else None, prob,
           nMax, findApproach)
    args = (pbs, placeB, graspB, conf, hand, tuple(base) if base else None, prob,
            nMax, findApproach)
    cache = graspConfGenCache
    val = cache.get(key, None)
    graspConfGenCacheStats[0] += 1
    if val != None:
        graspConfGenCacheStats[1] += 1
        memo = val.copy()
        if debug(tag): print tag, 'cached gen with len(values)=', memo.values
    else:
        memo = Memoizer('potentialGraspConfGen',
                        potentialGraspConfGenAux(*args))
        cache[key] = memo
        if debug(tag): print tag, 'new gen'
    for x in memo:
        assert len(x) == 3 and x[-1] != None
        yield x

class GraspConfHyp(object):
    def __init__(self, grasp, conf, approachConf, viol):
        self.grasp = grasp
        self.conf = conf
        self.approachConf = approachConf
        self.viol = viol
    def __str__(self):
        return 'GraspHyp(%s,%s,%s)'%(self.grasp, self.conf.baseConf(), self.viol)
    __repr__ = __str__
    
# Generator for grasp confs
def potentialGraspConfGenAux(pbs, placeB, graspB, conf, hand, base, prob,
                             nMax=10, findApproach = True):
    def validTestFn(hyp):
        return pbs.inWorkspaceConf(hyp.conf)
    def costFn(hyp):
        wkMargin = pbs.inWorkspaceConfMargin(hyp.conf)
        wkWeight = 5*max(0.1 - wkMargin, 0)
        # print 'wkMargin', wkMargin, 'wkWeight', wkWeight
        print hyp.viol
        print 'collisionDist=', collisionMargin(pbs, prob, hyp.conf)
        return hyp.viol.weight() + baseDist(pbs.getConf(), hyp.conf) + wkWeight
    pbsCopy = pbs.copy()                # so it can be modified 
    hypGen = graspConfHypGen(pbs, placeB, graspB, conf, hand, base, prob,
                             nMax=nMax, findApproach=findApproach)
    for hyp in sortedHyps(hypGen, validTestFn, costFn, nMax, 2*nMax,
                          size=(1 if glob.inHeuristic else 10)):
        if debug('potentialGraspConfGen'):
            pbs.draw(prob, 'W'); hyp.conf.draw('W', 'green')
            debugMsg('potentialGraspConfGen', 'v=%s'%hyp.viol, 'weight=%s'%str(hyp.viol.weight()),
                     'pose=%s'%placeB.poseD.mode(), 'grasp=%s'%graspB.grasp.mode())
        yield hyp.conf, hyp.approachConf, hyp.viol

def graspConfForBase(pbs, placeB, graspB, hand, basePose, prob,
                     wrist = None, counts = None, findApproach = True):
    robot = pbs.getRobot()
    if not wrist:
        wrist = objectGraspFrame(pbs, graspB, placeB, hand)
    basePose = basePose.pose()
    graspConfStats[0] += 1
    # If just the base collides with a perm obstacle, no need to continue
    if baseCollision(pbs, prob, basePose): return
    conf = robot.confFromBaseAndWrist(basePose, hand, wrist, pbs.getConf(), counts)
    if conf is None: return
    # check collisions
    ans = testConfs(pbs, placeB, conf, hand, prob,
                    findApproach=findApproach, counts=counts)
    if ans is None: return
    (_, ca, viol) = ans
    return conf, ca, viol

def testConfs(pbs, placeB, conf, hand, prob, findApproach=True, counts=None):
    viol = Violations()
    testc = [(conf, 'target')]
    ca = None
    if findApproach:
        # get approach conf
        ca = findApproachConf(pbs, placeB.obj, placeB, conf, hand, prob)
        if ca is None:
            if counts: counts[0] += 1       # kin failure
            if debug('potentialGraspConfsLose'):
                pbs.draw(prob, 'W'); conf.draw('W','orange')
                debugMsg('potentialGraspConfsLose', 'no approach conf')
            return
        testc = [(ca, 'approach')] + testc
    for c, ctype in testc:
        viol = pbs.confViolations(c, prob, initViol=viol,
                                  ignoreAttached=True, clearance=graspConfClear)
        if viol is None:                # illegal
            if debug('potentialGraspConfsLose'):
                pbs.draw(prob, 'W'); c.draw('W','red')
                debugMsg('potentialGraspConfsLose', 'collision at %s'%ctype)
            if counts: counts[1] += 1   # collision
            return
    if debug('potentialGraspConfsWin'):
        pbs.draw(prob, 'W'); conf.draw('W','green')
        debugMsg('potentialGraspConfsWin', ('->', conf.conf))
    return c, ca, viol

def baseCollision(pbs, prob, basePose, counts=None):
    baseShape = pbs.getRobot().baseLinkShape(basePose)
    shWorld = pbs.getShadowWorld(prob)
    for perm in shWorld.fixedObjects:
        obst = shWorld.objectShapes[perm]
        if obst.collides(baseShape):
            graspConfStats[1] += 1
            if counts: counts[1] += 1   # collision
            if debug('baseCollision'):
                pbs.draw(prob, 'W')
                obst.draw('W', 'magenta'); baseShape.draw('W', 'magenta')
            tr('baseCollision', 'Base collides with permanent', perm)
            return True
    return False

def gripperCollision(pbs, prob, conf, hand, wrist, counts=None):
    shWorld = pbs.getShadowWorld(prob)
    gripperShape = gripperPlace(conf, hand, wrist)
    for perm in shWorld.fixedObjects:
        obst = shWorld.objectShapes[perm]
        if obst.collides(gripperShape):
            graspConfStats[1] += 1
            if counts: counts[1] += 1   # collision
            if debug('gripperCollision'):
                pbs.draw(prob, 'W')
                obst.draw('W', 'magenta'); gripperShape.draw('W', 'magenta')
            tr('gripperCollision', 'Hand collides with permanent', perm)
            return True
    return False

def graspConfHypGen(pbs, placeB, graspB, conf, hand, base, prob,
                    nMax=10, findApproach=True):
    tag = 'potentialGraspConfs'
    if debug(tag): print 'Entering potentialGraspConfGenAux'
    grasp = graspB.grasp.mode()
    if conf:
        ans = testConfs(pbs, placeB, conf, hand, prob, findApproach=findApproach)
        if ans:
            yield GraspConfHyp(*(grasp,)+ans)
        tr(tag, 'Conf specified; viol is None or out of alternatives')
        return
    wrist = objectGraspFrame(pbs, graspB, placeB, hand)
    if gripperCollision(pbs, prob, pbs.getConf(), hand, wrist): return
    tr(tag, hand, placeB.obj, graspB.grasp, '\n', wrist,
       draw = [(pbs, prob, 'W')], snap = ['W'])
    count = 0
    tried = 0
    robot = pbs.getRobot()
    counts = [0, 0]
    if base:
        (x,y,th) = base
        nominalBasePose = hu.Pose(x, y, 0.0, th)
        for ans in [graspConfForBase(pbs, placeB, graspB, hand,
                                     nominalBasePose, prob, wrist=wrist, counts=counts)]:
            if ans:
                count += 1
                yield GraspConfHyp(*(grasp,)+ans)
        tr(tag, 'Base specified; out of grasp confs for base')
        return
    curBasePose = pbs.getConf().basePose()
    # Try current pose first
    ans = graspConfForBase(pbs, placeB, graspB, hand, curBasePose, prob,
                           wrist=wrist, counts=counts)
    if ans:
        count += 1
        yield GraspConfHyp(*(grasp,)+ans)
    # Try the rest
    # TODO: Sample subset?
    for basePose in robot.potentialBasePosesGen(wrist, hand):
        if nMax and count >= nMax: break
        tried += 1
        ans = graspConfForBase(pbs, placeB, graspB, hand, basePose, prob,
                               wrist=wrist, counts=counts)
        if ans:
            count += 1
            yield GraspConfHyp(*(grasp,)+ans)
    debugMsg('potentialGraspConfs',
             ('Tried', tried, 'found', count, 'potential grasp confs, with grasp', graspB.grasp),
             ('Failed', counts[0], 'invkin', counts[1], 'collisions'))
    return

# This needs generalization
def findApproachConf(pbs, obj, placeB, conf, hand, prob):
    approachConfCacheStats[0] += 1
    cached = pbs.getRoadMap().approachConfs.get(conf, False)
    if cached is not False:
        approachConfCacheStats[1] += 1
        return cached
    robot = pbs.getRobot()
    cart = conf.cartConf()
    wristFrame = cart[robot.armChainNames[hand]]
    if abs(wristFrame.matrix[2,0]) < 0.1: # horizontal
        offset = hu.Pose(-glob.approachBackoff,0.,glob.approachPerpBackoff,0.)
    else:                               # vertical
        offset = hu.Pose(-glob.approachBackoff,0.,0.,0.)
    wristFrameBack = wristFrame.compose(offset)
    cartBack = cart.set(robot.armChainNames[hand], wristFrameBack)
    confBack = robot.inverseKin(cartBack, conf = conf)
    if not None in confBack.values():
        pbs.getRoadMap().approachConfs[conf] = confBack
        return confBack
    else:
        pbs.getRoadMap().approachConfs[conf] = None
        return None
