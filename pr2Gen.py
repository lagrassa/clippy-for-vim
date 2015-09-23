import time
import pdb
from fbch import Function
from dist import DeltaDist
from pr2Util import supportFaceIndex, shadowWidths, trArgs, inside
from pr2PlanBel import getConf, getGoalPoseBels
from shapes import Box
from pr2Fluents import baseConfWithin
from pr2GenAux import *
from planUtil import PPResponse, ObjPlaceB, PoseD
from pr2Push import pushInRegionGenGen

Ident = hu.Transform(np.eye(4))            # identity transform

#  How many candidates to generate at a time...  Larger numbers will
#  generally lead to better solutions.
pickPlaceBatchSize = 3

easyGraspGenCacheStats = [0,0]

# Inside the heuristic, instead of planning a regrasp in detail, just
# generate a different grasp that might be easier to achieve.
class EasyGraspGen(Function):
    # noinspection PyUnusedLocal
    @staticmethod
    def fun(args, goalConds, bState):
        tag = 'easyGraspGen'
        graspVar = 4*(0.1,)                # make precondition even weaker
        graspDelta = 4*(0.005,)            # put back to prev value

        pbs = bState.pbs.copy()
        (obj, hand, face, grasp) = args
        assert obj != None and obj != 'none'
        tr(tag, '(%s,%s) h=%s'%(obj,hand,glob.inHeuristic))
        if obj == 'none' or (goalConds and getConf(goalConds, None)):
            tr(tag, '=> obj is none or conf in goal conds, failing')
            return
        prob = 0.75
        # Set up pbs
        newBS = pbs.copy()
        # Just placements specified in goal
        newBS = newBS.updateFromGoalPoses(goalConds)
        placeB = newBS.getPlaceB(obj)
        shWorld = newBS.getShadowWorld(prob)
        if obj == newBS.held[hand].mode():
            ans = PPResponse(placeB, newBS.graspB[hand], None, None, None, hand)
            tr(tag, 'inHand:'+ str(ans))
            yield ans.easyGraspTuple()
            return
        if obj == newBS.held[otherHand(hand)].mode():
            tr(tag, 'no easy grasp with this hand, failing')
            return
        rm = newBS.getRoadMap()
        graspB = ObjGraspB(obj, pbs.getWorld().getGraspDesc(obj), None,
                           placeB.support.mode(),
                           PoseD(None, graspVar), delta=graspDelta)
        cache = pbs.beliefContext.genCaches[tag]
        key = (newBS, placeB, graspB, hand, prob, face, grasp)
        easyGraspGenCacheStats[0] += 1
        val = cache.get(key, None)
        if val != None:
            if debug(tag): print tag, 'cached'
            easyGraspGenCacheStats[1] += 1
            cached = 'Cached'
            memo = val.copy()
        else:
            if debug(tag): print tag, 'new gen'
            memo = Memoizer(tag,
                            easyGraspGenAux(newBS, placeB, graspB, hand, prob,
                                            face, grasp))
            cache[key] = memo
            cached = ''
        for ans in memo:
            tr(tag, str(ans))
            yield ans.easyGraspTuple()
        tr(tag, '(%s,%s)='%(obj, hand)+'=> out of values')
        return

def easyGraspGenAux(newBS, placeB, graspB, hand, prob, oldFace, oldGrasp):
    tag = 'easyGraspGen'

    def graspApproachConfGen(firstConf):
        if firstConf:
            yield firstConf
        for c, ca, _ in graspConfGen:
            approached[ca] = c
            yield ca

    def pickable(ca, c, pB, gB):
        viol, reason = canPickPlaceTest(newBS, ca, c, hand, gB, pB, prob, op='pick')
        return viol

    if debug(tag): print 'easyGraspGenAux'
    obj = placeB.obj
    approached = {}
    for gB in graspGen(newBS, obj, graspB):
        if gB.grasp.mode() == oldFace and gB.poseD.modeTuple() == oldGrasp:
            tr(tag, 'Rejected %s because same'%gB)
            continue
        tr(tag, 'considering grasp=%s'%gB)

        # TODO: is there a middle road between this and full regrasp?
        #yield PPResponse(placeB, gB, None, None, None, hand)
        
        graspConfGen = potentialGraspConfGen(newBS, placeB, gB, None, hand, None, prob)
        firstConf = next(graspApproachConfGen(None), None)
        if not firstConf:
            tr(tag, 'no confs for grasp = %s'%gB)
            continue
        for ca in graspApproachConfGen(firstConf):
            tr(tag, 'considering conf=%s'%ca.conf)
            viol = pickable(ca, approached[ca], placeB, gB)
            if viol:
                tr(tag, 'pickable')
                yield PPResponse(placeB, gB, approached[ca], ca, viol, hand)
                break
            else:
                tr(tag, 'not pickable')

# R1: Pick bindings that make pre-conditions not inconsistent with goalConds
# R2: Pick bindings so that results do not make conditional fluents in the goalConds infeasible

# Preconditions (for R1):

# 1. Pose(obj) - since there is a Holding fluent in the goal (therefore
# we pick), there cannot be a conflicting Pose fluent

# 2. CanPickPlace(...) - has to be feasible given (permanent)
# placement of objects in the goalConds, but it's ok to violate
# shadows.

# 3. Conf() - if there is Conf in goalConds, then fail.  If there's a
# baseConf in goalConds, then we have to use that base.

# 4. Holding(hand)=none - should not be a different Holding(hand)
# value in goalConds, but there can't be.

# Results (for R2):

# Holding(hand)

class PickGen(Function):
    def fun(self, args, goalConds, bState):
        (obj, graspFace, graspPose,
         objV, graspV, objDelta, confDelta, graspDelta, hand, prob) = args

        base = sameBase(goalConds)          # base is (x, y, th)
        tr('pickGen', 'obj=%s, base=%s'%(obj, base))

        pbs = bState.pbs.copy()
        world = pbs.getWorld()
        graspB = ObjGraspB(obj, world.getGraspDesc(obj), graspFace, None,
                       PoseD(hu.Pose(*graspPose), graspV), delta=graspDelta)
        placeB = ObjPlaceB(obj, world.getFaceFrames(obj), None,
                       PoseD(None,  objV), delta=objDelta)
        # TODO: LPK it is possible that I messed up an onlyCurrent argument here
        for ans in pickGenTop((obj, graspB, placeB, hand, base, prob,),
                          goalConds, pbs):
            yield ans.pickTuple()

def pickGenTop(args, goalConds, pbs, onlyCurrent = False):
    (obj, graspB, placeB, hand, base, prob) = args
    tag = 'pickGen'
    graspDelta = pbs.domainProbs.pickStdev
    tr(tag, '(%s,%s,%d) b=%s h=%s'%(obj,hand,graspB.grasp.mode(),base,glob.inHeuristic))
    trArgs(tag, ('obj', 'graspB', 'placeB', 'hand', 'prob'), args, goalConds, pbs)
    if obj == 'none':                   # can't pick up 'none'
        tr(tag, '=> cannot pick up none, failing')
        return
    if goalConds:
        if getConf(goalConds, None):
            tr(tag, '=> conf is already specified')
            return
    # Set up pbs
    newBS = pbs.copy()
    # Just placements specified in goal
    newBS = newBS.updateFromGoalPoses(goalConds)
    if placeB.poseD.mode() is not None: # specified by, e.g. lookGen
        pose = placeB.poseD.mode()
        sup =  placeB.support.mode()
        newBS.resetPlaceB(obj, placeB)
        tr(tag, 'Setting placeB, support=%s, pose=%s'%(sup, pose.xyztTuple()))
    if obj == newBS.held[hand].mode():
        attachedShape = newBS.getRobot().attachedObj(newBS.getShadowWorld(prob),
                                                   hand)
        shape = newBS.getObjectShapeAtOrigin(obj).\
                applyLoc(attachedShape.origin())
        sup = supportFaceIndex(shape)
        pose = None
        conf = None
        confAppr = None
        tr(tag, 'Object already in hand, support=%s'%sup)
    elif obj == newBS.held[otherHand(hand)].mode():
        attachedShape = newBS.getRobot().attachedObj(newBS.getShadowWorld(prob),
                                                   otherHand(hand))
        shape = newBS.getObjectShapeAtOrigin(obj).\
                applyLoc(attachedShape.origin())
        sup = supportFaceIndex(shape)
        pose = None
        conf = None
        confAppr = None
        tr(tag, 'Object already in other hand, support=%s'%sup)
    else:
        # Use placeB from the current state
        pose = newBS.getPlaceB(obj).poseD.mode()
        sup =  newBS.getPlaceB(obj).support.mode()
        conf = None
        confAppr = None
        tr(tag, 'Using current state, support=%s, pose=%s'%(sup, pose.xyztTuple()))
    # Update placeB
    placeB = ObjPlaceB(obj, placeB.faceFrames, DeltaDist(sup),
                       PoseD(pose, placeB.poseD.var), placeB.delta)
    tr(tag, 'target placeB=%s'%placeB)
    shWorld = newBS.getShadowWorld(prob)
    tr('pickGen', 'Goal conditions', draw=[(newBS, prob, 'W')], snap=['W'])
    gen = pickGenAux(newBS, obj, confAppr, conf, placeB, graspB, hand, base, prob,
                     goalConds, onlyCurrent=onlyCurrent)
    for ans in gen:
        tr(tag, str(ans),
           draw=[(newBS, prob, 'W'),
                 (ans.c, 'W', 'orange', shWorld.attached)],
           snap=['W'])
        yield ans

def pickGenAux(pbs, obj, confAppr, conf, placeB, graspB, hand, base, prob,
               goalConds, onlyCurrent = False):
    def pickable(ca, c, pB, gB):
        return canPickPlaceTest(pbs, ca, c, hand, gB, pB, prob, op='pick')

    def checkInfeasible(conf):
        newBS = pbs.copy()
        newBS.updateConf(conf)
        newBS.updateHeldBel(graspB, hand)
        viol = rm.confViolations(conf, newBS, prob)
        if not viol:                # was valid when not holding, so...
            tr(tag, 'Held collision', draw=[(newBS, prob, 'W')], snap=['W'])
            return True            # punt.

    def graspApproachConfGen(firstConf):
        if firstConf:
            yield firstConf
        for c, ca, _ in graspConfGen:
            approached[ca] = c
            yield ca

    def currentGraspFeasible():
        wrist = objectGraspFrame(pbs, graspB, placeB, hand)

    tag = 'pickGen'
    shw = shadowWidths(placeB.poseD.var, placeB.delta, prob)
    if any(w > t for (w, t) in zip(shw, pbs.domainProbs.pickTolerance)):
        print 'pickGen shadow widths', shw
        print 'poseVar', placeB.poseD.var
        print 'delta', placeB.delta
        print 'prob', prob
        tr(tag, '=> Shadow widths exceed tolerance in pickGen')
        return
    shWorld = pbs.getShadowWorld(prob)
    approached = {}
    rm = pbs.getRoadMap()
    failureReasons = []
    if placeB.poseD.mode() is not None: # otherwise go to regrasp
        if not base:
            # Try current conf
            (x,y,th) = pbs.conf['pr2Base']
            currBasePose = hu.Pose(x, y, 0.0, th)
            confs = graspConfForBase(pbs, placeB, graspB, hand, currBasePose, prob)
            if confs:
                (c, ca, viol) = confs
                ans = PPResponse(placeB, graspB, c, ca, viol, hand)
                cachePPResponse(ans)
                tr(tag, '=>'+str(ans))
                yield ans
        graspConfGen = potentialGraspConfGen(pbs, placeB, graspB, conf, hand, base, prob)
        firstConf = next(graspApproachConfGen(None), None)
        # This used to have an or clause
        # (firstConf and checkInfeasible(firstConf))
        # but infeasibility of one of the grasp confs due to held
        # object does not guarantee there are no solutions.
        if (not firstConf):
            if not onlyCurrent:
                tr(tag, 'No potential grasp confs, will need to regrasp',
                   draw=[(pbs, prob, 'W')], snap=['W'])
                if True: # debug(tag):
                    raw_input('pickGen: Cannot find grasp conf for current pose of ' + obj)
                else: print 'pickGen: Cannot find graspconf for current pose of', obj
        else:
            targetConfs = graspApproachConfGen(firstConf)
            batchSize = 1 if glob.inHeuristic else pickPlaceBatchSize
            batch = 0
            while True:
                # Collect the next batch of trialConfs
                batch += 1
                trialConfs = []
                count = 0
                minCost = 1e6
                for ca in targetConfs:       # targetConfs is a generator
                    viol, reason = pickable(ca, approached[ca], placeB, graspB)
                    if viol:
                        trialConfs.append((viol.weight(), viol, ca))
                        minCost = min(viol.weight(), minCost)
                    else:
                        failureReasons.append(reason)
                        tr(tag, 'target conf failed: ' + reason)
                        continue
                    count += 1
                    if count == batchSize or minCost == 0: break
                if count == 0: break
                trialConfs.sort()
                for _, viol, ca in trialConfs:
                    c = approached[ca]
                    ans = PPResponse(placeB, graspB, c, ca, viol, hand)
                    cachePPResponse(ans)
                    tr(tag, 
                       'currently graspable ->'+str(ans), 'viol: %s'%(ans.viol),
                       draw=[(pbs, prob, 'W'),
                             (placeB.shape(pbs.getShadowWorld(prob)), 'W', 'navy'),
                             (c, 'W', 'navy', shWorld.attached)],
                       snap=['W'])
                    yield ans
    else:
        if debug(tag): raw_input('Current pose is not known, need to regrasp')
        else: print 'Current pose is not known, need to regrasp'

    if onlyCurrent:
        tr(tag, 'onlyCurrent: out of values')
        return
        
    # Try a regrasp... that is place the object somewhere else where it can be grasped.
    if glob.inHeuristic:
        return
    if failureReasons and all(['visibility' in reason for reason in failureReasons]):
        tr(tag, 'There were valid targets that failed due to visibility')
        return
    
    tr(tag, 'Calling for regrasping... h=%s'%glob.inHeuristic)

    # !! Needs to look for plausible regions...
    regShapes = regShapes = [shWorld.regionShapes[region] for region in pbs.awayRegions()]
    plGen = placeInGenTop((obj, regShapes, graspB, placeB, None, prob),
                          goalConds, pbs, regrasp = True)
    for ans in plGen:
        v, reason = pickable(ans.ca, ans.c, ans.pB, ans.gB)
        ans = ans.copy()
        ans.viol = v
        tr(tag, 'Regrasp pickable=%s'%ans,
           draw=[(pbs, prob, 'W'), (ans.c, 'W', 'blue', shWorld.attached)],
           snap=['W'])
        if v:
            yield ans
    tr(tag, '=> out of values')

# Preconditions (for R1):

# 1. CanPickPlace(...) - has to be feasible given (permanent)
# placement of objects in the goalConds, but it's ok to violate
# shadows.

# 2. Holding(hand) - should not suggest h if goalConds already has
# Holding(h)

# 3. Conf() - if there is Conf in goalConds, then fail.  If there's a
# baseConf in goalConds, then we have to use that base.

# Results (for R2):

# Pose(obj)
# Holding(hand) = none

# Returns (hand, graspMu, graspFace, graspConf,  preConf)

class PlaceGen(Function):
    def fun(self, args, goalConds, bState):
        for ans in placeGenGen(args, goalConds, bState):
            tr('placeGen', str(ans))
            yield ans.placeTuple()

# Either hand or poses will be specified, but generally not both.  They will never both be unspecified.
def placeGenGen(args, goalConds, bState):
    (obj, hand, poses, support, objV, graspV, objDelta, graspDelta, confDelta,
     prob) = args
    tag = 'placeGen'
    base = sameBase(goalConds)
    tr(tag, 'obj=%s, base=%s'%(obj, base))
    # tr(tag, ('args', args))
    if goalConds:
        if getConf(goalConds, None):
            tr(tag, '=> conf is already specified, failing')
            return

    pbs = bState.pbs.copy()
    world = pbs.getWorld()

    if poses == '*' or isVar(poses) or support == '*' or isVar(support):
        tr(tag, 'Unspecified pose')
        if base:
            # Don't try to keep the same base, if we're trying to
            # place the object away.
            tr(tag, '=> unspecified pose with same base constraint, failing')
            return
        assert not isVar(hand)
        
        # Just placements specified in goal (and excluding obj)
        # placeInGenAway does not do this when calling placeGen
        newBS = pbs.copy()
        newBS = newBS.updateFromGoalPoses(goalConds, updateConf=False)
        newBS = newBS.excludeObjs([obj])
        # v is viol
        for ans in placeInGenAway((obj, objDelta, prob), goalConds, newBS):
            yield ans
        return

    if isinstance(poses, tuple):
        placeB = ObjPlaceB(obj, world.getFaceFrames(obj), support,
                           PoseD(poses, objV), delta=objDelta)
        placeBs = frozenset([placeB])
    else:
        raw_input('placeGenGen - poses is not a tuple')

    graspB = ObjGraspB(obj, world.getGraspDesc(obj), None, None,
                       PoseD(None, graspV), delta=graspDelta)
        
    # Figure out whether one hand or the other is required;  if not, do round robin
    leftGen = placeGenTop((obj, graspB, placeBs, 'left', base, prob),
                                 goalConds, pbs)
    rightGen = placeGenTop((obj, graspB, placeBs, 'right', base, prob),
                                 goalConds, pbs)
    
    for ans in chooseHandGen(pbs, goalConds, obj, hand, leftGen, rightGen):
        yield ans

placeGenCacheStats = [0, 0]
placeGenCache = {}

# returns values for (?graspPose, ?graspFace, ?conf, ?confAppr)
def placeGenTop(args, goalConds, pbs, regrasp=False, away=False, update=True):
    (obj, graspB, placeBs, hand, base, prob) = args

    startTime = time.clock()
    tag = 'placeGen'
    tr(tag, '(%s,%s) h=%s'%(obj,hand, glob.inHeuristic))
    trArgs(tag, ('obj', 'graspB', 'placeBs', 'hand', 'prob'), args, goalConds, pbs)
    if obj == 'none' or not placeBs:
        tr(tag, '=> obj is none or no poses, failing')
        return
    if goalConds:
        if getConf(goalConds, None) and not away:
            tr(tag, '=> goal conf specified and not away, failing')
            return
        for (h, o) in getHolding(goalConds):
            if h == hand:
                tr(tag, '=> Hand=%s is already Holding, failing'%hand)
                return
    conf = None
    confAppr = None
    # Set up pbs
    newBS = pbs.copy()
    # Just placements specified in goal (and excluding obj)
    if update:                          # could be done by caller
        newBS = newBS.updateFromGoalPoses(goalConds, updateConf=not away)
        newBS = newBS.excludeObjs([obj])
    tr(tag, 'Goal conditions', draw=[(newBS, prob, 'W')], snap=['W'])

    if isinstance(placeBs, frozenset):
        for pB in placeBs:
            ans = drop(newBS, prob, obj, hand, pB)
            if ans:
                shWorld = newBS.getShadowWorld(prob)
                tr(tag, 'Cached place ->' + str(ans), 'viol=%s'%ans.viol,
                   draw=[(newBS, prob, 'W'),
                         (ans.pB.shape(shWorld), 'W', 'magenta'),
                         (ans.c, 'W', 'magenta', shWorld.attached)],
                   snap=['W'])
                yield ans

    key = (newBS, pbs,
           (obj, graspB, placeBs, hand, tuple(base) if base else None, prob),
           regrasp, away, update)
    val = placeGenCache.get(key, None)
    placeGenCacheStats[0] += 1
    if val is not None:
        placeGenCacheStats[1] += 1
        # Will restart the generator when it is retrieved
        memo = val.copy()
    else:
        if isinstance(placeBs, frozenset):
            def placeBGen():
                for placeB in placeBs: yield placeB
            placeBG = Memoizer('placeBGen_placeGen', placeBGen())
        else:
            placeBG = placeBs
        memo = Memoizer(tag,
                        placeGenAux(newBS, obj, confAppr, conf, placeBG.copy(),
                                    graspB, hand, base, prob,
                                    regrasp=regrasp, pbsOrig = pbs))
        placeGenCache[key] = memo
    for ans in memo:
        tr(tag, str(ans) +' (t=%s)'%(time.clock()-startTime))
        yield ans

def placeGenAux(pbs, obj, confAppr, conf, placeBs, graspB, hand, base, prob,
                regrasp=False, pbsOrig=None):
    def placeable(ca, c, quick=False):
        (pB, gB) = context[ca]
        return canPickPlaceTest(pbs, ca, c, hand, gB, pB, prob,
                                op='place', quick=quick)

    def checkRegraspable(pB):
        if pB in regraspablePB:
            return regraspablePB[pB]
        other =  [next(potentialGraspConfGen(pbs, pB, gBO, conf, hand, base, prob, nMax=1),
                       (None, None, None))[0] \
                  for gBO in gBOther]
        if any(other):
            tr(tag,
               ('Regraspable', pB.poseD.mode(), [gBO.grasp.mode() for gBO in gBOther]),
               draw=[(c, 'W', 'green') for c in \
                     [o for o in other if o != None]], snap=['W'])
            regraspablePB[pB] = True
            return True
        else:
            regraspablePB[pB] = False
            tr(tag, ('Not regraspable', pB.poseD.mode()))
            return False

    def checkOrigGrasp(gB):
        # 0 if currently true
        # 1 if could be used on object's current position
        # 2 otherwise
        
        # Prefer current grasp
        if obj == pbsOrig.held[hand].mode():
            currGraspB = pbsOrig.graspB[hand]
            match = (gB.grasp.mode() == currGraspB.grasp.mode()) and \
                      gB.poseD.mode().near(currGraspB.poseD.mode(), .01, .01)
            if match:
                tr(tag, 'current grasp is a match',
                   ('curr', currGraspB), ('desired', gB))
                return 0

        minConf = minimalConf(pbsOrig.conf, hand)
        if minConf in PPRCache:
            ppr = PPRCache[minConf]
            currGraspB = ppr.gB
            match = (gB.grasp.mode() == currGraspB.grasp.mode()) and \
                      gB.poseD.mode().near(currGraspB.poseD.mode(), .01, .01)
            if match:
                tr(tag, 'cached grasp for curr conf is a match',
                   ('curr', currGraspB), ('desired', gB))
                return 0

        pB = pbsOrig.getPlaceB(obj, default=False) # check we know where obj is.
        if pbsOrig and pbsOrig.held[hand].mode() != obj and pB:
            nextGr = next(potentialGraspConfGen(pbsOrig, pB, gB, conf, hand, base,
                                          prob, nMax=1),
                              (None, None, None))
            # !!! LPK changed this because next was returning None
            if nextGr and nextGr[0]:
                return 1
            else:
                if debug(tag) and gB.grasp.mode() == 0:
                    pbsOrig.draw(prob, 'W')
                    print 'cannot use grasp 0'
                return 2
        else:
            return 1

    def placeApproachConfGen(grasps):
        placeBsCopy = placeBs.copy()
        for pB in placeBsCopy:          # re-generate
            for gB in grasps:
                tr(tag, 
                   ('considering grasps for ', pB),
                   ('for grasp class', gB.grasp),
                   ('placeBsCopy.values', len(placeBsCopy.values)))
                if regrasp:
                    checkRegraspable(pB)
                graspConfGen = potentialGraspConfGen(pbs, pB, gB, conf, hand, base, prob)
                count = 0
                for c,ca,_ in graspConfGen:
                    tr(tag, 'Yielding grasp approach conf',
                       draw=[(pbs, prob, 'W'), (c, 'W', 'orange', shWorld.attached)],
                       snap=['W'])
                    approached[ca] = c
                    count += 1
                    context[ca] = (pB, gB)
                    yield ca
                    # if count > 2: break # !! ??
                tr(tag, 'found %d confs'%count)

    def regraspCost(ca):
        if not regrasp:
            # if debug('placeGen'): print 'not in regrasp mode, cost = 0'
            return 0
        (pB, gB) = context[ca]
        if pB in regraspablePB:
            if regraspablePB[pB]:
                # if debug('placeGen'): print 'regrasp cost = 0'
                return 0
            else:
                # if debug('placeGen'): print 'regrasp cost = 5'
                return 5
        else:
            # if debug('placeGen'): print 'unknown pB, cost = 0'
            return 0

    tag = 'placeGen'
    approached = {}
    context = {}
    regraspablePB = {}
    rm = pbs.getRoadMap()
    shWorld = pbs.getShadowWorld(prob)
    if regrasp:
         graspBOther = graspB.copy()
         otherGrasps = range(len(graspBOther.graspDesc))
         otherGrasps.remove(graspB.grasp.mode())
         if otherGrasps:
             graspBOther.grasp = UniformDist(otherGrasps)
             gBOther = list(graspGen(pbs, obj, graspBOther))
         else:
             gBOther = []

    allGrasps = [(checkOrigGrasp(gB), gB) for gB in graspGen(pbs, obj, graspB)]
    gClasses, gCosts = groupByCost(allGrasps)

    tr(tag, 'Top grasps', [g.grasp.mode() for g in gClasses[0]], 'costs', gCosts)

    for grasps, gCost in zip(gClasses, gCosts):
        targetConfs = placeApproachConfGen(grasps)
        batchSize = 1 if glob.inHeuristic else pickPlaceBatchSize
        batch = 0
        while True:
            # Collect the next batach of trialConfs
            batch += 1
            trialConfs = []
            count = 0
            minCost = 1e6
            for ca in targetConfs:   # targetConfs is a generator
                viol, reason = placeable(ca, approached[ca])
                if viol:
                    cost = viol.weight() + gCost + regraspCost(ca)
                    minCost = min(cost, minCost)
                    trialConfs.append((cost, viol, ca))
                else:
                    tr(tag, 'Failure of placeable: '+reason)
                    continue
                count += 1
                if count == batchSize or minCost == 0: break
            if count == 0: break
            pbs.getShadowWorld(prob)
            trialConfs.sort()
            for _, viol, ca in trialConfs:
                (pB, gB) = context[ca]
                c = approached[ca]
                ans = PPResponse(pB, gB, c, ca, viol, hand)
                cachePPResponse(ans)
                tr(tag, '->' + str(ans), 'viol=%s'%viol,
                   draw=[(pbs, prob, 'W'),
                         (pB.shape(shWorld), 'W', 'magenta'),
                         (c, 'W', 'magenta', shWorld.attached)],
                   snap=['W'])
                yield ans
    tr(tag, 'out of values')

# Preconditions (for R1):

# 1. Pose(obj) - pick pose that does not conflict with what goalConds
# say about this obj.  So, if Pose in goalConds with smaller variance
# works, then fine, but otherwise a Pose in goalConds should cause
# failure.

# Results (for R2):

# In(obj, Region)

class PoseInRegionGen(Function):
    # Return objPose, poseFace.
    def fun(self, args, goalConds, bState):
        for ans in roundrobin(placeInRegionGenGen(args, goalConds, bState, away = False),
                              pushInRegionGenGen(args, goalConds, bState, away = False)):
            if ans:
                yield ans.poseInTuple()

def placeInRegionGenGen(args, goalConds, bState, away = False, update=True):
    (obj, region, var, delta, prob) = args
    tag = 'placeInGen'
    pbs = bState.pbs.copy()
    world = pbs.getWorld()

    tr(tag, args)

    # If there are no grasps, just fail
    if not world.getGraspDesc(obj): return

    # Get the regions
    if not isinstance(region, (list, tuple, frozenset)):
        regions = frozenset([region])
    elif len(region) == 0:
        raise Exception, 'need a region to place into'
    else:
        regions = frozenset(region)
    shWorld = pbs.getShadowWorld(prob)
    regShapes = [shWorld.regionShapes[region] for region in regions]
    tr(tag, 'Target region in purple',
       draw=[(pbs, prob, 'W')] + [(rs, 'W', 'purple') for rs in regShapes],
       snap=['W'])
    # Set pose and support from current state
    pose = None
    if pbs.getPlaceB(obj, default=False):
        # If it is currently placed, use that support
        support = pbs.getPlaceB(obj).support.mode()
        pose = pbs.getPlaceB(obj).poseD.mode()
    elif obj == pbs.held['left'].mode():
        attachedShape = pbs.getRobot().attachedObj(pbs.getShadowWorld(prob),
                                                   'left')
        shape = pbs.getObjectShapeAtOrigin(obj).\
                applyLoc(attachedShape.origin())
        support = supportFaceIndex(shape)
    elif obj == pbs.held['right'].mode():
        attachedShape = pbs.getRobot().attachedObj(pbs.getShadowWorld(prob),
                                                   'right')
        shape = pbs.getObjectShapeAtOrigin(obj).\
                applyLoc(attachedShape.origin())
        support = supportFaceIndex(shape)
    else:
        raise Exception('Cannot determine support')

    graspV = bState.domainProbs.maxGraspVar
    graspDelta = bState.domainProbs.graspDelta
    graspB = ObjGraspB(obj, world.getGraspDesc(obj), None, None,
                       PoseD(None, graspV), delta=graspDelta)

    # Check if object pose is specified in goalConds
    poseBels = getGoalPoseBels(goalConds, world.getFaceFrames)
    if obj in poseBels:
        pB = poseBels[obj]
        shw = shadowWidths(pB.poseD.var, pB.delta, prob)
        shwMin = shadowWidths(graspV, graspDelta, prob)
        if any(w > mw for (w, mw) in zip(shw, shwMin)):
            args = (obj, None, pB.poseD.modeTuple(),
                    support, var, graspV,
                    delta, graspDelta, None, prob)
            gen = placeGenGen(args, goalConds, bState)
            for ans in gen:
                regions = [x.name() for x in regShapes]
                tr(tag, str(ans), 'regions=%s'%regions,
                   draw=[(pbs, prob, 'W')] + [(rs, 'W', 'purple') for rs in regShapes],
                   snap=['W'])
                yield ans
            return
        else:
            # If pose is specified and variance is small, return
            return

    # Check whether just "dropping" the object achieves the result
    ans = dropIn(pbs, prob, obj, regShapes)
    if ans:
        shWorld = pbs.getShadowWorld(prob)
        tr(tag, 'Cached placeIn ->' + str(ans), 'viol=%s'%ans.viol,
           draw=[(pbs, prob, 'W'),
                 (ans.pB.shape(shWorld), 'W', 'magenta'),
                 (ans.c, 'W', 'magenta', shWorld.attached)],
           snap=['W'])
        yield ans
    else:
        tr(tag, 'Drop in is not applicable')

    # The normal case

    # Use the input var and delta to select candidate poses in the
    # region.  We will use smaller values (in general) for actually
    # placing.
    placeB = ObjPlaceB(obj, world.getFaceFrames(obj), support,
                       PoseD(pose, var), delta=delta)

    gen = placeInGenTop((obj, regShapes, graspB, placeB, None, prob),
                          goalConds, pbs, away = away, update=update)
    for ans in gen:
        yield ans

placeVarIncreaseFactor = 3 # was 2
lookVarIncreaseFactor = 2

def dropIn(pbs, prob, obj, regShapes):
    hand = None
    for h in ('left', 'right'):
        minConf = minimalConf(pbs.conf, h)
        if minConf in PPRCache:
            hand = h
            break
    if not hand: return
    shWorld = pbs.getShadowWorld(prob)
    ppr = PPRCache[minConf]
    assert ppr.hand == hand
    if obj == pbs.held[hand].mode():
        robShape, attachedPartsDict = pbs.conf.placementAux(attached=shWorld.attached)
        shape = attachedPartsDict[hand]
        (x,y,z,t) = shape.origin().pose().xyztTuple()
        for regShape in regShapes:
            rz = regShape.bbox()[0,2] + 0.01
            dz = rz - shape.bbox()[0,2]
            pose = (x,y,z+dz,t)
            support = supportFaceIndex(shape)
            rshape = shape.applyLoc(hu.Pose(*pose))
            regShape.draw('W', 'purple')
            rshape.draw('W', 'green')
            if inside(rshape.xyPrim(), regShape):
                if canPickPlaceTest(pbs, ppr.ca, ppr.c, ppr.hand, ppr.gB, ppr.pB, prob,
                            op='place')[0]:
                    return ppr

def drop(pbs, prob, obj, hand, placeB):
    minConf = minimalConf(pbs.conf, hand)
    if minConf in PPRCache:
        ppr = PPRCache[minConf]
    else:
        return
    assert ppr.hand == hand
    shWorld = pbs.getShadowWorld(prob)
    if obj == pbs.held[hand].mode():
        xt = ppr.pB.poseD.mode().xyztTuple()
        cxt = placeB.poseD.mode().xyztTuple()
        if max([abs(a-b) for (a,b) in zip(xt,cxt)]) < 0.001:
            if canPickPlaceTest(pbs, ppr.ca, ppr.c, ppr.hand, ppr.gB, ppr.pB, prob,
                            op='place')[0]:
                return ppr

def placeInGenAway(args, goalConds, pbs):
    # !! Should search over regions and hands
    (obj, delta, prob) = args
    if not pbs.awayRegions():
        raw_input('Need some awayRegions')
        return
    tr('placeInGenAway', zip(('obj', 'delta', 'prob'), args),
       draw=[(pbs, prob, 'W')], snap=['W'])
    targetPlaceVar = tuple([placeVarIncreaseFactor * x \
                            for x in pbs.domainProbs.obsVarTuple])
    # Pass in the goalConds to get reachObsts, but don't do the update of
    # the pbs, since achCanXGen has already done it.
    for ans in placeInRegionGenGen((obj, pbs.awayRegions(),
                                    targetPlaceVar, delta, prob),
                                   goalConds, pbs, away=True, update=False):
        yield ans

placeInGenMaxPoses  = 300
placeInGenMaxPosesH = 300

def placeInGenTop(args, goalConds, pbs,
                  regrasp=False, away = False, update=True):
    (obj, regShapes, graspB, placeB, base, prob) = args
    tag = 'placeInGen'
    regions = [x.name() for x in regShapes]
    tr(tag, '(%s,%s) h=%s'%(obj,regions, glob.inHeuristic))
    tr(tag, 
       zip(('obj', 'regShapes', 'graspB', 'placeB', 'prob'), args))
    if obj == 'none' or not regShapes:
        # Nothing to do
        tr(tag, '=> object is none or no regions, failing')
        return
    if goalConds and getConf(goalConds, None) and not away:
        # if conf is specified, just fail
        tr(tag, '=> conf is specified, failing')
        return

    conf = None
    confAppr = None
    # Obstacles for all Reachable fluents
    reachObsts = getReachObsts(goalConds, pbs)
    if reachObsts is None:
        tr(tag, '=> No path for reachObst, failing')
        return
    tr(tag, '%d reachObsts - in orange'%len(reachObsts),
       draw=[(pbs, prob, 'W')] + [(obst, 'W', 'orange') for _,obst in reachObsts],
       snap=['W'])
    newBS = pbs.copy()           #  not necessary
    pB = placeB
    shWorld = newBS.getShadowWorld(prob)
    nPoses = placeInGenMaxPosesH if glob.inHeuristic else placeInGenMaxPoses
    poseGenLeft = Memoizer('regionPosesLeft',
                           potentialRegionPoseGen(newBS, obj, pB, graspB, prob, regShapes,
                                                  reachObsts, 'left', base,
                                                  maxPoses=nPoses))
    poseGenRight = Memoizer('regionPosesRight',
                            potentialRegionPoseGen(newBS, obj, pB, graspB, prob, regShapes,
                                                   reachObsts, 'right', base,
                                                   maxPoses=nPoses))
    # note the use of PB...
    leftGen = placeInGenAux(newBS, poseGenLeft, goalConds, confAppr,
                            conf, pB, graspB, 'left', base, prob,
                            regrasp=regrasp, away=away, update=update)
    rightGen = placeInGenAux(newBS, poseGenRight, goalConds, confAppr,
                             conf, pB, graspB, 'right', base, prob,
                             regrasp=regrasp, away=away, update=update)
    # Figure out whether one hand or the other is required;  if not, do round robin
    mainGen = chooseHandGen(newBS, goalConds, obj, None, leftGen, rightGen)

    # Picks among possible target poses and then try to place it in region
    for ans in mainGen:
        tr(tag, str(ans),
           draw=[(ans.c, 'W', 'green', shWorld.attached)] + \
           [(rs, 'W', 'purple') for rs in regShapes],
           snap=['W'])
        yield ans

# Don't try to place all objects at once
def placeInGenAux(pbs, poseGen, goalConds, confAppr, conf, placeB, graspB,
                  hand, base, prob, regrasp=False, away=False, update=True):

    def placeBGen():
        for pose in poseGen.copy():
            yield placeB.modifyPoseD(mu=pose)
    tries = 0
    shWorld = pbs.getShadowWorld(prob)
    gen = Memoizer('placeBGen_placeInGenAux1', placeBGen())
    for ans in placeGenTop((graspB.obj, graspB, gen, hand, base, prob),
                           goalConds, pbs, regrasp=regrasp, away=away, update=update):
        tr('placeInGen', ('=> blue', str(ans)),
           draw=[(pbs, prob, 'W'),
                 (ans.pB.shape(shWorld), 'W', 'blue'),
                 (ans.c, 'W', 'blue', shWorld.attached)],
           snap=['W'])
        yield ans

PPRCache = {}
def cachePPResponse(ppr):
    minConf = minimalConf(ppr.ca, ppr.hand)
    if minConf not in PPRCache:
        PPRCache[minConf] = ppr
    else:
        old = PPRCache[minConf]
        if old.gB.grasp.mode() != ppr.gB.grasp.mode() \
               or old.pB.poseD.mode() != ppr.pB.poseD.mode():
            print 'PPRCache collision'
            pdb.set_trace()

maxLookDist = 1.5

# Preconditions (for R1):

# 1. CanSeeFrom() - make a world from the goalConds and CanSeeFrom (visible) should be true.

# 2. Conf() - if there is Conf in goalConds, then fail.  If there's a
# baseConf in goalConds, then we have to use that base.

# Results (for R2):

# Condition to avoid violating future canReach
# If we're in shadow in starting state, ok.   Otherwise, don't walk into a shadow.

# Returns lookConf
# The lookDelta is a slop factor.  Ideally if the robot is within that
# factor, visibility should still hold.
class LookGen(Function):
    def fun(self, args, goalConds, bState):
        (obj, pose, support, objV_before, objV_after, objDelta, lookDelta, prob) = args
        pbs = bState.pbs.copy()
        world = pbs.getWorld()
        base = sameBase(goalConds)      # base specified in goalConds
        # Use current mean pose if pose is not specified.
        if pose == '*':
            # This could produce a mode of None if object is held
            pB = pbs.getPlaceB(obj, default=False)
            if pB is None:
                tr('lookGen', '=> Trying to reduce variance on object pose but obj is in hand')
                return
            pose = pB.poseD.mode()
        # Use current support if it is not specified.
        if isVar(support) or support == '*':
            support = pbs.getPlaceB(obj).support.mode()
        # Use the lookDelta is objDelta is not specified.
        if objDelta == '*':
            objDelta = lookDelta
        # Pose distributions before and after the look
        poseD_before = PoseD(pose, objV_before)
        poseD_after = PoseD(pose, objV_after)
        placeB_before = ObjPlaceB(obj, world.getFaceFrames(obj),
                                  support, poseD_before,
                                  delta = objDelta)
        placeB_after = ObjPlaceB(obj, world.getFaceFrames(obj),
                                 support, poseD_after,
                                delta = objDelta)
        # ans = (lookConf,)
        for ans, viol in lookGenTop((obj, placeB_before, placeB_after,
                                     lookDelta, base, prob),
                                goalConds, pbs):
            yield ans

# Returns (lookConf,), viol
def lookGenTop(args, goalConds, pbs):
    # This checks that from conf c, sh is visible (not blocked by
    # fixed obstacles).  The obst are "movable" obstacles.  Returns
    # boolean.
    def testFn(c, sh, shWorld):
        tr(tag, 'Trying base conf', c['pr2Base'], ol = True)
        obst = [s for s in shWorld.getNonShadowShapes() if s.name() != obj ]
        # We don't need to add the robot, since it can be moved out of the way.
        # obst_rob = obst + [c.placement(shWorld.attached)]
        # visible returns (bool, occluders), return only the boolean
        return visible(shWorld, c, sh, obst, prob, moveHead=True)[0]

    (obj, placeB_before, placeB_after, lookDelta, base, prob) = args
    tag = 'lookGen'
    tr(tag, '(%s) h=%s'%(obj, glob.inHeuristic))
    if placeB_before.poseD.mode() is None:
        tr(tag, '=> object is in the hand, failing')
        return    
    # Condition the pbs on goalConds
    newBS = pbs.copy()
    newBS = newBS.updateFromGoalPoses(goalConds)
    # Creat planning contexts for before look (large variance)...
    newBS_before = newBS.copy().updatePermObjPose(placeB_before)
    newBS_before.addAvoidShadow([obj])
    shWorld_before = newBS_before.getShadowWorld(prob)
    # ... and after look (lower variance)
    newBS_after = newBS.copy().updatePermObjPose(placeB_after)
    newBS_after.addAvoidShadow([obj])
    shWorld_after = newBS_after.getShadowWorld(prob)
    # Some temp values that are independent of before/after
    rm = newBS_before.getRoadMap()      # road map
    attached = shWorld_before.attached  # attached
    if any(attached.values()):
        tr(tag, 'attached=%s'%attached)
    shName = shadowName(obj)
    world = newBS_before.getWorld()
    # mode shape, ignoring variance and delta, use as target shape.
    shapeForLook = placeB_before.shape(shWorld_before)
    shapeShadow = shWorld_before.objectShapes[shName]

    # newBS.draw(prob, 'W')
    # shapeForLook.draw('W', 'orange')
    # shapeShadow.draw('W', 'gray')
    # raw_input('Me and my shadow')

    # Check if conf is specified in goalConds
    goalConf = getConf(goalConds, None)
    if goalConds and goalConf:
        # if conf is specified, just fail
        tr(tag, '=> Conf is specified, failing: ' + str(goalConf))
        return
    # Check if object is in the hand
    if obj in [newBS_before.held[hand].mode() for hand in ['left', 'right']]:
        tr(tag, '=> object is in the hand, failing')
        return

    # Handle case where the base is specified.
    if base:
        # Use the conf in goalConds to "fill in" the base information
        confAtTarget = targetConf(goalConds)
        if confAtTarget is None:
            print 'No conf found for lookConf with specified base'
            raw_input('This might be an error in regression')
            return
        # We must be able to reach confAtTarget after the look, but
        # someone else has/will make sure that is true.  Start by
        # checking that confAtTarget is not blocked by fixed
        # obstacles.
        if testFn(confAtTarget, shapeForLook, shWorld_before):
            # Is current conf good enough?  If not -
            # Modify the lookConf (if needed) by moving arm out of the
            # way of the viewCone.  Use the before shadow because this
            # conf needs to be safe before we look
            delta = pbs.domainProbs.moveConfDelta
            if baseConfWithin(pbs.conf['pr2Base'], base, delta):
                curLookConf = lookAtConfCanView(newBS_after, prob, newBS_after.conf,
                                                shapeForLook,
                                                shapeShadow=shapeShadow,
                                                findPath=False)
            else:
                curLookConf = None
            lookConf = curLookConf or \
                       lookAtConfCanView(newBS_after, prob, confAtTarget,
                                         shapeForLook, shapeShadow=shapeShadow)
            if lookConf:
                tr(tag, '=> Found a path to look conf with specified base.',
                   ('-> cyan', lookConf.conf),
                   draw=[(newBS_before, prob, 'W'),
                         (lookConf, 'W', 'cyan', attached)],
                   snap=['W'])
                yield (lookConf,), rm.confViolations(lookConf, newBS_after,prob)
            else:
                tr(tag,
                   '=> Failed to find path to look conf with specified base.',
                   ('target conf after look is magenta', confAtTarget.conf),
                   draw=[(newBS_before, prob, 'W'),
                         (confAtTarget, 'W', 'magenta', attached)],
                   snap=['W'])
        return

    # Check if the current conf will work for the look
    curr = newBS_before.conf
    if testFn(curr, shapeForLook, shWorld_before): # visible?
        # move arm out of the way if necessary, use after shadow
        lookConf = lookAtConfCanView(newBS_after, prob, curr,
                                     shapeForLook, shapeShadow=shapeShadow)
        if lookConf:
            tr(tag, '=> Using current conf.',
               draw=[(newBS_before, prob, 'W'),
                     (lookConf, 'W', 'cyan', attached)])
            yield (lookConf,), rm.confViolations(lookConf, newBS_after, prob)

    # If we're looking at graspable objects, prefer a lookConf from
    # which we could pick the object, so that we don't have to move
    # the base.

    # TODO: Generalize this to (pick or push)

    if len(world.getGraspDesc(obj)) > 0 and not glob.inHeuristic:
        graspVar = 4*(0.001,)
        graspDelta = 4*(0.001,)
        graspB = ObjGraspB(obj, world.getGraspDesc(obj), None, None,
                           PoseD(None, graspVar), delta=graspDelta)
        # Use newBS (which doesn't declare shadow to be permanent) to
        # generate candidate confs, since they will need to collide
        # with shadow of obj.
        for gB in graspGen(newBS, obj, graspB):
            for hand in ['left', 'right']:
                for ans in pickGenTop((obj, gB, placeB_after, hand, base, prob),
                                      goalConds, newBS, onlyCurrent=True):
                    # Modify the approach conf so that robot avoids view cone.
                    lookConf = lookAtConfCanView(newBS, prob, ans.ca,
                                                 shapeForLook, shapeShadow=shapeShadow)
                    if not lookConf:
                        tr(tag, 'canView failed')
                        continue
                    # We should be guaranteed that this is true, since
                    # ca was chosen so that the shape is visible.
                    if testFn(lookConf, shapeForLook, shWorld_before):
                        # Find the violations at that lookConf
                        viol = rm.confViolations(lookConf, newBS, prob)
                        vw = viol.weight() if viol else None
                        tr(tag, '(%s) canView cleared viol=%s'%(obj, vw))
                        yield (lookConf,), viol
                    else:
                        assert None, 'shape should be visible, but it is not'
    # Find a lookConf unconstrained by base
    lookConfGen = potentialLookConfGen(newBS_before, prob, shapeForLook, maxLookDist)
    for ans in rm.confReachViolGen(lookConfGen, newBS_before, prob,
                                   testFn = lambda c: testFn(c, shapeForLook, shWorld_before)):
        viol, cost, path = ans
        tr(tag, '(%s) viol=%s'%(obj, viol.weight() if viol else None))
        if not path:
            tr(tag,  'Failed to find a path to look conf.')
            raw_input('Failed to find a path to look conf.')
            continue
        conf = path[-1]                 # lookConf is at the end of the path
        # Modify the look conf so that robot does not block
        lookConf = lookAtConfCanView(newBS_before, prob, conf,
                                     shapeForLook, shapeShadow=shapeShadow)
        if lookConf:
            tr(tag, '(%s) general conf viol=%s'%(obj, viol.weight() if viol else None),
               ('-> cyan', lookConf.conf),
               draw=[(newBS_before, prob, 'W'),
                     (lookConf, 'W', 'cyan', attached)],
               snap=['W'])
            yield (lookConf,), viol

# Computes lookConf for shape, makes sure that the robot does not
# block the view cone.  It will construct a path from the input conf
# to the returned lookConf if necessary - the path is not returned,
# only the final conf.
def lookAtConfCanView(pbs, prob, conf, shape, hands=('left', 'right'),
                      shapeShadow=None, findPath=True):
    lookConf = lookAtConf(conf, shape)  # conf with head looking at shape
    if not glob.inHeuristic:            # if heuristic we'll ignore robot
        for hand in hands:              # consider each hand in turn
            if not lookConf:
                tr('lookAtConfCanView', 'lookAtConfCanView failed conf')
                return None
            # Find path from lookConf to some conf that does not
            # collide with viewCone.  The last conf in the path will
            # be the new lookConf.
            path = canView(pbs, prob, lookConf, hand, shape,
                           shapeShadow=shapeShadow, findPath=findPath)
            if not path:
                tr('lookAtConfCanView', 'lookAtConfCanView failed path')
                return None
            lookConf = path[-1]
    return lookConf

## lookHandGen
## obj, hand, graspFace, grasp, graspVar, graspDelta and gives a conf

# !! NEEDS UPDATING

# Preconditions (for R1):

# 1. CanSeeFrom() - make a world from the goalConds and CanSeeFrom
# should be true.

# 2. Conf() - if there is Conf in goalConds, then fail.  If there's a
# baseConf in goalConds, then we have to use that base.

# Returns lookConf
def lookHandGen(args, goalConds, bState, outBindings):
    (obj, hand, graspFace, grasp, graspV, graspDelta, prob) = args
    pbs = bState.pbs.copy()
    world = pbs.getWorld()
    if obj == 'none':
        graspB = None
    else:
        graspB = ObjGraspB(obj, world.getGraspDesc(obj), graspFace, None,
                           PoseD(grasp, graspV), delta=graspDelta)
    for ans, viol in lookHandGenTop((obj, hand, graspB, prob),
                                    goalConds, pbs, outBindings):
        yield ans

def lookHandGenTop(args, goalConds, pbs, outBindings):
    def objInHand(conf, hand):
        if (conf, hand) not in handObj:
            attached = shWorld.attached
            if not attached[hand]:
                attached = attached.copy()
                tool = conf.robot.toolOffsetX[hand]
                attached[hand] = Box(0.1,0.05,0.1, None, name='virtualObject').applyLoc(tool)
            _, attachedParts = conf.placementAux(attached, getShapes=[])
            handObj[(conf, hand)] = attachedParts[hand]
        return handObj[(conf, hand)]

    def testFn(c):
        if c not in placements:
            placements[c] = c.placement()
        ans = visible(shWorld, c, objInHand(c, hand),
                       [placements[c]]+obst, prob, moveHead=True)[0]
        return ans
    
    (obj, hand, graspB, prob) = args
    tag = 'lookHandGen'
    placements = {}
    handObj = {}
    tr(tag, '(%s) h=%s'%(obj, glob.inHeuristic))
    newBS = pbs.copy()
    newBS = newBS.updateFromGoalPoses(goalConds)
    newBS.updateHeldBel(graspB, hand)
    shWorld = newBS.getShadowWorld(prob)
    if glob.inHeuristic:
        lookConf = lookAtConf(newBS.conf, objInHand(newBS.conf, hand))
        if lookConf:
            tr(tag, ('->', lookConf))
            yield (lookConf,), Violations()
        return
    if goalConds and getConf(goalConds, None):
        tr(tag, '=> conf is specified, failing')
        return
    rm = newBS.getRoadMap()
    obst = [s for s in shWorld.getNonShadowShapes() if s.name() != obj ]
    lookConfGen = potentialLookHandConfGen(newBS, prob, hand)
    for ans in rm.confReachViolGen(lookConfGen, newBS, prob,
                                   startConf = newBS.conf,
                                   testFn = testFn):
        viol, cost, path = ans
        tr(tag, '(%s) viol=%s'%(obj, viol.weight() if viol else None))
        if not path:
            tr(tag, 'Failed to find a path to look conf.')
            continue
        lookConf = path[-1]
        tr(tag, ('-> cyan', lookConf.conf),
           draw=[(pbs, prob, 'W'),
                 (lookConf, 'W', 'cyan', shWorld.attached)],
           snap=['W'])
        yield (lookConf,), viol

# Generates (obst, pose, support, variance, delta)
# Uses goalConds to find taboo regions
def moveOut(pbs, prob, obst, delta, goalConds):
    tr('moveOut', 'obst=%s'%obst)
    domainPlaceVar = tuple([placeVarIncreaseFactor * x \
                            for x in pbs.domainProbs.obsVarTuple])
    if not isinstance(obst, str):
        obst = obst.name()
    for ans in placeInGenAway((obst, delta, prob), goalConds, pbs):
        ans = ans.copy()
        ans.var = domainPlaceVar; ans.delta = delta
        yield ans

# Preconditions (for R1):

# 1. CanReach(...) - new Pose fluent should not make the canReach
# infeasible (use fluent as taboo).
# new Pose fluent should not already be in conditions (any Pose for this obj).

# 2. Pose(obj) - new Pose has to be consistent with the goal (ok to
# reduce variance wrt goal but not cond). if Pose(obj) in goalConds,
# can only reduce variance.

'''
# returns
# ['Occ', 'Pose', 'PoseFace', 'PoseVar', 'PoseDelta']
# obj, pose, face, var, delta
class CanReachGen(Function):
    def fun(self, args, goalConds, bState):
        (conf, fcp, prob, cond) = args
        pbs = bState.pbs.copy()
        # Don't make this infeasible
        goalFluent = Bd([CanReachHome([conf, fcp, cond]), True, prob], True)
        goalConds = goalConds + [goalFluent]
        # Set up PBS
        newBS = pbs.copy()
        newBS = newBS.updateFromGoalPoses(goalConds)
        newBS = newBS.updateFromGoalPoses(cond, permShadows=True)
        shWorld = newBS.getShadowWorld(prob)
        tr('canReachGen', draw=[(newBS, prob, 'W'),
                     (conf, 'W', 'pink', shWorld.attached)], snap=['W'])
        tr('canReachGen', zip(('conf', 'fcp', 'prob', 'cond'),args))
        # Call
        def violFn(pbs):
            path, viol = canReachHome(pbs, conf, prob, Violations())
            return viol
        lookVar = tuple([lookVarIncreaseFactor * x \
                        for x in pbs.domainProbs.obsVarTuple])
        for ans in canXGenTop(violFn, (cond, prob, lookVar),
                                    goalConds, newBS, 'canReachGen'):
            tr('canReachGen', ('->', ans))
            yield ans.canXGenTuple()
            tr('canReachGen', 'exhausted')

# Preconditions (for R1):

# 1. CanPickPlace(...) - new Pose fluent should not make the
# canPickPlace infeasible.  new Pose fluent should not already be in
# conditions.

# 2. Pose(obj) - new Pose has to be consistent with the goal (ok to
# reduce variance wrt goal but not cond)

# LPK!! More efficient if we notice right away that we cannot ask to
# change the pose of an object that is in the hand in goalConds
class CanPickPlaceGen(Function):
    #@staticmethod
    def fun(self, args, goalConds, bState):
        (preconf, ppconf, hand, obj, pose, realPoseVar, poseDelta, poseFace,
         graspFace, graspMu, graspVar, graspDelta, op, prob, cond) = args
        pbs = bState.pbs.copy()
        # Don't make this infeasible
        cppFluent = Bd([CanPickPlace([preconf, ppconf, hand, obj, pose,
                                      realPoseVar, poseDelta, poseFace,
                                      graspFace, graspMu, graspVar, graspDelta,
                                      op, cond]), True, prob], True)
        poseFluent = B([Pose([obj, poseFace]), pose, realPoseVar, poseDelta, prob],
                        True)
        # Augment with conditions to maintain
        goalConds = goalConds + [cppFluent, poseFluent]
        
        world = pbs.getWorld()
        lookVar = tuple([lookVarIncreaseFactor * x \
                                for x in pbs.domainProbs.obsVarTuple])
        graspB = ObjGraspB(obj, world.getGraspDesc(obj), graspFace, poseFace,
                           PoseD(graspMu, graspVar), delta= graspDelta)
        placeB = ObjPlaceB(obj, world.getFaceFrames(obj), poseFace,
                           PoseD(pose, realPoseVar), delta=poseDelta)
        # Set up PBS
        newBS = pbs.copy()
        newBS = newBS.updateFromGoalPoses(goalConds)
        newBS = newBS.updateFromGoalPoses(cond, permShadows=True)
        # Debug
        shWorld = newBS.getShadowWorld(prob)
        tr('canPickPlaceGen',
           draw=[(newBS, prob, 'W'),
                 (preconf, 'W', 'blue', shWorld.attached),
                 (ppconf, 'W', 'pink', shWorld.attached),
                 (placeB.shape(shWorld), 'W', 'pink')],
           snap=['W'])
        tr('canPickPlaceGen',
           zip(('preconf', 'ppconf', 'hand', 'obj', 'pose', 'realPoseVar', 'poseDelta', 'poseFace',
                'graspFace', 'graspMu', 'graspVar', 'graspDelta', 'prob', 'cond', 'op'),
               args))
        # Initial test
        def violFn(pbs):
            v, r = canPickPlaceTest(pbs, preconf, ppconf, hand,
                                    graspB, placeB, prob, op=op)
            return v
        for ans in canXGenTop(violFn, (cond, prob, lookVar),
                              goalConds, newBS, 'canPickPlaceGen'):
            tr('canPickPlaceGen', ('->', ans))
            yield ans.canXGenTuple()
        tr('canPickPlaceGen', 'exhausted')
'''        

def canXGenTop(violFn, args, goalConds, newBS, tag):
    (cond, prob, lookVar) = args
    tr(tag, 'h=%s'%glob.inHeuristic)
    # Initial test
    viol = violFn(newBS)
    tr(tag, ('viol', viol),
       draw=[(newBS, prob, 'W')], snap=['W'])
    if not viol:                  # hopeless
        if tag == 'canPickPlaceGen':
            glob.debugOn.append('canPickPlaceTest')
            violFn(newBS)
            glob.debugOn.remove('canPickPlaceTest')
        tr(tag, 'Impossible dream')
        return
    if viol.empty():
        tr(tag, '=> No obstacles or shadows; returning')
        return

    #objBMinVarGrasp = tuple([x**2/2*x for x in newBS.domainProbs.obsVarTuple])
    
    # LPK Make this a little bigger?
    objBMinVarGrasp = tuple([x/2 for x in newBS.domainProbs.obsVarTuple])
    objBMinVarStatic = tuple([x**2 for x in newBS.domainProbs.odoError])
    objBMinProb = 0.95
    # The irreducible shadow
    objBMinDelta = newBS.domainProbs.shadowDelta
    
    lookDelta = objBMinDelta
    moveDelta = newBS.domainProbs.placeDelta
    shWorld = newBS.getShadowWorld(prob)
    fixed = shWorld.fixedObjects
    # Try to fix one of the violations if any...
    obstacles = [o.name() for o in viol.allObstacles() \
                 if o.name() not in fixed]
    shadows = [sh.name() for sh in viol.allShadows() \
               if not sh.name() in fixed]
    if not (obstacles or shadows):
        tr(tag, '=> No movable obstacles or shadows to fix')
        return       # nothing available
    if obstacles:
        obst = obstacles[0]
        for ans in moveOut(newBS, prob, obst, moveDelta, goalConds):
            yield ans
        return
    if shadows:
        shadowName = shadows[0]
        obst = objectName(shadowName)
        graspable = len(newBS.getWorld().getFraspDesc(obst)) > 0
        objBMinVar = objBMinVarGrasp if graspable else objBMinVarStatic
        placeB = newBS.getPlaceB(obst)
        tr(tag, '=> reduce shadow %s (in red):'%obst,
           draw=[(newBS, prob, 'W'),
                 (placeB.shadow(newBS.getShadowWorld(prob)), 'W', 'red')],
           snap=['W'])
        ans = PPResponse(placeB, None, None, None, None, objBMinVar, lookDelta)
        yield ans
        # Either reducing the shadow is not enough or we failed and
        # need to move the object (if it's movable).
        if obst not in fixed:
            for ans in moveOut(newBS, prob, obst, moveDelta, goalConds):
                yield ans
    tr(tag, '=> Out of remedies')

# Preconditions (for R1):

# 1. CanSeeFrom(...) - new Pose fluent should not make the CanSeeFrom
# infeasible.  new Pose fluent should not already be in conditions.

# 2. Pose(obj) - new Pose has to be consistent with the goal (ok to
# reduce variance wrt goal but not cond)

# returns
# ['Occ', 'PoseFace', 'Pose', 'PoseVar', 'PoseDelta']
def canSeeGen(args, goalConds, bState, outBindings):
    (obj, pose, support, objV, objDelta, lookConf, lookDelta, prob) = args
    pbs = bState.pbs.copy()
    world = pbs.getWorld()

    if pose == '*':
        poseD = pbs.getPlaceB(obj).poseD
    else: 
        poseD = PoseD(pose, objV)
    if isVar(support) or support == '*':
        support = pbs.getPlaceB(obj).support.mode()
    if objDelta == '*':
        objDelta = lookDelta
    
    placeB = ObjPlaceB(obj, world.getFaceFrames(obj), support,
                       poseD,
                       # Pretend that the object has bigger delta
                       delta=tuple([o+l for (o,l) in zip(objDelta, lookDelta)]))

    for ans in canSeeGenTop((lookConf, placeB, [], prob),
                            goalConds, pbs, outBindings):
        yield ans

def canSeeGenTop(args, goalConds, pbs, outBindings):
    (conf, placeB, cond, prob) = args
    obj = placeB.obj
    tr('canSeeGen', '(%s) h=%s'%(obj, glob.inHeuristic))
    tr('canSeeGen', zip(('conf', 'placeB', 'cond', 'prob'), args))

    newBS = pbs.copy()
    newBS = newBS.updateFromGoalPoses(goalConds)
    newBS = newBS.updateFromGoalPoses(cond, permShadows=True)
    newBS = newBS.updatePermObjPose(placeB)

    shWorld = newBS.getShadowWorld(prob)
    shape = shWorld.objectShapes[placeB.obj]
    obst = [s for s in shWorld.getNonShadowShapes() \
            if s.name() != placeB.obj ]
    p, occluders = visible(shWorld, conf, shape, obst, prob, moveHead=True)
    occluders = [oc for oc in occluders if oc not in newBS.fixObjBs]
    if not occluders:
        tr('canSeeGen', '=> no occluders')
        return
    obst = occluders[0] # !! just pick one
    moveDelta = pbs.domainProbs.placeDelta
    for ans in moveOut(newBS, prob, obst, moveDelta, goalConds):
        yield ans 

def groupByCost(entries):
    classes = []
    values = []
    sentries = sorted(entries)
    for (c, e) in sentries:
        if not(values) or values[-1] != c:
            classes.append([e])
            values.append(c)
        else:
            classes[-1].append(e)
    return classes, values


