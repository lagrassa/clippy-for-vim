import time
import pdb
from fbch import Function
from dist import DeltaDist
from pr2Util import supportFaceIndex, shadowWidths, trArgs, inside, graspable, objectName
from pr2PlanBel import getGoalConf, getGoalPoseBels
from shapes import Box
from pr2Fluents import baseConfWithin
from pr2GenAux import *
from planUtil import PPResponse, ObjPlaceB, PoseD
from pr2Push import pushInRegionGenGen

'''
class Candidate(object):
    # pre, during, post 
    pB0 = pb1 = pb2 = None
    # pre, during, post
    c0 = c1 = c2 = None
    gB = None
    viol = None
    hand = None
    # pre, post
    var0 = var1 = None
    delta = None

# The structure of a genrator
class Generator(Function):
    # Returns pargs, cpbs (processed args and conditioned pbs)
    # Test that preconds are not inherently inconsistent with goal
    def processArgs(self, args, goalConds, pbs):
        return None, None
    # Yields instances of Candidate
    def candidateGen(self, pargs, cpbs):
        return
    def scoreCandidate(self, candidate, pargs, cpbs):
        return 0
    # Test that candidate does not make conditional fluents infeasible
    def testCandidate(self, candidate, pargs, cpbs):
        return True
    def fun(self, args, goalConds, bState):
        pass

'''

def fixed(value):
    return value and isinstance(value, tuple) and len(value) == 2 and value[0]

#  How many candidates to generate at a time...  Larger numbers will
#  generally lead to better solutions but take longer.
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
        (obj, hand, face, grasp) = args
        assert obj != None and obj != 'none'
        tr(self.tag, '(%s,%s) h=%s'%(obj,hand,glob.inHeuristic))
        cpbs = pbs.conditioned(goalConds, []) # condition on goalConds
        if fixed(cpbs.conf):
            tr(tag, '=> conf fixed in goal, failing')
            return
        prob = 0.75
        placeB = cpbs.getPlaceB(obj)
        shWorld = cpbs.getShadowWorld(prob)
        if obj == cpbs.getHeld(hand):
            ans = PPResponse(placeB, cpbs.getGraspB(hand), None, None, None, hand)
            tr(tag, 'inHand:'+ str(ans))
            yield ans.easyGraspTuple()
            return
        if obj == cpbs.getHeld(otherHand(hand)):
            tr(tag, 'no easy grasp with this hand, failing')
            return
        graspB = ObjGraspB(obj, cpbs.getWorld().getGraspDesc(obj), None,
                           placeB.support.mode(),
                           PoseD(None, graspVar), delta=graspDelta)
        cache = cpbs.genCache(tag)
        key = (cpbs, placeB, graspB, hand, prob, face, grasp)
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
                            easyGraspGenAux(cpbs, placeB, graspB, hand, prob,
                                            face, grasp))
            cache[key] = memo
            cached = ''
        for ans in memo:
            tr(tag, str(ans))
            yield ans.easyGraspTuple()
        tr(tag, '(%s,%s)='%(obj, hand)+'=> out of values')
        return

def easyGraspGenAux(cpbs, placeB, graspB, hand, prob, oldFace, oldGrasp):
    tag = 'easyGraspGen'

    def graspApproachConfGen(firstConf):
        if firstConf:
            yield firstConf
        for c, ca, _ in graspConfGen:
            approached[ca] = c
            yield ca

    def pickable(ca, c, pB, gB):
        viol, reason = canPickPlaceTest(cpbs, ca, c, hand, gB, pB, prob, op='pick')
        return viol

    if debug(tag): print 'easyGraspGenAux'
    obj = placeB.obj
    approached = {}
    for gB in graspGen(cpbs, obj, graspB):
        if gB.grasp.mode() == oldFace and gB.poseD.modeTuple() == oldGrasp:
            tr(tag, 'Rejected %s because same'%gB)
            continue
        tr(tag, 'considering grasp=%s'%gB)

        # TODO: is there a middle road between this and full regrasp?
        #yield PPResponse(placeB, gB, None, None, None, hand)
        
        graspConfGen = potentialGraspConfGen(cpbs, placeB, gB, None, hand, None, prob)
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
        cpbs = bState.pbs.conditioned(goalConds, [])
        world = cpbs.getWorld()
        graspB = ObjGraspB(obj, world.getGraspDesc(obj), graspFace, None,
                       PoseD(hu.Pose(*graspPose), graspV), delta=graspDelta)
        placeB = ObjPlaceB(obj, world.getFaceFrames(obj), None,
                       PoseD(None,  objV), delta=objDelta)
        for ans in pickGenTop((obj, graspB, placeB, hand, prob,), cpbs):
            yield ans.pickTuple()

def pickGenTop(args, pbs, onlyCurrent = False):
    (obj, graspB, placeB, hand, prob) = args

    if traceGen:
        print '***', 'pickGenAux', obj, placeB.poseD.mode(), graspB.grasp.mode(), hand

    tag = 'pickGen'
    graspDelta = pbs.domainProbs.pickStdev
    tr(tag, '(%s,%s,%d) h=%s'%(obj,hand,graspB.grasp.mode(), glob.inHeuristic))
    trArgs(tag, ('obj', 'graspB', 'placeB', 'hand', 'prob'), args, pbs)
    if obj == 'none':                   # can't pick up 'none'
        tr(tag, '=> cannot pick up none, failing')
        return
    if fixed(pbs.conf):
        tr(tag, '=> conf fixed in goal, failing')
        return
    if placeB.poseD.mode() is not None: # specified by, e.g. lookGen
        pose = placeB.poseD.mode()
        sup =  placeB.support.mode()
        pbs.resetPlaceB(placeB)
        tr(tag, 'Setting placeB, support=%s, pose=%s'%(sup, pose.xyztTuple()))
    conf = None
    confAppr = None
    pose, support = getPoseAndSupport(tag, obj, pbs, prob)
    # Update placeB
    # Use 0 variance !!
    pickVar = 4*(0.0,)
    placeB = ObjPlaceB(obj, placeB.faceFrames, DeltaDist(support),
                       PoseD(pose, pickVar), placeB.delta)
    tr(tag, 'target placeB=%s'%placeB)

    shWorld = pbs.getShadowWorld(prob)
    tr('pickGen', 'Goal conditions', draw=[(pbs, prob, 'W')], snap=['W'])
    gen = pickGenAux(pbs, obj, confAppr, conf, placeB, graspB, hand, prob,
                     onlyCurrent=onlyCurrent)
    for ans in gen:
        tr(tag, str(ans),
           draw=[(pbs, prob, 'W'),
                 (ans.c, 'W', 'orange', shWorld.attached)],
           snap=['W'])
        yield ans

def pickGenAux(pbs, obj, confAppr, conf, placeB, graspB, hand, prob,
               onlyCurrent = False):
    def pickable(ca, c, pB, gB):
        return canPickPlaceTest(pbs, ca, c, hand, gB, pB, prob, op='pick')

    def checkInfeasible(conf):
        newBS = pbs.copy()
        newBS.updateConf(conf)
        newBS.updateHeldBel(graspB, hand)
        viol = newBS.confViolations(conf, prob)
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

    if traceGen:
        print ' **', 'pickGenAux', placeB.poseD.mode(), graspB.grasp.mode(), hand

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
    failureReasons = []
    base = pbs.getBase()
    if placeB.poseD.mode() is not None: # otherwise go to regrasp
        if not base:
            # Try current conf
            (x,y,th) = pbs.getConf()['pr2Base']
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
                    print 'pickGen: Cannot find grasp conf for current pose of ' + obj
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
        debugMsg(tag, 'Pose is not specified, need to place')

    if onlyCurrent:
        tr(tag, 'onlyCurrent: out of values')
        return
        
    # Try a regrasp... that is place the object somewhere else where
    # it can be grasped.
    if glob.inHeuristic:                # don't do regrasping in heuristic
        return
    
    # Try a regrasp... that is place the object somewhere else where it can be grasped.

    if failureReasons and all(['visibility' in reason for reason in failureReasons]):
        tr(tag, 'There were valid targets that failed due to visibility')
        return
    
    tr(tag, 'Calling for regrasping... h=%s'%glob.inHeuristic)

    # !! Needs to look for plausible regions...
    regShapes = [shWorld.regionShapes[region] for region in pbs.awayRegions()]
    plGen = placeInGenTop((obj, regShapes, graspB, placeB, None, prob),
                          pbs, regrasp = True)
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
        pbs = bState.pbs.conditioned(goalConds, [])
        for ans in placeGenGen(args, pbs):
            tr('placeGen', str(ans))
            yield ans.placeTuple()

# Either hand or poses will be specified, but generally not both.  They will never both be unspecified.
def placeGenGen(args, pbs):
    (obj, hand, poses, support, objV, graspV, objDelta, graspDelta, confDelta,
     prob) = args
    if traceGen:
        print '***', 'placeGenGen', obj, hand
    tag = 'placeGen'
    if fixed(pbs.conf):
        tr(tag, '=> conf fixed in goal, failing')
        return
    base = pbs.getBase()
    tr(tag, 'obj=%s, base=%s'%(obj, base))
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
        newBS = pbs.copy().excludeObjs([obj])
        # v is viol
        for ans in placeInGenAway((obj, objDelta, prob), newBS):
            yield ans
        return

    if isinstance(poses, tuple):
        placeVar = pbs.domainProbs.placeVar # instead of objV
        placeB = ObjPlaceB(obj, world.getFaceFrames(obj), support,
                           PoseD(poses, placeVar), delta=objDelta)
        placeBs = frozenset([placeB])
    else:
        raw_input('placeGenGen - poses is not a tuple')

    graspB = ObjGraspB(obj, world.getGraspDesc(obj), None, None,
                       PoseD(None, graspV), delta=graspDelta)
        
    # Figure out whether one hand or the other is required;  if not, do round robin
    leftGen = placeGenTop((obj, graspB, placeBs, 'left', base, prob), pbs)
    rightGen = placeGenTop((obj, graspB, placeBs, 'right', base, prob), pbs)
    
    for ans in chooseHandGen(pbs, obj, hand, leftGen, rightGen):
        yield ans

placeGenCacheStats = [0, 0]
placeGenCache = {}

# returns values for (?graspPose, ?graspFace, ?conf, ?confAppr)
def placeGenTop(args, pbs, regrasp=False, away=False):
    (obj, graspB, placeBs, hand, base, prob) = args

    if traceGen:
        print '***', 'placeGenTop', graspB.grasp.mode(), hand

    startTime = time.clock()
    tag = 'placeGen'
    tr(tag, '(%s,%s) h=%s'%(obj,hand, glob.inHeuristic))
    trArgs(tag, ('obj', 'graspB', 'placeBs', 'hand', 'prob'), args, pbs)
    if obj == 'none' or not placeBs:
        tr(tag, '=> obj is none or no poses, failing')
        return
    if fixed(pbs.conf):
        tr(tag, '=> conf fixed in goal, failing')
        return
    if fixed(pbs.held[hand]):
        tr(tag, '=> Hand=%s is already Holding, failing'%hand)
        return
    conf = None
    confAppr = None
    # Make sure we exclude obj
    newBS = pbs.copy().excludeObjs([obj])
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
           regrasp, away)
    val = placeGenCache.get(key, None)
    placeGenCacheStats[0] += 1
    if val is not None:
        placeGenCacheStats[1] += 1
        # Will restart the generator when it is retrieved
        memo = val.copy()
        tr(tag, 'Found generator in cache')
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
        tr(tag, 'Created new generator')
    for ans in memo:
        tr(tag, str(ans) +' (t=%s)'%(time.clock()-startTime))
        yield ans

def placeGenAux(pbs, obj, confAppr, conf, placeBs, graspB, hand, base, prob,
                regrasp=False, pbsOrig=None):
    def placeable(ca, c, quick=False):
        (pB, gB) = context[ca]
        return canPickPlaceTest(pbs, ca, c, hand, gB, pB, prob,
                                op='place', quick=quick)
    def currentGrasp(gB):
        if obj == pbsOrig.getHeld(hand):
            currGraspB = pbsOrig.getGraspB(hand)
            return  (gB.grasp.mode() == currGraspB.grasp.mode()) and \
                   gB.poseD.mode().near(currGraspB.poseD.mode(), .01, .01)
        return False

    def checkRegraspable(pB):
        if pB in regraspablePB:
            return regraspablePB[pB]
        regraspablePB[pB] = 5.
        curGrasp = None
        for gBO in gBOther:
            if currentGrasp(gBO):
                curGrasp = gBO
                break
        if curGrasp:
            c = next(potentialGraspConfGen(pbs, pB, gBO, conf, hand, base, prob, nMax=1),
                     (None, None, None))[0]
            if c:
                tr(tag, 'Regraspable for current grasp')
                regraspablePB[pB] = 0.
                return True
            else:
                tr(tag, 'Not regraspable for current grasp')
                regraspablePB[pB] = 5.
                # Returning False means that we insist that the
                # regrasp happen in one step.
                # return False
        for gBO in gBOther:
            if gBO == curGrasp: continue
            c = next(potentialGraspConfGen(pbs, pB, gBO, conf, hand, base, prob, nMax=1),
                     (None, None, None))[0]
            if c:
                tr(tag,
                   'Regraspable', pB.poseD.mode(), gBO.grasp.mode(),
                   draw=[(c, 'W', 'green')], snap=['W'])
                regraspablePB[pB] = 2.
                return True

    def checkOrigGrasp(gB):
        # 0 if currently true
        # 1 if could be used on object's current position
        # 2 otherwise
        
        # Prefer current grasp
        if currentGrasp(gB):
            tr(tag, 'current grasp is a match',
               ('curr', pbsOrig.graspB[hand]), ('desired', gB))
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
                    if not checkRegraspable(pB):
                        continue
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
            return regraspablePB[pB]
        else:
            # if debug('placeGen'): print 'unknown pB, cost = 1'
            return 1.

    if traceGen:
        print ' **', 'placeGenAux', graspB.grasp.mode(), hand

    tag = 'placeGen'
    approached = {}
    context = {}
    regraspablePB = {}
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

    tr(tag, 'Top grasps', [[g.grasp.mode() for g in gC] for gC in gClasses], 'costs', gCosts)

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
        pbs = bState.pbs.conditioned(goalConds, [])
        for ans in lookInRegionGenGen(args, pbs, away = False):
            yield ans
        for ans in roundrobin(placeInRegionGenGen(args, pbs, away = False),
                              pushInRegionGenGen(args, pbs, away = False)):
            if ans:
                yield ans.poseInTuple()

def lookInRegionGenGen(args, pbs, away = False):
    (obj, region, var, delta, prob) = args
    tag = 'lookInGen'
    world = pbs.getWorld()

    tr(tag, args)

    # Get the regions
    regions = getRegions(region)
    shWorld = pbs.getShadowWorld(prob)
    regShapes = [shWorld.regionShapes[region] if isinstance(region, str) else region\
                 for region in regions]
    tr(tag, 'Target region in purple',
       draw=[(pbs, prob, 'W')] + [(rs, 'W', 'purple') for rs in regShapes],
       snap=['W'])
    pose, support = getPoseAndSupport(tag, obj, pbs, prob)

    # Check if object pose is specified
    if fixed(pbs.objectBs.get(obj, None)):
        pB = pbs.objectBs[obj]
        pose = pB.poseD.mode()
        var = pB.poseD.var

    lookVar = pbs.domainProbs.objBMinVar(objectName(obj))
    lookDelta = pbs.domainProbs.shadowDelta
    placeB = ObjPlaceB(obj, world.getFaceFrames(obj), support,
                       PoseD(pose, lookVar), delta=lookDelta)
    if pose and any(inside(placeB.makeShadow(pbs, prob), regShape, strict=True) \
           for regShape in regShapes):
        ans = (pose.xyztTuple(), support)
        tr(tag, '=>', ans)
        yield ans
    tr(tag, '=> Look will not achieve In')

def placeInRegionGenGen(args, pbs, away = False):
    (obj, region, var, delta, prob) = args

    if traceGen:
        print '***', 'placeInRegionGenGen'
    tag = 'placeInGen'    
    tr(tag, args)
    world = pbs.getWorld()

    # If there are no grasps, just fail
    if not graspable(obj):
        tr(tag, obj, 'not graspable')
        return

    # Get the regions
    regions = getRegions(region)
    shWorld = pbs.getShadowWorld(prob)
    regShapes = [shWorld.regionShapes[region] for region in regions]
    tr(tag, 'Target region in purple',
       draw=[(pbs, prob, 'W')] + [(rs, 'W', 'purple') for rs in regShapes],
       snap=['W'])
    pose, support = getPoseAndSupport(tag, obj, pbs, prob)

    graspV = pbs.domainProbs.maxGraspVar
    graspDelta = pbs.domainProbs.graspDelta
    graspB = ObjGraspB(obj, world.getGraspDesc(obj), None, None,
                       PoseD(None, graspV), delta=graspDelta)

    # Check if object pose is specified
    if fixed(pbs.objectBs.get(obj, None)):
        pB = poseBels[obj]
        shw = shadowWidths(pB.poseD.var, pB.delta, prob)
        shwMin = shadowWidths(graspV, graspDelta, prob)
        if any(w > mw for (w, mw) in zip(shw, shwMin)):
            args = (obj, None, pB.poseD.modeTuple(),
                    support, var, graspV,
                    delta, graspDelta, None, prob)
            gen = placeGenGen(args, pbs)
            for ans in gen:
                regions = [x.name() for x in regShapes]
                tr(tag, str(ans), 'regions=%s'%regions,
                   draw=[(pbs, prob, 'W')] + [(rs, 'W', 'purple') for rs in regShapes],
                   snap=['W'])
                tr(tag, '-> pose already specified', ans)
                yield ans
            tr(tag, obj, 'pose already specified')
            return
        else:
            # If pose is specified and variance is small, return
            tr(tag, obj, 'pose already specified, small variance')
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
        tr(tag, '-> dropping obj', ans)
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
                          pbs, away = away)
    for ans in gen:
        tr(tag, '-> ', ans)
        yield ans
    tr(tag, 'XXX', ans)
        

placeVarIncreaseFactor = 3 # was 2
lookVarIncreaseFactor = 2

def dropIn(pbs, prob, obj, regShapes):
    hand = None
    for h in ('left', 'right'):
        minConf = minimalConf(pbs.getConf(), h)
        if minConf in PPRCache:
            hand = h
            break
    if not hand: return
    shWorld = pbs.getShadowWorld(prob)
    ppr = PPRCache[minConf]
    assert ppr.hand == hand
    if obj == pbs.getHeld(hand):
        robShape, attachedPartsDict = pbs.getConf().placementAux(attached=shWorld.attached)
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
    minConf = minimalConf(pbs.getConf(), hand)
    if minConf in PPRCache:
        ppr = PPRCache[minConf]
    else:
        return
    assert ppr.hand == hand
    shWorld = pbs.getShadowWorld(prob)
    if obj == pbs.getHeld(hand):
        xt = ppr.pB.poseD.mode().xyztTuple()
        cxt = placeB.poseD.mode().xyztTuple()
        if max([abs(a-b) for (a,b) in zip(xt,cxt)]) < 0.001:
            if canPickPlaceTest(pbs, ppr.ca, ppr.c, ppr.hand, ppr.gB, ppr.pB, prob,
                            op='place')[0]:
                return ppr

def placeInGenAway(args, pbs):
    # !! Should search over regions and hands
    (obj, delta, prob) = args

    if traceGen:
        print '***', 'placeInGenAway'

    if not pbs.awayRegions():
        raw_input('Need some awayRegions')
        return
    tr('placeInGenAway', zip(('obj', 'delta', 'prob'), args),
       draw=[(pbs, prob, 'W')], snap=['W'])
    targetPlaceVar = tuple([placeVarIncreaseFactor * x \
                            for x in pbs.domainProbs.obsVarTuple])
    for ans in placeInRegionGenGen((obj, pbs.awayRegions(),
                                    targetPlaceVar, delta, prob),
                                   pbs, away=True):
        yield ans

placeInGenMaxPoses  = 500

def placeInGenTop(args, pbs,
                  regrasp=False, away = False):
    (obj, regShapes, graspB, placeB, base, prob) = args
    tag = 'placeInGen'

    if traceGen:
        print '***', 'placeInGenTop', placeB.poseD.mode(), graspB.grasp.mode()

    regions = [x.name() for x in regShapes]
    tr(tag, '(%s,%s) h=%s'%(obj,regions, glob.inHeuristic))
    tr(tag, 
       zip(('obj', 'regShapes', 'graspB', 'placeB', 'prob'), args))
    if obj == 'none' or not regShapes:
        # Nothing to do
        tr(tag, '=> object is none or no regions, failing')
        return
    if fixed(pbs.conf):
        tr(tag, '=> conf fixed in goal, failing')
        return
    conf = confAppr = None
    nPoses = placeInGenMaxPoses
    poseGenLeft = Memoizer('regionPosesLeft',
                           potentialRegionPoseGen(pbs, obj, placeB, graspB, prob, regShapes,
                                                  'left', base, maxPoses=nPoses))
    poseGenRight = Memoizer('regionPosesRight',
                            potentialRegionPoseGen(pbs, obj, placeB, graspB, prob, regShapes,
                                                   'right', base, maxPoses=nPoses))
    # note the use of PB...
    leftGen = placeInGenAux(pbs, poseGenLeft, confAppr,
                            conf, placeB, graspB, 'left', base, prob,
                            regrasp=regrasp, away=away)
    rightGen = placeInGenAux(pbs, poseGenRight, confAppr,
                             conf, placeB, graspB, 'right', base, prob,
                             regrasp=regrasp, away=away)
    # Figure out whether one hand or the other is required;  if not, do round robin
    mainGen = chooseHandGen(pbs, obj, None, leftGen, rightGen)

    # Picks among possible target poses and then try to place it in region
    for ans in mainGen:
        shWorld = pbs.getShadowWorld(prob)
        tr(tag, str(ans),
           draw=[(ans.c, 'W', 'green', shWorld.attached)] + \
           [(rs, 'W', 'purple') for rs in regShapes],
           snap=['W'])
        yield ans

# Don't try to place all objects at once
def placeInGenAux(pbs, poseGen, confAppr, conf, placeB, graspB,
                  hand, base, prob, regrasp=False, away=False):

    def placeBGen():
        for pose in poseGen.copy():
            yield placeB.modifyPoseD(mu=pose)

    if traceGen:
        print '***', 'placeInGenAux', placeB.poseD.mode(), graspB.grasp.mode(), hand

    tries = 0
    shWorld = pbs.getShadowWorld(prob)
    gen = Memoizer('placeBGen_placeInGenAux1', placeBGen())
    for ans in placeGenTop((graspB.obj, graspB, gen, hand, base, prob),
                           pbs, regrasp=regrasp, away=away):
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
        pbs = bState.pbs.conditioned(goalConds, [])
        world = pbs.getWorld()
        base = pbs.getBase()
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
                                  support, poseD_before, delta = objDelta)
        placeB_after = ObjPlaceB(obj, world.getFaceFrames(obj),
                                 support, poseD_after, delta = objDelta)
        # ans = (lookConf,)
        for ans, viol in lookGenTop((obj, placeB_before, placeB_after,
                                     lookDelta, base, prob),
                                    pbs):
            yield ans

# Returns (lookConf,), viol
# The lookConf should be s.t.
# - has no collisions with beforeShadow or any other permanent objects in the pbs
# - can move to targetConf in the after world
# - has the same base, if base is specified

def lookGenTop(args, pbs):
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
    if fixed(pbs.conf):
        tr(tag, '=> conf fixed in goal, failing')
        return
    if placeB_before.poseD.mode() is None:
        tr(tag, '=> object is in the hand, failing')
        return    
    # Create planning contexts for before look (large variance)...
    pbs_before = pbs.copy().updatePermObjBel(placeB_before)
    pbs_before.addAvoidShadow([obj])
    shWorld_before = pbs_before.getShadowWorld(prob)
    # ... and after look (lower variance)
    pbs_after = pbs.copy().updatePermObjBel(placeB_after)
    pbs_after.addAvoidShadow([obj])
    shWorld_after = pbs_after.getShadowWorld(prob)
    # Some temp values that are independent of before/after
    attached = shWorld_before.attached  # attached
    if any(attached.values()):
        tr(tag, 'attached=%s'%attached)
    shName = shadowName(obj)
    world = pbs_before.getWorld()
    # mode shape, ignoring variance and delta, use as target shape.
    shapeForLook = placeB_before.shape(shWorld_before)
    shapeShadow = shWorld_before.objectShapes[shName]
    # Handle case where the base is specified.
    if base:
        confAtTarget = pbs.getTargetConf()
        # Use the conf in goalConds to "fill in" the base information
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
            # way of the viewCone and the shapeShadow.
            delta = pbs.domainProbs.moveConfDelta
            if baseConfWithin(pbs.getConf()['pr2Base'], base, delta):
                curLookConf = lookAtConfCanView(pbs_after, prob,
                                                pbs_after.conf,
                                                shapeForLook,
                                                shapeShadow=shapeShadow,
                                                findPath=False)
            else:
                curLookConf = None
            # Note that lookAtConfCanView avois the view cone and
            # shapeShadow in pbs_after, this ensures that this
            # motion can take us back to a safe conf in the after world.
            lookConf = curLookConf or \
                       lookAtConfCanView(pbs_after, prob, confAtTarget,
                                         shapeForLook, shapeShadow=shapeShadow,
                                         findPath=True)
            if lookConf:
                tr(tag, '=> Found a path to look conf with specified base.',
                   ('-> cyan', lookConf.conf),
                   draw=[(pbs_before, prob, 'W'),
                         (lookConf, 'W', 'cyan', attached)],
                   snap=['W'])
                yield (lookConf,), pbs_after.confViolations(lookConf, prob)
            else:
                tr(tag,
                   '=> Failed to find path to look conf with specified base.',
                   ('target conf after look is magenta', confAtTarget.conf),
                   draw=[(pbs_before, prob, 'W'),
                         (confAtTarget, 'W', 'magenta', attached)],
                   snap=['W'])
        return

    # Check if the current conf will work for the look
    curr = pbs_before.conf
    if testFn(curr, shapeForLook, shWorld_before): # visible?
        # move arm out of the way if necessary, use after shadow
        lookConf = lookAtConfCanView(pbs_after, prob, curr,
                                     shapeForLook, shapeShadow=shapeShadow)
        if lookConf:
            tr(tag, '=> Using current conf.',
               draw=[(pbs_before, prob, 'W'),
                     (lookConf, 'W', 'cyan', attached)])
            yield (lookConf,), pbs_after.confViolations(lookConf, prob)

    # If we're looking at graspable objects, prefer a lookConf from
    # which we could pick the object, so that we don't have to move
    # the base.

    # TODO: Generalize this to (pick or push)

    if graspable(obj) and not glob.inHeuristic:
        graspVar = 4*(0.001,)
        graspDelta = 4*(0.001,)
        graspB = ObjGraspB(obj, world.getGraspDesc(obj), None, None,
                           PoseD(None, graspVar), delta=graspDelta)
        # Use pbs (which doesn't declare shadow to be permanent) to
        # generate candidate confs, since they will need to collide
        # with shadow of obj.
        for gB in graspGen(pbs, obj, graspB):
            for hand in ['left', 'right']:
                for ans in pickGenTop((obj, gB, placeB_after, hand, base, prob),
                                      pbs, onlyCurrent=True):
                    # Modify the approach conf so that robot avoids view cone.
                    lookConf = lookAtConfCanView(pbs, prob, ans.ca,
                                                 shapeForLook, shapeShadow=shapeShadow)
                    if not lookConf:
                        tr(tag, 'canView failed')
                        continue
                    # We should be guaranteed that this is true, since
                    # ca was chosen so that the shape is visible.
                    if testFn(lookConf, shapeForLook, shWorld_before):
                        # Find the violations at that lookConf
                        viol = pbs.confViolations(lookConf, prob)
                        vw = viol.weight() if viol else None
                        tr(tag, '(%s) canView cleared viol=%s'%(obj, vw))
                        yield (lookConf,), viol
                    else:
                        assert None, 'shape should be visible, but it is not'
    # Find a lookConf unconstrained by base
    lookConfGen = potentialLookConfGen(pbs_before, prob, shapeForLook, maxLookDist)
    rm = pbs_before.getRoadMap()
    for ans in rm.confReachViolGen(lookConfGen, pbs_before, prob,
                                   testFn = lambda c: testFn(c, shapeForLook, shWorld_before)):
        viol, cost, path = ans
        tr(tag, '(%s) viol=%s'%(obj, viol.weight() if viol else None))
        if not path:
            tr(tag,  'Failed to find a path to look conf.')
            raw_input('Failed to find a path to look conf.')
            continue
        conf = path[-1]                 # lookConf is at the end of the path
        # Modify the look conf so that robot does not block
        lookConf = lookAtConfCanView(pbs_before, prob, conf,
                                     shapeForLook, shapeShadow=shapeShadow)
        if lookConf:
            tr(tag, '(%s) general conf viol=%s'%(obj, viol.weight() if viol else None),
               ('-> cyan', lookConf.conf),
               draw=[(pbs_before, prob, 'W'),
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
def lookHandGen(args, goalConds, bState):
    (obj, hand, graspFace, grasp, graspV, graspDelta, prob) = args
    pbs = bState.pbs.copy().conditioned(goalConds, [])
    world = pbs.getWorld()
    if obj == 'none':
        graspB = None
    else:
        graspB = ObjGraspB(obj, world.getGraspDesc(obj), graspFace, None,
                           PoseD(grasp, graspV), delta=graspDelta)
    for ans, viol in lookHandGenTop((obj, hand, graspB, prob), pbs):
        yield ans

def lookHandGenTop(args, pbs):
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
    newBS = pbs.updateHeldBel(graspB, hand)
    shWorld = newBS.getShadowWorld(prob)
    if glob.inHeuristic:
        lookConf = lookAtConf(newBS.getConf(), objInHand(newBS.conf, hand))
        if lookConf:
            tr(tag, ('->', lookConf))
            yield (lookConf,), Violations()
        return
    if fixed(pbs.conf):
        tr(tag, '=> conf fixed in goal, failing')
        return
    rm = newBS.getRoadMap()
    obst = [s for s in shWorld.getNonShadowShapes() if s.name() != obj ]
    lookConfGen = potentialLookHandConfGen(newBS, prob, hand)
    for ans in rm.confReachViolGen(lookConfGen, newBS, prob,
                                   startConf = newBS.getConf(),
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
def moveOut(pbs, prob, obst, delta):
    tr('moveOut', 'obst=%s'%obst)
    domainPlaceVar = tuple([placeVarIncreaseFactor * x \
                            for x in pbs.domainProbs.obsVarTuple])
    if not isinstance(obst, str):
        obst = obst.name()
    for ans in placeInGenAway((obst, delta, prob), newBS):
        ans = ans.copy()
        ans.var = pbs.domainProbs.objBMinVar(obst)
        ans.delta = delta
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


