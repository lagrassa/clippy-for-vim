import numpy as np
import util
from dist import DeltaDist, varBeforeObs, DDist, probModeMoved, MixtureDist,\
     UniformDist, chiSqFromP, MultivariateGaussianDistribution
from fbch import Function, getMatchingFluents, Operator, simplifyCond
from miscUtil import isVar, prettyString, makeDiag
from pr2Util import PoseD, shadowName, ObjGraspB, ObjPlaceB, Violations
from pr2Gen import pickGen, canReachHome, placeInGen, lookGen, canReachGen,canSeeGen,lookHandGen, easyGraspGen, canPickPlaceGen
from belief import Bd, B
from pr2Fluents import Conf, CanReachHome, Holding, GraspFace, Grasp, Pose,\
     SupportFace, In, CanSeeFrom, Graspable, CanPickPlace
from planGlobals import debugMsg, debug, useROS

zeroPose = zeroVar = (0.0,)*4
awayPose = (100.0, 100.0, 0.0, 0.0)
maxVarianceTuple = (.1,)*4
defaultPoseDelta = (0.0001, 0.0001, 0.0001, 0.001)
lookConfDelta = (0.0001, 0.0001, 0.0001, 0.001)

# Fixed accuracy to use for some standard preconditions
canPPProb = 0.95
otherHandProb = 0.95
canSeeProb = 0.95
# No prob can go above this
maxProbValue = 0.999
# How sure do we have to be of CRH for moving
movePreProb = 0.98
movePreProb = 0.8
# Prob for generators.  Keep it high.   Should this be = maxProbValue?
probForGenerators = 0.95

######################################################################
#
# Prim functions map an operator's arguments to some parameters that
#  are used during execution
#
######################################################################

tryDirectPath = True
def primPath(bs, cs, ce, p):
    if tryDirectPath:
        path, viols = canReachHome(bs, ce, p, Violations(), startConf=cs,
                                   draw=False)
        if not viols or viols.weight() > 0:
            print 'viol', viols
            raw_input('Collision in direct primitive path')
            # don't return, try the path via home
        else:
            smoothed = bs.getRoadMap().smoothPath(path, bs, p)
            return smoothed
    path1, v1 = canReachHome(bs, cs, p, Violations(), draw=False)
    path2, v2 = canReachHome(bs, ce, p, Violations(), draw=False)
    if v1.weight() > 0 or v2.weight() > 0:
        if v1.weight() > 0: print 'start viol', v1
        if v2.weight() > 0: print 'end viol', v2
        raw_input('Potential collision in primitive path')
        return path1[::-1] + path2
    else:
        print 'Success'
        return path1[::-1] + path2

def movePrim(args, details):
    vl = \
         ['CStart', 'CEnd', 'DEnd',
          'LObj', 'LFace', 'LGraspMu', 'LGraspVar', 'LGraspDelta',
          'RObj', 'RFace', 'RGraspMu', 'RGraspVar', 'RGraspDelta',
          'RealGraspVarL', 'RealGraspVarR',
          'P1', 'P2', 'PCR']
    (cs, ce, cd,
     lo, lf, lgm, lgv, lgd,
     ro, rf, rgm, rgv, rgd,
     rgvl, rgvr,
     p1, p2, pcr) = args

    bs = details.pbs.copy()
    # Make all the objects be fixed
    bs.fixObjBs.update(bs.moveObjBs)
    bs.moveObjBs = {}
    print 'movePrim (start, end)'
    printConf(cs); printConf(ce)

    path = primPath(bs, cs, ce, pcr)
    if debug('prim'):
        print '*** movePrim'
        print zip(vl, args)
        print 'path length', len(path)
    assert path
    return path

def printConf(conf):
    cart = conf.cartConf()
    pose = cart['pr2LeftArm'].pose(fail=False)
    if pose:
        hand = str(np.array(pose.xyztTuple()))
    else:
        hand = '\n'+str(cart['pr2LeftArm'].matrix)
    print 'base', conf['pr2Base'], 'hand', hand
    
def pickPrim(args, details):
    vl = \
         ['Obj', 'Hand', 'OtherHand', 'PoseFace', 'Pose', 'PoseDelta',
          'RObj', 'RFace', 'RGraspMu', 'RGraspVar', 'RGraspDelta',
          'GraspFace', 'GraspMu', 'GraspVar', 'GraspDelta',
          'PreConf', 'ConfDelta', 'PickConf', 'RealGraspVar', 'PoseVar',
          'P1', 'P2', 'P3', 'P4', 'PR1', 'PR2', 'PR3']
    (o, h, oh, pf, p, pd,
     ro, rf, rgm, rgv, rgd,
     gf, gm, gv, gd,
     prc, cd, pc, real, posev,
     p1, p2, p3, p4, pr1, pr2, pr3) = args

    bs = details.pbs.copy()

    print 'plckPrim (start, end)'
    
    # !! Should close the fingers as well?
    if debug('prim'):
        print '*** pickPrim'
        print zip(vl, args)
    return None

def lookPrim(args, details):
    # In the real vision system, we might pass in a more general
    # structure with all the objects (and/or types) we expect to see
    vl = ['Obj', 'LookConf', 'PoseFace', 'Pose',
     'PoseVarBefore', 'PoseDelta', 'PoseVarAfter',
     'P1', 'P2', 'PR1', 'PR2']
    if debug('prim'):
        print '*** lookPrim'
        print zip(vl, args)

    # The distributions for the placed objects, to guide looking
    return details.pbs.getPlacedObjBs()

def lookHandPrim(args, details):
    # In the real vision system, we might pass in a more general
    # structure with all the objects (and/or types) we expect to see
    vl = ['Obj', 'Hand', 'LookConf', 'GraspFace', 'Grasp',
     'GraspVarBefore', 'GraspDelta', 'GraspVarAfter',
     'P1', 'P2', 'PR1', 'PR2']
    if debug('prim'):
        print '*** lookHandPrim'
        print zip(vl, args)

    # The distributions for the grasped objects, to guide looking
    return details.pbs.graspB
    
def placePrim(args, details):
    vl = \
       ['Obj', 'Hand', 'OtherHand', 'Region',
        'PoseFace', 'Pose', 'PoseVar', 'RealPoseVar', 'PoseDelta',
        'GraspFace', 'Grasp', 'GraspVar', 'GraspDelta',
        'RObj', 'RFace', 'RGraspMu', 'RGraspVar', 'RGraspDelta',         
        'PreConf', 'ConfDelta', 'PlaceConf',
        'PR1', 'PR2', 'PR3', 'PR4', 'P1', 'P2', 'P3']
    (o, h, oh, r, pf, p, pv, rpv, pd,
     gf, gm, gv, gd,
     ro, rf, rgm, rgv, rgd,
     prc, cd, pc, 
     pr1, pr2, pr3, pr4, p1, p2, p3) = args

    bs = details.pbs.copy()
    # Plan a path from cs to ce
    # bs.updateHeld(ro, rf, PoseD(rgm, rgv), 'right', delta=rgd)

    print 'placePrim (start, end)'
    # printConf(prc); printConf(pc)
    # path = primPath(bs, prc, pc, p2)

    # !! Should open the fingers as well
    if debug('prim'):
        print '*** placePrim'
        print zip(vl, args)

    return None


################################################################
## Simple generators
################################################################

# Relevant fluents:
#  Holding(hand), GraspFace(obj, hand), Grasp(obj, hand, face)

smallDelta = (10e-4,)*4

def oh(h):
    return 'left' if h == 'right' else 'right'

def otherHand((hand,), goal, start, vals):
    if hand == 'left':
        return [['right']]
    else:
        return [['left']]

def getObjAndHands(args, goal, start, stuff):
    (o, h) = args
    heldLeft = start.pbs.getHeld('left').mode()        
    heldRight = start.pbs.getHeld('right').mode()        
    result = []
    if isVar(o):
        # Obj is unspecified, h should be bound.  
        hh = heldLeft if h == 'left' else heldRight
        if hh == 'none':
            # If there is nothing in the hand right now, then we
            # should technically iterate through all possible objects.
            # For now, fail.
            result = []
        else:
            result = [(hh, h, oh(h))]
    else:
        if o == 'none' and heldLeft == 'none' and heldRight == 'none':
            return []
        # Obj is specified
        if not isVar(h):
            # hand is specified
            hands = [h]
        elif heldLeft == o:
            # Try left first
            hands = ['left', 'right']
        elif heldRight == o:
            # Try right first
            hands = ['right', 'left']
        elif heldLeft == 'none':
            # Either order okay, but prefer empty one
            hands = ['left', 'right']
        else:
            hands = ['right', 'left']
        result = [(o, hand, oh(hand)) for hand in hands]
    return result

# Return a tuple (obj, face, mu, var, delta).  Values taken from the start state
def graspStuffFromStart(start, hand):
    pbs = start.pbs
    obj = pbs.getHeld(hand).mode()
    if obj == 'none':
        return ('none', 0, (0.0, 0.0, 0.0, 0.0),
                           (0.0, 0.0, 0.0, 0.0),
                           smallDelta)

    gd = pbs.getGraspB(obj, hand)
    face = gd.grasp.mode()
    mu = gd.poseD.mu.xyztTuple()
    var = gd.poseD.var
    return (obj, face, mu, var, smallDelta)

# Return a tuple (obj, face, mu, var, delta).  Defaults, overridden by
# what we find in the goal
def graspStuffFromGoal(goal, hand,
                       defaults = ('none', 0,
                                   (0.0, 0.0, 0.0, 0.0),
                                   (0.0, 0.0, 0.0, 0.0),
                                   smallDelta)):

    (obj, face, mu, var, delta) = defaults
    hb = getMatchingFluents(goal, Bd([Holding([hand]), 'Obj', 'P'], True))
    assert len(hb) < 2
    if len(hb) == 0:
        return (obj, face, mu, var, delta)
    obj = hb[0][1]['Obj']

    if obj == 'none':
        return (obj, face, mu, var, delta)

    fb = getMatchingFluents(goal, Bd([GraspFace([obj, hand]),'Face', 'P'],True))
    assert len(fb) < 2
    if len(fb) == 0:
        return (obj, face, mu, var, delta)
    face = fb[0][1]['Face']

    gb = getMatchingFluents(goal, B([Grasp([obj, hand, face]),
                                     'M', 'V', 'D', 'P'], True))
    assert len(gb) < 2
    if len(gb) == 0:
        return (obj, face, mu, var, delta)
    (mu, var, delta) = (gb[0][1]['M'], gb[0][1]['V'], gb[0][1]['D'])

    return (obj, face, mu, var, delta)

# See if would be useful to look at obj in order to reduce its variance
def graspVarCanPickPlaceGen(args, goal, start, vals):
    (obj, variance) = args
    if obj != 'none' and variance[0] > start.domainProbs.obsVarTuple[0]:
        return [[start.domainProbs.obsVarTuple]]
    else:
        return []

# Get all grasp-relevant information for both hands.  Used by move.
def genGraspStuff(args, goal, start, vals):
    return [a + b for (a, b) in \
      zip(genGraspStuffHand(('left', 'O1'), goal, start, vals),
          genGraspStuffHand(('right', 'O2'), goal, start, vals))]

# Get grasp-relevant stuff for one hand.  Used by move, place, pick
def genGraspStuffHand((hand, otherObj), goal, start, values):
    # See if there is a pose requirement in the goal (in which case that
    # obj can't be in the hand.
    pb = getMatchingFluents(goal,
                            B([Pose(['O', 'F']), 'PM', 'PV', 'PD', 'P'], True))
    fixedObjs = [b['O'] for (f, b) in pb]
    # If it's not required by the goal, then let it be the
    # value in the start state
    s1 = graspStuffFromGoal(goal, hand, graspStuffFromStart(start, hand)) 
    # Try with empty as a default, even if it's different in start
    s2 = graspStuffFromGoal(goal, hand)
    ans = []
    if s1[0] != otherObj and not s1[0] in fixedObjs: ans.append(s1)
    if s2[0] != otherObj: ans.append(s2)
    debugMsg('genGraspStuff', hand, ans)
    return ans

# Use this when we don't want to generate an argument (expecting to
# get it from goal bindings.)  Guaranteed to fail if that var isn't
# already bound.
def genNone(args, goal, start, vals):
    return None

def assign(args, goal, start, vals):
    return args

# Be sure the argument is not 'none'
def notNone(args, goal, start, vals):
    if args[0] == 'none':
        return None
    else:
        return [[]]

def notEqual(args, goal, start, vals):
    if args[0] == args[1]:
        result = None
    else:
        result = [[]]
    return result

# It pains me that this has to exist; but the args needs to be a list
# not a structure of variables.
def notEqual2(args, goal, start, vals):
    if (args[0], args[1]) == (args[2], args[3]):
        result = None
    else:
        result = [[]]
    return result
    
# Isbound
def isBound(args, goal, start, vals):
    if isVar(args[0]):
        return None
    else:
        return [[]]
    
# Return as many values as there are args; overwrite any that are
# variables with the minimum value
def minP(args, goal, start, vals):
    minVal = min([a for a in args if not isVar(a)])
    return [[minVal if isVar(a) else a for a in args]]

# Regression:  what does the mode need to be beforehand, assuming a good
# outcome
def obsModeProb(args, goal, start, vals):
    p = max([a for a in args if not isVar(a)])
    pFalsePos = pFalseNeg = start.domainProbs.obsTypeErrProb
    pr = p * pFalsePos / ((1 - p) * (1 - pFalseNeg) + p * pFalsePos)
    return [[max(0.1, pr)]]

# Compute the nth root of the maximum defined prob value

def regressProb(n, probName = None):
    def regressProbAux(args, goal, start, vals):
        failProb = getattr(start.domainProbs, probName) if probName else 0.0
        pr = max([a for a in args if not isVar(a)]) / (1 - failProb)
        val = np.power(pr, 1.0/n)
        if val < maxProbValue:
            return [[val]*n]
        else:
            return []
    return regressProbAux

def halveVariance((var,), goal, start, vals):
    return [[tuple([min(maxVarianceTuple[0], v / 2.0) for v in var])]*2]

def maxGraspVarFun((var,), goal, start, vals):
    assert not(isVar(var))
    maxGraspVar = (0.015**2, .015**2, .015**2, .03**2)

    # A conservative value to start with, but then try whatever the
    # variance is in the current grasp
    result = [[tuple([min(x,y) for (x,y) in zip(var, maxGraspVar)])]] 

    lgs = graspStuffFromStart(start, 'left')
    if lgs[0] != 'none':
        result.append([tuple([min(x,y) for (x,y) in zip(var, lgs[3])])])
    rgs = graspStuffFromStart(start, 'right')
    if rgs[0] != 'none':
        result.append([tuple([min(x,y) for (x,y) in zip(var, rgs[3])])])
    return result


def realPoseVar((graspVar,), goal, start, vals):
    placeVar = start.domainProbs.placeVar
    return [[tuple([gv+pv for (gv, pv) in zip(graspVar, placeVar)])]]

# For place, grasp var is desired poseVar minus fixed placeVar
# Don't let it be bigger than maxGraspVar 
def placeGraspVar((poseVar,), goal, start, vals):
    maxGraspVar = (0.0004, 0.0004, 0.0004, 0.008)
    placeVar = start.domainProbs.placeVar
    if isVar(poseVar):
        # For placing in a region; could let the place pick this, but
        # just do it for now
        defaultPoseVar = tuple([4*v for v in placeVar])
        poseVar = defaultPoseVar
    graspVar = tuple([min(gv - pv, m) for (gv, pv, m) \
                      in zip(poseVar, placeVar, maxGraspVar)])
    if any([x <= 0 for x in graspVar]):
        return []
    else:
        return [[graspVar]]

# For pick, pose var is desired graspVar minus fixed pickVar
def pickPoseVar((graspVar, prob), goal, start, vals):
    pickVar = start.domainProbs.pickVar
    pickTolerance = start.domainProbs.pickTolerance[0]
    # What does the variance need to be so that we are within
    # pickTolerance with probability prob?
    numStdDevs =  np.sqrt(chiSqFromP(1-prob, 3))
    # nstd * std < pickTol
    # std < pickTol / nstd
    tolerableVar = (pickTolerance / numStdDevs)**2
    poseVar = tuple([min(gv - pv, tolerableVar) \
                     for (gv, pv) in zip(graspVar, pickVar)])
    if any([x <= 0 for x in poseVar]):
        debugMsg('pickGen', 'pick pose var negative', poseVar)
        return []
    else:
        debugMsg('pickGen', 'pick pose var', poseVar)
        return [[poseVar]]

# If it's bigger than this, we can't just plan to look and see it
# Should be more subtle than this...
maxPoseVar = (0.1**2, 0.1**2, 0.1**2, 0.2**2)

# starting var if it's legal, plus regression of the result var
def genLookObjPrevVariance((ve, obj, face), goal, start, vals):
    epsilon = 10e-5
    lookVar = start.domainProbs.obsVarTuple
    vs = tuple(start.poseModeDist(obj, face).mld().sigma.diagonal().tolist()[0])
    vbo = varBeforeObs(lookVar, ve)
    cappedVbo = tuple([min(a, b) for (a, b) in zip(maxPoseVar, vbo)])
    result = []
    # We might be looking to increase the mode prob, so don't fail
    # !!!
    #if cappedVbo[0] > ve[0]:
    result.append([cappedVbo])

    startLessThanMax = any([a < b for (a, b) in zip(vs, maxPoseVar)])
    startUseful = any([a > b for (a, b) in zip(vs, ve)])

    if startLessThanMax and startUseful:
        # starting var is bigger, but not too big
        result.append([vs])
    return result

# starting var if it's legal, plus regression of the result var
def genLookObjHandPrevVariance((ve, hand, obj, face), goal, start, vals):
    epsilon = 10e-5
    lookVar = start.domainProbs.obsVarTuple
    maxGraspVar = (0.0008, 0.0008, 0.0008, 0.008)

    result = []
    hs = start.pbs.getHeld(hand).mode()
    vs = None
    if hs == obj:
        vs = tuple(start.graspModeDist(obj, hand, face)\
                        .mld().sigma.diagonal().tolist()[0])
        if vs[0] < maxPoseVar[0] and vs[0] > ve[0]:
            # starting var is bigger, but not too big
            result.append([vs])
    vbo = varBeforeObs(lookVar, ve)
    cappedVbo = tuple([min(a, b) for (a, b) in zip(maxGraspVar, vbo)])

    debugMsg('lookObjHand', ('postVar', ve), ('vstart', vs),
             ('preVar', vbo), ('capped preVar', cappedVbo))

    # It's tempting to fail in this case; but we may be looking to
    # increase the mode prob
    # !!!
    if True: #cappedVbo[0] > ve[0]:
        result.append([cappedVbo])
    return result

# Add a condition on the pose and face of an object
def addPosePreCond((postCond, obj, poseFace, pose, poseVar, poseDelta, p),
                   goal, start, vals):
    newFluents = [Bd([SupportFace([obj]), poseFace, p], True),
                  B([Pose([obj, poseFace]), pose, poseVar, poseDelta, p],
                     True)]
    fluentList = simplifyCond(postCond, newFluents)
    return [[fluentList]]


def awayRegionIfNecessary((region, pose), goal, start, vals):
    if not isVar(pose) or not isVar(region):
        return [[region]]
    else:
        return [[start.pbs.awayRegions()]]

################################################################
## Cost funs
################################################################

# So many ways to do this...
def costFun(primCost, prob):
    if prob == 0:
        print 'costFun: prob = 0, returning inf'
        raw_input('okay?')
    return float('inf') if prob == 0 else primCost / prob

# Cost depends on likelihood of success: canReach, plus objects in each hand
# Add in a real distance from cs to ce

def moveCostFun(al, args, details):
    (s, e, _, _, _, _, _, _, _, _, _, _, _, _, _, p1, p2, pcr) = args
    result = costFun(1.0, p1 * p2 * pcr)
    debugMsg('cost', ('move', (p1, p2, pcr), result))
    return result

def placeCostFun(al, args, details):
    (_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,p1,p2,p3) = args
    result = costFun(1.0, p1 * p2 * p3 * (1-details.domainProbs.placeFailProb))
    debugMsg('cost', ('place', (p1, p2, p3), result))
    return result

def pickCostFun(al, args, details):
    (_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,p1,p2,p3,p4,_,_,_) = args
    result = costFun(1.0, p1*p2*p3*p4 * (1 - details.domainProbs.pickFailProb))
    debugMsg('cost', ('pick', (p1, p2, p3, p4), result))
    return result

# Cost depends on likelihood of seeing the object and of moving the
# mean out of the delta bounds
# When we go to non-diagonal covariance, this will be fun...
# For now, just use the first term!
def lookAtCostFun(al, args, details):
    (_,_,_,_,vb,d,va,pb,pCanSee,pPoseR,pFaceR) = args
    placeProb = min(pPoseR, pFaceR)
    vo = details.domainProbs.obsVarTuple
    deltaViolProb = probModeMoved(d[0], vb[0], vo[0])
    # Switched to using var *after* look because if look reliability
    # is very high then var before is huge and so is the cost.
    deltaViolProb = probModeMoved(d[0], va[0], vo[0])
    result = costFun(1.0, pCanSee*placeProb*(1-deltaViolProb)*\
                     (1 - details.domainProbs.obsTypeErrProb))
    debugMsg('cost',
             ('lookAt',
              (pCanSee, placeProb,
               1-deltaViolProb, 1-details.domainProbs.obsTypeErrProb),
                result))
    return result

def lookAtHandCostFun(al, args, details):
    # Two parts:  the probability and the variance
    (_,_,_,_,_,vb,d,va,pb,pGraspR,pFaceR,pHoldingR) = args
    
    holdingProb = min(pGraspR, pFaceR, pHoldingR)
    if not isVar(d):
        vo = details.domainProbs.obsVarTuple
        #deltaViolProb = probModeMoved(d[0], vb[0], vo[0])
        # Switched to using var *after* look because if look
        # reliability is very high then var before is huge and so is
        # the cost.
        deltaViolProb = probModeMoved(d[0], va[0], vo[0])
    else:
        deltaViolProb = 0
    result = costFun(1.0, holdingProb*(1-deltaViolProb))
    debugMsg('cost', ('lookAtHand', (holdingProb, 1-deltaViolProb), result))
    return result

################################################################
## Action effects
################################################################

# All super-bogus until we get the real state estimator running

def moveBProgress(details, args, obs=None):
    (s, e, _, _, _, _, _, _, _, _, _, _, _, _, _, p1, p2, pcr) = args
    # Totally fake for now!  Just put the robot in the intended place
    details.pbs.updateConf(e)
    details.shadowWorld = None # force recompute
    debugMsg('beliefUpdate', 'moveBel')    

def pickBProgress(details, args, obs=None):
    # Assume robot ends up at preconf, so no conf change
    (o, h, _, _, _, _, _, _, _, _, _, gf, gm, gv, gd, _,_,_,_,_,_,_,_,_,_,_,_)=\
       args
    pickVar = details.domainProbs.pickVar
    failProb = details.domainProbs.pickFailProb
    # !! This is wrong!  The coordinate frames of the variances don't match.
    v = [x+y for x,y in zip(details.pbs.getPlaceB(o).poseD.var, pickVar)]
    v[2] = 1e-8
    gv = tuple(v)
    details.graspModeProb[h] = (1 - failProb) * details.poseModeProbs[o]
    details.pbs.updateHeld(o, gf, PoseD(gm, gv), h, gd)
    details.pbs.excludeObjs([o])
    details.pbs.shadowWorld = None # force recompute
    debugMsg('beliefUpdate', 'pickBel')

def placeBProgress(details, args, obs=None):
    # Assume robot ends up at preconf, so no conf change
    (o, h, _, r, pf, p, _, _, _,_,_,_, _, _, _, _, _, _,_,_,_,_,_,_,_,_,_, _) =\
       args
    placeVar = details.domainProbs.placeVar
    failProb = details.domainProbs.placeFailProb
    # !! This is wrong!  The coordinate frames of the variances don't match.
    v = [x+y for x,y in \
         zip(details.pbs.getGraspB(o,h).poseD.var, placeVar)]
    v[2] = 1e-8
    gv = tuple(v)
    details.poseModeProbs[o] = (1 - failProb) * details.graspModeProb[h]
    details.pbs.updateHeld('none', None, None, h, None)
    ff = details.pbs.getWorld().getFaceFrames(o)
    if isinstance(pf, int):
        pf = DeltaDist(pf)
    else:
        raw_input('place face is DDist, should be int')
        pf = pf.mode()
    details.pbs.moveObjBs[o] = ObjPlaceB(o, ff, pf, PoseD(p, gv))
    details.pbs.shadowWorld = None # force recompute
    debugMsg('beliefUpdate', 'placeBel')
    
# For now, assume obs has the form (obj, face, pose) or None
# Really, we'll get obj-type, face, relative pose
def lookAtBProgress(details, args, obs):
    (o, _, _, _, _, _, _, _, _, _, _) = args
    objectObsUpdate(details, obs, o)
    details.pbs.shadowWorld = None # force recompute
    debugMsg('beliefUpdate', 'look')

llMatchThreshold = -100  # very liberal

def objectObsUpdate(details, obs, soughtObject):
    # Assume for now a single observed object.  Make this fancier when
    # we can get multiple detections.
    if obs == None:
        debugMsg('obsUpdate', 'No good match for observation')
        # Update modeprob
        oldP = details.poseModeProbs[soughtObject]
        obsGivenH = details.domainProbs.obsTypeErrProb
        obsGivenNotH = (1 - details.domainProbs.obsTypeErrProb)
        newP = obsGivenH * oldP / (obsGivenH * oldP + obsGivenNotH * (1 - oldP))
        details.poseModeProbs[soughtObject] = newP
        return
    
    (oType, obsFace, obsPose) = obs
    pbs = details.pbs
    w = pbs.beliefContext.world
    symFacesType, symXformsType = w.getSymmetries(oType)
    canonicalFace = symFacesType[obsFace]
    symXForms = symXformsType[canonicalFace]
    # Could make this more efficient by mapping to a canonical one, but
    # seems risky
    symPoses = [obsPose] + [obsPose.compose(xf) for xf in symXForms]

    candidates = []

    ff = pbs.beliefContext.world.getFaceFrames(soughtObject)[canonicalFace]

    if debug('obsUpdate'):
        ## LPK!!  Should really draw the detected object but I don't have
        ## an immediate way to get the shape of a type.  Should fix that.
        objShape = pbs.getObjectShapeAtOrigin(soughtObject)
        objShape.applyLoc(obsPose.compose(ff.inverse())).draw('Belief', 'cyan')
        raw_input('obs is cyan')

    # Find the best matching pose mode.  Type must be equal, pose nearby.
    for o in pbs.objNames:
        # Type
        if w.getObjType(o) != oType: continue

        oldObjBel = pbs.getPlaceB(o)

        # Face
        oldPoseFace = oldObjBel.support.mode()
        assert symFacesType[oldPoseFace] == oldPoseFace, \
                                        'non canonical face in bel'
        if oldPoseFace != canonicalFace: continue
        
        # Pose
        # Create old distribution.
        oldPoseMu = oldObjBel.poseD.mode()
        oldSigma = oldObjBel.poseD.variance()
        obsSigma = [v1 + v2 for (v1, v2) in zip(oldSigma,
                                          details.domainProbs.obsVarTuple)]
        obsD = MultivariateGaussianDistribution(np.mat(oldPoseMu.xyztTuple()).T,
                                                makeDiag(obsSigma))
        bestObs, bestLL = None, -float('inf')
        for obsPoseCand in symPoses:
            ll = float(obsD.logProb(np.mat(obsPoseCand.pose().xyztTuple()).T))
            if ll > bestLL:
                bestObs, bestLL = obsPoseCand, ll

        debugMsg('obsUpdate', 'Potential match with', o, 'll', bestLL,
                 bestLL > llMatchThreshold)
        if bestLL > llMatchThreshold:
            candidates.append((bestLL, o, obsPoseCand, oldPoseMu, oldSigma))

    if len(candidates) == 0:
        # No match for this observation
        # !LPK: also do this if the best match isn't for the object we were
        # looking for?
        debugMsg('obsUpdate', 'No good match for observation')
        # Update modeprob
        oldP = details.poseModeProbs[soughtObject]
        obsGivenH = details.domainProbs.obsTypeErrProb
        obsGivenNotH = (1 - details.domainProbs.obsTypeErrProb)
        newP = obsGivenH * oldP / (obsGivenH * oldP + obsGivenNotH * (1 - oldP))
        details.poseModeProbs[soughtObject] = newP
    else:
        # Find the best candidate match and do the update.
        candidates.sort(reverse = True)
        (_, obj, pose, oldMu, oldSigma) = candidates[0]

        # Update mode prob
        oldP = details.poseModeProbs[obj]
        obsGivenH = (1 - details.domainProbs.obsTypeErrProb)
        obsGivenNotH = details.domainProbs.obsTypeErrProb
        newP = obsGivenH * oldP / (obsGivenH * oldP + obsGivenNotH * (1 - oldP))
        details.poseModeProbs[obj] = newP

        # Update mean and sigma
        obsVar = details.domainProbs.obsVarTuple
        (newMu, newSigma) = gaussObsUpdate(oldMu.pose().xyztTuple(),
                                           pose.pose().xyztTuple(),
                                           oldSigma, obsVar)
        details.pbs.updateObjB(ObjPlaceB(obj, w.getFaceFrames(obj),
                                         DeltaDist(oldPoseFace),
                                         PoseD(util.Pose(*newMu), newSigma)))
        if debug('obsUpdate'):
            objShape = pbs.getObjectShapeAtOrigin(soughtObject)
            objShape.applyLoc(util.Pose(*newMu).compose(ff.inverse())).\
              draw('Belief', 'magenta')
            raw_input('newMu is magenta')
         

# Temporary;  assumes diagonal cov; should use dist.MultivariateGaussian
def gaussObsUpdate(oldMu, obs, oldSigma, obsVar, noZ = True):
    # All tuples
    newMu = [(m * obsV + op * muV) / (obsV + muV) \
                       for (m, muV, op, obsV) in \
                       zip(oldMu, oldSigma, obs, obsVar)]
    newSigma = tuple([(a * b) / (a + b) for (a, b) in zip(oldSigma,obsVar)])
    if noZ:
        newMu[2] = oldMu[2]
    print 'new sigma', newSigma
    # raw_input('okay?')
    return (tuple(newMu), newSigma)
    
# For now, assume obs has the form (obj, face, grasp) or None
# Really, we'll get obj-type, face, relative pose
def lookAtHandBProgress(details, args, obs):
    (_, h, _, f, _, _, _,  _, _, _, _,_) = args
    # Awful!  Should have an attribute of the object
    universe = [o for o in details.pbs.beliefContext.world.objects.keys() \
                if o[0:3] == 'obj']
    heldDist = details.pbs.held[h]

    oldMlo = heldDist.mode()
    # Update dist on what we're holding if we got an observation
    if obs != None:
        obsObj = 'none' if obs == 'none' else obs[0]
        # Observation model
        def om(trueObj):
            return MixtureDist(DeltaDist(trueObj),
                               UniformDist(universe + ['none']),
                               1 - details.domainProbs.obsTypeErrProb)
        heldDist.obsUpdate(om, obsObj)

        # If we are fairly sure of the object, update the mode object's dist
        mlo = heldDist.mode()
        bigSigma = (0.01, 0.01, 0.01, 0.04)
        if mlo == 'none':
            newOGB = None
        # If we now have a new mode, we have to reinitialize the grasp dist!
        elif mlo != oldMlo:
            details.graspModeProb[h] = 1 - details.domainProbs.obsTypeErrProb
            gd = details.pbs.graspB[h].graspDesc
            newOGB = objGraspB(mlo, gd, PoseD(util.Pose(0, 0, 0, 0),
                                                  bigSigma))
        elif mlo != 'none' and obsObj != 'none':
            (_, ogf, ograsp) = obs            
            # Bayes update on mode prob
            oldP = details.graspModeProb[h]
            obsGivenH = (1 - details.domainProbs.obsTypeErrProb)
            obsGivenNotH = details.domainProbs.obsTypeErrProb
            newP = obsGivenH * oldP / \
                        (obsGivenH * oldP + obsGivenNotH * (1 - oldP))
            details.graspModeProb[h] = newP

            gd = details.pbs.graspB[h].graspDesc
            # Update the rest of the distributional info.
            # Consider only doing this if the mode prob is high
            mlop = heldDist.prob(mlo)
            faceDist = details.pbs.graspB[h].grasp
            oldMlf = faceDist.mode()

            # Should do an update, but since we only have one grasp for now it
            # doesn't make sense
            # faceDist.obsUpdate(fom, ogf)
            faceDist = DeltaDist(ogf)

            mlf = faceDist.mode()
            poseDist = details.pbs.graspB[h].poseD
            oldMu = poseDist.mode().xyztTuple()
            oldSigma = poseDist.variance()

            if mlf != oldMlf:
                # Most likely grasp face changed.
                # Keep the pose for lack of a better idea, but
                # increase sigma a lot!
                newPoseDist = PoseD(poseDist.mode(), bigSigma)
                raw_input('Grasp changed')
            else:
                # Cheapo obs update
                obsVar = details.domainProbs.obsVarTuple
                newMu = tuple([(m * obsV + op * muV) / (obsV + muV) \
                       for (m, muV, op, obsV) in \
                       zip(oldMu, oldSigma, ograsp, obsVar)])
                newSigma = tuple([(a * b) / (a + b) for (a, b) in \
                                  zip(oldSigma,obsVar)])
                newPoseDist = PoseD(util.Pose(*newMu), newSigma)
            newOGB = ObjGraspB(mlo, gd, faceDist, newPoseDist)
        details.pbs.updateHeldBel(newOGB, h)
    details.pbs.shadowWorld = None # force recompute
    debugMsg('beliefUpdate', 'look')

################################################################
## Operator descriptions
################################################################

# How sure the move preconditions need to be.  Making this lower makes
# the plan more likely to fail (cost be higher).  Making this higher
# makes the plan work harder to be sure the preconds are satisfied.


def genMoveProbs(args, goal, start, vals):
    return [[movePreProb]*3]

# Parameter PCR is the probability that its path is not blocked.
# Likelihood of success is p1 * p2 * pcr

move = Operator(\
    'Move',
    ['CStart', 'CEnd', 'DEnd',
     'LObj', 'LFace', 'LGraspMu', 'LGraspVar', 'LGraspDelta',
     'RObj', 'RFace', 'RGraspMu', 'RGraspVar', 'RGraspDelta',
     'RealGraspVarL', 'RealGraspVarR',
     'P1', 'P2', 'PCR'],
    # Pre
    {0 : {Bd([CanReachHome(['CEnd', 'left',
                    'LObj', 'LFace', 'LGraspMu', 'RealGraspVarL', 'LGraspDelta',
                    'RObj', 'RFace', 'RGraspMu', 'RealGraspVarR', 'RGraspDelta',
                            []]),  True, 'PCR'], True)},
     1 : {Conf(['CStart', 'DEnd'], True),
          Bd([Holding(['left']), 'LObj', 'P1'], True),
          Bd([GraspFace(['LObj', 'left']), 'LFace', 'P1'], True),
          B([Grasp(['LObj', 'left', 'LFace']),
             'LGraspMu', 'RealGraspVarL', 'LGraspDelta', 'P1'], True),
          Bd([Holding(['right']), 'RObj', 'P2'], True),
          Bd([GraspFace(['RObj', 'right']), 'RFace', 'P2'], True),
          B([Grasp(['RObj', 'right', 'RFace']),
             'RGraspMu', 'RealGraspVarR', 'RGraspDelta', 'P2'], True)
             }},
    # Results:  list of pairs: (fluent set, private preconds)
    [({Conf(['CEnd', 'DEnd'], True)}, {})],
    functions = [\
        Function(['CEnd'], [], genNone, 'genNone'),                 
        Function(['PCR', 'P1', 'P2'], [], genMoveProbs,
                 'genMoveProbs'),
        Function(['LObj', 'LFace', 'LGraspMu', 'LGraspVar', 'LGraspDelta',
                  'RObj', 'RFace', 'RGraspMu', 'RGraspVar', 'RGraspDelta'],
                 [], genGraspStuff, 'genGraspStuff'),
        Function(['RealGraspVarL'], ['LGraspVar'], maxGraspVarFun,
                     'realGraspVar'),
        Function(['RealGraspVarR'], ['RGraspVar'], maxGraspVarFun,
                     'realGraspVar')
                 ],
    cost = moveCostFun,
    f = moveBProgress,
    prim  = movePrim,
    argsToPrint = [0, 1],
    ignorableArgs = range(2,18))  # For abstraction

# Likelihood of success is p1 * p2 * p3

place = Operator(\
        'Place',
        ['Obj', 'Hand', 'OtherHand',
         'Region', 'PoseFace', 'Pose', 'PoseVar', 'RealPoseVar', 'PoseDelta',
         'GraspFace', 'GraspMu', 'GraspVar', 'GraspDelta',
         'OObj', 'OFace', 'OGraspMu', 'OGraspVar', 'OGraspDelta',         
         'PreConf', 'ConfDelta', 'PlaceConf',
         'PR1', 'PR2', 'PR3', 'PR4', 'P1', 'P2', 'P3'],
        # Pre
        {0 : {Graspable(['Obj'], True)},
         1 : {Bd([CanPickPlace(['PreConf', 'PlaceConf', 'Hand', 'Obj', 'Pose',
                               'RealPoseVar', 'PoseDelta', 'PoseFace',
                               'GraspFace', 'GraspMu', 'GraspVar', 'GraspDelta',
                               'OObj', 'OFace', 'OGraspMu', 'OGraspVar', 
                               'OGraspDelta', []]), True, 'P1'], True)},
         2 : {Bd([Holding(['Hand']), 'Obj', 'P2'], True),
              Bd([GraspFace(['Obj', 'Hand']), 'GraspFace', 'P2'], True),
              B([Grasp(['Obj', 'Hand', 'GraspFace']),
                 'GraspMu', 'GraspVar', 'GraspDelta', 'P2'], True),
              # Bookkeeping for other hand
              Bd([Holding(['OtherHand']), 'OObj', 'P3'], True),
              Bd([GraspFace(['OObj', 'OtherHand']), 'OFace', 'P3'], True),
              B([Grasp(['OObj', 'OtherHand', 'OFace']),
                       'OGraspMu', 'OGraspVar', 'OGraspDelta', 'P3'], True)},
         3 : {Conf(['PreConf', 'ConfDelta'], True)}
        },
        # Results
        [({Bd([In(['Obj', 'Region']), True, 'PR4'], True)}, {}),
         ({Bd([SupportFace(['Obj']), 'PoseFace', 'PR1'], True),
           B([Pose(['Obj', 'PoseFace']), 'Pose', 'PoseVar', 'PoseDelta','PR2'],
                 True)},{}),
         ({Bd([Holding(['Hand']), 'none', 'PR3'], True)}, {})],
        # Functions
        functions = [\
            # Get both hands and object!
            Function(['Obj', 'Hand', 'OtherHand'], ['Obj', 'Hand'],
                     getObjAndHands, 'getObjAndHands'),
            # Either Obj is bound (because we're trying to place it) or
            # Hand is bound (because we're trying to make it empty)
            # If Obj is not bound then: get it from the start state;
            #  also, let region be awayRegion
            Function(['Region'], ['Region', 'Pose'], awayRegionIfNecessary,
                                   'awayRegionIfNecessary'),
            
            # Be sure all result probs are bound.  At least one will be.
            Function(['PR1', 'PR2', 'PR3', 'PR4'],
                     ['PR1', 'PR2', 'PR3', 'PR4'], minP,'minP'),

            # Compute precond probs.  Assume that crash is observable.
            # So, really, just move the obj holding prob forward into
            # the result.  Use canned probs for the other ones.
            Function(['P2'], ['PR1', 'PR2', 'PR3', 'PR4'], 
                     regressProb(1, 'placeFailProb'), 'regressProb1'),
            Function(['P1', 'P3'], [[canPPProb, otherHandProb]],
                     assign, 'assign'),

            # PoseVar = GraspVar + PlaceVar,
            # GraspVar = min(maxGraspVar, PoseVar - PlaceVar)
            Function(['GraspVar'], ['PoseVar'], placeGraspVar, 'placeGraspVar'),

            # Real pose var might be much less than pose var if the
            # original pos var was very large
            # RealPoseVar = GraspVar + PlaceVar
            Function(['RealPoseVar'],
                     ['GraspVar'], realPoseVar, 'realPoseVar'),
            
            # In case PoseDelta isn't defined
            Function(['PoseDelta'],[],lambda a,g,s,v: [[defaultPoseDelta]],
                     'defaultPoseDelta'),
            # Divide delta evenly
            Function(['ConfDelta', 'GraspDelta'], ['PoseDelta'],
                      halveVariance, 'halveVar'),

            # Values for what is in the other hand
            Function(['OObj', 'OFace', 'OGraspMu', 'OGraspVar', 'OGraspDelta'],
                       ['OtherHand', 'Obj'], genGraspStuffHand,
                       'genGraspStuffHand'),

            # Not modeling the fact that the object's shadow should
            # grow a bit as we move to pick it.   Build that into pickGen.
            Function(['Pose', 'PoseFace', 'GraspMu', 'GraspFace', 'GraspVar',
                      'PlaceConf', 'PreConf'],
                     ['Obj', 'Region','Pose', 'PoseFace', 'PoseVar', 'GraspVar',
                      'PoseDelta', 'GraspDelta', 'ConfDelta', 'Hand',
                     probForGenerators],
                     placeInGen, 'placeInGen')

            ],
        cost = placeCostFun, 
        f = placeBProgress,
        prim = placePrim,
        argsToPrint = [0, 1, 3, 4, 5],
        ignorableArgs = range(2, 27))

pick = Operator(\
        'Pick',
        ['Obj', 'Hand', 'OtherHand', 'PoseFace', 'Pose', 'PoseDelta',
         'OObj', 'OFace', 'OGraspMu', 'OGraspVar', 'OGraspDelta',
         'GraspFace', 'GraspMu', 'GraspVar', 'GraspDelta',
         'PreConf', 'ConfDelta', 'PickConf', 'RealGraspVar', 'PoseVar',
         'P1', 'P2', 'P3', 'P4', 'PR1', 'PR2', 'PR3'],
        # Pre
        {0 : {Graspable(['Obj'], True),
              Bd([SupportFace(['Obj']), 'PoseFace', 'P1'], True),
              B([Pose(['Obj', 'PoseFace']), 'Pose', 'PoseVar', 'PoseDelta',
                 'P1'], True)},
         1 : {Bd([CanPickPlace(['PreConf', 'PickConf', 'Hand', 'Obj', 'Pose',
                               'PoseVar', 'PoseDelta', 'PoseFace',
                               'GraspFace', 'GraspMu', 'RealGraspVar',
                               'GraspDelta',
                               'OObj', 'OFace', 'OGraspMu', 'OGraspVar', 
                               'OGraspDelta', []]), True, 'P2'], True)},
            # Implicitly, CanPick should be true, too
         2  : {Conf(['PreConf', 'ConfDelta'], True),
             Bd([Holding(['Hand']), 'none', 'P3'], True),
             # Bookkeeping for other hand
             Bd([Holding(['OtherHand']), 'OObj', 'P4'], True),
             Bd([GraspFace(['OObj', 'OtherHand']), 'OFace', 'P4'], True),
             B([Grasp(['OObj', 'OtherHand', 'OFace']),
                       'OGraspMu', 'OGraspVar', 'OGraspDelta', 'P4'], True)
             }},

        # Results
        [({Bd([Holding(['Hand']), 'Obj', 'PR1'], True), 
           Bd([GraspFace(['Obj', 'Hand']), 'GraspFace', 'PR2'], True),
           B([Grasp(['Obj', 'Hand', 'GraspFace']),
             'GraspMu', 'GraspVar', 'GraspDelta', 'PR3'], True)}, {})],
        # Functions
        functions = [\
            # Be sure obj is not none -- don't use this to empty the hand
            Function([], ['Obj'], notNone, 'notNone', True),

            # Get other hand
            Function(['OtherHand'], ['Hand'], otherHand, 'otherHand'),
            
            # Be sure all result probs are bound.  At least one will be.
            Function(['PR1', 'PR2', 'PR3'], ['PR1', 'PR2', 'PR3'], minP,'minP'),

            # Compute precond probs.  Only regress object placecement P1.
            # Consider failure prob
            Function(['P1'], ['PR1', 'PR2'], 
                     regressProb(1, 'pickFailProb'), 'regressProb1'),
            Function(['P2', 'P3', 'P4'],[[canPPProb, canPPProb, otherHandProb]],
                    assign, 'assign'),
            Function(['RealGraspVar'], ['GraspVar'], maxGraspVarFun,
                     'realGraspVar'),
                     
            # GraspVar = PoseVar + PickVar
            Function(['PoseVar'], ['RealGraspVar', 'PR3'],
                     pickPoseVar, 'pickPoseVar'),
            
            # Divide delta evenly
            Function(['ConfDelta', 'PoseDelta'], ['GraspDelta'],
                      halveVariance, 'halveVar'),

            # Values for what is in the other
            Function(['OObj', 'OFace', 'OGraspMu', 'OGraspVar', 'OGraspDelta'],
                       ['OtherHand', 'Obj'], genGraspStuffHand,
                       'genGraspStuffHand'),

            # Not modeling the fact that the object's shadow should
            # grow a bit as we move to pick it.   Build that into pickGen.

            # Generate object pose and two confs
            Function(['Pose', 'PoseFace', 'PickConf', 'PreConf'],
                     ['Obj', 'GraspFace', 'GraspMu',
                      'PoseVar', 'RealGraspVar', 'PoseDelta', 'ConfDelta',
                      'GraspDelta', 'Hand', probForGenerators],
                     pickGen, 'pickGen')
            ],
        cost = pickCostFun,
        f = pickBProgress,
        prim = pickPrim,
        argsToPrint = [0, 1, 11, 12],
        ignorableArgs = range(2, 27))  # pays attention to pose

# P2 goes into success prob (cost)
# Consider reducing prob without increasing var
lookAt = Operator(\
    'LookAt',
    ['Obj', 'LookConf', 'PoseFace', 'Pose',
     'PoseVarBefore', 'PoseDelta', 'PoseVarAfter',
     'P1', 'P2', 'PR1', 'PR2'],
    # Pre
    {0: {Bd([SupportFace(['Obj']), 'PoseFace', 'P1'], True),
         B([Pose(['Obj', 'PoseFace']), 'Pose', 'PoseVarBefore', 'PoseDelta',
                 'P1'], True)},
     1: {Bd([CanSeeFrom(['Obj', 'Pose', 'PoseFace', 'LookConf', []]),
             True, 'P2'], True),
         Conf(['LookConf', lookConfDelta], True)}},
    # Results
    [({B([Pose(['Obj', 'PoseFace']), 'Pose', 'PoseVarAfter', 'PoseDelta',
         'PR1'],True),
       Bd([SupportFace(['Obj']), 'PoseFace', 'PR2'], True)}, {})
       ],
    # Functions
    functions = [\
        Function(['P2'], [[canSeeProb]], assign, 'assign'),
        # Look increases probability.  For now, just a fixed amount.  Ugly.
        Function(['P1'], ['PR1', 'PR2'], obsModeProb, 'obsModeProb'),
        # How confident do we need to be before the look?
        Function(['PoseVarBefore'], ['PoseVarAfter', 'Obj', 'PoseFace'],
                genLookObjPrevVariance, 'genLookObjPrevVariance'),
        Function(['LookConf'],
                 ['Obj', 'Pose', 'PoseFace', 'PoseVarBefore', 'PoseDelta',
                         lookConfDelta, probForGenerators],
                 lookGen, 'lookGen')
        ],
    cost = lookAtCostFun,
    f = lookAtBProgress,
    prim = lookPrim,
    argsToPrint = [0, 1],
    ignorableArgs = range(1, 11))

lookAtHand = Operator(\
    'LookAtHand',
    ['Obj', 'Hand', 'LookConf', 'GraspFace', 'Grasp',
     'GraspVarBefore', 'GraspDelta', 'GraspVarAfter',
     'P1', 'PR1', 'PR2', 'PR3'],
    # Pre
    {0: {Bd([GraspFace(['Obj', 'Hand']), 'GraspFace', 'P1'], True),
         Bd([Holding(['Hand']), 'Obj', 'P1'], True),
         B([Grasp(['Obj', 'Hand', 'GraspFace']), 'Grasp', 'GraspVarBefore',
            'GraspDelta', 'P1'], True)},
     1: {Conf(['LookConf', lookConfDelta], True)}},
    # Results
    [({B([Grasp(['Obj', 'Hand', 'GraspFace']), 'Grasp', 'GraspVarAfter',
         'GraspDelta', 'PR1'], True), 
      Bd([GraspFace(['Obj', 'Hand']), 'GraspFace', 'PR3'], True)}, {}),
     ({Bd([Holding(['Hand']), 'Obj', 'PR2'], True)}, {})],
    # Functions
    functions = [\
        # Look increases probability.  For now, just a fixed amount.  Ugly.
        Function(['P1'], ['PR1', 'PR2'], obsModeProb, 'obsModeProb'),
        # How confident do we need to be before the look?
        # Fix this to be grasp specific (was written for poses)
        Function(['GraspVarBefore'],
                 ['GraspVarAfter', 'Hand', 'Obj', 'GraspFace'],
                 genLookObjHandPrevVariance, 'genLookObjHandPrevVariance'),
        Function(['LookConf'],
                 ['Obj', 'Hand', 'GraspFace', 'Grasp', 'GraspVarBefore',
                  'GraspDelta', probForGenerators],
                 lookHandGen, 'lookHandGen')
        ],
    cost = lookAtHandCostFun,
    f = lookAtHandBProgress,
    prim = lookHandPrim,
    argsToPrint = [0, 1],
    ignorableArgs = range(1, 11)) 

# This is an inference step that is true for all pairs of a
# conditional fluent and a primitive one.
# FC(. | p) and p -> FC(.)
# If p is well chosen, then we think FC(. |p) will have lower heuristic value,
# than p

# We have to do this because FC is an implicit predicate and we
# haven't asserted all of the ways in which it can be made true by
# primitive actions.
#
# We are asserting that making p true is a kind of primitive action
# that can make this fluent become true.  And, now, because there are
# infinitely many such p's, we need generators to cause us to consider
# reasonable ones in a reasonable order.

# In these particular cases, there is inductive reasoning going on:
# moving this object won't make the region clear, but (we assert) it
# will make it closer to being clear.

# Would it be possible to write this completely generically?  In some
# sense, we need a generator that, given an FC will give us a P.  In
# principle possible, but kind of a big change to rule syntax.  For
# now, make one for each pair of conditional fluent FC and primitive
# fluent P.  Note that the action is irrelevant.  So, moveObjToAchReachable
# and lookObjToAchReachable have the some FC and P, so they can be
# folded into a single ChangePoseToMakeReachable, which might ask to
# change the mean or the variance of a pose dist.

poseAchCanReach = Operator(\
    'PoseAchCanReach',
    ['CEnd', 'Hand', 
     'LObj', 'LFace', 'LGraspMu', 'RealGraspVarL', 'LGraspDelta',
     'RObj', 'RFace', 'RGraspMu', 'RealGraspVarR', 'RGraspDelta',
     'PreCond', 'PostCond',
     'Occ', 'PoseFace', 'Pose', 'PoseVar', 'PoseDelta',
     'P1', 'P2', 'PR'],

    {0: {},
     1: {Bd([CanReachHome(['CEnd', 'Hand', 
                   'LObj', 'LFace', 'LGraspMu', 'RealGraspVarL', 'LGraspDelta',
                   'RObj', 'RFace', 'RGraspMu', 'RealGraspVarR', 'RGraspDelta',
                           'PreCond']),  True, 'P1'], True),
         Bd([SupportFace(['Occ']), 'PoseFace', 'P2'], True),
         B([Pose(['Occ', 'PoseFace']), 'Pose', 'PoseVar', 'PoseDelta', 'P2'],
            True)}},
    # Result
    [({Bd([CanReachHome(['CEnd', 'Hand',
                   'LObj', 'LFace', 'LGraspMu', 'RealGraspVarL', 'LGraspDelta',
                   'RObj', 'RFace', 'RGraspMu', 'RealGraspVarR', 'RGraspDelta',
                           'PostCond']),  True, 'PR'], True)}, {})],

    # Functions
    functions = [\
        # Compute precond probs
        Function(['P1', 'P2'], ['PR'], regressProb(2), 'regressProb2'),
        # Call generator
        Function(['Occ', 'Pose', 'PoseFace', 'PoseVar', 'PoseDelta'],
                  ['CEnd', 'Hand',
                   'LObj', 'LFace', 'LGraspMu', 'RealGraspVarL', 'LGraspDelta',
                   'RObj', 'RFace', 'RGraspMu', 'RealGraspVarR', 'RGraspDelta',
                   'P1', 'PostCond'], canReachGen, 'canReachGen'),
         # Add the appropriate condition
         Function(['PreCond'],
                  ['PostCond',
                   'Occ', 'PoseFace', 'Pose', 'PoseVar', 'PoseDelta', 'P2'],
                  addPosePreCond, 'addPosePreCond')],
    cost = lambda al, args, details: 0.1,
    argsToPrint = [0, 14, 16],
    ignorableArgs = range(1, 14) + range(17,22))  
    
poseAchCanPickPlace = Operator(\
    'PoseAchCanPickPlace',
    ['PreConf', 'PlaceConf', 'Hand', 'Obj', 'Pose',
                          'RealPoseVar', 'PoseDelta', 'PoseFace',
                          'GraspFace', 'GraspMu', 'GraspVar', 'GraspDelta',
                          'OObj', 'OFace', 'OGraspMu', 'OGraspVar', 
                          'OGraspDelta', 'PreCond', 'PostCond',
     'Occ', 'OccPose', 'OccPoseFace', 'OccPoseVar', 'OccPoseDelta',
     'P1', 'P2', 'PR'],
    {0: {},
     1: {Bd([CanPickPlace(['PreConf', 'PlaceConf', 'Hand', 'Obj', 'Pose',
                          'RealPoseVar', 'PoseDelta', 'PoseFace',
                          'GraspFace', 'GraspMu', 'GraspVar', 'GraspDelta',
                          'OObj', 'OFace', 'OGraspMu', 'OGraspVar', 
                          'OGraspDelta', 'PreCond']), True, 'P1'],True),
         Bd([SupportFace(['Occ']), 'OccPoseFace', 'P2'], True),
         B([Pose(['Occ', 'OccPoseFace']), 'OccPose', 'OccPoseVar',
                  'OccPoseDelta', 'P2'],
            True)}},
    # Result
    [({Bd([CanPickPlace(['PreConf', 'PlaceConf', 'Hand', 'Obj', 'Pose',
                          'RealPoseVar', 'PoseDelta', 'PoseFace',
                          'GraspFace', 'GraspMu', 'GraspVar', 'GraspDelta',
                          'OObj', 'OFace', 'OGraspMu', 'OGraspVar', 
                          'OGraspDelta', 'PostCond']), True, 'PR'],True)}, {})],
    # Functions
    functions = [\
        # Compute precond probs
        Function(['P1', 'P2'], ['PR'], regressProb(2), 'regressProb2'),
        # Call generator
        Function(['Occ', 'OccPose', 'OccPoseFace', 'OccPoseVar','OccPoseDelta'],
                  ['PreConf', 'PlaceConf', 'Hand', 'Obj', 'Pose',
                          'RealPoseVar', 'PoseDelta', 'PoseFace',
                          'GraspFace', 'GraspMu', 'GraspVar', 'GraspDelta',
                          'OObj', 'OFace', 'OGraspMu', 'OGraspVar', 
                          'OGraspDelta', 'P1', 'PostCond'],
                          canPickPlaceGen, 'canPickPlaceGen'),
         # Add the appropriate condition
         Function(['PreCond'],
                  ['PostCond',
                   'Occ', 'OccPoseFace', 'OccPose', 'OccPoseVar',
                   'OccPoseDelta', 'P2'],
                  addPosePreCond, 'addPosePreCond')],
    cost = lambda al, args, details: 0.1,
    argsToPrint = [3, 2, 4, 19, 20],
    ignorableArgs = range(0, 2) + range(5, 27))

# Need also graspAchCanReachHome

graspAchCanPickPlace = Operator(\
    'GraspAchCanPickPlace',
    ['PreConf', 'PlaceConf', 'Hand', 'Obj', 'Pose',
                          'RealPoseVar', 'PoseDelta', 'PoseFace',
                          'GraspFace', 'GraspMu', 'GraspVar', 'GraspDelta',
                          'OObj', 'OFace', 'OGraspMu', 'OGraspVar', 
                          'OGraspDelta', 'Cond',
                          'PreGraspVar', 'P1', 'P2', 'PR'],
    {0: {},
     1: {Bd([CanPickPlace(['PreConf', 'PlaceConf', 'Hand', 'Obj', 'Pose',
                          'RealPoseVar', 'PoseDelta', 'PoseFace',
                          'GraspFace', 'GraspMu', 'PreGraspVar', 'GraspDelta',
                          'OObj', 'OFace', 'OGraspMu', 'OGraspVar', 
                          'OGraspDelta', 'Cond']), True, 'P1'],True),
         Bd([GraspFace(['Obj', 'Hand']), 'GraspFace', 'P2'], True),
         Bd([Holding(['Hand']), 'Obj', 'P2'], True),
         B([Grasp(['Obj', 'Hand', 'GraspFace']),
             'GraspMu', 'PreGraspVar', 'GraspDelta', 'P2'], True)}},
    # Result
    [({Bd([CanPickPlace(['PreConf', 'PlaceConf', 'Hand', 'Obj', 'Pose',
                          'RealPoseVar', 'PoseDelta', 'PoseFace',
                          'GraspFace', 'GraspMu', 'GraspVar', 'GraspDelta',
                          'OObj', 'OFace', 'OGraspMu', 'OGraspVar', 
                          'OGraspDelta', 'Cond']), True, 'PR'],True)}, {})],
    # Functions
    functions = [\
        # Compute precond probs
        Function(['P1', 'P2'], ['PR'], regressProb(2), 'regressProb2'),
        # Call generator, just to see if reducing graspvar would be useful
        Function(['PreGraspVar'],['Obj', 'GraspVar'],
                   graspVarCanPickPlaceGen, 'graspVarCanPickPlaceGen')],
    cost = lambda al, args, details: 0.1,
    argsToPrint = [3, 2, 4, 18],
    ignorableArgs = range(0, 2) + range(5, 22))

poseAchCanSee = Operator(\
    'PoseAchCanSee',
    ['Obj', 'TargetPose', 'TargetPoseFace', 'TargetPoseVar', 'TargetPoseDelta',
     'Occ', 'OccPoseFace', 'OccPose', 'OccPoseVar', 'OccPoseDelta',
     'LookConf', 'ConfDelta', 'PreCond', 'PostCond',
      'P1', 'P2', 'PR'],

    {0: {Bd([CanSeeFrom(['Obj', 'TargetPose', 'TargetPoseFace', 'LookConf',
                         'PreCond']),
             True, 'P1'], True)},
     1: {Bd([SupportFace(['Occ']), 'OccPoseFace', 'P2'], True),
         B([Pose(['Occ', 'OccPoseFace']), 'OccPose', 'OccPoseVar',
            'OccPoseDelta', 'P2'], True)}},
    # Result
    [({Bd([CanSeeFrom(['Obj', 'TargetPose', 'TargetPoseFace', 'LookConf', 
                           'PostCond']),  True, 'PR'], True)}, {})],
    # Functions
    functions = [\
        # Compute precond probs
        Function(['P1', 'P2'], ['PR'], regressProb(2), 'regressProb2'),
        # Only want to see the mean, assume robot at conf
        Function(['TargetPoseVar', 'TargetPoseDelta', 'ConfDelta'], [],
                 lambda a, c, b, o: [[(0.0,)*4, (0.0,)*4, (0.0,)*4]], 'zeros'),
        # Call generator
        Function(['Occ', 'OccPose', 'OccPoseFace', 'OccPoseVar','OccPoseDelta'],
                  ['Obj', 'TargetPose', 'TargetPoseFace', 'TargetPoseVar',
                   'TargetPoseDelta', 'LookConf', 'ConfDelta', 'P2'],
                    canSeeGen, 'canSeeGen'),
         # Add the appropriate condition
         Function(['PreCond'],
                  ['PostCond',
                   'Occ', 'OccPoseFace', 'OccPose', 'OccPoseVar',
                   'OccPoseDelta', 'P1'],
                  addPosePreCond, 'addPosePreCond')],
    cost = lambda al, args, details: 0.1)


######################################################################
#
# Operators only used in the heuristic
#
######################################################################

magicRegraspCost = 10

hRegrasp = Operator(\
        'HeuristicRegrasp',
        ['Obj', 'Hand', 'GraspFace', 'GraspMu', 'GraspVar', 'GraspDelta',
         'PrevGraspFace', 'PrevGraspMu', 'PrevGraspVar', 'PrevGraspDelta',
         'PR1', 'PR2', 'PR3'],

        # Pre
        {0 : {Bd([Holding(['Hand']), 'Obj', 'PR1'], True),
              Bd([GraspFace(['Obj', 'Hand']), 'PrevGraspFace', 'PR2'], True),
              B([Grasp(['Obj', 'Hand', 'PrevGraspFace']),
                  'PrevGraspMu', 'GraspVar', 'GraspDelta', 'PR3'], True)}},

        # Results
        [({Bd([Holding(['Hand']), 'Obj', 'PR1'], True), 
           Bd([GraspFace(['Obj', 'Hand']), 'GraspFace', 'PR2'], True),
           B([Grasp(['Obj', 'Hand', 'GraspFace']),
            'GraspMu', 'GraspVar', 'GraspDelta', 'PR3'], True)}, {})],

        # Functions
        functions = [\
            # Be sure obj is not none
            Function([], ['Obj'], notNone, 'notNone', True),
            Function(['PrevGraspFace', 'PrevGraspMu', 'PrevGraspVar',
                      'PrevGraspDelta'],
                      ['Obj', 'Hand'], easyGraspGen, 'easyGraspGen'),
            # Only use to change grasp.
            Function([], ['GraspMu', 'GraspFace',
                          'PrevGraspMu', 'PrevGraspFace'],
                     notEqual2, 'notEqual2', True),
            ],
        cost = lambda al, args, details: magicRegraspCost,
        argsToPrint = [0, 1],
        ignorableArgs = range(2, 12))  # pays attention to pose
