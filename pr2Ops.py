import numpy as np
import util
from itertools import product, permutations, chain, imap
from dist import DeltaDist, varBeforeObs, DDist, probModeMoved, MixtureDist,\
     UniformDist, chiSqFromP, MultivariateGaussianDistribution
import fbch
from fbch import Function, getMatchingFluents, Operator, simplifyCond
from miscUtil import isVar, prettyString, makeDiag, isGround, argmax
from planUtil import PoseD, ObjGraspB, ObjPlaceB, Violations
from pr2Util import shadowName, shadowWidths
from pr2Gen import pickGen, lookGen, canReachGen,canSeeGen,lookHandGen, easyGraspGen, canPickPlaceGen, placeInRegionGen, placeGen
from belief import Bd, B
from pr2Fluents import Conf, CanReachHome, Holding, GraspFace, Grasp, Pose,\
     SupportFace, In, CanSeeFrom, Graspable, CanPickPlace,\
     findRegionParent, CanReachNB, BaseConf, BLoc, canReachHome, canReachNB
import planGlobals as glob
from planGlobals import debugMsg, debug, useROS
import pr2RRT as rrt
from pr2Visible import visible
import windowManager3D as wm

zeroPose = zeroVar = (0.0,)*4
awayPose = (100.0, 100.0, 0.0, 0.0)
maxVarianceTuple = (.1,)*4

# If it's bigger than this, we can't just plan to look and see it
# Should be more subtle than this...
maxPoseVar = (0.05**2, 0.05**2, 0.05**2, 0.1**2)

# Fixed accuracy to use for some standard preconditions
canPPProb = 0.9
otherHandProb = 0.9
canSeeProb = 0.9
# No prob can go above this
maxProbValue = 0.98  # was .999
# How sure do we have to be of CRH for moving
movePreProb = 0.98
movePreProb = 0.8
# Prob for generators.  Keep it high.   Should this be = maxProbValue?
probForGenerators = 0.98

# Generic large values for the purposes of planning
planVar = (0.04**2, 0.04**2, 0.04**2, 0.08**2)
planP = 0.95

hands = (0, 1)
handName = ('left', 'right')
handI = {'left' : 0, 'right' : 1}

######################################################################
#
# Prim functions map an operator's arguments to some parameters that
#  are used during execution
#
######################################################################

tryDirectPath = False
# canReachHome(conf) returns a path from conf to home!
def primPath(bs, cs, ce, p):
    home = bs.getRoadMap().homeConf
    path = None
    if tryDirectPath:
        path, viols = canReachHome(bs, cs, p, Violations(),
                                   homeConf=ce, optimize=True)
    if (not path) or viols.weight() > 0:
        path1, v1 = canReachHome(bs, cs, p, Violations(), optimize=True)
        if (not path1) or v1.weight() > 0:
            print 'Path1 failed, trying RRT'
            path1, v1 = rrt.planRobotPathSeq(bs, p, home, cs, None,
                                             maxIter=50, failIter=10)
            if (not path1) or v1.weight() > 0:
                print 'Path1 RRT failed, trying full RRT'
                path, viols = rrt.planRobotPathSeq(bs, p, cs, ce, None,
                                                   maxIter=50, failIter=10)            
                assert path and viols.weight() == 0
        if (not path) and path1:
            path2, v2 = canReachHome(bs, ce, p, Violations(),
                                     optimize=True, reversePath=True)
            if (not path2) or v2.weight() > 0:
                print 'Path2 failed, trying RRT'
                path2, v2 = rrt.planRobotPathSeq(bs, p, home, ce, None,
                                                 maxIter=50, failIter=10)                
            if (not path2) or v2.weight() > 0:
                print 'Path2 RRT failed, trying full RRT'
                path, viols = rrt.planRobotPathSeq(bs, p, cs, ce, None,
                                                   maxIter=50, failIter=10)            
                assert path and viols.weight == 0
        if (not path) and path1 and path2:
            path = path1 + path2[::-1]
    else:
        print 'Direct path succeeded'

    smoothed = bs.getRoadMap().smoothPath(path, bs, p)
    interpolated = interpolate(smoothed)
    verifyPaths(bs, p, path, smoothed, interpolated)
    return smoothed, interpolated

def primNBPath(bs, cs, ce, p):
    path, v = canReachNB(bs, cs, ce, p, Violations())
    if not path:
        print 'NB Path failed, trying RRT'
        path, v = rrt.planRobotPathSeq(bs, p, cs, ce, None,
                                       maxIter=50, failIter=10)
    assert path
    if v.weight() > 0:
        raw_input('Potential collision in primitive path')
    else:
        print 'Success'
    smoothed = bs.getRoadMap().smoothPath(path, bs, p)
    interpolated = interpolate(smoothed)
    verifyPaths(bs, p, path, smoothed, interpolated)
    return smoothed, interpolated

def interpolate(path):
    interpolated = []
    for i in range(1, len(path)):
        qf = path[i]
        qi = path[i-1]
        confs = rrt.interpolate(qf, qi, stepSize=0.25)
        print i, 'path segment has', len(confs), 'confs'
        interpolated.extend(confs)
    return interpolated

def verifyPath(pbs, prob, path, msg):
    print 'Verifying', msg, 'path'
    shWorld = pbs.getShadowWorld(prob)
    attached = shWorld.attached
    pbsDrawn = False
    for conf in path:
        viol = pbs.getRoadMap().confViolations(conf, pbs, prob)
        pbs.draw(prob, 'W')
        if not viol or viol.weight() > 0:
            print msg, 'path', viol
            conf.draw('W', 'red', attached=attached)
        else:
            conf.draw('W')            
        raw_input('Ok?')

def verifyPaths(bs, p, path, smoothed, interpolated):
    if debug('verifyPath'):
        verifyPath(bs, p, path, 'original')
        verifyPath(bs, p, interpolate(path), 'original interpolated')
        verifyPath(bs, p, smoothed, 'smoothed')
        verifyPath(bs, p, interpolated, 'smoothed interpolated')


def moveNBPrim(args, details):
    (base, cs, ce, cd) = args

    bs = details.pbs.copy()
    # Make all the objects be fixed
    bs.fixObjBs.update(bs.moveObjBs)
    bs.moveObjBs = {}
    print 'moveNBPrim (start, end)'
    printConf(cs); printConf(ce)

    path, interpolated = primNBPath(bs, cs, ce, movePreProb)
    if debug('prim'):
        print '*** movePrim No base'
        print list(enumerate(zip(vl, args)))
        print 'path length', len(path)
    assert path
    return path, interpolated, details.pbs.getPlacedObjBs()

def movePrim(args, details):
    (cs, ce, cd) = args

    bs = details.pbs.copy()
    # Make all the objects be fixed
    bs.fixObjBs.update(bs.moveObjBs)
    bs.moveObjBs = {}
    print 'movePrim (start, end)'
    printConf(cs); printConf(ce)

    path, interpolated = primPath(bs, cs, ce, movePreProb)
    if debug('prim'):
        print '*** movePrim'
        print list(enumerate(zip(vl, args)))
        print 'path length', len(path)
    assert path
    # path(s) and the distributions for the placed objects, to guide looking
    return path, interpolated, details.pbs.getPlacedObjBs()

def printConf(conf):
    cart = conf.cartConf()
    pose = cart['pr2LeftArm'].pose(fail=False)
    if pose:
        hand = str(np.array(pose.xyztTuple()))
    else:
        hand = '\n'+str(cart['pr2LeftArm'].matrix)
    print 'base', conf['pr2Base'], 'hand', hand
    
def pickPrim(args, details):
    (o,h,pf,p,pd,gf,gm,gv,gd,prc,cd,pc,rgv,pv,p1,pr1,pr2,pr3) = args
    bs = details.pbs.copy()
    print 'plckPrim (start, end)'
    # !! Should close the fingers as well?
    if debug('prim'):
        print '*** pickPrim'
        print list(enumerate(zip(vl, args)))
    return details.pbs.getPlacedObjBs()

def lookPrim(args, details):
    # In the real vision system, we might pass in a more general
    # structure with all the objects (and/or types) we expect to see
    if debug('prim'):
        print '*** lookPrim'
        print list(enumerate(zip(vl, args)))

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
        print list(enumerate(zip(vl, args)))

    # The distributions for the grasped objects, to guide looking
    return details.pbs.graspB
    
def placePrim(args, details):
    (o, h, pf, p, pv, rpv, pd,
     gf, gm, gv, gd,
     prc, cd, pc, ar,
     pr1, pr2, pr3, p1) = args

    bs = details.pbs.copy()
    print 'placePrim (start, end)'

    # !! Should open the fingers as well
    if debug('prim'):
        print '*** placePrim'
        print list(enumerate(zip(vl, args)))

    return details.pbs.getPlacedObjBs()


################################################################
## Simple generators
################################################################

# Relevant fluents:
#  Holding(hand), GraspFace(obj, hand), Grasp(obj, hand, face)

smallDelta = (10e-4,)*4

def oh(h):
    return 'left' if h == 'right' else 'right'

def getObj(args, goal, start, stuff):
    (h,) = args
    heldLeft = start.pbs.getHeld('left').mode()        
    heldRight = start.pbs.getHeld('right').mode()        
    result = []
    hh = heldLeft if h == 'left' else heldRight
    if hh == 'none' or h == 'right' and not start.pbs.useRight:
        # If there is nothing in the hand right now, then we
        # should technically iterate through all possible objects.
        # For now, fail.
        return []
    else:
        return [[hh]]

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

# See if would be useful to look at obj in order to reduce its variance
def graspVarCanPickPlaceGen(args, goal, start, vals):
    (obj, variance) = args
    if obj != 'none' and variance[0] > start.domainProbs.obsVarTuple[0]:
        return [[start.domainProbs.obsVarTuple]]
    else:
        return []

# Just return the base pose
def getBase(args, goal, start, vals):
    (conf,) = args
    if isVar(conf):
        return []
    else:
        return [[conf['pr2Base']]]

# LPK: make this more efficient by storing the inverse mapping
def regionParent(args, goal, start, vals):
    [region] = args
    if isVar(region):
        # Trying to place at a pose.  Doesn't matter.
        return [['none']]
    return [[findRegionParent(start, region)]]

def poseInStart(args, goal, start, vals):
    [obj] = args
    pbs = start.pbs
    pd = pbs.getPlaceB(obj)
    face = pd.support.mode()
    mu = pd.poseD.mu.xyztTuple()
    debugMsg('poseInStart', ('->', (face, mu)))
    return [(face, mu)]

# Use this when we don't want to generate an argument (expecting to
# get it from goal bindings.)  Guaranteed to fail if that var isn't
# already bound.
def genNone(args, goal, start, vals):
    return None

def assign(args, goal, start, vals):
    return args

# Be sure the argument is not 'none'
def notNone(args, goal, start, vals):
    assert args[0] != None
    if args[0] == 'none':
        return None
    else:
        return [[]]

# Be sure the argument is not '*'
def notStar(args, goal, start, vals):
    if args[0] == '*':
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

# It pains me that this has to exist; but the args needs to be a list
# not a structure of variables.
def notEqual3(args, goal, start, vals):
    if (args[0], args[1], args[2]) == (args[3], args[4], args[5]):
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

# Subtract
def subtract((a, b), goal, start, vals):
    ans = tuple([aa - bb for (aa, bb) in zip(a, b)])
    if any([x <=  0.0 for x in ans]):
        debugMsg('smallDelta', 'Delta would be negative or zero', ans)
        return []
    return [[ans]]
        
# Return as many values as there are args; overwrite any that are
# variables with the minimum value
def minP(args, goal, start, vals):
    minVal = min([a for a in args if not isVar(a)])
    return [[minVal if isVar(a) else a for a in args]]

# Regression:  what does the mode need to be beforehand, assuming a good
# outcome.  Don't let it go down too fast...
def obsModeProb(args, goal, start, vals):
    p = max([a for a in args if not isVar(a)])
    pFalsePos = pFalseNeg = start.domainProbs.obsTypeErrProb
    pr = p * pFalsePos / ((1 - p) * (1 - pFalseNeg) + p * pFalsePos)
    return [[max(0.1, pr, p - 0.2)]]

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

def maxGraspVarFun((var,), goal, start, vals):
    assert not(isVar(var))
    maxGraspVar = start.domainProbs.maxGraspVar

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

def moveConfDelta(args, goal, start, vals):
    return [[start.domainProbs.moveConfDelta]]
def defaultPlaceDelta(args, goal, start, vals):
    return [[start.domainProbs.placeDelta]]
def defaultGraspDelta(args, goal, start, vals):
    return [[start.domainProbs.graspDelta]]
def obsVar(args, goal, start, vals):
    return [[start.domainProbs.obsVar]]

def realPoseVar((graspVar,), goal, start, vals):
    placeVar = start.domainProbs.placeVar
    return [[tuple([gv+pv for (gv, pv) in zip(graspVar, placeVar)])]]

def placeInPoseVar(args, goal, start, vals):
    pv = [v * 2 for v in start.domainProbs.obsVarTuple]
    pv[2] = pv[0]
    return [[tuple(pv)]]

# Thing is a variance; compute a variance that corresponds to doubling
# the stdev.    (sqrt(v) * 2)^2 = v * 4
def stdevTimes2((thing,), goal, start, vals):
    return [[tuple([v*4 for v in thing])]]

def times2((thing,), goal, start, vals):
    return [[tuple([v*2 for v in thing])]]

# For place, grasp var is desired poseVar minus fixed placeVar
# Don't let it be bigger than maxGraspVar 
def placeGraspVar((poseVar,), goal, start, vals):
    maxGraspVar = start.domainProbs.maxGraspVar
    placeVar = start.domainProbs.placeVar
    graspVar = tuple([min(gv - pv, m) for (gv, pv, m) \
                      in zip(poseVar, placeVar, maxGraspVar)])
    if any([x <= 0 for x in graspVar]):
        if debug('placeVar'):
            print 'negative grasp var'
            print 'poseVar', poseVar
            print 'placeVar', placeVar
            print 'maxGraspVar', maxGraspVar
            raw_input('poseVar - placeVar is negative')
        return []
    else:
        return [[graspVar]]

# tol > n * sqrt(var) + d
# tol - d > n * sqrt(var)
# (tol - d) / n > sqrt(var)
# ((tol - d) / n)**2 > var

# For pick, pose var is desired graspVar minus fixed pickVar
def pickPoseVar((graspVar, graspDelta, prob), goal, start, vals):
    pickVar = start.domainProbs.pickVar
    pickTolerance = start.domainProbs.pickTolerance
    # What does the variance need to be so that we are within
    # pickTolerance with probability prob?
    numStdDevs =  np.sqrt(chiSqFromP(1-prob, 3))
    # nstd * std < pickTol
    # std < pickTol / nstd
    tolerableVar = [((pt - gd - .001) / numStdDevs)**2 for \
                    (pt, gd) in zip(pickTolerance, graspDelta)]
    poseVar = tuple([min(gv - pv, tv) \
                     for (gv, pv, tv) in zip(graspVar, pickVar, tolerableVar)])

    if debug('pickGenVar'):
        print 'pick pose var'
        print 'fixed pickVar', pickVar
        print 'tolerance', pickTolerance
        print 'num stdev', numStdDevs
        print 'tolerable var', tolerableVar
        print 'poseVar', poseVar
        print 'shadow width', shadowWidths(poseVar, graspDelta, prob)
        raw_input('okay?')
                     
    if any([x <= 0 for x in poseVar]):
        debugMsg('pickGenVar', 'pick pose var negative', poseVar)
        return []
    else:
        debugMsg('pickGenVar', 'pick pose var', poseVar)
        return [[poseVar]]


# starting var if it's legal, plus regression of the result var.
# Need to try several, so that looking doesn't put the robot into the shadow!
def genLookObjPrevVariance((ve, obj, face), goal, start, vals):
    lookVar = start.domainProbs.obsVarTuple
    vs = list(start.poseModeDist(obj, face).mld().sigma.diagonal().tolist()[0])
    vs[2] = .0001**2
    vs = tuple(vs)

    # Don't let variance get bigger than variance in the initial state, or
    # the cap, whichever is bigger
    cap = [max(a, b) for (a, b) in zip(maxPoseVar, vs)]

    vbo = varBeforeObs(lookVar, ve)
    cappedVbo1 = tuple([min(a, b) for (a, b) in zip(cap, vbo)])
    cappedVbo2 = tuple([min(a, b) for (a, b) in zip(vs, vbo)])

    # vbo > ve
    # This is useful if it's between:  vbo > vv > ve
    def useful(vv):
        return any([a > b for (a, b) in zip(vv, ve)]) and \
               any([a > b for (a, b) in zip(cappedVbo1, vv)])

    def sqrts(vv):
        return [np.sqrt(xx) for xx in vv]

    result = [[cappedVbo1]]

    v4 = tuple([v / 4.0 for v in cappedVbo1])
    v9 = tuple([v / 9.0 for v in cappedVbo1])
    v25 = tuple([v / 25.0 for v in cappedVbo1])
    ov = lookVar
        
    if useful(v4): result.append([v4])
    if useful(v9): result.append([v9])
    if useful(v25): result.append([v25])
    if useful(ov): result.append([ov])

    if cappedVbo2 != cappedVbo1:
        result.append([cappedVbo2])
    if vs != cappedVbo1 and vs != cappedVbo2:
        result.append([vs])

    if debug('varBeforeObs'):
        print '@@@@@@@@@@@@@ Prev var'
        print 'Target', prettyString(sqrts(ve))
        print 'Capped before', prettyString(sqrts(cappedVbo1))
        print 'Other suggestions'
        for xx in result: print '   ', prettyString(sqrts(xx)[0])

    debugMsg('genLookObjPrevVariance', result)

    return result

def realPoseVarAfterObs((varAfter,), goal, start, vals):
    obsVar = start.domainProbs.obsVarTuple
    thing = tuple([min(x, y) for (x, y) in zip(varAfter, obsVar)])
    return [[thing]]

# starting var if it's legal, plus regression of the result var
def genLookObjHandPrevVariance((ve, hand, obj, face), goal, start, vals):
    epsilon = 10e-5
    lookVar = start.domainProbs.obsVarTuple
    maxGraspVar = start.domainProbs.maxGraspVar

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

# Add a condition on the pose and face of an object
def addGraspPreCond((postCond, hand, obj, graspFace, grasp,
                     graspVar, graspDelta, p),
                   goal, start, vals):
    newFluents = [Bd([Holding([hand]), obj, p], True),
                  Bd([GraspFace([obj, hand]), graspFace, p], True),
                  B([Grasp([obj, hand, graspFace]), grasp, graspVar,
                     graspDelta, p], True)]
    fluentList = simplifyCond(postCond, newFluents)
    return [[fluentList]]

# Add a condition on the pose and face of an object
def addDropPreCond((postCond, hand, p),
                   goal, start, vals):
    newFluents = [Bd([Holding([hand]), 'none', p], True)]
    fluentList = simplifyCond(postCond, newFluents)
    return [[fluentList]]

def awayRegion(args, goal, start, vals):
    return [[start.pbs.awayRegions()]]

# Really just return true if reducing variance on the object in the
# hand will reduce the violations.
#####    Fix this!!!
def canReachHandGen(args, goal, start, vals):
    (conf, fcp, p, cond, hand) = args
    f = CanReachHome([conf, fcp, cond], True)
    path, viol = f.getViols(start, True, p)
    if viol and viol.heldShadows[handI[hand]] != []:
        return [[]]
    else:
        return []

# Really just return true if putting down the object in the
# hand will reduce the violations.
def canReachDropGen(args, goal, start, vals):
    (conf, fcp, p, cond) = args
    f = CanReachHome([conf, fcp, cond], True)
    result = []
    path, viol = f.getViols(start, True, p)
    for hand in ('left', 'right'):
        if viol:
            collidesWithHeld = viol.heldObstacles[handI[hand]]
            if len(collidesWithHeld) > 0:
                heldO = start.pbs.held[hand].mode()
                assert heldO != 'none'
                fbs = fbch.getMatchingFluents(goal,
                               Bd([Holding([hand]), 'Obj', 'P'], True))
                # should be 0 or 1 object names
                matches = [b['Obj'] for (f, b) in fbs if isGround(b.values)]
                if heldO != 'none' and len(matches) == 0 or matches == ['none']:
                    # Holding none is consistent with the goal
                    result.append([hand])
    return result
        
# Really just return true if putting down the object in the
# hand will reduce the violations.
def canPickPlaceDropGen(args, goal, start, vals):
    (preConf, placeConf, hand, obj, pose, poseVar, poseDelta, poseFace,
     graspFace, graspMu, graspVar, graspDelta, op, cond, p) = args
    f = CanPickPlace(args[:-1], True)
    result = []
    path, viol = f.getViols(start, True, p)
    for dropHand in ('left', 'right'):
        if viol:
            # objects that collide with the object in the hand
            collidesWithHeld = viol.heldObstacles[handI[dropHand]]
            if len(collidesWithHeld) > 0:  # maybe make sure not shadow
                heldO = start.pbs.held[dropHand].mode()
                fbs = fbch.getMatchingFluents(goal,
                               Bd([Holding([dropHand]), 'Obj', 'P'], True))
                # should be 0 or 1 object names
                matches = [b['Obj'] for (f, b) in fbs if isGround(b.values())]
                if heldO != obj and heldO != 'none' and \
                   (len(matches) == 0 or matches == ['none']):
                    print 'canPickPlaceDropGen dropping', heldO, dropHand
                    print 'held obstacles', collidesWithHeld
                    print 'goal held', matches
                    raw_input('okay?')
                    result.append([dropHand])
    return result

################################################################
## Special regression funs
################################################################

attenuation = 0.5
# During regression, apply to all fluents in goal;  returns f or a new fluent.
def moveSpecialRegress(f, details, abstractionLevel):

    # Only model these effects at the lower level of abstraction.
    if abstractionLevel == 0:
        return f.copy()

    # Assume that odometry error is controlled during motion, so not more than 
    # this.  It's a stdev
    odoError = details.domainProbs.odoError
    odoVar = [e * e for e in odoError]    # Variance due to odometry after move

    if f.predicate == 'B' and f.args[0].predicate == 'Pose':
        # Do something like this if odo error compounds
        # newVar = tuple([v - e for (v, e) in zip(f.args[2], totalOdoErr)])
        targetVar = f.args[2]
        if any([tv < ov for (tv, ov) in zip(targetVar, odoVar)]):
            print 'Move special regress failing; target var less than odo'
            print f
            # Can't achieve this 
            return None
        # Regress to odoVar: assumption will be that we can maintain
        # this variance if it's already this small
        # newF = f.copy()
        # newF.args[2] = odoVar
        # newF.update()
        # return newF

    elif f.predicate == 'BLoc':
        targetVar = f.args[1]
        if any([tv < ov for (tv, ov) in zip(targetVar, odoVar)]):
            print 'Move special regress failing; target var less than odo'
            print f.copy()
            # Can't achieve this 
            return None
        # newF = f.copy()
        # newF.args[1] = odoVar
        # newF.update()
        # return newF

    return f.copy()

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
    rawCost = 5
    result = costFun(rawCost, movePreProb)
    return result

def moveNBCostFun(al, args, details):
    rawCost = 1 
    result = costFun(rawCost, movePreProb)
    return result

def placeCostFun(al, args, details):
    rawCost = 3
    abstractCost = 5
    (_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,p1) = args
    result = costFun(rawCost,
                     p1*canPPProb*(1-details.domainProbs.placeFailProb)) + \
               (abstractCost if al == 0 else 0)
    return result

def pickCostFun(al, args, details):
    (o,h,pf,p,pd,gf,gm,gv,gd,prc,cd,pc,rgv,pv,p1,pr1,pr2,pr3) = args
    rawCost = 3
    abstractCost = 1
    result = costFun(rawCost, p1*canPPProb*canPPProb*\
                     (1 - details.domainProbs.pickFailProb)) + \
               (abstractCost if al == 0 else 0)
    return result

# Cost depends on likelihood of seeing the object and of moving the
# mean out of the delta bounds
# When we go to non-diagonal covariance, this will be fun...
# For now, just use the first term!
def lookAtCostFun(al, args, details):
    (_,_,_,_,vb,d,va,rva,cd,ov,p,pb,pPoseR,pFaceR) = args
    placeProb = min(pPoseR,pFaceR) if (not isVar(pPoseR) and not isVar(pFaceR))\
                           else p
    vo = details.domainProbs.obsVarTuple
    if d == '*':
        deltaViolProb = 0.0
    else:
        # Switched to using var *after* look because if look reliability
        # is very high then var before is huge and so is the cost.
        deltaViolProb = probModeMoved(d[0], rva[0], vo[0])
    result = costFun(1.0, canSeeProb*placeProb*(1-deltaViolProb)*\
                     (1 - details.domainProbs.obsTypeErrProb))
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
    return result

################################################################
## Action effects
################################################################

# All super-bogus until we get the real state estimator running

def moveBProgress(details, args, obs=None):
    (s, e, _) = args
    # Assume we've controlled the error during motion.  This is a stdev
    odoError = details.domainProbs.odoError
    odoVar = [x**2 for x in odoError]

    # Change robot conf
    details.pbs.updateConf(e)

    # Make variance of all objects not in the hand equal to the max of
    # the previous variance and odoError

    for ob in details.pbs.moveObjBs.values() + \
               details.pbs.fixObjBs.values():
        oldVar = ob.poseD.var
        # this is the additive version
        # ob.poseD.var = tuple([a + b*b for (a, b) in zip(oldVar, odoError)])
        ob.poseD.var = tuple([max(a, b) for (a, b) in zip(oldVar, odoVar)])
    details.pbs.reset()
    details.pbs.getShadowWorld(0)
    details.pbs.internalCollisionCheck()
    debugMsg('beliefUpdate', 'moveBel')    

def moveNBBProgress(details, args, obs=None):
    (b, s, e, _) = args
    # Just put the robot in the intended place
    details.pbs.updateConf(e)
    details.pbs.reset()
    details.pbs.getShadowWorld(0)
    details.pbs.internalCollisionCheck()
    debugMsg('beliefUpdate', 'moveBel')    

def pickBProgress(details, args, obs=None):
    # Assume robot ends up at preconf, so no conf change
    (o,h,pf,p,pd,gf,gm,gv,gd,prc,cd,pc,rgv,pv,p1,pr1,pr2,pr3) = args
    pickVar = details.domainProbs.pickVar
    failProb = details.domainProbs.pickFailProb
    # !! This is wrong!  The coordinate frames of the variances don't match.
    v = [x+y for x,y in zip(details.pbs.getPlaceB(o).poseD.var, pickVar)]
    v[2] = 1e-20
    gv = tuple(v)
    details.graspModeProb[h] = (1 - failProb) * details.poseModeProbs[o]
    details.pbs.updateHeld(o, gf, PoseD(gm, gv), h, gd)
    details.pbs.excludeObjs([o])
    details.pbs.reset()
    details.pbs.getShadowWorld(0)
    details.pbs.internalCollisionCheck()
    debugMsg('beliefUpdate', 'pickBel')

def placeBProgress(details, args, obs=None):
    # Assume robot ends up at preconf, so no conf change
    (o,h,pf,p,_,_,_,_,_,_,_,_,_,_,_,_,_,_,p1) = args
    placeVar = details.domainProbs.placeVar
    failProb = details.domainProbs.placeFailProb
    # !! This is wrong!  The coordinate frames of the variances don't match.
    v = [x+y for x,y in \
         zip(details.pbs.getGraspB(o,h).poseD.var, placeVar)]
    v[2] = 1e-20
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
    details.pbs.reset()
    details.pbs.getShadowWorld(0)
    details.pbs.internalCollisionCheck()
    debugMsg('beliefUpdate', 'placeBel')
    
# obs has the form (obj-type, face, relative pose)
def lookAtBProgress(details, args, obs):
    (_, lookConf, _, _, _, _, _, _, _, _, _, _, _, _) = args
    objectObsUpdate(details, lookConf, obs)
    details.pbs.reset()
    details.pbs.getRoadMap().confReachCache = {} # Clear motion planning cache
    details.pbs.getShadowWorld(0)
    details.pbs.internalCollisionCheck()
    debugMsg('beliefUpdate', 'look')

llMatchThreshold = -100.0

def objectObsUpdate(details, lookConf, obsList):
    def rem(l,x): return [y for y in l if y != x]
    prob = 0.95
    world = details.pbs.getWorld()
    shWorld = details.pbs.getShadowWorld(prob)
    rob = details.pbs.getRobot().placement(lookConf,
                                           attached=shWorld.attached)[0]
    fixed = [s.name() for s in shWorld.getNonShadowShapes()] + [rob.name()]
    obstacles = shWorld.getNonShadowShapes()
    # Objects that we expect to be visible
    objList = [s for s in obstacles \
               if visible(shWorld, lookConf, s, rem(obstacles,s)+[rob], 0.5,
                          moveHead=False, fixed=fixed)[0]]
    assert objList, 'Do not expect to see any objects!'
    scores = {}
    for obj in objList:
        scores[(obj, None)] = (llMatchThreshold, None, None)
        for obs in obsList:
            (oType, obsFace, obsPose) = obs
            if world.getObjType(obj.name()) != oType: continue
            scores[(obj, obs)] = scoreObsObj(details, obs, obj.name())
    # bestAssignment = argmax(allAssignments(objList, obsList),
    #                          lambda a: scoreAssignment(a, scores))
    bestAssignment = greedyBestAssignment(objList, obsList, scores)
    assert len(obsList)==0 or \
         any([xpose for (xobj, xpose, xf) in bestAssignment]), \
         'Best assignment maps all objects to no observation'
    for obj, obsPose, obsFace in bestAssignment:
        singleTargetUpdate(details, obj.name(), obsPose, obsFace)

# Each object is assigned an observation or None
# These are objects that we expected to be able to see.
# Doesn't initialize new objects

def allAssignments(aList, bList):
    m = len(aList)
    n = len(bList)
    if m > n:
        bList = bList + [None]*(m - n)
    
    def ablate(elts, pat):
        return map(lambda e, p: e if p else None, elts, pat)
    
    return set(imap(lambda bs: tuple(zip(aList, bs)), 
                 chain(*[chain([ablate(bPerm, pat) for pat in \
                          product((True, False), repeat = n)])\
                     for bPerm in permutations(bList)])))

def greedyBestAssignment(aList, bList, scores):
    result = []
    scoresLeft = scores.items()
    while scoresLeft:
        ((obj, obs), (val, tobs, face)) = argmax(scoresLeft,
                                              lambda ((oj, os), (v, o, f)): v)
        if val < llMatchThreshold:
            result.append((obj, None, None))
            return result
        print 'A', prettyString(val), obj.name(), tobs
        result.append((obj, tobs, face))
        # Better not to copy so much
        scoresLeft = [((oj, os), stuff) for ((oj, os), stuff) in scoresLeft \
                      if oj != obj and (os != obs or obs == None)]
    debugMsg('obsUpdate', 'greedy best assignment')
    return result

def obsDist(details, obj):
    obsVar = details.domainProbs.obsVarTuple
    objBel = details.pbs.getPlaceB(obj)
    poseFace = objBel.support.mode()
    poseMu = objBel.poseD.mode()
    var = objBel.poseD.variance()
    obsCov = [v1 + v2 for (v1, v2) in zip(var, obsVar)]
    obsPoseD = MultivariateGaussianDistribution(np.mat(poseMu.xyztTuple()).T,
                                                makeDiag(obsCov))
    return obsPoseD, poseFace

def scoreAssignment(obsAssignments, scores):
    LL = 0
    for obj, obs in obsAssignments:
        if not obs:
            LL += llMatchThreshold      # !! ??
            continue
        score = scores.get((obj, obs), None)
        if not score:
            LL += llMatchThreshold      # !! ??
            continue
        (obsPose, ll, face) = score
        if ll >= llMatchThreshold:
            LL += ll
    return LL


# Gets multiple observations and tries to find the one that best
# matches sought object
def singleTargetUpdate(details, objName, obsPose, obsFace):
    obsVar = details.domainProbs.obsVarTuple       
    oldPlaceB = details.pbs.getPlaceB(objName)
    w = details.pbs.beliefContext.world

    if obsPose == None:
        # Update modeprob if we don't get a good score
        oldP = details.poseModeProbs[objName]
        obsGivenH = details.domainProbs.obsTypeErrProb
        obsGivenNotH = (1 - details.domainProbs.obsTypeErrProb)
        newP = obsGivenH * oldP / (obsGivenH * oldP + obsGivenNotH * (1 - oldP))
        details.poseModeProbs[objName] = newP
        print 'No match above threshold', objName, oldP, newP

        newMu = oldPlaceB.poseD.mode().pose().xyztTuple()
        newSigma = [v + .001 for v in oldPlaceB.poseD.varTuple()]
        newSigma[2] = 1e-10
        newSigma = tuple(newSigma)
    else:
        # Update mode prob if we do get a good score
        oldP = details.poseModeProbs[objName]
        obsGivenH = (1 - details.domainProbs.obsTypeErrProb)
        obsGivenNotH = details.domainProbs.obsTypeErrProb
        newP = obsGivenH * oldP / (obsGivenH * oldP + obsGivenNotH * (1 - oldP))
        details.poseModeProbs[objName] = newP
        print 'Obs match for', objName, oldP, newP

        # Should update face!!
        # Update mean and sigma
        ## Be sure handling angle right.
        (newMu, newSigma) = \
                    gaussObsUpdate(oldPlaceB.poseD.mode().pose().xyztTuple(),
                                   obsPose.pose().xyztTuple(),
                                   oldPlaceB.poseD.variance(), obsVar)
        ff = w.getFaceFrames(objName)[obsFace]
        if debug('obsUpdate'):
            ## LPK!!  Should really draw the detected object but I don't have
            ## an immediate way to get the shape of a type.  Should fix that.
            objShape = details.pbs.getObjectShapeAtOrigin(objName)
            objShape.applyLoc(obsPose.pose().compose(ff.inverse())).\
                            draw('Belief', 'cyan')
            objShape = details.pbs.getObjectShapeAtOrigin(objName)
            objShape.applyLoc(util.Pose(*newMu).compose(ff.inverse())).\
              draw('Belief', 'magenta')
            raw_input('obs is Cyan; newMu is magenta')

    details.pbs.updateObjB(ObjPlaceB(objName, w.getFaceFrames(objName),
                                     DeltaDist(oldPlaceB.support.mode()),
                                     PoseD(util.Pose(*newMu), newSigma)))
    
missedObsLikelihoodPenalty = llMatchThreshold

# Return a triple:  (score, transObs, face)
# transObs might be a transformed version of obs (picking the best member
# of the class of symmetries for this object)
# face is a canonical face
def scoreObsObj(details, obs, object):
    if obs == None:
        return (missedObsLikelihoodPenalty, None, None)
    
    (oType, obsFace, obsPose) = obs
    (obsPoseD, poseFace) = obsDist(details, object)
    pbs = details.pbs
    w = pbs.beliefContext.world
    symFacesType, symXformsType = w.getSymmetries(oType)
    canonicalFace = symFacesType[obsFace]
    symXForms = symXformsType[canonicalFace]
    # Consider all object poses consistent with this observation
    symPoses = [obsPose] + [obsPose.compose(xf) for xf in symXForms]
    ff = pbs.getWorld().getFaceFrames(object)[canonicalFace]

    # Find the best matching pose mode.  Type and face must be equal,
    # pose nearby.
    #!!LPK handle faces better

    # Type
    if w.getObjType(object) != oType:
        return (-float(inf), None, None)
    # Face
    assert symFacesType[poseFace] == poseFace, 'non canonical face in bel'
    if poseFace != canonicalFace:
        return (-float(inf), None, None)
    # Iterate over symmetries for this object
    bestObs, bestLL = None, -float('inf')
    for obsPoseCand in symPoses:
        ll = float(obsPoseD.logProb(np.mat(obsPoseCand.pose().xyztTuple()).T))
        if ll > bestLL:
            bestObs, bestLL = obsPoseCand, ll
    return bestLL, bestObs, canonicalFace

# Temporary;  assumes diagonal cov; should use dist.MultivariateGaussian
def gaussObsUpdate(oldMu, obs, oldSigma, obsVar, noZ = True):
    # All tuples
    newMu = [(m * obsV + op * muV) / (obsV + muV) \
                       for (m, muV, op, obsV) in \
                       zip(oldMu, oldSigma, obs, obsVar)]
    # That was not the right way to do the angle update!  Quick and dirty here.
    oldTh, obsTh, muV, obsV = oldMu[3], obs[3], oldSigma[3], obsVar[3]
    oldX, oldY = np.cos(oldTh), np.sin(oldTh)
    obsX, obsY = np.cos(obsTh), np.sin(obsTh)
    newX = (oldX * obsV + obsX * muV) / (obsV + muV)
    newY = (oldY * obsV + obsY * muV) / (obsV + muV)
    newTh = np.arctan2(newY, newX)
    newMu[3] = newTh
    if debug('obsUpdate'):
        print 'Angle obs update'
        print '  oldTh', oldTh, 'obsTh', obsTh, 'newTh', newTh
        raw_input('okay?')

    newSigma = tuple([(a * b) / (a + b) for (a, b) in zip(oldSigma,obsVar)])
    if noZ:
        newMu[2] = oldMu[2]
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
    details.pbs.reset()
    details.pbs.getShadowWorld(0)
    details.pbs.internalCollisionCheck()

    debugMsg('beliefUpdate', 'look')

################################################################
## Operator descriptions
################################################################

# Parameter PCR is the probability that its path is not blocked.

# Allowed to move base
move = Operator(\
    'Move',
    ['CStart', 'CEnd', 'DEnd'],
    # Pre
    {0 : {Bd([CanReachHome(['CEnd', False, []]),  True, movePreProb], True)},
     1 : {Conf(['CStart', 'DEnd'], True)}},
    # Results:  list of pairs: (fluent set, private preconds)
    [({Conf(['CEnd', 'DEnd'], True)}, {})],
    functions = [\
        Function(['CEnd'], [], genNone, 'genNone'),
                 ],
    cost = moveCostFun,
    f = moveBProgress,
    prim  = movePrim,
    # Can't use conditional results here, because there are
    # arbitrarily many instances of the Pose fluent that need a
    # special regression condition.
    specialRegress = moveSpecialRegress,
    argsToPrint = [0, 1],
    ignorableArgs = range(3))  # For abstraction
    # Would like really to just pay attention to the base!!

# Not allowed to move base
moveNB = Operator(\
    'MoveNB',
    ['Base', 'CStart', 'CEnd', 'DEnd'],
    # Pre
    {0 : {Bd([CanReachNB(['CStart', 'CEnd', []]),  True, movePreProb], True),
          Conf(['CStart', 'DEnd'], True),
          BaseConf(['Base', 'DEnd'], True)
             }},
    # Results:  list of pairs: (fluent set, private preconds)
    [({Conf(['CEnd', 'DEnd'], True)}, {})],
    functions = [\
        Function(['CEnd'], [], genNone, 'genNone'),
        Function(['Base'], ['CEnd'], getBase, 'getBase')
                 ],
    cost = moveNBCostFun,
    f = moveNBBProgress,
    prim  = moveNBPrim,
    argsToPrint = [0],
    ignorableArgs = range(4))

# All this work to say you can know the location of something by knowing its
# pose or its grasp
bLoc1 = Operator(\
         'BLoc1', ['Obj', 'Var', 'P'],
         {0 : {B([Pose(['Obj', '*']), '*', 'Var', '*', 'P'], True),
               Bd([SupportFace(['Obj']), '*', 'P'], True)}},
         [({BLoc(['Obj', 'Var', 'P'], True)}, {})])

bLoc2 = Operator(\
         'BLoc2', ['Obj', 'Var', 'P'],
         {0 : {Graspable(['Obj'], True),
               B([Grasp(['Obj', 'left', '*']), '*', 'Var', '*', 'P'], True),
               Bd([Holding(['left']), 'Obj', 'P'], True),
               Bd([GraspFace(['Obj', 'left']), '*', 'P'], True)}},
         [({BLoc(['Obj', 'Var', 'P'], True)}, {})])

bLoc3 = Operator(\
         'BLoc3', ['Obj', 'Var', 'P'],
         {0 : {Graspable(['Obj'], True),
               B([Grasp(['Obj', 'right', '*']), '*', 'Var', '*', 'P'], True),
               Bd([Holding(['right']), 'Obj', 'P'], True),
               Bd([GraspFace(['Obj', 'right']), '*', 'P'], True)}},
         [({BLoc(['Obj', 'Var', 'P'], True)}, {})])

poseAchIn = Operator(\
             'PosesAchIn', ['Obj1', 'Region',
                            'ObjPose1', 'PoseFace1',
                            'Obj2', 'ObjPose2', 'PoseFace2',
                            'PoseVar', 'TotalVar', 'PoseDelta', 'TotalDelta',
                            'P1', 'P2', 'PR'],
            # Very prescriptive:  find objects, then nail down obj1,
            # then obj 2.  Changed so we don't try to maintain
            # detailed k of the table as we are picking other obj.
            {0 : set(),
             1 : {BLoc(['Obj1', planVar, planP], True), # 'PoseVar'
                  BLoc(['Obj2', planVar, planP], True)},
             2 : {B([Pose(['Obj1', 'PoseFace1']), 'ObjPose1', 'PoseVar',
                               'PoseDelta', 'P1'], True),
                  Bd([SupportFace(['Obj1']), 'PoseFace1', 'P1'], True)},
             3 : {B([Pose(['Obj2', 'PoseFace2']), 'ObjPose2', 'PoseVar',
                               'PoseDelta', 'P2'], True),
                  Bd([SupportFace(['Obj2']), 'PoseFace2', 'P2'], True)}},
            # Results
            [({Bd([In(['Obj1', 'Region']), True, 'PR'], True)},{})],
            functions = [\
              Function(['P1', 'P2'], ['PR'], regressProb(2), 'regressProb2'),
              # Object region is defined wrto
              Function(['Obj2'], ['Region'], regionParent, 'regionParent'),
              Function(['PoseDelta'], [], defaultPlaceDelta,
                       'defaultPlaceDelta'),
              # Assume it doesn't move
              Function(['PoseFace2', 'ObjPose2'], ['Obj2'],
                       poseInStart, 'poseInStart'),
              # totalVar = square(2 * sqrt(poseVar))
              Function(['PoseVar'], [], placeInPoseVar, 'placeInPoseVar'),
              Function(['TotalVar'], ['PoseVar'], stdevTimes2, 'stdevTimes2'),
              Function(['TotalDelta'], ['PoseDelta'], times2, 'times2'),
              # call main generator
              Function(['ObjPose1', 'PoseFace1'],
                     ['Obj1', 'Region', 'TotalVar', 'TotalDelta',
                      probForGenerators],
                     placeInRegionGen, 'placeInRegionGen')],
            argsToPrint = [0, 1],
            ignorableArgs = range(1,11))

place = Operator(\
        'Place',
        ['Obj', 'Hand',
         'PoseFace', 'Pose', 'PoseVar', 'RealPoseVar', 'PoseDelta',
         'GraspFace', 'GraspMu', 'GraspVar', 'GraspDelta',
         'PreConf', 'ConfDelta', 'PlaceConf', 'AwayRegion',
         'PR1', 'PR2', 'PR3', 'P1'],
        # Pre
        {0 : {Graspable(['Obj'], True)},
         1 : {Bd([CanPickPlace(['PreConf', 'PlaceConf', 'Hand', 'Obj', 'Pose',
                               'RealPoseVar', 'PoseDelta', 'PoseFace',
                               'GraspFace', 'GraspMu', 'GraspVar', 'GraspDelta',
                                'place', []]), True, canPPProb],True)},
         2 : {Bd([Holding(['Hand']), 'Obj', 'P1'], True),
              Bd([GraspFace(['Obj', 'Hand']), 'GraspFace', 'P1'], True),
              B([Grasp(['Obj', 'Hand', 'GraspFace']),
                 'GraspMu', 'GraspVar', 'GraspDelta', 'P1'], True)},
         3 : {Conf(['PreConf', 'ConfDelta'], True)}
        },
        # Results
        [#({BLoc(['Obj', planVar, 'PR2'], True)},{}),  # 'PoseVar'
         ({Bd([SupportFace(['Obj']), 'PoseFace', 'PR1'], True),
           B([Pose(['Obj', 'PoseFace']), 'Pose', 'PoseVar', 'PoseDelta','PR2'],
                 True)},{}),
         ({Bd([Holding(['Hand']), 'none', 'PR3'], True)}, {})],
        # Functions
        functions = [\
            # Not appropriate when we're just trying to decrease variance
            Function([], ['Pose'], notStar, 'notStar', True),
            Function([], ['PoseFace'], notStar, 'notStar', True),

            # Get object.  Only if the var is unbound.  Try first the
            # objects that are currently in the hands.
            Function(['Obj'], ['Hand'], getObj, 'getObj'),
            
            # Be sure all result probs are bound.  At least one will be.
            Function(['PR1', 'PR2', 'PR3'],
                     ['PR1', 'PR2', 'PR3'], minP,'minP'),

            # Compute precond probs.  Assume that crash is observable.
            # So, really, just move the obj holding prob forward into
            # the result.  
            Function(['P1'], ['PR1', 'PR2', 'PR3'], 
                     regressProb(1, 'placeFailProb'), 'regressProb1', True),
            
            # In case not specified
            Function(['PoseVar'], [], placeInPoseVar, 'placeInPoseVar'),

            # PoseVar = GraspVar + PlaceVar,
            # GraspVar = min(maxGraspVar, PoseVar - PlaceVar)
            Function(['GraspVar'], ['PoseVar'], placeGraspVar, 'placeGraspVar',
                     True),

            # Real pose var might be much less than pose var if the
            # original pos var was very large
            # RealPoseVar = GraspVar + PlaceVar
            Function(['RealPoseVar'],
                     ['GraspVar'], realPoseVar, 'realPoseVar'),
            
            # In case PoseDelta isn't defined
            Function(['PoseDelta'], [], defaultPlaceDelta, 'defPlaceDelta'),
            # Assume fixed conf delta
            Function(['ConfDelta'], [], moveConfDelta, 'moveConfDelta'),

            Function(['GraspDelta'], ['PoseDelta', 'ConfDelta'],
                      subtract, 'subtract'),

            # Not modeling the fact that the object's shadow should
            # grow a bit as we move to pick it.   Build that into pickGen.
            Function(['Hand', 
                      'GraspMu', 'GraspFace', 'PlaceConf', 'PreConf',
                      'Pose', 'PoseFace'],
                     ['Obj', 'Hand', 'Pose', 'PoseFace', 'RealPoseVar',
                      'GraspVar',
                      'PoseDelta', 'GraspDelta', 'ConfDelta',probForGenerators],
                     placeGen, 'placeGen'),

            ],
        cost = placeCostFun, 
        f = placeBProgress,
        prim = placePrim,
        argsToPrint = range(4),
        ignorableArgs = range(1, 19))


# Put the condition to know the pose precisely down at the bottom to
# try to decrease replanning.

pick = Operator(\
        'Pick',
        ['Obj', 'Hand', 'PoseFace', 'Pose', 'PoseDelta',
         'GraspFace', 'GraspMu', 'GraspVar', 'GraspDelta',
         'PreConf', 'ConfDelta', 'PickConf', 'RealGraspVar', 'PoseVar',
         'P1', 'PR1', 'PR2', 'PR3'],
        # Pre
        {0 : {Graspable(['Obj'], True),
              BLoc(['Obj', planVar, 'P1'], True)},    # was planP
         2 : {Bd([SupportFace(['Obj']), 'PoseFace', 'P1'], True),
              B([Pose(['Obj', 'PoseFace']), 'Pose', planVar, 'PoseDelta',
                 'P1'], True)},
         1 : {Bd([CanPickPlace(['PreConf', 'PickConf', 'Hand', 'Obj', 'Pose',
                               'PoseVar', 'PoseDelta', 'PoseFace',
                               'GraspFace', 'GraspMu', 'RealGraspVar',
                               'GraspDelta', 'pick', []]), True, canPPProb],
                               True),
              Bd([Holding(['Hand']), 'none', canPPProb], True)},
         3 : {Conf(['PreConf', 'ConfDelta'], True),
              B([Pose(['Obj', 'PoseFace']), 'Pose', 'PoseVar', 'PoseDelta',
                 'P1'], True)              
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
            
            # Be sure all result probs are bound.  At least one will be.
            Function(['PR1', 'PR2', 'PR3'], ['PR1', 'PR2', 'PR3'], minP,'minP'),

            # Compute precond probs.  Only regress object placecement P1.
            # Consider failure prob
            Function(['P1'], ['PR1', 'PR2'], 
                     regressProb(1, 'pickFailProb'), 'regressProb1', True),
            Function(['RealGraspVar'], ['GraspVar'], maxGraspVarFun,
                     'realGraspVar'),

            # Assume fixed conf delta
            Function(['ConfDelta'], [], moveConfDelta, 'moveConfDelta'),
                     
            # Subtract off conf delta
            Function(['PoseDelta'], ['GraspDelta', 'ConfDelta'],
                      subtract, 'subtract'),

            # GraspVar = PoseVar + PickVar
            # prob was pr3, but this keeps it tighter
            Function(['PoseVar'], ['RealGraspVar', 'GraspDelta',
                                   probForGenerators],
                     pickPoseVar, 'pickPoseVar'),

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
        argsToPrint = [0, 1, 3, 9],
        ignorableArgs = range(1, 18))

# We know that the resulting variance will always be less than obsVar.
# Would like the result to be the min of PoseVarAfter (which is in the
# goal) and obsVar.  Trying to make two different
# results...unfortunate increase in branching factor unless
# applicableOps handles them well.

lookAt = Operator(\
    'LookAt',
    ['Obj', 'LookConf', 'PoseFace', 'Pose',
     'PoseVarBefore', 'PoseDelta', 'PoseVarAfter', 'RealPoseVarAfter',
     'ConfDelta', 'ObsVar', 
     'P1', 'PR0', 'PR1', 'PR2'],
    # Pre
    {0: {Bd([SupportFace(['Obj']), 'PoseFace', 'P1'], True),
         B([Pose(['Obj', 'PoseFace']), 'Pose', 'PoseVarBefore', 'PoseDelta',
                 'P1'], True)},
     1: {Bd([CanSeeFrom(['Obj', 'Pose', 'PoseFace', 'LookConf', []]),
             True, canSeeProb], True),
         Conf(['LookConf', 'ConfDelta'], True)}},
    [({B([Pose(['Obj', 'PoseFace']), 'Pose', 'PoseVarAfter', 'PoseDelta',
         'PR1'],True),
       B([Pose(['Obj', 'PoseFace']), 'Pose', 'ObsVar', 'PoseDelta',
         'PR1'],True),         
       Bd([SupportFace(['Obj']), 'PoseFace', 'PR2'], True)}, {})
       ],
    # Functions
    functions = [\
        # In case these aren't bound
        Function(['PoseFace'], [['*']], assign, 'assign'),
        Function(['Pose'], [['*']], assign, 'assign'),
        Function(['PoseDelta'], [['*']], assign, 'assign'),
        Function(['ConfDelta'], [], moveConfDelta, 'moveConfDelta'),
        Function(['ObsVar'], [], obsVar, 'obsVar'),
        Function(['RealPoseVarAfter'], ['PoseVarAfter'],
                 realPoseVarAfterObs, 'realPoseVarAfterObs'),
        # Look increases probability.  
        Function(['P1'], ['PR0', 'PR1', 'PR2'], obsModeProb, 'obsModeProb'),
        # How confident do we need to be before the look?
        Function(['PoseVarBefore'], ['RealPoseVarAfter', 'Obj', 'PoseFace'],
                genLookObjPrevVariance, 'genLookObjPrevVariance'),
        Function(['LookConf'],
                 ['Obj', 'Pose', 'PoseFace', 'PoseVarBefore', 'PoseDelta',
                         'ConfDelta', probForGenerators],
                 lookGen, 'lookGen')
        ],
    cost = lookAtCostFun,
    f = lookAtBProgress,
    prim = lookPrim,
    argsToPrint = [0, 1, 3],
    ignorableArgs = [1, 2] + range(4, 11))

## Should have a CanSeeFrom precondition
## Needs major update

'''
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
'''    

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
    ['CEnd', 'FCP', 'PreCond', 'PostCond',
     'Occ', 'PoseFace', 'Pose', 'PoseVar', 'PoseDelta', 'P1', 'PR'],

    {0: {},
     1: {Bd([CanReachHome(['CEnd', 'FCP', 'PreCond']),  True, 'PR'], True),
         Bd([SupportFace(['Occ']), 'PoseFace', 'P1'], True),
         B([Pose(['Occ', 'PoseFace']), 'Pose', 'PoseVar', 'PoseDelta', 'P1'],
            True)}},
    # Result
    [({Bd([CanReachHome(['CEnd', 'FCP','PostCond']),  True, 'PR'], True)}, {})],

    # Functions
    functions = [\
        # Compute precond probs
        Function(['P1'], ['PR'], regressProb(1), 'regressProb1', True),
        # Call generator
        Function(['Occ', 'Pose', 'PoseFace', 'PoseVar', 'PoseDelta'],
                  ['CEnd', 'FCP', 'PR', 'PostCond'], canReachGen,'canReachGen'),
         # Add the appropriate condition
         Function(['PreCond'],
                  ['PostCond',
                   'Occ', 'PoseFace', 'Pose', 'PoseVar', 'PoseDelta', 'PR'],
                  addPosePreCond, 'addPosePreCond')],
    cost = lambda al, args, details: 0.1,
    argsToPrint = [0, 4, 6],
    ignorableArgs = range(1, 11))

poseAchCanPickPlace = Operator(\
    'PoseAchCanPickPlace',
    ['PreConf', 'PlaceConf', 'Hand', 'Obj', 'Pose',
     'RealPoseVar', 'PoseDelta', 'PoseFace',
     'GraspFace', 'GraspMu', 'GraspVar', 'GraspDelta',
     'PreCond', 'PostCond',
     'Occ', 'OccPose', 'OccPoseFace', 'OccPoseVar', 'OccPoseDelta', 'Op',
     'P1', 'PR'],
    {0: {},
     1: {Bd([CanPickPlace(['PreConf', 'PlaceConf', 'Hand', 'Obj', 'Pose',
                          'RealPoseVar', 'PoseDelta', 'PoseFace',
                          'GraspFace', 'GraspMu', 'GraspVar', 'GraspDelta',
                          'Op', 'PreCond']), True, 'PR'],True),
         Bd([SupportFace(['Occ']), 'OccPoseFace', 'P1'], True),
         B([Pose(['Occ', 'OccPoseFace']), 'OccPose', 'OccPoseVar',
                  'OccPoseDelta', 'P1'],
            True)}},
    # Result
    [({Bd([CanPickPlace(['PreConf', 'PlaceConf', 'Hand', 'Obj', 'Pose',
                          'RealPoseVar', 'PoseDelta', 'PoseFace',
                          'GraspFace', 'GraspMu', 'GraspVar', 'GraspDelta',
                          'Op', 'PostCond']), True, 'PR'],True)},
                          {})],
    # Functions
    functions = [\
        # Compute precond probs
        Function(['P1'], ['PR'], regressProb(1), 'regressProb1', True),
        # Call generator
        Function(['Occ', 'OccPose', 'OccPoseFace', 'OccPoseVar','OccPoseDelta'],
                  ['PreConf', 'PlaceConf', 'Hand', 'Obj', 'Pose',
                          'RealPoseVar', 'PoseDelta', 'PoseFace',
                          'GraspFace', 'GraspMu', 'GraspVar', 'GraspDelta',
                          'PR', 'PostCond', 'Op'],
                          canPickPlaceGen, 'canPickPlaceGen'),
         # Add the appropriate condition
         Function(['PreCond'],
                  ['PostCond',
                   'Occ', 'OccPoseFace', 'OccPose', 'OccPoseVar',
                   'OccPoseDelta', 'PR'],
                  addPosePreCond, 'addPosePreCond')],
    cost = lambda al, args, details: 0.1,
    argsToPrint = [3, 4, 14, 15],
    ignorableArgs = range(0, 2) + range(5, 22))

# Only useful if we can look at hand. 
graspAchCanReach = Operator(\
    'GraspAchCanReach',
    ['CEnd', 'FCP', 'PreCond', 'PostCond',
     'Obj', 'Hand', 'GraspFace', 'GraspMu', 'PreGraspVar', 'GraspDelta',
      'P1', 'PR'],
    {0: {},
     1: {Bd([CanReachHome(['CEnd', 'FCP', 'PreCond']),  True, 'PR'], True),
         Bd([GraspFace(['Obj', 'Hand']), 'GraspFace', 'P1'], True),
         Bd([Holding(['Hand']), 'Obj', 'P1'], True),
         B([Grasp(['Obj', 'Hand', 'GraspFace']),
             'GraspMu', 'PreGraspVar', 'GraspDelta', 'P1'], True)}},
    # Result
    [({Bd([CanReachHome(['CEnd', 'FCP','PostCond']),  True, 'PR'], True)}, {})],

    # Functions
    functions = [\
        # Compute precond probs
        Function(['P1'], ['PR'], regressProb(1), 'regressProb1', True),
        # Call generator
        Function([], ['CEnd', 'FCP', 'PR', 'PostCond', 'Hand'],
                 canReachHandGen,'canReachHandGen', True),
         # Add the appropriate condition
         Function(['PreCond'],
                  ['PostCond', 'Hand',
                   'Obj', 'GraspFace', 'GraspMu', 'PreGraspVar',
                   'GraspDelta', 'PR'],
                  addGraspPreCond, 'addGraspPreCond')],
    cost = lambda al, args, details: 0.1,
    argsToPrint = [0, 4],
    ignorableArgs = range(1, 12))

dropAchCanReach = Operator(\
    'DropAchCanReach',
    ['CEnd', 'FCP', 'PreCond', 'PostCond', 'Hand', 'P1', 'PR'],
    {0: {Bd([CanReachHome(['CEnd', 'FCP', 'PreCond']),  True, 'PR'], True)},
     1: {Bd([Holding(['Hand']), 'none', 'P1'], True)}},
    # Result
    [({Bd([CanReachHome(['CEnd', 'FCP','PostCond']),  True, 'PR'], True)}, {})],

    # Functions
    functions = [\
        # Compute precond probs
        Function(['P1'], ['PR'], regressProb(1), 'regressProb1', True),
        # Call generator
        Function(['Hand'], ['CEnd', 'FCP', 'PR', 'PostCond'],
                 canReachDropGen,'canReachHandGen', True),
         # Add the appropriate condition
         Function(['PreCond'],
                  ['PostCond', 'Hand', 'PR'],
                  addDropPreCond, 'addDropPreCond')],
    cost = lambda al, args, details: 0.1,
    argsToPrint = [0, 4],
    ignorableArgs = range(1, 7))

dropAchCanPickPlace = Operator(\
    'dropAchCanPickPlace',
    ['PreConf', 'PlaceConf', 'Hand', 'Obj', 'Pose',
                          'RealPoseVar', 'PoseDelta', 'PoseFace',
                          'GraspFace', 'GraspMu', 'GraspVar', 'GraspDelta',
                          'Op', 'DropHand', 'PreCond', 'PostCond', 'P1', 'PR'],
    {0: {Bd([CanPickPlace(['PreConf', 'PlaceConf', 'Hand', 'Obj', 'Pose',
                          'RealPoseVar', 'PoseDelta', 'PoseFace',
                          'GraspFace', 'GraspMu', 'GraspVar', 'GraspDelta',
                          'Op', 'PreCond']), True, 'PR'],True)},
     1: {Bd([Holding(['DropHand']), 'none', 'P1'], True)}},
    # Result
    [({Bd([CanPickPlace(['PreConf', 'PlaceConf', 'Hand', 'Obj', 'Pose',
                          'RealPoseVar', 'PoseDelta', 'PoseFace',
                          'GraspFace', 'GraspMu', 'GraspVar', 'GraspDelta',
                          'Op', 'PostCond']), True, 'PR'],True)}, {})],
    # Functions
    functions = [\
        # Compute precond probs
        Function(['P1'], ['PR'], regressProb(1), 'regressProb1', True),
        # Call generator
        Function(['DropHand'],
                 ['PreConf', 'PlaceConf', 'Hand', 'Obj', 'Pose',
                          'RealPoseVar', 'PoseDelta', 'PoseFace',
                          'GraspFace', 'GraspMu', 'GraspVar', 'GraspDelta',
                          'Op', 'PostCond', 'PR'],
                 canPickPlaceDropGen,'canPickPlaceDropGen', True),
         # Add the appropriate condition
         Function(['PreCond'],
                  ['PostCond', 'DropHand', 'PR'],
                  addDropPreCond, 'addDropPreCond')],
    cost = lambda al, args, details: 0.1,
    argsToPrint = [0, 4],
    ignorableArgs = [0, 1, 2] + range(4, 7))


'''
More generally:  have the generators return the relevant conditions.
So, one generator per conditional fluent.

We would need to add some mechanism to the rule definition so we can
programatically compute some extra preconditions.  That's a bit of 


                           
Additional connective operators:
dropAchCanPickPlace
graspAchCanPickPlace
graspAchCanReachNB
dropAchCanReachNB
poseAchCanReachNB
graspAchCanSee
dropAchCanSee
                            

'''    

# Never been tested

poseAchCanSee = Operator(\
    'PoseAchCanSee',
    ['Obj', 'TargetPose', 'TargetPoseFace', 'TargetPoseVar', 'TargetPoseDelta',
     'Occ', 'OccPoseFace', 'OccPose', 'OccPoseVar', 'OccPoseDelta',
     'LookConf', 'ConfDelta', 'PreCond', 'PostCond',
      'P1', 'P2', 'PR'],

    {0: {Bd([CanSeeFrom(['Obj', 'TargetPose', 'TargetPoseFace', 'LookConf',
                         'PreCond']),
             True, 'PR'], True)},
     1: {Bd([SupportFace(['Occ']), 'OccPoseFace', 'PR'], True),
         B([Pose(['Occ', 'OccPoseFace']), 'OccPose', 'OccPoseVar',
            'OccPoseDelta', 'PR'], True)}},
    # Result
    [({Bd([CanSeeFrom(['Obj', 'TargetPose', 'TargetPoseFace', 'LookConf', 
                           'PostCond']),  True, 'PR'], True)}, {})],
    # Functions
    functions = [\
        # Compute precond probs
        Function(['P1', 'P2'], ['PR'], regressProb(2), 'regressProb2', True),
        # Only want to see the mean, assume robot at conf
        Function(['TargetPoseVar', 'TargetPoseDelta', 'ConfDelta'], [],
                 lambda a, c, b, o: [[(0.0,)*4, (0.0,)*4, (0.0,)*4]], 'zeros'),
        # Call generator
        Function(['Occ', 'OccPose', 'OccPoseFace', 'OccPoseVar','OccPoseDelta'],
                  ['Obj', 'TargetPose', 'TargetPoseFace', 'TargetPoseVar',
                   'TargetPoseDelta', 'LookConf', 'ConfDelta', 'PR'],
                    canSeeGen, 'canSeeGen'),
         # Add the appropriate condition
         Function(['PreCond'],
                  ['PostCond',
                   'Occ', 'OccPoseFace', 'OccPose', 'OccPoseVar',
                   'OccPoseDelta', 'P1'],
                  addPosePreCond, 'addPosePreCond')],
    argsToPrint = [0, 1, 5],
    cost = lambda al, args, details: 0.1)


######################################################################
#
# Operators only used in the heuristic
#
######################################################################

magicRegraspCost = 10

# Not sure how to make this be called less often; it's relevant to any
# "Holding" goal.  Hopefully the easyGrasp cache works well.  Will try
# attenuating the probabilities.

hRegrasp = Operator(\
        'HeuristicRegrasp',
        ['Obj', 'Hand', 'GraspFace', 'GraspMu', 'GraspVar', 'GraspDelta',
         'PrevGraspFace', 'PrevGraspMu', 'PrevGraspVar', 'PrevGraspDelta',
         'P1', 'P2', 'P3', 'PR1', 'PR2', 'PR3'],

        # Pre
        {0 : {Bd([Holding(['Hand']), 'Obj', 'P1'], True),
              Bd([GraspFace(['Obj', 'Hand']), 'PrevGraspFace', 'P2'], True),
              B([Grasp(['Obj', 'Hand', 'PrevGraspFace']),
                  'PrevGraspMu', 'PrevGraspVar', 'PrevGraspDelta', 'P3'],
                  True)}},

        # Results
        [({Bd([Holding(['Hand']), 'Obj', 'PR1'], True), 
           Bd([GraspFace(['Obj', 'Hand']), 'GraspFace', 'PR2'], True),
           B([Grasp(['Obj', 'Hand', 'GraspFace']),
            'GraspMu', 'GraspVar', 'GraspDelta', 'PR3'], True)}, {})],

        # Functions
        functions = [\
            # Be sure obj is not none
            Function([], ['Obj'], notNone, 'notNone', True),
            Function(['P1'], ['PR1'], 
                     regressProb(1, 'pickFailProb'), 'regressProb1', True),
            Function(['P2'], ['PR2'], 
                     regressProb(1, 'pickFailProb'), 'regressProb1', True),
            Function(['P3'], ['PR3'], 
                     regressProb(1, 'pickFailProb'), 'regressProb1', True),

            Function(['PrevGraspFace', 'PrevGraspMu', 'PrevGraspVar',
                      'PrevGraspDelta'],
                      ['Obj', 'Hand'], easyGraspGen, 'easyGraspGen'),
            # Only use to change grasp.
            Function([], ['GraspMu', 'GraspFace', 'GraspVar',
                          'PrevGraspMu', 'PrevGraspFace', 'PrevGraspVar'],
                     notEqual3, 'notEqual3', True),
            ],
        cost = lambda al, args, details: magicRegraspCost,
        argsToPrint = [0, 1])
