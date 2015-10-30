import pdb
import numpy as np
import planGlobals as glob
from numpy import cos, sin, arctan2, sqrt, power
import hu
from traceFile import tr, trAlways
from itertools import product, permutations, chain, imap
from dist import DeltaDist, varBeforeObs, probModeMoved, MixtureDist,\
     UniformDist, chiSqFromP, MultivariateGaussianDistribution
from fbch import Function, Operator, simplifyCond, State
from miscUtil import isVar, prettyString, makeDiag, argmax, lookup, roundrobin
from planUtil import PoseD, ObjGraspB, ObjPlaceB, Violations
from pr2Util import shadowWidths, objectName
from pr2Gen import PickGen, LookGen,\
    EasyGraspGen, canPickPlaceTest, PoseInRegionGen, PlaceGen, moveOut
from belief import Bd, B, BMetaOperator
from pr2Fluents import Conf, CanReachHome, Holding, GraspFace, Grasp, Pose,\
     SupportFace, In, CanSeeFrom, Graspable, CanPickPlace,\
     findRegionParent, CanReachNB, BaseConf, BLoc, canReachHome, canReachNB,\
     Pushable, CanPush, canPush, graspable, pushable
from traceFile import debugMsg, debug
from pr2Push import PushGen, pushOut
import pr2RRT as rrt
from pr2Visible import visible
import itertools

zeroPose = zeroVar = (0.0,)*4
awayPose = (100.0, 100.0, 0.0, 0.0)
maxVarianceTuple = (.1,)*4

# If it's bigger than this, we can't just plan to look and see it
# Should be more subtle than this...
maxPoseVar = (0.05**2, 0.05**2, 0.05**2, 0.1**2)
#maxPoseVar = (0.03**2, 0.03**2, 0.03**2, 0.05**2)

# Don't allow delta smaller than this
minDelta = .0001

# Fixed accuracy to use for some standard preconditions
canPPProb = 0.9
otherHandProb = 0.9
canSeeProb = 0.9
# No prob can go above this
maxProbValue = 0.98  # was .999
# How sure do we have to be of CRH for moving
#movePreProb = 0.98
movePreProb = 0.8
# Prob for generators.  Keep it high.   Should this be = maxProbValue?
probForGenerators = 0.98

# Generic large values for the purposes of planning If they're small,
# it has to keep looking to maintain them.  If they're large, the
# plans become invalidated all the time.

#planVar = (0.08**2, 0.08**2, 0.04**2, 0.16**2)
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

tryDirectPath = True
# canReachHome(conf) returns a path from conf to home!

def primPath(bs, cs, ce, p):
    def onlyShadows(viols):
        return viols and not (viols.obstacles or any(viols.heldObstacles))
    home = bs.getRoadMap().homeConf
    path, viols = canReachHome(bs, cs, p, Violations(),
                               homeConf=ce, optimize=True)
    if not(path):
        path, viols = canReachHome(bs, ce, p, Violations(),
                                   homeConf=cs, optimize=True)
        path = path[::-1]               # reverse path
    if path:
        if viols.weight() > 0 and onlyShadows(viols):
            print 'Shadow collision in primPath', viols
            raw_input('Shadow collisions - continue?')
        trAlways('Direct path succeeded')
    else:
        assert 'primPath failed'

    smoothed = bs.getRoadMap().smoothPath(path, bs, p)
    interpolated = rrt.interpolatePath(smoothed)
    verifyPaths(bs, p, path, smoothed, interpolated)
    return smoothed, interpolated

def primNBPath(bs, cs, ce, p):
    path, v = canReachNB(bs, cs, ce, p, Violations())
    if not path:
        print 'NB Path failed, trying RRT'
        path, v = rrt.planRobotPathSeq(bs, p, cs, ce, None,
                                       maxIter=50, failIter=10, inflate=True)
    assert path
    if v.weight() > 0:
        raw_input('Potential collision in primitive path')
    else:
        trAlways('Success on primNB')
    smoothed = bs.getRoadMap().smoothPath(path, bs, p)
    interpolated = rrt.interpolatePath(smoothed)
    verifyPaths(bs, p, path, smoothed, interpolated)
    return smoothed, interpolated

def verifyPath(pbs, prob, path, msg):
    print 'Verifying', msg, 'path'
    shWorld = pbs.getShadowWorld(prob)
    attached = shWorld.attached
    for conf in path:
        viol = pbs.getRoadMap().confViolations(conf, pbs, prob)
        pbs.draw(prob, 'W')
        if not viol or viol.weight() > 0:
            print msg, 'path', viol
            conf.draw('W', 'red', attached=attached)
        else:
            conf.draw('W', 'blue', attached=attached)            
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
    cs = bs.conf
    tr('prim', 'moveNBPrim (start, end)', confStr(cs), confStr(ce),
       pause = False)
    path, interpolated = primNBPath(bs, cs, ce, movePreProb)
    assert path
    tr('prim', '*** movePrim no base', args, ('path length', len(path)))
    return path, interpolated, details.pbs.getPlacedObjBs()

def movePrim(args, details):
    (cs, ce, cd) = args

    bs = details.pbs.copy()
    # Make all the objects be fixed
    bs.fixObjBs.update(bs.moveObjBs)
    bs.moveObjBs = {}
    cs = bs.conf
    tr('prim', 'movePrim (start, end)', confStr(cs), confStr(ce),
       pause = False)
    path, interpolated = primPath(bs, cs, ce, movePreProb)
    assert path
    tr('prim', '*** movePrim no base', args, ('path length', len(path)))
    # path(s) and the distributions for the placed objects, to guide looking
    return path, interpolated, details.pbs.getPlacedObjBs()

def confStr(conf):
    cart = conf.cartConf()
    pose = cart['pr2LeftArm'].pose(fail=False)
    if pose:
        hand = str(np.array(pose.xyztTuple()))
    else:
        hand = '\n'+str(cart['pr2LeftArm'].matrix)
    return 'base: '+str(conf['pr2Base'])+'   hand: '+hand

def pickPrim(args, details):
    # !! Should close the fingers as well?
    tr('prim','*** pickPrim', args)
    return details.pbs.getPlacedObjBs()

def lookPrim(args, details):
    # In the real vision system, we might pass in a more general
    # structure with all the objects (and/or types) we expect to see
    tr('prim', '*** lookPrim', args)
    # The distributions for the placed objects, to guide looking
    return details.pbs.getPlacedObjBs()

def lookHandPrim(args, details):
    # In the real vision system, we might pass in a more general
    # structure with all the objects (and/or types) we expect to see
    tr('prim', '*** lookHandPrim', args)
    # The distributions for the grasped objects, to guide looking
    return details.pbs.graspB
    
def placePrim(args, details):
    # !! Should open the fingers as well
    tr('prim', '*** placePrim', args)
    return details.pbs.getPlacedObjBs()

def pushPrim(args, details):
    (obj, hand, pose, poseFace, poseVar, poseDelta, prePose, prePoseVar,
            preConf, pushConf, postConf, confDelta, resultProb, preProb1,
            preProb2) = args
    # TODO: Does it matter which prob we use?
    path, viol = canPush(details.pbs, obj, hand, poseFace, prePose, pose,
                         preConf, pushConf, postConf, prePoseVar, poseVar,
                         poseDelta, resultProb, Violations(), prim=True)
    assert path
    tr('prim', '*** pushPrim', args, ('path length', len(path)))
    if postConf in path:
        revIndex = path.index(postConf)
        revPath = path[revIndex:]
        revPath.reverse()
    else:
        revIndex = path.index(reverseConf(path, hand))
        revPath = path[revIndex:]
        revPath.reverse()
        revPath.append(postConf)
        print 'Did not find postConf in path - adding it'
    return path, revPath, details.pbs.getPlacedObjBs()    

# TODO: Similar to one defined in pr2Push
def reverseConf(pp, hand):
    cpost = pp[-1]                # end of the path, contact with object
    handFrameName = cpost.robot.armChainNames[hand]
    tpost = cpost.cartConf()[handFrameName] # final hand position
    for i in range(2, len(pp)):
        c = pp[-i]
        t = c.cartConf()[handFrameName] # hand position
        if t.distance(tpost) > 0.1:
            return c
    return pp[0]

################################################################
## Simple generators
################################################################

# Relevant fluents:
#  Holding(hand), GraspFace(obj, hand), Grasp(obj, hand, face)

smallDelta = (10e-4,)*4

class GetObj(Function):
    # noinspection PyUnusedLocal
    @staticmethod
    def fun((h,), goal, start):
        heldLeft = start.pbs.getHeld('left').mode()
        heldRight = start.pbs.getHeld('right').mode()
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
    mu = gd.poseD.modeTuple()
    var = gd.poseD.var
    return obj, face, mu, var, smallDelta

# Just return the base pose
# noinspection PyUnusedLocal
class GetBase(Function):
    @staticmethod
    def fun((conf,), goal, start):
        if isVar(conf):
            return []
        else:
            return [[conf['pr2Base']]]

# LPK: make this more efficient by storing the inverse mapping
class RegionParent(Function):
    # noinspection PyUnusedLocal
    @staticmethod
    def fun((region,), goal, start):
        if isVar(region):
            # Trying to place at a pose.  Doesn't matter.
            return [['none']]
        else:
            return [[findRegionParent(start, region)]]

class PoseInStart(Function):
    # noinspection PyUnusedLocal
    @staticmethod
    def fun((obj,), goal, start):
        pd = start.pbs.getPlaceB(obj)
        face = pd.support.mode()
        mu = pd.poseD.modeTuple()
        return [(face, mu)]

# Use this when we don't want to generate an argument (expecting to
# get it from goal bindings.)  Guaranteed to fail if that var isn't
# already bound.
class GenNone(Function):
    # noinspection PyUnusedLocal
    @staticmethod
    def fun(args, goal, start):
        return None

class GenList(Function):
    def __init__(self, outVars, inVars, listVals = [], isNecessary = None):
        super(GenList, self).__init__(outVars, inVars, isNecessary)
        self.listVals = listVals

    def fun(self, args, goal, start):
        return self.listVals

    def applyBindings(self, bindings):
        res = super(GenList, self).applyBindings(bindings)
        res.listVals = self.listVals
        return res

class Assign(Function):
    # noinspection PyUnusedLocal
    @staticmethod
    def fun(args, goal, start):
        return args

# Be sure the argument is not 'none'
class NotNone(Function):
    isNecessary = True
    # noinspection PyUnusedLocal
    @staticmethod
    def fun(args, goal, start):
        assert args[0] is not None
        if args[0] == 'none':
            return None
        else:
            return [[]]

class NotStar(Function):
    isNecessary = True
    # Be sure the argument is not '*'
    # noinspection PyUnusedLocal
    @staticmethod
    def fun(args, goal, start):
        if args[0] == '*':
            return None
        else:
            return [[]]

class NotEqual(Function):
    isNecessary = True
    # noinspection PyUnusedLocal
    @staticmethod
    def fun(args, goal, start):
        n = len(args)
        assert n%2 == 0
        m = n/2
        if args[:m] == args[m:]:
            result = None
        else:
            result = [[]]
        return result

class Subtract(Function):
    # noinspection PyUnusedLocal
    @staticmethod
    def fun((a, b), goal, start):
        if a == '*' or b == '*':
            return [['*']]
        result = tuple([aa - bb for (aa, bb) in zip(a, b)])
        if any([x < minDelta for x in [result[0], result[1], result[3]]]) or result[2] < 0.0:
            debugMsg('smallDelta', 'Delta would be negative or zero', result)
            return []
        return [[result]]
        
# Return as many values as there are args; overwrite any that are
# variables with the minimum value
class MinP(Function):
    # noinspection PyUnusedLocal
    @staticmethod
    def fun(args, goal, start):
        minVal = min([a for a in args if not isVar(a)])
        return [[minVal if isVar(a) else a for a in args]]

class ObsVar(Function):
    # noinspection PyUnusedLocal
    @staticmethod
    def fun(args, goal, start):
        return [[start.domainProbs.obsVar]]

# Regression:  what does the mode need to be beforehand, assuming a good
# outcome.  Don't let it go down too fast...
minProb = 0.5

class ObsModeProb(Function):
    # noinspection PyUnusedLocal
    @staticmethod
    def fun(args, goal, start):
        p = max([a for a in args if not isVar(a)])
        pFalsePos = pFalseNeg = start.domainProbs.obsTypeErrProb
        pr = p * pFalsePos / ((1 - p) * (1 - pFalseNeg) + p * pFalsePos)
        return [[max(minProb, pr, p - 0.2)]]

class RegressProb(Function):
    isNecessary = True
    # Compute the nth root of the maximum defined prob value.  Also, attenuated
    # by an error probability found in domainProbs
    def __init__(self, outVars, inVars, isNecessary = False, pn = None):
        self.n = len(outVars)
        self.probName = pn
        super(RegressProb, self).__init__(outVars, inVars, isNecessary)
    # noinspection PyUnusedLocal
    def fun(self, args, goal, start):
        failProb = getattr(start.domainProbs, self.probName) \
                                        if (self.probName is not None) else 0.0
        pr = max([a for a in args if not isVar(a)]) / (1 - failProb)
        # noinspection PyTypeChecker
        val = power(pr, 1.0/self.n)
        if val < maxProbValue:
            return [[val]*self.n]
        else:
            return []
    def applyBindings(self, bindings):
        return self.__class__([lookup(v, bindings) for v in self.outVars],
                              [lookup(v, bindings) for v in self.inVars],
                              self.isNecessary, self.probName)

class MaxGraspVarFun(Function):
    # noinspection PyUnusedLocal
    @staticmethod
    def fun((var,), goal, start):
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

class MoveConfDelta(Function):
    # noinspection PyUnusedLocal
    @staticmethod
    def fun(args, goal, start):
        return [[start.domainProbs.moveConfDelta]]

class DefaultPlaceDelta(Function):
    # noinspection PyUnusedLocal
    @staticmethod
    def fun(args, goal, start):
        return [[start.domainProbs.placeDelta]]

class RealPoseVar(Function):
    # noinspection PyUnusedLocal
    @staticmethod
    def fun((graspVar,), goal, start):
        placeVar = start.domainProbs.placeVar
        return [[tuple([gv+pv for (gv, pv) in zip(graspVar, placeVar)])]]

# Two constraints: one from the delta of the resulting pose and one
# from the tolerance for picking.
class GraspDelta(Function):
    # noinspection PyUnusedLocal
    @staticmethod
    def fun((poseDelta, confDelta, graspVar, p), goal, start):
        pickTol = start.domainProbs.pickTolerance
        numStdDevs =  sqrt(chiSqFromP(1-p, 3))
        graspStds = [sqrt(v) * numStdDevs for v in graspVar]
        maxForPose = [pd - cd for (pd, cd) in zip(poseDelta, confDelta)]
        maxForGraspTol = [gt - gstd for (gt, gstd) in zip(pickTol, graspStds)]

        result = tuple([min(d1, d2) for (d1, d2) in zip(maxForPose, maxForGraspTol)])
        if any([x < minDelta for x in [result[0], result[1], result[3]]]):
            debugMsg('smallDelta', 'Delta would be negative or zero', result)
            return []
        return [[result]]

# Realistically, push increases variance quite a bit.  For now, we'll just
# assume stdev needs to be halved
# Also have a max stdev
class PushPrevVar(Function):
    # noinspection PyUnusedLocal
    @staticmethod
    def fun((resultVar,), goal, start):
        pushVar = start.domainProbs.pushVar
        maxPushVar = start.domainProbs.maxPushVar
        # pretend it's lower
        res = tuple([min(x - y, m) for (x, y, m) \
                     in zip(resultVar, pushVar, maxPushVar)])
        #res = tuple([x/4.0 for x in resultVar])
        if any([v <= 0.0 for v in res]):
            tr('pushGenVar', 'Push previous var would be negative', res)
            return []
        else:
            return [[res]]
    
class PlaceInPoseVar(Function):
    # TODO: LPK: be sure this is consistent with moveOut
    # noinspection PyUnusedLocal
    @staticmethod
    def fun(args, goal, start):
        pv = [v * 2 for v in start.domainProbs.obsVarTuple]
        #pv = list(start.domainProbs.obsVarTuple)
        pv[2] = pv[0]
        return [[tuple(pv)]]

# Thing is a variance; compute a variance that corresponds to doubling
# the stdev.    (sqrt(v) * 2)^2 = v * 4
class StdevTimes2(Function):
    # noinspection PyUnusedLocal
    @staticmethod
    def fun((thing,), goal, start):
        return [[tuple([v*4 for v in thing])]]

class Times2(Function):
    # noinspection PyUnusedLocal
    @staticmethod
    def fun((thing,), goal, start):
        return [[tuple([v*2 for v in thing])]]

# For place, grasp var is desired poseVar minus fixed placeVar
# Don't let it be bigger than maxGraspVar
class PlaceGraspVar(Function):
    # noinspection PyUnusedLocal
    @staticmethod
    def fun((poseVar,), goal, start):
        maxGraspVar = start.domainProbs.maxGraspVar
        placeVar = start.domainProbs.placeVar
        graspVar = tuple([min(gv - pv, m) for (gv, pv, m) \
                                  in zip(poseVar, placeVar, maxGraspVar)])
        if any([x <= 0 for x in graspVar]):
            tr('placeVar', 'negative grasp var', ('poseVar', poseVar),
            ('placeVar', placeVar), ('maxGraspVar', maxGraspVar))
            return []
        else:
            return [[graspVar]]

# tol > n * sqrt(var) + d
# tol - d > n * sqrt(var)
# (tol - d) / n > sqrt(var)
# ((tol - d) / n)**2 > var

# For pick, pose var is desired graspVar minus fixed pickVar
class PickPoseVar(Function):
    # noinspection PyUnusedLocal
    @staticmethod
    def fun((graspVar, graspDelta, prob), goal, start):
        if graspDelta == '*':
            return [[graspVar]]
        pickVar = start.domainProbs.pickVar
        pickTolerance = start.domainProbs.pickTolerance
        # What does the variance need to be so that we are within
        # pickTolerance with probability prob?
        numStdDevs =  sqrt(chiSqFromP(1-prob, 3))
        # nstd * std < pickTol
        # std < pickTol / nstd
        tolerableVar = [((pt - gd - .001) / numStdDevs)**2 for \
                        (pt, gd) in zip(pickTolerance, graspDelta)]
        poseVar = tuple([min(gv - pv, tv) \
                         for (gv, pv, tv) in zip(graspVar, pickVar, tolerableVar)])

        if any([x <= 0 for x in poseVar]):
            tr('pickGenVar', 'failed',
               ('fixed pickVar', pickVar), ('tolerance', pickTolerance),
                ('num stdev', numStdDevs), ('tolerable var', tolerableVar),
                ('poseVar', poseVar))
            return []
        else:
            tr('pickGenVar',
                ('fixed pickVar', pickVar), ('tolerance', pickTolerance),
                ('num stdev', numStdDevs), ('tolerable var', tolerableVar),
                ('poseVar', poseVar),
                ('shadow width', shadowWidths(poseVar, graspDelta, prob)))
            return [[poseVar]]

# starting var if it's legal, plus regression of the result var.
# Need to try several, so that looking doesn't put the robot into the shadow!
class GenLookObjPrevVariance(Function):
    # noinspection PyUnusedLocal
    @staticmethod
    def fun((ve, obj, face), goal, start):
        lookVar = start.domainProbs.obsVarTuple
        odoVar = [e * e for e in start.domainProbs.odoError]
        
        if start.pbs.getHeld('left').mode() == obj or \
          start.pbs.getHeld('right').mode() == obj:
            vs = maxPoseVar
        else:
            vs = list(start.poseModeDist(obj, face).mld().sigma.diagonal().\
                      tolist()[0])
            vs[2] = .0001**2
            # This is the variance in the start state.  We have to be able
            # to move to look at it, which will increase the variance.
            # So increase it a bit here.
            vs = tuple([v + ov for (v, ov) in zip(vs, odoVar)])
        # Don't let variance get bigger than variance in the initial state, or
        # the cap, whichever is bigger
        cap = [max(a, b) for (a, b) in zip(maxPoseVar, vs)]
        vbo = varBeforeObs(lookVar, ve)
        cappedVbo1 = tuple([min(a, b) for (a, b) in zip(cap, vbo)])
        cappedVbo2 = tuple([min(a, b) for (a, b) in zip(vs, vbo)])
        # vbo > ve
        # This is useful if it's between:  vbo > vv > ve
        ve3 = (ve[0], ve[1], ve[3])
        cvbo3 = (cappedVbo1[0], cappedVbo2[1], cappedVbo1[3])
        def useful(vv):
            vv3 = (vv[0], vv[1], vv[3])
            # noinspection PyShadowingNames
            return any([a > b for (a, b) in zip(vv3, ve3)]) and \
                   any([a > b for (a, b) in zip(cvbo3, vv3)])
        def sqrts(vv):
            # noinspection PyShadowingNames
            return [sqrt(xx) for xx in vv]
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

        tr('genLookObsPrevVariance',
        ('Target', prettyString(sqrts(ve))),
        ('Capped before', prettyString(sqrts(cappedVbo1))),
        ('Other suggestions',
           [prettyString(sqrts(xx)[0]) for xx in result]))
        return result

class RealPoseVarAfterObs(Function):
    # noinspection PyUnusedLocal
    @staticmethod
    def fun((varAfter,), goal, start):
        obsVar = start.domainProbs.obsVarTuple
        thing = tuple([min(x, y) for (x, y) in zip(varAfter, obsVar)])
        return [[thing]]

'''
# starting var if it's legal, plus regression of the result var
# noinspection PyUnusedLocal
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
'''

'''
# Really just return true if reducing variance on the object in the
# hand will reduce the violations.
#####    Fix this!!!
# noinspection PyUnusedLocal
def canReachHandGen(args, goal, start, vals):
    (conf, p, cond, hand) = args
    f = CanReachHome([conf, cond], True)
    path, viol = f.getViols(start, True, p)
    if viol and viol.heldShadows[handI[hand]] != []:
        return [[]]
    else:
        return []

# Really just return true if putting down the object in the
# hand will reduce the violations.
# noinspection PyUnusedLocal
def canReachDropGen(args, goal, start, vals):
    (conf, p, cond) = args
    f = CanReachHome([conf, cond], True)
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
# noinspection PyUnusedLocal
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
                    tr('canPickPlaceDropGen', (heldO, dropHand),
                       ('held obstacles', collidesWithHeld), ('goal held', matches))
                    result.append([dropHand])
    return result
'''

################################################################
## Special regression funs
################################################################

attenuation = 0.5
# During regression, apply to all fluents in goal;  returns f or a new fluent.
# noinspection PyUnusedLocal
def moveSpecialRegress(f, details, abstractionLevel):

    # Only model these effects at the lower level of abstraction.
    # if abstractionLevel == 0:
    #     return f.copy()

    # Assume that odometry error is controlled during motion, so not more than 
    # this.  It's a stdev
    odoError = details.domainProbs.odoError
    # Variance due to odometry after move of a meter
    odoVar = [e * e for e in odoError]

    if f.predicate == 'B' and f.args[0].predicate == 'Pose':
        fNew = f.copy()
        newVar = tuple([v - e for (v, e) in zip(f.args[2], odoVar)])
        if any([nv <= 0.0 for nv in newVar]):
            tr('specialRegress',
               'Move special regress failing; cannot regress', f,
               f.args[2], odoVar, newVar)
            return None
        fNew.args[2] = newVar
        fNew.update()
        return fNew
        
    elif f.predicate == 'BLoc':
        targetVar = f.args[1]
        if any([tv < ov for (tv, ov) in zip(targetVar, odoVar)]):
            tr('specialRegress',
               'Move special regress failing; target var less than odo', f)
            return None
    return f.copy()

################################################################
## Cost funs
################################################################

# So many ways to do this...
def costFun(primCost, prob):
    if prob == 0:
        trAlways('costFun: prob = 0, returning inf')
    return float('inf') if prob == 0 else primCost / prob

# Cost depends on likelihood of success: canReach, plus objects in each hand
# Add in a real distance from cs to ce

# noinspection PyUnusedLocal
def moveCostFun(al, args, details):
    rawCost = 5
    result = costFun(rawCost, movePreProb)
    return result

# noinspection PyUnusedLocal
def moveNBCostFun(al, args, details):
    rawCost = 1 
    result = costFun(rawCost, movePreProb)
    return result

# noinspection PyUnusedLocal
def placeCostFun(al, args, details):
    rawCost = 10
    abstractCost = 20
    (_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,p1) = args
    if isVar(p1): p1 = 0.9
    result = costFun(rawCost,
                     p1*canPPProb*(1-details.domainProbs.placeFailProb)) + \
               (abstractCost if al == 0 else 0)
    return result

# noinspection PyUnusedLocal
def pushCostFun(al, args, details):
    # Make it cost more than pick/place for now, though not clear it
    # should always be like this.
    rawCost = 75
    abstractCost = 100
    (_, _, _, _, _, _, _, _, _, _, _, _, p, _, _)  = args
    result = costFun(rawCost,
                     p*canPPProb*(1-details.domainProbs.placeFailProb)) + \
               (abstractCost if al == 0 else 0)
    return result

# noinspection PyUnusedLocal
def pickCostFun(al, args, details):
    (o,h,pf,p,pd,gf,gm,gv,gd,prc,cd,pc,rgv,pv,p1,pr1,pr2,pr3) = args
    rawCost = 3
    abstractCost = 10
    result = costFun(rawCost, p1*canPPProb*canPPProb*\
                     (1 - details.domainProbs.pickFailProb)) + \
               (abstractCost if al == 0 else 0)
    return result

# Cost depends on likelihood of seeing the object and of moving the
# mean out of the delta bounds
# When we go to non-diagonal covariance, this will be fun...
# For now, just use the first term!
# noinspection PyUnusedLocal
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

# noinspection PyUnusedLocal
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

# noinspection PyUnusedLocal
def moveBProgress(details, args, obs=None):
    (s, e, _) = args
    # Let's say this is error per meter / radian
    odoError = details.domainProbs.odoError
    (endConf, (xyDisp, angDisp)) = obs
    obsConf = endConf.conf
    bp1 = (s['pr2Base'][0], s['pr2Base'][1], 0, s['pr2Base'][2])
    bp2 = (obsConf['pr2Base'][0], obsConf['pr2Base'][1], 0,
           obsConf['pr2Base'][2])
    # turn this up in case we have to 
    increaseFactor = 1
    # xyDisp is the total xy displacement along path
    # angDisp is the total angular displacement along path
    odoVar = ((xyDisp * increaseFactor * odoError[0])**2,
              (xyDisp * increaseFactor * odoError[1])**2,
              0.0,
              (angDisp * increaseFactor * odoError[3])**2)
    tr('beliefUpdate', 'About to move B progress', 
        ('start base', bp1),
        ('end base ', bp2), 
        ('odo error rate', odoError),
        ('xyDisp', xyDisp),
        ('added odo var', odoVar), 
        ('added odo stdev', [sqrt(v) for v in odoVar]), ol = False)
    
    # Change robot conf.  For now, trust the observation completely
    details.pbs.updateConf(endConf)

    for ob in details.pbs.moveObjBs.values() + \
               details.pbs.fixObjBs.values():
        oldVar = ob.poseD.var
        newVar = tuple([a + b for (a, b) in zip(oldVar, odoVar)])
        details.pbs.resetPlaceB(ob.modifyPoseD(var=newVar))
    details.pbs.reset()
    details.pbs.getShadowWorld(0)
    details.pbs.internalCollisionCheck()
    debugMsg('beliefUpdate', 'moveBel')

# noinspection PyUnusedLocal
def moveNBBProgress(details, args, obs=None):
    (b, s, e, _) = args
    # Just put the robot in the intended place
    details.pbs.updateConf(e)
    details.pbs.reset()
    details.pbs.getShadowWorld(0)
    details.pbs.internalCollisionCheck()
    debugMsg('beliefUpdate', 'moveBel')    

# noinspection PyUnusedLocal
def pickBProgress(details, args, obs=None):
    assert obs in ('success', 'failure')
    # Assume robot ends up at preconf, so no conf change
    (o,h,pf,p,pd,gf,gm,gv,gd,prc,cd,pc,rgv,pv,p1,pr1,pr2,pr3) = args
    pickVar = details.domainProbs.pickVar
    # TODO: Is this good?
    if obs == 'failure':
        failProb = 0.99
    else:
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

# noinspection PyUnusedLocal
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

# noinspection PyUnusedLocal
def pushBProgress(details, args, obs=None):
    # Conf of robot and pose of object change
    (o, h, pose, pf, pv, pd, pp, ppv, prec, pushc, postc, cd, p, pr1,pr2) = args

    failProb = details.domainProbs.pushFailProb
    pushVar = details.domainProbs.pushVar

    # Change robot conf
    details.pbs.updateConf(postc)

    v = [x + y for x,y in \
         zip(details.pbs.getPlaceB(o).poseD.varTuple(), pushVar)]
    v[2] = 1e-20
    gv = tuple(v)

    # Failure here would be to knock the object over
    details.poseModeProbs[o] = (1 - failProb) * details.poseModeProbs[o]
    ff = details.pbs.getWorld().getFaceFrames(o)
    details.pbs.moveObjBs[o] = ObjPlaceB(o, ff, pf, PoseD(pose, gv))
    details.pbs.reset()
    details.pbs.getShadowWorld(0)
    details.pbs.internalCollisionCheck()
    debugMsg('beliefUpdate', 'pushBel')
    
# obs has the form (obj-type, face, relative pose)
def lookAtBProgress(details, args, obs):
    (_, lookConf, _, _, _, _, _, _, _, _, _, _, _, _) = args
    objectObsUpdate(details, lookConf, obs)
    details.pbs.reset()
    details.pbs.getRoadMap().confReachCache = {} # Clear motion planning cache
    details.pbs.getShadowWorld(0)
    details.pbs.internalCollisionCheck()
    debugMsg('beliefUpdate', 'look')

#llMatchThreshold = -100.0
llMatchThreshold = -400.0

def objectObsUpdate(details, lookConf, obsList):
    def rem(l,x): return [y for y in l if y != x]
    prob = 0.95
    world = details.pbs.getWorld()
    shWorld = details.pbs.getShadowWorld(prob)
    rob = details.pbs.getRobot().placement(lookConf,
                                           attached=shWorld.attached)[0]
    fixed = shWorld.getNonShadowShapes() + [rob]
    obstacles = shWorld.getNonShadowShapes()
    # Objects that we expect to be visible
    heldLeft = details.pbs.held['left'].mode()
    heldRight = details.pbs.held['left'].mode()

    objList = [s for s in obstacles \
               if visible(shWorld, lookConf, s, rem(obstacles,s)+[rob], 0.5,
                          moveHead=False, fixed=fixed)[0] and \
                  s.name() not in (heldLeft, heldRight)]
    if len(objList) == 0:
        trAlways('Do not expect to see any objects!')
    scores = {}
    for obj in objList:
        scores[(obj, None)] = (-float('inf'), None, None)
        for obs in obsList:
            (oType, obsFace, obsPose) = obs
            if world.getObjType(obj.name()) != oType: continue
            scores[(obj, obs)] = scoreObsObj(details, obs, obj.name())
    # bestAssignment = argmax(allAssignments(objList, obsList),
    #                          lambda a: scoreAssignment(a, scores))
    bestAssignment = greedyBestAssignment(scores)
    assert len(obsList)==0 or \
         any([xpose for (xobj, xpose, xf) in bestAssignment]), \
         'Best assignment maps all objects to no observation'
    for obj, obsPose, obsFace in bestAssignment:
        singleTargetUpdate(details, obj.name(), obsPose, obsFace)

    if debug('pbsId'):
        print 'Just updated pbs', id(details.pbs)
        raw_input('Okay?')

# Each object is assigned an observation or None
# These are objects that we expected to be able to see.
# Doesn't initialize new objects

def allAssignments(aList, bList):
    m = len(aList)
    n = len(bList)
    if m > n:
        bList = bList + [None]*(m - n)
    
    def ablate(elts, pp):
        return map(lambda e, p: e if p else None, elts, pp)
    
    return set(imap(lambda bs: tuple(zip(aList, bs)), 
                 chain(*[chain([ablate(bPerm, pat) for pat in \
                          product((True, False), repeat = n)])\
                     for bPerm in permutations(bList)])))


# For now, assume we do not see the object in the hand
# noinspection PyShadowingNames
def greedyBestAssignment(scores):
    result = []
    scoresLeft = scores.items()
    while scoresLeft:
        ((obj, obs), (val, tobs, face)) = argmax(scoresLeft,
                                              lambda ((oj, os), (v, o, f)): v)
        if val < llMatchThreshold:
            result.append((obj, None, None))
            tr('assign', 'Fail', val, obj.name(), tobs)
            return result
        else:
            tr('assign', prettyString(val), obj.name(), tobs, old = True)
        result.append((obj, tobs, face))
        # Better not to copy so much
        scoresLeft = [((oj, os), stuff) for ((oj, os), stuff) in scoresLeft \
                      if oj != obj and (os != obs or obs is None)]
    return result

def obsDist(details, obj):
    obsVar = details.domainProbs.obsVarTuple
    objBel = details.pbs.getPlaceB(obj)
    poseFace = objBel.support.mode()
    poseMu = objBel.poseD.modeTuple()
    var = objBel.poseD.variance()
    obsCov = [v1 + v2 for (v1, v2) in zip(var, obsVar)]
    obsPoseD = MultivariateGaussianDistribution(np.mat(poseMu).T,
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

    if obsPose is None:

        # print '*****  Big hack until we get observation prediction right ****'
        # print 'Expected to see', objName, 'but did not'
        # return

        print 'Expected to see', objName, 'but did not'
        # Update modeprob if we don't get a good score
        oldP = details.poseModeProbs[objName]
        obsGivenH = details.domainProbs.obsTypeErrProb
        obsGivenNotH = (1 - details.domainProbs.obsTypeErrProb)
        newP = obsGivenH * oldP / (obsGivenH * oldP + obsGivenNotH * (1 - oldP))
        details.poseModeProbs[objName] = newP
        tr('assign',  'No match above threshold', objName, oldP, newP,
           ol = True)
        newMu = oldPlaceB.poseD.modeTuple()
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
        tr('assign', 'Obs match for', objName, oldP, newP, ol = True)
        # Should update face!!
        # Update mean and sigma
        ## Be sure handling angle right.
        (newMu, newSigma) = \
                    gaussObsUpdate(oldPlaceB.poseD.modeTuple(),
                                   obsPose.pose().xyztTuple(),
                                   oldPlaceB.poseD.variance(), obsVar)
        ns = list(newSigma); ns[2] = 1e-10; newSigma = tuple(ns)
        ff = w.getFaceFrames(objName)[obsFace]

        ## LPK!!  Should really draw the detected object but I don't have
        ## an immediate way to get the shape of a type.  Should fix that.
        tr('obsUpdate', (objName, obsPose, obsFace),
            'obs is Cyan; newMu is magenta',
            draw = [(details.pbs.getObjectShapeAtOrigin(objName).\
                        applyLoc(obsPose.pose().compose(ff.inverse())), 
                        'Belief', 'cyan'),
                    (details.pbs.getObjectShapeAtOrigin(objName).\
                         applyLoc(hu.Pose(*newMu).compose(ff.inverse())),
                         'Belief', 'magenta')],
               snap = ['Belief'])

    details.pbs.resetPlaceB(ObjPlaceB(objName, w.getFaceFrames(objName),
                                      DeltaDist(oldPlaceB.support.mode()),
                                      PoseD(hu.Pose(*newMu), newSigma)))

    if newP < 0.3:
        print 'Object has gotten lost and has very low probability in'
        print 'the current mode!!!!'
        print objName
        raw_input('Woe.')

    
missedObsLikelihoodPenalty = llMatchThreshold

# Return a triple:  (score, transObs, face)
# transObs might be a transformed version of obs (picking the best member
# of the class of symmetries for this object)
# face is a canonical face
def scoreObsObj(details, obs, objct):
    if obs is None:
        return missedObsLikelihoodPenalty, None, None

    # Don't consider matches against objects in the hand
    heldLeft = details.pbs.held['left'].mode()
    heldRight = details.pbs.held['right'].mode()
    if objct in (heldLeft, heldRight):
        return -float('inf'), None, None
    
    (oType, obsFace, obsPose) = obs
    (obsPoseD, poseFace) = obsDist(details, objct)
    pbs = details.pbs
    w = pbs.beliefContext.world
    symFacesType, symXformsType = w.getSymmetries(oType)
    canonicalFace = symFacesType[obsFace]
    symXForms = symXformsType[canonicalFace]
    # Consider all object poses consistent with this observation
    symPoses = [obsPose] + [obsPose.compose(xf) for xf in symXForms]
    # ff = pbs.getWorld().getFaceFrames(objct)[canonicalFace]

    # Find the best matching pose mode.  Type and face must be equal,
    # pose nearby.
    #!!LPK handle faces better

    # Type
    if w.getObjType(objct) != oType:
        # noinspection PyUnresolvedReferences
        return -float('inf'), None, None
    # Face
    assert symFacesType[poseFace] == poseFace, 'non canonical face in bel'
    if poseFace != canonicalFace:
        # noinspection PyUnresolvedReferences
        return -float('inf'), None, None
    # Iterate over symmetries for this object
    # noinspection PyUnresolvedReferences
    bestObs, bestLL = None, -float('inf')
    for obsPoseCand in symPoses:
        ll = float(obsPoseD.logProb(np.mat(obsPoseCand.pose().xyztTuple()).T))
        if ll > bestLL:
            bestObs, bestLL = obsPoseCand, ll
    return bestLL, bestObs, canonicalFace

# Temporary;  assumes diagonal cov; should use dist.MultivariateGaussian
def gaussObsUpdate(oldMu, obs, oldSigma, obsVariance, noZ = True):
    # All tuples
    newMu = [(m * obsV + op * muV) / (obsV + muV) \
                       for (m, muV, op, obsV) in \
                       zip(oldMu, oldSigma, obs, obsVariance)]
    # That was not the right way to do the angle update!  Quick and dirty here.
    oldTh, obsTh, muV, obsV = oldMu[3], obs[3], oldSigma[3], obsVariance[3]
    oldX, oldY = cos(oldTh), sin(oldTh)
    obsX, obsY = cos(obsTh), sin(obsTh)
    newX = (oldX * obsV + obsX * muV) / (obsV + muV)
    newY = (oldY * obsV + obsY * muV) / (obsV + muV)
    newTh = arctan2(newY, newX)
    newMu[3] = newTh
    newSigma = tuple([(a * b) / (a + b) for (a, b) in zip(oldSigma,obsVariance)])
    if noZ:
        newMu[2] = oldMu[2]
    return tuple(newMu), newSigma
    
# For now, assume obs has the form (obj, face, grasp) or None
# Really, we'll get obj-type, face, relative pose
# noinspection PyTypeChecker
def lookAtHandBProgress(details, args, obs):
    (_, h, _, f, _, _, _,  _, _, _, _,_) = args
    # Awful!  Should have an attribute of the object
    universe = [o for o in details.pbs.beliefContext.world.objects.keys() \
                if o[0:3] == 'obj']
    heldDist = details.pbs.held[h]

    oldMlo = heldDist.mode()
    # Update dist on what we're holding if we got an observation
    if obs is not None:
        obsObj = 'none' if obs is 'none' else obs[0]
        # Observation model
        def om(trueObj):
            return MixtureDist(DeltaDist(trueObj),
                               UniformDist(universe + ['none']),
                               1 - details.domainProbs.obsTypeErrProb)
        heldDist.obsUpdate(om, obsObj)

        # If we are fairly sure of the object, update the mode object's dist
        mlo = heldDist.mode()
        bigSigma = (0.01, 0.01, 0.01, 0.04)
        faceDist = details.pbs.graspB[h].grasp
        gd = details.pbs.graspB[h].graspDesc
        if mlo == 'none':
            # We think we are not holding anything
            newOGB = None
        # If we now have a new mode, we have to reinitialize the grasp dist!
        elif mlo != oldMlo:
            raw_input('Fix observation update to handle new mlo')
            #details.graspModeProb[h] = 1 - details.domainProbs.obsTypeErrProb
            #gd = details.pbs.graspB[h].graspDesc
            newOGB = ObjGraspB(mlo, gd, faceDist, None, 
                               PoseD(hu.Pose(0.0, 0.0, 0.0, 0.0), bigSigma))
        elif obsObj == 'none':
            # obj, graspDesc, graspD, poseD
            poseDist = details.pbs.graspB[h].poseD
            oldPose = poseDist.modeTuple()
            newOGB = ObjGraspB(mlo, gd, faceDist, PoseD(oldPose, bigSigma))
        else:
            # we observed the same object as the current mode; do a
            # belief update on the grasp
            (_, ogf, ograsp) = obs            
            # Bayes update on mode prob
            oldP = details.graspModeProb[h]
            obsGivenH = (1 - details.domainProbs.obsTypeErrProb)
            obsGivenNotH = details.domainProbs.obsTypeErrProb
            newP = obsGivenH * oldP / \
                        (obsGivenH * oldP + obsGivenNotH * (1 - oldP))
            details.graspModeProb[h] = newP
            # Update the rest of the distributional info.
            # Consider only doing this if the mode prob is high
            #mlop = heldDist.prob(mlo)
            oldMlf = faceDist.mode()
            # Should do an update, but since we only have one grasp for now it
            # doesn't make sense
            # faceDist.obsUpdate(fom, ogf)
            faceDist = DeltaDist(ogf)

            mlf = faceDist.mode()
            poseDist = details.pbs.graspB[h].poseD
            oldMu = poseDist.modeTuple()
            oldSigma = poseDist.variance()

            if mlf != oldMlf:
                # Most likely grasp face changed.
                # Keep the pose for lack of a better idea, but
                # increase sigma a lot!
                newPoseDist = PoseD(poseDist.mode(), bigSigma)
                trAlways('Grasp face changed.  Probably not okay')
            else:
                # Cheapo obs update
                obsVariance = details.domainProbs.obsVarTuple
                newMu = tuple([(m * obsV + op * muV) / (obsV + muV) \
                       for (m, muV, op, obsV) in \
                       zip(oldMu, oldSigma, ograsp, obsVariance)])
                newSigma = tuple([(a * b) / (a + b) for (a, b) in \
                                  zip(oldSigma,obsVariance)])
                newPoseDist = PoseD(hu.Pose(*newMu), newSigma)
            newOGB = ObjGraspB(mlo, gd, faceDist, None, newPoseDist)
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
move = Operator(
    'Move',
    ['CStart', 'CEnd', 'DEnd'],
    # Pre
    {0 : {Bd([CanReachHome(['CEnd', []]),  True, movePreProb], True)},
     1 : {Conf(['CStart', 'DEnd'], True)}},
    # Results:  list of pairs: (fluent set, private preconds)
    [({Conf(['CEnd', 'DEnd'], True)}, {})],
    functions = [GenNone(['CEnd'], [])],
    cost = moveCostFun,
    f = moveBProgress,
    prim  = movePrim,
    # Can't use conditional results here, because there are
    # arbitrarily many instances of the Pose fluent that need a
    # special regression condition.
    specialRegress = moveSpecialRegress,
    argsToPrint = [0, 1],
    ignorableArgs = range(3), # For hierarchy
    ignorableArgsForHeuristic = (0, 2),
    rebindPenalty = 100
    )


# Not allowed to move base
moveNB = Operator(
    'MoveNB',
    ['Base', 'CStart', 'CEnd', 'DEnd'],
    # Pre
    {0 : {Bd([CanReachNB(['CStart', 'CEnd', []]),  True, movePreProb], True),
          Conf(['CStart', 'DEnd'], True),
          BaseConf(['Base', 'DEnd'], True)
             }},
    # Results:  list of pairs: (fluent set, private preconds)
    [({Conf(['CEnd', 'DEnd'], True)}, {})],
    functions = [
        GenNone(['CEnd'], []),
        GetBase(['Base'], ['CEnd'])],
    cost = moveNBCostFun,
    f = moveNBBProgress,
    prim  = moveNBPrim,
    argsToPrint = [0],
    ignorableArgs = range(4), # For hierarchy
    ignorableArgsForHeuristic = (1, 3),
    rebindPenalty = 100
    )

# All this work to say you can know the location of something by knowing its
# pose or its grasp
bLoc1 = Operator(
         'BLocPose', ['Obj', 'Var', 'P'],
         {0 : {B([Pose(['Obj', '*']), '*', 'Var', '*', 'P'], True),
               Bd([SupportFace(['Obj']), '*', 'P'], True)}},
         [({BLoc(['Obj', 'Var', 'P'], True)}, {})],
         ignorableArgs = (1, 2),
         ignorableArgsForHeuristic = (1, 2),
         rebindPenalty = 100)

bLoc2 = Operator(
         'BLocGrasp', ['Obj', 'Var', 'Hand', 'P'],
         {0 : {Graspable(['Obj'], True),
               B([Grasp(['Obj', 'Hand', '*']), '*', 'Var', '*', 'P'], True),
               Bd([Holding(['Hand']), 'Obj', 'P'], True),
               Bd([GraspFace(['Obj', 'Hand']), '*', 'P'], True)}},
         [({BLoc(['Obj', 'Var', 'P'], True)}, {})],
         functions = [GenList(['Hand'], [],
                              listVals = [['left'], ['right']])],
         ignorableArgs = (1, 3),
         ignorableArgsForHeuristic = (1, 3),
         rebindPenalty = 100)

poseAchIn = Operator(
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
            functions = [
                RegressProb(['P1', 'P2'], ['PR']),
                # Object region is defined wrto
                RegionParent(['Obj2'], ['Region']),
                DefaultPlaceDelta(['PoseDelta'], []),
                # Assume the base doesn't move
                PoseInStart(['PoseFace2', 'ObjPose2'], ['Obj2']),
                PlaceInPoseVar(['PoseVar'], []),
                #StdevTimes2(['TotalVar'], ['PoseVar']),
                Times2(['TotalVar'], ['PoseVar']),
                Times2(['TotalDelta'], ['PoseDelta']),
                # call main generator
                PoseInRegionGen(['ObjPose1', 'PoseFace1'],
                   ['Obj1', 'Region', 'TotalVar', 'TotalDelta',
                    probForGenerators])],
            argsToPrint = [0, 1],
            ignorableArgs = range(2,14),
            ignorableArgsForHeuristic = range(2, 14),
            conditionOnPreconds = True,
            rebindPenalty = 100)

placeArgs = ['Obj', 'Hand',
         'PoseFace', 'Pose', 'PoseVar', 'RealPoseVar', 'PoseDelta',
         'GraspFace', 'GraspMu', 'GraspVar', 'GraspDelta',
         'PreConf', 'ConfDelta', 'PlaceConf', 'AwayRegion',
         'PR1', 'PR2', 'PR3', 'P1']
def placeOp(*args):
    assert len(args) == len(placeArgs)
    newB = dict([(a, v) for (a, v) in zip(placeArgs, args) if a != v]) 
    return place.applyBindings(newB)

place = Operator('Place', placeArgs,
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
        [({Bd([SupportFace(['Obj']), 'PoseFace', 'PR1'], True),
           B([Pose(['Obj', 'PoseFace']), 'Pose', 'PoseVar', 'PoseDelta','PR2'],
                 True)},{}),
         ({Bd([Holding(['Hand']), 'none', 'PR3'], True)}, {})],
        functions = [
            # Not appropriate when we're just trying to decrease variance
            NotStar([], ['Pose']),
            NotStar([], ['PoseFace']),
            # Get object.  Only if the var is unbound.  Try first the
            # objects that are currently in the hands.
            GetObj(['Obj'], ['Hand']),
            # Be sure all result probs are bound.  At least one will be.
            MinP(['PR1', 'PR2', 'PR3'], ['PR1', 'PR2', 'PR3']),
            # Compute precond probs.  Assume that crash is observable.
            # So, really, just move the obj holding prob forward into
            # the result.  
            RegressProb(['P1'], ['PR1', 'PR2', 'PR3'], pn = 'placeFailProb'),
            # In case not specified
            PlaceInPoseVar(['PoseVar'], []),
            # PoseVar = GraspVar + PlaceVar,
            # GraspVar = min(maxGraspVar, PoseVar - PlaceVar)
            PlaceGraspVar(['GraspVar'], ['PoseVar']),
            # Real pose var might be much less than pose var if the
            # original pos var was very large
            # RealPoseVar = GraspVar + PlaceVar
            RealPoseVar(['RealPoseVar'],['GraspVar']),
            # In case PoseDelta isn't defined
            DefaultPlaceDelta(['PoseDelta'], []),
            # Assume fixed conf delta
            MoveConfDelta(['ConfDelta'], []),
            # Grasp delta is min of  poseDelta - confDelta and
            # pickTolerance - shadow(realPoseVar, prob)
            GraspDelta(['GraspDelta'], ['PoseDelta', 'ConfDelta',
                                        'GraspVar', 'P1']),
            # Not modeling the fact that the object's shadow should
            # grow a bit as we move to pick it.   Build that into pickGen.
            PlaceGen(['Hand','GraspMu', 'GraspFace', 'PlaceConf', 'PreConf',
                      'Pose', 'PoseFace'],
                     ['Obj', 'Hand', 'Pose', 'PoseFace', 'RealPoseVar',
                      'GraspVar', 'PoseDelta', 'GraspDelta', 'ConfDelta',
                     probForGenerators])],
        cost = placeCostFun,
        f = placeBProgress,
        prim = placePrim,
        argsToPrint = range(4),
        ignorableArgs = range(1, 19),   # all place of same obj at same level
        ignorableArgsForHeuristic = range(4, 19),
        rebindPenalty = 30)
        

pushArgs = ['Obj', 'Hand', 'Pose', 'PoseFace', 'PoseVar', 'PoseDelta',
            'PrePose', 'PrePoseVar', 'PreConf', 'PushConf', 'PostConf',
            'ConfDelta', 'P', 'PR1', 'PR2']

# make an instance of the push operation with given arguments
def pushOp(*args):
    assert len(args) == len(pushArgs)
    newB = dict([(a, v) for (a, v) in zip(pushArgs, args) if a != v]) 
    return push.applyBindings(newB)

# TODO : LPK think through the deltas more carefully
pushDelta = (0.01, 0.01, 1e-4, 0.4)
push = Operator('Push', pushArgs,
        {0 : {Pushable(['Obj'], True),
              BLoc(['Obj', planVar, 'P'], True)},    # was planP
         1 : {Bd([CanPush(['Obj', 'Hand', 'PoseFace', 'PrePose', 'Pose',
                           'PreConf',
                            'PushConf', 'PostConf', 'PoseVar', 'PrePoseVar',
                            'PoseDelta', []]), True, canPPProb],True)},
        2 : {Bd([SupportFace(['Obj']), 'PoseFace', 'P'], True),
              B([Pose(['Obj', 'PoseFace']), 'PrePose',
                 'PrePoseVar',  pushDelta, 'P'], True)},
        3 : {Conf(['PreConf', 'ConfDelta'], True),
             Bd([Holding(['Hand']), 'none', canPPProb], True)}
        },
        # Results
        [({Bd([SupportFace(['Obj']), 'PoseFace', 'PR1'], True),
           B([Pose(['Obj', 'PoseFace']), 'Pose', 'PoseVar', 'PoseDelta','PR2'],
                 True)},{})],
        sideEffects = {3 : {Conf(['PostConf', 'ConfDelta'], True)}},
        functions = [
            # Not appropriate when we're just trying to decrease variance,
            # at least, for now.
            NotStar([], ['Pose']),
            NotStar([], ['PoseFace']),

            RegressProb(['P'], ['PR1', 'PR2'], pn = 'pushFailProb'),
            PushPrevVar(['PrePoseVar'], ['PoseVar']),
            MoveConfDelta(['ConfDelta'], []),
            PushGen(['Hand','PrePose', 'PreConf', 'PushConf', 'PostConf'],
                     ['Obj', 'Pose', 'PoseFace', 'PoseVar', 'PoseDelta',
                      'ConfDelta', probForGenerators])],
        cost = pushCostFun,
        f = pushBProgress,
        prim = pushPrim,
        argsToPrint = range(4),
        ignorableArgs = range(1, 19),
        ignorableArgsForHeuristic = range(3, 19),
        rebindPenalty = 30)

# Put the condition to know the pose precisely down at the bottom to
# try to decrease replanning.

# Debate about level 2 vs level 1 preconds.

# We want the holding none precond at the same level as pose, if the
# object is currently in the hand.

pick = Operator(
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
                 'P1'], True),
              Bd([Holding(['Hand']), 'none', 'P1'], True)},
         1 : {Bd([CanPickPlace(['PreConf', 'PickConf', 'Hand', 'Obj', 'Pose',
                               'PoseVar', 'PoseDelta', 'PoseFace',
                               'GraspFace', 'GraspMu', 'RealGraspVar',
                               'GraspDelta', 'pick', []]), True, canPPProb],
                               True)},
#              Bd([Holding(['Hand']), 'none', canPPProb], True)},
         3 : {Conf(['PreConf', 'ConfDelta'], True),
              B([Pose(['Obj', 'PoseFace']), 'Pose', 'PoseVar', 'PoseDelta',
                 'P1'], True)              
             }},

        # Results
        [({Bd([Holding(['Hand']), 'Obj', 'PR1'], True), 
           Bd([GraspFace(['Obj', 'Hand']), 'GraspFace', 'PR2'], True),
           B([Grasp(['Obj', 'Hand', 'GraspFace']),
             'GraspMu', 'GraspVar', 'GraspDelta', 'PR3'], True)}, {})],
        functions = [
            # Be sure obj is not none -- don't use this to empty the hand
            NotNone([], ['Obj']),
            # Be sure grasp is not *, we don't know what to do
            NotStar([], ['GraspMu']),

            # Be sure all result probs are bound.  At least one will be.
            MinP(['PR1', 'PR2', 'PR3'], ['PR1', 'PR2', 'PR3']),
            # Compute precond probs.  Only regress object placecement P1.
            # Consider failure prob
            RegressProb(['P1'], ['PR1', 'PR2'], pn = 'pickFailProb'),
            MaxGraspVarFun(['RealGraspVar'], ['GraspVar']),
            # Assume fixed conf delta
            MoveConfDelta(['ConfDelta'], []),
            # Subtract off conf delta
            Subtract(['PoseDelta'], ['GraspDelta', 'ConfDelta']),
            # GraspVar = PoseVar + PickVar
            # prob was pr3, but this keeps it tighter
            PickPoseVar(['PoseVar'], ['RealGraspVar', 'GraspDelta', probForGenerators]),
            # Generate object pose and two confs
            PickGen(['Pose', 'PoseFace', 'PickConf', 'PreConf'],
                     ['Obj', 'GraspFace', 'GraspMu',
                      'PoseVar', 'RealGraspVar', 'PoseDelta', 'ConfDelta',
                      'GraspDelta', 'Hand', probForGenerators])],
        cost = pickCostFun,
        f = pickBProgress,
        prim = pickPrim,
        argsToPrint = [0, 1, 5, 3, 9],
        ignorableArgs = range(1, 18),
        ignorableArgsForHeuristic = range(4, 18),
        rebindPenalty = 40)

# We know that the resulting variance will always be less than obsVar.
# Would like the result to be the min of PoseVarAfter (which is in the
# goal) and obsVar.  Trying to make two different
# results...unfortunate increase in branching factor unless
# applicableOps handles them well.

lookAtArgs = ['Obj', 'LookConf', 'PoseFace', 'Pose',
     'PoseVarBefore', 'PoseDelta', 'PoseVarAfter', 'RealPoseVarAfter',
     'ConfDelta', 'ObsVar',
     'P1', 'PR0', 'PR1', 'PR2']

# make an instance of the lookAt operation with given arguments
def lookAtOp(*args):
    assert len(args) == len(lookAtArgs)
    newB = dict([(a, v) for (a, v) in zip(lookAtArgs, args) if a != v]) 
    return lookAt.applyBindings(newB)

lookAt = Operator(
    'LookAt', lookAtArgs,
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
    functions = [
        # In case these aren't bound
        Assign(['PoseFace'], [['*']]),
        Assign(['Pose'], [['*']]),
        Assign(['PoseDelta'], [['*']]),
        MoveConfDelta(['ConfDelta'], []),
        ObsVar(['ObsVar'], []),
        RealPoseVarAfterObs(['RealPoseVarAfter'], ['PoseVarAfter']),
        # Look increases probability.
        ObsModeProb(['P1'], ['PR0', 'PR1', 'PR2']),
        # How confident do we need to be before the look?
        GenLookObjPrevVariance(['PoseVarBefore'],
                               ['RealPoseVarAfter', 'Obj', 'PoseFace']),
        LookGen(['LookConf'],
                 ['Obj', 'Pose', 'PoseFace', 'PoseVarBefore',
                  'RealPoseVarAfter', 'PoseDelta',
                  'ConfDelta', probForGenerators])],
    cost = lookAtCostFun,
    f = lookAtBProgress,
    prim = lookPrim,
    argsToPrint = [0, 1, 3],
    ignorableArgs = [1, 2] + range(5, 14),   # change 5 to 4 to ignore var
    ignorableArgsForHeuristic = [1, 2] + range(4, 14),
    rebindPenalty = 100)
    

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
######################################################################
#
# Meta generators:  make implicit fluents true
#
######################################################################

# The idea here is that there are infinitely many actions we could possibly
# take.  Of those, only some subset can make a canXX fluent true or reduce
# the number of violations it has.

# Our original strategy was to suggest conditions which, when added to the
# fluent would reduce the nubmer of viols.  It's relatively easy to generate
# such conditions, but more difficult to ensure that they are feasible.  To show
# that they are feasible we must construct a (fragment of a) plan for making
# them true.   Given that we already construct such a plan, we decide, here
# to directly suggest the final operation of that plan.  One might imagine,
# for longer plans, generating more than a single operation.

# Returns an operator and the new condition that should be added to the
# conditional operator by this fluent.

class AchCanReachGen(Function):
    @staticmethod
    def fun(args, goal, start):
        tag = 'canReachGen'
        (conf, prob, cond) = args
        crhFluent = Bd([CanReachHome([conf, cond]), True, prob], True)
        def violFn(pbs):
            p, v = canReachHome(pbs, conf, prob, Violations())
            return v
        for newCond in \
              achCanXGen(start.pbs, goal, cond, [crhFluent], violFn, prob, tag):
            if not State(goal).isConsistent(newCond):
                print 'AchCanReach suggestion inconsistent with goal'
                for c in newCond: print c
                debugMsg(tag, 'Inconsistent')
            else:
                yield [newCond]

class AchCanReachNBGen(Function):
    @staticmethod
    def fun(args, goal, start):
        tag = 'canReachNBGen'
        (startConf, endConf, prob, cond) = args
        crFluent = Bd([CanReachNB([startConf, endConf, cond]), True, prob],
                       True)
        def violFn(pbs):
            p, v = canReachNB(pbs, startConf, endConf, prob, Violations())
            return v
        for newCond in \
              achCanXGen(start.pbs, goal, cond, [crFluent], violFn, prob, tag):
            if not State(goal).isConsistent(newCond):
                print 'AchCanReachNB suggestion inconsistent with goal'
                for c in newCond: print c
                debugMsg(tag, 'Inconsistent')
            else:
                yield [newCond]

class AchCanPickPlaceGen(Function):
    @staticmethod
    def fun(args, goal, start):
        tag = 'canPickPlaceGen'
        (preconf, ppconf, hand, obj, pose, realPoseVar, poseDelta, poseFace,
         graspFace, graspMu, graspVar, graspDelta, op, prob, cond) = args
        cppFluent = Bd([CanPickPlace([preconf, ppconf, hand, obj, pose,
                                      realPoseVar, poseDelta, poseFace,
                                      graspFace, graspMu, graspVar, graspDelta,
                                      op, cond]), True, prob], True)
        poseFluent = B([Pose([obj, poseFace]), pose, realPoseVar,
                        poseDelta, prob], True)
        
        # cppFluent is here so we can get the right reachObsts
        # for picking, we have to nail down the obj at the pose we want to pick from
        addedConditions = [cppFluent, poseFluent] if op == 'pick' else [cppFluent]

        world = start.pbs.getWorld()
        graspB = ObjGraspB(obj, world.getGraspDesc(obj), graspFace, poseFace,
                           PoseD(graspMu, graspVar), delta= graspDelta)
        placeB = ObjPlaceB(obj, world.getFaceFrames(obj), poseFace,
                           PoseD(pose, realPoseVar), delta=poseDelta)
        
        def violFn(pbs):
            v, r = canPickPlaceTest(pbs, preconf, ppconf, hand,
                                    graspB, placeB, prob, op=op)
            return v
        return achCanXGen(start.pbs, goal, cond, addedConditions,
                          violFn, prob, tag)

class AchCanPushGen(Function):
    @staticmethod
    def fun(args, goal, start):
        tag = 'canPushGen'
        (obj, hand, poseFace, prePose, pose, preConf, pushConf, postConf,
         poseVar, prePoseVar, poseDelta, prob, cond) = args

        tr(tag, 'args', args)

        # Preconditions
        cpFluent = Bd([CanPush([obj, hand, poseFace, prePose, pose, preConf,
                                pushConf, postConf, poseVar, prePoseVar,
                                poseDelta, cond]), True, prob], True)
        poseFluent = B([Pose([obj, poseFace]), prePose, prePoseVar,
                        poseDelta, prob], True)
        
        def violFn(pbs):
            path, v = canPush(pbs, obj, hand, poseFace, prePose, pose,
                                   preConf, pushConf, postConf, prePoseVar,
                                   poseVar,
                                   poseDelta, prob, Violations())
            return v
        return achCanXGen(start.pbs, goal, cond, [cpFluent, poseFluent],
                          violFn, prob, tag)

    
# violFn specifies what we are trying to achieve tries all the ways we
# know how to achieve it targetFluents are the declarative version of
# the same condition; would be better if we didn't have to specify it
# both ways.

def achCanXGen(pbs, goal, originalCond, targetFluents, violFn, prob, tag):
        allConds = list(originalCond) + targetFluents
        newBS = pbs.conditioned(goal, allConds)
        shWorld = newBS.getShadowWorld(prob)
            
        viol = violFn(newBS)
        tr(tag, ('viol', viol), draw=[(newBS, prob, 'W')], snap=['W'])
        if viol is None:                  # hopeless
            trAlways('Impossible dream', pause = True); return []
        if viol.empty():
            tr(tag, '=> No obstacles or shadows; returning'); return []

        if debug('nagLeslie'):
            print 'need to see if base pose is specified and pass it in'

        lookG = lookAchCanXGen(newBS, shWorld, viol, violFn, prob)
        placeG = placeAchCanXGen(newBS, shWorld, viol, violFn, prob,
                                 allConds + goal)
        pushG = pushAchCanXGen(newBS, shWorld, viol, violFn, prob,
                               allConds + goal)
        # prefer looking
        return itertools.chain(lookG, roundrobin(placeG, pushG))
    

def lookAchCanXGen(newBS, shWorld, initViol, violFn, prob):
    tag = 'lookAchGen'
    reducibleShadows = [sh.name() for sh in initViol.allShadows() \
                        if (not sh.name() in shWorld.fixedObjects) and \
                        (not objectName(sh.name()) in \
                          [obst.name() for obst in initViol.allObstacles()])]

    if not reducibleShadows:
        tr(tag, '=> No shadows to fix')
        return       # nothing available

    obsVar = newBS.domainProbs.obsVarTuple
    lookDelta = newBS.domainProbs.shadowDelta
    for shadowName in reducibleShadows:
        obst = objectName(shadowName)
        objBMinVar = newBS.domainProbs.objBMinVar(obst)
        placeB = newBS.getPlaceB(obst)
        tr(tag, '=> reduce shadow %s (in red):'%obst,
           draw=[(newBS, prob, 'W'),
           (placeB.shadow(newBS.getShadowWorld(prob)), 'W', 'red')],
           snap=['W'])
        face = placeB.support.mode()
        poseMean = placeB.poseD.modeTuple()
        conds = frozenset([Bd([SupportFace([obst]), face, prob], True),
                           B([Pose([obst, face]), poseMean, objBMinVar,
                              lookDelta, prob], True)])
        resultBS = newBS.conditioned([], conds)
        resultViol = violFn(resultBS)
        if resultViol is not None and shadowName not in resultViol.allShadows():
            yield [conds]
        else:
            trAlways('Error? Looking could not dispel shadow')
    tr(tag, '=> Out of remedies')


# noinspection PyUnusedLocal
def ignore(thing):
    pass

def fixedHeld(pbs, obj):
    for hand in ('left', 'right'):
        if obj == pbs.held[hand].mode() and pbs.fixHeld[hand]:
            return True
    return False

def placeAchCanXGen(newBS, shWorld, initViol, violFn, prob, cond):
    tag = 'placeAchGen'
    obstacles = [o.name() for o in initViol.allObstacles() \
                  if (not o.name() in shWorld.fixedObjects) and \
                     (not fixedHeld(newBS, o.name())) and \
                     graspable(o.name())]
    if obstacles:
        tr(tag, '=> Pickable obstacles to fix', obstacles)
    else:
        tr(tag, '=> No pickable obstacles to fix')
        return       # nothing available

    moveDelta = newBS.domainProbs.placeDelta
    # LPK: If this fails, it could be that we really want to try a
    # different obstacle (so we can do the obstacles in a different order)
    # than a different placement of the first obst;  not really sure how to
    # arrange that.
    for obst in obstacles:
        for r in moveOut(newBS, prob, obst, moveDelta, cond):
            # TODO: LPK: concerned about how graspVar and graspDelta
            # are computed
            graspFace = r.gB.grasp.mode()
            graspMean = r.gB.poseD.modeTuple()
            graspVar = r.gB.poseD.varTuple()
            graspDelta = r.gB.delta
            supportFace = r.pB.support.mode()
            poseMean = r.pB.poseD.modeTuple()
            poseVar = r.pB.poseD.varTuple()

            newConds = frozenset(
                {Bd([SupportFace([obst]), supportFace, prob], True),
                 B([Pose([obst, supportFace]), poseMean, poseVar,
                           moveDelta, prob], True)})
            print '*** moveOut', obst
            yield [newConds]
    tr(tag, '=> Out of remedies')

def pushAchCanXGen(newBS, shWorld, initViol, violFn, prob, cond):
    tag = 'pushAchGen'
    obstacles = [o.name() for o in initViol.allObstacles() \
                  if (not o.name() in shWorld.fixedObjects) and \
                  (not fixedHeld(newBS, o.name()))]
    if obstacles:
        tr(tag, 'Movable obstacles to fix', obstacles)        
    else:
        tr(tag, '=> No movable obstacles to fix')
        return       # nothing available

    moveDelta = newBS.domainProbs.placeDelta
    for obst in obstacles:
        for r in pushOut(newBS, prob, obst, moveDelta, cond):

            supportFace = r.postPB.support.mode()
            postPose = r.postPB.poseD.modeTuple()
            postPoseVar = r.postPB.poseD.var
            prePose = r.prePB.poseD.modeTuple()
            prePoseVar = r.prePB.poseD.var

            newConds = frozenset(
                {Bd([SupportFace([obst]), supportFace, prob], True),
                 B([Pose([obst, supportFace]), postPose, postPoseVar,
                           moveDelta, prob], True)})
            print '*** pushOut', obst
            yield [newConds]
    tr(tag, '=> Out of remedies')
    

######################################################################
#
# Meta operators:  make implicit fluents true
#
######################################################################

# Could be place, push, or look

achCanReach = BMetaOperator('AchCanReach', CanReachHome, ['CEnd'],
                           AchCanReachGen,
                           argsToPrint = [0])

achCanReachNB = BMetaOperator('AchCanReachNB', CanReachNB, ['CStart', 'CEnd'],
                           AchCanReachNBGen,
                           argsToPrint = [0, 1])

achCanPickPlace = BMetaOperator('AchCanPickPlace', CanPickPlace,
    ['PreConf', 'PlaceConf', 'Hand', 'Obj', 'Pose',
     'RealPoseVar', 'PoseDelta', 'PoseFace',
     'GraspFace', 'GraspMu', 'GraspVar', 'GraspDelta','PPOp'],
     AchCanPickPlaceGen,
     argsToPrint = (1, 2, 3))

achCanPush = BMetaOperator('AchCanPush', CanPush,
    ['Obj', 'Hand', 'PoseFace',
     'PrePose', 'Pose', 'PreConf', 'PushConf', 'PostConf',
     'PoseVar', 'PrePoseVar', 'PoseDelta'],
    AchCanPushGen,
    argsToPrint = (0, 1, 4))

# Never been tested
'''
achCanSee = Operator('AchCanSee',
    ['Obj', 'TargetPose', 'TargetPoseFace', 'TargetPoseVar',
     'LookConf', 'PreCond', 'PostCond', 'NewCond', 'Op', 'PR'],
    {0: {Bd([CanSeeFrom(['Obj', 'TargetPose', 'TargetPoseFace', 'LookConf',
                         'PreCond']), True, 'PR'], True)}},
    # Result
    [({Bd([CanSeeFrom(['Obj', 'TargetPose', 'TargetPoseFace', 'LookConf', 
                           'PostCond']),  True, 'PR'], True)}, {})],
    functions = [
        CanSeeGen(['Op', 'NewCond'],
                  ['Obj', 'TargetPose', 'TargetPoseFace', 'TargetPoseVar',
                   'LookConf', 'PR', 'PostCond']),
         AddPreConds(['PreCond'], ['PostCond', 'NewCond'])],
    metaGenerator = True)
'''

######################################################################
#
# Operators only used in the heuristic
#
######################################################################

magicRegraspCost = 40

hRegrasp = Operator(
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
        functions = [
            # Be sure obj is not none
            NotNone([], ['Obj']),
            # Take out error in backchaining just so we hit cache
            RegressProb(['P1'], ['PR1']),
            RegressProb(['P2'], ['PR2']),
            RegressProb(['P3'], ['PR3']),
            # RegressProb(['P1'], ['PR1'], pn = 'pickFailProb'),
            # RegressProb(['P2'], ['PR2'],pn = 'pickFailProb'),
            # RegressProb(['P3'], ['PR3'], pn = 'pickFailProb'),
            EasyGraspGen(['PrevGraspFace', 'PrevGraspMu', 'PrevGraspVar',
                          'PrevGraspDelta'],['Obj', 'Hand', 'GraspFace',
                                             'GraspMu'])],
        cost = lambda al, args, details: magicRegraspCost,
        ignorableArgsForHeuristic = range(1, 16),
        argsToPrint = [0, 1])
