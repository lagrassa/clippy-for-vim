import pdb
import math
import numpy as np
import hu
import time
import shapes
from transformations import rotation_matrix
import planGlobals as glob
from traceFile import debugMsg, debug, tr
from planUtil import Violations, ObjPlaceB, ObjGraspB
from pr2Util import shadowName, objectName, permanent, inside, PoseD, GDesc, otherHand, \
     objectGraspFrame, robotGraspFrame
from pr2Visible import visible, lookAtConf, viewCone, findSupportTableInPbs
from pr2RRT import planRobotGoalPath, interpolatePath
from dist import DeltaDist
import windowManager3D as wm

Ident = hu.Transform(np.eye(4))            # identity transform
tiny = 1.0e-6

################
# Basic tests for fluents and generators
################

def legalGrasp(pbs, conf, hand, objGrasp, objPlace):
    deltaThreshold = (0.01, 0.01, 0.01, 0.02)
    # !! This should check for kinematic feasibility over a range of poses.
    of = objectGraspFrame(pbs, objGrasp, objPlace, hand)
    rf = robotGraspFrame(pbs, conf, hand)
    result = of.withinDelta(rf, deltaThreshold)
    return result

def findRegionParent(pbs, region):
    # In case, this is a bState and not a pbs
    pbs = pbs.pbs
    regs = pbs.getWorld().regions
    for (obj, stuff) in regs.items():
        for (regName, regShape, regTr) in stuff:
            if regName == region:
                return obj
    raw_input('No parent object for region '+str(region))
    return None

# probability is: pObj * pParent * pObjFitsGivenRelativeVar = prob
def inTest(pbs, obj, regName, prob, pB=None):
    regs = pbs.getWorld().regions
    parent = findRegionParent(pbs, regName)
    pObj = pbs.poseModeProbs[obj]
    pParent = pbs.poseModeProbs[parent]
    pFits = prob / (pObj * pParent)
    if pFits > 1: return False

    # compute a shadow for this object
    placeB = pB or pbs.getPlaceB(obj)
    placeBSmall = placeB.modifyPoseD(delta = (0.0, 0.0, 0.0, 0.0))
    faceFrame = placeB.faceFrames[placeB.support.mode()]

    # Setting delta to be 0
    sh = pbs.objShadow(obj, shadowName(obj), pFits, placeBSmall, faceFrame)
    shadow = sh.applyLoc(placeB.objFrame()) # !! is this right?
    shWorld = pbs.getShadowWorld(prob)
    region = shWorld.regionShapes[regName]

    ans = any([np.all(np.all(np.dot(r.planes(), shadow.prim().vertices()) <= tiny, axis=1)) \
               for r in region.parts()])

    tr('testVerbose', 'In test, shadow in orange, region in purple',
       (shadow, region, ans), draw = [(pbs, prob, 'W'),
                                      (shadow, 'W', 'orange'),
                                      (region, 'W', 'purple')], snap=['W'])
    return ans

# Just see if the the object, with irreducible shadow, would fit
def inTestMinShadow(pbs, obj, pose, poseFace, regName):
    regs = pbs.getWorld().regions

    placeB = ObjPlaceB(obj,
                       pbs.getWorld().getFaceFrames(obj),
                       DeltaDist(poseFace),
                       hu.Pose(* pose),
                       (0, 0, 0, 0),
                       (0, 0, 0, 0))
    faceFrame = placeB.faceFrames[poseFace]

    # !! Clean this up
    prob = 0.1
    sh = pbs.objShadow(obj, shadowName(obj), prob, placeB, faceFrame)
    shadow = sh.applyLoc(placeB.objFrame()) # !! is this right?
    shWorld = pbs.getShadowWorld(prob)

    region = shWorld.regionShapes[regName]
    ans = any([np.all(np.all(np.dot(r.planes(), shadow.prim().vertices()) <= \
                             tiny, axis=1)) \
               for r in region.parts()])

    tr('testVerbose', 'In test, shadow in orange, region in purple',
       (shadow, region, ans), draw = [(pbs, prob, 'W'),
                                      (shadow, 'W', 'orange'),
                                      (region, 'W', 'purple')], snap=['W'])
    return ans


# LPK: canReachNB which is like canReachHome, but without moving
# the base.  
def canReachNB(pbs, startConf, conf, prob, initViol,
               optimize = False):
    # canReachHome goes towards its homeConf arg, that is it's destination.
    return canReachHome(pbs, startConf, prob, initViol,
                        homeConf=conf, moveBase=False,
                        optimize=optimize) 

# returns a path (conf -> homeConf (usually home))
def canReachHome(pbs, conf, prob, initViol, homeConf = None, reversePath = False,
                 optimize = False, moveBase = True):
    rm = pbs.getRoadMap()
    if not homeConf: homeConf = rm.homeConf
    robot = pbs.getRobot()
    tag = 'canReachHome' if moveBase else 'canReachNB'
    viol, cost, path = rm.confReachViol(conf, pbs, prob, initViol,
                                        startConf=homeConf,
                                        reversePath = reversePath,
                                        moveBase = moveBase,
                                        optimize = optimize)

    if path:
        assert path[0] == conf, 'Start of path'
        assert path[-1] == homeConf, 'End of path'

    if viol is None or viol.weight() > 0:
        # Don't log the "trivial" ones...
        tr('CRH', '%s h=%s'%(tag, glob.inHeuristic) + \
           ' viol=%s'%(viol.weight() if viol else None))

    if path and debug('backwards'):
        backSteps = []
        # unless reversePath is True, the direction of motion is
        # "backwards", that is, from conf to home.
        for i, c in enumerate(path[::-1] if reversePath else path ):
            if i == 0: continue
            # that is, moving from i to i-1 should be forwards (valid)
            if not validEdgeTest(c['pr2Base'], path[i-1]['pr2Base']):
                backSteps.append((c['pr2Base'], path[i-1]['pr2Base']))
        if backSteps:
            for (pre, post) in backSteps:
                trAlways('Backward step:', pre, '->', post, ol = True,
                         pause = False)
            # raw_input('CRH - Backwards steps')

        if debug('canReachHome'):
            pbs.draw(prob, 'W')
            if path:
                drawPath(path, viol=viol,
                         attached=pbs.getShadowWorld(prob).attached)
        tr('canReachHome', ('viol', viol), ('cost', cost), ('path', path),
               snap = ['W'])

    if not viol and debug('canReachHome'):
        pbs.draw(prob, 'W')
        conf.draw('W', 'blue', attached=pbs.getShadowWorld(prob).attached)
        raw_input('CRH Failed')

    return path, viol

ppConfs = {}

def canPickPlaceTest(pbs, preConf, ppConf, hand, objGrasp, objPlace, p,
                     op='pick', quick = False):
    tag = 'canPickPlaceTest'
    obj = objGrasp.obj
    collides = pbs.getRoadMap().checkRobotCollision
    if debug(tag):
        print zip(('preConf', 'ppConf', 'hand', 'objGrasp', 'objPlace', 'p', 'pbs'),
                  (preConf, ppConf, hand, objGrasp, objPlace, p, pbs))
    if not legalGrasp(pbs, ppConf, hand, objGrasp, objPlace):
        debugMsg(tag, 'Grasp is not legal in canPickPlaceTest')
        return None, 'Legal grasp'
    # pbs.getRoadMap().approachConfs[ppConf] = preConf
    violations = Violations()           # cumulative
    # 1.  Can move from home to pre holding nothing with object placed at pose
    if preConf:
        pbs1 = pbs.copy().updatePermObjBel(objPlace).updateHeldBel(None, hand)
        if op == 'pick':
            oB = objPlace.modifyPoseD(var=4*(0.0,)) # ignore uncertainty
            oB.delta = 4*(0.0,)
        else:
            oB = objPlace.modifyPoseD(var=pbs.domainProbs.placeVar)
            oB.delta = pbs.domainProbs.placeDelta
        pbs1 = pbs1.updatePermObjBel(oB).addAvoidShadow([obj])
        if debug(tag):
            pbs1.draw(p, 'W')
            debugMsg(tag, 'H->App, obj@pose (condition 1)')
        if quick:
            violations = pbs.getRoadMap().confViolations(preConf, pbs1, p, violations)
            path = [preConf]
        else:
            path, violations = canReachHome(pbs1, preConf, p, violations)
        if not violations:
            debugMsg(tag, 'Failed H->App, obj=pose (condition 1)')
            return None, '1. H->App obj@pose'
        elif debug(tag):
            for c in path: c.draw('W', attached = pbs1.getShadowWorld(p).attached)
            debugMsg(tag, 'path 1')

    # preConfShape = preConf.placement(attached = pbs1.getShadowWorld(p).attached)
    objShadow = objPlace.shadow(pbs1.getShadowWorld(p))
    # Check visibility at preConf (for pick)
    if op =='pick' and not (glob.inHeuristic or quick):
        path = canView(pbs1, p, preConf, hand, objShadow)
        if path:
            debugMsg(tag, 'Succeeded visibility test for pick')
            preConfView = path[-1]
            if preConfView != preConf:
                path, violations = canReachHome(pbs1, preConfView, p, violations)
                if not violations:
                    debugMsg(tag, 'Cannot reachHome with retracted arm')
                    pbs1.draw(p, 'W'); preConfView.draw('W', 'red')
                    debugMsg(tag, 'canPickPlaceTest - Cannot reachHome with retracted arm')
                    return None, 'Obj visibility'
        else:
            debugMsg(tag, 'Failed visibility test for pick')
            return None, 'Obj visibility'
            
    # 2 - Can move from home to pre holding the object
    pbs2 = pbs.copy().excludeObjs([obj]).updateHeldBel(objGrasp, hand)
    if debug(tag):
        pbs2.draw(p, 'W'); preConf.draw('W', attached = pbs2.getShadowWorld(p).attached)
        debugMsg(tag, 'H->App, obj=held (condition 2)')
    if quick:
        violations = pbs.getRoadMap().confViolations(preConf, pbs2, p, violations)
        path = [preConf]
    else:
        path, violations = canReachHome(pbs2, preConf, p, violations)
    if not violations:
        debugMsg(tag + 'Failed H->App, obj=held (condition 2)')
        return None, '2. H->App, held=obj'
    elif debug(tag):
        for c in path: c.draw('W', attached = pbs2.getShadowWorld(p).attached)
        debugMsg(tag, 'path 2')

    # Check visibility of support table at preConf (for pick AND place)
    if op in ('pick', 'place') and not (glob.inHeuristic or quick):
        tableB = findSupportTableInPbs(pbs1, objPlace.obj) # use pbs1 so obj is there
        assert tableB
        if debug(tag): print 'Looking at support for', obj, '->', tableB.obj
        lookDelta = pbs2.domainProbs.minDelta
        lookVar = pbs2.domainProbs.obsVarTuple
        tableB2 = tableB.modifyPoseD(var = lookVar)
        tableB2.delta = lookDelta
        prob = 0.95
        shadow = tableB2.shadow(pbs2.updatePermObjBel(tableB2).getShadowWorld(prob))
        if collides(preConf, shadow, attached = pbs2.getShadowWorld(p).attached):
            preConfShape = preConf.placement(attached = pbs2.getShadowWorld(p).attached)
            pbs2.draw(p, 'W'); preConfShape.draw('W', 'cyan'); shadow.draw('W', 'cyan')
            debugMsg('Preconf collides for place in canPickPlaceTest')
            return None, 'Support shadow collision'
        if collides(ppConf, shadow): # ppConfShape.collides(shadow):
            ppConfShape = ppConf.placement() # no attached
            pbs2.draw(p, 'W'); ppConfShape.draw('W', 'magenta'); shadow.draw('W', 'magenta')
            debugMsg(tag, 'PPconf collides for place in canPickPlaceTest')
            return None, 'Support shadow collision'
        if not canView(pbs2, p, preConf, hand, shadow):
            preConfShape = preConf.placement(attached = pbs2.getShadowWorld(p).attached)
            pbs2.draw(p, 'W'); preConfShape.draw('W', 'orange'); shadow.draw('W', 'orange')
            debugMsg(tag, 'Failing to view for place in canPickPlaceTest')
            return None, 'Support visibility'

    # 3.  Can move from home to pick with object placed at pose (0 var)
    oB = objPlace.modifyPoseD(var=4*(0.0,)) # ignore uncertainty
    oB.delta = 4*(0.0,)
    pbs3 = pbs.copy().updatePermObjBel(oB).updateHeldBel(None, hand)
    if debug(tag):
        pbs3.draw(p, 'W')
        debugMsg(tag, 'H->Target, obj placed (0 var) (condition 3)')
    if quick:
        violations = pbs.getRoadMap().confViolations(ppConf, pbs3, p, violations)
        path = [ppConf]
    else:
        path, violations = canReachHome(pbs3, ppConf, p, violations)
    if not violations:
        debugMsg(tag, 'Failed H->Target  (condition 3)')
        return None, '3. H->Target obj@pose 0var'
    elif debug(tag):
        for c in path: c.draw('W', attached = pbs3.getShadowWorld(p).attached)
        debugMsg(tag, 'path 3')
    # 4.  Can move from home to pick while holding obj with zero grasp variance
    gB = objGrasp.modifyPoseD(var=4*(0.0,)) # ignore uncertainty
    gB.delta = 4*(0.0,)
    pbs4 = pbs.copy().excludeObjs([obj]).updateHeldBel(gB, hand)
    if debug(tag):
        pbs4.draw(p, 'W'); ppConf.draw('W', attached = pbs4.getShadowWorld(p).attached)
        debugMsg(tag, 'H->Target, holding obj (0 var) (condition 4)')
    if quick:
        violations = pbs.getRoadMap().confViolations(ppConf, pbs4, p, violations)
        path = [ppConf]
    else:
        path, violations = canReachHome(pbs4, ppConf, p, violations)
    if not violations:
        debugMsg(tag, 'Failed H->Target held=obj(condition 4)')
        return None, '4. H->Target held=obj 0var'
    elif debug(tag):
        for c in path: c.draw('W', attached = pbs4.getShadowWorld(p).attached)
        debugMsg(tag, 'path 4')
    debugMsg(tag, ('->', violations))

    if debug('lookBug'):
        base = tuple(preConf['pr2Base'])
        entry = (preConf, tuple([x.getShadowWorld(p) for x in (pbs1, pbs2, pbs3, pbs4)]))
        if base in ppConfs:
            ppConfs[base] = ppConfs[base].union(frozenset([entry]))
        else:
            ppConfs[base] = frozenset([entry])

    return violations, None


# Find a path to a conf such that the arm (specified) by hand does not
# collide with the view cone to the target shape and maybe shadow.
def canView(pbs, prob, conf, hand, shape,
            shapeShadow = None, maxIter = 50, findPath = True):
    def armShape(c, h):
        parts = dict([(o.name(), o) for o in c.placement(attached=attached).parts()])
        armShapes = [parts[pbs.getRobot().armChainNames[h]],
                     parts[pbs.getRobot().gripperChainNames[h]]]
        if attached[h]:
            armShapes.append(parts[attached[h].name()])
        return shapes.Shape(armShapes, None)
    collides = pbs.getRoadMap().checkRobotCollision
    robot = pbs.getRobot()
    vc = viewCone(conf, shape)
    if not vc: return None
    shWorld = pbs.getShadowWorld(prob)
    attached = shWorld.attached
    if shapeShadow:
        avoid = shapes.Shape([vc, shape, shapeShadow], None)
    else:
        avoid = shapes.Shape([vc, shape], None)
    # confPlace = conf.placement(attached=attached)
    if not collides(conf, avoid, attached=attached):
        if debug('canView'):
            print 'canView - no collisions'
        return [conf]
    elif not findPath:
        return []
    # !! don't move arms to clear view of fixed objects
    if not permanent(objectName(shape.name())):
        if debug('canView'):
            avoid.draw('W', 'red')
            conf.draw('W', attached=attached)
            debugMsg('canView', 'ViewCone collision')
        pathFull = []
        for h in ['left', 'right']:     # try both hands
            chainName = robot.armChainNames[h]
            armChains = [chainName, robot.gripperChainNames[h]]
            if not (collides(conf, avoid, attached=attached, selectedChains=armChains) \
                   if glob.useCC else avoid.collides(armShape(conf, h))):
                continue
            if debug('canView'):
                print 'canView collision with', h, 'arm', conf['pr2Base']
            path, viol = planRobotGoalPath(pbs, prob, conf,
                                lambda c: not (collides(c, avoid, attached=attached, selectedChains=armChains) \
                                                          if glob.useCC else avoid.collides(armShape(c,h))),
                                           None, [chainName], maxIter = maxIter)
            if debug('canView'):
                pbs.draw(prob, 'W')
                if path:
                    for c in path: c.draw('W', 'blue', attached=attached)
                    path[-1].draw('W', 'orange', attached=attached)
                    avoid.draw('W', 'green')
                    debugMsg('canView', 'Retract arm')
            if debug('canView') or debug('canViewFail'):
                if not path:
                    pbs.draw(prob, 'W')
                    conf.draw('W', attached=attached)
                    avoid.draw('W', 'red')
                    raw_input('canView - no path')
            if path:
                pathFull.extend(path)
            else:
                return []
        return pathFull
    else:
        if debug('canView'):
            print 'canView - ignore view cone collision for perm object', shape
        return [conf]

# Computes lookConf for shape, makes sure that the robot does not
# block the view cone.  It will construct a path from the input conf
# to the returned lookConf if necessary - the path is not returned,
# only the final conf.
def lookAtConfCanView(pbs, prob, conf, shape, hands=('left', 'right'),
                      shapeShadow=None, findPath=True):
    lookConf = lookAtConf(conf, shape)  # conf with head looking at shape
    if not lookConf:
        tr('lookAtConfCanView', 'lookAtConfCanView failed conf')
        return None
    if True: # not glob.inHeuristic:            # if heuristic we'll ignore robot
        path = None
        for hand in hands:              # consider each hand in turn
            # Find path from lookConf to some conf that does not
            # collide with viewCone.  The last conf in the path will
            # be the new lookConf.
            path = canView(pbs, prob, lookConf, hand, shape,
                           shapeShadow=shapeShadow, findPath=findPath)
            if path: break
        if not path:
            tr('lookAtConfCanView', 'lookAtConfCanView failed path')
            return None
        lookConf = path[-1]
    tr('lookAtConfCanView', '->', lookConf)
    return lookConf

