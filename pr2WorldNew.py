
################
# Basic tests for pick and place
################

def legalGrasp(bState, conf, hand, objGrasp, objPlace, delPose):
    of = objectGraspFrame(bState, objGrasp, objPlace)
    rf = robotGraspFrame(bState, conf, hand)
    (distDelta, angleDelta) = delPose
    result1 = of.near(rf, distDelta, angleDelta)

###################
#!! Need implementations of InPickApproach and InPlaceDeproach
###################

# Pick conditions
# pick1. move home->pre with obj at pick pose
# pick2. move home->pick without obj
# pick3. move home->pick with obj in hand 

def canPickTest(bState, preConf, pickConf, hand, objGrasp, objPlace, delPose, p):
    args = (preConf, pickConf, hand, objGrasp, objPlace, p, bState)
    debugMsg('canPickTest',
             zip(('preConf', 'pickConf', 'hand', 'objGrasp', 'objPlace', 'p', 'bState'),
                args))
    if not legalGrasp(bState, pickConf, hand, objGrasp, objPlace, delPose):
        debugMsg('canPickTest', 'Grasp is not legal')
        return None
    if not inPickApproach(bState, preConf, pickConf, hand, objGrasp, objPlace):
        debugMsg('canPickTest', 'Grasp is not legal')
        return None
    violations = Violations()           # cumulative
    # 1.  Can move from home to pre holding nothing with object placed at pose
    bState1 = bState.copy().updatePermObjPose(objPlace).updateHeld(None)
    debugMsg('canPickTest', 'Test initial approach (condition 1)')
    path, violations = canReachHome(bState1, preConf, p, violations)
    if not path:
        debugMsg('canPickTest', 'Failed initial approach (condition 1)')
        return None
    # 2.  Can move from home to pick in bState without obj
    bState2 = bState.copy().excludeObjs([obj]).updateHeld(None)
    debugMsg('canPickTest', 'Test final approach (condition 2)')
    path, violations = canReachHome(bState2, pickConf, p, violations)
    if not path:
        debugMsg('canPickTest', 'Failed final approach (condition 2)')
        return None
    # 3.  Can move from home to pick while holding obj
    bState3 = bState.copy().excludeObjs([obj]).updateHeld(objGrasp)
    debugMsg('canPickTest', 'Test de-pproach (condition 3)')
    path, violations = canReachHome(bState3, pickConf, p, violations)
    if not path:
        debugMsg('canPickTest', 'Failed de-pproach (condition 3)')
        return None
    debugMsg('canPickTest', ('->', violations))
    return violations

# Place conditions
# place1. move home->place with obj in hand
# place2. move post->home with obj at place pose

def canPlaceTest(bState, preConf, placeConf, hand, objGrasp, objPlace, delPose, p):
    args = (preConf, placeConf, hand, objGrasp, objPlace, p, bState)
    debugMsg('canPlaceTest',
             zip(('preConf', 'placeConf', 'hand', 'objGrasp', 'objPlace', 'p', 'bState'),
                args))
    if not legalGrasp(bState, placeConf, hand, objGrasp, objPlace, delPose):
        debugMsg('canPlaceTest', 'Grasp is not legal')
        return None
    if not inPlaceDeproach(bState, placeConf, postConf, hand, objGrasp, objPlace):
        debugMsg('canPlaceTest', 'Grasp is not legal')
        return None
    violations = Violations()           # cumulative
    # 1.  Can move from home to placeConf holding the object
    bState1 = bState.copy().updateHeld(objGrasp)
    debugMsg('canPlaceTest', 'Test place condition (condition 1)')
    path, violations = canReachHome(bState1, placeConf, p, violations)
    if not path:
        debugMsg('canPlaceTest', 'Failed place condition (condition 1)')
        return None
    # 2.  Can move from postconf to home with hand empty and object placed
    bState2 = bState.copy().updatePermObjPose(objPlace).updateHeld(None)
    debugMsg('canPlaceTest', 'Test de-pproach (condition 2)')
    path, violations = canReachHome(bState2, postConf, p, violations)
    if not path:
        debugMsg('canPlaceTest', 'Failed de-pproach (condition 2)')
        return None
    debugMsg('canPlaceTest', ('->', violations))
    return violations

def inTest(bState, obj, region, prob, objD = None):
    # compute a shadow for this object
    poseD = objD or bState.getPoseD(obj)
    shadow = bState.objShadow(obj, True, prob, poseD)
    return all([region.contains(p) for p in shadow.vertices()])

################
## GENERATORS
################

memoizerBufferN = 5
class Memoizer:
    def __init__(self, name, generator, values = None, bufN = memoizerBufferN):
        self.name = name
        self.generator = generator               # shared
        self.values = values if values else [] # shared
        self.bufN = bufN                       # shared
        self.done = set([])             # not shared
    def __iter__(self):
        return self
    def copy(self):
        # shares the generator and values list, only index differs.
        new = Memoizer(self.name, self.generator, self.values, self.bufN)
        return new
    def next(self):
        dif = len(self.values) - len(self.done)
        # Fill up the buffer, if possible
        if dif < self.bufN:
            for i in range(self.bufN - dif):
                try:
                    val = self.generator.next()
                    self.values.append(val)
                    if val[1].weight() < 1.0: break
                except StopIteration:
                    break
        if len(self.values) > len(self.done):
            elegible = set(range(len(self.values))) - self.done
            # Find min weight index among elegible
            nextI = argmax(list(elegible), lambda i: -self.values[i][1].weight())
            self.done.add(nextI)
            chosen = self.values[nextI]
            debugMsg('Memoizer',
                     self.name,
                     ('weights', [self.values[i][1].weight() for i in elegible]),
                     ('chosen', chosen[1].weight()))
            if chosen[1].weight() > 5:
                raw_input('Big weight - Ok?')
            return chosen
        else:
            raise StopIteration

# Generators:
#   INPUT:
#   list of specific args such as region, object(s), variance, probability
#   conditions from the goal state, e.g. Pose, Conf, Grasp, Reachable, In, are constraints
#   initial state
#   some pre-bindings of output variables.
#   OUTPUT:
#   ordered list of ordered value lists

# returns values for (?pose, ?conf, ?confAppr). i.e. pcc

def pickGen(args, goalConds, bState, outBindings, onlyCurrent = False):
    for ans, viol in pickGenTop(args, goalConds, bState, outBindings, onlyCurrent):
        debugMsg('pickGen', ('->', ans, viol))
        yield ans

def pickGenTop(args, goalConds, bState, outBindings,
               onlyCurrent = False):
    debugMsg('pickGen',
             zip(('obj', 'grasp', 'graspPose', 'objV', 'graspV',
                  'objDelta', 'graspDelta', 'prob'), args),
             outBindings)
    (obj, grasp, graspPose, objV, graspV, objDelta, graspDelta, prob)\
       = args
    (opose, oconf, oconfAppr) = outBindings
    if obj == 'none':                   # can't pick up 'none'
        return
    if obj == bState.held.mode():
        pose = None
        conf = None
        confAppr = None
        raw_input('Trying to pick object already in hand -- pose is undefined')
    else:
        # Check if any output bindings have been specified.
        pose = bState.getPoseD(obj).mode() if isAnyVar(opose) else opose
        conf = None if isAnyVar(oconf) else oconf   # need to choose
        confAppr = None if isAnyVar(oconfAppr) else oconfAppr   # need to choose
    # Set up bState
    newBS = bState.copy()
    # Just placements specified in goal
    # !! Do the args override goalConds or viceversa?
    newBS = updateFromGoalPoses(goalConds) if goalConds else newBS
    newBS.updateGraspV(graspV, graspDelta).updateObjV(objV, objDelta)
    # If onlyCurrent or outBindings were specified, then we don't regrasp
    onlyCurrent = onlyCurrent or any([not isAnyVar(v) for v in outBindings])
    # I don't know what to do about the dependence on goalConds, so we
    # won't cache for now.  placeInGen caches in any case.
    gen = pickGenAux(obj, confAppr, conf, pose, grasp, graspPose, prob, bState,
                     goalConds, onlyCurrent=onlyCurrent)
    memo = Memoizer('pickGen', gen)
    # call the memo to generate
    for x in memo: yield x

def pickGenAux(obj, confAppr, conf, pose, grasp, graspPose, prob,
               bState, goalConds, onlyCurrent = False):
    objV = bState.objV
    objDelta = bState.objDelta
    (_, graspV, graspDelta) = bState.graspD
    bState = bState.bState
    cviol = None
    if pose != None:
        if confAppr is None:
            confAppr = findApproachConf(obj, poseD(bState, pose),
                                        conf, prob, bState)
        # Try picking at the current location of the object
        cviol = canPickTest(obj, confAppr, conf, pose, grasp, prob, bState)
        if cviol:
            pcc = (pose, conf, confAppr)
            if debug('pickGen'):
                drawPoseConf(bState, obj, pose, conf, confAppr, curWin, color = 'green')
                debugMsg('pickGen', ('currently graspable', pcc), ('cviol', cviol))
            yield (pcc, cviol)
            if onlyCurrent:
                debugMsg('pickGen', 'onlyCurrent: out of values')
                return
    # Try a regrasp... that is place the object somewhere else where it can be grasped.
    debugMsg('placeInGen', ('Called by pick', obj, bState.awayRegion))

    pgccGen = placeInGenTop((obj, bState.awayRegion, objV, graspV,
                             objDelta, graspDelta, prob),
                            goalConds, bState, ['?x', grasp, '?x', '?x'],
                            considerOtherIns = False)
    for pgcc, viol in pgccGen:
        (p, g, c, ca) = pgcc
        # The regrasp option should never be cheaper than the non-regrasp.
        penalty = cviol.weight()+1 if cviol else 1
        debugMsg('pickGen', ('Adding penalty', penalty, 'to', viol.penalty, viol))
        yield (p, c, ca), Violations(viol.obstacles, viol.shadows,
                                     viol.penalty+penalty)
    debugMsg('pickGen', 'out of values')

def placeGen(args, goalConds, bState, outBindings):
    for ans, viol in placeGenTop(args, goalConds, bState, outBindings):
        debugMsg('placeGen', ('->', ans))
        yield ans

# returns values for (?grasp, ?conf, ?confDep)
def placeGenTop(args, goalConds, bState, outBindings):
    (obj, pose, objV, graspV, confV, objDelta, graspDelta, confDelta,  prob) = args
    (ograsp, oconf, oconfDep) = outBindings
    if obj == 'none':
        return
    # Have any output bindings been specified?
    grasps = bState.legalGrasps[obj] if isAnyVar(ograsp) else [ograsp]
    conf = None if isAnyVar(oconf) else oconf
    confDep = None if isAnyVar(oconfDep) else oconfDep
    # Set up world
    # Just placements specified in goal
    world = W(bState, objV=objV, objDelta=objDelta).updateFromGoalPoses(goalConds).excludeObjs([obj])
    world.updateConfV(confV, confDelta).updateGraspV(graspV, graspDelta)
    debugMsg('placeGen',
             zip(('obj', 'pose', 'objV', 'graspV', 'confV', 'prob'), args),
             outBindings, world)
    key = ((obj, confDep, conf, pose, tuple(grasps), prob), world)
    debugMsg('placeGen', key)
    if key in bState.genCaches['placeGen']:
        debugMsg('placeGen', 'placeGenCache hit')
        memo = bState.genCaches['placeGen'][key].copy()
    else:
        gen = placeGenAux(obj, confDep, conf, pose, grasps, prob, world)
        memo = Memoizer('placeGen', gen)
        bState.genCaches['placeGen'][key] = memo
    # call the memo to generate
    for x in memo: yield x

def placeGenAux(obj, confDep, conf, pose, grasps, prob, world):
    objV = world.objV
    objDelta = world.objDelta
    (_, confV, confDelta) = world.confD
    bState = world.bState
    graspsSorted = []
    for grasp in grasps:
        # Check whether obj is graspable in current location
        currentPose = modeTuple(bState.objects[obj])
        pickConf = graspConf(bState, currentPose, grasp)
        wmin = world.minimizeVariances()
        viol = canPickTest(obj, None, pickConf, currentPose, grasp, prob, wmin)
        if viol:
            graspsSorted.append((viol.weight(), grasp))
        else:
            graspsSorted.append((10., grasp))
    graspsSorted.sort()
    debugMsg('placeGen', ('sorted grasps', graspsSorted))
    for (w, g) in graspsSorted:
        c = conf or graspConf(bState, pose, g)
        cd = confDep or findApproachConf(obj, (pose, objV, objDelta),
                                         (c, confV, confDelta), prob, bState)
        viol = canPlaceTest(obj, c, cd, pose, g, prob, world)
        if viol:
            gcc = (g, c, cd)
            if debug('placeGen'):
                drawPoseConf(bState, obj, pose, c, cd, curWin, color='magenta')
                path, violations = canReachHome(c, prob, world)
                drawPath(path, prob, world)
            debugMsg('placeGen', ('->', gcc), ('viol', viol))
            if debug('placeGen'):
                wm.getWindow(curWin).clear()
            yield gcc, viol
        else:
            debugMsg('placeGen', 'canPlaceTest failed', ('grasp', g), ('conf', c), ('pose', pose))
    debugMsg('placeGen', 'out of values')

def placeInGen(args, goalConds, bState, outBindings, considerOtherIns = True):
    for ans, viol in placeInGenTop(args, goalConds, bState, outBindings, considerOtherIns):
        debugMsg('placeInGen', ('->', ans))
        yield ans

def placeInGenTop(args, goalConds, bState, outBindings, considerOtherIns = True):
    if debug('placeInGen'): bState.draw(curWin)
    debugMsg('placeInGen',
             zip(('obj', 'regShape', 'objV', 'graspV', 'confV',
                  'objDelta', 'graspDelta', 'confDelta', 'prob'), args),
             outBindings)
    (obj, regShape, objV, graspV, confV, objDelta, graspDelta, confDelta, prob) = args
    (opose, ograsp, oconf, oconfDep) = outBindings
    if obj == 'none':
        return
    # Have any output bindings been specified?
    if not isAnyVar(opose):             # opose is specified, just do placeGen
        for gcc, viol in placeGenTop((obj, opose, objV, graspV, confV,
                                      objDelta, graspDelta, confDelta,prob),
                               goalConds, bState, (ograsp, oconf, oconfDep)):
            yield (opose,) + gcc, viol
        return
    legalGrasps = bState.legalGrasps if isAnyVar(ograsp) else {obj:[ograsp]}
    conf = None if isAnyVar(oconf) else oconf
    confDep = None if isAnyVar(oconfDep) else oconfDep
    # Conditions for already placed objects
    placed = [o for (o, r, p) in getGoalInConds(goalConds) \
              if o != obj and \
              inTest(bState, o, r, p,
                    (modeTuple(bState.getObjectD(o), objV, zeroConfDeltaTuple)))]
    placedDs = dict([(o, (modeTuple(bState.objects[o]), objV, objDelta)) for o in placed])
    inConds = [(o, r, objV, objDelta,  p) for (o, r, p) in getGoalInConds(goalConds) \
               if o != obj and o not in placed]
    # Set up world
    # Just placements specified in goal
    world = W(bState, objV=objV, objDelta=objDelta).updateFromGoalPoses(goalConds).excludeObjs([obj]+placed)
    world.updateConfV(confV, confDelta).updateGraspV(graspV, graspDelta)
    world.moveObjBs.update(placedDs)        # the placed are optional
    debugMsg('placeInGen', ('inConds', inConds), ('placed', placed), world)
    # Obstacles for all Reachable fluents (conf, confV, prob)
    reachObsts = getReachObsts(goalConds, bState, confV, confDelta, graspV, graspDelta)
    debugMsg('placeInGen', ('reachObsts', reachObsts))
    if reachObsts == None:
        debugMsg('placeInGen', 'quitting because no path')
        return
    # PlaceInGen has a randomized component -- the results can't be cached!!
    # key = (obj, regShape, conf, confDep, prob, world, frozenset(reachObsts), frozenset(inConds),
    #        frozenset([(o, frozenset(legalGrasps[o])) for o in legalGrasps]),
    #        considerOtherIns)
    # debugMsg('placeInGen', key)
    # if key in bState.genCaches['placeInGen']:
    #     debugMsg('placeInGen', 'placeInGenCache hit')
    #     memo = bState.genCaches['placeInGen'][key].copy()
    # else:
    gen = placeInGenAux(obj, regShape, conf, confDep, prob, reachObsts, inConds,
                        legalGrasps, considerOtherIns, world)
    memo = Memoizer('placeInGen', gen)
    # bState.genCaches['placeInGen'][key] = memo
    for x in memo: yield x

def placeInGenAux(obj, regShape, conf, confDep, prob, reachObsts, inConds,
                  legalGrasps, considerOtherIns, world):
    objV = world.objV
    objDelta = world.objDelta
    graspV = varTuple(world.graspD)
    confV = varTuple(world.confD)
    bState = world.bState
    fixedObst = bState.obstShape
    if reachObsts: debugDraw('placeInGen', Shape(reachObsts), curWin)
    thisInCond = [(obj, regShape, objV, objDelta, prob)]
    # Pick an order for achieving other In conditions
    # Recall we're doing regression, so placement order is reverse of inConds
    if considerOtherIns:
        perm = permutations(inConds)
    else:
        perm = [[]]
    for otherInConds in perm:
        # Note that we have (pose + fixed) that constrain places and
        # paths and (reachable) that constrain places but not paths.
        # Returns only distinct choices for obj (first of inConds)
        pgccGen = candidatePGCC(thisInCond + list(otherInConds),
                                 reachObsts, legalGrasps, prob, world)
        for pgcc, viol in pgccGen:
            (p, g, cf, ca) = pgcc
            if debug('placeInGen'):
                drawObjAndShadow(bState, obj,
                                 p, objV, objDelta, prob, curWin, color='cyan')
                regShape.draw(curWin, 'magenta')
                wm.getWindow(curWin).update()
                debugMsg('placeInGen', ('->', pgcc))
            yield pgcc, viol
        # for pgcc, viol in pgccGen:
        #     (p, g, c, ca) = pgcc
        #     cf = conf or c
        #     ca = confDep or findApproachConf(obj, p, objV, cf, confV, prob, bState)
        #     viol = canPlaceTest(obj, cf, ca, p, g, prob, world, True)
        #     if viol:
        #         pgcc = (p, g, cf, ca)
        #         if debug('placeInGen'):
        #             drawObjAndShadow(bState, obj,
        #                              pgcc[0], objV, prob, curWin, 'cyan')
        #             regShape.draw(curWin, 'magenta')
        #             wm.getWindow(curWin).update()
        #         debugMsg('placeInGen', ('->', pgcc))
        #         yield pgcc, viol
    debugMsg('placeInGenFail', 'out of values')

# returns values for (?obj, ?pose, ?var) for an object to clear the way to
# conf (while holding heldObj) in grasp (heldObj and grasp may be None).
# confV here is the uncertainty while the robot is traversing the path

# Really, free variables used when placing objects out of the way
clearObjVar = (0.01, 0.01)
clearConfVar = (0.005, 0.005)
clearGraspVar = (0.005, 0.005)

clearObjDelta = (0.05, 0.05)
clearConfDelta = (0.02, 0.02)
clearGraspDelta = (0.01, 0.01)

clearPlaceObjVar = (0.01, 0.01)
clearLookObjVar = (0.001, 0.001)

# returns (blocker.name, pose, clearObjVar)
def clearToHomeGen(args, goalConds, bState, outBindings):
    for ans, viol in clearToHomeGenTop(args, goalConds, bState, outBindings):
        debugMsg('clearGen', ('->', ans))
        yield ans

def clearToHomeGenTop(args, goalConds, bState, outBindings):
    (conf, heldObj, grasp, graspV, graspDelta, confV, confDelta, prob, cond) = args
    assert isGround((heldObj, grasp, graspV, confV, graspDelta, confDelta,  prob))
    if isAnyVar(conf):
        return
    # Use immovable obsts when finding a path
    debugMsg('clearGen', ('Cond in fluent', cond))
    world = W(bState, held=heldObj).updateFromGoalPoses(goalConds)
    gc = getConf(cond)
    if gc: world.updateConf(modeTuple(gc[0]))
    world.excludeObjs([heldObj])
    world.updateGrasp(grasp, graspV, graspDelta)
    # Use min variance, we might need to localize later to reach conf
    world.updateConfV(minRobotVarianceTuple, confDelta)
    debugMsg('clearGen',
             zip(('conf', 'heldObj', 'grasp', 'graspV', 'confV', 'prob', 'cond'), args),
             outBindings, world)
    path, viol = canReachHome(conf, prob, world)
    if not path:
        debugMsg('clearGen', 'no path')
        return
    blockers = set(blockingObjs(path, prob, world))
    debugMsg('clearGen', ('blockers', blockers))
    if not blockers:
        return
    for candidate in clearToHomeGenAux(blockers, prob, goalConds, bState,
                                       clearPlaceObjVar):
        debugMsg('clearGen', ('->', candidate))
        yield candidate
    debugMsg('clearGen', 'out of candidates')

def clearToHomeGenAux(blockers, prob, goalConds, bState, clearObjVar):
    # Look for out of the way poses for one of the blockers, ideally
    # the "innermost" blocker.
    candidates = []
    for blocker in blockers:
        ## placeInGen makes sure that we avoid any reachObsts, which
        ## will include one for this region that we're trying to clear.

        ## For a call to placeInGen, we really need to know the objV to
        ## know whether the object will fit in the region.   Need confV
        ## and graspV to test likely reachability.

        debugMsg('placeInGen', ('Called for blockers', blocker.name, bState.awayRegion))
        pgccGen = placeInGenTop((blocker.name, bState.awayRegion, clearObjVar,
                                 clearGraspVar, clearConfVar,
                                 clearObjDelta, clearGraspDelta, clearConfDelta, prob),
                                goalConds, bState, 4*['?x'], 
                                considerOtherIns = False)
        for (p, g, c, ca), viol in pgccGen:
            if (blocker.name, p) in candidates: continue
            if debug('clearGen'):
                drawPoseConf(bState, blocker.name, p, c, ca, curWin,
                             color = 'brown')
            candidates.append((blocker.name, p))
            yield (blocker.name, p, clearObjVar), viol

def clearLookGen(args, goalConds, bState, outBindings):
    (conf, heldObj, grasp, graspV, graspDelta, confV, confDelta, objDelta, prob) = args
    assert isGround((heldObj, grasp, graspV, graspDelta, confV, confDelta, prob))
    if isAnyVar(conf):
        return
    world = W(bState, held=heldObj).updateFromGoalPoses(goalConds)
    world.excludeObjs([heldObj])
    world.updateGrasp(grasp, graspV, graspDelta)
    world.updateConfV(confV, confDelta)
    debugMsg('clearLookGen',
             zip(('conf', 'heldObj', 'grasp', 'graspV', 'confV', 'prob'), args),
             outBindings, world)
    path, viol = canReachHome(conf, prob, world)
    if not path:
        debugMsg('clearLookGen', 'no path')
        return
    bo = set([o.name for o in blockingObjs(path, prob, world)])
    boLook = set([o.name for o in blockingObjs(path, prob, world, minObjectVarianceTuple)])
    # if an object is still blocking with minVariance, then ignore
    blockers = bo - boLook
    debugMsg('clearLookGen', ('bo', bo), ('boLook', boLook))
    debugMsg('clearLookGen', ('blockers', blockers))
    for blocker in blockers:
        ans = (blocker,
               modeTuple(bState.objects[blocker]),
               clearLookObjVar)
        debugMsg('clearLookGen', ('->', ans))
        yield ans

# returns (conf)
# Made it avoid moving on top of other objects
def lookGen(args, goalConds, bState, outBindings):
    for ans, viol in lookGenTop(args, goalConds, bState, outBindings):
        debugMsg('lookGen', ('->', ans))
        yield [ans]

def lookGenTop(args, goalConds, bState, outBindings):
    (obj, pose, objV, objDelta, confV, confDelta, prob) = args
    assert isGround((obj, pose, objV, confV, prob))
    # Set up world
    world = W(bState, objV=objV, objDelta=objDelta).updateFromGoalPoses(goalConds).excludeObjs([obj])
    world.updateConfV(confV,confDelta).updatePermObjPose(obj, pose, objV, objDelta)
    debugMsg('lookGen',
             zip(('obj', 'pose', 'objV', 'objDelta', 'confV', 'confDelta', 'prob'), args),
             outBindings, world)
    goalConfs = getConf(goalConds)
    # lookGen has randomization -- it can't be cached
    # key = (tuple(args), world, frozenset(goalConfs))
    # if key in bState.genCaches['lookGen']:
    #     memo = bState.genCaches['lookGen'][key].copy()
    # else:
    gen = lookGenAux(bState, obj, pose, prob, world, goalConfs)
    memo = Memoizer('lookGen', gen)
    # bState.genCaches['lookGen'][key] = memo
    for x in memo:
        yield x

def lookGenAux(bState, obj, pose, prob, world, goalConfs):
    objV = world.objV
    objDelta = world.objDelta
    (_, confV, confDelta) = world.confD
    bState = world.bState
    robot = bState.robot.shape
    sensor = bboxCenter(robot.bbox())
    shape = getShadow(obj, pose, objV, objDelta, prob, bState)
    fixedObst = bState.obstShape
    poseObsts = getShadows(world.fixObjBs, prob, bState)
    obstRects = Shape([fixedObst] + [o for o in poseObsts if o.name != obj]).rects()
    confs = goalConfs if goalConfs else [modeTuple(bState.conf)]
    for conf in confs:                  # there should only be one (at most)
        if visibleFromPoint(shape, compose(confs[0], sensor), obstRects):
            # Don't go into the valley of the shadow of the obj
            path, viol = canReachHome(conf, prob, world, avoidShadow=[obj])
            if viol:
                if debug('lookGen'):
                    robot.applyLoc(conf).draw(curWin, color='green')
                    wm.getWindow(curWin).update()
                debugMsg('lookGen', 'current Conf ->', conf)
                yield conf, Violations()
    # If there was a goal conf and it didn't work, we're done. -- LPK
    if goalConfs:
        return

    # find other confs if necessary
    visReg = visibleRegion(obj, pose, objV, objDelta, prob, bState)
    if debug('lookGen'):
        bState.draw()
        visReg.draw(curWin, 'purple')
    # These are robot poses, that is, confs...
    bbl = candidatePoses(robot, visReg, None, [fixedObst, shape] + poseObsts)
    if debug('lookGen'):
        drawBBL(bbl, curWin, 'pink')
    for conf in interiorBBLGen(bbl):
        if debug('lookGen'):
            shape.draw(curWin, 'magenta')
        if visibleFromPoint(shape, compose(conf, sensor), obstRects):
            # Don't go into the valley of the shadow of the obj
            path, viol = canReachHome(conf, prob, world, avoidShadow=[obj])
            if viol:
                if debug('lookGen'):
                    robot.applyLoc(conf).draw(curWin, color='green')
                    wm.getWindow(curWin).update()
                    debugMsg('lookGen', '->', conf)
                yield conf, viol
        else:
            if debug('lookGen'):
                robot.applyLoc(conf).draw(curWin, color='red')
                wm.getWindow(curWin).update()
                debugMsg('lookGen', 'this one does not work!', conf)
    debugMsg('lookGen', 'failed to find conf')

# simple for now, a box of a given (half-)width
maxLookDistance = 3.0
def visibleRegion(obj, pose, objV, objDelta, prob, bState):
    ctr = bboxCenter(bState.objShapes[obj].applyLoc(pose).bbox())
    d = maxLookDistance
    return bboxRect([[ctr[0]-d, ctr[1]-d], [ctr[0]+d, ctr[1]+d]],
                    name='visRegion')

def clearLocalizeGen(args, goalConds, bState, outBindings):
    (conf, heldObj, grasp, graspV, confV, graspDelta, confDelta,  prob) = args
    if isAnyVar(conf): return
    assert isGround((heldObj, grasp, graspV, confV, graspDelta, confDelta,  prob))
    world = W(bState, held=heldObj).updateFromGoalPoses(goalConds)
    world.excludeObjs([heldObj])
    world.updateGrasp(grasp, graspV, graspDelta)
    # We want to know: if confV is minimal, can we reach home
    world.updateConfV(minRobotVarianceTuple, confDelta)
    debugMsg('clearLocalizeGen',
             zip((conf, heldObj, grasp, graspV, confV, graspDelta, confDelta,  prob), args),
             outBindings, world)
    path, viol = canReachHome(conf, prob, world)
    if viol:                              # viol is None for failure
        debugMsg('clearLocalizeGen', 'found path')
        return [[True]]
    else:
        debugMsg('clearLocalizeGen', 'no path')
        return [[False]]

################
# SUPPORT FUNCTIONS
################

# Find objects in the initial state that are blocking the path to conf
def blockingObjs(path, prob, world, lookVar = None):
    robShadow = robotShadow(world.bState, (0,0),
                            varTuple(world.confD), delTuple(world.confD),
                            world.held, world.graspD, prob)
    # Objects that can be suggested to move
    # Use lookVar if it's specified
    obstShadows = getShadows(world.moveObjBs, prob, world.bState, lookVar)
    debugDraw('blockingObjs', Shape(obstShadows), curWin, color = 'black')
    blockers = set([])
    for (_, p) in path:
        rob = robShadow.applyLoc(p)
        debugDraw('blockingObjs', rob, curWin, color = 'orchid')
        for obst in obstShadows:
            if obst not in blockers and rob.collides(obst):
                blockers.add(obst)
    if debug('blockingObjs'):
        raw_input('blocking path')
    return blockers

def robotGraspShadow(bState, conf, confV, confDelta, obj, objPose, objV, objDelta, prob):

    ## !! Give a little "cushion"
    confV = tuple([v+0.01 for v in confV])
    
    ca = findApproachConf(obj, (objPose, objV, objDelta),
                          (conf, confV, confDelta), prob, bState, multiplier=1.5)
    sh1 = robotShadow(bState, conf, confV, confDelta,  'none', None, prob)
    sh2 = robotShadow(bState, ca, confV, confDelta, 'none', None, prob)
    bb = bboxUnion(sh1.bbox(), sh2.bbox())
    org = sh1.origin
    sh3 = Rect(bb[1][0] - org[0], bb[1][1] - org[1], org,
               org[0] - bb[0][0], org[1] - bb[0][1], name='robotGraspShadow')
    if debug('robotGraspShadow'):
        print 'sh1', sh1.bbox()
        print 'sh2', sh2.bbox()
        print 'bb ', bb
        print 'sh3', sh3.bbox()
        sh1.draw(curWin, 'green')
        sh2.draw(curWin, 'red')
        sh3.draw(curWin, 'cyan')
        if pause('robotGraspShadow'):
            raw_input('robotGraspShadow-Go?')
    return sh3

def findApproachConf(obj, (objPose, objV, objDelta),
                     (graspConf, confV, confDelta), probability, bState,
                     multiplier = 1.2):
    debugMsg('findApproachConf',
             ('obj', obj, 'objPose', objPose, 'objV', objV),
             ('objDelta', objDelta, 'graspConf', graspConf, 'confV', confV),
             ('confDelta', confDelta, 'probability', probability))
    objShWidths = shadowWidths(objV, objDelta, probability)
    robShWidths = shadowWidths(confV, confDelta, probability)
    sumShWidths = [1.2*(o+r) for (o,r) in zip(objShWidths, robShWidths)]
    bbObj = bState.objShapes[obj].bbox()
    bbRob = bState.robot.shape.bbox()
    if bbObj[1][0] + objPose[0] <= bbRob[0][0] + graspConf[0]:
        off = (sumShWidths[0], 0)
    elif bbObj[0][0] + objPose[0] >= bbRob[1][0] + graspConf[0]:
        off = (-sumShWidths[0], 0)
    elif bbObj[1][1] + objPose[1] <= bbRob[0][1] + graspConf[1]:
        off = (0, sumShWidths[1])
    elif bbObj[0][1] + objPose[1] >= bbRob[1][1] + graspConf[1]:
        off = (0, -sumShWidths[1])
    else:
        assert False, 'Bad findApproachConf - likely bad grasp definition for %s'%obj
    return compose(graspConf, off)

maxPoses = 20

# returns lists of (pose, grasp, conf)
def candidatePGCC(inCondsRev, reachObsts, legalGrasps, prob, iworld):
    # REVERSE THE INCONDS -- because regression is in opposite order
    inConds = inCondsRev[::-1]
    debugMsg('candidatePGCC', ('inConds - reversed', inConds))
    objs = [obj for (obj,_,_,_,_) in inConds]
    objVs = [ov for (_,_,ov,_,_) in inConds]
    objDeltas = [ov for (_,_,_,od,_) in inConds]
    world = iworld.copy().excludeObjs(objs) # we'll position objs
    bState = world.bState
    (_, confV, confDelta) = world.confD
    poseObsts = getShadows(world.moveObjBs, prob, world.bState)
    fixedObst = bState.obstShape         # Grown?
    # inCond is (obj, regShape, objV, objDelta, prob)
    allObsts = [fixedObst] + poseObsts + reachObsts
    robObsts = [fixedObst] + poseObsts
    ## !! Gives it a little cushion
    objShadows = [getShadow(obj, (0,0), tuple([v+0.01 for v in objV]), objDelta, prob, bState) \
                  for (obj, _, objV, objDelta, prob) in inConds]
    regShapes = [regShape for (_,regShape,_,_,_) in inConds]
    robot = bState.robot
    robShadow = robotShadow(bState, (0,0), confV, confDelta, 'none', None, prob)
    # 1. Find plausible grasps -- could skip and just use legalGrasps
    graspsPerObj1 = {}
    ho = bState.heldObj.mode()
    hoGrasp = bState.grasp[ho].modeTuple() if ho != 'none' else None
    hoConf = bState.conf.modeTuple() if ho != 'none' else None
    hoPose = graspPose(bState, hoConf, hoGrasp) if ho != 'none' else None
    for (obj, objShadow, regShape, objV) in zip(objs, objShadows, regShapes, objVs):
        # using held object saves one motion... so weight=0
        graspsPerObj1[obj] = [(0.0, hoGrasp)] if ho == obj else []
        for grasp in legalGrasps.get(obj, None) or bState.legalGrasps[obj]:
            if grasp == hoGrasp and ho == obj: continue
            if not candidatePoses(objShadow, regShape, grasp, allObsts,
                                  robShadow, robObsts): continue
            # Check whether obj is graspable in current location
            pose = modeTuple(bState.objects[obj])
            pickConf = graspConf(bState, pose, grasp)
            viol = canPickTest(obj, None, pickConf, pose, grasp, prob, world)
            if viol:
                # weight is slightly higher for objct not in hand
                graspsPerObj1[obj].append((viol.weight()+1, grasp))
            else:
                graspsPerObj1[obj].append((10., grasp))
        if not graspsPerObj1[obj]:
            debugMsg('candidatePGCCFail', ('failed to find grasp for', obj))
            return
    debugMsg('candidatePGCC', ('weighted grasps', graspsPerObj1))
    prefPose = {}
    if ho != 'none' and ho in objs:     #  specify desired pose
        prefPose[ho] = hoPose
    graspsPerObj = graspWeightsToProb(graspsPerObj1)
    # 2. Find combinations of poses and grasps
    posesForGrasps = {}
    graspDistDict = {}
    totalPoses = 0
    for weightedGrasps in product(*[graspsPerObj[obj] for obj in objs]):
        # a list of lists of poses (one for each objShape)
        robShadowFn = lambda obj, grasp: robotGraspShadow(bState, (0,0), confV, confDelta, \
                                                          obj, grasp, objV, objDelta,  prob)
        grasps = tuple([g for (w, g) in weightedGrasps])
        posesList = candidatePosesSamples(objShadows, regShapes, grasps, allObsts,
                                      robShadowFn, robObsts, prefPose)
        if debug('candidatePGCC'):
            bState.draw(curWin)
            for poses in posesList:
                for pose, obj in zip(poses, objs):
                    bState.objShapes[obj].applyLoc(pose).draw(curWin)
            if pause('candidatePGCC'): raw_input('Poses')
        
        posesForGrasps[grasps] = posesList
        graspDistDict[grasps] = reduce(lambda x,y: x*y, [w for (w, g) in weightedGrasps])
        totalPoses += len(posesList)
        if debug('candidatePGCC'):
            print grasps, 'p=', graspDistDict[grasps], 'Found %d poses'%len(posesList)
    worldCopy = world.copy()
    graspDist = DDist(graspDistDict)
    graspDist.normalize()
    first = True if prefPose else False
    genPoses = 0
    if prefPose: print 'prefPose', prefPose
    print graspDist
    for i in xrange(totalPoses):
        if genPoses > maxPoses: return
        grasps = graspDist.draw()
        if not grasps in posesForGrasps: continue
        print 'Chose grasps', grasps, 'with prob', graspDist.prob(grasps)
        # debugMsg('candidatePGCC', ('Chose grasps', grasps, 'with prob', graspDist.prob(grasps)))
        posesList = posesForGrasps[grasps]
        if posesList:
            if first:
                poses = posesList[0]
                debugMsg('candidatePGCC', ('poses (first)', poses))
                first = False
            else:
                poses = random.choice(posesList)
                debugMsg('candidatePGCC', ('poses (random)', poses))
            posesList.remove(poses)
            if not posesList: del posesForGrasps[grasps]
        else:
            continue
        # 3. Pick confs and check reachability (to home)
        # Place the other objects in the world
        worldCopy.fixObjBs = poseReplace(objs[:-1], poses[:-1], world.fixObjBs)
        # worldCopy.moveObjBs = poseReplace(objs[:-1], poses[:-1], world.moveObjBs)
        obj = objs[-1]
        pose = poses[-1]
        grasp = grasps[-1]
        conf = graspConf(bState, pose, grasp)
        ca = findApproachConf(obj, (pose, objV, objDelta), (conf, confV, confDelta), prob, bState)
        viol = canPlaceTest(obj, conf, ca, pose, grasp, prob, worldCopy)
        if viol:
            # print 'viol.penalty', viol.penalty, viol
            # viol = Violations(viol.obstacles, viol.shadows, viol.penalty+(1-graspDist.prob(grasps)))
            if debug('candidatePGCC'):
                for obj, grasp, pose in zip(objs, grasps, poses):
                    bState.objShapes[obj].applyLoc(pose).draw(curWin)
                    robShadow = robotGraspShadow(bState, (0,0), confV, confDelta, obj, grasp, objV, objDelta, prob)
                    robShadow.applyLoc(conf).draw(curWin, 'magenta')
                    debugMsg('candidatePGCC', list(zip(objs, grasps, poses)))
            genPoses += 1
            yield (tuple(poses[-1]), grasps[-1], conf, ca), viol

def graspWeightsToProb(graspsPerObj):
    graspProbPerObj = {}
    for obj in graspsPerObj:
        weights = [1./(w+1)**4 for (w,g) in graspsPerObj[obj]]
        total = sum(weights)
        normalWeights = [w/total for w in weights]
        graspProbPerObj[obj] = zip(normalWeights, [g for (w,g) in graspsPerObj[obj]])
    return graspProbPerObj

def poseReplace(objs, poses, poseDs):
    if not objs:
        return poseDs
    poseDsNew = poseDs.copy()
    for obj, pose in zip(objs, poses):
        if obj in poseDs:
            poseDsNew[obj] = (pose,) +  poseDs[obj][1:]
        else:
            poseDsNew[obj] = (pose, minObjectVarianceTuple, (0., 0.))
    return poseDsNew

# An alternative approach: Find bdry(CO_B(A)) = bBA, placements of B
# in contact with A We want to find a placement for B relative to A
# that lie on this boundary, while pose of A is in CI_A(R)=IA and pose
# of B is in CI_B(R)=IB.  That is, we want point p in IB and q in IA
# st p-q in in bBA.  The following is a brute-force "sampling" method.

# It is important that the order of targets is the actual order that
# they will be placed in the region.  If it is the reverse order, then
# there is more subtle manipulation of obstacles required.
def candidatePosesSamples(targets, regions, grasps, obsts, robotFn, robotObsts, prefPose):
    # find placements for first target
    # place target at that location, recurse for second target
    # recurse returns a list of lists of poses, each sublist a pose for each target.
    def recurse(tg, rg, gr, ob, ro, parent):
        # parent is a list of poses, one for each target
        if not tg:
            return [parent]             # a list of a single single answer
        bbl = candidatePoses(tg[0], rg[0], gr[0], ob, robotFn(tg[0].name, gr[0]), ro)
        if not bbl: return []
        ans = []
        # a list of answers
        ## !! Modify to randomly generate poses from the bbl
        for p in boundaryBBL(bbl, prefPose.get(tg[0].name, None)):
            tg0 = tg[0].applyLoc(p)
            ob1 = ob+[tg0]
            ro1 = ro+[tg0]
            ans.extend(recurse(tg[1:], rg[1:], gr[1:], ob1, ro1, parent+[p]))
        return ans
    assert len(targets) == len(grasps)
    debugMsg('candidatePosesSamples', ('targets', targets), ('regions', regions), ('grasps', grasps))
    poses = recurse(targets, regions, grasps, obsts, robotObsts, [])
    ## !! The poses should be sorted.
    debugMsg('candidatePosesSamples', ('->', poses))
    return poses

def visibleFromPoint(object, point, rectObstacles):
    debugMsg('visibleFromPoint', object, point, rectObstacles)
    robj = object.rects()
    robst = [r.vertices() for r in rectObstacles + robj]
    for rect in robj:
        segs = segments(rect.vertices())
        random.shuffle(segs)
        for segment in segs:
            if not all([confDist(point, s) <= maxLookDistance for s in segment]):
                continue
            vp = viewPoly(point, segment)
            if any(polyPolyCollide(vp, r) for r in robst):
                continue
            debugMsg('visibleFromPoint', ('->', True))
            return True
    debugMsg('visibleFromPoint', ('->', False))
    return False

def viewPoly(pt, seg):
    def combine(p1, p2):
        return tuple([0.01*a + 0.99*b for (a, b) in zip(p1, p2)])
    return [pt, combine(pt, seg[0]), combine(pt, seg[1])]

def pathShape(path, prob, world):
    robShadow = robotShadow(world.bState, (0,0), varTuple(world.confD),
                            delList(world.confD), world.held, world.graspD, prob)
    if isinstance(path, (list, tuple)):
        pass
    elif path == True:
        path = [(None, conf)]
    else:
        return
    return Shape([robShadow.applyLoc(p) for (_, p) in path])

####################
# Drawing
####################

def drawPoseConf(bState, obj, pose, conf, confAppr, win, color = None):
    bState.objShapes[obj].applyLoc(pose).draw(win, color=color)
    bState.robot.placement(conf).draw(win, color=color)
    wm.getWindow(win).update()

def drawObjAndShadow(bState, obj, pose, var, delta, prob, win, color = None):
    bState.objShapes[obj].applyLoc(pose).draw(win, color)
    getShadow(obj, pose, var, delta, prob, bState).draw(win, color = color)

def drawPolys(polys, win):
    w = wm.getWindow(win)
    for poly, color in polys:
        for ((x0, y0), (x1, y1)) in segments(poly):
            w.window.drawLineSeg(x0, y0, x1, y1, color=color)

def drawPath(path, prob, world, color = 'brown'):
    pathShape(path, prob, world).draw(curWin, color = color)


####################
# Utilities for distributions
####################

def modD(dist, mode=None, var=None, delta=None):
    return (mode or dist[0], var or dist[1], delta or dist[2])
def poseD(bState, pose):
    return (pose, bState.objV, bState.objDelta)
