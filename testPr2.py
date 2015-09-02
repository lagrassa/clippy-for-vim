import pdb
from pr2ROS import testReactive
import testRig
reload(testRig)
from testRig import *

from timeout import timeout, TimeoutError

# 10 min timeout for all tests
@timeout(1000)
def testFunc(n, skeleton=None, heuristic=habbs, hierarchical=True, easy=False, rip=True):
    eval('test%s(skeleton=skeleton, heuristic=heuristic, hierarchical=hierarchical, easy=easy, rip=rip)'%str(n))

def testRepeat(n, repeat=3, **args):
    for i in range(repeat):
        try:
            testFunc(n, **args)
        except TimeoutError:
            print '************** Timed out **************'

def testAll(indices, repeat=3, crashIsError=True, **args):
    pr2Sim.crashIsError = crashIsError
    for i in indices:
        if i == 0: continue
        testRepeat(i, repeat=repeat, **args)
    print testResults

######################################################################
# Various Settings
######################################################################

sd = 1.0e-3
tz = 0.68
class Experiment:
    defaultVar = (sd**2, sd**2, 1.0e-10, sd**2)
    defaultDelta = 4*(0.0,)
    domainProbs = typicalErrProbs
    operators = allOperators
    varDict = {}
    movePoses = {}
    fixPoses = {}
    regions = []

table1Pose = hu.Pose(1.3, 0.0, 0.0, math.pi/2.0)
table2Pose = hu.Pose(1.0, -1.2, 0.0, 0.0)
table3Pose = hu.Pose(1.6,0.0,0.0, math.pi/2.0),
coolShelvesPose = hu.Pose(1.4, 0.0, tZ, math.pi/2)

bigVar = (0.1**2, 0.1**2, 1e-10, 0.3**2)
medVar = (0.05**2, 0.05**2, 1e-10, 0.1**2)
smallVar = (0.03**2, 0.03**2, 1e-10, 0.06**2)
tinyVar = (0.001**2, 0.001**2, 1e-10, 0.002**2)

targetSmallVar = (0.01**2, 0.01**2, 0.01**2, 0.02**2)

def makeExp(fix, move, reg, easy=False):
    exp = Experiment()
    exp.fixPoses = { x: pose for (x,(pose, var)) in fix.items()}
    exp.movePoses = { x: pose for (x,(pose, var)) in move.items()}
    exp.varDict = {} if easy \
                  else { x: var for (x,(pose, var)) in move.items() + fix.items()}
    exp.regions = reg
    return exp

# Goals
def holding(obj, hand='left', graspType=2, goalProb=0.95,
            targetVar=targetSmallVar, targetDelta=(0.01, 0.01, 0.01, 0.05)):
    return State([Bd([Holding([hand]), obj, goalProb], True),
                  Bd([GraspFace([obj, hand]), graspType, goalProb], True),
                  B([Grasp([obj, hand,  graspType]),
                     (0,-0.025,0,0), targetSmallVar, targetDelta,
                     goalProb], True)])

def placed(obj, pose, hand='left', goalProb=0.95,
           targetVar=targetSmallVar, targetDelta=(0.01, 0.01, 0.01, 0.05)) :
    return State([Bd([SupportFace([obj]), 4, goalProb], True),
                  B([Pose([obj, 4]),
                     pose.xyztTuple(), targetVar, targetDelta,
                     goalProb], True)])

def emptyHand(hand='left', goalProb=0.95):
    return State([Bd([Holding([hand]), 'none', goalProb], True)])
    
def inRegion(objs, regions, goalProb=0.95):
    if isinstance(objs, str): objs = [objs]
    if isinstance(regions, str):
        regions = [regions for o in objs] # if multiple objs, each in region
    assert len(objs) == len(regions)
    return State([Bd([In([obj, reg]), True, goalProb], True) \
                  for obj,reg in zip(objs, regions)])

defaultArgs = {'hpn' : True,
               'hierarchical' : True,
               'heuristic' : habbs,
               'rip' : False,
               'skeleton' : None,
               'easy' : False,
               'multiplier' : 6,
               'initBelief' : None,
               'initWorld' : None}

def doTest(name, exp, goal, skel, args):

    glob.monotonicFirst = True          # always...

    tArgs = defaultArgs.copy()
    tArgs.update(args)
    tArgs['skeleton'] = skel if tArgs['skeleton'] else None
    tArgs['regions'] = exp.regions
    t = PlanTest(name, exp, multiplier=tArgs['multiplier'])
    for x,y in tArgs.items():
        if x not in {'initBelief', 'initWorld'}:
            print x, '<-', y
    t.run(goal, **tArgs)
    return t

######################################################################
# Test 0: 1 table move 1 object
######################################################################

# 1 table; move 1 object
# Assumption is that odometry error is kept in check during motions.
# Use domainProbs.odoError as the stdev of any object.

def test0(**args):
    glob.rebindPenalty = 10
    exp = makeExp({'table1' : (table1Pose, medVar)},
                  {'objA' : (hu.Pose(1.1, 0.0, tZ, 0.0), medVar)},
                  ['table1Top', 'table1Left'], easy=args.get('easy', False))
    goal = inRegion(['objA'], 'table1Left')
    # pick/place, flat
    skel = [[poseAchIn, lookAt.applyBindings({'Obj' : 'objA'}), move, 
             place.applyBindings({'Obj' : 'objA'}),
             move, pick, moveNB,
             lookAt.applyBindings({'Obj' : 'objA'}),
             moveNB, lookAt, move]]
    # pick/place, hierarchical
    hskel = [[poseAchIn],
            [poseAchIn],
            [poseAchIn, lookAt, place],
            [place],
            [place, pick],
            [pick],
            [pick],
            [pick, moveNB, lookAt, move],
            [place, move]]
    return doTest('test0', exp, goal, skel, args)

######################################################################
# Test 1: 2 tables move 1 object
######################################################################

def test1(**args):
    glob.rebindPenalty = 700
    exp = makeExp({'table1' : (table1Pose, smallVar),
                   'table2' : (table2Pose, smallVar)},
                  {'objA' : (hu.Pose(1.1, 0.0, tZ, 0.0), medVar)},
                  ['table1Top', 'table1Left',
                   'table2Top', 'table2Left'], easy=args.get('easy', False))
    goal = inRegion(['objA'], 'table2Left')
    skel = [[poseAchIn, lookAt, moveNB, lookAt, move,
             place, move, pick, moveNB, lookAt, moveNB, lookAt, move]]
    return doTest('test1', exp, goal, skel, args)

######################################################################
# Test 1.5: 2 tables move 1 object: more error on table 1
######################################################################

# pick and place into region.  2 tables
def test1Point5(**args):
    glob.rebindPenalty = 700
    exp = makeExp({'table1' : (table1Pose, medVar),
                   'table2' : (table2Pose, tinyVar)},
                  {'objA' : (hu.Pose(1.2, 0.0, tZ, 0.0), medVar)},
                  ['table1Top', 'table1Left',
                   'table2Top', 'table2Left'], easy=args.get('easy', False))
    goal = inRegion(['objA'], 'table2Left')
    skel = [[poseAchIn, lookAt.applyBindings({'Obj' : 'objA'}),
             move, place.applyBindings({'Obj' : 'objA'}),
             move, pick.applyBindings({'Obj' : 'objA'}),
             moveNB, lookAt.applyBindings({'Obj' : 'objA'}),
             moveNB, lookAt.applyBindings({'Obj' : 'objA'}),
             move, achCanPickPlace,
             move]]
    return doTest('test1', exp, goal, skel, args)

######################################################################
# Test 2: one table, move two objects
######################################################################

def test2(**args):
    glob.rebindPenalty = 700
    exp = makeExp({'table1' : (table1Pose, smallVar)},
                  {'objA' : (hu.Pose(1.1, 0.0, tZ, 0.0), medVar),
                   'objB' : (hu.Pose(1.1, -0.4, tZ, 0.0), medVar)},
                  ['table1Top', 'table1Left'], easy=args.get('easy', False))
    goal = inRegion(['objA', 'objB'], 'table1Left')
    skel = None
    return doTest('test2', exp, goal, skel, args)

######################################################################
# Test 3: Put object in shelves
######################################################################

def test3(**args):
    glob.rebindPenalty = 700
    right1 = hu.Pose(1.1, 0.0, tZ, 0.0) 
    right2 = hu.Pose(1.5, -0.5, tZ, 0.0)
    left1 = hu.Pose(1.1, 0.5, tZ, 0.0)
    left2 = hu.Pose(1.5, 0.5, tZ, 0.0)
    region = 'coolShelves_space_2'
    easy=args.get('easy', False)
    exp = makeExp({'table1' : (table1Pose, smallVar),
                   'coolShelves' : (coolShelvesPose , smallVar)},  # was medVar
                  {'objA' : (right1, medVar),
                   'objB' : (left1, medVar)},
                  [region, 'table1Top'], easy=easy)
    goal1 = inRegion('objA', region)
    goal2 = inRegion(['objA', 'objB'], region)
    # pick/place, one object, flat
    if easy:
        skel = [[poseAchIn, lookAt.applyBindings({'Obj' : 'objA'}),
                 move, place.applyBindings({'Obj' : 'objA'}),
                 move, pick, move]]
    else:
        skel = [[poseAchIn, lookAt.applyBindings({'Obj' : 'objA'}),
             move, place.applyBindings({'Obj' : 'objA'}),
             move, pick.applyBindings({'Obj' : 'objA'}),
             moveNB, lookAt.applyBindings({'Obj' : 'objA'}),
             moveNB, lookAt.applyBindings({'Obj' : 'objA'}),
             move, achCanPickPlace,
             moveNB, lookAt.applyBindings({'Obj' : 'coolShelves'}),
             move, achCanReach,
             moveNB, lookAt.applyBindings({'Obj' : 'objA'}),
             move]]
    return doTest('test3', exp, goal1, skel, args)

######################################################################
# Test 4: Boring pick
######################################################################

def test4(gt=0, **args):
    glob.rebindPenalty = 100
    exp = makeExp({'table1' : (table1Pose, smallVar)},
                  {'objA' : (hu.Pose(1.2, 0.0, tZ, 0.0), medVar)},
                  [], easy=args.get('easy', False))
    goal = holding('objA', hand='right', graspType=gt)
    skel =  [[pick, moveNB, lookAt, move, lookAt, move, lookAt]]
    return doTest('test4', exp, goal, skel, args)

######################################################################
# Test Put Down
#       Start with something in the hand
######################################################################

def testWithBInHand(name, goal, args):
    glob.rebindPenalty = 150
    front = hu.Pose(1.1, 0.0, tZ, 0.0)
    back = hu.Pose(1.4, 0.0, tZ, 0.0)
    exp = makeExp({'table1' : (table1Pose, smallVar),
                   'table2' : (table2Pose, smallVar)},
                  {'objA' : (back, medVar),
                   'objB' : (front, medVar)},
                  ['table1Left'], easy=args.get('easy', False))
    grasped = 'objB'; hand = 'left'
    args['initBelief'] = lambda bs: makeInitBel(bs, grasped, hand, gf=0)
    args['initWorld'] = lambda bs,rw: makeAttachedWorldFromPBS(bs.pbs, rw, grasped, hand)
    skel = None
    return doTest(name, exp, goal, skel, args)

def test5(**args):
    goal = emptyHand()
    testWithBInHand('test5', goal, args)
    
######################################################################
# Test Pick Up
#       Start with something in the hand
######################################################################

def test6(**args):
    goal = holding('objA', 'left', 2)
    testWithBInHand('test6', goal, args)
    
######################################################################
# Test Placing
#       Start with something in the hand
######################################################################

def test7(**args):
    front = hu.Pose(1.1, 0.0, tZ, 0.0)
    goal = placed('objA', front, 'left')
    testWithBInHand('test7', goal, args)

######################################################################
# Test Regrasp
#       Start with something in the hand
######################################################################

def test8(**args):
    goal = holding('objB', 'left', 1, goalProb=0.7)
    testWithBInHand('test8', goal, args)

######################################################################
#       Another test.  Picking something up from the back.
#       shouldn't be hard.
######################################################################

def test9(**args):
    glob.rebindPenalty = 150
    front = hu.Pose(1.1, 0.0, tZ, 0.0)
    back = hu.Pose(1.4, 0.0, tZ, 0.0)
    exp = makeExp({'table1' : (table1Pose, smallVar),
                   'table2' : (table2Pose, smallVar)},
                  {'objA' : (back, medVar),
                   'objB' : (front, medVar)},
                  ['table1Left'], easy=args.get('easy', False))
    goal1 = holding('objA', 'left', 2)
    goal2 = placed('objA', front)
    skel = None
    return doTest('test9', exp, goal1, skel, args)

######################################################################
# Test Swap
######################################################################

def testSwap(hardSwap = False, **args):
    glob.rebindPenalty = 150
    front = hu.Pose(1.1, 0.0, tZ, 0.0)
    back = hu.Pose(1.25, 0.0, tZ, 0.0)
    parking1 = hu.Pose(0.95, 0.3, tZ, 0.0)
    parking2 = hu.Pose(0.95, -0.3, tZ, 0.0)
    exp = makeExp({'table1' : (table1Pose, smallVar),
                   'table2' : (table2Pose, smallVar)},
                  {'objA' : (back, medVar),
                   'objB' : (front, medVar)},
                  ['table1Top', 'table2Top', 'table1MidFront',
                   'table1MidRear'], easy=args.get('easy', False))
    goal = inRegion(['objA', 'objB'], ['table1MidRear', 'table1MidFront'])
    # A on other table
    goal1 = inRegion('objA', 'table2Top')
    skel1 = [[poseAchIn, 
              place, moveNB, lookAt, move,
              pick, moveNB, lookAt, moveNB, lookAt, move, lookAt, moveNB]]
    # A and B on other table
    goal2 = inRegion(['objA', 'objB'], 'table2Top')
    # B in back
    goal3 = inRegion('objB', 'table1MidRear')
    actualGoal = goal if hardSwap else goal3
    skel = None
    return doTest('testSwap', exp, actualGoal, skel, args)

def testHardSwap(**keys):
    return testSwap(hardSwap = True, **keys)

######################################################################
# Test Swap with clutter
######################################################################

def testBusy(hardSwap = False, **args):
    glob.rebindPenalty = 150
    # Put this back to make the problem harder
    #back = hu.Pose(1.1, 0.0, tZ, 0.0)
    back = hu.Pose(1.25, 0.0, tZ, 0.0)
    parking1 = hu.Pose(0.95, 0.3, tZ, 0.0)
    parking2 = hu.Pose(0.95, -0.3, tZ, 0.0)
    exp = makeExp({'table1' : (table1Pose, smallVar),
                   'table2' : (table2Pose, smallVar)},
                  {'objA' : (back, medVar),
                   'objB': (hu.Pose(0.95, -0.4, tZ, 0.0), medVar),
                   'objC': (hu.Pose(0.45, -1.2, tZ, 0.0), medVar),
                   'objD': (hu.Pose(0.95, -0.2, tZ, 0.0), medVar),
                   'objE': (hu.Pose(0.95, 0.0, tZ, 0.0), medVar),
                   'objF': (hu.Pose(0.95, 0.2, tZ, 0.0), medVar),
                   'objG': (hu.Pose(0.95, 0.4, tZ, 0.0), medVar)},
                  ['table1Top', 'table2Top', 'table1MidFront',
                   'table1MidRear'], easy=args.get('easy', False))
    goal = inRegion(['objA', 'objB'], ['table1MidRear', 'table1MidFront'])
    # A on other table
    goal1 = inRegion('objA', 'table2Top')
    skel1 = [[poseAchIn, 
              place, moveNB, lookAt, move,
              pick, moveNB, lookAt, moveNB, lookAt, move, lookAt, moveNB]]
    # A and B on other table
    goal2 = inRegion(['objA', 'objB'], 'table2Top')
    # B in back
    goal3 = inRegion('objB', 'table1MidRear')
    actualGoal = goal if hardSwap else goal3
    skel = None
    return doTest('testBusy', exp, actualGoal, skel, args)

######################################################################
# Test One Push
######################################################################

def testPush(name, objName, startPose, targetPose, **args):
    glob.rebindPenalty = 50
    exp = makeExp({'table1' : (table1Pose, smallVar)},
                  {objName : (startPose, medVar)},
                  ['table1Top'], easy=args.get('easy', False))
    goal = placed(objName, targetPose, targetDelta = (0.1, .1, .1, .5))
    # One push, no uncertainty
    skel = [[lookAt, move, push, moveNB, lookAt,
             move, lookAt, moveNB]]
    return doTest(name, exp, goal, skel, args)

def testPush0(objName='bigA', **args):
    testPush('testPush0', objName,
             hu.Pose(1.2, 0.0, tZ, 0.0),
             hu.Pose(1.2, 0.4, tZ, 0.0), **args)

######################################################################
# Test Two Pushes, easier
######################################################################

def testPush1(objName='bigA', **args):
    testPush('testPush1', objName,
             hu.Pose(1.2, 0.0, tZ, 0.0),
             hu.Pose(1.4, 0.4, tZ, 0.0), **args)
    
######################################################################
# Test TWo Pushes, harder
######################################################################

def testPush2(objName='bigA', **args):
    testPush('testPush2', objName,
             hu.Pose(1.2, 0.0, tZ, 0.0),
             hu.Pose(1.5, 0.4, tZ, 0.0), **args)

######################################################################
# Test TWo Pushes around shelves
######################################################################

def testPushShelves(name, objName, startPose, targetPose,
                    startPoseB, **args):
    glob.rebindPenalty = 50
    exp = makeExp({'table1' : (table1Pose, smallVar),
                   'coolShelves' : (coolShelvesPose, smallVar)},
                  {objName : (startPose, medVar),
                   'objB' : (startPoseB, medVar)},
                  ['table1Top'], easy=args.get('easy', False))
    goal = placed(objName, targetPose, targetDelta = (0.1, .1, .1, .5))
    # pick and place!
    skel = [[lookAt, move, place, move, 
             pick, moveNB, lookAt, moveNB, lookAt, move]]
    # One push, no uncertainty
    skel = [[lookAt, move, push, moveNB, lookAt,
             move, lookAt, moveNB]]
    return doTest(name, exp, goal, skel, args)

def testPush3(objName='bigA', **args):
    testPushShelves('testPush3', objName,
                    hu.Pose(1.05, 0.0, tZ, 0.0),
                    hu.Pose(1.5, 0.5, tZ, 0.0),
                    hu.Pose(1.05, -0.4, tZ, 0.0), # out of the way
                    **args)

######################################################################
# Move obj b out of the way to push bigA
######################################################################

def testPush4(objName='bigA', **args):
    testPushShelves('testPush3', objName,
                    hu.Pose(1.05, 0.0, tZ, 0.0),
                    hu.Pose(1.5, 0.5, tZ, 0.0),
                    hu.Pose(1.05, 0.4, tZ, 0.0), # in the way
                    **args)

######################################################################
# Push to Region
######################################################################

def testPush5(objName = 'bigA', **args):
    glob.rebindPenalty = 50
    exp = makeExp({'table1' : (table1Pose, smallVar)},
                  {objName : (hu.Pose(1.1, 0.0, tZ, 0.0), medVar)},
                  ['table1Top', 'table1Left'], easy=args.get('easy', False))
    goal = inRegion(objName, 'table1Left')
    skel = None
    return doTest('testPush5', exp, goal, skel, args)

######################################################################
# Push objects out of the way to gras bigA
######################################################################

def testPush6(objName = 'objA', **args):
    glob.rebindPenalty = 50
    front = hu.Pose(1.1, 0.0, tZ, 0.0)
    back = hu.Pose(1.4, 0.0, tZ, 0.0)
    right = hu.Pose(1.25, -0.16, tZ, 0.0)
    left = hu.Pose(1.25, 0.16, tZ, 0.0)
    middle = hu.Pose(1.25, 0.0, tZ, 0.0)
    exp = makeExp({'table1' : (table1Pose, smallVar)},
                  {'objA': (middle, medVar),
                   'bigB': (front, medVar),
                   'tallC': (back, medVar),
                   'barD': (right, medVar),
                   'barE': (left, medVar)},
                  ['table1Top'], easy=args.get('easy', False))
    goal = holding('objA', 'left', 2)
    skel = None
    return doTest('testPush6', exp, goal, skel, args)

######################################################################
# Utilities
######################################################################

def prof(test, n=100):
    import cProfile
    import pstats
    cProfile.run(test, 'prof')
    p = pstats.Stats('prof')
    p.sort_stats('cumulative').print_stats(n)
    # p.sort_stats('cumulative').print_callers(n)

def profPrint(n=100):
    import pstats
    p = pstats.Stats('prof')
    p.sort_stats('cumulative').print_stats(n)
    p.sort_stats('cumulative').print_callers(n)

def clearCachesS(s):
    s.valueCache.clear()
    s.relaxedValueCache.clear()
    clearCaches(s.details)

# Evaluate on details and a fluent to flush the caches and evaluate
def firstAid(details, fluent = None):
    clearCaches(details)
    glob.debugOn.extend(['confReachViol', 'confViolations'])
    if fluent:
        fluent.args[0].viols = {}
        return fluent.valueInDetails(details)

def canPPDebug(details, fluent):
    glob.debugOn.extend(['confViolations', 'canPickPlace'])
    pbs = details.pbs
    # pbs.getRoadMap().confReachCache.clear()
    # bc = pbs.beliefContext
    # bc.pathObstCache.clear()
    # bc.objectShadowCache.clear()
    # for c in bc.genCaches.values():
    #     c.clear()
    # pr2GenAux.graspConfGenCache.clear()
    # bc.world.robot.cacheReset()
    # pr2Visible.cache.clear()
    # belief.hCacheReset()

    rf = fluent.args[0]
    conds = rf.getConds()
    for c in conds:
        c.viols = {}; c.hviols = {}
        c.getViols(details, True, fluent.args[-1])

# Get false fluents
def ff(g, details):
    return [thing for thing in g.fluents if thing.isGround() \
            and thing.valueInDetails(details) == False]

def testReact():
    t = PlanTest('testReact', typicalErrProbs, allOperators, multiplier = 1)
    startConf = makeConf(t.world.robot, 0.0, 0.0, 0.0)
    result, cnfOut, _ = pr2GoToConf(startConf, 'move')
    result, cnfOut, _ = pr2GoToConf(startConf, 'look')
    # Reset the internal coordinate frames
    result, cnfOut, _ = pr2GoToConf(startConf, 'reset')
    testReactive(startConf)

def gripOpen(conf, hand, width=0.08):
    return conf.set(conf.robot.gripperChainNames[hand], [width])

def testOpen(hand='left'):
    t = PlanTest('testReact', typicalErrProbs, allOperators, multiplier = 1)

    startConf = makeConf(t.world.robot, 0.0, 0.0, 0.0)
    result, cnfOut, _ = pr2GoToConf(gripOpen(startConf, hand), 'open')    

def testIt():
    import gjk
    fizz = shapes.Box(0.1,0.2,0.3, None)
    print fizz.parts()[0].basePrim.baseRings
    fuzz = shapes.Box(0.1,0.2,0.3, None)
    win = wm.getWindow('W')
    for x in range(100):
        f = fuzz.applyTrans(hu.Pose(x*0.005, 0., 0., 0.))
        print f.origin().matrix
        print x*0.005, gjk.gjkDist(fizz, f)**0.5
        win.clear()
        fizz.draw('W'); f.draw('W', 'red')
        raw_input('Ok')



