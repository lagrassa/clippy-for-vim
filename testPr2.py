import pdb
from pr2ROS import testReactive
import testRig
reload(testRig)
from testRig import *

from timeout import timeout, TimeoutError

# 10 min timeout for all tests
@timeout(600)
def testFunc(n, skeleton=None, heuristic=habbs, hierarchical=True, easy=False, rip=True):
    eval('test%s(skeleton=skeleton, heuristic=heuristic, hierarchical=hierarchical, easy=easy, rip=rip)'%str(n))

def testRepeat(n, repeat=3, **args):
    for i in range(repeat):
        try:
            testFunc(n, **args)
        except TimeoutError:
            print '************** Timed out **************'

# defined in testRig.
#testResults = {}

def testAll(indices, repeat=3, crashIsError=True, **args):
    pr2Sim.crashIsError = crashIsError
    for i in indices:
        if i == 0: continue
        testRepeat(i, repeat=repeat, **args)
    print testResults

######################################################################
# Test 0: 1 table move 1 object
######################################################################

# 1 table; move 1 object
# Assumption is that odometry error is kept in check during motions.
# Use domainProbs.odoError as the stdev of any object.

def test0(hpn = True, skeleton = False, hierarchical = False, heuristic=habbs,
          easy = False, rip = False):

    glob.rebindPenalty = 100
    glob.monotonicFirst = True

    goalProb, errProbs = (0.5,tinyErrProbs) if easy else (0.95,typicalErrProbs)

    varDict = {} if easy else {'table1': (0.07**2, 0.03**2, 1e-10, 0.2**2),
                               'objA': (0.1**2, 0.1**2, 1e-10, 0.3**2)} 
    front = hu.Pose(1.1, 0.0, tZ, 0.0)
    table1Pose = hu.Pose(1.3, 0.0, 0.0, math.pi/2)

    region = 'table1Left'
    goal = State([Bd([In(['objA', region]), True, goalProb], True)])

    t = PlanTest('test0',  errProbs, allOperators,
                 objects=['table1', 'objA'],
                 fixPoses={'table1': table1Pose},
                 movePoses={'objA': front},
                 varDict = varDict)

    actualSkel = None

    t.run(goal,
          hpn = hpn,
          skeleton = actualSkel if skeleton else None,
          hierarchical = hierarchical,
          regions=[region],
          heuristic = heuristic,
          rip = rip
          )
    return t

######################################################################
# Test 1: 2 tables move 1 object
######################################################################

# pick and place into region.  2 tables
def test1(hpn = True, skeleton = False, hierarchical = False, heuristic=habbs,
          easy = False, rip = False, multiplier=6):

    glob.rebindPenalty = 700
    glob.monotonicFirst = True

    goalProb, errProbs = (0.5,tinyErrProbs) if easy else (0.95,typicalErrProbs)

    varDict = {} if easy else {'table1': (0.07**2, 0.03**2, 1e-10, 0.2**2),
                               'table2': (0.07**2, 0.03**2, 1e-10, 0.2**2),
                               'objA': (0.1**2, 0.1**2, 1e-10, 0.3**2)} 
    front = hu.Pose(1.1, 0.0, tZ, 0.0)
    table2Pose = hu.Pose(1.0, -1.00, 0.0, 0.0)
    table1Pose = hu.Pose(1.3, 0.0, 0.0, math.pi/2)

    region = 'table2Left'
    goal = State([Bd([In(['objA', region]), True, goalProb], True)])

    t = PlanTest('test1',  errProbs, allOperators,
                 objects=['table1', 'objA', 'table2'],
                 fixPoses={'table1': table1Pose,
                           'table2': table2Pose},
                 movePoses={'objA': front},
                 varDict = varDict,
                 multiplier = multiplier)

    t.run(goal,
          hpn = hpn,
          hierarchical = hierarchical,
          regions=[region],
          heuristic = heuristic,
          rip = rip
          )
    return t

######################################################################
# Test 4: one table, move two objects
######################################################################

# pick and place into region... one table, for robot.
def test4(hpn = True, skeleton = False, hierarchical = False, heuristic=habbs,
          easy = False, rip = False):

    glob.rebindPenalty = 700
    glob.monotonicFirst = True

    goalProb, errProbs = (0.5, tinyErrProbs) if easy else (0.95,typicalErrProbs)

    varDict = {} if easy else {'table1': (0.07**2, 0.03**2, 1e-10, 0.2**2),
                               'objA': (0.1**2, 0.1**2, 1e-10, 0.3**2),
                               'objB': (0.1**2, 0.1**2, 1e-10, 0.3**2)} 
    front = hu.Pose(1.1, 0.0, tZ, 0.0)
    right = hu.Pose(1.1, -0.4, tZ, 0.0)
    table1Pose = hu.Pose(1.3, 0.0, 0.0, math.pi/2)

    region = 'table1Left'
    goal = State([Bd([In(['objA', region]), True, goalProb], True),
                  Bd([In(['objB', region]), True, goalProb], True)])

    t = PlanTest('test4',  errProbs, allOperators,
                 objects=['table1', 'objA', 'objB'],
                 fixPoses={'table1': table1Pose},
                 movePoses={'objA': right, 'objB':front},
                 varDict = varDict)

    t.run(goal,
          hpn = hpn,
          hierarchical = hierarchical,
          regions=[region],
          heuristic = heuristic,
          rip = rip
          )
    return t

def testShelves(hpn = True, skeleton = False, hierarchical = False, heuristic=habbs,
                easy = False, rip = False, multiplier=6):

    glob.rebindPenalty = 700
    glob.monotonicFirst = True

    goalProb, errProbs = (0.5, tinyErrProbs) if easy else (0.95,typicalErrProbs)

    varDict = {} if easy else {'coolShelves': (0.07**2, 0.03**2, 1e-10, 0.2**2),
                               'table1': (0.07**2, 0.03**2, 1e-10, 0.2**2),
                               'objA': (0.1**2, 0.1**2, 1e-10, 0.3**2),
                               'objB': (0.1**2, 0.1**2, 1e-10, 0.3**2)
                               } 
    right1 = hu.Pose(1.1, -0.5, tZ, 0.0)
    right2 = hu.Pose(1.5, -0.5, tZ, 0.0)
    left1 = hu.Pose(1.1, 0.5, tZ, 0.0)
    left2 = hu.Pose(1.5, 0.5, tZ, 0.0)
    coolShelvesPose = hu.Pose(1.3, 0.0, tZ, math.pi/2)
    table1Pose = hu.Pose(1.3, 0.0, 0.0, math.pi/2)
    
    region = 'coolShelves_space_2'
    goal1 = State([Bd([In(['objA', region]), True, goalProb], True)])
    goal2 = State([Bd([In(['objB', region]), True, goalProb], True),
                  Bd([In(['objA', region]), True, goalProb], True)])

    t = PlanTest('testShelves',  errProbs, allOperators,
                 objects=['table1', 'coolShelves', 'objA',
                          # 'objB'
                          ],
                 fixPoses={'table1' : table1Pose, 'coolShelves': coolShelvesPose},
                 movePoses={'objA': right1,
                            # 'objB': left1
                            },
                 varDict = varDict,
                 multiplier = multiplier)

    t.run(goal1,
          hpn = hpn,
          hierarchical = hierarchical,
          regions=[region],
          heuristic = heuristic,
          rip = rip,
          )
    return t

def testPick(hpn = True, skeleton = False, hierarchical = False, heuristic=habbs,
          easy = False, rip = False, multiplier=6):

    glob.rebindPenalty = 700
    glob.monotonicFirst = True

    goalProb, errProbs = (0.5,tinyErrProbs) if easy else (0.95,typicalErrProbs)

    varDict = {} if easy else {'table1': (0.07**2, 0.03**2, 1e-10, 0.2**2),
                               'objA': (0.1**2, 0.1**2, 1e-10, 0.3**2)} 
    front = hu.Pose(1.1, 0.0, tZ, 0.0)
    table1Pose = hu.Pose(1.3, 0.0, 0.0, math.pi/2)

    hand = 'right'
    graspType = 0
    targetDelta = (0.01, 0.01, 0.01, 0.05)
    goal = State([Bd([Holding([hand]), 'objA', goalProb], True),
                   Bd([GraspFace(['objA', hand]), graspType, goalProb], True),
                   B([Grasp(['objA', hand,  graspType]),
                     (0,-0.025,0,0), (0.01, 0.01, 0.01, 0.01), targetDelta,
                     goalProb], True)])

    t = PlanTest('test1',  errProbs, allOperators,
                 objects=['table1', 'objA'],
                 fixPoses={'table1': table1Pose},
                 movePoses={'objA': front},
                 varDict = varDict,
                 multiplier = multiplier)

    t.run(goal,
          hpn = hpn,
          hierarchical = hierarchical,
          heuristic = heuristic,
          rip = rip
          )
    return t


######################################################################
# Test Put Down
#       Start with something in the hand
######################################################################

def makeAttachedWorldFromPBS(pbs, realWorld, grasped, hand):
    attachedShape = pbs.getRobot().\
                    attachedObj(pbs.getShadowWorld(0.9), hand)
    shape = pbs.getWorld().\
            getObjectShapeAtOrigin(grasped).applyLoc(attachedShape.origin())
    realWorld.robot.attach(shape, realWorld, hand)
    robot = pbs.getRobot()
    cart = realWorld.robotConf.cartConf()
    handPose = cart[robot.armChainNames[hand]].compose(robot.toolOffsetX[hand])
    pose = shape.origin()
    realWorld.held[hand] = grasped
    realWorld.grasp[hand] = handPose.inverse().compose(pose)
    realWorld.delObjectState(grasped)
    realWorld.setRobotConf(realWorld.robotConf) # to compute a new robotPlace

def testPutDown(hpn = True, skeleton = False, hierarchical = False,
                heuristic = habbs, easy = False, rip = False):

    # Seems to need this
    global useRight, useVertical
    useRight, useVertical = True, True

    glob.rebindPenalty = 150
    goalProb, errProbs = (0.4, tinyErrProbs) if easy else (0.95,typicalErrProbs)
    glob.monotonicFirst = True
    
    front = hu.Pose(0.95, 0.0, tZ, 0.0)
    back = hu.Pose(1.25, 0.0, tZ, 0.0)

    varDict = {} if easy else {'table1': (0.07**2, 0.03**2, 1e-10, 0.2**2),
                               'table2': (0.07**2, 0.03**2, 1e-10, 0.2**2),
                               'objA': (0.05**2,0.05**2, 1e-10,0.2**2),
                               'objB': (0.05**2,0.05**2, 1e-10,0.2**2)}

    t = PlanTest('testPutDown',  errProbs, allOperators,
                 objects=['table1', 'objA','objB'], 
                 movePoses={'objA': back,
                            'objB': front},
                    #fixPoses={'table2': table2Pose},
                 varDict = varDict)

    targetVar = (0.0001, 0.0001, 0.0001, 0.0005)
    targetDelta = (0.01, 0.01, 0.01, 0.05)
    # Just empty the hand.  No problem.
    goal1 = State([Bd([Holding(['left']), 'none', goalProb], True)])
    # Pick obj A
    graspType = 2
    goal2 = State([Bd([Holding(['left']), 'objA', goalProb], True),
                   Bd([GraspFace(['objA', 'left']), graspType, goalProb], True),
                   B([Grasp(['objA', 'left',  graspType]),
                     (0,-0.025,0,0), (0.01, 0.01, 0.01, 0.01), targetDelta,
                     goalProb], True)])
    # Put A somewhere.  Ideally use right hand!
    goal3 = State([Bd([SupportFace(['objA']), 4, goalProb], True),
                  B([Pose(['objA', 4]),
                     front.xyztTuple(), targetVar, targetDelta,
                     goalProb], True)])

    grasped = 'objB'
    hand = 'left'
    initGraspVar = (1e-6,)*4    # very small
    def initBel(bs):
        # Change pbs so obj B is in the left hand
        gm = (0, -0.025, 0, 0)
        gv = initGraspVar
        gd = (1e-4,)*4
        gf = 0
        bs.pbs.updateHeld(grasped, gf, PoseD(gm, gv), hand, gd)
        bs.pbs.excludeObjs([grasped])
        bs.pbs.reset()

    def initWorld(bs, realWorld):
        makeAttachedWorldFromPBS(bs.pbs, realWorld, grasped, hand)

    t.run(goal3,
          hpn = hpn,
          hierarchical = hierarchical,
          heuristic = heuristic,
          regions = ['table1Top'],
          initBelief = initBel,
          initWorld = initWorld
          )

def testChangeGrasp(hpn = True, skeleton = False, hierarchical = False,
                heuristic = habbs, easy = False, rip = False):

    # Seems to need this
    global useRight, useVertical
    useRight, useVertical = True, True

    glob.rebindPenalty = 150
    goalProb, errProbs = (0.4, tinyErrProbs) if easy else (0.95,typicalErrProbs)
    glob.monotonicFirst = True
    
    front = hu.Pose(0.95, 0.0, tZ, 0.0)
    back = hu.Pose(1.25, 0.0, tZ, 0.0)

    varDict = {} if easy else {'table1': (0.07**2, 0.03**2, 1e-10, 0.2**2),
                               'table2': (0.07**2, 0.03**2, 1e-10, 0.2**2),
                               'objA': (0.05**2,0.05**2, 1e-10,0.2**2),
                               'objB': (0.05**2,0.05**2, 1e-10,0.2**2)}

    t = PlanTest('testChangeGrasp',  errProbs, allOperators,
                 objects=['table1', 'objA','objB'], 
                 varDict = varDict)

    targetDelta = (0.01, 0.01, 0.01, 0.05)

    # Pick obj B
    graspType = 0
    goalProb = 0.7
    goal = State([Bd([Holding(['left']), 'objB', goalProb], True),
                  Bd([GraspFace(['objB', 'left']), graspType, goalProb], True),
                  B([Grasp(['objB', 'left',  graspType]),
                     (0,-0.025,0,0), (0.01**2, 0.01**2, 0.01**2, 0.02**2), targetDelta,
                     goalProb], True)])

    grasped = 'objB'
    hand = 'left'
    initGraspVar = (1e-6,)*4    # very small
    def initBel(bs):
        # Change pbs so obj B is in the left hand
        gm = (0, -0.025, 0, 0)
        gv = (0.011**2, 0.011**2, 0.011**2, 0.021**2)  # initGraspVar
        gd = (1e-4,)*4
        gf = 0
        bs.pbs.updateHeld(grasped, gf, PoseD(gm, gv), hand, gd)
        bs.pbs.excludeObjs([grasped])
        bs.pbs.reset()

    def initWorld(bs, realWorld):
        makeAttachedWorldFromPBS(bs.pbs, realWorld, grasped, hand)

    t.run(goal,
          hpn = hpn,
          hierarchical = hierarchical,
          heuristic = heuristic,
          regions = ['table1Top'],
          initBelief = initBel,
          initWorld = initWorld
          )


######################################################################
#       Another test.  Picking something up from the back.
#       shouldn't be hard.
######################################################################

def test5(hpn = True, skeleton = False, hierarchical = False,
                heuristic = habbs, easy = False, rip = False):

    # Seems to need this
    global useRight, useVertical
    useRight, useVertical = True, True

    glob.rebindPenalty = 150
    goalProb, errProbs = (0.4, tinyErrProbs) if easy else (0.95,typicalErrProbs)
    glob.monotonicFirst = True
    
    front = hu.Pose(0.95, 0.0, tZ, 0.0)
    back = hu.Pose(1.25, 0.0, tZ, 0.0)

    varDict = {} if easy else {'table1': (0.07**2, 0.03**2, 1e-10, 0.2**2),
                               'table2': (0.07**2, 0.03**2, 1e-10, 0.2**2),
                               'objA': (0.05**2,0.05**2, 1e-10,0.2**2),
                               'objB': (0.05**2,0.05**2, 1e-10,0.2**2)}

    t = PlanTest('test5',  errProbs, allOperators,
                 objects=['table1', 'objA','objB'], 
                 movePoses={'objA': back,
                            'objB': front},
                 varDict = varDict)

    targetVar = (0.0001, 0.0001, 0.0001, 0.0005)
    targetDelta = (0.01, 0.01, 0.01, 0.05)

    # Pick obj A
    graspType = 2
    goal2 = State([Bd([Holding(['left']), 'objA', goalProb], True),
                  Bd([GraspFace(['objA', 'left']), graspType, goalProb], True),
                  B([Grasp(['objA', 'left',  graspType]),
                     (0,-0.025,0,0), (0.01, 0.01, 0.01, 0.01), targetDelta,
                     goalProb], True)])
    # Put A somewhere.  
    goal3 = State([Bd([SupportFace(['objA']), 4, goalProb], True),
                  B([Pose(['objA', 4]),
                     front.xyztTuple(), targetVar, targetDelta,
                     goalProb], True)])

    t.run(goal2,
          hpn = hpn,
          hierarchical = hierarchical,
          heuristic = heuristic,
          regions = ['table1Top'])

######################################################################
# Test Swap:
#     with goal3, it just puts object b in back
#     with goal, it does whole swap
######################################################################


#  Swap!
def testSwap(hpn = True, skeleton = False, hierarchical = False,
           heuristic = habbs, easy = False, rip = False,
           hardSwap = False):


    # Seems to need this
    global useRight, useVertical
    useRight, useVertical = True, True

    glob.rebindPenalty = 150
    goalProb, errProbs = (0.4, tinyErrProbs) if easy else (0.95,typicalErrProbs)
    glob.monotonicFirst = True
    table2Pose = hu.Pose(1.0, -1.2, 0.0, 0.0)
    
    front = hu.Pose(0.95, 0.0, tZ, 0.0)
    # Put this back to make the problem harder
    #back = hu.Pose(1.1, 0.0, tZ, 0.0)
    back = hu.Pose(1.25, 0.0, tZ, 0.0)
    parking1 = hu.Pose(0.95, 0.3, tZ, 0.0)
    parking2 = hu.Pose(0.95, -0.3, tZ, 0.0)

    varDict = {} if easy else {'table1': (0.07**2, 0.03**2, 1e-10, 0.2**2),
                               'table2': (0.07**2, 0.03**2, 1e-10, 0.2**2),
                               'objA': (0.05**2,0.05**2, 1e-10,0.2**2),
                               'objB': (0.05**2,0.05**2, 1e-10,0.2**2)}


    t = PlanTest('testSwap',  errProbs, allOperators,
                 objects=['table1', 'table2', 'objA',
                          'objB'], #,'cupboardSide1', 'cupboardSide2'],
                 movePoses={'objA': back,
                            'objB': front},
                 fixPoses={'table2': table2Pose},
                 varDict = varDict)

    goal = State([Bd([In(['objB', 'table1MidRear']), True, goalProb], True),
                  Bd([In(['objA', 'table1MidFront']), True, goalProb], True)])

    # A on other table
    goal1 = State([Bd([In(['objA', 'table2Top']), True, goalProb], True)])
    skel1 = [[poseAchIn, 
              place, moveNB, lookAt, move,
              pick, moveNB, lookAt, moveNB, lookAt, move, lookAt, moveNB]]

    # A and B on other table
    goal2 = State([Bd([In(['objB', 'table2Top']), True, goalProb], True),
                   Bd([In(['objA', 'table2Top']), True, goalProb], True)])

    # B in back
    goal3 = State([Bd([In(['objB', 'table1MidRear']), True, goalProb], True)])

    actualGoal = goal if hardSwap else goal3

    t.run(actualGoal,
          hpn = hpn,
          heuristic = heuristic,
          hierarchical = hierarchical,
          rip = rip,
          regions=['table1Top', 'table2Top', 'table1MidFront',
                   'table1MidRear']
          )

def testHold(hpn = True, skeleton = False, hierarchical = False,
           heuristic = habbs, easy = False, rip = False):

    glob.rebindPenalty = 150
    goalProb, errProbs = (0.4, tinyErrProbs) if easy else (0.95,typicalErrProbs)
    glob.monotonicFirst = True
    table2Pose = hu.Pose(1.0, -1.20, 0.0, 0.0)
    
    front = hu.Pose(0.95, 0.0, tZ, 0.0)
    # Put this back to make the problem harder
    #back = hu.Pose(1.1, 0.0, tZ, 0.0)
    back = hu.Pose(1.25, 0.0, tZ, 0.0)

    easyVarDict = {'table1': (0.001**2, 0.001**2, 1e-10, 0.001**2),
                               'table2': (0.001**2, 0.001**2, 1e-10, 0.001**2),
                               'objA': (0.0001**2,0.0001**2, 1e-10,0.001**2),
                               'objB': (0.0001**2,0.0001**2, 1e-10,0.001**2)}

    varDict = easyVarDict if easy else \
                            {'table1': (0.07**2, 0.03**2, 1e-10, 0.2**2),
                               'table2': (0.07**2, 0.03**2, 1e-10, 0.2**2),
                               'objA': (0.05**2,0.05**2, 1e-10,0.2**2),
                               'objB': (0.05**2,0.05**2, 1e-10,0.2**2)}

    t = PlanTest('testHold',  errProbs, allOperators,
                 objects=['table1', 'table2', 'objA', 'objB'],
#                          'cupboardSide1', 'cupboardSide2'],
                 movePoses={'objA': back,
                            'objB': front},
                 fixPoses={'table2': table2Pose},
                 varDict = varDict)

    obj = 'objA'
    hand = 'left'
    grasp = 3
    delta = (0.01,)*4

    goal = State([Bd([Holding([hand]), obj, goalProb], True),
                  Bd([GraspFace([obj, hand]), grasp, goalProb], True),
                  B([Grasp([obj, hand,  grasp]),
                     (0,-0.025,0,0), (0.01, 0.01, 0.01, 0.01), delta,
                     goalProb], True)])

    t.run(goal,
          hpn = hpn,
          heuristic = heuristic,
          hierarchical = hierarchical,
          rip = rip,
          regions=['table1Top']
          )


def testSim():
    varDict =  {'table1': (0.07**2, 0.03**2, 1e-10, 0.2**2),
                'objA': (0.1**2, 0.1**2, 1e-10, 0.3**2),
                'objB': (0.1**2, 0.1**2, 1e-10, 0.3**2)} 
    front = hu.Pose(1.1, 0.0, tZ, 0.0)
    right = hu.Pose(1.1, -0.4, tZ, 0.0)
    table1Pose = hu.Pose(1.3, 0.0, 0.0, math.pi/2)

    t = PlanTest('testSim', typicalErrProbs, allOperators,
                 objects=['table1', 'objA', 'objB'],
                 fixPoses={'table1': table1Pose},
                 movePoses={'objA': right, 'objB':front},
                 varDict = varDict,
                 multiplier=1)

    t.run(None)

    return t

def prof(test, n=100):
    import cProfile
    import pstats
    cProfile.run(test, 'prof')
    p = pstats.Stats('prof')
    p.sort_stats('cumulative').print_stats(n)
    p.sort_stats('cumulative').print_callers(n)

def profPrint(n=100):
    import pstats
    p = pstats.Stats('prof')
    p.sort_stats('cumulative').print_stats(n)
    p.sort_stats('cumulative').print_callers(n)


# Evaluate on details and a fluent to flush the caches and evaluate
def firstAid(details, fluent = None):
    glob.debugOn.extend(['confReachViol', 'confViolations'])

    pbs = details.pbs
    bc = pbs.beliefContext

    pbs.getRoadMap().confReachCache.clear()
    bc.pathObstCache.clear()
    bc.objectShadowCache.clear()
    for c in bc.genCaches.values():
        c.clear()
    pr2GenAux.graspConfGenCache.clear()
    bc.world.robot.cacheReset()
    pr2Visible.cache.clear()
    belief.hCacheReset()
    
    if fluent:
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

def testBusy(hpn = True, skeleton = False, hierarchical = False,
           heuristic = habbs, easy = False, rip = False,
           hardSwap = False):


    # Seems to need this
    global useRight, useVertical
    useRight, useVertical = True, True

    glob.rebindPenalty = 150
    goalProb, errProbs = (0.4, tinyErrProbs) if easy else (0.95,typicalErrProbs)
    glob.monotonicFirst = True
    table2Pose = hu.Pose(1.0, -1.2, 0.0, 0.0)
    
    front = hu.Pose(0.95, 0.0, tZ, 0.0)
    back = hu.Pose(1.25, 0.0, tZ, 0.0)

    varDict = {} if easy else {'table1': (0.07**2, 0.03**2, 1e-10, 0.2**2),
                               'table2': (0.07**2, 0.03**2, 1e-10, 0.2**2),
                               'objA': (0.05**2,0.05**2, 1e-10,0.2**2),
                               'objB': (0.05**2,0.05**2, 1e-10,0.2**2),
                               'objC': (0.05**2,0.05**2, 1e-10,0.2**2),
                               'objD': (0.05**2,0.05**2, 1e-10,0.2**2),
                               'objE': (0.05**2,0.05**2, 1e-10,0.2**2),
                               'objF': (0.05**2,0.05**2, 1e-10,0.2**2),
                               'objG': (0.05**2,0.05**2, 1e-10,0.2**2)}


    t = PlanTest('testSwap',  errProbs, allOperators,
                 objects=['table1', 'table2', 'objA',
                          'objB', 'objC', 'objD', 'objE', 'objF', 'objG'], 
                 movePoses={'objA': back},
                 fixPoses={'table2': table2Pose},
                 varDict = varDict)

    goal = State([Bd([In(['objB', 'table1MidRear']), True, goalProb], True),
                  Bd([In(['objA', 'table1MidFront']), True, goalProb], True)])

    # A on other table
    goal1 = State([Bd([In(['objA', 'table2Top']), True, goalProb], True)])

    # A and B on other table
    goal2 = State([Bd([In(['objB', 'table2Top']), True, goalProb], True),
                   Bd([In(['objA', 'table2Top']), True, goalProb], True)])

    # B in back
    goal3 = State([Bd([In(['objB', 'table1MidRear']), True, goalProb], True)])

    actualGoal = goal if hardSwap else goal3

    t.run(actualGoal,
          hpn = hpn,
          heuristic = heuristic,
          hierarchical = hierarchical,
          rip = rip,
          regions=['table1Top', 'table2Top', 'table1MidFront',
                   'table1MidRear']
          )

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
    
