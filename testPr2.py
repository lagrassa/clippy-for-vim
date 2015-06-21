from itertools import *
import testRig
reload(testRig)
from testRig import *

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

    goalProb, errProbs = (0.5,smallErrProbs) if easy else (0.95,typicalErrProbs)

    varDict = {} if easy else {'table1': (0.07**2, 0.03**2, 1e-10, 0.2**2),
                               'objA': (0.1**2, 0.1**2, 1e-10, 0.3**2)} 
    front = util.Pose(1.1, 0.0, tZ, 0.0)
    table1Pose = util.Pose(1.3, 0.0, 0.0, math.pi/2)

    region = 'table1Left'
    goal = State([Bd([In(['objA', region]), True, goalProb], True)])

    t = PlanTest('test0',  errProbs, allOperators,
                 objects=['table1', 'objA'],
                 fixPoses={'table1': table1Pose},
                 movePoses={'objA': front},
                 varDict = varDict)

    hskel = [[poseAchIn],
             [poseAchIn, place.applyBindings({'Obj' : 'objA'}),
              lookAt.applyBindings({'Obj' : 'table1'})], #1
             [lookAt.applyBindings({'Obj' : 'table1'}), moveNB], #2
             [place.applyBindings({'Obj' : 'objA'})], #3
             [lookAt.applyBindings({'Obj' : 'objA'})], #4
             [lookAt.applyBindings({'Obj' : 'objA'}), moveNB], #5
             [poseAchIn], #6
             [poseAchIn, lookAt.applyBindings({'Obj' : 'objA'}),
              place.applyBindings({'Obj' : 'objA'})], #7
             [place.applyBindings({'Obj' : 'objA'}),
              pick.applyBindings({'Obj' : 'objA'})], #8
             [pick.applyBindings({'Obj' : 'objA'})], #9
             [pick.applyBindings({'Obj' : 'objA'})], #10
             [pick.applyBindings({'Obj' : 'objA'}), move], #11
             [moveNB, lookAt.applyBindings({'Obj' : 'objA'}), move], #12
             [move]
            ]

    skel = [[poseAchIn,
             lookAt.applyBindings({'Obj' : 'objA'}), moveNB,
             lookAt.applyBindings({'Obj' : 'table1'}), moveNB,
             lookAt.applyBindings({'Obj' : 'table1'}), move,
             place.applyBindings({'Obj': 'objA'}),
             poseAchCanPickPlace,
             moveNB,
             lookAt.applyBindings({'Obj' : 'table1'}), 
             move,      
             pick,
             #poseAchCanReach,
             moveNB,
             lookAt.applyBindings({'Obj' : 'table1'}), moveNB, 
             lookAt.applyBindings({'Obj' : 'table1'}), moveNB, 
             lookAt.applyBindings({'Obj' : 'table1'}), moveNB, 
             lookAt.applyBindings({'Obj' : 'objA'}), moveNB,
             lookAt.applyBindings({'Obj' : 'objA'}), moveNB,
             lookAt.applyBindings({'Obj' : 'objA'}),
             move,
             lookAt.applyBindings({'Obj' : 'table1'}), moveNB,
             lookAt.applyBindings({'Obj' : 'objA'}), moveNB]]

    if not easy:
        actualSkel = hskel if hierarchical else skel
    else:
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
          easy = False, rip = False, multiplier=8):

    glob.rebindPenalty = 700
    glob.monotonicFirst = True

    goalProb, errProbs = (0.5,smallErrProbs) if easy else (0.95,typicalErrProbs)

    varDict = {} if easy else {'table1': (0.07**2, 0.03**2, 1e-10, 0.2**2),
                               'table2': (0.07**2, 0.03**2, 1e-10, 0.2**2),
                               'objA': (0.1**2, 0.1**2, 1e-10, 0.3**2)} 
    front = util.Pose(1.1, 0.0, tZ, 0.0)
    table2Pose = util.Pose(1.0, -1.00, 0.0, 0.0)
    table1Pose = util.Pose(1.3, 0.0, 0.0, math.pi/2)

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
          skeleton = skel if skeleton else None,
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

    goalProb, errProbs = (0.5,smallErrProbs) if easy else (0.95,typicalErrProbs)

    varDict = {} if easy else {'table1': (0.07**2, 0.03**2, 1e-10, 0.2**2),
                               'objA': (0.1**2, 0.1**2, 1e-10, 0.3**2),
                               'objB': (0.1**2, 0.1**2, 1e-10, 0.3**2)} 
    front = util.Pose(1.1, 0.0, tZ, 0.0)
    right = util.Pose(1.1, -0.4, tZ, 0.0)
    table1Pose = util.Pose(1.3, 0.0, 0.0, math.pi/2)

    region = 'table1Left'
    goal = State([Bd([In(['objA', region]), True, goalProb], True),
                  Bd([In(['objB', region]), True, goalProb], True)])

    t = PlanTest('test4',  errProbs, allOperators,
                 objects=['table1', 'objA', 'objB'],
                 fixPoses={'table1': table1Pose},
                 movePoses={'objA': right, 'objB':front},
                 varDict = varDict)

    skel = [[poseAchIn.applyBindings({'Obj1' : 'objA'}),
             poseAchIn.applyBindings({'Obj1' : 'objB'}),
             lookAt, lookAt, lookAt, lookAt]]

    t.run(goal,
          hpn = hpn,
          skeleton = skel if skeleton else None,
          hierarchical = hierarchical,
          regions=[region],
          heuristic = heuristic,
          rip = rip
          )
    return t

def testShelves(hpn = True, skeleton = False, hierarchical = False, heuristic=habbs,
                easy = False, rip = False):

    glob.rebindPenalty = 700
    glob.monotonicFirst = True

    goalProb, errProbs = (0.5,smallErrProbs) if easy else (0.95,typicalErrProbs)

    varDict = {} if easy else {'tableShelves': (0.07**2, 0.03**2, 1e-10, 0.2**2),
                               'objA': (0.1**2, 0.1**2, 1e-10, 0.3**2),
                               'objB': (0.1**2, 0.1**2, 1e-10, 0.3**2)
                               } 
    right1 = util.Pose(1.1, -0.5, tZ, 0.0)
    right2 = util.Pose(1.5, -0.5, tZ, 0.0)
    left1 = util.Pose(1.1, 0.5, tZ, 0.0)
    left2 = util.Pose(1.5, 0.5, tZ, 0.0)
    tableShelvesPose = util.Pose(1.3, 0.0, 0.0, math.pi/2)

    region = 'tableShelves_space_2'
    goal1 = State([Bd([In(['objA', region]), True, goalProb], True)])
    goal2 = State([Bd([In(['objB', region]), True, goalProb], True),
                  Bd([In(['objA', region]), True, goalProb], True)])

    t = PlanTest('testShelves',  errProbs, allOperators,
                 objects=['tableShelves', 'objA',
                          # 'objB'
                          ],
                 fixPoses={'tableShelves': tableShelvesPose},
                 movePoses={'objA': right1,
                            # 'objB': left1
                            },
                 varDict = varDict)

    t.run(goal1,
          hpn = hpn,
          hierarchical = hierarchical,
          regions=[region],
          heuristic = heuristic,
          rip = rip,
          )
    return t

def testPick(hpn = True, skeleton = False, hierarchical = False, heuristic=habbs,
          easy = False, rip = False, multiplier=8):

    glob.rebindPenalty = 700
    glob.monotonicFirst = True

    goalProb, errProbs = (0.5,smallErrProbs) if easy else (0.95,typicalErrProbs)

    varDict = {} if easy else {'table1': (0.07**2, 0.03**2, 1e-10, 0.2**2),
                               'objA': (0.1**2, 0.1**2, 1e-10, 0.3**2)} 
    front = util.Pose(1.1, 0.0, tZ, 0.0)
    table1Pose = util.Pose(1.3, 0.0, 0.0, math.pi/2)

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
          skeleton = skel if skeleton else None,
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
    
    front = util.Pose(0.95, 0.0, tZ, 0.0)
    back = util.Pose(1.25, 0.0, tZ, 0.0)

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

    easySkel2 = [[pick, move,
                  place, move, 
                  lookAt, move,
                  lookAt, move]]

    easyHSkel2 = [[pick],
                  [pick, lookAt],
                  [lookAt, moveNB],
                  [pick],
                  [pick, move, place]]

    hardSkel2 = [[pick, moveNB,
                  poseAchCanPickPlace,
                  lookAt, moveNB,
                  lookAt, move,
                  place, moveNB,
                  lookAt, move]]

    easySkel = easyHSkel2 if hierarchical else easySkel2

    skel = easySkel if easy else hardSkel2
                   
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
          skeleton = easySkel if skeleton else None,
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
    
    front = util.Pose(0.95, 0.0, tZ, 0.0)
    back = util.Pose(1.25, 0.0, tZ, 0.0)

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
          skeleton = easySkel if skeleton else None,
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
    
    front = util.Pose(0.95, 0.0, tZ, 0.0)
    back = util.Pose(1.25, 0.0, tZ, 0.0)

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

    easySkel2 = [[pick, moveNB, poseAchCanPickPlace, lookAt, move]]

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
          skeleton = easySkel2 if skeleton else None,
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
    table2Pose = util.Pose(1.0, -1.2, 0.0, 0.0)
    
    front = util.Pose(0.95, 0.0, tZ, 0.0)
    # Put this back to make the problem harder
    #back = util.Pose(1.1, 0.0, tZ, 0.0)
    back = util.Pose(1.25, 0.0, tZ, 0.0)
    parking1 = util.Pose(0.95, 0.3, tZ, 0.0)
    parking2 = util.Pose(0.95, -0.3, tZ, 0.0)

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

    skel3h = [[poseAchIn],
              [poseAchIn, bLoc1.applyBindings({'Obj' : 'table1'}),
               lookAt, bLoc1.applyBindings({'Obj' : 'objB'}),
               lookAt],
             [lookAt.applyBindings({'Obj' : 'objB'}), moveNB],
             [poseAchIn],
             [poseAchIn, lookAt.applyBindings({'Obj' : 'objB'}), place],
             [place, move, pick]]

    hardSkel = [[poseAchIn.applyBindings({'Obj' : 'objB'}), poseAchIn],
                [poseAchIn.applyBindings({'Obj' : 'objA'}),
                 bLoc1.applyBindings({'Obj' : 'table1'}),
                 lookAt.applyBindings({'Obj' : 'table1'}),
                 bLoc1.applyBindings({'Obj' : 'objA'}),
                 lookAt.applyBindings({'Obj' : 'objA'})],
                [lookAt.applyBindings({'Obj' : 'objA'}), #2
                 moveNB],
                [poseAchIn.applyBindings({'Obj' : 'objA'})], #3
                [poseAchIn.applyBindings({'Obj' : 'objA'}),  #4
                 lookAt.applyBindings({'Obj' : 'objA'}),
                 place.applyBindings({'Obj' : 'objA'})],
                [place.applyBindings({'Obj' : 'objA'}),   #5
                 poseAchCanPickPlace,
                 lookAt.applyBindings({'Obj' : 'objB'}),
                 place.applyBindings({'Obj' : 'objB'})],
                [place.applyBindings({'Obj' : 'objB'}),    #6
                 poseAchCanPickPlace,
                 lookAt.applyBindings({'Obj' : 'objA'}),
                 place.applyBindings({'Obj' : 'objA'}),
                 pick.applyBindings({'Obj' : 'objA'})],
                [pick.applyBindings({'Obj' : 'objA'})], #7
                [pick.applyBindings({'Obj' : 'objA'})], #8
                [pick.applyBindings({'Obj' : 'objA'}),  #9
                 move],
                [moveNB,
                 lookAt.applyBindings({'Obj' : 'objA'}),
                 move],
                [place.applyBindings({'Obj' : 'objB'}), #11
                 pick.applyBindings({'Obj' : 'objB'})],
                [pick.applyBindings({'Obj' : 'objB'}),
                 place.applyBindings({'Obj' : 'objA'}),
                 move],
                [move],
                [pick.applyBindings({'Obj' : 'objB'})],
                [pick.applyBindings({'Obj' : 'objB'}),
                 moveNB,
                 lookAt.applyBindings({'Obj' : 'objB'}),
                 move,
                 lookAt.applyBindings({'Obj' : 'table1'}),
                 moveNB]]
                 

    t.run(actualGoal,
          hpn = hpn,
          skeleton = hardSkel if skeleton else None,
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
    table2Pose = util.Pose(1.0, -1.20, 0.0, 0.0)
    
    front = util.Pose(0.95, 0.0, tZ, 0.0)
    # Put this back to make the problem harder
    #back = util.Pose(1.1, 0.0, tZ, 0.0)
    back = util.Pose(1.25, 0.0, tZ, 0.0)

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


    skel2 = [[pick, moveNB, poseAchCanPickPlace,
              lookAt, moveNB, lookAt,
              move, lookAt, moveNB, lookAt, moveNB]]

    skel3 = [[pick, moveNB, poseAchCanPickPlace,
              lookAt.applyBindings({'Obj' : 'table1'}),
              move, lookAt, moveNB, lookAt, moveNB]]
        
    goal = State([Bd([Holding([hand]), obj, goalProb], True),
                  Bd([GraspFace([obj, hand]), grasp, goalProb], True),
                  B([Grasp([obj, hand,  grasp]),
                     (0,-0.025,0,0), (0.01, 0.01, 0.01, 0.01), delta,
                     goalProb], True)])

    t.run(goal,
          hpn = hpn,
          skeleton = skel3 if skeleton else None,
          heuristic = heuristic,
          hierarchical = hierarchical,
          rip = rip,
          regions=['table1Top']
          )


######################################################################
#    Old tests    
'''
def test4(hpn = True, hierarchical = False, skeleton = False,
          heuristic = habbs, easy = False):

    goalProb, errProbs = (0.2,smallErrProbs) if easy else (0.95,typicalErrProbs)
    glob.rebindPenalty = 50


    t = PlanTest('test4',  errProbs, allOperators,
                 objects=['table1', 'objA', 'objB'])
    targetPose = (1.05, 0.25, tZ, 0.0)
    targetPoseB = (1.05, -0.2, tZ, 0.0)
    targetVar = (0.001, 0.001, 0.001, 0.005)
    targetDelta = (.02, .02, .02, .04)

    goal = State([Bd([In(['objA', 'table1MidFront']), True, goalProb], True),
                  Bd([In(['objB', 'table1MidRear']), True, goalProb], True)])

    t.run(goal,
          hpn = hpn,
          skeleton = skel if skeleton else None,
          hierarchical = hierarchical,
          regions=['table1Top'],
          heuristic = heuristic
          )

# Test placing in a region    
def test5(hpn = True, skeleton = False, heuristic=habbs, hierarchical = False,
          easy = False):

    goalProb, errProbs = (0.4,smallErrProbs) if easy else (0.95,typicalErrProbs)

    p1 = util.Pose(0.95, 0.0, tZ, 0.0)
    p2 = util.Pose(1.1, 0.0, tZ, 0.0)
    t = PlanTest('test5',  errProbs, allOperators,
                 objects=['table1', 'objA', 'table2'],
                 movePoses={'objA': p1,
                            'objB': p2})

    goal = State([Bd([In(['objA', 'table2Top']), True, goalProb], True)])

    t.run(goal,
          hpn = hpn,
          skeleton = [[place, move, pick, move]] if skeleton else None,
          regions=['table1Top', 'table2Top'],
          heuristic = heuristic
          )

# Test looking
def test6(hpn = True, skeleton=False, heuristic=habbs, hierarchical = False,
          easy = False, rip = False):

    goalProb, errProbs = (0.8,smallErrProbs) if easy else (0.95,typicalErrProbs)
        
    p2 = util.Pose(0.9, 0.0, tZ, 0.0)
    t = PlanTest('test6', errProbs, allOperators,
                 objects=['table1', 'objA'],
                 movePoses={'objA': p2},
                 varDict = {'table1': (0.1**2, 0.08**2, 0.000001, 0.1**2),
                            'objA': (0.075**2,0.075**2,0.000001,0.2**2)})

    goal = State([B([Pose(['objA', 4]), p2.xyztTuple(),
                     (0.001, 0.001, 0.001, 0.005),
                          (0.025,)*4, goalProb], True),
                  Bd([SupportFace(['objA']), 4, goalProb], True)])

    t.run(goal,
          hpn = hpn,
          skeleton = [[lookAt, move],
                      # need lookat to decrease prob
                      [lookAt,
                       move,
                       place.applyBindings({'Hand' : 'left'}),
                       move, pick, move,
                       poseAchCanPickPlace, lookAt, move, lookAt, move],
                      [lookAt,
                       move,
                       place.applyBindings({'Hand' : 'left'}),
                       move, pick, move, lookAt, move]] \
                      if skeleton else None,
          hierarchical = hierarchical,
          regions=['table1Top'],
          heuristic = heuristic,
          rip = rip
          )

# Test look, pick
def test7(hpn = True, flip=False, skeleton = False, heuristic=habbs,
          hierarchical = False, easy = False, gd = 0, rip = False):
    glob.rebindPenalty = 50
    goalProb, errProbs = (0.8,smallErrProbs) if easy else (0.99,typicalErrProbs)

    p1 = util.Pose(0.95, 0.0, tZ, 0.0)
    p2 = util.Pose(1.1, 0.0, tZ, 0.0)
    delta = (0.02, 0.02, 0.02, 0.05)

    t = PlanTest('test7',  errProbs, allOperators,
                 objects=['table1', 'objA', 'table2'],
                 movePoses={'objA': p2,
                            'objB': p1},
                 varDict = {'objA': (0.075**2,0.075**2, 1e-10,0.2**2)})
    targetPose = (1.05, 0.25, tZ, 0.0)

    goal = State([Bd([Holding(['left']), 'objA', goalProb], True),
                  Bd([GraspFace(['objA', 'left']), 0, goalProb], True),
                  B([Grasp(['objA', 'left', 0]),
                     (0,-0.025,0,0), (0.01, 0.01, 0.01, 0.01), delta,
                     goalProb], True)])
    homeConf = makeConf(t.world.robot, 0.0, 0.0, math.pi) if flip else None
    skel = [[lookAtHand, move, pick, move, lookAt, move]]*3
    t.run(goal,
          hpn = hpn,
          skeleton = skel if skeleton else None,
          heuristic = heuristic,
          hierarchical = hierarchical,
          regions=['table1Top'],
          home=homeConf,
          rip = rip
          )

# Regrasp!
def test8(hpn = True, skeleton=False, hierarchical = False, 
            heuristic = habbs, hand='left', flip = False, gd=1,
            easy = False):

    glob.rebindPenalty = 50
    goalProb, errProbs = (0.4, tinyErrProbs) if easy else (0.95,typicalErrProbs)
    #glob.monotonicFirst = False

    t = PlanTest('test8', errProbs, allOperators,
                 objects=['table1', 'table3', 'objA'])
    goalConf = makeConf(t.world.robot, 0.5, 1.0, 0.0)
    goal = State([Bd([Holding([hand]), 'objA', goalProb], True),
                  Bd([GraspFace(['objA', hand]), gd, goalProb], True),
                  B([Grasp(['objA', hand, gd]),
                     (0,-0.025,0,0), (0.001, 0.001, 0.001, 0.005),
                     (0.02,)*4, goalProb], True)])
    homeConf = makeConf(t.world.robot, 0.0, 0.0, math.pi) if flip else None
    goodSkel = [[pick,
                 move,
                 place.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
                 move,
                 lookAtHand,
                 move,
                 pick,
                 move]]
    t.run(goal,
          hpn = hpn,
          skeleton = goodSkel if skeleton else None,
          heuristic = heuristic, 
          hierarchical = hierarchical,
          home=homeConf,
          regions=['table1Top']
          )

def test9(hpn=True, skeleton = False, heuristic=habbs, hierarchical = False,
          easy = False, rip = False):
    glob.rebindPenalty = 50
    goalProb, errProbs = (0.4, tinyErrProbs) if easy else (0.95,typicalErrProbs)

    t = PlanTest('test9', errProbs, allOperators,
                 objects = ['table1'],
                 varDict = {'table1': (0.1**2, 0.05**2, 0.0000001, 0.1**2)},
                 # fixPoses={'table1': util.Pose(1.3, 0.0, 0.0, math.pi/2)}
                 )

    #goalConf = makeConf(t.world.robot, 1.1, 1.3, 0, 0.0)
    goalConf = makeConf(t.world.robot, 1.2, 1.4, 0, 0.0)
    confDeltas = (0.05, 0.05, 0.05, 0.05)
    goal = State([Conf([goalConf, confDeltas], True)])
    t.run(goal,
          greedy = .8,
          hpn = hpn,
          heuristic = heuristic,
          hierarchical = hierarchical,
          regions=['table1Top'],
          rip = rip,
          skeleton = [[move, poseAchCanReach,
                       lookAt, move]] if skeleton else None,
          )
    return t

def test10(hpn = True, skeleton = False, hierarchical = False, heuristic=habbs,
           easy = False, rip = False):
    # need to look at A to clear path to b
    glob.rebindPenalty = 50
    goalProb, errProbs = (0.4, tinyErrProbs) if easy else (0.99,typicalErrProbs)
    
    t = PlanTest('test10',  errProbs, allOperators,
                 objects=['table1', 'objA', 'objB'],
                 varDict = {'objA': (0.075**2,0.075**2, 1e-10,0.2**2)})
    targetPose = (1.05, 0.25, tZ, 0.0)
    targetPoseB = (1.05, -0.2, tZ, 0.0)
    targetVar = (0.01, 0.01, 0.01, 0.05) 

    goalConf = makeConf(t.world.robot, 0.5, 1.0, 0.0)
    confDeltas = (0.05, 0.05, 0.05, 0.05)

    goal = State([\
                  Bd([SupportFace(['objB']), 4, goalProb], True),
                  B([Pose(['objB', 4]),
                     targetPoseB, targetVar, (0.02,)*4,
                     goalProb], True)])
    t.run(goal,
          hpn = hpn,
          skeleton = [[place, move, pick, move, poseAchCanPickPlace,
                       lookAt, move]]*3 if skeleton else None,
          hierarchical = hierarchical,
          regions=['table1Top'],
          heuristic = heuristic,
          rip = rip
          )


def test11(hpn = True, skeleton = False, hierarchical = False,
           heuristic = habbs, easy = False, rip = False):
    glob.rebindPenalty = 50
    glob.monotonicFirst = hierarchical
    glob.monotonicFirst = True
    goalProb, errProbs = (0.4, tinyErrProbs) if easy else (0.99,typicalErrProbs)
    t = PlanTest('test11',  errProbs, allOperators,
                 objects=['table1', 'objA', 'objB'],
    # varDict = {} if easy else {'objA': (0.075**2, 0.1**2, 1e-10,0.2**2),
    #                            'objB': (0.075**2, 0.1**2, 1e-10,0.2**2),
    #                            'table1': (0.1**2, 0.03**2, 1e-10, 0.3**2)}
    varDict = {} if easy else {'objA': (0.075**2, 0.1**2, 1e-10,0.2**2),
                                'objB': (0.075**2, 0.1**2, 1e-10,0.2**2),
                                'table1': (0.05**2, 0.03**2, 1e-10, 0.2**2)})

    targetPose = (1.05, 0.25, tZ, 0.0)
    targetPoseB = (1.05, -0.25, tZ, 0.0)
    # targetVar = (0.002, 0.002, 0.002, 0.005)  make this work!
    targetVar = (0.0005, 0.0005, 0.0005, 0.001) 

    goal = State([\
                  Bd([SupportFace(['objA']), 4, goalProb], True),
                  B([Pose(['objA', 4]),
                     targetPose, targetVar, (0.02, 0.02, 0.02, 0.08),
                     goalProb], True),
                  Bd([SupportFace(['objB']), 4, goalProb], True),
                  B([Pose(['objB', 4]),
                     targetPoseB, targetVar, (0.02, 0.02, 0.02, 0.08),
                     goalProb], True)])

    skel = [[place.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
             move,
             place.applyBindings({'Obj' : 'objB', 'Hand' : 'right'}),
             move,
             lookAtHand.applyBindings({'Obj' : 'objB'}),
              move,
             pick.applyBindings({'Obj' : 'objB', 'Hand' : 'right'}),
             move,
             lookAt.applyBindings({'Obj' : 'objB'}),
             move,
             lookAtHand.applyBindings({'Obj' : 'objA'}),
             move,
             pick.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
             poseAchCanPickPlace,
             move,
             lookAt.applyBindings({'Obj' : 'objB'}),
             move,
             lookAt.applyBindings({'Obj' : 'objA'}),
             move]]*5


    hardSkel = [[lookAt.applyBindings({'Obj' : 'objA'}),
                move,
             place.applyBindings({'Obj' : 'objA', 'Hand' : 'right'}),
             move,
             pick.applyBindings({'Obj' : 'objA', 'Hand' : 'right'}),
             move,
             poseAchCanPickPlace,
             lookAt.applyBindings({'Obj' : 'objB'}),
             move,
             place.applyBindings({'Obj' : 'objB', 'Hand' : 'right'}),
             move,
             pick.applyBindings({'Obj' : 'objB', 'Hand' : 'right'}),
             move,
             lookAt.applyBindings({'Obj' : 'objA'}),
             move,
             lookAt.applyBindings({'Obj' : 'objB'}),
             move,
             lookAt.applyBindings({'Obj' : 'objA'}),
             move,
             lookAt.applyBindings({'Obj' : 'objB'}),
             move]]*5

    hierSkel = [[lookAt.applyBindings({'Obj' : 'objB'}),
                 lookAt.applyBindings({'Obj' : 'objA'}),
                 place.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
                 place.applyBindings({'Obj' : 'objB', 'Hand' : 'left'})],
                [place.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
                 poseAchCanPickPlace.applyBindings({'Obj' : 'objB'})],
                [poseAchCanPickPlace.applyBindings({'Obj' : 'objB'}),
                 lookAt.applyBindings({'Obj' : 'objA'}),
                 move,
                 poseAchCanPickPlace.applyBindings({'Obj' : 'objB'}),
                 lookAt.applyBindings({'Obj' : 'table1'}),
                 move]]
                 
    t.run(goal,
          hpn = hpn,
          skeleton = hierSkel if skeleton else None,
          hierarchical = hierarchical,
          #regions=['table1Top'],
          heuristic = heuristic,
          rip = rip
          )
    
def test12(hpn = True, skeleton = False, hierarchical = False,
           heuristic = habbs, easy = False, rip = False):
    glob.rebindPenalty = 50
    goalProb, errProbs = (0.4, tinyErrProbs) if easy else (0.99,typicalErrProbs)

    t = PlanTest('test12',  errProbs, allOperators,
                 objects=['table1', 'table2',
                          'objA', 'objB', 'objD',
                          'objE', 'objF', 'objG'])
    targetPose = (1.05, 0.25, tZ, 0.0)
    targetPoseB = (1.05, -0.2, tZ, 0.0)
    targetVar = (0.001, 0.001, 0.001, 0.005) 

    goalConf = makeConf(t.world.robot, 0.5, 1.0, 0.0)
    confDeltas = (0.05, 0.05, 0.05, 0.05)
    hand = 'left'
    gd = 0
    goal = State([Bd([Holding([hand]), 'objA', goalProb], True),
                  Bd([GraspFace(['objA', hand]), 0, goalProb], True),
                  B([Grasp(['objA', hand, gd]),
                     (0,-0.025,0,0), (0.001, 0.001, 0.001, 0.001),
                     (0.02,)*4, goalProb], True)])

    easySkel = [[pick,
                       move,
                       poseAchCanPickPlace,
                       place.applyBindings({'Hand' : 'left'}),
                       move,
                       pick,
                       move,
                       ]]

    hardSkel = [[lookAtHand, move, pick,
                       move,
                       poseAchCanPickPlace,
                       lookAt,
                       move,
                       place.applyBindings({'Hand' : 'left'}),
                       move,
                       pick,
                       move,
                       ]]
        
    t.run(goal,
          hpn = hpn,
          skeleton = hardSkel if skeleton else None,
          hierarchical = hierarchical,
          # regions=['table2Top'],
          regions=['table2Top', 'table1Top'],
          rip = rip,
          heuristic = heuristic
          )

def test13(hpn = True, skeleton = False, hierarchical = False, heuristic=habbs,
           easy = False, rip = False):
    glob.rebindPenalty = 50
    goalProb, errProbs = (0.4, tinyErrProbs) if easy else (0.99,typicalErrProbs)
    t = PlanTest('test13',  errProbs, allOperators,
                 objects=['table1', 'table2', 'objA', 'objB', 'objD',
                          'objE', 'objF', 'objG'])
    targetPose = (1.05, 0.25, tZ, 0.0)
    targetPoseB = (1.05, -0.2, tZ, 0.0)
    targetVar = (0.001, 0.001, 0.001, 0.005)  # should be this

    if skeleton and not hierarchical:
        raw_input('skeleton wrong for this problem')

    hskel = [[place],  #0
             [place, poseAchCanReach, poseAchCanReach, poseAchCanReach], #1
             [poseAchCanReach, lookAt, place], #2
             [place], #3
             [place, pick], #4
             [pick], #5
             [pick, move], #6
             [move], #7
             [place, pick], #8
             [pick, poseAchCanReach, poseAchCanReach, poseAchCanReach], #9
             [poseAchCanReach, lookAt, place], #10
             [place], #11
             [place, pick], #12
             [pick], #13
             [pick, move], #14
             [move], #15
             [place.applyBindings({'Obj' : 'objE'}),
              move], #16
             [move], #17
             [pick.applyBindings({'Obj' : 'objA'}),   # 18
              move,
              place.applyBindings({'Obj' : 'objF'}), move],
             [move], 
             [place],
             [move], 
             [place.applyBindings({'Hand' : 'left', 'Obj' : 'objA'})
              ],  # a, left
             [pick, move],
             [move],
             [place, move],
             [move]]

    goal = State([\
                  Bd([SupportFace(['objA']), 4, goalProb], True),
                  B([Pose(['objA', 4]),
                     targetPose, targetVar, (0.02,)*4,
                     goalProb], True)])
    t.run(goal,
          hpn = hpn,
          skeleton = hskel if skeleton else None,
          hierarchical = hierarchical,
          regions=['table1Top', 'table2Top'],
          rip = rip,
          heuristic = heuristic
          )
    
def test14(hpn = True, skeleton = False, hierarchical = False, heuristic=habbs,
           easy = False, rip = False):
    # Move A so we can look at B
    # Example isn't really constructed right

    glob.rebindPenalty = 50
    goalProb, errProbs = (0.4, tinyErrProbs) if easy else (0.99,typicalErrProbs)

    p1 = util.Pose(0.9, 0.0, tZ, 0.0)
    p2 = util.Pose(1.3, 0.0, tZ, 0.0)
    t = PlanTest('test14', errProbs, allOperators,
                 objects=['table1', 'objA', 'objB', 'table2',
                          'cupboardSide1', 'cupboardSide2'],
                 movePoses={'objA': p1,
                            'objB': p2},
                 varDict = {'objA': (0.05**2,0.05**2, 1e-10,0.2**2),
                            'objB': (0.05**2,0.05**2, 1e-10,0.2**2)})

    goal = State([ Bd([SupportFace(['objB']), 4, goalProb], True),
                   B([Pose(['objB', 4]), p2.xyztTuple(),
                     (0.00001, 0.00001, 0.00001, 0.0001),
                          (0.02,)*4, goalProb], True)])

    t.run(goal,
          hpn = hpn,
          skeleton = [[lookAt, poseAchCanSee, move,
                       place, move, pick, move]] \
                             if skeleton else None,
          hierarchical = hierarchical,
          regions=['table1Top'],
          heuristic = heuristic,
          rip = rip
          )


# Test look at hand
def test15(hpn = True, skeleton=False, hand='left', flip = False, gd = 0,
           heuristic=habbs, hierarchical=False, easy = False):

    glob.rebindPenalty = 50
    goalProb, errProbs = (0.4, tinyErrProbs) if easy else (0.99,typicalErrProbs)

    t = PlanTest('test15', errProbs, allOperators,
                 objects=['table1', 'objA'])
    goalConf = makeConf(t.world.robot, 0.5, 1.0, 0.0)
    confDeltas = (0.05, 0.05, 0.05, 0.05)
    goal = State([Bd([Holding([hand]), 'objA', .6], True),
                  Bd([GraspFace(['objA', hand]), gd, .6], True),
                  B([Grasp(['objA', hand, gd]),
                     (0,-0.025,0,0), (0.0001, 0.0001, 0.0001, 0.0001),
                     (0.02,)*4, 0.6], True),
                  Conf([goalConf, confDeltas], True)])
    homeConf = makeConf(t.world.robot, 0.0, 0.0, math.pi) \
                         if flip else None
    t.run(goal,
          hpn = hpn,
          skeleton = [[move, lookAtHand, move, pick, move]] \
          if skeleton else None,
          heuristic = heuristic,
          regions=['table1Top'],
          home=homeConf
          )

# pick and place with more noise 
def test16(hpn = True, skeleton = False, hierarchical = False,
           heuristic = habbs, easy = False, rip = False):
    glob.rebindPenalty = 50
    goalProb, errProbs = (0.4, tinyErrProbs) if easy else (0.99,typicalErrProbs)

    t = PlanTest('test16',  errProbs, allOperators,
                 objects=['table1', 'objA', 'objB'],
                 varDict = {'objA': (0.05**2,0.05**2, 1e-10,0.2**2),
                            'objB': (0.05**2,0.05**2, 1e-10,0.2**2)})

    targetPose = (1.05, 0.25, tZ, 0.0)
    targetVar = (0.0001, 0.0001, 0.0001, 0.0005)
    goal = State([Bd([SupportFace(['objA']), 4, .5], True),
                  B([Pose(['objA', 4]),
                     targetPose, targetVar, (0.05,)*4,
                     goalProb], True)])
    t.run(goal,
          hpn = hpn,
          skeleton = [[lookAt, move, place, move,
                       pick, move, lookAt]]*5 \
                       if skeleton else None,
          hierarchical = hierarchical,
          regions=['table1Top'],
          rip = rip,
          heuristic = heuristic
          )

# Do first with tiny error probs, and then gradually relax.

# A sequence of tests, working up to swap.
# 17.  A in back, B in parking -> B in back   (move out of the way)
# 18.  A in parking, B in parking -> A in front, B in back  (ordering)
# 19.  A in back, B in parking -> A in front, B in back (combination)
# 20.  A in front, B in parking -> A in front, B in back (combination, non-mon)
# 21.  A in back, B in front -> A in front, B in back (whole enchilada)

def test17(hpn = True, skeleton = False, hierarchical = False,
           heuristic = habbs, easy = False, rip = False):
    glob.rebindPenalty = 50
    goalProb, errProbs = (0.4, tinyErrProbs) if easy else (0.99,typicalErrProbs)

    front = util.Pose(0.95, 0.0, tZ, 0.0)
    back = util.Pose(1.1, 0.0, tZ, 0.0)
    parking = util.Pose(0.95, 0.3, tZ, 0.0)
    t = PlanTest('test17',  errProbs, allOperators,
                 objects=['table1', 'objA', 'objB'],
                 movePoses={'objA': back,
                            'objB': parking},
                 varDict = {'objA': (0.05**2,0.05**2, 1e-10,0.2**2),
                            'objB': (0.05**2,0.05**2, 1e-10,0.2**2)})

    skel = [[place.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
             move,
             pick.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
             move,
             poseAchCanPickPlace,
             place.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
             move,
             lookAtHand,
             move,
             pick.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
             move]]*5

    # Small var
    targetVar = (0.0001, 0.0001, 0.0001, 0.0005)
    targetDelta = (0.02, 0.02, 0.02, 0.05)
    
    goal = State([Bd([SupportFace(['objB']), 4, goalProb], True),
                  B([Pose(['objB', 4]),
                     back.xyztTuple(), targetVar, targetDelta,
                     goalProb], True)])

    t.run(goal,
          hpn = hpn,
          skeleton = skel if skeleton else None,
          heuristic = heuristic,
          hierarchical = hierarchical,
          rip = rip,
          regions=['table1Top']
          )


# 18.  A in parking, B in parking -> A in front, B in back  (ordering)
# Trivial with skeleton
def test18(hpn = True, skeleton = False, hierarchical = False,
           heuristic = habbs, easy = False, rip = False):

    glob.rebindPenalty = 50
    goalProb, errProbs = (0.4, tinyErrProbs) if easy else (0.99,typicalErrProbs)

    front = util.Pose(0.95, 0.0, tZ, 0.0)
    back = util.Pose(1.1, 0.0, tZ, 0.0)
    parking1 = util.Pose(0.95, 0.3, tZ, 0.0)
    parking2 = util.Pose(0.95, -0.3, tZ, 0.0)
    t = PlanTest('test18',  errProbs, allOperators,
                 objects=['table1', 'objA', 'objB'],
                 fixPoses={'objA': parking1,
                            'objB': parking2},
                 varDict = {'objA': (0.05**2,0.05**2, 1e-10,0.2**2),
                            'objB': (0.05**2,0.05**2, 1e-10,0.2**2)})

    skel = [[place.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
             move,
             pick.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
             move,
             place.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
             move,
             pick.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
             move]]*5

    # Small var
    targetVar = (0.0001, 0.0001, 0.0001, 0.0005)
    targetDelta = (0.02, 0.02, 0.02, 0.05)
    
    goal = State([Bd([SupportFace(['objA']), 4, goalProb], True),
                  B([Pose(['objA', 4]),
                     front.xyztTuple(), targetVar, targetDelta,
                     goalProb], True),
                  Bd([SupportFace(['objB']), 4, goalProb], True),
                  B([Pose(['objB', 4]),
                     back.xyztTuple(), targetVar, targetDelta,
                     goalProb], True)])

    t.run(goal,
          hpn = hpn,
          skeleton = skel if skeleton else None,
          hierarchical = hierarchical,
          heuristic = heuristic,
          regions=['table1Top'],
          rip = rip
          )

# 19.  A in back, B in parking -> A in front, B in back (combination)    
def test19(hpn = True, skeleton = False, hierarchical = False,
           heuristic = habbs, easy = False, rip = False):

    glob.rebindPenalty = 50
    goalProb, errProbs = (0.4, tinyErrProbs) if easy else (0.99,typicalErrProbs)

    front = util.Pose(0.95, 0.0, tZ, 0.0)
    back = util.Pose(1.1, 0.0, tZ, 0.0)
    parking1 = util.Pose(0.95, 0.3, tZ, 0.0)
    parking2 = util.Pose(0.95, -0.3, tZ, 0.0)
    t = PlanTest('test19',  errProbs, allOperators,
                 objects=['table1', 'objA', 'objB'],
                 movePoses={'objA': back,
                            'objB': parking2},
                 varDict = {'objA': (0.05**2,0.05**2, 1e-10,0.2**2),
                            'objB': (0.05**2,0.05**2, 1e-10,0.2**2)})

    skel = [[place.applyBindings({'Obj' : 'objA', 'Hand' : 'right'}),
             move,
             place.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
             move,
             pick.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
             move,
             pick.applyBindings({'Obj' : 'objA', 'Hand' : 'right'}),
             move]]*5

    # Small var
    targetVar = (0.0001, 0.0001, 0.0001, 0.0005)
    targetDelta = (0.02, 0.02, 0.02, 0.05)
    
    goal = State([Bd([SupportFace(['objA']), 4, goalProb], True),
                  B([Pose(['objA', 4]),
                     front.xyztTuple(), targetVar, targetDelta,
                     goalProb], True),
                  Bd([SupportFace(['objB']), 4, goalProb], True),
                  B([Pose(['objB', 4]),
                     back.xyztTuple(), targetVar, targetDelta,
                     goalProb], True)])

    t.run(goal,
          hpn = hpn,
          skeleton = skel if skeleton else None,
          hierarchical = hierarchical,
          heuristic = heuristic,
          regions=['table1Top'],
          rip = rip
          )


def test19a(hpn = True, skeleton = False, hierarchical = False,
           heuristic = habbs, easy = False):
    glob.rebindPenalty = 50
    goalProb, errProbs = (0.4, tinyErrProbs) if easy else (0.99,typicalErrProbs)
    glob.monotonicFirst = True

    front = util.Pose(0.95, 0.0, tZ, 0.0)
    back = util.Pose(1.1, 0.0, tZ, 0.0)
    parking1 = util.Pose(0.95, 0.3, tZ, 0.0)
    parking2 = util.Pose(0.95, -0.3, tZ, 0.0)
    parkingBad = util.Pose(1.183, 0.222, tZ0, 1.571)
    t = PlanTest('test19a',  errProbs, allOperators,
                 objects=['table1', 'objA', 'objB'],
                 movePoses={'objA': parkingBad,
                            'objB': back})

    # Small var
    targetVar = (0.0001, 0.0001, 0.0001, 0.0005)
    targetDelta = (0.02, 0.02, 0.02, 0.05)

    easyGoal = State([Bd([SupportFace(['objA']), 4, goalProb], True),
                  B([Pose(['objA', 4]),
                     (0.45, -0.4, tZ, 0.0), targetVar, targetDelta,
                     goalProb], True)])
    
    goal = State([Bd([SupportFace(['objA']), 4, goalProb], True),
                  B([Pose(['objA', 4]),
                     front.xyztTuple(), targetVar, targetDelta,
                     goalProb], True),
                  Bd([SupportFace(['objB']), 4, goalProb], True),
                  B([Pose(['objB', 4]),
                     back.xyztTuple(), targetVar, targetDelta,
                     goalProb], True)])

    flatSkeleton = [[place.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
                     move,
                     pick.applyBindings({'Obj' : 'objA'}),
                     move,
                     place.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
                     move,
                     lookAtHand.applyBindings({'Hand' : 'left'}),
                     move, 
                     pick.applyBindings({'Obj' : 'objB'}),
                     move,                                              
                     place.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
                     move,
                     lookAtHand.applyBindings({'Hand' : 'left'}),
                     move, 
                     pick.applyBindings({'Obj' : 'objA'}),
                     move,
                     place.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
                     move,
                     lookAtHand.applyBindings({'Hand' : 'left'}),
                     move, 
                     pick.applyBindings({'Obj' : 'objB'}),
                     move]]


    easySkeleton = [[place.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
                     move,
                     pick.applyBindings({'Obj' : 'objA'}),
                     move, 
                     poseAchCanPickPlace,
                     place.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
                     move,
                     lookAtHand.applyBindings({'Hand' : 'left'}),
                     move, 
                     pick.applyBindings({'Obj' : 'objB'}),
                     move]]

    t.run(goal,  #goal
          hpn = hpn,
          skeleton = easySkeleton if skeleton else None, #flatSkeleton,
          hierarchical = hierarchical,
          heuristic = heuristic,
          regions=['table1Top']
          )

# 20.  Swap!
def test20(hpn = True, skeleton = False, hierarchical = False,
           heuristic = habbs, easy = False, rip = False):

    glob.rebindPenalty = 50
    goalProb, errProbs = (0.4, tinyErrProbs) if easy else (0.99,typicalErrProbs)
    glob.monotonicFirst = True


    front = util.Pose(0.95, 0.0, tZ, 0.0)
    back = util.Pose(1.1, 0.0, tZ, 0.0)
    parking1 = util.Pose(0.95, 0.3, tZ, 0.0)
    parking2 = util.Pose(0.95, -0.3, tZ, 0.0)
    t = PlanTest('test20',  errProbs, allOperators,
                 objects=['table1', 'objA', 'objB'],
                 movePoses={'objA': back,
                            'objB': front},
                 varDict = {'objA': (0.05**2,0.05**2, 1e-10,0.2**2),
                            'objB': (0.05**2,0.05**2, 1e-10,0.2**2)})

    # This just gets us down the first left expansion
    hierSkel = [[place, place], #0
                [place, poseAchCanPickPlace], #1
                [poseAchCanPickPlace, place], #2
                [place, pick], #3
                [lookAtHand.applyBindings({'Hand' : 'left', 'Obj':'objA'}),
                 pick,
                 poseAchCanPickPlace], #4
                [poseAchCanPickPlace, lookAt, place], #5
                [place], #6
                [place, lookAtHand, pick],#7
                [pick],  #8
                [pick, move], #9
                [move]] #10


    swapSkel = [[place.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
                 move,
                 pick.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
                 move,
                 place.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
                 move,
                 pick.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
                 move,
                 lookAt.applyBindings({'Obj' : 'objA'}),
                 move,
                 place.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
                 move,
                 pick.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
                 move,
                 place.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
                 move,
                 pick.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
                 move]]*5

    # Small var
    targetVar = (0.0001, 0.0001, 0.0001, 0.0005)
    targetDelta = (0.02, 0.02, 0.02, 0.05)
    
    goal = State([Bd([SupportFace(['objA']), 4, goalProb], True),
                  B([Pose(['objA', 4]),
                     front.xyztTuple(), targetVar, targetDelta,
                     goalProb], True),
                  Bd([SupportFace(['objB']), 4, goalProb], True),
                  B([Pose(['objB', 4]),
                     back.xyztTuple(), targetVar, targetDelta,
                     goalProb], True)])

    t.run(goal,
          hpn = hpn,
          skeleton = hierSkel if skeleton else None,
          heuristic = heuristic,
          hierarchical = hierarchical,
          rip = rip,
          regions=['table1Top']
          )

# 20a.  A situation we encounter if we serialize badly.  A is in
# front, B is in the hand.
def test20a(hpn = True, skeleton = False, hierarchical = False,
           heuristic = habbs, easy = False, rip = False):

    glob.rebindPenalty = 50
    goalProb, errProbs = (0.4, tinyErrProbs) if easy else (0.99,typicalErrProbs)
    glob.monotonicFirst = False

    front = util.Pose(0.95, 0.0, tZ, 0.0)
    back = util.Pose(1.15, 0.0, tZ, 0.0)

    t = PlanTest('test20a',  errProbs, allOperators,
                 objects=['table1', 'objA', 'objB'],
                 movePoses={'objA': front,
                            'objB': back},
                 varDict = {'objA': (.01**2, .01**2, .001**2, .01**2)})

    flatSkel = [[lookAt.applyBindings({'Obj' : 'objA'}),
                move,
                place.applyBindings({'Obj' : 'objA', 'Hand' : 'right'}),
                move,
                lookAt.applyBindings({'Obj' : 'objB'}),
                move,                
                place.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
                move,
                pick.applyBindings({'Obj' : 'objA', 'Hand' : 'right'}),
                move,
                lookAt.applyBindings({'Obj' : 'objA'}),
                move]]

    flatSkelRegrasp = [[lookAt.applyBindings({'Obj' : 'objB'}),
                move,
                place.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
                move,
                pick.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
                move,
                lookAt.applyBindings({'Obj' : 'objB'}),
                move,                
                place.applyBindings({'Obj' : 'objB', 'Hand' : 'left'}),
                move]]

    grasped = 'objB'
    hand = 'left'
    def initBel(bs):
        # Change pbs so obj B is in the hand
        gm = (0, -0.025, 0, 0)
        gv = (0.014**2, 0.014**2, 0.0001**2, 0.022**2)
        gd = (1e-4,)*4
        gf = 0
        bs.pbs.updateHeld(grasped, gf, PoseD(gm, gv), hand, gd)
        bs.pbs.excludeObjs([grasped])
        bs.pbs.shadowWorld = None # force recompute

    def initWorld(bs, realWorld):
        attachedShape = bs.pbs.getRobot().\
                         attachedObj(bs.pbs.getShadowWorld(0.9), hand)
        shape = bs.pbs.getWorld().\
               getObjectShapeAtOrigin(grasped).applyLoc(attachedShape.origin())
        realWorld.robot.attach(shape, realWorld, hand)
        robot = bs.pbs.getRobot()
        cart = realWorld.robotConf.cartConf()
        handPose = cart[robot.armChainNames[hand]].compose(robot.toolOffsetX[hand])
        pose = shape.origin()
        realWorld.held[hand] = grasped
        realWorld.grasp[hand] = handPose.inverse().compose(pose)
        realWorld.delObjectState(grasped)    

    # Small var
    targetVar = (0.0001, 0.0001, 0.0001, 0.0005)
    targetDelta = (0.02, 0.02, 0.02, 0.05)
    
    goal = State([Bd([SupportFace(['objA']), 4, goalProb], True),
                  B([Pose(['objA', 4]),
                     front.xyztTuple(), targetVar, targetDelta,
                     goalProb], True),
                  Bd([SupportFace(['objB']), 4, goalProb], True),
                  B([Pose(['objB', 4]),
                     back.xyztTuple(), targetVar, targetDelta,
                     goalProb], True)])

    t.run(goal,
          hpn = hpn,
          skeleton = flatSkelRegrasp if skeleton else None,
          heuristic = heuristic,
          hierarchical = hierarchical,
          rip = rip,
          regions=['table1Top'],
          initBelief = initBel,
          initWorld = initWorld
          )

    
# stack objects?
def testStack(hpn = True, skeleton = False, hierarchical = False,
           heuristic = habbs):
    p1 = util.Pose(0.95, 0.0, tZ, 0.0)
    p2 = util.Pose(1.1, 0.0, tZ, 0.0)
    p3 = util.Pose(0.95, 0.2, tZ, 0.0)
    t = PlanTest('test18',  smallErrProbs, allOperators,
                 objects=['table1', 'objA', 'objB', 'objC'],
                 movePoses={'objA': p1,
                            'objB': p2,
                            'objC': p3})

    skel = [[place.applyBindings({'Obj' : 'objC', 'Hand' : 'left'}),
             move,
             lookAtHand.applyBindings({'Obj' : 'objC'}),
             move,
             pick.applyBindings({'Obj' : 'objC', 'Hand' : 'left'}),
             move,
             place.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
             move,
             lookAtHand.applyBindings({'Obj' : 'objA'}),
             move,
             pick.applyBindings({'Obj' : 'objA', 'Hand' : 'left'}),
             move]]*5

    goal = State([Bd([In(['objA', 'objBTop']), True, .4], True),
                  # Bd([In(['objC', 'objATop']), True, .4], True)
                  ])

    t.run(goal,
          hpn = hpn,
          skeleton = skel if skeleton else None,
          heuristic = heuristic,
          hierarchical = hierarchical,
          regions=['objATop', 'objBTop', 'table1Top']
          )

# Empty hand
def test21(hpn = True, skeleton = False, hierarchical = False,
           heuristic = habbs, easy=False):
    p1 = util.Pose(0.95, 0.0, tZ, 0.0)
    p2 = util.Pose(0.95, 0.4, tZ, 0.0)

    t = PlanTest('test21',  smallErrProbs, allOperators,
                 objects=['table1', 'objA', 'objB'],
                 movePoses={'objA': p1,
                            'objB': p2})

    # Init var
    initGraspVar = (1e-4,)*4    # very small
    # initGraspVar = (0.01, 0.01, 0.01, 0.05)    # very big

    # Small var
    targetVar = (0.0001, 0.0001, 0.0001, 0.0005)
    targetDelta = (0.001, 0.001, 0.001, 0.005)
    # Increase this
    goalProb = 0.4
    # Need to empty the hand in order to achieve this
    goal3 = State([Bd([SupportFace(['objA']), 4, goalProb], True),
                  B([Pose(['objA', 4]),
                     p2.xyztTuple(), targetVar, targetDelta,
                     goalProb], True)])
    # Just empty the hand.  No problem.
    goal1 = State([Bd([Holding(['left']), 'none', goalProb], True)])
    # Empty the hand.  Other goal condition is already true.
    goal2 = State([Bd([Holding(['left']), 'none', goalProb], True),
                   Bd([SupportFace(['objA']), 4, goalProb], True),
                   B([Pose(['objA', 4]),
                     p1.xyztTuple(), targetVar, targetDelta,
                     goalProb], True)])

    grasped = 'objB'
    hand = 'left'
    def initBel(bs):
        # Change pbs so obj B is in the hand
        gm = (0, -0.025, 0, 0)
        gv = initGraspVar
        gd = (1e-4,)*4
        gf = 0
        bs.pbs.updateHeld(grasped, gf, PoseD(gm, gv), hand, gd)
        bs.pbs.excludeObjs([grasped])
        bs.pbs.shadowWorld = None # force recompute

    def initWorld(bs, realWorld):
        attachedShape = bs.pbs.getRobot().attachedObj(bs.pbs.getShadowWorld(0.9), hand)
        shape = bs.pbs.getWorld().getObjectShapeAtOrigin(grasped).applyLoc(attachedShape.origin())
        realWorld.robot.attach(shape, realWorld, hand)
        robot = bs.pbs.getRobot()
        cart = realWorld.robotConf.cartConf
        handPose = cart[robot.armChainNames[hand]].compose(robot.toolOffsetX[hand])
        pose = shape.origin()
        realWorld.held[hand] = grasped
        realWorld.grasp[hand] = handPose.inverse().compose(pose)
        realWorld.delObjectState(grasped)    

    skeleton1 = [[place, move, lookAtHand, move, lookAtHand,
                  move]]
    t.run(goal1,
          hpn = hpn,
          skeleton = skeleton1 if skeleton else None,
          heuristic = heuristic,
          regions = ['table1Top'],
          initBelief = initBel,
          initWorld = initWorld
          )

# Need to verify that hand is empty
def test22(hpn = True, skeleton = False, hierarchical = False,
           heuristic = habbs, easy=False):
    p1 = util.Pose(0.95, 0.0, tZ, 0.0)
    p2 = util.Pose(0.95, 0.4, tZ, 0.0)

    t = PlanTest('test22',  smallErrProbs, allOperators,
                 objects=['table1', 'objA', 'objB'],
                 movePoses={'objA': p1,
                            'objB': p2})

    # Small var
    targetVar = (0.0001, 0.0001, 0.0001, 0.0005)
    # Bigger for testing
    targetVar = (0.001, 0.001, 0.001, 0.005)
    targetDelta = (0.001, 0.001, 0.001, 0.005)
    # Increase this
    goalProb = 0.4

    goal1 = State([Bd([Holding(['left']), 'none', .97], True)])


    # Need to be very sure
    goal3 = State([Bd([Holding(['left']), 'objA', .97], True),
                   Bd([GraspFace(['objA', 'left']), 0, .97], True),
                   B([Grasp(['objA', 'left', 0]),
                     (0,-0.025,0,0), (0.001, 0.001, 0.001, 0.001), (0.001,)*4,
                     .97], True)])

    # Need to verify empty hand in order to achieve this
    goal2 = State([Bd([SupportFace(['objA']), 4, goalProb], True),
                  B([Pose(['objA', 4]),
                     p2.xyztTuple(), targetVar, targetDelta,
                     goalProb], True)])

    def initBel(bs):
        # Change pbs so we are unsure whether both hands are empty
        bs.pbs.held['left'] = dist.DDist({'none' : .6, 'objA' : .3, 'objB': .1})
        bs.pbs.held['right'] = dist.DDist({'none' : .6, 'objA' : .1, 'objB': .3})
        bs.pbs.shadowWorld = None # force recompute
        bs.pbs.draw(0.9, 'W')
        bs.pbs.draw(0.9, 'Belief')

    skeleton1 = [[lookAtHand.applyBindings({'Hand' : 'left'}),
                  move,
                  lookAtHand.applyBindings({'Hand' : 'left'}),
                  move]]
    skeleton2 = [[lookAt, move, place, move, pick, move, lookAtHand, move]]

    t.run(goal2,
          hpn = hpn,
          skeleton = skeleton1 if skeleton else None,
          heuristic = heuristic,
          initBelief = initBel,
          regions = ['table1Top']
          )


'''

def testSim():
    varDict =  {'table1': (0.07**2, 0.03**2, 1e-10, 0.2**2),
                'objA': (0.1**2, 0.1**2, 1e-10, 0.3**2),
                'objB': (0.1**2, 0.1**2, 1e-10, 0.3**2)} 
    front = util.Pose(1.1, 0.0, tZ, 0.0)
    right = util.Pose(1.1, -0.4, tZ, 0.0)
    table1Pose = util.Pose(1.3, 0.0, 0.0, math.pi/2)

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

# Get false fluents
def ff(g, details):
    return [thing for thing in g.fluents if thing.isGround() \
            and thing.valueInDetails(details) == False]

def testReact():
    t = PlanTest('testReact', typicalErrProbs, allOperators, multiplier = 1)
    startConf = makeConf(t.world.robot, 0.0, 0.0, 0.0)
    result, cnfOut = pr2GoToConf(startConf, 'move')
    result, cnfOut = pr2GoToConf(startConf, 'look')
    # Reset the internal coordinate frames
    result, cnfOut = pr2GoToConf(startConf, 'reset')
    glob.debugOn.append('invkin')
    testReactive(startConf)

def gripOpen(conf, hand, width=0.08):
    return conf.set(conf.robot.gripperChainNames[hand], [width])

def testOpen(hand='left'):
    t = PlanTest('testReact', typicalErrProbs, allOperators, multiplier = 1)
    startConf = makeConf(t.world.robot, 0.0, 0.0, 0.0)[0]
    result, cnfOut = pr2GoToConf(gripOpen(startConf, hand), 'open')    

def testBusy(hpn = True, skeleton = False, hierarchical = False,
           heuristic = habbs, easy = False, rip = False,
           hardSwap = False):


    # Seems to need this
    global useRight, useVertical
    useRight, useVertical = True, True

    glob.rebindPenalty = 150
    goalProb, errProbs = (0.4, tinyErrProbs) if easy else (0.95,typicalErrProbs)
    glob.monotonicFirst = True
    table2Pose = util.Pose(1.0, -1.2, 0.0, 0.0)
    
    front = util.Pose(0.95, 0.0, tZ, 0.0)
    back = util.Pose(1.25, 0.0, tZ, 0.0)

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
          skeleton = hardSkel if skeleton else None,
          heuristic = heuristic,
          hierarchical = hierarchical,
          rip = rip,
          regions=['table1Top', 'table2Top', 'table1MidFront',
                   'table1MidRear']
          )
