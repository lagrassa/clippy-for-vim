    
# regrasp
def testRegrasp(hpn = False, draw=False):
    t = PlanTest('testRegrasp',  typicalErrProbs, 
    objects=['table1', 'table2', 'objA'], multiplier=4)
    targetPose = (0.55, 0.25, 0.65, 0.0)

    goalProb = 0.5

    goal = State([Bd([SupportFace(['objA']), 4, .5], True),
                  B([Pose(['objA', 4]),
                     targetPose, (0.001, 0.001, 0.001, 0.005), (0.001,)*4,
                     0.5], True)])
    t.run(goal,
          hpn = hpn,
          skeleton = [['place', 'move', 'pick', 'move']],
          operators=['move', 'pick', 'place']
          )
    raw_input('Done?')

def testWorldState(draw = True):
    world = testWorld(['table1', 'table2', 'objA', 'objB'])
    ws = WorldState(world)
    conf = JointConf(pr2Init.copy(), world.robot.chains)
    ws.setRobotConf(conf)
    ws.setObjectPose('objA', util.Pose(-1.0, 0.0, 0.0, 0.0))
    ws.setObjectPose('objB', util.Pose(0.6, 0.0, 0.7, 1.57))
    ws.setObjectPose('table1', util.Pose(0.6, 0.0, 0.3, math.pi/2))
    ws.setObjectPose('table2', util.Pose(0.9, 0.0, 0.9, math.pi/2))
    if draw: ws.draw('W')
    return ws

def testScan():
    scan = (0.3, 0.7, 2.0, 2., 30)      # len should be 5
    scan = (0.3, 0.1, 0.1, 2., 30)      # len should be 5
    s = Scan(util.Pose(0,0,1.5,0).compose(util.Transform(transf.rotation_matrix(math.pi/4, (0,1,0)))), scan)
    wm.makeWindow('W', viewPort)
    s.draw('W', 'pink')
    ws = testWorldState()
    dm = np.zeros(3722)
    dm.fill(10.0)
    contacts = 3722*[None]
    print updateDepthMap(s, ws.objectShapes['table1'], dm, contacts)
    print updateDepthMap(s, ws.objectShapes['table2'], dm, contacts)
    print updateDepthMap(s, ws.objectShapes['objB'], dm, contacts)
    r = 0.01
    pointBox = shapes.BoxAligned(np.array([(-r, -r, -r), (r, r, r)]), None)
    count = 0
    for c in contacts:
        if c != None:
            pose = util.Pose(c[0], c[1], c[2], 0.0)
            pointBox.applyTrans(pose).draw('W', 'red')
            count += 1
    return count

    
