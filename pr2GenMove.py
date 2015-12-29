import pdb
import time
import hu
from pr2RRT import planRobotGoalPath
from pr2GenTests import canReachHome
from planUtil import Violations
from pr2Util import Hashable
from pr2Visible import viewCone, visible
from shapes import Shape
from traceFile import debugMsg, debug, tr, trAlways
from ucSearchPQ import search
import planGlobals as glob
import windowManager3D as wm

"""Do a 'belief space' search for a move/look sequence to move between
two confs. If the motion does not involve the base or the base motion
is short, then just do vanilla RRT.  Otherwise consider moves, which
increase variance on all objects and look(x) which reduce variance of
one object.  Looks require being in a valid look zone.  For long
moves, first tuck in the arm, do displacement and looks and finally do
an untuck."""

lookCost = 1.0
delta = 0.1                # step size in x and y
deltaTh = 0.21             # step size in theta pi/15 (12 degrees)
thConvert = 1.0
maxDirect = 0.2

def ptDist(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (thConvert*hu.angleDiff(p1[2],p2[2]))**2)**0.5

# Is v1 less than v2+eps? 
def varLess(v1, v2, eps=0):
    return any(v1[i] < (v2[i]+eps) for i in (0,1,3))

# does vs1 subsume vs2?  That is, for every object, is variance in vs1
# somewhat smaller (or about the same) as that in vs2?
def varSubsumes(vs1, vs2, eps=0):
    return all(varLess(vs1[o], vs2[o], eps) for o in vs1)

class MoveState(Hashable):
    def __init__(self, q, objVars):
        self.q = q
        self.objVars = objVars
        Hashable.__init__(self)
    def pt(self):
        return self.q['pr2Base']
    def __str__(self):
        return 'MoveState(%s,%s)'%(self.pt(), self.objVars)
    def desc(self):
        return (self.q, frozenset(self.objVars.items()))
    __repr__ = __str__

# Only do this when the base distance between the two confs is large.
def moveLookPath(pbs, prob, q1, q2):
    if q1['pr2Base'] == q2['pr2Base']:
        return []
    startTime = time.time()
    # Get confs with hands retracted
    retractArm1 = findRetractArmConf(pbs, prob, q1, q2)
    if not retractArm1:
        print 'Failed to retract arm'
        pdb.set_trace()
        return None
    retractArm2 = retractArm1.set('pr2Base', q2['pr2Base'])
    retract1 = findRetractBaseConf(pbs, prob, retractArm1) or retractArm1
    retract2 = findRetractBaseConf(pbs, prob, retractArm2) or retractArm2
    # Find path in belief space
    pBs = pbs.getPlacedObjBs().values()
    initialState = MoveState(retract1,
                             {pB.obj:pB.poseD.var for pB in pBs})
    goalState = MoveState(retract2, None)
    goalPt = goalState.pt()
    print 'goalPt', goalPt
    odoError = pbs.domainProbs.odoError
    obsVar = pbs.domainProbs.obsVarTuple
    # The 'move' actions are incremental, 'moveGoal' is an absolute displacement
    thStep = hu.angleDiff(goalPt[-1], initialState.pt()[-1])
    actionList = \
               [('move', -delta, 0., 0.), ('move', delta, 0., 0.),
                ('move', 0., -delta, 0.), ('move', 0., delta, 0.),
                # ('move', 0., 0., -thStep),
                ('move', 0., 0., thStep), # change orientation.
                ('moveGoal', ) + tuple(goalPt)] + \
                [('look', pB.obj) for pB in pBs]
    workspace = pbs.getWorld().workspace
    bestStateVars = {}                  # {statePt:bestVars, ...}
    def actions(state):
        return actionList
    def successor(state, action):
        nstate, cost = resultState(state, action, odoError, obsVar) # (state, cost)
        if action[0] == 'moveGoal':
            p0 = state.pt(); p1 = nstate.pt()
            directDist = ((p0[0]-p1[0])**2 + (p0[1]-p1[1])**2)**0.5
            if directDist < maxDirect:
                return nstate, cost
            else:
                return None
        pt = tuple(nstate.pt())
        if any(pt[i] < workspace[0,i] or pt[i] > workspace[1,i] for i in (0,1)):
            return None
        vs = nstate.objVars
        if pt in bestStateVars:
            # if none of the previous entries subsume this, then keep it.
            for entry in bestStateVars[pt]:
                if varSubsumes(entry, vs, 1.0e-4):
                    return None
            if vs in bestStateVars[pt]:
                pdb.set_trace()
            bestStateVars[pt].append(vs)
        else:
            bestStateVars[pt] = [vs]
        return nstate, cost
    def heuristic(state):
        h = ptDist(state.pt(), goalPt)
        if debug('moveLookPath'): print 'h=', h, state.pt()
        return h
    def goalTest(state):
        return ptDist(state.pt(), goalPt) <= 0.0001
    def expandF(node):
        newPBS = pbsUpdateVar(pbs, node.state.objVars)
        if debug('moveLookPath'):
            print node.action
            newPBS.draw(prob, 'W')
            goalState.q.draw('W', 'pink')
            node.state.q.draw('W')
            wm.getWindow('W').update()
    debugH = False
    if debug('h'):
        glob.debugOn.remove('h')
        debugH = True
    path, cost = search(initialState, goalTest, actions, successor,
                        heuristic,
                        # Check legality on expansion and possibly modify node
                        checkExpandF = \
                        lambda node: feasibleAction(pbs, prob, node, odoError, obsVar,
                                                    first = (node.state == initialState)),
                        expandF = expandF,
                        greedy = 0.75,
                        printFinal = debug('moveLookPath'),
                        verbose = debug('moveLookPath'),
                        fail = False)
    print 'moveLookPath time =', time.time() - startTime
    if debugH:
        glob.debugOn.append('h')

    if not path:
        pdb.set_trace()

    path1, viol1 = canReachHome(pbs, q1, prob, Violations(),
                                homeConf=retract1, optimize=True)
    assert viol1 and viol1.weight() == 0
    path2, viol2 = canReachHome(pbs, retract2, prob, Violations(),
                                homeConf=q2, optimize=True)
    assert viol2 and viol2.weight() == 0

    # a list of (action, conf
    return [(None, q) for q in path1] + \
           [(act, state.q) for (act, state) in path] + \
           [(None, q) for q in path2]

# node is a SearchNode from ucSearchPQ.py
def feasibleAction(pbs, prob, node, odoError, obsVar, first=False):
    if node.parent is None:
        return node
    # The node.state is an absolute state obtained from resultState
    ostate = node.parent.state          # s
    act = node.action                   # a
    state = node.state                  # s'
    actType = act[0]
    if actType in ('move', 'moveGoal'):
        newState, newCost = feasibleMove(pbs, prob, ostate, act, state,
                                         odoError, first)
    elif actType == 'look':
        newState, newCost = feasibleLook(pbs, prob, ostate, act, state,
                                         obsVar, first)
    else: assert False, 'Illegal action: %s'%(act,)
    if not newState:
        if debug('moveLookPath'):
            print ostate.pt(), '+', act, '->', None
        return None                     # infeasible action
    else:
        if node.state != newState:      # update the node...
            node.state = newState
            node.cost += newCost - node.actionCost # update total cost
            node.actionCost = newCost
        if debug('moveLookPath'):
            print ostate.pt(), '+', act, '->', node.cost, newState
    return node

def feasibleMove(pbs, prob, ostate, act, state, odoError, first=False):
   newPBS = pbsUpdateVar(pbs, (ostate if first else state).objVars)
   q_f = stopConf(newPBS, prob, state.q, ostate.q, ['pr2Base'])
   if q_f == ostate.q:                  # couldn't move
       return None, None
   else:
       # updated state and cost for an absolute move, as far as we could go
       newState, newCost = resultStateMoveGoal(ostate, ('moveGoal',) + tuple(q_f['pr2Base']), odoError)
       newvs = newState.objVars
       oldvs = state.objVars
       for o in newvs:
           if any(thisV+1.0e-8 < prevV for (thisV, prevV) in zip(newvs[o],oldvs[o])):
               raw_input('Whoa')
       return newState, newCost

def feasibleLook(pbs, prob, ostate, act, state, obsVar, first):
    (_, obj) = act
    shWorld = pbs.getShadowWorld(prob)
    obstacles = [s for s in shWorld.getObjectShapes() if \
                 s.name() != obj ]  + [shWorld.robotPlace]
    vis, occl = visible(shWorld, ostate.q,
                        shWorld.objectShapes[obj],
                        obstacles, prob, moveHead=True,
                        fixed=[shWorld.robotPlace])
    if vis or len(occl) == 0:
        return state, lookCost
    else:
        return None, None

def eqChains(conf1, conf2, moveChains):
    return all([conf1.conf[c]==conf2.conf[c] for c in moveChains])

def stopConf(pbs, prob, q_f, q_i, moveChains,
             stepSize = glob.rrtInterpolateStepSize,
             maxSteps = 1000):
    q_init = q_i
    if eqChains(q_f, q_i, moveChains): return q_init
    step = 0
    for step in xrange(maxSteps):
        q_new = q_i.robot.stepAlongLine(q_f, q_i, stepSize,
                                        moveChains = moveChains)
        viol = pbs.confViolations(q_new, prob)
        if viol and viol.weight() == 0: # no collisions
            if eqChains(q_f, q_new, moveChains):
                return q_f
            q_i = q_new
        else:                           # a collision
            return q_init               # fail -  could be q_i
    return q_init                       # fail -  could be q_i

def pbsUpdateVar(pbs, objVars):
    newPBS = pbs.copy()
    newPBS.objectBs = {pB.obj:(fix, pB.modifyPoseD(var=objVars.get(pB.obj, pB.poseD.var))) \
                       for (fix,pB) in pbs.objectBs.values()}
    return newPBS

# There are the nominal state and cost for the actions, the actual
# moves can be modified when checked for feasibility

def resultState(state, act, odoError, obsVar):
    actType = act[0]
    if actType == 'move': return resultStateMove(state, act, odoError)
    elif actType == 'moveGoal': return resultStateMoveGoal(state, act, odoError)
    elif actType == 'look': return resultStateLook(state, act, obsVar)
    else: assert False, 'Illegal action: %s'%(act,)

def resultStateMove(state, act, odoError):
    (_, dx, dy, dth) = act
    cost = (dx**2 + dy**2 + (thConvert*dth)**2)**0.5
    return (newMoveState(state, dx, dy, dth, odoError),
            cost)

def resultStateMoveGoal(state, act, odoError):
    (_, gx, gy, gth) = act
    px, py, pth = state.pt()
    dx, dy, dth = (gx-px, gy-py, hu.angleDiff(gth,pth))
    cost = (dx**2 + dy**2 + (thConvert*dth)**2)**0.5
    return (newMoveState(state, dx, dy, dth, odoError),
            cost)

def newMoveState(state, dx, dy, dth, odoError):
    # dx, dy is xy displacement along move
    # dth is the angular displacement along move
    increaseFactor = 3.
    dxdy = (dx**2 + dy**2)**0.5

    # Displacement should affect the angular variance...
    
    odoVar = ((dxdy * increaseFactor * odoError[0])**2,
              (dxdy * increaseFactor * odoError[1])**2,
              0.0,
              (abs(dth) * increaseFactor * odoError[3])**2)
    return MoveState(state.q.set('pr2Base', [a+b for (a,b) in zip(state.pt(), (dx,dy,dth))]),
                     {o:tuple([a + b for (a, b) in zip(oldVar, odoVar)]) \
                      for (o,oldVar) in state.objVars.iteritems()})
    
def resultStateLook(state, act, obsVar):
    (_, obj) = act
    oldVar = state.objVars[obj]
    newVar = tuple([(a * b) / (a + b) for (a, b) in zip(oldVar,obsVar)])
    newState = MoveState(state.q, state.objVars.copy())
    newState.objVars[obj] = newVar
    return newState, lookCost

### Add constraint on distance of hands from body!!

# Returns conf that avoids collisions and start and goal, keeps a view
# cone clear and tries to minimize the 
def findRetractArmConf(pbs, prob, q1, q2, maxIter = 50):
    collides = pbs.getRoadMap().checkRobotCollision
    robot = pbs.getRobot()
    shWorld = pbs.getShadowWorld(prob)
    attached = shWorld.attached
    shape1 = q1.handWorkspace()
    # obstacles at q1
    avoid1 = shWorld.objectShapes.values()
    if q2['pr2Base'] != q1['pr2Base']:
        # obstacles at q2, placed on top of q1
        tr = q1.basePose().compose(q2.basePose().inverse())
        avoid2 = [s.applyTrans(tr) for s in avoid1]
    else:
        avoid2 = []
    avoid = Shape(avoid1 + [viewCone(q1,shape1)] + avoid2, None)
    if debug('retract'):
        avoid.draw('W', 'red')
        q1.draw('W', 'blue', attached=attached)
        q2.draw('W', 'pink', attached=attached)
        debugMsg('retract', 'Retract obstacles')
    conf = q1
    for h in ['left', 'right']:     # try both hands
        chainName = robot.armChainNames[h]
        armChains = [chainName, robot.gripperChainNames[h]]
        if not (collides(conf, avoid, attached=attached, selectedChains=armChains) \
               if glob.useCC else avoid.collides(conf.armShape(h,attached))):
            continue
        if debug('retract'):
            print 'retract collision with', h, 'arm', conf['pr2Base']
        path, viol = \
              planRobotGoalPath(pbs, prob, conf,
                                lambda c: not (collides(c, avoid, attached=attached, selectedChains=armChains) \
                                               if glob.useCC else avoid.collides(c.armShape(h,attached))),
                                None, [chainName], maxIter = maxIter)
        if debug('retract'):
            pbs.draw(prob, 'W')
            if path:
                for c in path: c.draw('W', 'blue', attached=attached)
                path[-1].draw('W', 'orange', attached=attached)
                avoid.draw('W', 'green')
                debugMsg('canView', 'Retract arm')
            else:
                pbs.draw(prob, 'W')
                conf.draw('W', attached=attached)
                avoid.draw('W', 'red')
                raw_input('retract - no path')
        if path:
            conf = path[-1]
        else:
            print 'Failed to retract arm=', h
            pdb.set_trace()
            return None
    return conf

maxRetractAttempts = 10
def findRetractBaseConf(pbs, prob, conf, maxIter=10):
    # Inflate the objects and move the base
    newBS = pbs.copy()
    newBS.conf = (False, conf)
    for obj in newBS.objectBs:
        fix, objB = newBS.objectBs[obj]
        inflatedVar = (0.05**2, 0.05**2, 0.05**2, 0.1**2)
        newBS.updatePlaceB(objB.modifyPoseD(var=inflatedVar))

    def anyColl(c):
        v = newBS.confViolations(c, prob)
        return not (v is None or v.weight() > 0)

    bestRetractConf = None
    bestRetractDist = float('inf')
    conf.draw('W', 'blue')
    for attempt in range(maxRetractAttempts):
        path, viol = \
              planRobotGoalPath(newBS, prob, conf, anyColl,
                                None, ['pr2Base'], maxIter = maxIter)
        path[-1].draw('W', 'pink') if path else None
        if path:
            retractConf = path[-1]
            dist = ptDist(conf['pr2Base'], retractConf['pr2Base'])
            print 'retractDist', dist
            if dist == 0:
                return retractConf
            elif dist < bestRetractDist:
                bestRetractConf = retractConf
                bestRetractDist = dist
    bestRetractConf.draw('W', 'green')
    raw_input('retract')
    return bestRetractConf

