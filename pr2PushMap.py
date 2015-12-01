
from shapes import Shape
from geom import bboxOverlap

"""
A simple xy planner to push from a specified target location towards a
specified final location.  In the away mode, then we must reach the
initial state.  Note that all positions need to remain inside a given
region shape.

A (small) set of pushing directions are allowed.  Note that these are
one-way, pushing opposite to a face normal.

A set of C-obstacles are defined by the pushed object and the hand at
the push pose.  The tangents of the obstacles for each of the push
directions (and rays through the start and target define a sort of
grid that is searched for a path.

Turns in a path are expensive because they require breaking contact
and moving to a new contact.

Each path segment can be checked for feasibility (and mapped to robot)
separately - canReachHome at beginning and end and then kinematics and
collision checks along the segment - without moving the base.  If
kinematics fails partway, we can break the segment and check
separately for feasibility of the rest.  That way, we may be able to
move the base to make the kinematics feasible.
"""

# Used for searching for an "object" push sequence (ignoring the robot)
class ObjPushState:
    def __init__(self, pt, dirx):
        self.pt = pt
        self.dirx = dirx
    def displace(self, stepSizes):
        displacement = [self.dirx[i]*stepSizes[i] for i in (0,1)]
        npt = tuple([self.step[i] + displacement[i] for i in (0,1)])
        return ObjPushState(npt, self.dirx)
    def turn(self, ndirx):
        return ObjPushState(self.pt, ndirx)
    def __str__(self):
        return "ObjPushState(%s, %s)"%(pt, dirx)
    __repr__ = __str__

class PushEnv:
    def __init__(self, pbs, prob, regShape, dirxs, steps):
        self.pbs = pbs
        self.prob = prob
        self.dirxs = dirxs
        self.steps = steps
        self.regShape = regShape
        self.turnCost = 10
    def successors(self, pstate):       # returns (nstate, cost)
        slist = []
        for dirx in self.dirxs:
            if dirx == pstate.dirx:     # forward
                nstate = pstate.displace(penv.steps)
                if self.legalStep(pstate, nstate):
                    slist.append((nstate, 1.))
            else:                       # turn
                nstate = pstate.turn(dirx)
                if self.legalTurn(pstate, nstate):
                    slist.append((nstate, self.turnCost))
        return slist
    def legalStep(self, pstate, nstate):
        pass
    def legalTurn(self, pstate, nstate):
        pass

# Returns a list of dictionaries of C-obstacles, one for each
# potential push.  We don't distinguish permanents from not.
def CObstacles(pbs, prob, potentialPushes, pB, hand, supportRegShape):
    shWorld = pbs.getShadowWorld(prob)

    newPB = pB.modifyPoseD(var=4*(0.02**2,))
    newPB.delta = delta=4*(0.001,)
    objShape = pB.makeShadow(pbs, prob)
    (x,y,_,_) = newPB.poseD.mode().pose().xyztTuple()
    centeringOffset = hu.Pose(-x,-y,0.,0.)

    xyObst = {name: obst.xyPrim() \
              for (name, obst) in shWorld.objectShapes.iteritems()}

    CObsts = []
    for (gB, width, direction) in potentialContacts:
        wrist = objectGraspFrame(pbs, gB, newPB, hand)
        # the gripper at push pose relative to pB
        gripShape = gripperPlace(pbs.getConf(), hand, wrist)
        # the combinaed gripper and object, origin is that of object
        objGripShape = Shape([gripShape, objShape], \
                             objShape.origin(),
                             name='gripper').xyPrim() # note xyPrim
        # center the shape at the origin (only in x,y translation)
        objGripShapeCtr = objGripShape.applyTrans(centeringOffset)
        bb = bboxExtendXY(supportRegShape.bbox(), objGripShapeCtr.bbox())
        CO = {}
        CObsts.append(CO)
        for (name, obst) in xyObst.iteritems():
            if bboxOverlap(bb, obst.bbox()):
                c = xyCO(objGripShape, obst)
                CO[name] = c
                c.properties['name'] = name
            else:
                CO[name] = None

    return CObsts
        
def bboxExtendXY(bbB, bbA):
    bb = np.array([bbB[0] - bbA[0], bbB[1] + bbA[1]])
    bb[0,2] = bbB[0,2]
    bb[1,2] = bbB[1,2]
    return bb

# Given obst shape and direction vector [x,y,0], returns two line
# equations [-y,x,0,-dmin] and [-y,x,0,-dmax]
def tangents(obst, dir, eps = 0.):
    x = dir[0]; y = dir[1]
    mag = (x**2 + y**2)**0.5
    x /= mag; y /= mag
    verts = obst.getVertices()
    ext = [float('inf'), -float('inf')]
    for i in xrange(verts.shape[1]):
        d = -verts[0,i]*y + verts[1,i]*x
        ext[0] = min(ext[0], d)
        ext[1] = max(ext[1], d)
    return np.array([-y,x,0,-(ext[0]-eps)]), np.array([-y,x,0,-(ext[1]+eps)])
    
def edgeCross((A,B), (C,D)):
    """
Suppose the two segments have endpoints A,B and C,D. The numerically
robust way to determine intersection is to check the sign of the four
determinants:

| Ax-Cx  Bx-Cx |    | Ax-Dx  Bx-Dx |
| Ay-Cy  By-Cy |    | Ay-Dy  By-Dy |

| Cx-Ax  Dx-Ax |    | Cx-Bx  Dx-Bx |
| Cy-Ay  Dy-Ay |    | Cy-By  Dy-By |

For intersection, each determinant on the left must have the opposite
sign of the one to the right,
"""
    def det(a,b,c):
        return (a[0]-c[0])*(b[1]-c[1]) - (a[1]-c[1])*(b[0]-c[0])
    return (np.sign(det(A,B,C)) == -np.sign(A,B,D)) and \
           (np.sign(det(C,D,A)) == -np.sign(C,D,B))

def edgeCollides(edge, poly):
    # First, check if completely inside
    (z0, z1) = poly.zRange()
    p = np.array([edge[0][0], edge[0][1], 0.5*(z0+z1), 1.0])
    if np.all(np.dot(poly.planes(),
                     np.resize(p,(4,1)))<=0): return True
    # Check for crossings
    verts = poly.vertices()
    edges = poly.edges()
    face0 = poly.faces()[0]
    for e in edges:
        if edgeCross(edge, (verts[edges[e,0]], verts[edges[e,1]])):
            return True
    return False

def lineIsectDist(pt, dirx, line):
    cos = sum([dirx[i]*line[i] for i in (0,1)])
    if abs(cos) < 1e-6: return None
    return -(sum([line[i]*p[i] for i in (0,1)]) + line[3]) / cos

# tangents is a dictionary {dirx: list of tangent lines}
def nextCrossing(pt, dirx, tangents, goal):
    d = sum((goal[i] - p[i])*dirx[i])
    bestDir = None
    if all(abs(p[i] + dirx[i]*goal[i]) < 1e-4 for i in (0,1)):
        # aligned with goal
        bestDist = d
    else:
        bestDist = None
    for ndirx, tanLines in tangents.iteritems():
        if dirx == ndirx: continue
        for line in tanLines:
            d = lineIsectDist(pt, dirx, line)
            if d and d > 0 and (bestDist is None or d < bestDist):
                bestDist = d
                bestDir = ndirx
    if bestDist:
        return [p[i] + bestDist*dirx[i] for i in (0,1)], bestDir
    else:
        return None, None




def pushGenPaths(pbs, prob, potentialContacts, targetPB, curPB,
                 hand, base, prim, supportRegShape, away=False):
    tag = 'pushGen'
    # Define a collection of potential (straight line) pushes, defined
    # by the direction and the "grasp", the position of the hand
    # relative to the object.
    potentialPushes = []                # (graspB, width, direction)
    for (vertical, contactFrame, width) in potentialContacts:
        # construct a graspB corresponding to the push hand pose,
        # determined by the contact frame
        graspB = graspBForContactFrame(pbs, prob, contactFrame,
                                       0.0,  targetPB, hand, vertical)
        # This is negative z axis of face
        direction = -contactFrame.matrix[:3,2].reshape(3)
        direction[2] = 0.0            # we want z component exactly 0.
        potentialPushes.append((graspB, width, direction))
    
    for prPath in searchPushPath(pbs, prob, potentialPushes, targetPB, curPB,
                                 hand, base, prim, supportRegShape, away=away):
        # ptPath is a list of PushResponse instances.
        # yield the first segment of the path
        prPath[0].debug(tag)            # do debugging display
        yield prPath[0]

def searchPushPath(pbs, prob, potentialPushes, targetPB, curPB,
                   hand, base, prim, supportRegShape, away=False):
    curPose = curPB.poseD.mode().pose()
    targetPose = targetPB.poseD.mode().pose()
    


    # sort contacts and compute a distance when applicable, entries
    # are: (distance, vertical, contactFrame, width)
    # Sort contacts by nearness to current pose of object
    sortedContacts = sortPushContacts(potentialContacts, targetPB.poseD.mode(),
                                      curPose)
    if not sortPushContacts:
        tr(tag, '=> No sorted contacts')
    # Now we have frames for contact with object, we have to generate
    # potential answers following the face normal in the given
    # direction.
    for (dist, vertical, contactFrame, width) in sortedContacts:
        if debug(tag): print 'dist=', dist
        # construct a graspB corresponding to the push hand pose,
        # determined by the contact frame
        graspB = graspBForContactFrame(pbs, prob, contactFrame,
                                       0.0,  targetPB, hand, vertical)
        # This is negative z axis of face
        direction = -contactFrame.matrix[:3,2].reshape(3)
        direction[2] = 0.0            # we want z component exactly 0.
        prePoseOffset = hu.Pose(*(dist*direction).tolist()+[0.0])
        # initial pose for object along direction
        prePose = prePoseOffset.compose(targetPB.poseD.mode())
        if debug(tag):
            pbs.draw(prob, 'W')
            shape.draw('W', 'blue'); prim.draw('W', 'green')
            print 'vertical', vertical, 'dist', dist
            drawFrame(contactFrame)
            raw_input('graspDesc frame')
        pushPaths = []                  # for different base positions
        # Generate confs to place the hand at graspB
        # Try going directly to goal and along the direction of face
        # The direct route only works for vertical pushing...
        for direct in (True, False) if (useDirectPush and vertical and curPose) else (False,):
            count = 0                       # how many tries
            doneCount = 0                   # how many went all the way
            initPose = curPose.pose() if direct else prePose.pose()
            appPoseOffset = hu.Pose(*(glob.pushBuffer*direction).tolist()+[0.0])
            appPose = appPoseOffset.compose(initPose).pose()
            if debug(tag):
                print 'Calling potentialConfs, with direct =', direct, 'appPose', appPose
            for ans in potentialConfs(pbs, prob, targetPB, appPose, graspB, hand, base):
                if not ans:
                    tr(tag+'_kin', 'potential grasp conf is empy')
                    continue
                (prec, postc) = ans         # conf, approach, violations
                preConf = gripSet(prec, hand, 2*width) # open fingers
                pushConf = gripSet(postc, hand, 2*width) # open fingers
                if debug(tag+'_kin'):
                    pushConf.draw('W', 'orange')
                    raw_input('Candidate conf')
                count += 1
                pathAndViols, reason = pushPath(pbs, prob, graspB, targetPB, pushConf,
                                                initPose, preConf, supportRegShape, hand)
                if reason == 'done':
                    pushPaths.append((pathAndViols, reason))
                    doneCount +=1 
                    if doneCount >= maxDone: break
                tr(tag, 'pushPath reason = %s, path len = %d'%(reason, len(pathAndViols)))
                if count > maxPushPaths: break
            pose1 = targetPB.poseD.mode()
            pose2 = appPose
            if debug('pushFail'):
                if count == 0:
                    print 'No push', direction[:2], 'between', pose1, pose2, 'with', hand, 'vert', vertical
                    debugMsg('pushFail', 'Could not find conf for push along %s'%direction[:2])
                else:
                    print 'Found conf for push', direction[:2], 'between', pose1, pose2 
        # Sort the push paths by violations
        sorted = sortedPushPaths(pushPaths, curPose)
        for i in range(min(len(sorted), maxDone)):
            pp = sorted[i]
            ppre, cpre, ppost, cpost = getPrePost(pp)
            if not ppre or not ppost: continue
            crev = reverseConf(pp, hand)
            if debug(tag):
                robot = cpre.robot
                handName = robot.armChainNames[hand]
                print 'pre pose\n', ppre.matrix
                for (name, conf) in (('pre', cpre), ('post', cpost), ('rev', crev)):
                    print name, 'conf tool'
                    print conf.cartConf()[handName].compose(robot.toolOffsetX[hand]).matrix
            tr(tag, 'pre conf (blue), post conf (pink), rev conf (green)',
               draw=[(pbs, prob, 'W'),
                     (cpre, 'W', 'blue'), (cpost, 'W', 'pink'),
                     (crev, 'W', 'green')], snap=['W'])
            viol = Violations()
            for (c,v,p) in pathAndViols:
                viol.update(v)
            ans = PushResponse(targetPB.modifyPoseD(ppre.pose()),
                               targetPB.modifyPoseD(ppost.pose()),
                               cpre, cpost, crev, viol, hand,
                               targetPB.poseD.var, targetPB.delta)
            if debug('pushFail'):
                print 'Yield push', ppre.pose(), '->', ppost.pose()
            cachePushResponse(ans)
            yield ans
