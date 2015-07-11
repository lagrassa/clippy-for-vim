
def pushGen(args, goalConds, bState, outBindings):
    (obj, hand, poses, support, objV, objDelta, prob) = args
    tag = 'pushGen'
    base = sameBase(goalConds)
    tr(tag, 0, 'obj=%s, base=%s'%(obj, base))
    if goalConds:
        if getConf(goalConds, None):
            tr(tag, 1, '=> conf is already specified, failing')
            return

    # Just placements specified in goal (and excluding obj)
    # placeInGenAway does not do this when calling placeGen
    newBS = bState.pbs.copy()
    newBS = newBS.updateFromGoalPoses(goalConds, updateConf=False)
    newBS = newBS.excludeObjs([obj])
    world = newBS.getWorld()

    if poses == '*' or isVar(poses) or support == '*' or isVar(support):
        tr(tag, 2, 'Unspecified pose')
        if base:
            # Don't try to keep the same base, if we're trying to push the object away.
            tr(tag, 1, '=> unspecified pose with same base constraint, failing')
            return
        assert not isVar(hand)
        # v is viol
        for ans,v in pushAway((obj, objDelta, prob),
                              goalConds, newBS, outBindings):
            (pB1, pB2, c1, ca1, c2, ca2) = ans
            yield ans, v, hand
        return

    if not isinstance(poses[0], (list, tuple, frozenset)):
        poses = frozenset([poses])

    def placeBGen():
        for pose in poses:
            yield ObjPlaceB(obj, world.getFaceFrames(obj), support,
                            PoseD(pose, objV), delta=objDelta)
    placeBs = Memoizer('placeBGen_placeGen', placeBGen())

    # Figure out whether one hand or the other is required;  if not, do round robin
    leftGen = pushGenTop((obj, placeBs, 'left', base, prob),
                         goalConds, pbs, newBS, outBindings)
    rightGen = pushGenTop((obj, placeBs, 'right', base, prob),
                          goalConds, pbs, newBS, outBindings)
    
    for ans in chooseHandGen(newBS, goalConds, obj, hand, leftGen, rightGen):
        yield ans
    
def pushGenTop(args, goalConds, pbsOrig, newBS, outBindings, away=False):
    (obj, placeBs, hand, base, prob) = args

    startTime = time.clock()
    tag = 'pushGen'
    tr(tag, 0, '(%s,%s) h=%s'%(obj,hand, glob.inHeuristic))
    tr(tag, 2, 
       zip(('obj', 'placeBs', 'hand', 'prob'), args),
       ('goalConds', goalConds),
       ('moveObjBs', newBS.moveObjBs),
       ('fixObjBs', newBS.fixObjBs),
       ('held', (newBS.held['left'].mode(),
                 newBS.held['right'].mode(),
                 newBS.graspB['left'],
                 newBS.graspB['right'])))
    if obj == 'none' or not placeBs:
        tr(tag, 1, '=> obj is none or no placeB, failing')
        return
    if goalConds:
        if getConf(goalConds, None) and not away:
            tr(tag, 1, '=> goal conf specified and not away, failing')
            return
        for (h, o) in getHolding(goalConds):
            if h == hand:
                tr(tag, 1, '=> Hand=%s is already Holding, failing'%hand)
                return
    conf = None
    confAppr = None
    tr(tag, 2, 'Goal conditions', draw=[(newBS, prob, 'W')], snap=['W'])
    gen = pushGenAux(pbsOrig, newBS, obj, confAppr, conf, placeBs.copy(),
                      hand, base, prob)

    for ans in gen:
        tr(tag, 1, str(ans) +' (t=%s)'%(time.clock()-startTime))
        yield ans

