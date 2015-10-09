
# Can cause non-zero heuristic at goal state, because it is trying to
# achieve a rounded-up value
heuristicPrecision = 1000.0

def canonicalizeUp(f, prec = heuristicPrecision):

    def roundUp(x, prec): return x

    newF = f.copy()
    if f.predicate == 'Bd':
        newF.args[2] = roundUp(f.args[2], prec)
        if str(newF.args[1])[0] == '?': newF.args[1] = '?'
    elif f.predicate == 'B':
        # round prob up; round variances up; round deltas up
        newF.args[2] = tuple([roundUp(v, prec) for v in f.args[2]])
        newF.args[3] = tuple([roundUp(v, prec) for v in f.args[3]])
        newF.args[4] = roundUp(f.args[4], prec)
        if str(newF.args[1])[0] == '?': newF.args[1] = '?'
    newF.update()
    return newF

def canonicalizeUp(f, prec = 0):
    return f



'''
place = Operator(\
        'Place',
        ['Obj', 'Hand', 'OtherHand',
         'Region', 'PoseFace', 'Pose', 'PoseVar', 'RealPoseVar', 'PoseDelta',
         'GraspFace', 'GraspMu', 'GraspVar', 'GraspDelta',
         'OObj', 'OFace', 'OGraspMu', 'OGraspVar', 'OGraspDelta',         
         'PreConf', 'ConfDelta', 'PlaceConf',
         'PR1', 'PR2', 'PR3', 'PR4', 'P1', 'P2', 'P3'],
        # Pre
        {0 : {Graspable(['Obj'], True)},
         1 : {Bd([CanPickPlace(['PreConf', 'PlaceConf', 'Hand', 'Obj', 'Pose',
                               'RealPoseVar', 'PoseDelta', 'PoseFace',
                               'GraspFace', 'GraspMu', 'GraspVar', 'GraspDelta',
                               'OObj', 'OFace', 'OGraspMu', 'OGraspVar', 
                               'OGraspDelta', []]), True, 'P1'], True)},
         2 : {Bd([Holding(['Hand']), 'Obj', 'P2'], True),
              Bd([GraspFace(['Obj', 'Hand']), 'GraspFace', 'P2'], True),
              B([Grasp(['Obj', 'Hand', 'GraspFace']),
                 'GraspMu', 'GraspVar', 'GraspDelta', 'P2'], True),
              # Bookkeeping for other hand
              Bd([Holding(['OtherHand']), 'OObj', 'P3'], True),
              Bd([GraspFace(['OObj', 'OtherHand']), 'OFace', 'P3'], True),
              B([Grasp(['OObj', 'OtherHand', 'OFace']),
                       'OGraspMu', 'OGraspVar', 'OGraspDelta', 'P3'], True)},
         3 : {Conf(['PreConf', 'ConfDelta'], True)}
        },
        # Results
        [({Bd([In(['Obj', 'Region']), True, 'PR4'], True)}, {}),
         ({Bd([SupportFace(['Obj']), 'PoseFace', 'PR1'], True),
           B([Pose(['Obj', 'PoseFace']), 'Pose', 'PoseVar', 'PoseDelta','PR2'],
                 True)},{}),
         ({Bd([Holding(['Hand']), 'none', 'PR3'], True)}, {})],
        # Functions
        functions = [\
            # Get both hands and object!
            Function(['Obj', 'Hand', 'OtherHand'], ['Obj', 'Hand'],
                     getObjAndHands, 'getObjAndHands'),
            # Either Obj is bound (because we're trying to place it) or
            # Hand is bound (because we're trying to make it empty)
            # If Obj is not bound then: get it from the start state;
            #  also, let region be awayRegion
            Function(['Region'], ['Region', 'Pose'], awayRegionIfNecessary,
                                   'awayRegionIfNecessary'),
            
            # Be sure all result probs are bound.  At least one will be.
            Function(['PR1', 'PR2', 'PR3', 'PR4'],
                     ['PR1', 'PR2', 'PR3', 'PR4'], minP,'minP'),

            # Compute precond probs.  Assume that crash is observable.
            # So, really, just move the obj holding prob forward into
            # the result.  Use canned probs for the other ones.
            Function(['P2'], ['PR1', 'PR2', 'PR3', 'PR4'], 
                     regressProb(1, 'placeFailProb'), 'regressProb1'),
            Function(['P1', 'P3'], [[canPPProb, otherHandProb]],
                     assign, 'assign'),

            # PoseVar = GraspVar + PlaceVar,
            # GraspVar = min(maxGraspVar, PoseVar - PlaceVar)
            Function(['GraspVar'], ['PoseVar'], placeGraspVar, 'placeGraspVar'),

            # Real pose var might be much less than pose var if the
            # original pos var was very large
            # RealPoseVar = GraspVar + PlaceVar
            Function(['RealPoseVar'],
                     ['GraspVar'], realPoseVar, 'realPoseVar'),
            
            # In case PoseDelta isn't defined
            Function(['PoseDelta'],[],lambda a,g,s,v: [[defaultPoseDelta]],
                     'defaultPoseDelta'),
            # Divide delta evenly
            Function(['ConfDelta', 'GraspDelta'], ['PoseDelta'],
                      halveVariance, 'halveVar'),

            # Values for what is in the other hand
            Function(['OObj', 'OFace', 'OGraspMu', 'OGraspVar', 'OGraspDelta'],
                       ['OtherHand', 'Obj'], genGraspStuffHand,
                       'genGraspStuffHand'),

            # Not modeling the fact that the object's shadow should
            # grow a bit as we move to pick it.   Build that into pickGen.
            Function(['Pose', 'PoseFace', 'GraspMu', 'GraspFace', 'GraspVar',
                      'PlaceConf', 'PreConf'],
                     ['Obj', 'Region','Pose', 'PoseFace', 'PoseVar', 'GraspVar',
                      'PoseDelta', 'GraspDelta', 'ConfDelta', 'Hand',
                     probForGenerators],
                     placeInGen, 'placeInGen')

            ],
        cost = placeCostFun, 
        f = placeBProgress,
        prim = placePrim,
        argsToPrint = [0, 1, 3, 4, 5],
        ignorableArgs = range(2, 27))
'''        

place = Operator(\
        'Place',
        ['Obj1', 'Hand', 'OtherHand',
         'Region', 'PoseFace', 'Pose', 'PoseVar', 'RealPoseVar',
         'PoseDelta',
         'Obj2', 'ObjPose2', 'PoseVar2', 'PoseDelta2', 'PoseFace2',
         'TotalVar', 'TotalDelta',
         'GraspFace', 'GraspMu', 'GraspVar', 'GraspDelta',
         'OObj', 'OFace', 'OGraspMu', 'OGraspVar', 'OGraspDelta',         
         'PreConf', 'ConfDelta', 'PlaceConf',
         'PR1', 'PR2', 'PR3', 'PR4', 'P1', 'P2', 'P3'],
        # Pre
        {0 : {Graspable(['Obj1'], True)},
         1 : {Bd([CanPickPlace(['PreConf', 'PlaceConf', 'Hand', 'Obj1', 'Pose',
                               'RealPoseVar', 'PoseDelta', 'PoseFace',
                               'GraspFace', 'GraspMu', 'GraspVar', 'GraspDelta',
                               'OObj', 'OFace', 'OGraspMu', 'OGraspVar', 
                               'OGraspDelta', []]), True, 'P1'], True)},
         2 : {Bd([Holding(['Hand']), 'Obj1', 'P2'], True),
              Bd([GraspFace(['Obj1', 'Hand']), 'GraspFace', 'P2'], True),
              B([Grasp(['Obj1', 'Hand', 'GraspFace']),
                 'GraspMu', 'GraspVar', 'GraspDelta', 'P2'], True),
              # Bookkeeping for other hand
              Bd([Holding(['OtherHand']), 'OObj', 'P3'], True),
              Bd([GraspFace(['OObj', 'OtherHand']), 'OFace', 'P3'], True),
              B([Grasp(['OObj', 'OtherHand', 'OFace']),
                       'OGraspMu', 'OGraspVar', 'OGraspDelta', 'P3'], True)},
         3 : {Conf(['PreConf', 'ConfDelta'], True)}
        },
        # Results
        [({Bd([In(['Obj1', 'Region']), True, 'PR4'], True)},
          {3 : {B([Pose(['Obj2', 'PoseFace2']), 'ObjPose2', 'PoseVar2',
                               'PoseDelta2', 'P1'], True),
                Bd([SupportFace(['Obj2']), 'PoseFace2', 'P1'], True),}}),
         ({Bd([SupportFace(['Obj1']), 'PoseFace', 'PR1'], True),
           # B([RelPose(['Obj1', 'PoseFace1', 'Obj2', 'PoseFace2']),
           #    'RelPose', 'RelPoseVar', 'RelPoseDelta','PR2'], True)
           B([Pose(['Obj1', 'PoseFace']),
              'Pose', 'PoseVar', 'PoseDelta','PR2'], True)},{}),
         ({Bd([Holding(['Hand']), 'none', 'PR3'], True)}, {})],
        # Functions
        functions = [\
            # Get both hands and object!
            Function(['Obj1', 'Hand', 'OtherHand'], ['Obj1', 'Hand'],
                     getObjAndHands, 'getObjAndHands'),
            # Either Obj is bound (because we're trying to place it) or
            # Hand is bound (because we're trying to make it empty)
            # If Obj is not bound then: get it from the start state;
            #  also, let region be awayRegion
            Function(['Region'], ['Region', 'Pose'], awayRegionIfNecessary,
                                   'awayRegionIfNecessary'),
            
            # Object region is defined wrto
            Function(['Obj2'], ['Region'], regionParent, 'regionParent'),
            
            # Be sure all result probs are bound.  At least one will be.
            Function(['PR1', 'PR2', 'PR3', 'PR4'],
                     ['PR1', 'PR2', 'PR3', 'PR4'], minP,'minP'),

            # Compute precond probs.  Assume that crash is observable.
            # So, really, just move the obj holding prob forward into
            # the result.  Use canned probs for the other ones.
            Function(['P2'], ['PR1', 'PR2', 'PR3', 'PR4'], 
                     regressProb(1, 'placeFailProb'), 'regressProb1'),
            Function(['P1', 'P3'], [[canPPProb, otherHandProb]],
                     assign, 'assign'),

            # Pick a total variance (for placement into the region) if not
            # otherwise specified.
            Function(['TotalVar'], [[(.05**2, .05**2, .02**2, .1**2)]],
                     assign, 'assign'),
            # Same for the pose of the base object.  
            Function(['PoseVar2'], [], obsVar, 'obsVar'),

            # TotalVar = PoseVar2 + GraspVar + PlaceVar,
            # GraspVar = min(maxGraspVar, TotalVar - PlaceVar - PoseVar2)
            Function(['GraspVar'], ['TotalVar', 'PoseVar2'],
                     placeGraspVar2, 'placeGraspVar'),

            # Real pose var might be much less than pose var if the
            # original pos var was very large
            # RealPoseVar = GraspVar + PlaceVar
            Function(['RealPoseVar'],
                     ['GraspVar'], realPoseVar, 'realPoseVar'),

            Function(['PoseFace2', 'ObjPose2'],
                     ['Obj2'], poseInStart, 'poseInStart'),

            # In case PoseDelta isn't defined
            Function(['PoseDelta'],[[defaultPoseDelta]], assign, 'assign'),
            Function(['TotalDelta'],[[defaultTotalDelta]], assign, 'assign'),
            # Divide delta evenly
            Function(['ConfDelta', 'GraspDelta', 'PoseDelta2'],
                     ['TotalDelta'], thirdVariance, 'thirdVar'),

            # Values for what is in the other hand
            Function(['OObj', 'OFace', 'OGraspMu', 'OGraspVar', 'OGraspDelta'],
                       ['OtherHand', 'Obj1'], genGraspStuffHand,
                       'genGraspStuffHand'),

            # Not modeling the fact that the object's shadow should
            # grow a bit as we move to place it.   Build that into placeGen.
            Function(['Pose', 'PoseFace', 'GraspMu', 'GraspFace', 'GraspVar',
                      'PlaceConf', 'PreConf'],
                     ['Obj1', 'Region','Pose', 'PoseFace', 'TotalVar',
                      'GraspVar', 'TotalDelta', 'GraspDelta', 'ConfDelta',
                      'Hand',
                     probForGenerators],
                     placeInGen, 'placeInGen')

            ],
        cost = placeCostFun, 
        f = placeBProgress,
        prim = placePrim,
        argsToPrint = [0, 1, 3, 4, 5],
        ignorableArgs = range(1, 34))

# -----------------------------------------


    # Cached version of the call to minViolPath
    def confReachViolOld(self, targetConf, pbs, prob, initViol=viol0,
                         startConf = None,
                         optimize = False, noViol = False):
        realInitViol = initViol
        initViol = viol0
        initConf = startConf or self.homeConf
        initConfNode = self.addNode(initConf)
        attached = pbs.getShadowWorld(prob).attached
        
        def grasp(hand):
            g = pbs.graspB[hand]
            if g == None or g.obj == 'none':
                return None
            else:
                return g
        def exitWithAns(ans):
            if not optimize:
                if not key in self.confReachCache:
                    self.confReachCache[key] = []
                self.confReachCache[key].append((pbs, prob, ans))
            if ans and ans[0] and ans[2]:
                (viol, cost, nodePath) = ans
                path = self.confPathFromNodePath(nodePath)
                if debug('confReachViol') and (not fbch.inHeuristic or debug('drawInHeuristic')):
                   drawPath(path, viol=viol, attached=attached)
                   newViol = self.checkPath(path, pbs, prob)
                   if newViol.weight() != viol.weight():
                       print 'viol', viol.weight(), viol
                       print 'newViol', newViol.weight(), newViol
                       raw_input('checkPath failed')
                   debugMsg('confReachViol', ('->', (viol, cost, 'path len = %d'%len(path))))
                return (viol.update(realInitViol) if viol else viol, cost, path)
            else:
                if debug('confReachViol'):
                    drawProblem(forceDraw=True)
                    debugMsg('confReachViol', ('->', ans))
                return (None, None, None)

        def drawProblem(forceDraw=False):
            if forceDraw or \
                   (debug('confReachViol') and \
                    (not fbch.inHeuristic  or debug('drawInHeuristic'))):
                pbs.draw(prob, 'W')
                initConf.draw('W', 'blue', attached=attached)
                targetConf.draw('W', 'pink', attached=attached)
                print 'startConf is blue; targetConf is pink'
                raw_input('confReachViol')

        key = (targetConf, initConf, initViol, fbch.inHeuristic)
        # Check the endpoints
        cv = self.confViolations(targetConf, pbs, prob)[0]
        cv = self.confViolations(initConf, pbs, prob, initViol=cv)[0]
        if cv is None or (noViol and cv.weight() > 0):
            if debug('traceCRH'): print '    unreachable conf',
            if debug('confReachViol'):
                print 'targetConf is unreachable'
            return exitWithAns((None, None, None, None))
            
        if not fbch.inHeuristic and initConf in self.approachConfs:
            keya = (targetConf, self.approachConfs[initConf], initViol, fbch.inHeuristic)
            if keya in self.confReachCache:
                if debug('confReachViolCache'): print 'confReachCache approach tentative hit'
                cacheValues = self.confReachCache[keya]
                sortedCacheValues = sorted(cacheValues,
                                           key=lambda v: v[-1][0].weight() if v[-1][0] else v[-1][0])
                ans = bsEntails(pbs, prob, sortedCacheValues, loose=True)
                if ans != None:
                    if debug('traceCRH'): print '    approach cache hit',
                    if debug('confReachViolCache'):
                        debugMsg('confReachViolCache', 'confReachCache approach actual hit')
                    (viol2, cost2, path2, nodePath2) = ans
                    return (viol2.update(cv).update(realInitViol) if viol2 else viol2,
                            cost2,
                            [initConf] + path2)
        if debug('confReachViolCache'):
            debugMsg('confReachViolCache',
                     ('targetConf', targetConf.conf),
                     ('initConf', initConf),
                     ('prob', prob),
                     ('moveObjBs', pbs.moveObjBs),
                     ('fixObjBs', pbs.fixObjBs),
                     ('held', (pbs.held['left'].mode(),
                               pbs.held['right'].mode(),
                               grasp('left'), grasp('right'))),
                     ('initViol', ([x.name() for x in initViol.obstacles],
                                   [x.name() for x in initViol.shadows]) if initViol else None),
                     ('avoidShadow', pbs.avoidShadow))
        if optimize or noViol:
            pass
        elif key in self.confReachCache:
            if debug('confReachViolCache'): print 'confReachCache tentative hit'
            cacheValues = self.confReachCache[key]
            sortedCacheValues = sorted(cacheValues,
                                       key=lambda v: v[-1][0].weight() if v[-1][0] else v[-1][0])
            ans = bsEntails(pbs, prob, sortedCacheValues)
            if ans != None:
                if debug('traceCRH'): print '    actual cache hit',
                if debug('confReachViolCache'):
                    debugMsg('confReachViolCache', 'confReachCache actual hit')
                    print '    returning', ans
                (viol2, cost2, path2, nodePath2) = ans
                return (viol2.update(realInitViol) if viol2 else viol2, cost2, path2) # actual answer
            elif not fbch.inHeuristic:
                for cacheValue in sortedCacheValues:
                    (bs2, p2, avoid2, ans) = cacheValue
                    (viol2, cost2, path2, nodePath2) = ans
                    if viol2:
                        newViol = self.checkNodePath(nodePath2, pbs, prob)
                        # Don't accept unless the new violations don't add to the initial viols
                        if newViol and newViol.weight() <= cv.weight():
                            ans = (newViol.update(realInitViol) if newViol else newViol, cost2, path2,
                                   nodePath2)
                            if debug('traceCRH'): print '    reusing path',
                            if debug('confReachViolCache'):
                                debugMsg('confReachViolCache', 'confReachCache reusing path')
                                print '    returning', ans
                            self.confReachCache[key].append((pbs, prob, ans))
                            return ans[:-1]
        else:
            self.confReachCache[key] = []
            if debug('confReachViolCache'): print 'confReachCache miss'

        drawProblem()
        cvi = initViol.update(cv)
        node = self.addNode(targetConf)
        if initConf == targetConf:
            if debug('traceCRH'): print '    init=target',
            return exitWithAns((cvi, 0, [targetConf], [node]))
        if fbch.inHeuristic:
            ans = cvi, objCollisionCost*len(cvi.obstacles)+shCollisionCost*len(cvi.shadows), \
                  [initConf, targetConf], [initConfNode, node]
            return exitWithAns(ans)
        if debug('traceCRH'): print '    find path',
        if debug('expand'):
            pbs.draw(prob, 'W')
        hsave = fbch.inHeuristic
        gen = self.minViolPathGen([node], pbs, prob,
                                  initConf, cvi,
                                  optimize=optimize, noViol=noViol)
        ans = gen.next()        
        return exitWithAns(ans)

    def confReachViolGen(self, targetConfs, pbs, prob, initViol=viol0,
                         testFn = lambda x: True, goalCostFn = lambda x: 0,
                         startConf = None, draw=False):
        attached = pbs.getShadowWorld(prob).attached
        initConf = startConf or self.homeConf
        batchSize = confReachViolGenBatch
        batch = 0
        while True:
            # Collect the next batach of trialConfs
            batch += 1
            trialConfs = []
            count = 0
            for c in targetConfs:       # targetConfs is a generator
                if self.confViolations(c, pbs, prob)[0] != None:
                    count += 1
                    trialConfs.append(c)
                    if initConf == c and testFn(c):
                        ans = initViol or Violations(), 0, [initConf]
                        yield ans
                        return
                if count == batchSize: break
            if debug('confReachViolGen'):
                print '** Examining batch', batch, 'of', count, 'confs'
            if count == 0:              # no more confs
                if debug('confReachViolGen'):
                    print '** Finished the batches'
                break
            random.shuffle(trialConfs)
            if debug('confReachViolGen') and not fbch.inHeuristic:
                pbs.draw(prob, 'W')
                initConf.draw('W', 'blue', attached=attached)
                for trialConf in trialConfs:
                    trialConf.draw('W', 'pink', attached=attached)
                print 'startConf is blue; targetConfs are pink'
                debugMsg('confReachViolGen', 'Go?')
            
            # keep track of the original conf for the nodes
            nodeConf = dict([(self.addNode(tc), tc) for tc in trialConfs])
            origConf = dict([(nc.conf, c) for (nc, c) in nodeConf.items()])
            nodeTestFn = lambda n: testFn(nodeConf[n])
            goalNodeCostFn = lambda n: goalCostFn(nodeConf[n])
            gen = self.minViolPathGen(nodeConf.keys(),
                                      pbs, prob,
                                      initConf, initViol or Violations(),
                                      nodeTestFn,
                                      goalCostFn=goalNodeCostFn, draw=draw)
            for ans in gen:
                if ans and ans[0] and ans[2]:
                    (viol, cost, path, _) = ans
                    pathOrig = path[:-1] + [origConf[path[-1]]]
                    if debug('confReachViolGen') and not fbch.inHeuristic:
                        drawPath(pathOrig, viol=viol, attached=attached)
                        newViol = self.checkPath(path, pbs, prob)
                        if newViol.weight() != viol.weight():
                            print 'viol', viol
                            print 'newViol', newViol
                            raw_input('checkPath failed')
                        debugMsg('confReachViolGen', ('->', (viol, cost, 'path len = %d'%len(pathOrig))))
                    yield (viol, cost, pathOrig)
                else:
                    if not fbch.inHeuristic:
                        debugMsg('confReachViolGen', ('->', ans))
                    break
        ans = None, 0, []
        debugMsg('confReachViolGen', ('->', ans))
        yield ans
        return


# ===========

            if initConf in self.approachConfs:
                keya = (targetConf, self.approachConfs[initConf])
                if keya in self.confReachCache:
                    if debug('confReachViolCache'): print 'confReachCache approach tentative hit'
                    cacheValues = self.confReachCache[keya]
                    sortedCacheValues = sorted(cacheValues,
                                               key=lambda v: v[-1][0].weight() if v[-1][0] else v[-1][0])
                    ans = bsEntails(pbs, prob, sortedCacheValues, loose=True)
                    if ans != None:
                        if debug('traceCRH'): print '    approach cache hit',
                        if debug('confReachViolCache'):
                            debugMsg('confReachViolCache', 'confReachCache approach actual hit')
                        (viol, cost, edgePath) = ans
                        return (viol.update(initViol) if viol else viol,
                                cost,
                                [initConf] + self.confPathFromEdgePath(edgePath))

                for cacheValue in sortedCacheValues:
                    (bs, p, ans) = cacheValue
                    (viol, cost, edgePath) = ans
                    if viol == None: return None
                        newViol = self.checkEdgePath(edgePath, pbs, prob)
                        cv = self.confViolations(targetConf, pbs, prob)[0]
                        cv = self.confViolations(initConf, pbs, prob, initViol=cv)[0]
                        # Don't accept unless the new violations don't add to the initial viols
                        if newViol and newViol.weight() <= cv.weight():
                            v = newViol.update(initViol) if newViol else newViol
                            ans = (v, cost, edgePath)
                            if debug('traceCRH'): print '    reusing path',
                            if debug('confReachViolCache'):
                                debugMsg('confReachViolCache', 'confReachCache reusing path')
                                print '    returning', ans
                            # self.confReachCache[key].append((pbs, prob, ans))
                            return  (v, cost, self.confPathFromEdgePath(edgePath))

==============


def canReachGenTop(args, goalConds, pbs, outBindings):
    (conf, cond, prob, lookVar) = args
    trace('canReachGen() h=', fbch.inHeuristic)
    skip = (fbch.inHeuristic and not debug('inHeuristic'))
    # Set up PBS
    newBS = pbs.copy()
    newBS = newBS.updateFromGoalPoses(goalConds)
    newBS = newBS.updateFromGoalPoses(cond, permShadows=True)
    # Initial test
    path, viol = canReachHome(newBS, conf, prob, Violations())
    if debug('canReachGen'):
        newBS.draw(prob, 'W')
    debugMsg('canReachGen', ('viol', viol))
    if not viol:                  # hopeless
        tracep('canReachGen', 'Impossible dream')
        return
    if viol.empty():
        tracep('canReachGen', 'No obstacles or shadows; returning')
        return
    
    # If possible, it might be better to make the deltas big; but we
    # have to be sure to use the same delta when generating paths.

    objBMinDelta = newBS.domainProbs.minDelta
    objBMinVar = newBS.domainProbs.obsVarTuple
    objBMinProb = 0.95

    lookDelta = objBMinDelta
    moveDelta = objBMinDelta

    # Try to fix one of the violations if any...
    if viol.obstacles:
        obsts = [o.name() for o in viol.obstacles \
                 if o.name() not in newBS.fixObjBs]
        if not obsts:
            debugMsg('canReachGen', 'No movable obstacles to fix')
            return       # nothing available
        # !! How carefully placed this object needs to be
        for ans in moveOut(newBS, prob, obsts[0], moveDelta, goalConds):
            yield ans 
    else:
        shWorld = newBS.getShadowWorld(prob)
        fixed = shWorld.fixedObjects
        shadows = [sh.name() for sh in shWorld.getShadowShapes() \
                   if not sh.name() in fixed]
        if not shadows:
            debugMsg('canReachGen', 'No shadows to clear')
            return       # nothing available
        shadowName = shadows[0]
        obst = objectName(shadowName)
        placeB = newBS.getPlaceB(obst)
        # !! It could be that sensing is not good enough to reduce the
        # shadow so that we can actually reach conf.
        newBS2 = newBS.copy()
        placeB2 = placeB.modifyPoseD(var = lookVar)
        placeB2.delta = lookDelta
        newBS2.updatePermObjPose(placeB2)
        path2, viol2 = canReachHome(newBS2, conf, prob, Violations())
        if path2 and viol2:
            if shadowName in [x.name() for x in viol2.shadows]:
                print 'canReachGen could not reduce the shadow for', obst
                drawObjAndShadow(newBS, placeB, prob, 'W', color='red')
                print 'brown is as far as it goes'
                drawObjAndShadow(newBS2, placeB2, prob, 'W', color='brown')
                raw_input('Go?')
            if debug('canReachGen', skip=skip):
                drawObjAndShadow(newBS, placeB, prob, 'W', color='red')
                debugMsg('canReachGen', 'Trying to reduce shadow (on W in red) %s'%obst)
                trace('    canReachGen() shadow:', obst)
            yield (obst, placeB.poseD.mode().xyztTuple(), placeB.support.mode(),
                   lookVar, lookDelta)
        # Either reducing the shadow is not enough or we failed and
        # need to move the object (if it's movable).
        if obst not in newBS.fixObjBs:
            for ans in moveOut(newBS, prob, obst, moveDelta, goalConds):
                yield ans

def canPickPlaceGen(args, goalConds, bState, outBindings):
    (preconf, ppconf, hand, obj, pose, realPoseVar, poseDelta, poseFace,
     graspFace, graspMu, graspVar, graspDelta, prob, cond, op) = args
    # Don't make this infeasible
    cppFluent = Bd([CanPickPlace([preconf, ppconf, hand, obj, pose, realPoseVar, poseDelta, poseFace,
                                  graspFace, graspMu, graspVar, graspDelta, op, cond]), True, prob], True)
    poseFluent = B([Pose([obj, poseFace]), pose, realPoseVar, poseDelta, prob], True)

    goalConds = goalConds + [cppFluent, poseFluent]
    # Debug
    skip = (fbch.inHeuristic and not debug('inHeuristic'))
    debugMsg('canPickPlaceGen', args)
    trace('canPickPlaceGen() h=', fbch.inHeuristic)
    # Set up the PBS
    world = bState.pbs.getWorld()
    lookVar = bState.domainProbs.obsVarTuple
    graspB = ObjGraspB(obj, world.getGraspDesc(obj), graspFace,
                       PoseD(graspMu, graspVar), delta= graspDelta)
    placeB = ObjPlaceB(obj, world.getFaceFrames(obj), poseFace,
                       PoseD(pose, realPoseVar), delta=poseDelta)
    newBS = bState.pbs.copy()   
    newBS = newBS.updateFromGoalPoses(goalConds)
    newBS = newBS.updateFromGoalPoses(cond, permShadows=True)
    # Initial test
    viol = canPickPlaceTest(newBS, preconf, ppconf, hand,
                             graspB, placeB, prob, op=op)
    if debug('canPickPlaceGen'):
        newBS.draw(prob, 'W')
    debugMsg('canPickPlaceGen', ('viol', viol))
    if not viol:                  # hopeless
        tracep('canPickPlaceGen', 'Violation is permanent; returning')
        newBS.draw(prob, 'W')
        raw_input('Impossible CanPickPlace')
        return
    if viol.empty():
        tracep('canPickPlaceGen', 'No obstacles or shadows; returning')
        return

    objBMinDelta = newBS.domainProbs.minDelta
    objBMinVar = newBS.domainProbs.obsVarTuple
    objBMinProb = 0.95

    lookDelta = objBMinDelta
    moveDelta = objBMinDelta

    # Try to fix one of the violations if any...
    # Treat object target as permanent
    newBS.updatePermObjPose(placeB)

    goalPoseObjs = getPoseObjs(cond)
    if viol.obstacles:
        obsts = [o.name() for o in viol.obstacles \
                 if o.name() not in newBS.fixObjBs.keys()]
        if not obsts:
            tracep('canPickPlaceGen', 'No movable obstacles to remove')
            return       # nothing available
        # !! How carefully placed this object needs to be
        for ans in moveOut(newBS, prob, obsts[0], moveDelta, goalConds):
            debugMsg('canPickPlaceGen', 'move out -> ', ans)
            yield ans 
    else:
        shWorld = newBS.getShadowWorld(prob)
        fixed = shWorld.fixedObjects
        shadows = [sh.name() for sh in shWorld.getShadowShapes() \
                   if not sh.name() in fixed]
        if not shadows:
            debugMsg('canPickPlaceGen', 'No shadows to clear')
            return       # nothing available
        shadowName = shadows[0]
        obst = objectName(shadowName)
        pB = newBS.getPlaceB(obst)
        # !! It could be that sensing is not good enough to reduce the
        # shadow so that we can actually reach conf.
        newBS2 = newBS.copy()
        pB2 = pB.modifyPoseD(var = lookVar)
        pB2.delta = lookDelta
        newBS2.updatePermObjPose(pB2)
        viol2 = canPickPlaceTest(newBS2, preconf, ppconf, hand,
                                 graspB, placeB, prob, op=op)
        debugMsg('canPickPlaceGen', ('viol2', viol2))
        if viol2:
            if debug('canPickPlaceGen', skip=skip):
                newBS.draw(prob, 'W')
                drawObjAndShadow(newBS, pB, prob, 'W', color = 'cyan')
                drawObjAndShadow(newBS2, pB2, prob, 'W', color='magenta')
                debugMsg('canPickPlaceGen',
                         'Trying to reduce shadow on %s'%obst + \
                         'Origin shadow cyan, reduced magenda')
            trace('    canPickPlaceGen() shadow:', obst, pB.poseD.mode().xyztTuple())
            ans = (obst, pB.poseD.mode().xyztTuple(), pB.support.mode(),
                   lookVar, lookDelta)
            debugMsg('canPickPlaceGen', 'reduce shadow -> ', ans)
            yield ans
        # Either reducing the shadow is not enough or we failed and
        # need to move the object (if it's movable).
        if obst not in newBS.fixObjBs:
            for ans in moveOut(newBS, prob, obst, moveDelta, goalConds):
                debugMsg('canPickPlaceGen', 'move out -> ', ans)
                yield ans
        else:
            tracep('canPickPlaceGen', 'found fixed obstacle', obst)

    # placeInGenCache = pbs.beliefContext.genCaches['placeInGen']
    # key = (obj, tuple(regShapes), graspB, placeB, prob, regrasp, away, fbch.inHeuristic)
    # val = placeInGenCache.get(key, None)
    # if val != None:
    #     ff = placeB.faceFrames[placeB.support.mode()]
    #     objShadow = pbs.objShadow(obj, True, prob, placeB, ff)
    #     for ans in val:
    #         ((pB, gB, cf, ca), viol) = ans
    #         pose = pB.poseD.mode() if pB else None
    #         sup = pB.support.mode() if pB else None
    #         grasp = gB.grasp.mode() if gB else None
    #         pg = (sup, grasp)
    #         sh = objShadow.applyTrans(pose)
    #         if all(not sh.collides(obst) for (ig, obst) in reachObsts if obj not in ig):
    #             viol2 = canPickPlaceTest(pbs, ca, cf, gB, pB, prob, op='place')
    #             print 'viol', viol
    #             print 'viol2', viol2
    #             if viol2 and viol2.weight() <= viol.weight():
    #                 if debug('traceGen'):
    #                     w = viol2.weight() if viol2 else None
    #                     print '    reusing placeInGen',
    #                     print '    placeInGen(%s,%s,%s) h='%(obj,[x.name() for x in regShapes],hand), \
    #                           fbch.inHeuristic, 'v=', w, '(p,g)=', pg, pose
    #                 yield ans[0], viol2
    # else:
    #     placeInGenCache[key] = val

###################################

# returns values for (?pose, ?poseFace, ?graspPose, ?graspFace, ?graspvar, ?conf, ?confAppr)
def placeInGenOld(args, goalConds, bState, outBindings,
               considerOtherIns = False, regrasp = False, away=False):
    (obj, region, pose, support, objV, graspV, objDelta,
     graspDelta, confDelta, hand, prob) = args
    if not isinstance(region, (list, tuple, frozenset)):
        regions = frozenset([region])
    else:
        regions = frozenset(region)

    skip = (fbch.inHeuristic and not debug('inHeuristic'))
    pbs = bState.pbs.copy()
    world = pbs.getWorld()

    # !! Should derive this from the clearance in the region
    domainPlaceVar = bState.domainProbs.obsVarTuple 

    if isVar(graspV):
        graspV = domainPlaceVar
    if isVar(objV):
        objV = graspV
    if isVar(support):
        if pbs.getPlaceB(obj, default=False):
            support = pbs.getPlaceB(obj).support # !! Don't change support
        elif obj == pbs.held[hand].mode():
            attachedShape = pbs.getRobot().attachedObj(pbs.getShadowWorld(prob), hand)
            shape = pbs.getWorld().getObjectShapeAtOrigin(obj).applyLoc(attachedShape.origin())
            support = supportFaceIndex(shape)
        else:
            assert None, 'Cannot determine support'

    graspB = ObjGraspB(obj, world.getGraspDesc(obj), None,
                       PoseD(None, graspV), delta=graspDelta)
    placeB = ObjPlaceB(obj, world.getFaceFrames(obj), support,
                       PoseD(None, objV), delta=objDelta)


    # If pose is specified, just call placeGen
    if pose and not isVar(pose):
        if debug('placeInGen'):
            pbs.draw(prob, 'W')
            debugMsgSkip('placeInGen', skip, ('Pose specified', pose))
        oplaceB = placeB.modifyPoseD(mu=hu.Pose(*pose))
        def placeBGen():
            yield oplaceB
        placeBs = Memoizer('placeBGen', placeBGen())
        for ans, viol in placeGenTop((obj, graspB, placeBs, hand, prob),
                                     goalConds, pbs, [], away=away):
            (gB, pB, cf, ca) = ans
            yield (pose, oplaceB.support.mode(),
                   gB.poseD.mode().xyztTuple(), gB.grasp.mode(), graspV, cf, ca)
        return

    # !! Needs to consider uncertainty in region -- but how?

    shWorld = pbs.getShadowWorld(prob)
    regShapes = [shWorld.regionShapes[region] for region in regions]
    if debug('placeInGen'):
        if len(regShapes) == 0:
            debugMsg('placeInGen', 'no region specified')
        shWorld.draw('W')
        for rs in regShapes: rs.draw('W', 'purple')
        debugMsgSkip('placeInGen', skip, 'Target region in purple')
    for ans, viol in placeInGenTop((obj, regShapes, graspB, placeB, hand, prob),
                                   goalConds, pbs, outBindings, considerOtherIns,
                                   regrasp=regrasp, away=away):
        (pB, gB, cf, ca) = ans
        yield (pB.poseD.mode().xyztTuple(), pB.support.mode(),
               gB.poseD.mode().xyztTuple(), gB.grasp.mode(), graspV, cf, ca)


def makeTableShelves(dx=shelfDepth/2.0, dy=0.305, dz=0.45,
                width = shelfWidth, nshelf = 2,
                name='tableShelves', color='brown'):
    sidePrims = [\
        Ba([(-dx, -dy-width, 0), (dx, -dy, dz)],
           name=name+'_side_A', color=color),
        Ba([(-dx, dy, 0), (dx, dy+width, dz)],
           name=name+'_side_B', color=color),
        Ba([(dx, -dy, 0), (dx+width, dy, dz)],
           name=name+'_backside', color=color),
        ]
    coolerPose = hu.Pose(0.0, 0.0, tZ, -math.pi/2)
    shelvesPose = hu.Pose(0.0, 0.0, tZ+coolerZ, -math.pi/2)
    tH = 0.67                           # table height
    shelfSpaces = []
    shelfRungs = []
    for i in xrange(nshelf+1):
        frac = i/float(nshelf)
        bot = dz*frac
        top = dz*frac+width
        shelf = Ba([(-dx, -dy-width, bot),
                    (dx, dy+width, bot+width)],
                   color=color,
                   name=name+'_shelf_'+string.ascii_uppercase[i])
        shelfRungs.append(shelf)
        spaceName = name+'_space_'+str(i+1)
        space = Ba([(-dx+eps, -dy-width+eps, eps),
                    (dx-eps, dy+width-eps, (dz/nshelf) - width - eps)],
                   color='green', name=spaceName)
        space = Sh([space], name=spaceName, color='green').applyTrans(shelvesPose)
        shelfSpaces.append((space, hu.Pose(0,0,bot+eps-(dz/2)-(tH/2)-(coolerZ/2),0)))
    cooler = Sh([Ba([(-0.12, -0.165, 0), (0.12, 0.165, coolerZ)],
                    name='cooler', color=color)],
                name='cooler', color=color)
    table = makeTable(0.603, 0.298, tH, name = 'table1', color=color)
    shelves = Sh(sidePrims + shelfRungs, name = name+'Body', color=color)
    obj = Sh( shelves.applyTrans(shelvesPose).parts() \
              + cooler.applyTrans(coolerPose).parts() \
              + table.parts(),
              name=name, color=color)
    return (obj, shelfSpaces)

    def climbTree(self, graph, targetNode, initNode, pbs, prob, initViol):
        def climb(node):
            nodePath = [node]
            cost = 0.0
            while node:
                if node == initNode:
                    edgePath = self.edgePathFromNodePath(graph, nodePath)
                    return (initViol, cost, edgePath)
                if node.parent:
                    edge = graph.edges.get((node, node.parent), None) or graph.edges.get((node.parent, node), None)
                    if not edge:
                        print 'climb: no edge between nodes', node, node.parent
                        return None
                    nviol = self.colliders(edge, pbs, prob, initViol)
                    if nviol != initViol:
                        print 'new violation', nviol
                        print 'old violation', initViol
                        return None
                    nodePath.append(node.parent)
                    cost += node.hVal
                    node = node.parent
                else:
                    return False
        print 'climbTree', targetNode, initNode
        if targetNode.parent:
            ans = climb(targetNode)
            print 'targetNode has parent, ans=', ans
        else:
            for edge in graph.incidence.get(targetNode, []):
                (a, b)  = edge.ends
                node = a if a != targetNode else b
                if node.parent:
                    targetNode.parent = node
                    targetNode.hVal = pointDist(targetNode.point, node.point)
                    ans = climb(targetNode)
                    print 'connected to tree at', node, 'ans=', ans
                    if ans:
                        return ans
                else:
                    raw_input('could not connect to tree')

    # !! Debugging hack
    # if glob.realWorld:
    #     rw = glob.realWorld
    #     held = rw.held.values()
    #     objShapes = [rw.objectShapes[obj] \
    #                  for obj in rw.objectShapes if not obj in held]
    #     attached = bs.getShadowWorld(p).attached
    #     for path in (path1, path2):
    #         for conf in path:
    #             for obst in objShapes:
    #                 if conf.placement(attached=attached).collides(obst):
    #                     wm.getWindow('W').clear(); rw.draw('W');
    #                     conf.draw('W', 'magenta'); obst.draw('W', 'magenta')
    #                     raw_input('RealWorld crash! with '+obst.name())
    
        # !! It could be that sensing is not good enough to reduce the
        # shadow so that we can actually achieve goal
        # LPK:  now that we have the irreducible shadow, this is not an issue.
        # newBS2 = newBS.copy()
        # placeB2 = placeB.modifyPoseD(var = lookVar)
        # placeB2.delta = lookDelta
        # newBS2.updatePermObjPose(placeB2)
        # viol2 = violFn(newBS2)
        # if viol2:
        #     if shadowName in [x.name() for x in viol2.allShadows()]:
        #         print 'could not reduce the shadow for', obst, 'enough to avoid'
        #         drawObjAndShadow(newBS, placeB, prob, 'W', color='red')
        #         print 'brown is as far as it goes'
        #         drawObjAndShadow(newBS2, placeB2, prob, 'W', color='brown')
        #         raw_input('Go?')
        #     else:
        #         if debug(tag, skip=skip):
        #             drawObjAndShadow(newBS, placeB, prob, 'W', color='red')
        #             debugMsg(tag,'Trying to reduce shadow (on W in red) %s'%obst)
        #             trace('    %s() shadow:'%tag, obst)
        #         yield (obst, placeB.poseD.mode().xyztTuple(),
        #                placeB.support.mode(), lookVar, lookDelta)

'''
def potentialRegionPoseGenCut(pbs, obj, placeB, prob, regShapes, reachObsts, maxPoses = 30):
    def genPose(rs, angle,  chunk):
        for i in xrange(5):
            (x,y,z) = bboxRandomDrawCoords(chunk.bbox())
            # Support pose, we assume that sh is on support face
            pose = hu.Pose(x,y,z + clearance, angle)
            sh = shRotations[angle].applyTrans(pose)
            if debug('potentialRegionPoseGen'):
                sh.draw('W', 'brown')
                wm.getWindow('W').update()
            if all([rs.contains(p) for p in sh.vertices().T]) and \
               all(not sh.collides(obst) for (ig, obst) in reachObsts if obj not in ig):
                return pose
    clearance = 0.01
    for rs in regShapes: rs.draw('W', 'purple')
    ff = placeB.faceFrames[placeB.support.mode()]
    shWorld = pbs.getShadowWorld(prob)
    objShadowBase = pbs.objShadow(obj, True, prob, placeB, ff)
    objShadow = objShadowBase.applyTrans(objShadowBase.origin().inverse())
    shRotations = dict([(angle, objShadow.applyTrans(hu.Pose(0,0,0,angle)).prim()) \
                        for angle in angleList])
    count = 0
    bICost = 5.
    safeCost = 1.
    chunks = []                         # (cost, ciReg)
    for rs in regShapes:
        for (angle, shRot) in shRotations.items():
            bI = CI(shRot, rs.prim())
            if bI == None:
                if debug('potentialRegionPoseGen'):
                    pbs.draw(prob, 'W')
                    print 'bI is None for angle', angle
                continue
            elif debug('potentialRegionPoseGen'):
                bI.draw('W', 'cyan')
                debugMsg('potentialRegionPoseGen', 'Region interior in cyan')
            chunks.append(((angle, bI), 1./bICost))
            co = shapes.Shape([xyCO(shRot, o) \
                               for o in shWorld.getObjectShapes()], o.origin())
            if debug('potentialRegionPoseGen'):
                co.draw('W', 'brown')

            pbs.draw('W')
            safeI = bI.cut(co)

            if not safeI:
                if debug('potentialRegionPoseGen'):
                    print 'safeI is None for angle', angle
            elif debug('potentialRegionPoseGen'):
                safeI.draw('W', 'pink')
                debugMsg('potentialRegionPoseGen', 'Region interior in pink')
            chunks.append(((angle, safeI), 1./safeCost))
    angleChunkDist = DDist(dict(chunks))
    angleChunkDist.normalize()

    for i in range(maxPoses):
        (angle, chunk) = angleChunkDist.draw()
        pose = genPose(rs, angle, chunk)
        if pose:
            count += 1
            yield pose
    if debug('potentialRegionPoseGen'):
        print 'Returned', count, 'for regions', [r.name() for r in regShapes]
    return
'''

    # Makes all objects permanent
    def updateFromAllPoses(self, goalConds,
                           updateHeld=True, updateConf=True, permShadows=False):
        initialObjects = self.objectsInPBS()
        world = self.getWorld()
        if updateHeld:
            (held, graspB) = \
                   getHeldAndGraspBel(goalConds, world.getGraspDesc)
            for h in ('left', 'right'):
                if held[h]:
                    self.fixHeld[h] = True
                    self.held[h] = held[h]
                if graspB[h]:
                    self.fixGrasp[h] = True
                    self.graspB[h] = graspB[h]
            for gB in self.graspB.values():
                if gB: self.excludeObjs([gB.obj])
        if updateConf:
            self.conf = getConf(goalConds, self.conf)
        self.fixObjBs = getAllPoseBels(goalConds, world.getFaceFrames,
                                       self.getPlacedObjBs())
        self.moveObjBs = {}
        # The shadows of Pose(obj) in the cond are also permanent
        if permShadows:
            self.updateAvoidShadow(getPoseObjs(goalConds))
        self.reset()
        finalObjects = self.objectsInPBS()
        if debug('conservation') and initialObjects != finalObjects:
            print 'Failure of conservation'
            print '    initial', sorted(list(initialObjects))
            print '    final', sorted(list(finalObjects))
            raw_input('conservation')
        return self

def getPoseObjs(goalConds):
    pfbs = fbch.getMatchingFluents(goalConds,
                                   B([Pose(['Obj', 'Face']), 'Mu', 'Var', 'Delta', 'P'], True))
    objs = []
    for (pf, pb) in pfbs:
        if isGround(pb.values()):
            objs.append(pb['Obj'])
    return objs

# Overrides is a list of fluents
# Returns dictionary of objects in belief state, overriden as appropriate.
def getAllPoseBels(overrides, getFaceFrames, curr):
    if not overrides: return curr
    objects = curr.copy()
    for (o, p) in getGoalPoseBels(overrides, getFaceFrames).iteritems():
        objects[o] = p
    return objects

<<<<<<< HEAD

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
    ws.setObjectPose('objA', hu.Pose(-1.0, 0.0, 0.0, 0.0))
    ws.setObjectPose('objB', hu.Pose(0.6, 0.0, 0.7, 1.57))
    ws.setObjectPose('table1', hu.Pose(0.6, 0.0, 0.3, math.pi/2))
    ws.setObjectPose('table2', hu.Pose(0.9, 0.0, 0.9, math.pi/2))
    if draw: ws.draw('W')
    return ws

def testScan():
    scan = (0.3, 0.7, 2.0, 2., 30)      # len should be 5
    scan = (0.3, 0.1, 0.1, 2., 30)      # len should be 5
    s = Scan(hu.Pose(0,0,1.5,0).compose(hu.Transform(transf.rotation_matrix(math.pi/4, (0,1,0)))), scan)
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
            pose = hu.Pose(c[0], c[1], c[2], 0.0)
            pointBox.applyTrans(pose).draw('W', 'red')
            count += 1
    return count


=======
        
    def setObjectConfOld(self, objName, conf):
        obj = self.world.objects[objName]
        if not isinstance(conf, dict):
            conf = {objName:conf}
        if not all([(cname in conf.keys()) for cname in obj.chainsByName]):
            conf = conf.copy()
            for chain in obj.chainsInOrder[1:]:
                # print 'Setting region chain', chain.name, 'of', obj
                assert isinstance(chain, Permanent)
                conf[chain.name] = []
        # Update the state of the world
        self.objectConfs[objName] = conf
        for chain in obj.chainsInOrder:
            chainPlace, chainTrans = chain.placement(self.frames[chain.baseFname],
                                                     conf[chain.name])
            for part in chainPlace.parts():
                if part.name() == objName:
                    self.objectShapes[part.name()] = part
                    # print 'Adding', part.name(), 'to objectShapes'
                else:
                    self.regionShapes[part.name()] = part
                    # print 'Adding', part.name(), 'to regionShapes'
            for fname, tr in zip(chain.fnames, chainTrans):
                self.frames[fname] = tr

    def addObjectRegionOld(self, objName, regName, regShape, regTr):
        chain = self.objects[objName]
        regChain = Permanent(regName, objName, regShape, regTr)
        newChain = MultiChain(objName, chain.chainsInOrder + [regChain])
        self.objects[objName] = newChain

==========


def pushPathOld(pbs, prob, gB, pB, conf, direction, dist, prePose, shape, regShape, hand,
                pushBuffer = glob.pushBuffer, prim=False):
    tag = 'pushPath'
    key = (pbs, prob, gB, pB, conf, tuple(direction.tolist()),
           dist, shape, regShape, hand, pushBuffer)
    pushPathCacheStats[0] += 1
    val = pushPathCache.get(key, None)
    if val is not None:
        pushPathCacheStats[1] += 1
        print tag, 'cached ->', val[-1]
        return val
    rm = pbs.getRoadMap()
    newBS = pbs.copy()
    newBS = newBS.updateHeldBel(gB, hand)
    viol = rm.confViolations(conf, newBS, prob)
    if not viol:
        print 'Conf collides in pushPath'
        pdb.set_trace()
        return None, None
    oldBS = pbs.copy()
    if debug(tag): newBS.draw(prob, 'W'); raw_input('Go?')
    pathViols = []
    reason = 'done'
    nsteps = 10
    if prim:
        pushBuffer -= handTiltOffset    # reduce due to tilt
    while float(dist+pushBuffer)/nsteps > pushStepSize:
        nsteps *= 2
    delta = float(dist+pushBuffer)/nsteps
    # Move extra dist (pushBuffer) to make up for the displacement from object
    if prim:
        offsetPose = hu.Pose(*(-1.1*pushBuffer*direction).tolist()+[0.0])
        firstConf = displaceHand(conf, hand, offsetPose)

    # NB!!! This is going to be executed in REVERSE -- so initial part
    # of the path returned is before contact.  This is why step starts
    # at -pushBuffer.

    for step_i in xrange(nsteps+1):
        if step_i == nsteps:
            step = dist
        else:
            step = (step_i * delta) - pushBuffer
        offsetPose = hu.Pose(*(step*direction).tolist()+[0.0])
        if shape and step >= 0:
            nshape = shape.applyTrans(offsetPose)
            if not inside(nshape, regShape):
                reason = 'outside'
                break
        nconf = displaceHand(conf, hand, offsetPose)
        if not nconf:
            reason = 'invkin'
            break
        offsetPB = pB.modifyPoseD(offsetPose.compose(pB.poseD.mode()).pose(),
                                  var=4*(0.0,))
        offsetPB.delta=4*(0.0,)
        oldBS.updateObjB(offsetPB)      # side effect
        viol1 = rm.confViolations(nconf, newBS, prob)
        viol2 = rm.confViolations(nconf, oldBS, prob)
        if (not viol2 or viol2.weight() > 0) and debug('pushPath'):
            print 'Collision with object along pushPath'
        if viol1 is None or viol2 is None:
            reason = 'collide'
            break
        viol = viol1.update(viol2)
        if prim:
            nconf = displaceHandRot(firstConf, conf, hand, offsetPose)
        if prim or debug('pushPath'):
            print 'step=', step, viol
            drawState(newBS, prob, nconf)
            if tag in glob.pauseOn: raw_input('Next?')
        pathViols.append((nconf, viol,
                          offsetPB.poseD.mode() if 0 <= step <= dist else None))
        if debug('pushPath'):
            print 'offset:', offsetPose
    if debug('pushPath'):
        raw_input('Path:'+reason)
    pushPathCache[key] = (pathViols, reason)
    print tag, '->', reason
    return pathViols, reason

def displaceHandRot(firstConf, conf, hand, offsetPose, nearTo=None, doRot=True, angle=0.0):
    cart = conf.cartConf()
    handFrameName = conf.robot.armChainNames[hand]
    trans = cart[handFrameName]         # initial hand position
    # wrist x points down, so we negate angle to get rotation around z.
    xrot = hu.Transform(rotation_matrix(-angle, (1,0,0)))
    nTrans = offsetPose.compose(trans).compose(xrot) # final hand position
    if doRot and trans.matrix[2,0] < -0.9:     # rot and vertical (wrist x along -z)
        firstCart = firstConf.cartConf()
        firstTrans = firstCart[handFrameName]
        handOff = firstTrans.inverse().compose(nTrans).pose()
        if abs(handOff.z) > 0.001:
            sign = -1.0 if handOff.z < 0 else 1.0
            print handOff, angle, '->', sign
            rot = hu.Transform(rotation_matrix(sign*math.pi/15., (0,1,0)))
            nTrans = nTrans.compose(rot)
    nCart = cart.set(handFrameName, nTrans)
    nConf = conf.robot.inverseKin(nCart, conf=(nearTo or conf)) # use conf to resolve
    if all(nConf.values()):
        return nConf

def displaceHand(conf, hand, offsetPose, nearTo=None, angle=0.0):
    cart = conf.cartConf()
    handFrameName = conf.robot.armChainNames[hand]
    trans = cart[handFrameName]
    xrot = hu.Transform(rotation_matrix(-angle, (1,0,0)))
    nTrans = (offsetPose.compose(trans)).compose(xrot)
    nCart = cart.set(handFrameName, nTrans)
    nConf = conf.robot.inverseKin(nCart, conf=(nearTo or conf)) # use conf to resolve
    if all(nConf.values()):
        return nConf


class KDTreeFull:
    def __init__(self, entries, kdLeafSize = 20):
        self.entries = entries
        points = [e.point for e in entries]
        self.points = np.array(points)  # array of point arrays
        self.kdTree = cKDTree(self.points, kdLeafSize)
        self.kdLeafSize = kdLeafSize
        self.size = len(points)
        self.newEntries = []
        self.newPoints = []             # list of point lists
        self.newSize = 0
        self.newKDTree = None
        self.entryTooClose = 0.001 # is there a rational way of picking this?

    def allEntries(self):
        return self.entries + self.newEntries

    def batchAddEntries(self, entries):
        if not entries: return
        last = len(entries) - 1
        for i, entry in enumerate(entries):
            self.addEntry(entry, merge = (i == last))   # merge on the last one

    def mergeKDTrees(self):
        if self.newSize == 0: return
        self.points = np.vstack((self.points,
                                 np.array(self.newPoints)))
        self.kdTree = cKDTree(self.points, self.kdLeafSize)
        self.entries.extend(self.newEntries)
        self.size += self.newSize
        self.newPoints = []
        self.newEntries = []
        self.newSize = 0
        self.newKDTree = None

    def findEntry(self, entry):
        (d, ne) = self.nearest(entry, 1)[0]
        if d <= self.entryTooClose:
            return ne
        
    def addEntry(self, entry, merge = False):
        point = entry.point
        if merge:
            ne = self.findEntry(entry)
            if ne:
                if debug('addEntry'):
                    print 'New', entry.point
                    print 'Old', ne.point
                    raw_input('Entry too close')
                return ne
        self.newPoints.append(point)
        self.newEntries.append(entry)
        self.newSize += 1
        self.newKDTree = None
        if merge and self.newSize > self.kdLeafSize:
            self.mergeKDTrees()
        return entry

    # Is it better just to make a new global tree when we do addConf
    def nearest(self, entry, k):
        merge = []
        dists, ids = self.kdTree.query(entry.point, k)
        if k == 1:
            dists = [dists]; ids = [ids]
        if self.newPoints:
            if not self.newKDTree:
                self.newKDTree = cKDTree(self.newPoints, self.kdLeafSize)
        else:
            return [(d, self.entries[i]) for d, i in zip(dists, ids) if d < np.inf]
        for (d, i) in zip(dists, ids):
            if d < np.inf: merge.append((d, i, False))
        assert self.newKDTree
        newDists, newIds = self.newKDTree.query(entry.point, k)
        if k == 1:
            newDists = [newDists]; newIds = [newIds]
        for (d, i) in zip(newDists, newIds):
            if d < np.inf: merge.append((d, i, True))
        merge.sort()
        return [(d, self.newEntries[i]) if new else (d, self.entries[i]) \
                for (d, i, new) in merge[:k]]


    if debug('collectGraspConfs'):
        xAxisZ = wrist.matrix[2,0]
        bases = robot.potentialBasePosesGen(wrist, hand)
        if abs(xAxisZ) < 0.01:
            if not glob.horizontal:
                glob.horizontal = [(b,
                                    CartConf({'pr2BaseFrame': b,
                                              robot.armChainNames[hand]+'Frame' : wrist,
                                              'pr2Torso':[glob.torsoZ]}, robot)) \
                                   for b in bases]
        elif abs(xAxisZ + 1.0) < 0.01:
            if not glob.vertical:
                glob.vertical = [(b,
                                  CartConf({'pr2BaseFrame': b,
                                            robot.armChainNames[hand]+'Frame' : wrist,
                                            'pr2Torso':[glob.torsoZ]}, robot)) \
                                 for b in bases]
        else:
            assert None


'''
# returns
# ['Occ', 'Pose', 'PoseFace', 'PoseVar', 'PoseDelta']
# obj, pose, face, var, delta
class CanReachGen(Function):
    def fun(self, args, goalConds, bState):
        (conf, fcp, prob, cond) = args
        pbs = bState.pbs.copy()
        # Don't make this infeasible
        goalFluent = Bd([CanReachHome([conf, fcp, cond]), True, prob], True)
        goalConds = goalConds + [goalFluent]
        # Set up PBS
        newBS = pbs.copy()
        newBS = newBS.updateFromGoalPoses(goalConds)
        newBS = newBS.updateFromGoalPoses(cond, permShadows=True)
        shWorld = newBS.getShadowWorld(prob)
        tr('canReachGen', draw=[(newBS, prob, 'W'),
                     (conf, 'W', 'pink', shWorld.attached)], snap=['W'])
        tr('canReachGen', zip(('conf', 'fcp', 'prob', 'cond'),args))
        # Call
        def violFn(pbs):
            path, viol = canReachHome(pbs, conf, prob, Violations())
            return viol
        lookVar = tuple([lookVarIncreaseFactor * x \
                        for x in pbs.domainProbs.obsVarTuple])
        for ans in canXGenTop(violFn, (cond, prob, lookVar),
                                    goalConds, newBS, 'canReachGen'):
            tr('canReachGen', ('->', ans))
            yield ans.canXGenTuple()
            tr('canReachGen', 'exhausted')

# Preconditions (for R1):

# 1. CanPickPlace(...) - new Pose fluent should not make the
# canPickPlace infeasible.  new Pose fluent should not already be in
# conditions.

# 2. Pose(obj) - new Pose has to be consistent with the goal (ok to
# reduce variance wrt goal but not cond)

# LPK!! More efficient if we notice right away that we cannot ask to
# change the pose of an object that is in the hand in goalConds
class CanPickPlaceGen(Function):
    #@staticmethod
    def fun(self, args, goalConds, bState):
        (preconf, ppconf, hand, obj, pose, realPoseVar, poseDelta, poseFace,
         graspFace, graspMu, graspVar, graspDelta, op, prob, cond) = args
        pbs = bState.pbs.copy()
        # Don't make this infeasible
        cppFluent = Bd([CanPickPlace([preconf, ppconf, hand, obj, pose,
                                      realPoseVar, poseDelta, poseFace,
                                      graspFace, graspMu, graspVar, graspDelta,
                                      op, cond]), True, prob], True)
        poseFluent = B([Pose([obj, poseFace]), pose, realPoseVar, poseDelta, prob],
                        True)
        # Augment with conditions to maintain
        goalConds = goalConds + [cppFluent, poseFluent]
        
        world = pbs.getWorld()
        lookVar = tuple([lookVarIncreaseFactor * x \
                                for x in pbs.domainProbs.obsVarTuple])
        graspB = ObjGraspB(obj, world.getGraspDesc(obj), graspFace, poseFace,
                           PoseD(graspMu, graspVar), delta= graspDelta)
        placeB = ObjPlaceB(obj, world.getFaceFrames(obj), poseFace,
                           PoseD(pose, realPoseVar), delta=poseDelta)
        # Set up PBS
        newBS = pbs.copy()
        newBS = newBS.updateFromGoalPoses(goalConds)
        newBS = newBS.updateFromGoalPoses(cond, permShadows=True)
        # Debug
        shWorld = newBS.getShadowWorld(prob)
        tr('canPickPlaceGen',
           draw=[(newBS, prob, 'W'),
                 (preconf, 'W', 'blue', shWorld.attached),
                 (ppconf, 'W', 'pink', shWorld.attached),
                 (placeB.shape(shWorld), 'W', 'pink')],
           snap=['W'])
        tr('canPickPlaceGen',
           zip(('preconf', 'ppconf', 'hand', 'obj', 'pose', 'realPoseVar', 'poseDelta', 'poseFace',
                'graspFace', 'graspMu', 'graspVar', 'graspDelta', 'prob', 'cond', 'op'),
               args))
        # Initial test
        def violFn(pbs):
            v, r = canPickPlaceTest(pbs, preconf, ppconf, hand,
                                    graspB, placeB, prob, op=op)
            return v
        for ans in canXGenTop(violFn, (cond, prob, lookVar),
                              goalConds, newBS, 'canPickPlaceGen'):
            tr('canPickPlaceGen', ('->', ans))
            yield ans.canXGenTuple()
        tr('canPickPlaceGen', 'exhausted')
'''        

    elif (not path) or viols.weight() > 0:
        path1, v1 = canReachHome(bs, cs, p, Violations(), optimize=True)
        if (not path1) or v1.weight() > 0:
            print 'Path1 failed, trying RRT'
            path1, v1 = rrt.planRobotPathSeq(bs, p, home, cs, None,
                                             maxIter=50, failIter=10)
            if path1 and v1.weight() > 0:
                raw_input('Potential collision in path1')
            if (not path1):
                raw_input('Path1 RRT failed')
        if (not path) and path1:
            path2, v2 = canReachHome(bs, ce, p, Violations(),
                                     optimize=True, reversePath=True)
            if (not path2) or v2.weight() > 0:
                print 'Path2 failed, trying RRT'
                path2, v2 = rrt.planRobotPathSeq(bs, p, home, ce, None,
                                                 maxIter=50, failIter=10)                
            if (not path2) or v2.weight() > 0:
                print 'Path2 RRT failed, trying full RRT'
                path, viols = rrt.planRobotPathSeq(bs, p, cs, ce, None,
                                                   maxIter=50, failIter=10)            
                assert path and viols.weight() == 0
        else:
            path2, v2 = None, None
        if (not path) and path1 and path2:
            # make sure to interpolate paths in their original directions.
            path = rrt.interpolatePath(path1) + rrt.interpolatePath(path2)[::-1]
