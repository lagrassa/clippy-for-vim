
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
