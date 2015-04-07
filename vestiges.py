
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
