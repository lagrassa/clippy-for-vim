import os
import platform
libkinDir = './IK/'
if platform.system() == 'Linux':
    outDir = '/mit/tlp/search/'
    genDir = '/mit/tlp/gen/'
else:
    outDir = '/Users/tlp/Documents/search/'
    genDir = '/Users/tlp/Documents/gen/'
dotSearch = outDir + 's%s%s.dot'
dotSearchX = outDir + 'sX%s%s.dot'
pngGen = '%s/g%s.png'
htmlGen = '%s/a%s.html'
htmlGenH = '%s/aH%s.html'

rebindPenalty = 100

monotonicFirst = True
drawFailedNodes = False
drawRebindNodes = True

# Dictionary of common trace keyworrds
'''
'applicableOps'
'canPickPlaceGen'
'canPickPlaceTest'
'canReachGen'
'canReachHome'
'canReachNB'
'canViewFail'
'checkCRH'
'colliders:collision'
'confReachViol'
'confReachViolCache'
'confReachViolGen'
'confViolations'
'drawInHeuristic'
'easyGraspGen'
'executePath'
'executionFail'
'getShadowWorld'
'getShadowWorldGrasp'
'hAddBackV'
'infeasible'
'lookGen'
'lookHandGen'
'minViolPath'
'nonmon'
'pathObst'
'pickGen'
'placeGen'
'placeInGen'
'potentialGraspConfs'
'potentialGraspConfsWin'
'potentialLookHandConfs'
'potentialRegionPoseGen'
'potentialRegionPoseGenWeight'
'path'
'prim'
'regrasping'
'regression:fail'
'regression:inconsistent'
'simpleAbstractCostEstimates'
'skeleton'
'successors'
'traceCRH'
'traceGen'
'visible'
'''

# debugOn = ['traceGen', 'skeleton', 'simpleAbstractCostEstimates', 'nonmon',
#            'useGJK', 'h', 'traceCRH', 'debugInHeuristic',
#            # 'traceCRH'
#            #'pushGen', 'handContactFrames', 'graspBForContactFrame',
#            #'pushPath', 'canPush',
#            #'lookGen', 'visible', 'canView', 'lookAtConfCanView'
#            ]

# if platform.system() == 'Linux':
#     for x in ['robotEnv', 'tables', 'obsUpdate', 'bigAngleChange']:
#         if not x in debugOn: debugOn.append(x)

# pauseOn = debugOn[:]
# if 'h' in pauseOn:
#     pauseOn.remove('h')

# debugOn = ['nonmon', 'skeleton', 'easyGraspGen', 'pickGen', 'debugInHeuristic']
#            # 'regression:fail',
#            #'appOp:number'
#            # 'pushPath', 'graspBForContactFrame'
#            #'regression:fail', 'appOp:number', 'regression', 'lookGen',
#            #'canReachHome'

# pauseOn = debugOn[:]
# logOn = debugOn + ['traceCRH', 'pickGen', 'placeGen', 'easyGraspGen', 'pushGen',
#                    'placeInGen', 'lookGen', 'lookHandGen', 'canPickPlaceGen', 'sim']
                   
# debugOn.append('h')

# if platform.system() == 'Linux':
#     for x in ['robotEnv', 'tables', 'obsUpdate', 'bigAngleChange']:
#         if not x in debugOn: debugOn.append(x)
#         if not x in logOn: debugOn.append(x)

# 'primitiveHeuristicAlways' for more accurate (but slower) heuristic.
# 'simpleAbstractCostEstimates' for cheaper heuristc
usualTags = ['nonmon', 'skeleton', 'simpleAbstractCostEstimates']
heuristicTags = ['hAddBack', 'hAddBackV', 'heuristic', 'hAddBackInf',
                 'debugInHeuristic']
skeletonTags = ['regression:fail', 'appOp:number']
traceOnly = ['traceCRH', 'pickGen', 'placeGen', 'easyGraspGen', 'sim',
                   'placeInGen', 'lookGen', 'lookHandGen', 'canPickPlaceGen',
                   'pushGen', 'assign']
debugOnly = ['h', 'assign']  # don't pause


debugOn = usualTags
# + ['assign', 'pushGenVar'] + heuristicTags
# ['easyGraspGen', 'applicableOps', 'pickGen', 'regressionFail']


pauseOn = debugOn[:]
logOn = debugOn + traceOnly
debugOn.extend(debugOnly)

