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

usualTags = ['nonmon']
flags = ['simpleAbstractCostEstimates',
         # 'primitiveHeuristicAlways',
          'useBBinH']
heuristicTags = ['hAddBackV', 'heuristic', 'hAddBackInf',
                 'debugInHeuristic', 'h']  # 'hAddBack'
skeletonTags = ['skeleton', 'regression:fail', 'appOp:number']
traceOnly = ['traceCRH', 'pickGen', 'placeGen', 'easyGraspGen',
                   'placeInGen', 'lookGen', 'lookHandGen', 'canPickPlaceGen',
                   'pushGen', 'assign']
debugOnly = ['h', 'assign']  # don't pause
#------------------------------------
# Add tags that you want to debug and pause on to this list

debugOn = usualTags + ['disablePickPlace'] + ['animate', 'pushGen'] + skeletonTags

if platform.system() == 'Linux':
    for x in ['robotEnv', 'tables', 'obsUpdate', 'bigAngleChange']:
        if not x in debugOn: debugOn.append(x)

#------------------------------------
pauseOn = debugOn[:]

if 'pushPath' in pauseOn: pauseOn.remove('pushPath')

logOn = debugOn + traceOnly
debugOn.extend(debugOnly)
debugOn.extend(flags)

