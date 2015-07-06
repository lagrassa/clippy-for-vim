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
pngGen = genDir + 'g%s_%s.png'
htmlGen = genDir + 'a%s_%s.html'
htmlGenH = genDir + 'aH%s_%s.html'

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

debugOn = ['traceGen', 'traceCRH', 'skeleton', 'simpleAbstractCostEstimates', 'nonmon',
           'heuristic0', 'collectGraspConfs'
           'verifyPath', 'pickGen', 'placeGen', 'placeInGen', 'lookGen', 
           'confReachViol', 'debugInHeuristic', 'getShadowWorld',
           ]

if platform.system() == 'Linux':
    for x in ['robotEnv', 'tables', 'obsUpdate', 'bigAngleChange']:
        if not x in debugOn: debugOn.append(x)

pauseOn = debugOn
