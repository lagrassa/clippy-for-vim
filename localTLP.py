import os
import platform
libkinDir = './IK/'
if platform.system() == 'Linux':
    outDir = '/mit/tlp/search/'
else:
    outDir = '/Users/tlp/Documents/search/'
dotSearch = outDir + 's%s%s.dot'
dotSearchX = outDir + 'sX%s%s.dot'

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
           #'verifyPath', 'backwards', 'verifyRRTPath'
           ]

if platform.system() == 'Linux':
    for x in ['robotEnv', 'tables', 'obsUpdate', 'bigAngleChange']:
        if not x in debugOn: debugOn.append(x)

pauseOn = debugOn
