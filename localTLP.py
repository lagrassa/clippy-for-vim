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


debugOn = ['prim', 'skeleton', 'executionFail', 'executePath',
           'pickGen', 'placeGen', 'placeInGen',
           'canReachHome', 'confReachViol',
           'confReachViolCache'] 
debugOn = ['prim', 'skeleton', 'executionFail', 'executePath',
           'canReachHome', 'canReachGen', 'placeInGen', 'placeGen', 'canPickPlaceTest']
debugOn = ['prim', 'skeleton', 'executionFail', 'executePath', 'hAddBackV', 'regression:fail',
           'pickGen', 'placeGen', 'placeInGen', 'confReachViol', 'confReachViolGen', 'confReachViolCache',
           'canPickPlaceTest', 'canReachHome', 
           'potentialLookHandConfs', 'lookHandGen', 'visible'
           ]
debugOn = ['prim', 'skeleton', 'executionFail', 'executePath', 'traceGen', 'traceCRH', 'hAddBackV', 'regression:fail',
           'pickGen', 'placeGen', 'lookGen', 'lookHandGen', 'pathObst', 'visible', 'regrasping',
           'confReachViolGen', 'confReachViol', 'confReachViolCache', 'colliders:collision', 'successors',
           'canPickPlaceTest', 'canReachHome', 'confViolations']

debugOn = ['prim', 'skeleton', 'executionFail', 'executePath', 'traceGen', 'traceCRH',
           'regression:fail', 'applicableOps', 'regression:inconsistent', 'easyGraspGen',
           'pickGen', 'placeGen', 'placeInGen', 'lookGen', 'lookHandGn', 'canReachGen',
           'visible', 'potentialRegionPoseGen', 'canPickPlaceGen',
           'minViolPath', 'confReachViolGen', 'confReachViol', 'confReachViolCache'
           ]

# debugOn = ['confReachViolGen', 'visible', 'canReachHome', 'confReachViol', 'confReachViolCache', ]
# debugOn = ['hAddBackV', 'traceGen']
# debugOn = ['nonmon', 'executionFail', 'hAddBackV', 'skeleton', 'applicableOps', 'regression:fail']
# debugOn = ['traceGen', 'traceCRH', 'potentialRegionPoseGen']
# debugOn = ['hAddBackV', 'traceGen', 'traceCRH']
# debugOn = ['regression:inconsistent', 'traceGen', 'traceCRH']
# debugOn = ['checkCRH']
#debugOn = ['traceGen', 'traceCRH', 'confReachViol', 'confReachViolCache', 'drawInHeuristic']
debugOn = ['traceGen', 'traceCRH', 'confReachViol', 'canPickPlaceGen', 'pickGen']
debugOn = ['traceGen', 'traceCRH', 'prim', 'getShadowWorldGrasp']
# debugOn = []
debugOn = ['traceGen', 'traceCRH', 'skeleton', 'hAddBackInf']
#           'potentialGraspConfs', 'potentialRegionPoseGen', 'infeasible'
#          'getShadowWorld', 'confReachViol', 'confViolations', 'lookGen', 'canReachGen', 'visible', 'lookGen'

if platform.system() == 'Linux':
    for x in ['robotEnv', 'tables', 'obsUpdate', 'bigAngleChange']:
        if not x in debugOn: debugOn.append(x)

pauseOn = debugOn
