import os
# path = os.getcwd()
# parent = path[:path.rfind('/')] + '/../'
libkinDir = './IK/'
# print 'parent', parent
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
debugOn = ['traceGen', 'traceCRH', 'placeInGen']
# debugOn = []
pauseOn = debugOn
if 'canPickPlaceTest' in pauseOn:
    pauseOn.remove('canPickPlaceTest')

