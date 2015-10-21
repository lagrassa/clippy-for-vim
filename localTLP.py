import os
import platform
libkinDir = './IK/'
if platform.system() == 'Linux':
    outDir = '/mit/tlp/search/'
    genDir = '/mit/tlp/gen/'
    movieDir = '/mit/tlp/mathMovies/'
else:
    outDir = '/Users/tlp/Documents/search/'
    genDir = '/Users/tlp/Documents/gen/'
    movieDir = '/Users/tlp/Documents/mathMovies/'
dotSearch = outDir + 's%s%s.dot'
dotSearchX = outDir + 'sX%s%s.dot'
pngGen = '%s/g%s.png'
htmlGen = '%s/a%s.html'
htmlGenH = '%s/aH%s.html'

rebindPenalty = 100

monotonicFirst = True
drawFailedNodes = False
drawRebindNodes = True

usualTags = ['nonmon', 'animate']
flags = ['simpleAbstractCostEstimates',
         'primitiveHeuristicAlways',
         'pushSim',
         'helpfulActions'
         ]
heuristicTags = ['hAddBackV', 'heuristic', 'hAddBackInf',
                 'debugInHeuristic', 'h', 'hAddBack']
skeletonTags = ['skeleton', 'regression:fail', 'appOp:number', 'rebind',
                'clobber', 'regression:fail:bindings']
traceOnly = ['traceCRH', 'pickGen', 'placeGen', 'easyGraspGen',
             'placeInGen', 'lookGen', 'lookHandGen', 'canPickPlaceGen',
             'pushGen', 'pushGen', 'pushInGen',
             'assign', 'beliefUpdate', 'regression', 'regression:fail']
debugOnly = ['h', 'assign']  # don't pause
#------------------------------------
# Add tags that you want to debug and pause on to this list

debugOn = usualTags + ['pushGen', 'pushInGen', 'pushGen_kin', 'pushPath']

# + ['debugInHeuristic', 'pushGen', 'pushInGen', 'pushGen_kin', 'pushPath']
# + ['lookGen', 'pickGen', 'placeGen', 'placeInGen', 'getReachObsts', 'CanReachHome']
# + ['debugInHeuristic', 'pickGen', 'placeGen', 'placeInGen', 'lookGen', 'visible', 'CanSeeFrom', 'canView']
# + ['potentialRegionPoseGen', 'regionPoseHyps', 'debugInHeuristic', 'potentialGraspConfs', 'potentialGraspConfsWin', 'potentialGraspConfsLose']
# + ['pickGen', 'placeGen', 'placeInGen', 'lookGen']
# + ['visible', 'CanSeeFrom', 'canView', 'pickGen', 'achCanPush', 'lookAchGen', 'canPush']
# + ['noWriteSearch', 'noTrace', 'noPlayback'] 
# + ['animate']
# + ['noWriteSearch', 'noTrace', 'noPlayback'] 
#+ ['canReachGen', 'pushPath', 'pushGen', 'debugInHeuristic', 'sim']
#+ ['pickGen', 'debugInHeuristic', 'potentialGraspConfs', 'potentialGraspConfsWin', 'potentialGraspConfsLose']
# + ['noWriteSearch', 'noTrace', 'noPlayback'] 

# tables, obsUpdate
if platform.system() == 'Linux':
    for x in ['robotEnv']:
        if not x in debugOn: debugOn.append(x)

#------------------------------------
pauseOn = debugOn[:]

logOn = debugOn + traceOnly
debugOn.extend(debugOnly)
debugOn.extend(flags)

