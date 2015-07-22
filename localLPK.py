import os
import platform
path = os.getcwd()
parent = path[:path.rfind('/')] + '/../'
libkinDir = './IK/'

if platform.system() == 'Linux':
   outDir = '/mit/tlp/search/'
   genDir = '/mit/tlp/gen/'
else:
   outDir = '/Users/lpk/Desktop/search/'
   genDir = '/Users/lpk/Desktop/genFiles/'
dotSearch = outDir + 's%s%s.dot'
dotSearchX = outDir + 'sX%s%s.dot'

pngGen = '%s/g%s.png'
htmlGen = '%s/a%s.html'
htmlGenH = '%s/aH%s.html'

rebindPenalty = 100

monotonicFirst = True
drawFailedNodes = False
drawRebindNodes = True

'''
abstractCost
animate
applicableOps
appOp:detail
appOp:number
appOp:result
assign  : data association
btbind
canPickPlaceGen
canReachGen
canReachHome
canReachNB (two confs, no base)
canSeeGen
checkCRH (compares CRH in and out of heuristic; draws good picture)
clobber
confReachViol
confViolations
cost
debugInHeuristic (turn on debugging inside heuristic in regression)
drawInHeuristic
executionFail
extraTests (test for rendundancy and contradiction within state fluent set;
            test that cached fluent strings are right)
fluentCache (be sure the fluent cache is behaving)
hAddBack   (super verbose)
hAddBackInf
hAddBackInfFinal
hAddBackV  (just values)
h (really just values)
heuristic (in ucSearch: positive value at goal or 0 value elsewhere)
heuristicInversion  (prim actually achieves more than abs)
infeasible (hierarchical subgoal)
inHeuristic (turn on debugging inside heuristic in generators)
inTest
lookGen
lookHandGen
nextStep
nonmon
obsUpdate : draw detection, update
pickPlaceTest
pickTol (be sure we're not giving pick too big a variance)
placeGen
placeInGen
placeInRegionGen
potentialLookConfs
prim
regression
regression:fail
regression:inconsistent (maybe evidence that generator could have failed
                          earlier)
satisfies                          
sim
simpleAbstractCostEstimates  (cut down on generator calls high in the hierarchy)
skeleton
testVerbose
traceCRH
traceGen  (compact summary of generators)
visible
visibleEx (show visibility stuff during execution even if visible is false)

'''

usualTags = ['nonmon', 'simpleAbstractCostEstimates']
             #'primitiveHeuristicAlways']
heuristicTags = ['hAddBack', 'hAddBackV', 'heuristic', 'hAddBackInf',
                 'debugInHeuristic', 'h']
skeletonTags = ['skeleton', 'regression:fail', 'appOp:number']
traceOnly = ['traceCRH', 'pickGen', 'placeGen', 'easyGraspGen',
                   'placeInGen', 'lookGen', 'lookHandGen', 'canPickPlaceGen',
                   'pushGen', 'assign']
debugOnly = ['h', 'assign']  # don't pause

debugOn = usualTags + ['hAddBackInf']

pauseOn = debugOn[:]

logOn = debugOn + traceOnly
debugOn.extend(debugOnly)


