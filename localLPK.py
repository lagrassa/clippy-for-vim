import os
path = os.getcwd()
parent = path[:path.rfind('/')] + '/../'
libkinDir = './IK/'

outDir = '/Users/lpk/Desktop/search/'
dotSearch = outDir + 's%s%s.dot'
dotSearchX = outDir + 'sX%s%s.dot'

rebindPenalty = 100

monotonicFirst = True
drawFailedNodes = False
drawRebindNodes = True

'''
abstractCost
applicableOps
appOp:detail
appOp:number
appOp:result
btbind
canPickPlaceGen
canReachGen
canReachHome
canReachNB (two confs, no base)
canSeeGen
checkCRH (compares CRH in and out of heuristic; draws good picture)
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
hAddBackV  (just values)
heuristic (in ucSearch: positive value at goal or 0 value elsewhere)
heuristicInversion  (prim actually achieves more than abs)
infeasible (hierarchical subgoal)
inHeuristic (turn on debugging inside heuristic in generators)
inTest
lookGen
lookHandGen
nextStep
nonmon
obsUpdate
pickPlaceTest
placeGen
placeInGen
placeInRegionGen
potentialLookConfs'
prim
regression
regression:fail
regression:inconsistent (maybe evidence that generator could have failed
                          earlier)
sim
skeleton
traceCRH
traceGen  (compact summary of generators)
visible

'''
debugOn = ['skeleton', 'nonmon', 'executionFail', 'executionSurprise',
           'traceGen', 'traceCRH', 'pickTol']



pauseOn = debugOn

