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
canReachGen
canReachHome
canSeeGen
checkCRH (compares CRH in and out of heuristic; draws good picture)
confReachViol
confViolations
cost
drawInHeuristic
executionFail
executionSurprise
hAddBack   (super verbose)
hAddBackInf
hAddBackV  (just values)
heuristic (in ucSearch: positive value at goal or 0 value elsewhere)
heuristicInversion  (prim actually achieves more than abs)
infeasible (hierarchical subgoal)
lookHandGen
nextStep
nonmon
pickPlaceTest
placeInGen
prim
regression
regression:fail
regression:inconsistent (may be evidence that generator could have failed earlier)

sim
skeleton
traceGen  (compact summary of generators)

'''
debugOn = ['nonmon', 'executionFail', 
           'appOp:number', 'traceGen', 
           'executionSurprise', 'prim', 'canPickPlaceGen']


pauseOn = debugOn



'''
- change HPN to hierarchically highest true kernel
Current version is much too attached to its subgoals



Python mode
C-C C-C should leave you in bottom buffer
Indentation should be better.

LPK to do

- Test7


- If we want to make moveObjToClear hierarchical, and postpone the pose
fluent, then we have to find a way to conditionalize so that everybody
else knows the object is out of the way, even if not where

- Look at hand

- In HPN, when replanning, try to re-use the old plan
- Monotonic first, then try non-mon
- Get HFF working again; figure out how to integrate with lower-level heuristic

- Use the heuristic value to estimate the cost for an abstract
operator!?  Some args might not yet be bound.

- Probabilities
o One problem is that probs are not allocated uniformly over eventual leaf
nodes
o Add failure probabilities into the rules

- Estimator that handles discrete events (e.g. dropping, failing to place)
- make canSeeFrom true

- Make estimator and simulator better: mixture probs, possible failure of pick and/or place, add noise to traj sim


- Make rebind penalty be the heuristic value
- Deal with some kind of constrained estimation

Notes
- Conceptual problem:  robot moves (an object) near the wall, then localizes, and discovers it is very close.  Our model for moving is to use the terminal variance, but that variance might put the robot in collision in the initial state.  **That should not be a showstopper**

- How do generate and order suggestions?  Use some sort of buffer of ones we've
found.

- Add fog

- Make trees show compressed version of conditions by default
- Rewrite regress to have some subroutines
- Can we make variance a function of motion length? (or, at least, 

- Random scenario generator


-----------------



Notes
- Conceptual problem:  robot moves (an object) near the wall, then localizes, and discovers it is very close.  Our model for moving is to use the terminal variance, but that variance might put the robot in collision in the initial state.  **That should not be a showstopper**

- How do generate and order suggestions?  Use some sort of buffer of ones we've
found.

- Add fog

--------
- Make trees show compressed version of conditions by default
- Rewrite regress to have some subroutines
- Can we make variance a function of motion length? (or, at least, 
- Monotonic first; then try again with nonmon
- Random scenario generator
- Hierarchy




----------------------------------------------------------------------

Improvements to belief state update
- do observation update for Gaussian mixture right

Improvements to rule specification


Improvements to heuristic
- Pick relaxed progress
- Be sure caching as much as possible
- indexicals or other strategy for finding appropriate bindings for
forward search
- figure out how it can apply if we have postponed binding (so subgoal has
variables)

Improvements to handling belief fluents
- think through what to do with delta
- think through how to handle the probabilities and variances in the heuristic

Improvements to planner
- try monotonic first
- allow costs to depend on variables
- draw a failure node in the HPN execution file when a precondition fail
happens

2D domain
- Improve modeling of likelihood of succeeding to grasp an object
- Make it so that look can verify held object and grasp

Bigger ideas
- Automatic belief rule generation!!!
- Automatic decision to achieve BV and postpone other preconditions
- Take big problem and scope it down by limiting space, objects
- Go back to relative uncertainty model

Major punted problems
- Can't move twice in a row:  no way to generate intermediate confs

Thinking:
---------

Conditions just have a mean and a variance.  Belief and relaxed belief
may have more complicated mixture dists.  For discrete RVs, conditions
just have a single value.

--------------

A subtlety:  decreased variance on a conf can come from a look, so Conf is a
sideEffect of look

--------------

Putting a max distance on motions and using the variance increase from
that during planning.

--------------

Are we burying some preconditions in the generators?  Is that okay?
It's fine if the condition are in the generators, but they should also
be in the fluents, so that the execution monitoring will work.  This
is a principle for writing generators:  the test should be in a fluent and
be sufficient.

------------

It would be good to have some sort of "suggester" mechanism for
ordering operator applications.  A problem is that we seem to need a product of operators:
- for every type of thing we might want to make clear (canMove, canPick, canPlace, canMoveHolding)
- for every thing we can do to clear out the region: move something, localize the robot, localize an object
'''
