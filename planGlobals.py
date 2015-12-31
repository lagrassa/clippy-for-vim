import platform

########################################
# Default values of global values, can be overriden in local.py
########################################

libkinDir = ''

########################################
# System: LINUX and Mac OS X are supported
########################################

LINUX = (platform.system() == 'Linux')

########################################
# ROS: Controlling robot 
########################################

useROS = LINUX
speedFactor = 0.5

######################################################################
# Right Arm??
######################################################################

useCartesian = False
useLookAtHand = False

useRight = True                        # DEBUG
useVertical = True                     # DEBUG
useHorizontal = True

if useROS:
    useRight = True
    useVertical = True
    useHorizontal = True

useMPL = False                           # Amruth

########################################
# HACKS
########################################

traceGen = False

useRegionPoseCache = False

realWorld = None
horizontal = None
vertical = None

mergeShadows = False
approachBackoff = 0.1
approachPerpBackoff = 0.025

useHandTiltForPush = True              # Use for real robot

inHeuristic = False

useCC = True                            # Use compiled chains collision checks
skipSearch = False                      # Don't plan paths except for prims
useMathematica = False

ignoreShadowZ = True
useInflation = True

########################################
# Name prefixes for objects that are graspable, pushable, etc.
# These are set when defining objects.
########################################

objectTypes = {}
graspableNames = []
pushableNames = []
crashableNames = []
objectSymmetries = {}
graspDesc = {}
constructor = {}

########################################
# Perception Parameters
########################################

# laserScanParams = (0.3, 0.2, 0.1, 3., 30)
laserScanParams = (0.3, 0.075, 0.075, 3., 15)

########################################
# Robot Parameters
########################################

IKfastStep = 0.1
useRRT = True                           # use RRT exclusively
rrtStep = 0.025
rrtInterpolateStepSize = 8*rrtStep
smoothSteps = 100
smoothPasses = 20
torsoZ = 0.2                            # normally 0.3 or 0.2
skipRRT = False
maxRRTIter = 100
failRRTIter = 10
smoothPathObst = True
rrtPlanAttempts = 5                     # to try to get smaller base displacement

maxOpenLoopDist = 8.0                   # Assumes we're doing moveLook paths

########################################
# Obstacle growing, should be determined by uncertainty
########################################

baseGrowthX = 0.1
baseGrowthY = 0.1

########################################
# FBCH
########################################

savedMaxNodesHPN = maxNodesHPN = 100
rebindPenalty = 30
monotonicFirst = True
monotonicNow = True
drawFailedNodes = False
drawRebindNodes = True

########################################
# Searcing for paths in roadmap
########################################

objCollisionCost = 10.0
shCollisionCost = 5.0
maxSearchNodes = 10000.
searchGreedy = 0.5                      # 0.5 is A*, 0 is UCS, 1 is best-first


########################################
# Pushing params
########################################

pushBuffer = 0.12

########################################
# Heuristic
########################################

numOpInstances = 1                      # used to be 4

########################################
# Table parameters
########################################

tableMaxShrink = 0.1
minTableDim = 2.0
cloudPointsResolution = 0.01            # should be 0.1
tableBadWeight = 5

########################################
# Debugging
########################################
debugOn = []
pauseOn = []
logOn = []
planNum = 0


########################################
# Temporary
########################################
pushGenCalls = 0
pushGenCallsH = 0
pushGenFail = 0
pushGenFailH = 0
pushGenCache = 0
pushGenCacheH = 0
pushGenCacheMiss = 0
pushGenCacheMissH = 0

# This has to be ugly here so that Cython can see the symbols.
from local import *

print 'Loaded planGlobals.py'
