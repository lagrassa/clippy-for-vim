import platform

########################################
# Default values of global values, can be overriden in local.py
########################################

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

########################################
# HACKS
########################################

realWorld = None
horizontal = None
vertical = None

mergeShadows = False
approachBackoff = 0.1
approachPerpBackoff = 0.025

useHandTiltForPush = True              # Use for real robot

inHeuristic = False

useCC = False                            # Use compiled chains collision checks
skipSearch = False                       # Don't plan paths except for prims
useMathematica = False

ignoreShadowZ = True
useInflation = False

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
# Robot Parameters
########################################

IKfastStep = 0.1
useRRT = True                           # use RRT exclusively
rrtStep = 0.025
rrtInterpolateStepSize = 10*rrtStep
smoothSteps = 100
smoothPasses = 20
torsoZ = 0.1                            # normally 0.3 or 0.2
skipRRT = False
maxRRTIter = 200
failRRTIter = 20
smoothPathObst = True

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

numOpInstances = 4

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

import local
reload(local)
from local import *

