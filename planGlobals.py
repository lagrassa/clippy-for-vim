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

########################################
# Windows
########################################

# import windowManager3D as wm
# windowWidth = 400

# stderrWin = 'Error'
# debugWin = 'Debug'
# beliefWin = 'Belief'
# worldWin = 'World'
# for win in [stderrWin, debugWin, beliefWin, worldWin]:
#     if not win in wm.windows:
#         wm.makeWindow(win, viewPort = [-6, 6, -6, 6])

########################################
# HACKS
########################################

realWorld = None
horizontal = None
vertical = None

mergeShadows = False
approachBackoff = 0.1
approachPerpBackoff = 0.025

inHeuristic = False

useCC = True                            # Use compiled chains collision checks
skipSearch = False                       # Don't plan paths except for prims
useMathematica = False

########################################
# Robot Parameters
########################################

IKfastStep = 0.1
rrtStep = 0.025
rrtInterpolateStepSize = 10*rrtStep
smoothSteps = 100
torsoZ = 0.15                            # normally 0.3
skipRRT = False
maxRRTIter = 200
failRRTIter = 20

########################################
# Obstacle growing, should be determined by uncertainty
########################################

baseGrowthX = 0.1
baseGrowthY = 0.1

########################################
# FBCH
########################################

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
# Variance parameters
########################################

## LPK Deprecated...try to be sure we're not using them

from miscUtil import makeDiag
# Super good looking!
lookVarianceTuple = (0.001, 0.001)
lookVariance = makeDiag(lookVarianceTuple)
minVarianceTuple = (0.001, 0.001)
minVariance = makeDiag(minVarianceTuple)
minRobotVarianceTuple = minVarianceTuple
minGraspVarianceTuple = minVarianceTuple
minObjectVarianceTuple = minVarianceTuple
maxVarianceTuple = (.1, .1)
minPlaceInVarTuple = (.005, .005)        # Target variance for placing
maxPlaceVarTuple = (.02, .02)            # Target variance for placing

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

