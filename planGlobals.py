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

useROS = False
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
# Robot Parameters
########################################

IKfastStep = 0.1
rrtStep = 0.025
rrtInterpolateStepSize = 10*rrtStep
smoothSteps = 100
torsoZ = 0.3
skipRRT = False
maxRRTIter = 200
failRRTIter = 10

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
minTableDim = 5.
cloudPointsResolution = 0.1            # should be 0.1
tableBadWeight = 5

########################################
# Debugging
########################################
debugOn = []
pauseOn = []

import local
reload(local)
from local import *

def debug(tag, skip=False):
    return (not skip) and tag in debugOn
def pause(tag, skip=False):
    return (not skip) and tag in pauseOn

def debugMsg(tag, *msgs):
    if debug(tag):
        print tag, ':'
        for m in msgs:
            print '    ', m
    if pause(tag):
        raw_input(tag+'-Go?')

def debugMsgSkip(tag, skip, *msgs):
    if skip: return
    if debug(tag):
        print tag, ':'
        for m in msgs:
            print '    ', m
    if pause(tag):
        raw_input(tag+'-Go?')

def debugDraw(tag, obj, window, color = None, skip=False):
    if skip: return
    if debug(tag):
        obj.draw(window, color = color)
    if pause(tag):
        raw_input(tag+'-Go?')

