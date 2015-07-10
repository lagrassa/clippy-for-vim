import math
import hu
import numpy as np
import transformations as transf

def getBindings():
    return globals()

########################################
# Directories
########################################

#path = os.getcwd()
#parent = path[:path.rfind('/')] + '/../'
#outDir = parent + 'search/'
outDir = '/Users/tlp/Documents/'
dotSearch = outDir + 's%d%s.dot'
dotSearchX = outDir + 'sx%d%s.dot'
pr2_hpnDir = path + '/'
libkinDir = './IK/'

########################################
# System: LINUX and Mac OS X are supported
########################################

LINUX = False if path[:6] == '/Users' else True

PDB = True

########################################
# ROS: Controlling robot 
########################################

useROS = LINUX
speedFactor = 0.5

########################################
# Useful constant
########################################

ZtoXTr = hu.Transform(np.dot(transf.rotation_matrix(-math.pi/2, (0,1,0)),
                               transf.rotation_matrix(math.pi/2, (1,0,0))))


########################################
# Windows
########################################

stderrWindow = 'Error'

########################################
# Parameters
########################################

IKfastStep = 0.1
rrtStep = 0.025
rrtInterpolateStepSize = 10*rrtStep
smoothSteps = 100
torsoZ = 0.2
skipRRT = False
maxRRTIter = 200
failRRTIter = 10

########################################
# Obstacle growing, should be determined by uncertainty
########################################

baseGrowthX = 0.1
baseGrowthY = 0.1

########################################
# Global communication for 3D CSP
########################################

movableObjectNames = []
CSPworld = None
CSPdomInfo = None
CSPVariableOrder = []
objectApproachOnly = True

########################################
# Debugging
########################################
debugOn = []
pauseOn = []

import local
reload(local)
from local import *

def debug(tag):
    return tag in debugOn
def pause(tag):
    return tag in pauseOn

def debugMsg(tag, *msgs):
    if debug(tag):
        print tag, ':'
        for m in msgs:
            print '    ', m
    if pause(tag):
        raw_input(tag+'-Go?')

def debugDraw(tag, obj, window, color = None):
    if debug(tag):
        obj.draw(window, color = color)
    if pause(tag):
        raw_input(tag+'-Go?')

