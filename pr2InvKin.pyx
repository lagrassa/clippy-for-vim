import hu
import math
#import numpy as np
import numpy
# noinspection PyUnresolvedReferences
cimport numpy as np
import planGlobals as glob
from traceFile import debug
from miscUtil import prettyString
from transformations import rotation_matrix

# We need this for inverse kinematics
from ctypes import *
if glob.LINUX:
    ik = CDLL(glob.libkinDir+"libkin.so.1.1")
else:
    ik = CDLL(glob.libkinDir+"libkin.dylib")
ik.fkLeft.restype = None
ik.fkRight.restype = None

# The nominal tool offset in ikFast
gripperTip = hu.Pose(0.18,0.0,0.0,0.0)
gripperToolOffset = numpy.dot(gripperTip.matrix,
                           rotation_matrix(math.pi/2,(0,1,0)))

def armInvKin(chains, arm, torso, target, conf, robot,
              collisionAware = False, returnAll = False):
    name = 'ArmInvKin'
    armName = 'pr2LeftArm' if arm=='l' else 'pr2RightArm'
    cfg = conf.copy()             # this will be modified by safeTest
    def safeTest(armAngles):
        #if collisionAware:
        #    cfg[armName] = armAngles
        #    return wstate.robot.safeConf(cfg, wstate)
        #else:
        return True
    # The tool pose relative to the torso frame 
    newHandPoseRel = reduce(numpy.dot, [torso.inverse().matrix,
                                     target.matrix,
                                     gripperToolOffset])
    newArmAngles = pr2KinIKfast(arm, newHandPoseRel, conf[armName],
                                chain=chains.chainsByName[armName],
                                safeTest=safeTest, returnAll=returnAll)
    if False:                            # debug
        print 'arm', arm
        print 'newHandPoseRel\n', newHandPoseRel
        print 'current angles', conf[armName]
        print 'IKfast angles:', newArmAngles
    return newArmAngles

# This should be in Cython...
#create a Ctypes array of type 'type' ('double' or 'int')
def createCtypesArr(ttype, size):
    if ttype == 'int':
        arraytype = c_int * size
    else:
        arraytype = c_double * size
    return arraytype()

nsolnsKin = 10
rotKin = createCtypesArr('float', 9)
transKin = createCtypesArr('float', 3)
freeKin = createCtypesArr('float', 1)
solnsKin = createCtypesArr('float', 7*nsolnsKin)
jtKin = createCtypesArr('float', 7)

cpdef pr2KinIKfast(arm, T, current, chain, safeTest, returnAll = False):
    cdef float bestDist = float('inf')
    cdef float dist
    sols = pr2KinIKfastAll(arm, T, current, chain, safeTest)
    if not sols: return None
    if returnAll:
        return sols
    else:
        if debug('invkin'):
            print 'arm', arm
            print 'current', prettyString(current)
        bestSol = None
        for s in sols:
            dist = solnDist(current, s)
            if dist < bestDist:
                bestDist = dist
                bestSol = s
            if debug('invkin'):
                print 'd=', dist, prettyString(s)
        if debug('invkin'):
            print 'best', prettyString(bestSol)
        return bestSol

cpdef float solnDist(sol1, sol2):
    return max([abs(hu.angleDiff(th1, th2)) for (th1, th2) in zip(sol1, sol2)])

def pr2KinIKfastAll(arm, T, current, chain, safeTest):
    def collectSafe(n):
        sols = []
        for i in range(n):
            sol = solnsKin[i*7 : (i+1)*7]
            if sol and chain.valid(sol):  # inside joint limits
                if (not safeTest) or safeTest(sol): # doesn't collide
                    sols.append(sol)
        return sols
    for i in range(3):
        for j in range(3):
            rotKin[i*3+j] = T[i, j]
    for i in range(3):
        transKin[i] = T[i, 3]
    if arm=='r':
        # bug in right arm kinematics, this is the distance between the shoulders
        transKin[1] += 0.376
    step = glob.IKfastStep
    lower, upper = chain.limits()[2]
    th0 = current[2]
    if not lower <= th0 <= upper: return []
    nsteps = max((upper-th0)/step, (th0-lower)/step)
    solver = ik.ikRight if arm=='r' else ik.ikLeft
    sols = []
    for i in range(int(nsteps)):
        stepsize = i*step
        freeKin[0] = th0 + stepsize
        if freeKin[0] <= upper:
            n = solver(transKin, rotKin, freeKin, nsolnsKin, solnsKin)
            # print 'th', th0 + stepsize, 'n', n
            if n > 0:
                sols.extend(collectSafe(n))
                # if sols: break
        freeKin[0] = th0 - stepsize
        if freeKin[0] >= lower:
            n = solver(transKin, rotKin, freeKin, nsolnsKin, solnsKin)
            # print 'th', th0 - stepsize, 'n', n
            if n > 0:
                sols.extend(collectSafe(n))
                # if sols: break
    # print 'IKFast sols', sols
    return sols
