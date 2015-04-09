import util
import numpy as np
cimport numpy as np
import planGlobals as glob

# We need this for inverse kinematics
from ctypes import *
if glob.LINUX:
    ik = CDLL(glob.libkinDir+"libkin.so.1.1")
else:
    ik = CDLL(glob.libkinDir+"libkin.dylib")
ik.fkLeft.restype = None
ik.fkRight.restype = None

def armInvKin(chains, arm, torso, target, tool, conf, robot,
              collisionAware = False, returnAll = False):
    name = 'ArmInvKin'
    armName = 'pr2LeftArm' if arm=='l' else 'pr2RightArm'
    cfg = conf.copy()             # this will be modified by safeTest
    def safeTest(armAngles):
        if collisionAware:
            cfg[armName] = armAngles
            return wstate.robot.safeConf(cfg, wstate)
        else:
            return True
    # The tool pose relative to the torso frame 
    newHandPoseRel = reduce(np.dot, [torso.inverse().matrix,
                                     target.matrix,
                                     tool.matrix])
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
def createCtypesArr(type, size):
    if type == 'int':
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
        bestSol = None
        for s in sols:
            dist = solnDist(current, s)
            if dist < bestDist:
                bestDist = dist
                bestSol = s
        return bestSol

cpdef float solnDist(sol1, sol2):
    total = 0.0
    for (th1, th2) in zip(sol1, sol2):
        total += abs(angleDiff(th1, th2))
    return total

cpdef float angleDiff(float x, float y):
    cdef:
        float twoPi = 2*math.pi
        float z = (x - y)%twoPi
        float w = float(int((x - y)/twoPi))
    if z > math.pi:
        return w*twoPi + (z - twoPi)
    else:
        return w*twoPi + z

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
