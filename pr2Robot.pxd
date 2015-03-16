
cdef class JointConf:
     cdef public conf, robot
     cpdef copy(self)
     cpdef list values(self)
     cpdef list keys(self)
     cpdef get(self, name, default = *)
     cpdef set(self, name, value)
     cpdef placement(self, attached=*)
     cpdef placementMod(self, place, attached=*)
     cpdef placementAux(self, attached=*, getShapes=*)
     cpdef placementModAux(self, place, attached=*, getShapes=*)
     cpdef draw(self, win, color=*, attached=*)
     cpdef str prettyString(self, eq = *)
     cpdef frozenset confItems(self, moveChains=*)
     

cdef class CartConf(JointConf):
     cpdef str frameName(self, name)
     cpdef get(self, name, default = *)
     cpdef set(self, name, value)
     cpdef copy(self)
     
cdef class PR2:
     cdef public chains, color, name, scanner, chainNames, moveChainNames, armChainNames, 
     cdef public gripperChainNames, wristFrameNames, baseChainName, gripperTip, nominalConf
     cdef public horizontalTrans, verticalTrans
     cpdef fingerSupportFrame(self, hand, width)
     cpdef list limits(self, chainNames = *)
     cpdef JointConf randomConf(self, moveChains=*)
     cpdef attach(self, objectPlace, wstate, hand=*)
     cpdef attachRel(self, objectPlace, wstate, hand=*)
     cpdef detach(self, wstate, hand=*)
     cpdef detachRel(self, wstate, hand=*)
     cpdef attachedObj(self, wstate, hand=*)
     cpdef placement(self, conf, wstate=*, getShapes=*, attached=*)
     cpdef placementMod(self, conf, place, wstate=*, getShapes=*, attached=*)
     cpdef placementAux(self, conf, wstate=*, getShapes=*, attached=*)
     cpdef placementModAux(self, conf, place, wstate=*, getShapes=*, attached=*)
     cpdef completeJointConf(self, conf, wstate=*, baseConf=*)
     cpdef stepAlongLine(self, q_f, q_i, stepSize, forward = *, moveChains = *)
     cpdef distConf(self, q1, q2)
     cpdef normConf(self, target, source)
     cpdef forwardKin(self, conf, wstate=*, complain = *, fail = *)

cpdef headInvKin(chains, torso, targetFrame, wstate, collisionAware=*, allowedViewError = *)
cdef list tangentSol(float x, float y, float x0, float y0)
cpdef torsoInvKin(chains, base, target, wstate, collisionAware = *)
cpdef solnDist(sol1, sol2)
