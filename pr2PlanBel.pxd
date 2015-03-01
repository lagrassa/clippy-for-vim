cdef list diagToSq(tuple d)

cdef class PBS:
     cdef public beliefContext, conf, held, graspB, fixObjBs, moveObjBs, regions, 
     cdef public pbs, shadowWorld, shadowProb, avoidShadow, defaultGraspBCache, defaultPlaceBCache, heuristic

     cpdef getWorld(self)
     cpdef getRobot(self)
     cpdef getRoadMap(self)
     cpdef getObjectShapeAtOrigin(self, obj)
     cpdef awayRegions(self)
     cpdef getPlaceB(self, obj, face=*, default=*)
     cpdef defaultPlaceB(self, obj)
     cpdef getGraspB(self, obj, hand, face=*)
     cpdef defaultGraspB(self, obj)
     cpdef getPlacedObjBs(self)
     cpdef getHeld(self, hand)
     cpdef copy(self)
     cpdef updateFromAllPoses(self, goalConds, updateHeld=*, updateConf=*)
     cpdef updateFromGoalPoses(self, goalConds, updateHeld=*, updateConf=*)
     cpdef updateHeld(self, obj, face, graspD, hand, delta=*)
     cpdef updateHeldBel(self, graspB, hand)
     cpdef updateConf(self, c)
     cpdef updatePermObjPose(self, objPlace)
     cpdef updateObjB(self, objPlace)
     cpdef excludeObjs(self, objs)
     cpdef extendFixObjBs(self, objBs, objShapes)
     cpdef getShadowWorld(self, prob, avoidShadow=*)
     cpdef objShadow(self, obj, shName, prob, poseBel, faceFrame)
     cpdef draw(self, p=*, win=*, clear=*)
     cpdef items(self)

cpdef list shadowWidths(tuple variance, tuple delta, float probability)
cpdef sigmaPoses(float prob, poseD, poseDelta)
cpdef makeShadow(shape, prob, bel, name=*)
cpdef LEQ(x, y)
cpdef getGoalPoseBels(goalConds, getFaceFrames)
cpdef getAllPoseBels(overrides, getFaceFrames, curr)
cpdef getConf(overrides, curr)
cpdef getHeldAndGraspBel(overrides, getGraspDesc, currHeld, currGrasp)

		 
