cdef class Node:
     cdef public node_idnum, id, conf, cartConf, point

cdef class Edge:
     cdef public edge_idnum, id, nodes, heldCollisions, robotCollisions, heldShapes, robotShapes, bbox

cdef class RoadMap:
     cdef public robot, cartesian, moveChains, kdLeafSize, homeConf, robotShapes
     cdef public root, nodeTooClose, nodes, points, size, kdTree, kNearest, newNodes, 
     cdef public newPoints, newSize, newKDTree, edges, confReachCache, confReachCacheHits
     cdef public confReachPathFails, confReachCacheTotal
     cpdef batchAddNodes(self, list confs)
     cpdef mergeKDTrees(self)
     cpdef pointFromCart(self, cart, alpha=*)
     cpdef addNode(self, conf, merge=*)
     cpdef nearest(self, Node node, int k)
     cpdef close(self, node_f, node_i, thr, cart)
     cpdef drawNodes(self, color=*)
     cpdef interpPose(self, pose_f, pose_i, minLenth, ratio=*)
     cpdef cartInterpolators(self, n_f, n_i, minLength, depth=*)
     cpdef cartLineSteps(self, node_f, node_i, minLength)
     cpdef robotSelfCollide(self, shape)
     cpdef heldSelfCollide(self, shape)
     cpdef colliders(self, node_f, node_i, bState, prob, viol, avoidShadow=*, attached=*)
     