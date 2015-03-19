cimport util
cimport kContents as KC
cimport objModels as om
from cpython cimport bool

cpdef list emptyPair (int i, int j)
cpdef tuple bestTable(om.Polygon zone, om.Object table, KC.Scan scan, list exclude,
                     float res=*, float zthr =*, list angles = *, bool debug = *)
cpdef fillSummed(util.Pose centerPoseInv, tuple ci, list good, list bad, list summed, float res, int size)
cpdef fillSummedIndex(list points, int id, list summed, int size)
cpdef tuple scoreTable(list summed, int i, int j, int dimI, int dimJ)
cpdef tuple bestTablePose(om.Object table, util.Point center, float angle,
                          list good, list bad, list summed, float res, int size, bool debug)
cpdef list anglesList(int n = *)
