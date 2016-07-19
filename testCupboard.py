import testRig
reload(testRig)
from testRig import *

import string

def offsetDims(dims, off):
    def s(d): return [x-y for (x,y) in zip(d,off)]
    return [[s(p) for p in dim] for dim in dims]

cupHalfZ = 0.073172
cupDims = offsetDims([[(0, 0, 0), (0.06, 0.062, 0.145)],
                      [(.06, 0.02, 0.03), (0.088, 0.042, 0.123)]],
                     (0.036347, 0.031000, 0))

bigCupDims = offsetDims([[(0, 0, 0), (0.06, 0.30, 0.25)],
                         [(.06, 0.14, 0.03), (0.088, 0.16, 0.123)]],
                        (0.036347, 0.15, 0))

bigCupDims = offsetDims([[(0, 0, 0), (0.06, 0.30, 0.32)],
                         [(.06, 0.14, 0.03), (0.088, 0.16, 0.123)]],
                        (0.036347, 0.15, 0))

def makeCup(name, color, dims = cupDims):
    return Sh([Ba(cDims, color = color, name=name)  for cDims in dims],
              name = name, color=color)

tableDX, tableDY, tableDZ = 0.3, 0.605, 0.67 # z of table is 0.65 but PR2 thinks not

cupADelta = -0.2
cupPose1 = hu.Pose(cupADelta, 0, tableDZ+0.01, math.pi)
errorDeltaX = -0.1  
errorDeltaY = -0.2
cupPose3 = hu.Pose(errorDeltaX, errorDeltaY, tableDZ+0.01, math.pi)

errorDeltaX4 = -0.1
errorDeltaY4 = .1
cupPose4 = hu.Pose(errorDeltaX4, errorDeltaY4, tableDZ+0.01, math.pi)

cupBDelta = 0
cupPose2 = hu.Pose(cupBDelta, 0, tableDZ+0.01, math.pi)

blockLDelta = -0.2
blockLHalfZ = 0.057412
LDims = offsetDims([[(0.00, 0.00, 0.00),(0.06, 0.06, 0.13)],
                    [(0.06, 0.00, 0.00), (0.096, 0.06, 0.04)]],
                   (0.039328, 0.030000, 0))
blockLBoxes = [Ba(lDims, color = 'gold') for lDims in LDims]
blockLPose = hu.Pose(blockLDelta, 0, tableDZ+0.01, 1.57)
blockL = Sh(blockLBoxes, name = 'blockL')

def cupboard(dx, dy, dz, width, name = 'cupboard', color='brown',
             shelf=False):
    sidePrims = [\
        Ba([(-dx, -dy-width, 0), (dx, -dy, dz)], name='side1'),
        Ba([(-dx, dy, 0), (dx, dy+width, dz)], name='side2'),
        Ba([(dx, -dy, 0), (dx+width, dy, dz)], name='back')]
    shelfPrims = [Ba([(-0.25*dx, -dy-width, dz*shelf),
                      (dx, dy+width, dz*shelf+width)], name='top')] \
                      if shelf else []
    obj = Sh( sidePrims + shelfPrims, name=name)
    return obj

# Full top
def cupboard2(dx, dy, dz, width, name = 'cupboard', color='brown',
             shelf=False):
    sidePrims = [\
        Ba([(-dx, -dy-width, 0), (dx, -dy, dz)], name='side1'),
        Ba([(-dx, dy, 0), (dx, dy+width, dz)], name='side2'),
        Ba([(dx, -dy, 0), (dx+width, dy, dz)], name='back')]
    shelfPrims = [Ba([(-dx, -dy-width, dz*shelf),
                      (dx, dy+width, dz*shelf+width)], name='top')] \
                      if shelf else []
    obj = Sh( sidePrims + shelfPrims, name=name)
    return obj

eps = 0.01
epsz = 0.02
shelfDepth = 0.3
shelfWidth = 0.02

def makeShelves(dx, dy, dz,
                width = 0.02, nshelf = 0,
                name='shelves', color='brown'):
    sidePrims = [\
        Ba([(-dx, -dy-width, 0), (dx, -dy, dz)],
           name=name+'_side_A', color=color),
        Ba([(-dx, dy, 0), (dx, dy+width, dz)],
           name=name+'_side_B', color=color),
        Ba([(dx, -dy, 0), (dx+width, dy, dz)],
           name=name+'_backside', color=color),
        ]
    shelfSpaces = []
    shelfRungs = []
    for i in xrange(nshelf+1):
        frac = i/float(nshelf)
        bot = dz*frac
        top = dz*frac+width
        shelf = Ba([(-dx, -dy-width, bot),
                    (dx, dy+width, bot+width)],
                   color=color,
                   name=name+'_shelf_'+string.ascii_uppercase[i])
        shelfRungs.append(shelf)
        space = Ba([(-dx+eps, -dy-width+eps, eps),
                    (dx-eps, dy+width-eps, (dz/nshelf) - width - eps)],
                   color='green',
                   name=name+'_space_'+str(i+1))
        space = Sh([space], name=name+'_space_'+str(i+1))
        shelfSpaces.append(space)
    obj = Sh( sidePrims + shelfRungs, name=name )
    return (obj, shelfSpaces)

'''
######################################################################
# Define object tree:  cupMap, cupMapMovedTable
######################################################################

def makeTableOnlyMap():
    tableOnlyMap = bm.ObjMap()
    table1 = makeTable(tableDX, tableDY, tableDZ, 'table')
    tableOnlyMap.addObj(table1, hu.Pose(1, 0, 0, 0), None)
    return tableOnlyMap

def makeTwistTableOnlyMap():
    tableOnlyMap = bm.ObjMap()
    table1 = makeTable(tableDX, tableDY, tableDZ, 'table')
    tableOnlyMap.addObj(table1, tableLocCloseTurned, None)
    return tableOnlyMap

useCupboard = True
def makeCupMap(addCupboard = useCupboard):
    cupMap = bm.ObjMap()
    cupMap.addFixedRelObj(room(roomDX0, roomDX1, roomDY0, roomDY1, roomDZ),
                          originPose, None)
    table1 = makeTable(tableDX, tableDY, tableDZ, 'table')
    cupMap.addObj(table1, tableLocFar, None)
    cupMap.addFixedRelObj(easyARegion, easyARegionPose, 'table')
    cupMap.addFixedRelObj(hardARegion, easyARegionPose, 'table')
    cupMap.addFixedRelObj(goal1Region, cupPose1, 'table')
    cupMap.addFixedRelObj(goal2Region, cupPose2, 'table')
    cupMap.addFixedRelObj(warehouse, originPose, 'table')
    cupMap.addObj(makeCup('cupA', 'red'), cupPose1, 'table')
    cupMap.addObj(makeCup('cupB', 'blue'), cupPose2, 'table')
    if addCupboard:
        cupMap.addObj(cupboard(0.25, 0.2, 0.5, 0.02),
                      hu.Pose(0,0,tableDZ+0.001,0), 'table')
    return cupMap

def makeCupMapMovedAll(addCupboard = useCupboard):
    cupMap = bm.ObjMap()
    cupMap.addFixedRelObj(room(roomDX0, roomDX1, roomDY0, roomDY1, roomDZ),
                          originPose, None)
    table1 = makeTable(tableDX, tableDY, tableDZ, 'table')
    tlf = tableLocFar
    tlf2 = hu.Pose(tlf.x+0.2, tlf.y+0.1, tlf.z, tlf.theta + 0.1)
    cp1 = cupPose1
    cp1m = hu.Pose(cp1.x+0.1, cp1.y-0.03, cp1.z, cp1.theta-0.05)
    cp2 = cupPose2
    cp2m = hu.Pose(cp2.x+0.1, cp2.y+0.04, cp2.z, cp2.theta+0.1)
    cupMap.addObj(table1, tlf2, None, variance = landmarkVariance)
    cupMap.addFixedRelObj(easyARegion, easyARegionPose, 'table')
    cupMap.addFixedRelObj(hardARegion, easyARegionPose, 'table')
    cupMap.addFixedRelObj(goal1Region, cupPose1, 'table')
    cupMap.addFixedRelObj(goal2Region, cupPose2, 'table')
    cupMap.addFixedRelObj(warehouse, originPose, 'table')
    cupMap.addObj(makeCup('cupA', 'red'), cp1m, 'table')
    cupMap.addObj(makeCup('cupB', 'blue'), cp2m, 'table', variance = cupBVariance)
    if addCupboard:
        cupMap.addObj(cupboard(0.25, 0.2, 0.5, 0.02),
                      hu.Pose(0,0,tableDZ+0.001,0), 'table')
    return cupMap

def makeEmptyMap():
    cupMap = bm.ObjMap()
    cupMap.addFixedRelObj(room(roomDX0, roomDX1, roomDY0, roomDY1, roomDZ),
                          originPose, None)
    cupMap.addFixedRelObj(roomContents, originPose, 'room')
    return cupMap

def makeBigCupMap(addCupboard = useCupboard):
    cupMap = bm.ObjMap()
    cupMap.addFixedRelObj(room(roomDX0, roomDX1, roomDY0, roomDY1, roomDZ),
                          originPose, None)
    table1 = makeTable(tableDX, longTableDY, tableDZ, 'table')
    cupMap.addObj(table1, tableLocFar, None, variance = landmarkVariance)
    cupMap.addFixedRelObj(easyARegion, easyARegionPose, 'table')
    cupMap.addFixedRelObj(goal1Region, cupPose1, 'table')
    cupMap.addFixedRelObj(goal2Region, cupPose2, 'table')
    cupMap.addFixedRelObj(warehouse, originPose, 'table')
    cupMap.addObj(makeCup('cupA', 'red', dims = bigCupDims), cupPose1, 'table')
    cupMap.addObj(makeCup('cupB', 'blue'), cupPose2, 'table', variance = cupBVariance)
    if addCupboard:
        cupMap.addObj(cupboard(0.25, 0.2, 0.5, 0.02),
                      hu.Pose(0,0,tableDZ+0.001,0), 'table')
    return cupMap

def makeBigCupMapTurnedTable(addCupboard = useCupboard, tableVar = landmarkVariance):
    cupMap = bm.ObjMap()
    cupMap.addFixedRelObj(room(roomDX0, roomDX1, roomDY0, roomDY1, roomDZ),
                          originPose, None)
    table1 = makeTable(tableDX, longTableDY, tableDZ, 'table')
    cupMap.addObj(table1, tableLocCloseTurned, None, variance = tableVar)
    cupMap.addFixedRelObj(easyARegion, easyARegionPose, 'table')
    cupMap.addFixedRelObj(goal1Region, cupPose1, 'table')
    cupMap.addFixedRelObj(goal2Region, cupPose2, 'table')
    cupMap.addFixedRelObj(warehouse, originPose, 'table')
    cupMap.addObj(makeCup('cupA', 'red', dims = bigCupDims), cupPose1, 'table')
    cupMap.addObj(makeCup('cupB', 'blue'), cupPose2, 'table', variance=cupBVariance)
    if addCupboard:
        cupMap.addObj(cupboard(0.25, 0.2, 0.5, 0.02),
                      hu.Pose(0,0,tableDZ+0.001,0), 'table')
    return cupMap


def makeCupMapTurnedTable(addCupboard = useCupboard):
    cupMap = bm.ObjMap()
    cupMap.addFixedRelObj(room(roomDX0, roomDX1, roomDY0, roomDY1, roomDZ),
                          originPose, None)
    table1 = makeTable(tableDX, tableDY, tableDZ, 'table')
    cupMap.addObj(table1, tableLocCloseTurned, None, variance = landmarkVariance)
    cupMap.addFixedRelObj(easyARegion, easyARegionPose, 'table')
    cupMap.addFixedRelObj(goal1Region, cupPose1, 'table')
    cupMap.addFixedRelObj(goal2Region, cupPose2, 'table')
    cupMap.addFixedRelObj(warehouse, originPose, 'table')
    cupMap.addObj(makeCup('cupA', 'red'), cupPose1, 'table')
    cupMap.addObj(makeCup('cupB', 'blue'), cupPose2, 'table')
    if addCupboard:
        cupMap.addObj(cupboard(0.25, 0.2, 0.5, 0.02),
                      hu.Pose(0,0,tableDZ+0.001,0), 'table')
    return cupMap

def makeCupMapTurnedTableNoA(addCupboard = useCupboard):
    cupMap = bm.ObjMap()
    cupMap.addFixedRelObj(room(roomDX0, roomDX1, roomDY0, roomDY1, roomDZ),
                          originPose, None)
    table1 = makeTable(tableDX, longTableDY, tableDZ, 'table')
    cupMap.addObj(table1, tableLocCloseTurned, None, variance = landmarkVariance)
    cupMap.addFixedRelObj(easyARegion, easyARegionPose, 'table')
    cupMap.addFixedRelObj(goal1Region, cupPose1, 'table')
    cupMap.addFixedRelObj(goal2Region, cupPose2, 'table')
    cupMap.addFixedRelObj(warehouse, originPose, 'table')
    cupMap.addObj(makeCup('cupB', 'blue'), cupPose2, 'table')
    if addCupboard:
        cupMap.addObj(cupboard(0.25, 0.2, 0.5, 0.02),
                      hu.Pose(0,0,tableDZ+0.001,0), 'table')
    return cupMap

def makeThreeCupboardMap():
    roomDY0 = roomDY1 = 1.1
    roomDX0, roomDX1 = 1.25, 2.5
    roomDZ = 1.25
    workspace = om.BoxAligned([(-roomDX0,-roomDY0, 0), (roomDX1, roomDY1, roomDZ)],
                              name = 'workspace')
    workspace.solid = False

    tableDX, tableDY, tableDZ = 0.3, 1.0, 0.66
    tableLoc = hu.Pose(2.1, 0, 0, 0.0)

    cbDX, cbDY, cbDZ = 0.25, 0.2, 0.5
    wX, wY = 2.1, 0.6

    tableAbove = om.Object([om.BoxAligned([(wX - 0.3, wY - 0.4, tableDZ+0.01),
                                           (wX + 0.3, wY + 0.4, tableDZ+0.5)])],
                           name='warehouseT', color = 'gray')
    warehouse = om.ParkingRegion(tableAbove,
                                 hu.Point(1,0,0,0),
                                 hu.Point(0,-1,0,0),
                                 name='warehouseT')

    m = bm.ObjMap()
    m.addFixedRelObj(room(roomDX0, roomDX1, roomDY0, roomDY1, roomDZ),
                          originPose, None)
    table1 = makeTable(tableDX, tableDY, tableDZ, 'table')
    m.addObj(table1, tableLoc, None, variance = landmarkVariance)
    m.addFixedRelObj(warehouse, originPose, 'table')
    m.addObj(cupboard(cbDX, cbDY, cbDZ, 0.02, 'cupboard1'),
             hu.Pose(0,0,tableDZ+0.001,0), 'table')
    m.addObj(cupboard(cbDX, cbDY, cbDZ, 0.02, 'cupboard2'),
             hu.Pose(0,-0.4,tableDZ+0.001,0), 'table')
    m.addObj(cupboard(cbDX, cbDY, cbDZ, 0.02, 'cupboard3'),
             hu.Pose(0,-0.8,tableDZ+0.001,0), 'table')

    m.addFixedRelObj(easyARegion, hu.Pose(0, 0.6, tableDZ+0.01,0), 'table')

    m.addObj(makeCup('cupA', 'red'), hu.Pose(-0.1, 0, 0, math.pi),
             'cupboard1')
    m.addObj(makeCup('cupB', 'blue'),  hu.Pose(-0.1, 0, 0, math.pi),
             'cupboard2', variance=cupBVariance)
    m.addObj(makeCup('cupC', 'orange'), hu.Pose(-0.1, 0, 0, math.pi),
             'cupboard3')

    # m.addFixedRelObj(goal1Region, cupPose1, 'table')
    # m.addFixedRelObj(goal2Region, cupPose2, 'table')
    # m.addFixedRelObj(warehouse, originPose, 'table')

    return m, workspace

def makeConstraintTestMap():
    roomDY0 = roomDY1 = 1.1
    roomDX0, roomDX1 = 1.25, 2.5
    roomDZ = 1.25
    workspace = om.BoxAligned([(-roomDX0,-roomDY0, 0), (roomDX1, roomDY1, roomDZ)],
                              name = 'workspace')
    workspace.solid = False

    tableDX, tableDY, tableDZ = 0.3, 1.0, 0.66
    tableLoc = hu.Pose(2.1, 0, 0, 0.0)

    cbDX, cbDY, cbDZ = 0.25, 0.2, 0.5
    m = bm.ObjMap()
    m.addFixedRelObj(room(roomDX0, roomDX1, roomDY0, roomDY1, roomDZ),
                          originPose, None)
    table1 = makeTable(tableDX, tableDY, tableDZ, 'table')
    m.addObj(table1, tableLoc, None, variance = landmarkVariance)
    m.addObj(makeSodaBox('b1'), hu.Pose(0, 0, tableDZ+0.01, 0.4), 'table')
    m.addObj(makeSodaBox('b2'), hu.Pose(.12, 0, tableDZ+0.01, 0.4), 'table')
    m.addObj(makeSodaBox('b3'), hu.Pose(.24, 0, tableDZ+0.01, 0.4), 'table')
    m.addObj(makeSodaBox('b4'), hu.Pose(-.12, 0, tableDZ+0.01, 0.4), 'table')
    m.addObj(makeSodaBox('b5'), hu.Pose(0, -.08, tableDZ+0.01, 0.4), 'table')
    m.addObj(makeSodaBox('b6'), hu.Pose(0, -.16, tableDZ+0.01, 0.4), 'table')
    m.addObj(makeSodaBox('b7'), hu.Pose(0, .08, tableDZ+0.01, 0.4), 'table')
    m.addObj(makeSodaBox('b8'), hu.Pose(0, .16, tableDZ+0.01, 0.4), 'table')

    return m, workspace

def makeConstraintTestMapAllCentered():
    roomDY0 = roomDY1 = 1.1
    roomDX0, roomDX1 = 1.25, 2.5
    roomDZ = 1.25
    workspace = om.BoxAligned([(-roomDX0,-roomDY0, 0), (roomDX1, roomDY1, roomDZ)],
                              name = 'workspace')
    workspace.solid = False

    tableDX, tableDY, tableDZ = 0.3, 1.0, 0.66
    tableLoc = hu.Pose(2.1, 0, 0, 0.0)

    cbDX, cbDY, cbDZ = 0.25, 0.2, 0.5
    m = bm.ObjMap()
    m.addFixedRelObj(room(roomDX0, roomDX1, roomDY0, roomDY1, roomDZ),
                          originPose, None)
    table1 = makeTable(tableDX, tableDY, tableDZ, 'table')
    m.addObj(table1, tableLoc, None, variance = landmarkVariance)
    m.addObj(makeSodaBox('b1'), hu.Pose(0, 0, tableDZ+0.01, 0.4), 'table')
    m.addObj(makeSodaBox('b2'), hu.Pose(0, 0, tableDZ+0.01, 0.4), 'table')
    m.addObj(makeSodaBox('b3'), hu.Pose(0, 0, tableDZ+0.01, 0.4), 'table')
    m.addObj(makeSodaBox('b4'), hu.Pose(0, 0, tableDZ+0.01, 0.4), 'table')

    m.addObj(makeSodaBox('b5'), hu.Pose(0, 0, tableDZ+0.01, 0.4), 'table')
    m.addObj(makeSodaBox('b6'), hu.Pose(0, 0, tableDZ+0.01, 0.4), 'table')
    m.addObj(makeSodaBox('b7'), hu.Pose(0, 0, tableDZ+0.01, 0.4), 'table')
    m.addObj(makeSodaBox('b8'), hu.Pose(0, 0, tableDZ+0.01, 0.4), 'table')

    return m, workspace


def makeSimpleCupMap():
    cupMap = bm.ObjMap()
    cupMap.addFixedRelObj(room(roomDX0, roomDX1, roomDY0, roomDY1, roomDZ),
                          originPose, None)
    table1 = makeTable(tableDX, tableDY, tableDZ, 'table')
    cupMap.addObj(table1, tableLocClose, None, variance = landmarkVariance)
    cupMap.addFixedRelObj(warehouse, originPose, 'table')
    cupMap.addObj(makeCup('cupA', 'red'), cupPose1, 'table')
    #cupMap.addObj(makeCup('cupB', 'blue'), cupPose2, 'table')
    #cupMap.addObj(cupboard(0.25, 0.2, 0.25, 0.02),
    #              hu.Pose(0,0,tableDZ+0.001,0), 'table')
    return cupMap

######################################################################
# Define object tree:  soupSodaMap
######################################################################
sodaBoxHalfZ = 0.06
sodaBoxDims = [(-0.045, -0.025, 0.0), (0.045, 0.025, 0.12)]

def makeSodaBox(name = 'soda'):
    return om.Object([om.BoxAligned(sodaBoxDims, color = 'blue')], name=name,
                     typeName = 'soda')

sodaBoxPose = hu.Pose(0, 0, tableDZ+0.01, 0.4)
sodaBoxPose2 = hu.Pose(-0.1, 0.1, tableDZ+0.01, 0)
sodaBoxPoseStraight = hu.Pose(0, 0, tableDZ+0.01, 0.0)

soupCanHalfZ = 0.05
soupCan = om.Object([om.Ngon(0.0675/2, (0, 2*soupCanHalfZ), 6, color = 'red')],
                    name='soup')
soupCanPose = hu.Pose(-0.2, 0, tableDZ+0.01, math.pi/16)
soupCanOverPose = hu.Pose(0.2, 0, tableDZ+0.01, math.pi/16)
soupCanPose2 = hu.Pose(-0.1, -0.15, tableDZ+0.01, math.pi/16)

def makeSSMapTwoTable(tableVar = landmarkVariance, includeSS = True):
    tableLocClose = hu.Pose(1.25, tableYOffset, 0, 0.1)
    tableLoc2 = hu.Pose(2.0, -1.45, 0, 1.57)
    ## remember, half widths in x and y
    tableDX2, tableDY2, tableDZ2 = [x*0.0254 for x in (12, 21, 28.75)]
    map = bm.ObjMap()
    map.addFixedRelObj(room(roomDX0, roomDX1, roomDY0, roomDY1, roomDZ),
                          originPose, None)
    table1 = makeTable(tableDX, tableDY, tableDZ, 'table')
    map.addObj(table1, tableLocClose, None, variance = tableVar)
    table2 = makeTable(tableDX2, tableDY2, tableDZ2, 'table2')
    map.addObj(table2, tableLoc2, None, variance = tableVar)
    easyARegion = om.Region(om.BoxAligned(om.bboxGrow(sodaBoxDims, 0.1, 0.005),
                                          opacity=0.5, color='purple'),
                            name='easyARegion')
    easyARegionPose = hu.Pose(0.2, 0.4, tableDZ2+0.01, 0)
    map.addFixedRelObj(easyARegion, easyARegionPose, 'table2')
    map.addFixedRelObj(warehouse, originPose, 'table')

    aboveTable = om.Object([\
        om.BoxAligned([(-tableDX*2, -tableDY*2, 0),
                       (tableDX*2, tableDY*2, 0.5)],
                      name='workspace', color='cyan',
                      perm=False, solid = False)], name = 'workspace')
    aboveTable.solid = False
    map.addFixedRelObj(aboveTable, hu.Pose(0, 0, tableDZ, 0), 'table')
    if includeSS: 
        map.addObj(soupCan('red', 'chicken'), soupCanPose, 'table')
        map.addObj(makeSodaBox(), sodaBoxPoseStraight, 'table')
    return map

def makeSSMap(tableVar = landmarkVariance, includeSS = True):
    tableLocClose = hu.Pose(1.25, tableYOffset, 0, 0.1)
    tableLocFar = hu.Pose(2.0, tableYOffset, 0, 0.1)
    map = bm.ObjMap()
    map.addFixedRelObj(room(roomDX0, roomDX1, roomDY0, roomDY1, roomDZ),
                          originPose, None)
    table1 = makeTable(tableDX, tableDY, tableDZ, 'table')
    tableAbove = om.Object([om.BoxAligned([(-tableDX+0.05,
                                            -tableDY, tableDZ+0.01),
                                           (tableDX-0.05, # 0 for smaller
                                            -0.2, tableDZ+0.5)])],
                           name='warehouseT', color = 'gray')
    warehouseT = om.ParkingRegion(tableAbove,
                                 hu.Point(1,0,0,0),
                                 hu.Point(0,-1,0,0),
                                 name='warehouseT')
    easyARegion = om.Region(om.BoxAligned([(-tableDX+0.05, 0.3, 0),
                                       (tableDX-0.05, tableDY, 0.5)],
                                          opacity=0.5, color='purple'),
                        name='easyARegion')
    easyARegionPose = hu.Pose(0, 0, tableDZ+0.01, 0)
    if includeSS: 
        map.addObj(table1, tableLocFar, None, variance = tableVar)
        map.addObj(soupCan('red', 'chicken'), soupCanPose, 'table')
        map.addObj(makeSodaBox(), sodaBoxPoseStraight, 'table')
    else:
        map.addUnlocalizedObj(table1, dist.DDist({'workspace': 0.2,
                                                  'workspaceFront':0.8}),
                              'table')
        map.addUnlocalizedObj(makeSodaBox(), dist.DDist({'aboveTable' : 0.9,
                                                         'workspace': 0.1}),
                              'soda')
    map.addFixedRelObj(easyARegion, easyARegionPose, 'table') 
    map.addFixedRelObj(warehouseT, originPose, 'table')
    map.addImplicitObj(['warehouseT'], [], 'warehouse')

    aboveTable = om.Object([\
        om.BoxAligned([(-tableDX, -tableDY, 0),
                       (tableDX, tableDY, 0.5)],
                      name='aboveTable', color='cyan',
                      perm=False, solid = False)], name = 'aboveTable')
    aboveTable.solid = False
    map.addFixedRelObj(aboveTable, hu.Pose(0, 0, tableDZ, 0), 'table')
    workspaceObj = om.Object([workspace], name = 'workspace')
    workspaceObj.solid = False
    map.addFixedRelObj(workspaceObj, hu.Pose(0, 0, 0, 0), None)
    
    workspaceBackObj = om.Object([workspaceBack], name = 'workspaceBack')
    workspaceBackObj.solid = False
    map.addFixedRelObj(workspaceBackObj, hu.Pose(0, 0, 0, 0), None)

    workspaceFrontObj = om.Object([workspaceFront], name = 'workspaceFront')
    workspaceFrontObj.solid = False
    map.addFixedRelObj(workspaceFrontObj, hu.Pose(0, 0, 0, 0), None)

    return map


def makeSodaMap(tableVar = landmarkVariance):
    tableLocClose = hu.Pose(1.25, tableYOffset, 0, 0.1)
    tableLocFar = hu.Pose(2.0, tableYOffset, 0, 0.1)
    map = bm.ObjMap()
    map.addFixedRelObj(room(roomDX0, roomDX1, roomDY0, roomDY1, roomDZ),
                          originPose, None)
    table1 = makeTable(tableDX, tableDY, tableDZ, 'table')
    easyARegion = om.Region(om.BoxAligned(om.bboxGrow(sodaBoxDims, 0.1, 0.1),
                                          opacity=0.5, color='purple'),
                            name='easyARegion')
    easyARegionPose = hu.Pose(-0.2, 0.2, tableDZ+0.11, 0)
    easyBRegion = om.Region(om.BoxAligned(om.bboxGrow(sodaBoxDims, 0.1, 0.1),
                                          opacity=0.5, color='orange'),
                            name='easyBRegion')
    easyBRegionPose = hu.Pose(-0.2, -0.2, tableDZ+0.11, 0)
    map.addObj(table1, tableLocFar, None, variance = tableVar)
    map.addObj(makeSodaBox(), sodaBoxPoseStraight, 'table')
    map.addFixedRelObj(easyARegion, easyARegionPose, 'table') 
    map.addFixedRelObj(easyBRegion, easyBRegionPose, 'table')
    map.addFixedRelObj(warehouse, originPose, 'table')

    aboveTable = om.Object([\
        om.BoxAligned([(-tableDX, -tableDY, tableDZ),
                       (tableDX, tableDY, tableDZ + 0.5)],
                      name='aboveTable', color='cyan',
                      perm=False, solid = False)], name = 'aboveTable')
    aboveTable.solid = False
    map.addFixedRelObj(aboveTable, hu.Pose(0, 0, 0, 0), 'table')
    workspaceObj = om.Object([workspace], name = 'workspace')
    workspaceObj.solid = False
    map.addFixedRelObj(workspaceObj, hu.Pose(0, 0, 0, 0), None)
    return map

tableDX2, tableDY2, tableDZ2 = [x*0.0254 for x in (12, 21, 28.75)]
tableLoc2 = hu.Pose(-0.2, -1.3, 0, 1.0)


# New means that it has shelves
# Declare that there is a single 'soda'
def makeSSMapNew(tableVar = landmarkVariance, chickOff = 0.1, includeSS = True,
                 twoTables = True):
    tableLocClose = hu.Pose(2.0, tableYOffset, 0, 0.1)
    offZ = 0.19                         # was 0.22
    map = bm.ObjMap()
    map.addFixedRelObj(room(roomDX0, roomDX1, roomDY0, roomDY1, roomDZ),
                          originPose, None)    
    table1 = makeTable(tableDX, tableDY, tableDZ, 'table1', typeName = 'table1')
    if twoTables:
        easyARegion = om.Region(om.BoxAligned([(-tableDX+0.05, -0.2, 0),
                                               (tableDX-0.05, 0.2, 0.5)],
                                              opacity=0.5, color='purple'),
                                name='easyARegion')
    else:
        easyARegion = om.Region(om.BoxAligned([(-tableDX+0.05, -0.05, 0),
                                               (tableDX-0.05, 0.2, 0.5)],
                                              opacity=0.5, color='purple'),
                                name='easyARegion')
    (shelves, aboveShelves) = makeShelves(shelfDepth/2.0, 0.305,
                                          0.45, width=0.02, nshelf=2)
    cooler = om.Object([Ba([(-0.12, -0.165, 0), (0.12, 0.165, offZ)])],
                       name='cooler')
    cooler.permanent = True
    soupCanPose = hu.Pose(-0.1, 0 + chickOff, 0.04, math.pi/16)
    sodaBoxPoseStraight = hu.Pose(-0.1, 0.0, 0.26, 0.0)

    workspaceObj = om.Object([workspace], name = 'workspace')
    workspaceObj.solid = False
    map.addFixedRelObj(workspaceObj, hu.Pose(0, 0, 0, 0), None)

    workspaceBackObj = om.Object([workspaceBack], name = 'workspaceBack')
    workspaceBackObj.solid = False
    map.addFixedRelObj(workspaceBackObj, hu.Pose(0, 0, 0, 0), None)

    workspaceFrontObj = om.Object([workspaceFront], name = 'workspaceFront')
    workspaceFrontObj.solid = False
    map.addFixedRelObj(workspaceFrontObj, hu.Pose(0, 0, 0, 0), None)

    if twoTables:
        table2 = makeTable(tableDX2, tableDY2, tableDZ2, 'table2',
                           typeName = 'table2')
        tableLoc2 = hu.Pose(-0.8, -1.1, 0, 0)
        # Doesn't work
        # tableLoc2 = hu.Pose(-0.3, -1.1, 0, 1.0)
    
    
    if includeSS: 
        map.addObj(table1, tableLocClose, None, variance = tableVar)
        if twoTables:
            map.addObj(table2, tableLoc2, None, variance = tableVar)    
        map.addFixedRelObj(shelves,
                       hu.Pose(0,0,tableDZ+0.001+offZ,0), 'table1')
        map.addObj(soupCan('red', 'chicken'), soupCanPose, 'shelves')
        map.addObj(makeSodaBox(), sodaBoxPoseStraight, 'shelves')
    else:
        # These objects are guaranteed to be singular instances of
        # their type
        map.addUnlocalizedObj(table1, dist.DDist({'workspaceFront': 1.0}),
                              'table1')
        if twoTables:
            map.addUnlocalizedObj(table2, dist.DDist({'workspaceBack': 1.0}),
                                  'table2')
        map.addFixedRelObj(shelves,
                       hu.Pose(0,0,tableDZ+0.001+offZ,0), 'table1')
        map.addFixedRelObj(aboveShelves[0][0], aboveShelves[0][1], 'shelves')
        map.addFixedRelObj(aboveShelves[1][0], aboveShelves[1][1], 'shelves')
        map.addUnlocalizedObj(makeSodaBox(),
                              dist.DDist({aboveShelves[0][0].getBaseName() : 0.45,
                                          aboveShelves[1][0].getBaseName() : 0.45,
                                          'workspace': 0.1}),
                              'soda')        
    map.addFixedRelObj(cooler, hu.Pose(0,0,tableDZ+0.001,0), 'table1')
    if twoTables:
        easyARegionPose = hu.Pose(0, 0.3, 0.01+tableDZ2, 0) # 0.04
        map.addFixedRelObj(easyARegion, easyARegionPose, 'table2')
    else:
        easyARegionPose = hu.Pose(0, 0.4, 0.01+tableDZ, 0) # 0.04
        map.addFixedRelObj(easyARegion, easyARegionPose, 'table1')

    wh = om.Object([om.BoxAligned([(-tableDX+0.05,-tableDY, tableDZ+0.01),
                                   (tableDX-0.05, -0.35, tableDZ+0.5)])],
                   name='warehouseT', color = 'gray')
    warehouseT = om.ParkingRegion(wh,
                                 hu.Point(1,0,0,0),
                                 hu.Point(0,-1,0,0),
                                 name='warehouseT')
    map.addFixedRelObj(warehouseT, originPose, 'table1')
    map.addImplicitObj(['warehouseT'], [], 'warehouse')
    return map

# old
def makeSSMapSmallRoom():
    map = bm.ObjMap()
    map.addFixedRelObj(room(0.75, 2.0, roomDY0, roomDY1, roomDZ),
                          originPose, None)
    table1 = makeTable(tableDX, tableDY, tableDZ, 'table')
    map.addObj(table1, tableLocClose, None, variance = landmarkVariance)
    map.addFixedRelObj(easyARegion, easyARegionPose, 'table')
    map.addFixedRelObj(goal1Region, cupPose1, 'table')
    map.addFixedRelObj(goal2Region, cupPose2, 'table')
    map.addFixedRelObj(warehouse, originPose, 'table')
    #map.addObj(soupCan, soupCanPose, 'table')
    map.addObj(makeSodaBox(), sodaBoxPoseStraight, 'table')
    #map.addObj(cupboard(0.25, 0.2, 0.25, 0.02),
                   #hu.Pose(0,0,tableDZ+0.001,0), 'table')
    return map


# old
def makeSSMapMovedTable():
    map = bm.ObjMap()
    map.addFixedRelObj(room(roomDX0, roomDX1, roomDY0, roomDY1, roomDZ),
                          originPose, None)
    table1 = makeTable(tableDX, tableDY, tableDZ, 'table')
    map.addObj(table1, tableLocMed, None, variance = landmarkVariance)
    map.addFixedRelObj(easyARegion, easyARegionPose, 'table')
    map.addFixedRelObj(goal1Region, cupPose1, 'table')
    map.addFixedRelObj(goal2Region, cupPose2, 'table')
    map.addFixedRelObj(warehouse, originPose, 'table')
    #map.addObj(soupCan, soupCanPose, 'table')
    map.addObj(makeSodaBox(), sodaBoxPoseStraight, 'table')
    #map.addObj(cupboard(0.25, 0.2, 0.25, 0.02),
                   #hu.Pose(0,0,tableDZ+0.001,0), 'table')
    return map

if glob.observable:
    percepts = {}
else:
    ident = transf.rotation_matrix(0, (1,0,0))
    trans =transf.translation_matrix
    rot = transf.rotation_matrix
    percepts = {'table' : (
                          trans((0.0, 0.0, 0.0)),
                          trans((0.0, 0.0, 0.0)),
                          2),
                'table1' : (
                          trans((0.0, 0.0, 0.0)),
                          trans((0.0, 0.0, 0.0)),
                          2),
                'table2' : (
                          trans((0.0, 0.0, 0.0)),
                          trans((0.0, 0.0, 0.0)),
                          2),
                'RectCup' : (
                             numpy.dot(rot(-math.pi/2, (1,0,0)),
                                       trans((0.0, 0.0, -cupHalfZ))),
                             numpy.dot(rot(math.pi/2, (1,0,0)),
                                       trans((0.0, 0.0, cupHalfZ))),
                             1),
                'LBlock' : (
                             numpy.dot(rot(-math.pi/2, (1,0,0)),
                                       trans((0.0, 0.0, -blockLHalfZ))),
                             numpy.dot(rot(math.pi/2, (1,0,0)),
                                       trans((0.0, 0.0, blockLHalfZ))),
                            1),
                'soda' : (
                          trans((0.0, 0.0, -sodaBoxHalfZ)),
                          numpy.dot(rot(math.pi, (1,0,0)),
                                    trans((0.0, 0.0, -sodaBoxHalfZ))),
                          2),
                'soup16' : (
                            trans((0.0, 0.0, -soupCanHalfZ)),
                            numpy.dot(rot(math.pi, (1,0,0)),
                                    trans((0.0, 0.0, -soupCanHalfZ))),
                          90)               # within 4 degrees
                }

def makeTwoTableMap(deltaX=0.02, deltaY=0.02, deltaTheta=0.05, tableVar = 0.0001):
    def rn(x): return 2*x*random.random() - x
    def rnx(): return rn(deltaX)
    def rny(): return rn(deltaY)
    def rnt(): return rn(deltaTheta)

    glob.referenceObject = 'table2'

    diff = deltaX or deltaY or deltaTheta
    roomDY0, roomDY1 = 1.5, 2.0
    roomDX0, roomDX1 = 1.25, 2.5
    roomDZ = 1.25
    workspace = om.BoxAligned([(-roomDX0,-roomDY0, 0), (roomDX1, roomDY1, roomDZ)],
                              name = 'workspace')
    workspace.solid = False

    tableDX, tableDY, tableDZ = 0.3, 0.8, 0.68
    tableLoc = hu.Pose(2.1+rnx(), 0+rny(), 0, 0.0+rnt())
    tableLoc2 = hu.Pose(0.5+rnx(), 1.6+rny(), 0, 1.57+rnt())
    cbDX, cbDY, cbDZ = 0.25, 0.2, 0.5
    warehouseDims = [(-tableDX+0.05, -tableDY, tableDZ+0.01),
                     ( tableDX-0.05, -0.2, tableDZ+0.5)]

    tableAbove1 = om.Object([om.BoxAligned(warehouseDims)],
                            name='warehouse1', color = 'gray')
    warehouse1 = om.ParkingRegion(tableAbove1,
                                 hu.Point(1,0,0,0), hu.Point(0,-1,0,0),
                                 name='warehouse1')
    tableAbove2 = om.Object([om.BoxAligned(warehouseDims)],
                            name='warehouse2', color = 'green')
    warehouse2 = om.ParkingRegion(tableAbove2,
                                 hu.Point(1,0,0,0), hu.Point(0,-1,0,0),
                                 name='warehouse2')
    m = bm.ObjMap()
    m.addFixedRelObj(room(roomDX0, roomDX1, roomDY0, roomDY1, roomDZ),
                          originPose, None)
    table1 = makeTable(tableDX, tableDY, tableDZ, 'table')
    m.addObj(table1, tableLoc, None, variance = tableVar)
    table2 = makeTable(tableDX, tableDY, tableDZ, 'table2')
    m.addObj(table2, tableLoc2, None, variance = tableVar)
    m.addFixedRelObj(warehouse1, originPose, 'table')
    m.addFixedRelObj(warehouse2, originPose, 'table2')

    m.addObj(cupboard(cbDX, cbDY, cbDZ, 0.02, 'cupboard1'),
             hu.Pose(0+rnx(),0+rny(),tableDZ+0.001,0+rnt()), 'table')
    m.addObj(cupboard(cbDX, cbDY, cbDZ, 0.02, 'cupboard2', shelf=0.5),
             hu.Pose(0+rnx(),0+rny(),tableDZ+0.001,0+rnt()), 'table2')

    targetBox = om.bboxGrow(cupDims[0], 0.1, 0.02) # was 0.05
    cupboard2TargetRegion = om.Region(om.BoxAligned(targetBox,
                                                    opacity=0.5, color='purple'),
                                      name='cupboard2TargetRegion')
    m.addFixedRelObj(cupboard2TargetRegion, hu.Pose(0, 0, 0.5*cbDZ+0.05,0), 'cupboard2') # added to z
    table2TargetRegion = om.Region(om.BoxAligned(targetBox,
                                                 opacity=0.5, color='purple'),
                                   name='table2TargetRegion')
    m.addFixedRelObj(table2TargetRegion, hu.Pose(0, 0, tableDZ+0.03,0), 'table2') # added to z

    cupSep = 0.01
    m.addObj(makeCup('cupA', 'red'), hu.Pose(-0.1+rnx(), 0+rny(), cupSep,
                                               math.pi+rnt()),
             'cupboard1', variance = 0.001)
    m.addObj(makeCup('cupB', 'blue'),  hu.Pose(0.1+rnx(), 0+rny(), cupSep,
                                                 math.pi+rnt()),
             'cupboard1', variance = 0.001)

    rot = 0.5 if diff else 0.0
    cupPoseC = hu.Pose(-0.2+rnx(), 0+rny(), tableDZ+0.01, math.pi-rot+rnt())
    m.addObj(makeCup('cupC', 'green'), cupPoseC, 'table2')
    return m, workspace
    
# def makeMultiCupboardMap(withObjects = True,
#                          deltaX=0.0, deltaY=0.0, deltaTheta=0.0):
#     def rn(x): return 2*x*random.random() - x
#     def rnx(): return rn(deltaX)
#     def rny(): return rn(deltaY)
#     def rnt(): return rn(deltaTheta)

#     glob.referenceObject = 'table2'
#     smallVariance = 0.01**2
#     roomDY0 = roomDY1 = 1.75
#     roomDX0, roomDX1 = 1.25, 2.5
#     roomDZ = 1.25
#     workspace = om.BoxAligned([(-roomDX0,-roomDY0, 0),
#                                (roomDX1, roomDY1, roomDZ)],
#                               name = 'workspace')
#     workspace.solid = False

#     tableDX, tableDY, tableDZ = 0.3, 1.0, 0.68
#     tableLoc = hu.Pose(2.1+rnx(), 0+rny(), 0, 0.0+rnt())
#     tableLoc2 = hu.Pose(0.5+rnx(), 1.25+rny(), 0, 1.57+rnt())
#     cbDX, cbDY, cbDZ = 0.25, 0.2, 0.5
#     warehouseDims = [(-tableDX+0.05, -tableDY, tableDZ+0.01),
#                      ( tableDX-0.05, -0.2, roomDZ+0.01)]

#     tableAbove1 = om.Object([om.BoxAligned(warehouseDims)],
#                             name='warehouse1', color = 'gray')
#     warehouse1 = om.ParkingRegion(tableAbove1,
#                                  hu.Point(1,0,0,0), hu.Point(0,-1,0,0),
#                                  name='warehouse1')
#     tableAbove2 = om.Object([om.BoxAligned(warehouseDims)],
#                             name='warehouse2', color = 'green')
#     warehouse2 = om.ParkingRegion(tableAbove2,
#                                  hu.Point(1,0,0,0), hu.Point(0,-1,0,0),
#                                  name='warehouse2')
#     m = bm.ObjMap()
#     m.addFixedRelObj(room(roomDX0, roomDX1, roomDY0, roomDY1, roomDZ),
#                           originPose, None)
#     table1 = makeTable(tableDX, tableDY, tableDZ, 'table',
#                        typeName = 'table')
#     m.addObj(table1, tableLoc, None, variance = smallVariance)
#     table2 = makeTable(tableDX, tableDY, tableDZ, 'table2',
#                        typeName = 'table2')
#     m.addObj(table2, tableLoc2, None, variance = smallVariance)
#     m.addFixedRelObj(warehouse1, originPose, 'table')
#     m.addFixedRelObj(warehouse2, originPose, 'table2')

#     cDims = [(-cbDX-0.05, -cbDY, 0), (cbDX, cbDY, 0.145)]
#     m.addObj(cupboard(cbDX, cbDY, cbDZ, 0.02, 'cupboard1A'),
#              hu.Pose(0+rnx(),0+rny(),tableDZ+0.001,0+rnt()), 'table',
#              variance = smallVariance)
#     inC1A = om.Region(om.BoxAligned(cDims, opacity=0.5, color='purple'),
#                       name='inC1A')
#     m.addFixedRelObj(inC1A, hu.Pose(0, 0, 0, 0), 'cupboard1A')
#     m.addObj(cupboard(cbDX, cbDY, cbDZ, 0.02, 'cupboard1B'),
#              hu.Pose(0+rnx(),2*cbDY+rny(),tableDZ+0.001,0+rnt()), 'table',
#              variance = smallVariance)
#     inC1B = om.Region(om.BoxAligned(cDims, opacity=0.5, color='purple'),
#                       name='inC1B')
#     m.addFixedRelObj(inC1B, hu.Pose(0, 0, 0, 0), 'cupboard1B')
#     m.addObj(cupboard(cbDX, cbDY, cbDZ, 0.02, 'cupboard2A'),
#              hu.Pose(0+rnx(),0+rny(),tableDZ+0.001,0+rnt()), 'table2',
#              variance = smallVariance)
#     inC2A = om.Region(om.BoxAligned(cDims, opacity=0.5, color='purple'),
#                       name='inC2A')
#     m.addFixedRelObj(inC2A, hu.Pose(0, 0, 0, 0), 'cupboard2A')
#     m.addObj(cupboard(cbDX, cbDY, cbDZ, 0.02, 'cupboard2B'),
#              hu.Pose(0+rnx(),2*cbDY+rny(),tableDZ+0.001,0+rnt()), 'table2',
#              variance = smallVariance)
#     inC2B = om.Region(om.BoxAligned(cDims, opacity=0.5, color='purple'),
#                       name='inC2B')
#     m.addFixedRelObj(inC2B, hu.Pose(0, 0, 0, 0), 'cupboard2B')

#     belowTables = om.BoxAligned([(-roomDX0,-roomDY0, 0),
#                                  (roomDX1, roomDY1, tableDZ-0.05)])
#     openRoom = om.BoxAligned([(-roomDX0,-roomDY0, 0), (1.75, 0.9, roomDZ+0.1)])
#     glob.knownSpaces = [belowTables, openRoom,
#                         warehouse1.applyRefPose(m.worldPoseMean('warehouse1')),
#                         warehouse2.applyRefPose(m.worldPoseMean('warehouse2'))]

#     if not withObjects:
#         return m, workspace

#     cupSep = 0.01
#     dx = (2*cbDX)/3.

#     for c, rows in (('1A', [[(bigCup, 'red')], [(None, None)], [(bigCup, 'red')]]),
#                     ('1B', [[(bigCup, 'red')], 2*[(soupCan, 'red')]]),
#                     ('2A', [[(bigCup, 'red')], 2*[(sodaBox, 'blue')], 2*[(soupCan,'red')]]),
#                     ('2B', [[(smallCup, 'red'), (sodaBox, 'blue')],
#                             [(smallCup, 'red'), (smallCup, 'green')]])):
#         for i in range(len(rows)):
#             col = rows[i]
#             ncol = len(col)
#             if ncol == 1: dy = (2.*cbDY)/(ncol+1)
#             else: dy = (2.*cbDY+0.02)/(ncol+1)
#             for j, (obj, color) in enumerate(col):
#                 if obj:
#                     m.addObj(obj('c%s_%d_%d'%(c,i+1,j+1), color),
#                              hu.Pose(-cbDX+0.02+i*dx+rnx(),
#                                        -cbDY+(j+1)*dy+rny(),
#                                        cupSep, math.pi+rnt()),
#                              'cupboard%s'%c, variance = 0.001)
#     return m, workspace

def makeMultiCupboardMap2(withObjects = True,
                          deltaX=0.0, deltaY=0.0, deltaTheta=0.0,
                          randomWorld = False,
                          greenKnown = False):
    def rn(x): return 2*x*random.random() - x
    def rnx(): return rn(deltaX)
    def rny(): return rn(deltaY)
    def rnt(): return rn(deltaTheta)

    glob.referenceObject = 'table2'
    smallVariance = 0.01**2
    roomDY0 = roomDY1 = 1.75
    roomDX0, roomDX1 = 1.25, 2.5
    roomDZ = 1.25

    tableDX, tableDY, tableDZ = 0.3, 1.25, 0.68
    tableLoc = hu.Pose(2.1, -0.25, 0, 0.0)
    #tableLoc2 = hu.Pose(0.5, 1.25, 0, 1.57)
    tableLoc2 = hu.Pose(0.1, 1.4, 0, 1.57)
    cbDX, cbDY, cbDZ = 0.2, 0.24, 0.4
    
    workspace = om.BoxAligned([(-roomDX0,-roomDY0, 0),
                               (roomDX1, roomDY1, roomDZ)],
                              name = 'workspace')
    workspace.solid = False
    workspaceObj = om.Object([workspace], name = 'workspace')
    workspaceObj.solid = False

    knownspace = om.BoxAligned([(-roomDX0,-roomDY0, 0),
                                (roomDX1-2*tableDX, roomDY1-2*tableDX, roomDZ)],
                              name = 'knownspace')
    knownspace.solid = False
    knownspaceObj = om.Object([knownspace], name = 'knownspace')
    knownspaceObj.solid = False


    warehouseDims = [(-tableDX+0.05, -tableDY+0.05, tableDZ+0.01),
                     ( tableDX-0.05, -0.3, tableDZ+0.36)]

    bigwarehouseDims = [(-tableDX-0.05, -tableDY, tableDZ+0.01),
                        ( tableDX+0.05, -0.2, tableDZ+0.8)]

    tableAbove1 = om.Object([om.BoxAligned(warehouseDims)],
                            name='warehouse1', color = 'gray')
    warehouse1 = om.ParkingRegion(tableAbove1,
                                 hu.Point(-1,0,0.01,0), hu.Point(0,-1,0,0),
                                 name='warehouse1')
    tableAbove2 = om.Object([om.BoxAligned(warehouseDims)],
                            name='warehouse2', color = 'green')
    warehouse2 = om.ParkingRegion(tableAbove2,
                                 hu.Point(-1,0,0.01,0), hu.Point(0,-1,0,0),
                                 name='warehouse2')

    cDims = [(-cbDX+0.01, -cbDY+0.01, 0.0), (cbDX-0.01, cbDY-0.01, 0.33)]
    inC1A = om.Region(om.BoxAligned(cDims, opacity=0.5, color='purple'),
                      name='inC1A')
    inC2A = om.Region(om.BoxAligned(cDims, opacity=0.5, color='purple'),
                      name='inC2A')
    inC1B = om.Region(om.BoxAligned(cDims, opacity=0.5, color='purple'),
                      name='inC1B')
    inC2B = om.Region(om.BoxAligned(cDims, opacity=0.5, color='purple'),
                      name='inC2B')
    m = bm.ObjMap()
    m.addFixedRelObj(workspaceObj, hu.Pose(0, 0, 0, 0), None)
    m.addFixedRelObj(knownspaceObj, hu.Pose(0, 0, 0, 0), None)
    m.addFixedRelObj(room(roomDX0, roomDX1, roomDY0, roomDY1, roomDZ),
                          originPose, None)
    table1 = makeTable(tableDX, tableDY, tableDZ, 'table')
    m.addObj(table1, tableLoc, None, variance = smallVariance)
    table2 = makeTable(tableDX, tableDY, tableDZ, 'table2')
    m.addObj(table2, tableLoc2, None, variance = smallVariance)
    m.addFixedRelObj(warehouse1, originPose, 'table')
    m.addFixedRelObj(warehouse2, originPose, 'table2')
    m.addImplicitObj(['warehouse1', 'warehouse2'], [], 'warehouse')

    m.addObj(cupboard2(cbDX, cbDY, cbDZ, 0.02, 'cupboard1A', shelf=1.0),
             hu.Pose(0,0,tableDZ+0.001,0), 'table',
             variance = smallVariance)
    m.addFixedRelObj(inC1A, hu.Pose(0, 0, 0, 0), 'cupboard1A')
    m.addObj(cupboard2(cbDX, cbDY, cbDZ, 0.02, 'cupboard1B', shelf=1.0),
             hu.Pose(0,2*cbDY+0.35,tableDZ+0.001,0), 'table',
             variance = smallVariance)
    m.addFixedRelObj(inC1B, hu.Pose(0, 0, 0, 0), 'cupboard1B')
    m.addObj(cupboard2(cbDX, cbDY, cbDZ, 0.02, 'cupboard2A', shelf=1.0),
             hu.Pose(0,0,tableDZ+0.001,0), 'table2',
             variance = smallVariance)
    m.addFixedRelObj(inC2A, hu.Pose(0, 0, 0, 0), 'cupboard2A')
    m.addObj(cupboard2(cbDX, cbDY, cbDZ, 0.02, 'cupboard2B', shelf = 1.0),
             hu.Pose(0,2*cbDY+0.35,tableDZ+0.001,0), 'table2',
             variance = smallVariance)
    m.addFixedRelObj(inC2B, hu.Pose(0, 0, 0, 0), 'cupboard2B')

    belowTables = om.BoxAligned([(-roomDX0,-roomDY0, 0),
                                 (roomDX1, roomDY1, tableDZ-0.05)])


    # Can take this out soon
    openRoom = om.BoxAligned([(-roomDX0,-roomDY0, 0), (1.75, 0.9, roomDZ+0.1)])
    openWarehouse1 = om.Object([om.BoxAligned(bigwarehouseDims)],
                               name='openWarehouse1', color = 'gray')
    openWarehouse1.solid = False
    #m.addFixedRelObj(openWarehouse1, originPose, 'table')
    openWarehouse2 = om.Object([om.BoxAligned(bigwarehouseDims)],
                               name='openWarehouse2', color = 'gray')
    openWarehouse2.solid = False
    #m.addFixedRelObj(openWarehouse2, originPose, 'table2')
    glob.knownSpaces = [belowTables, openRoom,
              openWarehouse1.applyRefPose(m.worldPoseMean('warehouse1')),
              openWarehouse2.applyRefPose(m.worldPoseMean('warehouse2'))]

    if not withObjects:
        if greenKnown:
            greenCup = smallCup('green')
            greenCup.name = 'myFavoriteCup'
            greenCup.setBaseName('myFavoriteCup')
            m.addUnlocalizedObj(greenCup,
                                  dist.DDist({'inC1A': 0.2,
                                              'inC1B': 0.1,
                                              'inC2A': 0.4,
                                              'inC2B': 0.3}),
                                  'myFavoriteCup')

        return m, workspace
    

    cupSep = 0.01
    dx = (2*cbDX)/3.
    if randomWorld:
        maxNumRows = 3
        maxNumCols = 3
        choices = [(smallCup, 'green'), (bigCup, 'green'), (smallCup, 'red'),
                   (soupCan, 'red'), (sodaBox, 'blue'), (bigCup, 'red')]

        def randomObj(d):
            if random.random() > 0.9:
                return (None, None)
            else:
                objtypeind = np.nonzero(np.random.multinomial(1, d))[0][0]
                return choices[objtypeind]
            
        def randomCupboard():
            # Sample a cupboard distribution
            univSample = randomWorld.SamplePosterior(\
                np.mat([0]*len(choices)))[0]
            d = np.asarray(univSample[:,20]).T[0]
            print 'Generating random cupboard with dist', d
        
            result = []
            rowNum = 0
            row = []
            while rowNum < maxNumRows:
                o = randomObj(d)
                if o and o[0] == bigCup:
                    if len(row) > 0:
                        result.append(row)
                        row = []
                        rowNum += 1
                    result.append([o])
                    rowNum += 1
                else:
                    row.append(o)
                    if len(row) >= maxNumCols:
                        result.append(row)
                        rowNum += 1
                        row = []
            print '    Cupboard contents', result
            if (smallCup, 'green') in result[0]:
                # Try again...we don't want greein in front
                return randomCupboard()
            else:
                return result[:maxNumRows]
        
        stuff = (('2B', randomCupboard()),
                 ('1B', randomCupboard()),
                 ('2A', randomCupboard()),
                 ('1A', randomCupboard()))
    else:
        stuff = (('2B', [[(bigCup, 'red')], [(None, None)],
                         [(bigCup, 'red')]]),
                 ('1B', [[(bigCup, 'red')], 2*[(soupCan, 'red')],
                         2*[(soupCan, 'red')]]),
                 ('2A', [[(bigCup, 'red')], 2*[(sodaBox, 'blue')],
                         2*[(soupCan,'red')]]),
                 ('1A', [[(bigCup, 'green')],
                         # 2*[(smallCup, 'red')],
                         [(bigCup, 'red')],
                         [(smallCup, 'green'), (smallCup, 'red')]]))
    
    for c, rows in stuff:
        for i in range(len(rows)):
            col = rows[i]
            ncol = len(col)
            if ncol == 1:
                dy = (2.*cbDY)/(ncol+1)
            else:
                # only randomize y in rows of two 
                dy = rny() + (2.*cbDY+0.02)/(ncol+1)
            for j, (obj, color) in enumerate(col):
                if obj:
                    m.addObj(obj(color),
                             hu.Pose(-cbDX+0.05+i*dx+rnx(),
                                       -cbDY+(j+1)*dy,
                                       cupSep, math.pi+rnt()),
                             'cupboard%s'%c, variance = 0.001)
    return m, workspace

def makeMultiCupboardMap3(withObjects = True,
                          deltaX=0.0, deltaY=0.0, deltaTheta=0.0,
                          randomWorld = False,
                          greenKnown = False):
    def rn(x): return 2*x*random.random() - x
    def rnx(): return rn(deltaX)
    def rny(): return rn(deltaY)
    def rnt(): return rn(deltaTheta)

    glob.referenceObject = 'table2'
    smallVariance = 0.01**2
    roomDY0 = roomDY1 = 1.75
    roomDX0, roomDX1 = 2.5, 2.5
    roomDZ = 1.25

    tableDX, tableDY, tableDZ = 0.3, 0.75, 0.75 # 0.68
    tableLoc = hu.Pose(2.1, 0, 0, 0.0)
    tableLoc2 = hu.Pose(0.1, 1.4, 0, 1.57)
    tableLoc3 = hu.Pose(-2.1, 0, 0, 0.0)
    tableLoc4 = hu.Pose(0.1, -1.4, 0, 1.57)

    cbDX, cbDY, cbDZ = 0.2, 0.24, 0.4
    
    workspace = om.BoxAligned([(-roomDX0,-roomDY0, 0),
                               (roomDX1, roomDY1, roomDZ)],
                              name = 'workspace')
    workspace.solid = False
    workspaceObj = om.Object([workspace], name = 'workspace')
    workspaceObj.solid = False

    dx = 2*tableDX+0.1 # was 0.25 -- lpk
    knownspace = om.BoxAligned([(-roomDX0+dx,-roomDY0+dx, 0),
                                (roomDX1-dx, roomDY1-dx, roomDZ)],
                              name = 'knownspace')
    knownspace.solid = False
    knownspaceObj = om.Object([knownspace], name = 'knownspace')
    knownspaceObj.solid = False


    warehouseDims = [(-tableDX+0.05, -tableDY+0.05, tableDZ+0.01),
                     ( tableDX-0.05, 0.0, tableDZ+0.36)]

    bigwarehouseDims = [(-tableDX-0.05, -tableDY, tableDZ+0.01),
                        ( tableDX+0.05, -0.2, tableDZ+0.8)]

    tableAbove1 = om.Object([om.BoxAligned(warehouseDims)],
                            name='warehouse1', color = 'gray')
    warehouse1 = om.ParkingRegion(tableAbove1,
                                 hu.Point(-1,0,0.01,0), hu.Point(0,-1,0,0),
                                 name='warehouse1')
    tableAbove2 = om.Object([om.BoxAligned(warehouseDims)],
                            name='warehouse2', color = 'green')
    warehouse2 = om.ParkingRegion(tableAbove2,
                                 hu.Point(-1,0,0.01,0), hu.Point(0,-1,0,0),
                                 name='warehouse2')

    tableAbove3 = om.Object([om.BoxAligned(warehouseDims)],
                            name='warehouse3', color = 'gray')
    warehouse3 = om.ParkingRegion(tableAbove3,
                                 hu.Point(-1,0,0.01,0), hu.Point(0,-1,0,0),
                                 name='warehouse3')
    tableAbove4 = om.Object([om.BoxAligned(warehouseDims)],
                            name='warehouse4', color = 'green')
    warehouse4 = om.ParkingRegion(tableAbove4,
                                 hu.Point(-1,0,0.01,0), hu.Point(0,-1,0,0),
                                 name='warehouse4')

    cDims = [(-cbDX+0.01, -cbDY+0.01, 0.0), (cbDX-0.01, cbDY-0.01, 0.33)]
    inC1A = om.Region(om.BoxAligned(cDims, opacity=0.5, color='purple'),
                      name='inE')
    inC2A = om.Region(om.BoxAligned(cDims, opacity=0.5, color='purple'),
                      name='inN')
    inC1B = om.Region(om.BoxAligned(cDims, opacity=0.5, color='purple'),
                      name='inW')
    inC2B = om.Region(om.BoxAligned(cDims, opacity=0.5, color='purple'),
                      name='inS')
    m = bm.ObjMap()
    m.addFixedRelObj(workspaceObj, hu.Pose(0, 0, 0, 0), None)
    m.addFixedRelObj(knownspaceObj, hu.Pose(0, 0, 0, 0), None)
    m.addFixedRelObj(room(roomDX0, roomDX1, roomDY0, roomDY1, roomDZ),
                          originPose, None)

    table1 = makeTable(tableDX, tableDY, tableDZ, 'table')
    m.addObj(table1, tableLoc, None, variance = smallVariance)
    table2 = makeTable(tableDX, tableDY, tableDZ, 'table2')
    m.addObj(table2, tableLoc2, None, variance = smallVariance)
    m.addFixedRelObj(warehouse1, originPose, 'table')
    m.addFixedRelObj(warehouse2, originPose, 'table2')
    
    table3 = makeTable(tableDX, tableDY, tableDZ, 'table3')
    m.addObj(table3, tableLoc3, None, variance = smallVariance)
    table4 = makeTable(tableDX, tableDY, tableDZ, 'table4')
    m.addObj(table4, tableLoc4, None, variance = smallVariance)
    m.addFixedRelObj(warehouse3, originPose, 'table3')
    m.addFixedRelObj(warehouse4, originPose, 'table4')

    m.addImplicitObj(['warehouse1', 'warehouse2',
                      'warehouse3', 'warehouse4'], [], 'warehouse')

    m.addObj(cupboard2(cbDX, cbDY, cbDZ, 0.02, 'cupboardE', shelf=1.0),
             hu.Pose(0,2*cbDY,tableDZ+0.001,0), 'table',
             variance = smallVariance)
    m.addFixedRelObj(inC1A, hu.Pose(0, 0, 0, 0), 'cupboardE')
    m.addObj(cupboard2(cbDX, cbDY, cbDZ, 0.02, 'cupboardW', shelf=1.0),
             hu.Pose(0,2*cbDY,tableDZ+0.001,math.pi), 'table3',
             variance = smallVariance)
    m.addFixedRelObj(inC1B, hu.Pose(0, 0, 0, 0), 'cupboardW')
    m.addObj(cupboard2(cbDX, cbDY, cbDZ, 0.02, 'cupboardN', shelf=1.0),
             hu.Pose(0,2*cbDY,tableDZ+0.001,0), 'table2',
             variance = smallVariance)
    m.addFixedRelObj(inC2A, hu.Pose(0, 0, 0, 0), 'cupboardN')
    m.addObj(cupboard2(cbDX, cbDY, cbDZ, 0.02, 'cupboardS', shelf = 1.0),
             hu.Pose(0,2*cbDY,tableDZ+0.001,math.pi), 'table4',
             variance = smallVariance)
    m.addFixedRelObj(inC2B, hu.Pose(0, 0, 0, 0), 'cupboardS')

    belowTables = om.BoxAligned([(-roomDX0,-roomDY0, 0),
                                 (roomDX1, roomDY1, tableDZ-0.05)])


    # Can take this out soon
    openRoom = om.BoxAligned([(-roomDX0,-roomDY0, 0), (1.75, 0.9, roomDZ+0.1)])
    openWarehouse1 = om.Object([om.BoxAligned(bigwarehouseDims)],
                               name='openWarehouse1', color = 'gray')
    openWarehouse1.solid = False
    #m.addFixedRelObj(openWarehouse1, originPose, 'table')
    openWarehouse2 = om.Object([om.BoxAligned(bigwarehouseDims)],
                               name='openWarehouse2', color = 'gray')
    openWarehouse2.solid = False
    #m.addFixedRelObj(openWarehouse2, originPose, 'table2')
    openWarehouse3 = om.Object([om.BoxAligned(bigwarehouseDims)],
                               name='openWarehouse3', color = 'gray')
    openWarehouse3.solid = False
    #m.addFixedRelObj(openWarehouse3, originPose, 'table3')
    openWarehouse4 = om.Object([om.BoxAligned(bigwarehouseDims)],
                               name='openWarehouse4', color = 'gray')
    openWarehouse4.solid = False
    #m.addFixedRelObj(openWarehouse4, originPose, 'table4')
    glob.knownSpaces = [belowTables, openRoom,
              openWarehouse1.applyRefPose(m.worldPoseMean('warehouse1')),
              openWarehouse2.applyRefPose(m.worldPoseMean('warehouse2')),
              openWarehouse3.applyRefPose(m.worldPoseMean('warehouse3')),
              openWarehouse4.applyRefPose(m.worldPoseMean('warehouse4'))]

    if not withObjects:
        if greenKnown:
            greenCup = smallCup('green')
            greenCup.name = 'myFavoriteCup'
            greenCup.setBaseName('myFavoriteCup')
            m.addUnlocalizedObj(greenCup,
                                  dist.DDist({'inC1A': 0.2,
                                              'inC1B': 0.1,
                                              'inC2A': 0.4,
                                              'inC2B': 0.3}),
                                  'myFavoriteCup')

        return m, workspace
    

    cupSep = 0.01

    # if randomWorld:
    #     maxNumRows = 3
    #     maxNumCols = 3
    #     choices = [(smallCup, 'green'), (bigCup, 'green'), (smallCup, 'red'),
    #                (soupCan, 'red'), (sodaBox, 'blue'), (bigCup, 'red')]

    #     def randomObj(d):
    #         if random.random() > 0.9:
    #             return (None, None)
    #         else:
    #             objtypeind = np.nonzero(np.random.multinomial(1, d))[0][0]
    #             return choices[objtypeind]
            
    #     def randomCupboard():
    #         # Sample a cupboard distribution
    #         univSample = randomWorld.SamplePosterior(\
    #             np.mat([0]*len(choices)))[0]
    #         d = np.asarray(univSample[:,20]).T[0]
    #         print 'Generating random cupboard with dist', d
        
    #         result = []
    #         rowNum = 0
    #         row = []
    #         while rowNum < maxNumRows:
    #             o = randomObj(d)
    #             if o and o[0] == bigCup:
    #                 if len(row) > 0:
    #                     result.append(row)
    #                     row = []
    #                     rowNum += 1
    #                 result.append([o])
    #                 rowNum += 1
    #             else:
    #                 row.append(o)
    #                 if len(row) >= maxNumCols:
    #                     result.append(row)
    #                     rowNum += 1
    #                     row = []
    #         print '    Cupboard contents', result
    #         if (smallCup, 'green') in result[0]:
    #             # Try again...we don't want greein in front
    #             return randomCupboard()
    #         else:
    #             return result[:maxNumRows]
        
    #     stuff = (('2B', randomCupboard()),
    #              ('1B', randomCupboard()),
    #              ('2A', randomCupboard()),
    #              ('1A', randomCupboard()))
    # else:
    #     # LSW: Change contents here
    #     stuff = (('2B', [[(bigCup, 'red')], [(None, None)],
    #                      [(bigCup, 'red')]]),
    #              ('1B', [[(bigCup, 'red')], 2*[(soupCan, 'red')],
    #                      2*[(soupCan, 'red')]]),
    #              ('2A', [[(bigCup, 'red')], 2*[(sodaBox, 'blue')],
    #                      2*[(soupCan,'red')]]),
    #              ('1A', [[(bigCup, 'green')],
    #                      # 2*[(smallCup, 'red')],
    #                      [(bigCup, 'red')],
    #                      [(smallCup, 'green'), (smallCup, 'red')]]))

    def placeRowsInCupboard(c, rows):
        dx = (2.*cbDX) / len(rows)
        for i in range(len(rows)):
            col = rows[i]
            ncol = len(col)
            if ncol == 1:
                dy = (2.*cbDY)/(ncol+1)
            else:
                # only randomize y in rows of two 
                dy = rny() + (2.*cbDY+0.02)/(ncol+1)
            for j, (obj, color) in enumerate(col):
                if obj:
                    m.addObj(obj(color),
                             hu.Pose(-cbDX+0.05+i*dx+rnx(),
                                       -cbDY+(j+1)*dy,
                                       cupSep, math.pi+rnt()),
                             'cupboard%s'%c, variance = 0.001)        

    stuff = (('E', [[(None, None)], [(bigCup, 'red')], 2*[(soupCan, 'red')]]),
             ('W', [[(sodaBox, 'blue'), (soupCan, 'red')], [(bigCup, 'red')],
                     [(soupCan, 'red'), (sodaBox, 'blue'), (soupCan, 'red')]]),
             ('S', [[(None, None)], [(bigCup, 'green')], [(bigCup, 'green')]]),
             # ('S', [2*[(smallCup, 'brown')], [(bigCup, 'green')]]),
             ('N', [[(bigCup, 'green')], [(bigCup, 'red')],
                     [(smallCup, 'brown'), (smallCup, 'green')]]))
    

    # LSW: Change placement here
    for c, rows in stuff:
        placeRowsInCupboard(c, rows)

    return m, workspace

def smallCup(color):
    c = makeCup(hu.gensym('smallCup'), color, dims = cupDims)
    c.typeName = 'smallCup_'+color
    return c
def bigCup(color):
    c = makeCup(hu.gensym('bigCup'), color, dims = bigCupDims)
    c.typeName = 'bigCup_'+color
    return c
def sodaBox(color):
    s = om.Object([om.BoxAligned(sodaBoxDims, color=color)],
                     name=hu.gensym('soda'))
    s.typeName = 'soda'
    return s
def soupCan(color, name = None):
    s = om.Object([om.Ngon(0.0675/2, (0, 2*soupCanHalfZ), 8, color=color)],
                     name=name+'_soup' if name else hu.gensym('soup'))
    s.typeName = 'soup16'
    return s
    

###########
# Caelan test
###########

# old
def makeSmallMap2(): 
    #cupPose1 = hu.Pose(-2.5, 0, tableDZ+0.01, math.pi) #Set the cup behind the robot
    #tablePose = hu.Pose(-0.2 - 100, 0.4 + 100 , tableDZ+100, 0) #Set the cup behind the robot
    cupMap = bm.ObjMap()
    cupMap.addFixedRelObj(room(roomDX0, roomDX1, roomDY0, roomDY1, roomDZ),
                          originPose, None)
    table1 = makeTable(0.05, 1.5, tableDZ+.8, 'table')
    cupMap.addObj(table1, tableLocFar, None, variance = landmarkVariance)
    cupMap.addFixedRelObj(easyARegion, easyARegionPose, 'table')
    # cupMap.addObj(makeCup('cupA', 'red'), cupPose1, 'table')
    return cupMap
#########

######################################################################
#            With object types
######################################################################

def makeSSSMap(tableVar = landmarkVariance, includeSS = True):
    tableLocClose = hu.Pose(1.25, tableYOffset, 0, 0.1)
    tableLocFar = hu.Pose(2.0, tableYOffset, 0, 0.1)
    map = bm.ObjMap()
    map.addFixedRelObj(room(roomDX0, roomDX1, roomDY0, roomDY1, roomDZ),
                          originPose, None)
    table1 = makeTable(tableDX, tableDY, tableDZ, 'table1',
                       typeName = 'table1')
    easyARegion = om.Region(om.BoxAligned(om.bboxGrow(sodaBoxDims, 0.1, 0.1),
                                          opacity=0.5, color='purple'),
                            name='easyARegion')
    easyARegionPose = hu.Pose(-0.2, 0.4, tableDZ+0.11, 0)
    if includeSS: 
        map.addObj(table1, tableLocFar, None, variance = tableVar)
        map.addObj(soupCan('red', 'tomato'), sodaBoxPoseStraight, 'table1')
        map.addObj(makeSodaBox('soda1'), soupCanOverPose, 'table1')
        map.addObj(makeSodaBox('soda2'), soupCanPose, 'table1')
    else:
        # These objects are guaranteed to be singular instances of
        # their type
        map.addUnlocalizedObj(table1, dist.DDist({'workspace': 1.0}), 'table1')
        map.addUnlocalizedObj(soupCan('red', 'tomato'),
                              dist.DDist({'aboveTable' : 0.9,
                                          'workspace': 0.1}),
                              'tomato_soup')
    map.addFixedRelObj(easyARegion, easyARegionPose, 'table1') 
    map.addFixedRelObj(warehouse, originPose, 'table1')

    aboveTable = om.Object([\
        om.BoxAligned([(-tableDX, -tableDY, 0),
                       (tableDX, tableDY, 0.5)],
                      name='aboveTable', color='cyan',
                      perm=False, solid = False)], name = 'aboveTable')
    aboveTable.solid = False
    map.addFixedRelObj(aboveTable, hu.Pose(0, 0, tableDZ, 0), 'table1')
    workspaceObj = om.Object([workspace], name = 'workspace')
    workspaceObj.solid = False
    map.addFixedRelObj(workspaceObj, hu.Pose(0, 0, 0, 0), None)
    return map


def makeSSSMap2(tableVar = landmarkVariance, includeSS = True):
    tableLocClose = hu.Pose(1.25, tableYOffset, 0, 0.1)
    tableLocFar = hu.Pose(2.0, tableYOffset, 0, 0.1)
    map = bm.ObjMap()
    map.addFixedRelObj(room(roomDX0, roomDX1, roomDY0, roomDY1, roomDZ),
                          originPose, None)
    table1 = makeTable(tableDX, tableDY, tableDZ, 'table1',
                       typeName = 'table1')
    easyARegion = om.Region(om.BoxAligned(om.bboxGrow(sodaBoxDims, 0.1, 0.1),
                                          opacity=0.5, color='purple'),
                            name='easyARegion')
    easyARegionPose = hu.Pose(-0.2, 0.4, tableDZ+0.11, 0)
    if includeSS:
        p1 = hu.Pose(-0.2, 0.1, tableDZ+0.01, math.pi/16)
        p2 = hu.Pose(-0.2, 0.3, tableDZ+0.01, math.pi/16)
        p3 = hu.Pose(-0.2, 0.5, tableDZ+0.01, math.pi/16)
        p4 = hu.Pose(-0.2, -0.1, tableDZ+0.01, math.pi/16)
        p5 = hu.Pose(-0.2, -0.3, tableDZ+0.01, math.pi/16)
        p6 = hu.Pose(0.2, -0.3, tableDZ+0.01, math.pi/16)
        p7 = hu.Pose(0.2, -0.1, tableDZ+0.01, math.pi/16)
        p8 = hu.Pose(0.2, 0.1, tableDZ+0.01, math.pi/16)
        p9 = hu.Pose(0.2, 0.3, tableDZ+0.01, math.pi/16)

        map.addObj(table1, tableLocFar, None, variance = tableVar)
        map.addObj(soupCan('red', 'tomato'), sodaBoxPoseStraight, 'table1')
        map.addObj(makeSodaBox('soda1'), p1, 'table1')
        #map.addObj(makeSodaBox('soda2'), p2, 'table1')
        #map.addObj(makeSodaBox('soda3'), p3, 'table1')
        map.addObj(makeSodaBox('soda4'), p4, 'table1')
        map.addObj(makeSodaBox('soda5'), p5, 'table1')
        # map.addObj(makeSodaBox('soda6'), p6, 'table1')
        map.addObj(makeSodaBox('soda7'), p7, 'table1')
        map.addObj(makeSodaBox('soda8'), p8, 'table1')
        # map.addObj(makeSodaBox('soda9'), p9, 'table1')

    else:
        # These objects are guaranteed to be singular instances of
        # their type
        map.addUnlocalizedObj(table1, dist.DDist({'workspace': 1.0}), 'table1')
        map.addUnlocalizedObj(soupCan('red', 'tomato'),
                              dist.DDist({'aboveTable' : 0.9,
                                          'workspace': 0.1}),
                              'tomato_soup')
    map.addFixedRelObj(easyARegion, easyARegionPose, 'table1') 
    map.addFixedRelObj(warehouse, originPose, 'table1')

    aboveTable = om.Object([\
        om.BoxAligned([(-tableDX, -tableDY, tableDZ),
                       (tableDX, tableDY, tableDZ + 0.5)],
                      name='aboveTable', color='cyan',
                      perm=False, solid = False)], name = 'aboveTable')
    aboveTable.solid = False
    map.addFixedRelObj(aboveTable, hu.Pose(0, 0, 0, 0), 'table1')
    workspaceObj = om.Object([workspace], name = 'workspace')
    workspaceObj.solid = False
    map.addFixedRelObj(workspaceObj, hu.Pose(0, 0, 0, 0), None)
    return map

######################################################################
#            Export
######################################################################

def exportWorld(objects, filename):
    exportObjects(objects.values(), filename)

##############

def foo():
    m, workspace = makeSSMap()
    glob.e = worldForTomas(m, workspace,
                           startingKinConfR, startingKinConfL)
'''
