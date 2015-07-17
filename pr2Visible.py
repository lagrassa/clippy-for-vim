import math
import numpy as np
from pointClouds import Scan, updateDepthMap
import hu
import planGlobals as glob
from geom import bboxCenter
from shapes import pointBox, BoxScale
import transformations as transf
from pr2Util import shadowName
from traceFile import debugMsg, debug
import windowManager3D as wm
from miscUtil import argmax

Ident = hu.Transform(np.eye(4))            # identity transform
laserScanGlobal = None
laserScanSparseGlobal = None
laserScanParams = (0.3, 0.1, 0.1, 2., 20) # narrow
laserScanParamsSparse = (0.3, 0.1, 0.1, 2., 15) # narrow
minVisiblePoints = 5

colors = ['red', 'green', 'blue', 'orange', 'cyan', 'purple']

# This is a cache for visibility computations, the key is formed from
# the args to visible.
cache = {}
cacheStats = [0, 0, 0, 0, 0, 0]                   # h tries, h hits, h easy, real tries, real hits, easy

# ws - shadow world
# conf - robot conf, head direction can be moved (if moveHead is True)
# shape - target shape
# obstacles - a list of "movable" obstacles
# prob - used (loosely) to determine how much partial occlusion is allowed
# moveHead - if True, allows the head orientation to change
# fixed - list of (additional) fixed objects, besides those marked as fixed in ws
# Returns (bool, list of occluders).  The bool indicates whether the
# target shape is potentially visible if the occluding obsts are removed.

def visible(ws, conf, shape, obstacles, prob, moveHead=True, fixed=[]):
    global laserScanGlobal, laserScanSparseGlobal
    key = (ws, conf, shape, tuple(obstacles), prob, moveHead, tuple(fixed), glob.inHeuristic)
    cacheStats[0 if glob.inHeuristic else 3] += 1
    if key in cache:
        cacheStats[1 if glob.inHeuristic else 4] += 1
        return cache[key]
    if debug('visible'):
        print 'visible', shape.name(), 'from base=', conf['pr2Base'], 'head=', conf['pr2Head']
    lookConf = lookAtConf(conf, shape) if moveHead else conf
    if not lookConf:
        if debug('visible'):
            print 'lookConf failed'
        cache[key] = (False, [])
        return False, []
    vc = viewCone(conf, shape, moveHead=moveHead)
    if debug('visible'):
        vc.draw('W', 'red')
        shape.draw('W', 'cyan')
        lookConf.draw('W')
        debugMsg('visible', 'look conf and view cone')

    potentialOccluders = []
    fixed = list(ws.fixedObjects)+fixed
    fix = [obj for obj in obstacles if obj.name() in fixed]
    move = [obj for obj in obstacles if obj.name() not in fixed]
    for objShape in fix+move:
        if objShape.name() == 'PR2': continue # already handled
        if objShape.collides(vc):
            potentialOccluders.append(objShape)
    if debug('visible'):
        print 'potentialOccluders', potentialOccluders
    if not potentialOccluders:
        cacheStats[2 if glob.inHeuristic else 5] += 1
        return True, []
    occluders = []

    scan = lookScan(lookConf)
    n = scan.edges.shape[0]
    dm = np.zeros(n); dm.fill(10.0)
    contacts = n*[None]
    for objPrim in shape.toPrims():
        updateDepthMap(scan, objPrim, dm, contacts, 0)
    total = n - contacts.count(None)
    if total < minVisiblePoints:
        if debug('visible'):
            scan.draw('W')
            print total, 'hit points'
            debugMsg('visible', 'Not enough hit points')
        cache[key] = (False, [])
        return False, []
    if 'table' in shape.name() or glob.inHeuristic:
        threshold = 0.5*prob            # generous
    else:
        threshold = 0.75*prob

    for i, objShape in enumerate(fix):
        if objShape not in potentialOccluders: continue
        if debug('visible'):
            print 'updating depth with', objShape.name()
        for objPrim in objShape.toPrims():
            updateDepthMap(scan, objPrim, dm, contacts, i+1, onlyUpdate=range(i+2))
        count = countContacts(contacts, i+1)
        if count > 0:                   #  should these be included?
            occluders.append((count, objShape.name()))
    if debug('visible'):
        print 'fixed occluders', occluders
    # acceptance is based on occlusion by fixed obstacles
    final = countContacts(contacts, 0)
    ratio = float(final)/float(total)
    for j, objShape in enumerate(move):
        if objShape not in potentialOccluders: continue
        i = len(fix) + j
        if debug('visible'):
            print 'updating depth with', objShape.name()
        for objPrim in objShape.toPrims():
            updateDepthMap(scan, objPrim, dm, contacts, i+1, onlyUpdate=range(i+2))
        count = countContacts(contacts, i+1)
        if count > 0:
            occluders.append((count, objShape.name()))
    if debug('visible'):
        wm.getWindow('W').clear()
        ws.draw('W')
        lookConf.draw('W', attached=ws.attached)
        for c in contacts:
            if c:
                pointBox(c[0]).draw('W', colors[c[1]%len(colors)])
        wm.getWindow('W').update()
        debugMsg('visible', 'Admire')
    occluders.sort(reverse=True)
    occluders = [x[1] for x in occluders]
    ans = ratio >= threshold, occluders
    if debug('visible'):
        print 'sorted occluders', occluders
        print 'total', total, 'final', final, '(', ratio, ')', 'thr', threshold, '->', ans
    cache[key] = ans
    if debug('visible'):
        if ans[1] and any([vc.collides(obj) for obj in fix]):
            print 'visible ->', ans
            raw_input('Visibility is compromised')
    return ans

def countContacts(contacts, id):
    final = 0
    for c in contacts:
        if c is None: continue
        if c[1] == id: final += 1
    return final

def lookAtConf(conf, shape):
    center = bboxCenter(shape.bbox())   # base=True?
    z = shape.bbox()[1,2]       # at the top
    for dz in (0, 0.02, 0.04, 0.06):
        center[2] = z + dz
        cartConf = conf.cartConf()
        assert cartConf['pr2Head']
        lookCartConf = cartConf.set('pr2Head', hu.Pose(*center.tolist()+[0.,]))
        lookConf = conf.robot.inverseKin(lookCartConf, conf=conf)
        if all(lookConf.values()):
            return lookConf
    print 'Failed to look at', shape.name(), center.tolist()

def viewCone(conf, shape, offset = 0.1, moveHead=True):
    if moveHead:
        lookConf = lookAtConf(conf, shape)
    else:
        lookConf = conf
    if not lookConf:
        return
    lookCartConf = lookConf.cartConf()
    headTrans = lookCartConf['pr2Head']
    sensor = headTrans.compose(hu.Transform(transf.rotation_matrix(-math.pi, (1,0,0))))
    sensorShape = shape.applyTrans(sensor.inverse())
    ((x0,y0,z0),(x1,y1,z1)) = sensorShape.bbox()
    dz = -0.15-z1
    cone = BoxScale((x1-x0), (y1-y0), dz, None, 0.01,name='ViewConeFor%s'%shape.name())
    final = cone.applyTrans(hu.Pose(0.,0.,-(dz+0.15)/2,0.)).applyTrans(sensor)

    if debug('viewCone'):
        wm.getWindow('W').clear()
        shape.draw('W', 'cyan')
        final.draw('W', 'red')
        print shape.name(), 'sensor bbox\n', sensorShape.bbox(), 'moveHead=', moveHead
        raw_input('viewCone')

    return final

def findSupportTable(targetObj, world, placeBs):
    tableBs = [pB for pB in placeBs.values() \
               if ('table' in pB.obj or 'shelves' in pB.obj)]
    # print 'tablesBs', tableBs
    tableCenters = [pB.poseD.mode().point() for pB in tableBs]
    targetB = placeBs[targetObj]
    assert targetB
    targetCenter = targetB.poseD.mode().point()
    bestCenter = argmax(tableCenters, lambda c: -targetCenter.distance(c))
    ind = tableCenters.index(bestCenter)
    return tableBs[ind]

def findSupportTableInPbs(pbs, targetObj):
    return findSupportTable(targetObj, pbs.getWorld(), pbs.getPlacedObjBs())

def lookScan(lookConf):
    global laserScanSparseGlobal, laserScanGlobal
    lookCartConf = lookConf.cartConf()
    headTrans = lookCartConf['pr2Head']
    if glob.inHeuristic:
        if not laserScanSparseGlobal:
            laserScanSparseGlobal = Scan(Ident, laserScanParamsSparse)
        laserScan = laserScanSparseGlobal
    else:
        if not laserScanGlobal:
            laserScanGlobal = Scan(Ident, laserScanParams)
        laserScan = laserScanGlobal
    scanTrans = headTrans.compose(hu.Transform(transf.rotation_matrix(-math.pi/2, (0,1,0))))
    scan = laserScan.applyTrans(scanTrans)
    return scan
