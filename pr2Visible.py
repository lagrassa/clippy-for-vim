import pdb
import math
import numpy as np
from pointClouds import Scan, updateDepthMap, Raster
import hu
import planGlobals as glob
from geom import bboxCenter
from shapes import pointBox, BoxScale
import transformations as transf
from pr2Util import shadowName, objectName
from traceFile import debugMsg, debug
import windowManager3D as wm
from miscUtil import argmax

Ident = hu.Transform(np.eye(4))         # identity transform
laserScanGlobal = None
minVisiblePoints = 10
minVisibleRasterAreaFrac = 0.05**2       # 5% on each image axis

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

def visibleOLD(ws, conf, shape, obstacles, prob, moveHead=True, fixed=[]):
    global laserScanGlobal
    key = (ws, conf, shape, tuple(obstacles), prob, moveHead, tuple(fixed))
    cacheStats[0 if glob.inHeuristic else 3] += 1
    if key in cache:
        cacheStats[1 if glob.inHeuristic else 4] += 1
        return cache[key]
    headChainName = conf.robot.headChainName
    if debug('visible'):
        print 'visible', shape.name(), 'from base=', conf.baseConf(), 'head=', conf[headChainName]
        print 'obstacles', obstacles
        print 'fixed', fixed
    lookConf = lookAtConf(conf, shape) if moveHead else conf
    if not lookConf:
        if debug('visible'):
            print 'lookConf failed'
        cache[key] = (False, [])
        return False, []
    scan = lookScan(lookConf)
    # centerPoint = hu.Point(np.resize(np.hstack([bboxCenter(shape.bbox()), [1]]), (4,1)))
    # if not scan.visible(centerPoint): # is origin in FOV
    #     if debug('visible'):
    #         print 'shape not in FOV', shape
    #         print 'center', centerPoint
    #         viewCone(conf, shape).draw('W', 'red')
    #         raw_input('FOV')
    #     cache[key] = (False, [])
    #     return False, []
    vc = viewCone(conf, shape)
    if not vc:
        if debug('visible'):
            print 'viewCone failed'
        cache[key] = (False, [])
        return False, []

    if debug('visible'):
        vc.draw('W', 'red')
        shape.draw('W', 'cyan')
        lookConf.draw('W')
        debugMsg('visible', 'look conf and view cone')

    potentialOccluders = []
    fix = [obj for obj in obstacles if obj.name() in ws.fixedObjects]
    for f in fixed: fix.append(f)
    move = [obj for obj in obstacles if obj not in fix]
    for objShape in fix+move:
        if objShape.collides(vc):
            potentialOccluders.append(objShape)
    if debug('visible'):
        print 'potentialOccluders', potentialOccluders

    # If we can't move the head, then the object might not be visible
    # because of FOV issues (not enough points on the object).
    if moveHead and not potentialOccluders:
        cacheStats[2 if glob.inHeuristic else 5] += 1
        return True, []

    occluders = []

    n = scan.edges.shape[0]
    dm = np.zeros(n); dm.fill(10.0)
    contacts = n*[None]
    for objPrim in shape.toPrims():
        updateDepthMap(scan, objPrim, dm, contacts, 0)
    total = n - contacts.count(None)
    if total < minVisiblePoints:
        if debug('visible'):
            scan.draw('W')
            print total, 'hit points for', shape
            debugMsg('visible', 'Not enough hit points')
        cache[key] = (False, [])
        return False, []

    # if 'table' in shape.name():
    #     threshold = 0.5*prob            # generous
    # else:
    #     # threshold = 0.75*prob
    #     threshold = 0.5

    threshold = 0.5                     # for consistency across simulation and planning

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
    if ratio < threshold:
        if debug('visible'): print 'visible ->', (False, [])
        return False, []            # No hope
    # find a list of movable occluders that could be removed to
    # achieve visibility
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
    # remove enough occluders to make it visible, be greedy
    occluders.sort(reverse=True)        # biggest occluder first
    ans = None
    for i in xrange(len(occluders)):
        if float(final - sum([x[0] for x in occluders[i:]]))/float(total) >= threshold:
            ans = True, [x[1] for x in occluders[:i]]
            break
    if ans is None:
        ans = True, [x[1] for x in occluders]
    if debug('visible'):
        print 'sorted occluders', occluders
        print 'total', total, 'final', final, '(', ratio, ')', 'thr', threshold, '->', ans
    cache[key] = ans
    if debug('visible'):
        if ans[1] and any([vc.collides(obj) for obj in fix]):
            print 'visible ->', ans
            debugMsg('visible', 'Visibility is compromised')
    return ans

def countContacts(contacts, id):
    final = 0
    for c in contacts:
        if c is None: continue
        if c[1] == id: final += 1
    return final

def lookAtConf(conf, shape):
    center = bboxCenter(shape.bbox())   # base=True?
    z = shape.bbox()[1,2]               # at the top
    z = 0.5*(shape.bbox()[0,2] + shape.bbox()[1,2])       # at the middle
    z = shape.bbox()[0,2]               # at the bottom
    headChainName = conf.robot.headChainName
    for dz in (0, 0.02, 0.04, 0.06):
        center[2] = z + dz
        cartConf = conf.cartConf()
        assert cartConf[headChainName]
        lookCartConf = cartConf.set(headChainName, hu.Pose(*center.tolist()+[0.,]))
        lookConf = conf.robot.inverseKin(lookCartConf, conf=conf)
        if all(lookConf.values()):
            return lookConf
    if debug('visible'):
        print 'Failed head kinematics trying to look at', shape.name(), 'from', center.tolist()

def viewCone(conf, shape, offset = 0.1):
    lookConf = lookAtConf(conf, shape)
    if not lookConf:
        return
    lookCartConf = lookConf.cartConf()
    headChainName = conf.robot.headChainName
    headTrans = lookCartConf[headChainName]
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
        print shape.name(), 'sensor bbox\n', sensorShape.bbox()
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
    if targetObj in pbs.getPlacedObjBs():
        return findSupportTable(targetObj, pbs.getWorld(), pbs.getPlacedObjBs())
    else:
        return None

def lookScan(lookConf):
    global laserScanGlobal
    lookCartConf = lookConf.cartConf()
    headChainName = conf.robot.headChainName
    headTrans = lookCartConf[headChainName]

    if not laserScanGlobal:
        laserScanGlobal = Scan(Ident, glob.laserScanParams)
    laserScan = laserScanGlobal

    scanTrans = headTrans.compose(hu.Transform(transf.rotation_matrix(-math.pi/2, (0,1,0))))
    scan = laserScan.applyTrans(scanTrans)
    return scan

#######################################

rasterGlobal = None

def lookRaster():
    global rasterGlobal
    if rasterGlobal is None:
        (focal, height, width, length, n) = glob.laserScanParams
        rasterGlobal = Raster(width, height, 2*n, 2*n, 0.5, length, focal)
        viewPort = [0, rasterGlobal.imageWidth, 0, rasterGlobal.imageHeight, 0.0, 0.1]
        wm.makeWindow('Raster', viewPort, 2*n+10)
    return rasterGlobal

def render(raster, lookConf, shape):
    raster.reset()
    lookCartConf = lookConf.cartConf()
    headTrans = lookCartConf[lookConf.robot.headChainName]
    sensor = headTrans.compose(hu.Transform(transf.rotation_matrix(-math.pi, (1,0,0))))
    trans = sensor.inverse()
    shape1 = shape.applyTrans(trans)
    for objPrim in shape1.toPrims():
        raster.update(objPrim, 1)
    return trans

nCalls = 0
nWins = 0
notEnoughPoints = 0
belowThresh = 0

# prob == 0. when doing simulation
def visible(ws, conf, shape, obstacles, prob, moveHead=True, fixed=[]):
    global nCalls, nWins, notEnoughPoints, belowThresh
    nCalls +=1 
    if debug('visible'):
        print 'visible', shape.name(), 'from base=', conf.baseConf(), 'head=', conf[conf.robot.headChainName]
        print 'obstacles', obstacles
        print 'fixed', fixed
    key = (ws, conf, shape, tuple(obstacles), prob==0., moveHead, tuple(fixed))
    cacheStats[0 if glob.inHeuristic else 3] += 1
    if key in cache:
        cacheStats[1 if glob.inHeuristic else 4] += 1
        if debug('visible'):
            print 'cached visibility', cache[key]
        return cache[key]
    raster = lookRaster()
    rasterArea = raster.imageWidth * raster.imageHeight

    lookConf = lookAtConf(conf, shape)  # assume we can move head
    if not lookConf:
        if debug('visible'):
            print 'lookConf failed'
        cache[key] = (False, [])
        return False, []
    trans = render(raster, lookConf, shape)
    total = raster.countId(1)           #  pixels on target if we can look directly
    minVis = minVisibleRasterAreaFrac * rasterArea
    if prob:                            # when predicting
        minVis *= 1.5                   # be more conservative
    if total < minVis:
        if debug('visible'):
            print total, 'hit points for', shape
            debugMsg('visible', 'Not enough hit points')
        cache[key] = (False, [])
        notEnoughPoints += 1
        return False, []
    if not moveHead:
        lookConf = conf
        trans = render(raster, lookConf, shape) # re-render
    fix = [obj for obj in obstacles if obj.name() in ws.fixedObjects]
    for f in fixed:
        if f not in fix: fix.append(f)
    move = [obj for obj in obstacles if obj not in fix]
    occluders = []
    threshold = 0.6 if prob else 0.4
    for i, objShape in enumerate(fix):
        if objectName(shape) == objectName(objShape): continue
        if debug('visible'):
            print 'updating depth with', objShape.name()
        sh = objShape.applyTrans(trans)
        for objPrim in sh.toPrims():
            raster.update(objPrim, i+2, onlyUpdate=set(range(1,i+2)))
        count = raster.countId(i+2)
        if count > 0:                   #  should these be included?
            occluders.append((count, objShape.name()))
    if debug('visible'):
        print 'fixed occluders', occluders
    # acceptance is based on occlusion by fixed obstacles
    final = raster.countId(1)
    ratio = float(final)/float(total)
    if ratio < threshold:
        if debug('visible'): print 'visible ->', (False, [])
        belowThresh += 1
        return False, []            # No hope
    # find a list of movable occluders that could be removed to
    # achieve visibility
    for j, objShape in enumerate(move):
        if objectName(shape) == objectName(objShape): continue
        i = len(fix) + j
        if debug('visible'):
            print 'updating depth with', objShape.name()
        sh = objShape.applyTrans(trans)
        for objPrim in sh.toPrims():
            raster.update(objPrim, i+2,
                          onlyUpdate=set(range(1,i+2)))
        count = raster.countId(i+2)
        if count > 0:
            occluders.append((count, objShape.name()))
    # remove enough occluders to make it visible, be greedy
    occluders.sort(reverse=True)        # biggest occluder first
    ans = None
    for i in xrange(len(occluders)):
        if float(final - sum([x[0] for x in occluders[i:]]))/float(total) >= threshold:
            ans = True, [x[1] for x in occluders[:i]]
            break
    if ans is None:
        ans = True, [x[1] for x in occluders]
    if debug('visible'):
        print 'sorted occluders', occluders
        print 'total', total, 'final', final, '(', ratio, ')', 'thr', threshold, '->', ans
        wm.getWindow('W').clear()
        ws.draw('W')
        lookConf.draw('W', attached=ws.attached)
        wm.getWindow('W').update()
        raster.draw('Raster')
        debugMsg('visible', 'Admire')

    # For entertainment...
    if debug('visible_raster'): raster.draw('Raster')

    cache[key] = ans
    nWins += 1
    return ans

