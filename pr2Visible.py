import math
import numpy as np
from pointClouds import Scan, updateDepthMap
import util
from geom import bboxCenter
from shapes import toPrims, pointBox, BoxScale
import transformations as transf
from pr2Util import shadowName
import fbch
from planGlobals import debugMsg, debugDraw, debug, pause
import windowManager3D as wm
from miscUtil import argmax

Ident = util.Transform(np.eye(4))            # identity transform
laserScanGlobal = None
laserScanSparseGlobal = None
laserScanParams = (0.3, 0.1, 0.1, 2., 20) # narrow
laserScanParamsSparse = (0.3, 0.1, 0.1, 2., 15) # narrow
minVisiblePoints = 5

colors = ['red', 'green', 'blue', 'orange', 'cyan', 'purple']

cache = {}
cacheStats = [0, 0, 0, 0]                   # h tries, h hits, real tries, real hits

# !! cache visibility computations
def visible(ws, conf, shape, obstacles, prob, moveHead=True, fixed=[]):
    global laserScanGlobal, laserScanSparseGlobal
    key = (ws, conf, shape, tuple(obstacles), prob, moveHead, tuple(fixed), fbch.inHeuristic)
    cacheStats[0 if fbch.inHeuristic else 2] += 1
    if key in cache:
        cacheStats[1 if fbch.inHeuristic else 3] += 1
        return cache[key]
    if debug(visible):
        print 'visible from base=', conf['pr2Base'], 'head=', conf['pr2Head']
    lookConf = lookAtConf(conf, shape) if moveHead else conf
    if not lookConf:
        cache[key] = (False, [])
        return False, []
    vc = viewCone(conf, shape, moveHead=moveHead)
    if debug('visible'):
        vc.draw('W', 'red')
        shape.draw('W', 'cyan')
        lookConf.draw('W')
        debugMsg('visible', 'look conf and view cone')

    lookCartConf = lookConf.cartConf()
    headTrans = lookCartConf['pr2Head']
    if fbch.inHeuristic:
        if not laserScanSparseGlobal:
            laserScanSparseGlobal = Scan(Ident, laserScanParamsSparse)
        laserScan = laserScanSparseGlobal
    else:
        if not laserScanGlobal:
            laserScanGlobal = Scan(Ident, laserScanParams)
        laserScan = laserScanGlobal
    scanTrans = headTrans.compose(util.Transform(transf.rotation_matrix(-math.pi/2, (0,1,0))))
    scan = laserScan.applyTrans(scanTrans)
    n = scan.edges.shape[0]
    dm = np.zeros(n); dm.fill(10.0)
    contacts = n*[None]
    for objPrim in toPrims(shape):
        updateDepthMap(scan, objPrim, dm, contacts, 0)
    total = n - contacts.count(None)
    if total < minVisiblePoints:
        if debug('visible'):
            scan.draw('W')
            print total, 'hit points'
            debugMsg('visible', 'Not enough hit points')
        cache[key] = (False, [])
        return False, []
    if fbch.inHeuristic:
        threshold = 0.5*prob            # generous
    else:
        threshold = 0.75*prob
    occluders = []
    fixed = list(ws.fixedObjects)+fixed
    fix = [obj for obj in obstacles if obj.name() in fixed]
    move = [obj for obj in obstacles if obj.name() not in fixed]
    for i, objShape in enumerate(fix):
        if debug('visible'):
            print 'updating depth with', objShape.name()
        for objPrim in toPrims(objShape):
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
        i = len(fix) + j
        if debug('visible'):
            print 'updating depth with', objShape.name()
        for objPrim in toPrims(objShape):
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
    center = bboxCenter(shape.bbox(), base=True)
    cartConf = conf.cartConf()
    assert cartConf['pr2Head']
    lookCartConf = cartConf.set('pr2Head', util.Pose(*center.tolist()+[0.,]))
    lookConf = conf.robot.inverseKin(lookCartConf, conf=conf)
    if all(lookConf.values()):
        return lookConf

def viewCone(conf, shape, offset = 0.1, moveHead=True):

    if moveHead:
        lookConf = lookAtConf(conf, shape)
    else:
        lookConf = conf
    if not lookConf:
        return
    lookCartConf = lookConf.cartConf()
    headTrans = lookCartConf['pr2Head']
    sensor = headTrans.compose(util.Transform(transf.rotation_matrix(-math.pi, (1,0,0))))
    # Note xyPrim
    sensorShape = shape.applyTrans(sensor.inverse())
    ((x0,y0,z0),(x1,y1,z1)) = sensorShape.bbox()
    # print 'sensorShape bbox\n', sensorShape.bbox()
    dz = -0.15-z1
    cone = BoxScale((x1-x0), (y1-y0), dz, None, 0.01,name='ViewConeFor%s'%shape.name())
    return cone.applyTrans(util.Pose(0.,0.,-(dz+0.15)/2,0.)).applyTrans(sensor)

def findSupportTable(targetObj, world, placeBs):
    tableBs = [pB for pB in placeBs.values() if 'table' in pB.obj]
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
