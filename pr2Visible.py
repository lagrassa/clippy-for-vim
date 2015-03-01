import math
import numpy as np
from pointClouds import Scan, updateDepthMap
import util
from shapes import toPrims, pointBox
import transformations as transf
from pr2Util import shadowName
import fbch
from planGlobals import debugMsg, debugDraw, debug, pause
import windowManager3D as wm

Ident = util.Transform(np.eye(4))            # identity transform
laserScanGlobal = None
laserScanSparseGlobal = None
laserScanParams = (0.3, 0.1, 0.1, 2., 20) # narrow
laserScanParamsSparse = (0.3, 0.1, 0.1, 2., 15) # narrow
minVisiblePoints = 5

colors = ['red', 'green', 'blue', 'orange', 'cyan', 'purple']

# !! Cache visibility computations
def visible(ws, conf, shape, obstacles, prob, moveHead=True):
    global laserScanGlobal, laserScanSparseGlobal
    if debug(visible):
        print 'visible from base=', conf['pr2Base'], 'head=', conf['pr2Head']
    lookConf = lookAtConf(conf, shape) if moveHead else conf
    lookCartConf = conf.robot.forwardKin(lookConf)
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
        scan.draw('W')
        print total, 'hit points'
        debugMsg('visible', 'Not enough hit points')
        return False, []
    occluders = []
    for i, objShape in enumerate(obstacles):
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
    final = countContacts(contacts, 0)
    if fbch.inHeuristic:
        ans = float(final)/float(total) >= 0.5*prob, occluders # !! be generous
    else:
        ans = float(final)/float(total) >= 0.8*prob, occluders
    if debug('visible'):
        print 'sorted occluders', occluders
        print 'total', total, 'final', final, '->', ans
    return ans

def countContacts(contacts, id):
    final = 0
    for c in contacts:
        if c is None: continue
        if c[1] == id: final += 1
    return final

def lookAtConf(conf, shape):
    center = shape.center()
    cartConf = conf.robot.forwardKin(conf)
    assert cartConf['pr2Head']
    lookCartConf = cartConf.set('pr2Head', util.Pose(*center.tolist()+[0.,]))
    return conf.robot.inverseKin(lookCartConf, conf=conf)
