import DrawingWindowStandalonePIL as dw
reload(dw)
from graphics3D import Window3D

######################################################################
##  Window management
######################################################################

use3D = False
windows = {}

def addWindow(w, title):
    windows[title] = w

def makeWindow(title, viewPort = [-1.,1., -1.,1., 0.,1.], windowWidth = 500):
    if not title in windows:
        print 'Could not find', title, 'in', windows
        print 'Creating new window:', title
        if use3D:
            # windows[title] = om.VisualWindow(title = title,
            #                      windowDims = (windowWidth, windowWidth))
            pass
        else:
            print viewPort
            windows[title] = Window3D(title = title,
                                      windowWidth = windowWidth,
                                      viewport = viewPort)
    return windows[title]

def getWindow(w):
    if isinstance(w, str):
        return makeWindow(w)
    else:
        return w

def makeSimpleWindow(dx, dy, windowWidth, title):
    if not title in windows:
        windows[title] = dw.DrawingWindow(windowWidth, windowWidth,
                                          0, max(dx, dy), 0, max(dx, dy), 
                                          title)
    return windows[title]


def makeDrawingWindow(windowWidth, windowHeight, xMin, xMax, yMin, yMax,
                      title):
    if not title in windows:
        windows[title] = dw.DrawingWindow(windowWidth, windowHeight,
                                          xMin, xMax, yMin, yMax,
                                          title)
    return windows[title]
