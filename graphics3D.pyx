import math
import time
import numpy as np
cimport numpy as np
from cpython cimport bool
cimport shapes
import shapes
import DrawingWindowStandalonePIL as dw
import colorNames

noWindow = False
use3D = False

if use3D:
    import visual
    reload(visual)

#################################
# Windows
#################################

cdef class Window3D:
    def __init__(self, viewport = None, str title = 'View', int windowWidth = 500):

        if noWindow:
            self.window = None
            return

        self.capture = []
        self.capturing = False

        cdef float lox, hix, loy, hiy, loz, hiz
        (lox, hix, loy, hiy, loz, hiz) = viewport
        windowHeight = (hiy + hiz - loy - loz) * (float(windowWidth)/(hix-lox))
        self.xzOffset = (0, hiy)
        self.window = dw.DrawingWindow(windowWidth, windowHeight, lox, hix, loy,
                                       hiy + hiz - loz, title)

    def clear(self):
        if noWindow:
            return
        self.window.clear()
        if self.capturing:
            if not self.capture or not self.capture[-1][0] == 'clear':
                self.capture.append(['clear', time.clock()])

    def pause(self):
        print 'Pausing window'
        if self.capturing:
            if not self.capture or not self.capture[-1][0] == 'pause':
                self.capture.append(['pause', time.clock()])

    def startCapture(self):
        self.capture = []
        self.capturing = True

    def stopCapture(self):
        self.capturing = False
        return self.capture

    def playback(self, capture = None, delay = 0):
        capturing = self.capturing
        self.capturing = False
        for entry in capture or self.capture:
            if entry[0] == 'clear':
                self.clear()
                time.sleep(delay)
            elif entry[0] == 'pause':
                time.sleep(delay)
            else:
                self.draw(entry[0], color=entry[1], opacity=entry[2])
                # self.update()
        self.capturing = capturing

    cpdef draw(self, shapes.Thing thing, str color = None, float opacity = 1.0):
        cdef:
            shapes.Prim prim
            np.ndarray[np.float64_t, ndim=2] verts
            np.ndarray[np.float64_t, ndim=1] p1, p2
            np.ndarray[np.int_t, ndim=2] edges
            float dx, dy
            int i, n, e
        if noWindow:
            return
        if self.capturing:
            self.capture.append([thing,
                                 color,
                                 opacity,
                                 time.clock()])
        (dx, dy) = self.xzOffset
        for part in thing.parts():
            partColor = color or part.properties.get('color', None) or 'black'
            if isinstance(part, shapes.Shape):
                for p in part.parts(): self.draw(p, color=partColor, opacity=opacity)
            elif isinstance(part, (shapes.Thing, shapes.Prim)):
                prim = part.prim()
                verts = prim.vertices()
                edges = prim.edges()
                for e in range(edges.shape[0]):
                    p1 = verts[:,edges[e,0]]
                    p2 = verts[:,edges[e,1]]
                    self.window.drawLineSeg(p1[0], p1[1], p2[0], p2[1],
                                            color = partColor)
                    self.window.drawLineSeg(p1[0]+dx, p1[2]+dy, p2[0]+dx, p2[2]+dy,
                                            color = partColor)

    cpdef update(self):
        self.window.canvas.update()


cdef class VisualWindow:
    colors = colorNames.colors
    def __init__(self, title = "Robot Simulation",
                 windowDims = (600, 600)):
        self.window = visual.display()
        self.disp = {}
        self.window.width = windowDims[0]
        self.window.height = windowDims[1]
        self.window.up = (0, 0, 1)
        self.window.forward = (0, -1, 0)
        self.window.title = title
        self.window.visible = True

    def color(self, c):
        if isinstance(c, str):
            if c[0] == '#':
                return map(float.fromhex, [c[1:3], c[3:5], c[5:7]])
            else:
                return [x/256.0 for x in self.colors[c]]
        elif isinstance(c, (tuple, list)):
            return [x/256.0 for x in c]
        elif c is None:
            return [x/256.0 for x in self.colors['white']]
        else:
            return c

    def clear(self):
        for (name, b) in self.disp.items():
            self.disp[name].visible = False
            del self.disp[name]

    def pause(self):
        print 'Pausing window'

    def playback(self, capture = None, delay = 1):
        for entry in capture:
            if entry[0] == 'clear':
                self.clear()
                time.sleep(delay)
            elif entry[0] == 'pause':
                time.sleep(delay)
            else:
                self.draw(entry[0], color=entry[1], opacity=entry[2])
                self.update()

    def update(self):
        pass

    cpdef draw (self, prim, str color = None, float opacity = 1.0):
        if noWindow: return
        self.window.select()
        name = prim.properties['name']
        if name in self.disp:
            b = self.disp[name]
            b.visible = False
            del self.disp[name]
            del b
        c = self.color(color)
        self.disp[name] = visual.convex(pos = prim.vertices().tolist(), color = c)
