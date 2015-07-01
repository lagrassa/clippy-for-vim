############################################################################
# DrawingWindow.py - Michael Haimes, Leslie Kaelbling                      #
############################################################################
#    MIT SoaR 2.0 - A Python abstraction layer for MobileRobots            #
#    Pioneer3DX robots, and a simulator to simulate their operation        #
#    in a python environment (for testing)                                 #
#                                                                          #
#    Copyright (C) 2006-2007 Michael Haimes <mhaimes@mit.edu>              #
#                                                                          #
#   This program is free software; you can redistribute it and/or modify   #
#   it under the terms of the GNU General Public License as published by   #
#   the Free Software Foundation; either version 2 of the License, or      #
#   (at your option) any later version.                                    #
#                                                                          #
#   This program is distributed in the hope that it will be useful,        #
#   but WITHOUT ANY WARRANTY; without even the implied warranty of         #
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the          #
#   GNU General Public License for more details.                           #
#                                                                          #
#   You should have received a copy of the GNU General Public License along#
#   with this program; if not, write to the Free Software Foundation, Inc.,#
#   51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.            #
############################################################################

### Hacked by LPK to be independent of SoaR
### Hacked by TLP to write PIL images

from Tkinter import *
import math
import Image, ImageDraw

class Thing:
    pass

class DrawingWindow:

    def __init__(self, windowWidth, windowHeight, xMin, xMax, yMin, yMax,
                 title, parent=None): 
	self.title = title
        # Make a window to draw on
	if parent:	
            self.parent = parent
            self.top = parent.getWindow(title)
	else:
            self.tk=Tk()
            self.tk.withdraw()
            self.top=Toplevel(self.tk)
            self.top.wm_title(title)
            self.top.protocol('WM_DELETE_WINDOW', self.top.destroy)
	self.windowWidth = windowWidth
        self.windowHeight = windowHeight
        self.canvas = Canvas(self.top, width=self.windowWidth,
                             height=self.windowHeight, background="white")
        self.canvas.pack()
        # multiply an input value by this to get pixels
        self.xScale = windowWidth / float(xMax - xMin)
        self.yScale = windowHeight / float(yMax - yMin)

        self.drawingframe = Thing()
        self.drawingframe.width_px = windowWidth

        self.xMin = xMin
        self.yMin = yMin
        self.xMax = xMax
        self.yMax = yMax

        self.image = Image.new("RGB", (int(self.windowWidth), int(self.windowHeight)), 'white')
        self.draw = ImageDraw.Draw(self.image)

    def update(self):
        self.canvas.update()

    def scaleX(self, x):
        return self.xScale * (x - self.xMin)
    
    def scaleY(self, y):
        return self.windowHeight - self.yScale * (y - self.yMin)

    def drawPoint(self, x, y, color = "blue"):
        windowX = self.scaleX(x)
        windowY = self.scaleY(y)
        self.draw.point([(windowX, windowY)], fill = color)
        return self.canvas.create_rectangle(windowX-1, windowY-1, windowX+1,
                                     windowY+1, fill = color, outline = color)

    def drawText(self, x, y, label):
        windowX = self.scaleX(x)
        windowY = self.scaleY(y)
        self.draw.text((int(windowX), int(windowY)), label, fill='black')
        return self.canvas.create_text(windowX, windowY, text=label)
        # font="Arial 20",fill="#ff0000"


    def drawRect(self, (x1,y1), (x2,y2), color = "black", outline = 'black'):
        points = (self.scaleX(x1), self.scaleY(y1), self.scaleX(x2), self.scaleY(y2))
        self.draw.rectangle([int(x) for x in points], outline=outline, fill=color)
        return self.canvas.create_rectangle(*points,
                                            fill = color,
                                            outline = outline)

    def drawOval(self, (x1,y1), (x2,y2), color = "black", outline = 'black'):
        points = (self.scaleX(x1), self.scaleY(y1), self.scaleX(x2), self.scaleY(y2))
        self.draw.ellipse([int(x) for x in points], outline=outline, fill=color)
        return self.canvas.create_oval(*points,
                                       fill = color)

    def drawLineSeg(self, x1, y1, x2, y2, color = "black", width = 2):
        points = (self.scaleX(x1),self.scaleY(y1), self.scaleX(x2),self.scaleY(y2))
        self.draw.line([int(x) for x in points], fill=color, width=width)
        return self.canvas.create_line(*points,
                                       fill = color,
                                       width = width)

    def drawLine(self, (a,b,c), color = "black"):
        if abs(b) < 0.001:
            startX = self.scaleX(-c/a)
            endX = self.scaleX(-c/a)
            startY = self.scaleY(self.yMin)
            endY = self.scaleY(self.yMax)
        else:
            startX = self.scaleX(self.xMin)
            startY = self.scaleY(- (a * self.xMin + c) / b)
            endX = self.scaleX(self.xMax)
            endY = self.scaleY(- (a * self.xMax + c) / b)
        points = (startX, startY, endX, endY)
        self.draw.line([int(x) for x in points], fill=color)
        return self.canvas.create_line(*points, fill = color)

    def delete(self, thing):
        self.canvas.delete(thing)
  
    def clear(self):
        self.canvas.delete("all")
        del self.draw
        del self.image
        self.image = Image.new("RGB", (int(self.windowWidth), int(self.windowHeight)), 'white')
        self.draw = ImageDraw.Draw(self.image)

    def saveImage(self, file):
        self.image.save(file, 'PNG')
