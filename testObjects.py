import math
import string
import hu
import numpy as np
import shapes
from pr2Util import GDesc
import planGlobals as glob

def Ba(bb, **prop): return shapes.BoxAligned(np.array(bb), None, **prop)
def Sh(args, **prop): return shapes.Shape(list(args), None, **prop)

def hor((x0, x1), y, dz, w):
    return Ba([(x0, y-w/2, 0), (x1, y+w/2.0, dz)])

def ver(x, (y0, y1), dz, w, extendSingleSide=False):
    if not extendSingleSide:
        return Ba([(x-w/2., y0, 0), (x+w/2.0, y1, dz)])
    return Ba([(x-w, y0, 0.), (x, y1, dz)])

#def place((x0, x1), (y0, y1), (z0, z1)):
#    return Ba([(x0, y0, z0), (x1, y1, z1)])

# Symmetries
sym0 = ({4 : 4}, {4 : [hu.Pose(0.,0.,0.,0.)]})
sym2 = ({4 : 4}, {4 : [hu.Pose(0.,0.,0.,0.), hu.Pose(0.,0.,0.,math.pi)]})
sym4 = ({4 : 4}, {4 : [hu.Pose(0.,0.,0.,(1./2.)*math.pi*x) for x in range(4)]})
sym6 = ({4 : 4}, {4 : [hu.Pose(0.,0.,0., (1./3.)*math.pi*x) for x in range(6)]})
# Grasps
# from the side
gMat0 = hu.Transform(np.array([(0.,1.,0.,0.),
                               (0.,0.,1.,-0.025),
                               (1.,0.,0.,0.01),
                               (0.,0.,0.,1.)]))
gMat1 = hu.Transform(np.array([(0.,-1.,0.,0.),
                               (0.,0.,-1.,0.025),
                               (1.,0.,0.,0.01),
                               (0.,0.,0.,1.)]))
# from the top
gMat2= hu.Transform(np.array([(-1.,0.,0.,0.),
                              (0.,0.,-1.,0.025),
                              (0.,-1.,0.,0.),
                              (0.,0.,0.,1.)]))
gMat3= hu.Transform(np.array([(1.,0.,0.,0.),
                              (0.,0.,1.,-0.025),
                              (0.,-1.,0.,0.),
                              (0.,0.,0.,1.)]))

gdesc0 = lambda obj: GDesc(obj, gMat0, 0.05, 0.05, 0.025)
gdesc1 = lambda obj: GDesc(obj, gMat1, 0.05, 0.05, 0.025)
gdesc2 = lambda obj: GDesc(obj, gMat2, 0.05, 0.05, 0.025)
gdesc3 = lambda obj: GDesc(obj, gMat3, 0.05, 0.05, 0.025)

colors = ['red', 'green', 'blue', 'cyan', 'purple', 'pink', 'orange']
def pickColor(name):
    if name[-1] in string.uppercase:
        cn = len(colors)
        return colors[string.uppercase.index(name[-1])%cn]
    else:
        return 'black'

# Need to add Top to manipulable objects...
# extraHeight = 1.5*height+0.01
# bbox = bboxGrow(thing.bbox(), np.array([0.075, 0.075, extraHeight]))
# regName = objName+'Top'
# tr('rig', ('Region', regName), bbox)
# world.addObjectRegion(objName, regName, Sh([Ba(bbox)], name=regName),
#                       hu.Pose(0.0,0.0,2.0* height +extraHeight,0.0))

def makeSoda(dx=0.0445, dy=0.027, dz=0.1175, name='objA', color=None):
    glob.graspDesc['soda'] = []
    if glob.useHorizontal: glob.graspDesc['soda'].extend([gdesc0(name), gdesc1(name)])
    if glob.useVertical: glob.graspDesc['soda'].extend([gdesc2(name), gdesc3(name)])
    color = color or pickColor(name)
    return (Sh([Ba([(-dx, -dy, 0.), (dx, dy, dz)])], name=name, color=color), [])
glob.graspableNames.append('obj'); glob.pushableNames.append('obj')
glob.graspableNames.append('soda'); glob.pushableNames.append('soda')
glob.objectSymmetries['soda'] = sym2
glob.objectTypes['obj'] = 'soda'
glob.objectTypes['soda'] = 'soda'
glob.constructor['soda'] = makeSoda

def makeBig(dx=0.0445, dy=0.055, dz=0.1175, name='bigA', color=None):
    color = color or pickColor(name)
    return (Sh([Ba([(-dx, -dy, 0.), (dx, dy, dz)])],
               name=name, color=color), [])
glob.pushableNames.append('big')
glob.objectSymmetries['big'] = sym4
glob.objectTypes['big'] = 'big'
glob.constructor['big'] = makeBig

def makeTall(dx=0.0445, dy=0.0445, dz=0.2, name='tallA', color=None):
    color = color or pickColor(name)
    return (Sh([Ba([(-dx, -dy, 0.), (dx, dy, dz)])],
               name=name, color=color), [])
glob.pushableNames.append('tall')
glob.objectSymmetries['tall'] = sym2
glob.objectTypes['tall'] = 'tall'
glob.constructor['tall'] = makeTall

def makeBigBar(dx=0.1, dy=0.0445, dz=0.2, name='bar', color=None):
    color = color or pickColor(name)
    return (Sh([Ba([(-dx, -dy, 0.), (dx, dy, dz)])],
               name=name, color=color), [])
glob.pushableNames.append('bar')
glob.objectSymmetries['bar'] = sym2
glob.objectTypes['bar'] = 'bar'
glob.constructor['bar'] = makeBigBar

soupZ = 0.1
def makeSoup(radius=0.0675/2, height=0.1, name='soup', color='red'):
    color = color or pickColor(name)
    # TODO: grasps...
    raw_input('No grasps defined for soup')
    glob.objectSymmetries['bar'] = sym6
    return (Sh([shapes.Ngon(radius, height, 6)], name=name, color=color), [])
glob.graspableNames.append('soup'); glob.pushableNames.append('soup')
glob.objectTypes['soup'] = 'soup'
glob.constructor['soup'] = makeSoup

def makeSolidTable(dx=0.603, dy=0.298, dz=0.67, name='tableSolid', width=0.1, color = 'orange'):
    table = Ba([(-dx, -dy, 0.), (dx, dy, dz)], name=name+'Body', color=color)
    reg = [Ba([(-dx, -dy, 0.), (dx, dy, dz)], name=name+'Top', color=color),
           Ba([(0.2, -dy, 0.), (dx, dy, dz)], name=name+'Left', color=color),
           Ba([(-dx, -dy, 0.), (-0.2, dy, dz)], name=name+'Right', color=color)]
    regions = [(r, hu.Pose(0.,0.,dz/2,0.)) for r in reg]
    return (Sh([table], name=name, color=color), regions)
glob.objectSymmetries['solidTable'] = sym2
glob.objectTypes['solidTable'] = 'solidTable'
glob.constructor['solidTable'] = makeSolidTable

tZ = 0.68                               # table height + 0.01
def makeLegTable(dx=0.603, dy=0.298, dz=0.67, name='table1', width=0.1, color = 'orange'):
    reg = [Ba([(-dx, -dy, 0.), (dx, dy, dz)], name=name+'Top', color=color),
           Ba([(0.2, -dy, 0.), (dx, dy, dz)], name=name+'Left', color=color),
           Ba([(-dx, -dy, 0.), (-0.2, dy, dz)], name=name+'Right', color=color),
           Ba([(-0.2*dx, 0, 0.), (0.2*dx, dy, dz)], name=name+'MidFront', color=color),
           Ba([(-0.2*dx, -dy, 0.), (0.2*dx, 0, dz)], name=name+'MidRear', color=color)]
    regions = [(r, hu.Pose(0.,0.,dz/2,0.)) for r in reg]
    table = [\
        Ba([(-dx, -dy, dz-width), (dx, dy, dz)],
           name=name+ 'top', color=color),
        Ba([(-dx,      -dy, 0.0),
            (-dx+width, dy, dz-width)],
           name=name+' leg 1', color=color),
        Ba([(dx-width, -dy, 0.0),
            (dx,       dy, dz-width)],
           name=name+' leg 2', color=color)
        ]
    return (Sh(table, name = name, color=color), regions)
glob.objectSymmetries['table'] = sym2
glob.objectTypes['table'] = 'table'
glob.constructor['table'] = makeLegTable

tZ = 0.74                               # table height + 0.01
def makeIkeaTable(dx=0.50, dy=0.30, dz=0.73, name='tableIkea', width=0.1, color = 'brown'):
    reg = [Ba([(-dx, -dy, 0.), (dx, dy, dz)], name=name+'Top', color=color),
           Ba([(0.2, -dy, 0.), (dx, dy, dz)], name=name+'Left', color=color),
           Ba([(-dx, -dy, 0.), (-0.2, dy, dz)], name=name+'Right', color=color),
           Ba([(-0.2*dx, 0, 0.), (0.2*dx, dy, dz)], name=name+'MidFront', color=color),
           Ba([(-0.2*dx, -dy, 0.), (0.2*dx, 0, dz)], name=name+'MidRear', color=color)]
    regions = [(r, hu.Pose(0.,0.,dz/2,0.)) for r in reg]
    table = [\
        Ba([(-dx, -dy, dz-width), (dx, dy, dz)],
           name=name+ 'top', color=color),
        Ba([(-dx,      -dy, 0.0),
            (-dx+width, -dy+width, dz-width)],
           name=name+' leg 1', color=color),
        Ba([(dx-width, -dy, 0.0),
            (dx,       -dy+width, dz-width)],
           name=name+' leg 2', color=color),
        Ba([(-dx,      dy-width, 0.0),
            (-dx+width, dy, dz-width)],
           name=name+' leg 3', color=color),
        Ba([(dx-width,  dy-width, 0.0),
            (dx,        dy, dz-width)],
           name=name+' leg 4', color=color)
        ]
    return (Sh(table, name = name, color=color), regions)
glob.objectSymmetries['tableIkea'] = sym2
glob.objectTypes['tableIkea'] = 'tableIkea'
glob.constructor['tableIkea'] = makeIkeaTable
def makeIkeaShelves(dx=0.185, dy=0.305, dz=0.51,
                width = 0.02, nshelf = 2,
                name='ikeaShelves', color='brown'):
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
        spaceName = name+'_space_'+str(i+1)
        space = Ba([(-dx+eps, -dy-width+eps, eps),
                    (dx-eps, dy+width-eps, (dz/nshelf) - width - eps)],
                   color='green', name=spaceName)
        space = Sh([space], name=spaceName, color='green')
        shelfSpaces.append((space, hu.Pose(0,0,bot+eps-(dz/2),0)))
    shelves = Sh(sidePrims + shelfRungs, name = name, color=color)
    return (shelves, shelfSpaces)
glob.objectSymmetries['ikeaShelves'] = sym0
glob.objectTypes['ikeaShelves'] = 'ikeaShelves'
glob.constructor['ikeaShelves'] = makeIkeaShelves

# def makeTableShelves(dx=shelfDepth/2.0, dy=0.305, dz=0.45,
#                 width = shelfWidth, nshelf = 2,
#                 name='tableShelves', color='orange'):
#     coolerPose = hu.Pose(0.0, 0.0, tZ, -math.pi/2)
#     shelvesPose = hu.Pose(0.0, 0.0, tZ+coolerZ, -math.pi/2)
#     tH = 0.67                           # table height
#     cooler = Sh([Ba([(-0.12, -0.165, 0), (0.12, 0.165, coolerZ)],
#                     name='cooler', color=color)],
#                 name='cooler', color=color)
#     table = makeTable(0.603, 0.298, tH, name = 'table1', color=color)
#     shelves, shelfSpaces = makeShelves(dx, dy, dz, width, nshelf, name='tableShelves', color=color)
#     offset = shelvesPose.compose(hu.Pose(0.,0.,-(tH/2+coolerZ/2),0.))
#     shelfSpaces = [(s, pose.compose(offset)) for (s, pose) in shelfSpaces]
#     obj = Sh( shelves.applyTrans(shelvesPose).parts() \
#               + cooler.applyTrans(coolerPose).parts() \
#               + table.parts(),
#               name=name, color=color)
#     return (obj, shelfSpaces)
# glob.objectSymmetries['shelves'] = sym0
# glob.objectTypes['shelves'] = 'shelves'

eps = 0.01
epsz = 0.02
shelfDepth = 0.3
shelfWidth = 0.02
coolerZ = 0.225

def makeCoolShelves(dx=shelfDepth/2.0, dy=0.305, dz=0.45,
                      width = shelfWidth, nshelf = 2,
                      name='coolShelves', color='orange'):
    coolerPose = hu.Pose(0.0, 0.0, 0.0, -math.pi/2)
    shelvesPose = hu.Pose(0.0, 0.0, coolerZ, -math.pi/2)
    cooler = Sh([Ba([(-0.12, -0.165, 0), (0.12, 0.165, coolerZ)],
                    name='cooler', color=color)],
                name='cooler', color=color)
    shelves, shelfSpaces = makeShelves(dx, dy, dz, width, nshelf, name='coolShelves', color=color)
    offset = shelvesPose.compose(hu.Pose(0.,0.,-(coolerZ/2),0.))
    shelfSpaces = [(s,pose.compose(offset)) for (s, pose) in shelfSpaces]
    obj = Sh( shelves.applyTrans(shelvesPose).parts() \
              + cooler.applyTrans(coolerPose).parts(),
              name=name, color=color)
    return (obj, shelfSpaces)
glob.objectSymmetries['coolShelves'] = sym0
glob.objectTypes['cool'] = 'coolShelves'
glob.constructor['coolShelves'] = makeCoolShelves

# Other permanent objects
# cupboard1 = Sh([place((-0.25, 0.25), (-0.05, 0.05), (0.0, 0.4))],
#                  name = 'cupboardSide1', color='orange')
# cupboard2 = Sh([place((-0.25, 0.25), (-0.05, 0.06), (0.0, 0.4))],
#                  name = 'cupboardSide2', color='orange')



