
import numpy as np
import shapes

def Ba(bb, **prop): return shapes.BoxAligned(np.array(bb), None, **prop)
def Sh(args, **prop): return shapes.Shape(list(args), None, **prop)

def makeSoda(dx=0.0445, dy=0.027, dz=0.1175, name='objA', color='blue'):
    return Sh([Ba([(-dx, -dy, 0.), (dx, dy, dz)])], name=name, color=color)

soupZ = 0.1
def makeSoup(radius=0.0675/2, height=0.1, name='soup', color='red'):
    return Sh([shapes.Ngon(radius, height, 6)], name=name, color=color)

def makeSolidTable(dx=0.603, dy=0.298, dz=0.67, name='table2', width=0.1, color = 'orange'):
    return Sh([Ba([(-dx, -dy, 0.), (dx, dy, dz)])], name=name, color=color)

tZ = 0.68
def makeLegTable(dx=0.603, dy=0.298, dz=0.67, name='table1', width=0.1, color = 'orange'):
    legInset = 0.02
    legOver = 0.02
    return Sh([\
        Ba([(-dx, -dy, dz-width), (dx, dy, dz)],
           name=name+ 'top', color=color),
        Ba([(-dx,      -dy, 0.0),
            (-dx+width, dy, dz-width)],
           name=name+' leg 1', color=color),
        Ba([(dx-width, -dy, 0.0),
            (dx,       dy, dz-width)],
           name=name+' leg 2', color=color)
        ], name = name, color=color)

eps = 0.01
epsz = 0.02
shelfDepth = 0.3
shelfWidth = 0.02
coolerZ = 0.225

def makeShelves(dx=shelfDepth/2.0, dy=0.305, dz=0.45,
                width = shelfWidth, nshelf = 2,
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
        spaceName = name+'_space_'+str(i+1)
        space = Ba([(-dx+eps, -dy-width+eps, eps),
                    (dx-eps, dy+width-eps, (dz/nshelf) - width - eps)],
                   color='green', name=spaceName)
        space = Sh([space], name=spaceName, color='green')
        shelfSpaces.append((space, hu.Pose(0,0,bot+eps-(dz/2),0)))
    shelves = Sh(sidePrims + shelfRungs, name = name+'Body', color=color)
    return (shelves, shelfSpaces)

def makeTableShelves(dx=shelfDepth/2.0, dy=0.305, dz=0.45,
                width = shelfWidth, nshelf = 2,
                name='tableShelves', color='brown'):
    coolerPose = hu.Pose(0.0, 0.0, tZ, -math.pi/2)
    shelvesPose = hu.Pose(0.0, 0.0, tZ+coolerZ, -math.pi/2)
    tH = 0.67                           # table height
    cooler = Sh([Ba([(-0.12, -0.165, 0), (0.12, 0.165, coolerZ)],
                    name='cooler', color=color)],
                name='cooler', color=color)
    table = makeTable(0.603, 0.298, tH, name = 'table1', color=color)
    shelves, shelfSpaces = makeShelves(dx, dy, dz, width, nshelf, name='tableShelves', color=color)
    offset = shelvesPose.compose(hu.Pose(0.,0.,-(tH/2+coolerZ/2),0.))
    shelfSpaces = [(s,pose.compose(offset)) for (s, pose) in shelfSpaces]
    obj = Sh( shelves.applyTrans(shelvesPose).parts() \
              + cooler.applyTrans(coolerPose).parts() \
              + table.parts(),
              name=name, color=color)
    return (obj, shelfSpaces)

def makeCoolShelves(dx=shelfDepth/2.0, dy=0.305, dz=0.45,
                      width = shelfWidth, nshelf = 2,
                      name='coolShelves', color='brown'):
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
