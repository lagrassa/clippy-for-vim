from shapes import Prim, Shape

# Reading WRL files
def readWrl(filename, name='wrlObj', scale=1.0):

    def readOneObj():
        vl = []
        while True:
            line = fl.readline()
            split = line.split(',')
            if len(split) != 2:
                break
            split = split[0].split()
            if len(split) == 3:
                vl.append(np.array([scale*float(x) for x in split]+[1.0]))
            else:
                break
        print '    verts', len(vl), 
        verts = np.vstack(vl).T
        while line.split()[0] != 'coordIndex':
            line = fl.readline()
        line = fl.readline()
        faces = []
        while True:
            line = fl.readline()
            split = line.split(',')
            if len(split) > 3:
                faces.append(np.array([int(x) for x in split[:3]]))
            else:
                break
        print 'faces', len(faces)
        return Prim(verts, faces, hu.Pose(0,0,0,0), None, name=name+str(len(prims)))

    fl = open(filename)
    assert fl.readline().split()[0] == '#VRML'
    prims = []
    while True:
        line = fl.readline()
        if not line: break
        split = line.split()
        if not split or split[0] != 'point':
            continue
        else:
            print 'Object', len(prims)
            prims.append(readOneObj())
    return Shape(prims, None)

