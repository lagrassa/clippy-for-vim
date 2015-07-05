from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import sys
import os
import numpy

def convert(filename, i):
    name = filename[:i]
    ext_modules = [Extension(name, [filename], include_dirs=[numpy.get_include()])]
    setup(
      cmdclass = {'build_ext': build_ext},
      ext_modules = ext_modules
    )

print "Input filename (name.pyx)"
filename = raw_input()
#filename = "all"
num = 0

if filename.lower() == "all":
    path = os.getcwd()
    dirList=os.listdir(path)
    dirList = sorted(dirList)
    # setup(
    #     name = "gjk",
    #     ext_modules=[
    #     Extension("gjk",
    #               sources=["./gjk_def.pyx", "./gjk/gjk.c"],
    #               include_dirs=[".", "./gjk"],
    #               language="c"),
    #     ],
    #     cmdclass={"build_ext": build_ext},
    #     )
    for fname in dirList:
        if fname.find("gjk_def") >= 0: continue
	i = fname.find(".pyx")
	if i != -1 and fname.find("~") == -1:
	    convert(fname, i)
	    num+=1
    print str(num) + " files processed"
    quit()

if filename.lower() == "clear":
    path = os.getcwd()
    dirList=os.listdir(path)
    dirList = sorted(dirList)
    for fname in dirList:
	i1 = fname.find(".c")
	i2 = fname.find(".so")
	if i1 != -1 or i2 != -1:
 	    print "Removing: " + fname
	    os.remove(fname)
	    num+=1
    print str(num) + " files processed"
    quit()
    
i = filename.find(".pyx")
if i == -1:
	    print "Error: Invalid filetype (name.pyx)"
	    quit()
else:
    convert(filename, i)
    num+=1
    print str(num) + " files processed"

#python setup.py build_ext --inplace
# all
#Caelan Garrett
