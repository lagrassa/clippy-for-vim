#!/usr/local/bin/python
# View a file in graphviz.  Add termination character, if necessary.
from sys import argv
from os import SEEK_CUR, SEEK_END, system # SEEK_SET

script, fileName = argv

f = open(fileName, 'r+')
# Go to last character
f.seek(-1, SEEK_END)
c = f.read(1)
while c.isspace() and f.tell() > 1:
    f.seek(-2, SEEK_CUR)
    c = f.read(1)

if c.isspace():
    print 'No non-space chars in file'
else:
    print 'last nonspace char:', c, 'at loc', f.tell()-1
    if c != '}':
        f.truncate()
        f.write('}')
f.close()

# Open in graphviz, letting Mac OS figure out what to do
system("open "+fileName)

