import sys
from testPr2 import testFunc

print sys.argv
n = 0
hierarchical = True
alwaysReplan = False
if len(sys.argv) > 1:
    n = sys.argv[1]
if len(sys.argv) > 2:
    hierarchical = eval(sys.argv[2])
if len(sys.argv) > 3:
    alwaysReplan = eval(sys.argv[3])

testFunc(n, easy=True, rip = False, hierarchical = hierarchical, alwaysReplan = alwaysReplan)

