import pdb
import math

class Test:
    def __init__(self):
        raw_input('?')
        self.a = {}
        self.b = {1: {}, 2: {}}
    def __str__(self):
        return "Test a=%s, b=%s"%(self.a,self.b)
    __repr__ = __str__


Test()
