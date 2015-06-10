import operator
import datetime
import numpy as np
import collections
from itertools import chain, combinations, cycle, islice
import copy

def timeString():
    return str(datetime.datetime.now()).replace(' ','').replace(':','_').\
           replace('.','_')

class SymbolGenerator:
    """
    Generate new symbols guaranteed to be different from one another
    Optionally, supply a prefix for mnemonic purposes
    Call gensym("foo") to get a symbol like 'foo37'
    """
    def __init__(self):
        self.count = 0
    def gensym(self, prefix = 'i', zeroPadded = False):
        self.count += 1
        if zeroPadded:
            return (prefix + '_%08i') % self.count
        else:
            return prefix + '_' + str(self.count)

gensym = SymbolGenerator().gensym
"""Call this function to get a new symbol"""

def tuplify(x):
    if type(x) in (tuple, list):
        return tuple([tuplify(y) for y in x])
    elif type(x) == dict:
        return tuple(sorted(tuplify(y) for y in x.iteritems()))
    else:
        return x

def floatify(x):
    if type(x) in (tuple, list):
        return tuple([floatify(y) for y in x])
    elif isAnyVar(x) or type(x) == str:
        return x
    else:
        return float(x)

# Return a flat list of elements.  Stuff should be a list.
def squash(stuff):
    result = []
    for thing in stuff:
        if type(thing) == list:
            result += squash(thing)
        else:
            result.append(thing)
    return result

# Return a flat set of elements
def squashSets(stuff):
    result = set([])
    for thing in stuff:
        if type(thing) == set:
            result = result.union(thing)
        else:
            result.add(thing)
    return result

# Return a dictionary
def squashDicts(stuff):
    result = {}
    for thing in stuff:
        result.update(thing)
    return result

# stuff is a list of dictionaries, where the values are sets.
#  merge so
# that, in the result, each key is associated with the union of the
# sets that were their values in the individual dictionaries.
def mergeDicts(dicts):
    result = {}
    for d in dicts:
        for (k, v) in d.items():
            if k in result:
                result[k] = result[k].union(v)
            else:
                result[k] = v
    return result

# Squash one level
def squashOne(stuff):
    result = []
    for thing in stuff:
        result.extend(thing)
    return result

def powerset(xs, includeEmpty = True):
    start = 0 if includeEmpty else 1
    # Ugly because I lost patience with getting the generators to work
    combos = [list(combinations(xs,n)) for n in range(start, len(xs)+1)]
    return chain(*combos)

# If equiv is true, don't map two objects that should be different
# into the same string

# if eq = True, use 6 digits otherwise 3

def prettyString(struct, eq = True):
    if type(struct) == list:
        return '[' + ', '.join([prettyString(item, eq) for item in struct]) +']'
    elif type(struct) == tuple:
        return '(' + ', '.join([prettyString(item,eq) for item in struct]) + ')'
    elif type(struct) == dict:
        return '{' + ', '.join([str(item)+':'+ prettyString(struct[item], eq) \
                                             for item in struct]) + '}'
    elif isinstance(struct, np.ndarray):
        # Could make this prettier...
        return str(struct)
    elif type(struct)!= int and type(struct) != bool and \
             hasattr(struct, '__float__'):
        struct = round(struct, 6 if eq else 3)
        if struct == 0: struct = 0      #  catch stupid -0.0
        return "%5.3f" % struct
    elif hasattr(struct, 'prettyString'):
        return struct.prettyString(eq)
    else:
        return str(struct)

class Stack:
    def __init__(self):
        self.__storage = []

    def isEmpty(self):
        return len(self.__storage) == 0

    def push(self,p):
        p.level = len(self.__storage)
        self.__storage.append(p)

    def pop(self):
        return self.__storage.pop()

    def top(self):
        return self.__storage[-1]

    def guts(self):
        return self.__storage

    def elt(self, i):
        return self.__storage[i]

    def popToTop(self, i):
        self.__storage = self.__storage[0:i]

    def size(self):
        return len(self.__storage)

def within(v1, v2, eps):
    return abs(v1 - v2) < eps

def clip(v, vMin, vMax):
    """
    @param v: number
    @param vMin: number (may be None, if no limit)
    @param vMax: number greater than C{vMin} (may be None, if no limit)
    @returns: If C{vMin <= v <= vMax}, then return C{v}; if C{v <
    vMin} return C{vMin}; else return C{vMax}
    """
    if vMin == None:
        if vMax == None:
            return v
        else:
            return min(v, vMax)
    else:
        if vMax == None:
            return max(v, vMin)
        else:
            return max(min(v, vMax), vMin)

def argmax(l, f):
    """
    @param l: C{List} of items
    @param f: C{Procedure} that maps an item into a numeric score
    @returns: the element of C{l} that has the highest score
    """
    vals = [f(x) for x in l]
    return l[vals.index(max(vals))]

def isStruct(thing):
    return isinstance(thing, collections.Iterable) and not type(thing)==str

def combineBindings(a, b):
    if a == False or b == False:
        return False
    return dict([(k, applyBindings(v, b)) for (k, v) in \
                 a.iteritems()] + \
                [(k, applyBindings(v, a)) for (k, v) in \
                 b.iteritems()])

# Doesn't extend bindings if there are bindings in s
def applyBindings1(s, bindings):
    assert goodBindings(bindings)
    if not bindings:
        return s
    tps = type(s)
    if tps == str:
        if s in bindings:
            return applyBindings1(bindings[s], bindings)
        else:
            return s
    if tps == dict:
        return dict(applyBindings1(s.items(), bindings)) 
    elif tps == tuple:
        return tuple([applyBindings1(thing, bindings) for thing in s])
    elif tps == set:
        return set([applyBindings1(thing, bindings) for thing in s])
    elif tps == list:
        return [applyBindings1(thing, bindings) for thing in s]
    elif hasattr(s, 'applyBindings'):
        return s.applyBindings(bindings)
    else:
        return s

applyBindings = applyBindings1

def customCopy(s):
    tps = type(s)
    if tps == str:
        return s[:]
    if tps == dict:
        return dict(customCopy(s.items()))
    elif tps == tuple:
        return tuple([customCopy(thing) for thing in s])
    elif tps == set:
        return set([customCopy(thing) for thing in s])
    elif tps == list:
        return [customCopy(thing) for thing in s]
    elif hasattr(s, 'copy'):
        return s.copy()
    else:
        return copy.copy(s)


# A string starting with an upper case letter is a planning operator
# schema variable
def isVar(thing):
    return type(thing) == str and thing[0].isalpha() and thing[0].isupper()

def isConstraintVar(thing):
    return type(thing) == str and thing[0] == '?'

def makeConstraintVar(v):
    return gensym('?'+v)

def makeVar(v):
    return gensym(v)

def isAnyVar(thing):
    # This gets called millions of times, so it's worth making more efficient - TLP
    # return isVar(thing) or isConstraintVar(thing)
    if isinstance(thing, str):
        return thing[0] == '?' or (thing[0].isalpha() and thing[0].isupper())

def lookup(thing, bindings):
    if isAnyVar(thing) and thing in bindings:
        return bindings[thing]
    else:
        return thing

def isGround(thing):
    if hasattr(thing, 'isGround'):
        return thing.isGround()
    elif isStruct(thing):
        return all(isGround(x) for x in thing)
    else:
        return not isAnyVar(thing)

def listUnion(l1, l2):
    return list(l1) + [x for x in l2 if not x in l1]

def average(items):
    return sum(items)/float(len(items))

def floatRange(minVal, maxVal, numSteps):
    vals = []
    v = minVal
    step = float(maxVal - minVal) / numSteps
    while v < maxVal:
        vals.append(v)
        v += step
    return vals

# Assumes funTable is monotonic
def inverseTableLookup(pQuery, table):
    for i in range(len(table)):
        (r, p) = funTable[i]
        if pQuery < p:
            if i == 0: return r
            # interpolate
            (rp, pp) = table[i-1]
            # dr / dp
            rate = (r - rp) / (p - pp)
            return pp + (pQuery - pp) * rate
    return table[-1][p]

    
# Variable name manipulation
# names are of the form: pred(obj)_numbers

def varNameToPred(vname):
    if vname[0] == '?': vname = vname[1:]
    name = vname.rstrip('_1234567890')
    if '(' in name:
        return name[:name.index('(')]
    else:
        return name

def varNameToObj(vname):
    if '(' in vname:
        return vname[vname.index('(')+1 : vname.index(')')]

def matchLists(s1, s2, bindings = {}, starIsWild = True):
    if bindings == None or len(s1) != len(s2):
        return None
    if bindings != {}:
        bindings = copy.copy(bindings)
    for (a1, a2) in zip(s1, s2):
        bindings = matchTerms(a1, a2, bindings, starIsWild = starIsWild)
        if bindings == None: return bindings
    return bindings

# Returns bindings, no side effect.  One-sided wrt '*': that is a
# constant in t1 will match a * in t2, but not vv.
def matchTerms(t1, t2, bindings = {}, starIsWild = True):
    if bindings == None: return None
    bt1 = applyBindings1(t1, bindings)
    bt2 = applyBindings1(t2, bindings)
    if bt1 == bt2 or t1 == t2:
        pass
    elif isVar(bt1):
        bindings = extendBindings(bindings, bt1, t2)
    elif isVar(bt2):
        bindings = extendBindings(bindings, bt2, t1)
    elif bt2 == '*' and starIsWild:
        pass
    elif hasattr(bt1, 'matches') and hasattr(bt2, 'matches'):
        # maybe not a great test test...trying to see if they are both B fluents
        bindings = bt1.matches(bt2, bindings)
    else:
        bindings = None
        
    return bindings

def extendBindings(b, k, v):
    newB = copy.copy(b)
    newB[k] = v
    return newB


# Assert that all variables in the bindings are constraint vars.
# A bound value of None means a function failed
def goodBindings(b):
    return type(b) == dict and \
           all(var != val and val != None \
               for (var, val) in b.iteritems())

def getCVBindings(bindings):
    return dict([(var, val) for (var, val) in bindings.iteritems() if \
                 isConstraintVar(var)])

def getRegularBindings(bindings):
    return dict([(var, val) for (var, val) in bindings.iteritems() if \
                 not isConstraintVar(var)])
    
def extractVars(struct):
    if isConstraintVar(struct):
        return [struct]
    elif isStruct(struct):
        return reduce(orderedUnion, [extractVars(part) for part in struct], [])
    else:
        return []

def orderedUnion(s1, s2):
    result = s1
    for e in s2:
        if not e in result:
            result.append(e)
    return result    

# Return the largest element of dom that is less than v.
# in two dimensions
def floorInDomain2(v, dom):
    if v == None: return None
    # assumes dom is sorted
    for (i, vd) in enumerate(dom):
        if vd[0] > v[0] or vd[1] > v[1]:
            if i > 0:
                #print 'FID', v, dom[i-1]
                return dom[i-1]
            else:
                #print 'FID', v, None
                return None
    #print 'FID', v, dom[-1]
    return dom[-1]

def makeDiag(v):
    return np.diag(v)

def undiag(m):
    if m == None: return None
    return tuple(np.diag(m))

def roundUp(v, prec):
    vt = int(v * prec) / prec
    if v > vt:
        return int(v * prec + 1) / prec
    else:
        return vt

def roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    pending = len(iterables)
    nexts = cycle(iter(it).next for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = cycle(islice(nexts, pending))    
