import pdb
import math

def kl(p, q, domain):
    return sum([p(x) * (math.log(p(x)) - math.log(q(x))) for x in domain])

def iter(ta, tb, p, n):
    def q((a, b),):
        return (ta if a else (1-ta)) * (tb if b else (1-tb))

    def updatea(tb, p):
        # ta is prob a = 1.  Actually, needs to be normalized!
        ta = math.exp(tb*math.log(p((1, 1))) + (1 - tb)*math.log(p((1, 0))))
        tNota = math.exp(tb*math.log(p((0, 1))) + (1 - tb)*math.log(p((0, 0))))
        (ta, tNota) =  ta / (ta + tNota), tNota / (ta + tNota)
        return ta
    def updateb(ta, p):
        # tb is prob b = 1
        tb = math.exp(ta*math.log(p((1, 1))) + (1 - ta)*math.log(p((0, 1))))
        tNotb = math.exp(ta*math.log(p((1, 0))) + (1 - ta)*math.log(p((0, 0))))
        (tb, tNotb) =  tb / (tb + tNotb), tNotb / (tb + tNotb)
        return tb

    for i in range(n):
        #print ta, tb, kl(q, p, [(0, 0), (0, 1), (1, 0), (1, 1)])
        print '{', ta, ',', tb, '},'
        ta = updatea(tb, p)
        print '{', ta, ',', tb, '},'
        tb = updateb(ta, p)
def p2((a, b),):
    return ((0.05, 0.45), (0.45, 0.05))[a][b]
def p3((a, b),):
    return ((0.24, 0.06), (0.14, 0.56))[a][b]
    
print 'loaded'
