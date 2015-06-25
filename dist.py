import pdb
import math
import operator
import miscUtil
from miscUtil import prettyString, makeDiag, undiag
import numpy as np
from numpy import *
import random
import operator as op
import copy

#from pylab import figure, show, rand
#from matplotlib.patches import Ellipse

class DiscreteDist:
    """
    Probability distribution over a discrete domain.  This is a
    generic superclass;  do not instantiate it.  Any subclass has to
    provide methods: C{prob}, C{setProb}, C{support}
    """
    def draw(self):
        """
        @returns: a randomly drawn element from the distribution
        """
        r = random.random()
        sum = 0.0
        for val in self.support():
            sum += self.prob(val)
            if r < sum:
                return val
        raise Exception, 'Failed to draw from:' + str(self)

    def __call__(self, val):
        return self.prob(val)

    def expectation(self, vals):
        return sum([self.prob(x) * vals[x] for x in self.support()])

    def expectationF(self, f):
        return sum([self.prob(x) * f(x) for x in self.support()])

    def maxProbElt(self):
        """
        @returns: The element in this domain with maximum probability
        """
        best = []                       # list of pairs (elt, p)
        for elt in self.support():
            p = self.prob(elt)
            if not best or p >= best[0][1]:
                if best and p == best[0][1]:
                    best.append((elt, p))
                else:
                    best = [(elt, p)]
        #return random.choice(best) if len(best) > 1 else best[0]
        return best[0]

    # Returns the x with highest weight
    def mode(self):
        return self.maxProbElt()[0]

    # Estimate the mean.  Assumes that + and * are defined on the elements
    def mean(self):
        return sum([x * self.prob(x) for x in self.support()])

    # Estimate the variance.  Assumes we can do x * x, and then scale
    # and sum the results.
    def variance(self):
        mu = self.mean()
        return sum([(x - mu)**2 * self.prob(x) for x in self.support()])

    def conditionOnVar(self, index, value):
        """
        @param index: index of a variable in the joint distribution
        @param value: value of that variable

        @returns: new distribution, conditioned on variable C{i}
        having value C{value}, and with variable C{i} removed from all
        of the elements (it's redundant at this point).
        """
        newElements = [e for e in self.support() if e[index] == value]
        z = sum([self.prob(e) for e in newElements])
        return DDist(dict([(removeElt(e, index), self.prob(e)/z) \
                                                       for e in newElements]))

    def verify(self, verbose = False):
        probs = [self.prob(e) for e in self.support()]
        z = sum(probs)
        if verbose:
            print '* in verify *'
            print 'min', min(probs), 'max', max(probs)
        assert z > 0.99999 and z < 1.00001, 'degenerate distribution ' + str(self)

       
def convertToDDist(oldD):
    newD = DDist({})
    for e in oldD.support():
        newD.addProb(e, oldD.prob(e))
    return newD

class DDist(DiscreteDist):
    """Discrete distribution represented as a dictionary.  Can be
    sparse, in the sense that elements that are not explicitly
    contained in the dictionary are assuemd to have zero probability."""
    def __init__(self, dictionary = {}, name = None):
        self.d = copy.copy(dictionary)
        """ Dictionary whose keys are elements of the domain and values
        are their probabilities. """
        self.name = name
        """Optional name;  used in sum-out operations"""

    def addProb(self, val, p):
        """
        Increase the probability of element C{val} by C{p}
        """
        self.setProb(val, self.prob(val) + p)
    def mulProb(self, val, p):
        """
        Multiply the probability of element C{val} by C{p}
        """
        self.setProb(val, self.prob(val) * p)

    def prob(self, elt):
        """
        @returns: the probability associated with C{elt}
        """
        if self.d.has_key(elt):
            return self.d[elt]
        else:
            return 0

    def setProb(self, elt, p):
        """
        @param elt: element of the domain
        @param p: probability
        Sets probability of C{elt} to be C{p}
        """
        self.d[elt] = p

    def support(self):
        """
        @returns: A list (in any order) of the elements of this
        distribution with non-zero probabability.
        """
        return [x for x in self.d.keys() if self.d[x] > 0]

    def normalize(self):
        """
        Divides all probabilities through by the sum of the values to
        ensure the distribution is normalized.

        Changes the distribution!!  (And returns it, for good measure)

        Generates an error if the sum of the current probability
        values is zero.
        """
        z = sum([self.prob(e) for e in self.support()])
        assert z > 0.0, 'degenerate distribution ' + str(self)
        alpha = 1.0 / z
        newD = {}
        for e in self.support():
            newD[e] = self.d[e] * alpha
        self.d = newD
        return self

    def normalizeOrSmooth(self):
        """
        If total prob is < 1, then spread that mass out uniformly
        (among elements with non-zero probabilty, seince we don't
        really know the universe; otherwiese, just normalize.
        """
        z = sum([self.prob(e) for e in self.support()])
        assert z > 0.0, 'degenerate distribution ' + str(self)
        newD = DDist({})
        if z < 1:
            beta = (1 - z) / len(self.support())
            for e in self.support():
                newD.addProb(e, beta)
        return self.normalize()

    def smooth(self, totalP):
        # redistribute p of the probability mass to the values that have
        # positive support
        n = len(self.support())
        p = totalP / n
        for e in self.support():
            self.addProb(e, p)
        self.normalize()

    def blur(self, blurDomain, blurFactor):
        """
        Adds small prob to all elements in C{blurDomain} and
        renormalizes.  Side effects the distribution.
        @param blurDomain: set of elements to be blurred
        @param blurFactor: how much to blur; 1 obliterates everything,
                           0 has no effect
        """
        eps = blurFactor * (1 / float(len(blurDomain)))
        for elt in blurDomain:
            self.addProb(elt, eps)
        self.normalize()

    # Used to be called map
    def project(self, f):
        """
        Return a new distribution with the same probabilities, but
        with each element replaced by f(elt)
        """
        newD = {}
        for (key, val) in self.d.items():
            newK = f(key)
            newD[newK] = val + newD[newK] if newK in newD else val
        return DDist(newD)

    def transitionUpdate(self, tDist):
        new = {}
        for sPrime in self.support(): #for each old state sPrime
            tDistS = tDist(sPrime) #distribution over ending states
            oldP = self.prob(sPrime)
            for s in tDistS.support():
                #prob of transitioning to s from sPrime                
                new[s] = new.get(s,0) + tDistS.prob(s)*oldP 
        self.d = new

    def obsUpdate(d, om, obs):
        for si in d.support():
            d.mulProb(si, om(si).prob(obs))
        d.normalize()

    # def pnm(self, delta, distMetric = lambda x, y: abs(x - y)):
    #     m = self.mode()
    #     return sum([p for (x, p) in self.d.items() if distMetric(x,m) < delta])

    def __repr__(self):
        if self.d.items():
            dictRepr = reduce(operator.add,[repr(k)+": "+prettyString(p)+", " \
                                            for (k, p) in self.d.items()])
        else:
            dictRepr = '{}'
        return "DDist(" + dictRepr[:-2] + ")"

    def ensureJDist(self):
        """
        If the argument is a C{DDist}, make it into a C{JDist}.
        """
        if isinstance(self, JDist):
            return self
        else:
            result = JDist([self.support()], name = [self.name])
            result.d = dict([((e,), self.prob(e)) for e in self.support()])
            return result

    def __hash__(self):
        return hash(frozenset(self.d.items()))
    def __eq__(self, other):
        return self.d == other.d
    def __ne__(self, other):
        return self.d != other.d
            
######################################################################
#   Special cases

def DeltaDist(v):
    """
    Distribution with all of its probability mass on value C{v}
    """
    return DDist({v:1.0})

def UniformDist(elts):
    """
    Uniform distribution over a given finite set of C{elts}
    @param elts: list of any kind of item
    """
    p = 1.0 / len(elts)
    return DDist(dict([(e, p) for e in elts]))

class MixtureDist(DiscreteDist):
    """
    A mixture of two probabability distributions, d1 and d2, with
    mixture parameter p.  Probability of an
    element x under this distribution is p * d1(x) + (1 - p) * d2(x).
    It is as if we first flip a probability-p coin to decide which
    distribution to draw from, and then choose from the approriate
    distribution.

    This implementation is lazy;  it stores the component
    distributions.  Alternatively, we could assume that d1 and d2 are
    DDists and compute a new DDist.
    """
    def __init__(self, d1, d2, p):
        self.d1 = d1
        self.d2 = d2
        self.p = p
        self.binom = DDist({True: p, False: 1 - p})
        
    def prob(self, elt):
        return self.p * self.d1.prob(elt) + (1 - self.p) * self.d2.prob(elt)

    def draw(self):
        if self.binom.draw():
            return self.d1.draw()
        else:
            return self.d2.draw()

    def support(self):
        return list(set(self.d1.support()).union(set(self.d2.support())))

    def __str__(self):
        result = 'MixtureDist({'
        elts = self.support()
        for x in elts[:-1]:
            result += str(x) + ' : ' + str(self.prob(x)) + ', '
        result += str(elts[-1]) + ' : ' + str(self.prob(elts[-1])) + '})'
        return result
    
    __repr__ = __str__

def MixtureDD(d1, d2, p):
    """
    A mixture of two probabability distributions, d1 and d2, with
    mixture parameter p.  Probability of an
    element x under this distribution is p * d1(x) + (1 - p) * d2(x).
    It is as if we first flip a probability-p coin to decide which
    distribution to draw from, and then choose from the approriate
    distribution.

    This implementation is eager: it computes a DDist
    distributions.  Alternatively, we could assume that d1 and d2 are
    DDists and compute a new DDist.
    """
    return DDist(dict([(e, p * d1.prob(e) + (1-p) * d2.prob(e)) \
                       for e in set(d1.support()).union(set(d2.support()))]))
    
def triangleDist(peak, halfWidth, lo = None, hi = None):
    """
    Construct and return a DDist over integers. The
    distribution will have its peak at index C{peak} and fall off
    linearly from there, reaching 0 at an index C{halfWidth} on
    either side of C{peak}.  Any probability mass that would be below
    C{lo} or above C{hi} is assigned to C{lo} or C{hi}
    """
    d = {}
    d[clip(peak, lo, hi)] = 1
    total = 1
    fhw = float(halfWidth)
    for offset in range(1, halfWidth):
        p = (halfWidth - offset) / fhw
        incrDictEntry(d, clip(peak + offset, lo, hi), p)
        incrDictEntry(d, clip(peak - offset, lo, hi), p)
        total += 2 * p
    for (elt, value) in d.items():
        d[elt] = value / total
    return DDist(d)

def squareDist(lo, hi, loLimit = None, hiLimit = None):
    """
    Construct and return a DDist over integers.  The
    distribution will have a uniform distribution on integers from
    lo to hi-1 (inclusive).
    Any probability mass that would be below
    C{lo} or above C{hi} is assigned to C{lo} or C{hi}.
    """
    d = {}
    p = 1.0 / (hi - lo)
    for i in range(lo, hi):
        incrDictEntry(d, clip(i, loLimit, hiLimit), p)
    return DDist(d)

def JDist(PA, PBgA):
    """
    Create a joint distribution on P(A, B) (in that order),
    represented as a C{DDist}
        
    @param PA: a C{DDist} on some random var A
    @param PBgA: a conditional probability distribution specifying
    P(B | A) (that is, a function from elements of A to C{DDist}
    on B)
    """
    d = {}
    for a in PA.support():
        for b in PBgA(a).support():
            d[(a, b)] = PA.prob(a) * PBgA(a).prob(b)
    return DDist(d)

def JDistIndep(PA, PB):
    """
    Create a joint distribution on P(A, B) (in that order),
    represented as a C{DDist}.  Assume independent.
        
    @param PA: a C{DDist} on some random var A
    @param PB: a C{DDist} on some random var B
    """
    d = {}
    for a in PA.support():
        for b in PB.support():
            d[(a, b)] = PA.prob(a) * PB.prob(b)
    return DDist(d)


def bayesEvidence(PA, PBgA, b):
    """
    @param PBgA: conditional distribution over B given A (function
    from values of a to C{DDist} over B)
    @param PA: prior on A
    @param b: evidence value for B = b 
    @returns: P(A | b)
    """
    # Remember that the order of the variables will be A, B
    return JDist(PA, PBgA).conditionOnVar(1, b)

def totalProbability(PA, PBgA):
    return JDist(PA, PBgA).project(lambda (a, b): b)

######################################################################
#   Continuous distribution

class GaussianDistribution:
    """
    Basic one-dimensional Gaussian.  
    """
    def __init__(self, mean, variance = None, stdev = None):
        self.mean = mean
        if variance:
            self.var = variance
            self.stdev = math.sqrt(self.var)
        elif stdev:
            self.stdev = stdev
            self.var = stdev**2
        else:
            raise Exception, 'Have to specify variance or stdev'

    def __str__(self):
        return 'Normal('+prettyString(self.mean)+', '+\
               prettyString(self.var)+')'

    def prob(self, v):
        return gaussian(v, self.mean, self.stdev)

    def mean(self):
        return self.mean

    def mode(self):
        return self.mean

    def variance(self):
        return self.var

    def cdf(self, value):
        return stats.norm(self.mean, self.stdev).cdf(value)

    def pnm(self, delta):
        return gaussPNM(self.stdev, delta)

    def draw(self):
        return random.normalvariate(self.mean, self.stdev)

    def update(self, obs, variance):
        varianceSum = self.var + variance
        self.mean = ((variance * self.mean) + (self.var * obs)) / \
                    varianceSum
        self.var = self.var * variance / varianceSum
        self.stdev = math.sqrt(self.var)

    def move(self, new, variance):
        self.mean = new
        self.var = self.var + variance
        self.stdev = math.sqrt(self.var)

    def reset(self, new, variance):
        self.mean = new
        self.var = variance
        self.stdev = math.sqrt(self.var)

class LogNormalDistribution:
    """
    log d ~ Normal(mu, sigma)
    Note that in some references we use \rho = 1/\sigma^2
    """
    def __init__(self, mu, sigma):
        self.mu = mu    # this is log of what we'd think of as mu   
        self.sigma = sigma

    def prob(self, v):
        return gaussian(math.log(v), self.mu, self.sigma)

    def mean(self):
        return exp(self.mu + self.sigma^2/2)

    def mode(self):
        return exp(self.mu - self.sigma^2)

    def median(self):
        return exp(self.mu)

    def variance(self):
        ss = self.sigma^2
        return exp(2 * self.mu + ss) * (exp(ss) - 1)

    def draw(self):
        return exp(random.normalvariate(self.mu, self.sigma))

def fixSigma(sigma, ridge = 0):
    # Can pass in ridge > 0 to ensure minimum eigenvalue is always >= ridge
    good = True
    for i in range(len(sigma)):
        for j in range(i):
            if sigma[i, j] != sigma[j, i]:
                #print 'found asymmetry mag:', abs(sigma[i, j] - sigma[j, i])
                good = False
    if not good:
        sigma = (sigma + sigma.T) / 2

    eigs = linalg.eigvalsh(sigma)
    if any([type(e) in (complex, complex128) for e in eigs]):
        print eigs
        print '** Symmetric, but complex eigs **'
    minEig = min(eigs)
    if minEig < 0:
        raw_input('** Not positive definite **'+str(minEig))
    elif minEig < ridge:
        print '** Adding ridge', ridge, 'because minEig is', minEig, '**'
    if minEig < ridge:
        sigma = sigma + 2 * (ridge - minEig) * identity(len(sigma))
    return sigma

# Uses numpy matrices
class MultivariateGaussianDistribution:
    def __init__(self, mu, sigma):
        self.mu = mat(mu)       # column vector
        self.sigma = fixSigma(mat(sigma))     # square pos def matrix

    def copy(self):
        return MultivariateGaussianDistribution(np.copy(self.mu),
                                                np.copy(self.sigma))

    def prob(self, v):
        d = len(v)
        norm = math.sqrt((2 * math.pi)**d * linalg.det(self.sigma))
        diff = v - self.mu
        if diff.shape == (1, 4):
            diff = diff.T
        return exp(-0.5 * diff.T * self.sigma.I * diff) / norm

    def logProb(self, v):
        d = len(v)
        norm = math.sqrt((2 * math.pi)**d * linalg.det(self.sigma))
        diff = v - self.mu
        return -0.5 * diff.T * self.sigma.I * diff - np.log(norm)
    
    def marginal(self, indices):
        mmu = self.mu.take(indices).T
        mcov = self.sigma.take(indices, axis = 0).take(indices, axis = 1)
        return MultivariateGaussianDistribution(mmu, mcov)

    def conditional(self, indices2, values2, indices1, xadd = op.add):
        # Mean of indices1, conditioned on indices2 = values2
        mu1 = self.mu.take(indices1).T
        mu2 = self.mu.take(indices2).T
        sigma11 = self.sigma.take(indices1, axis = 0).take(indices1, axis = 1)
        sigma12 = self.sigma.take(indices1, axis = 0).take(indices2, axis = 1)
        sigma21 = self.sigma.take(indices2, axis = 0).take(indices1, axis = 1)
        sigma22 = self.sigma.take(indices2, axis = 0).take(indices2, axis = 1)
        sigma22I = sigma22.I
        mu1g2 = xadd(mu1, sigma12 * sigma22I * xadd(values2, -mu2))
        sigma1g2 = fixSigma(sigma11 - sigma12 * sigma22I * sigma21)
        return MultivariateGaussianDistribution(mu1g2, sigma1g2)

    def difference(self, indices1, indices2, xadd = op.add):
        # dist of indices1 - indices2
        mu1 = self.mu.take(indices1).T
        mu2 = self.mu.take(indices2).T
        sigma11 = self.sigma.take(indices1, axis = 0).take(indices1, axis = 1)
        sigma21 = self.sigma.take(indices2, axis = 0).take(indices1, axis = 1)
        sigma12 = self.sigma.take(indices1, axis = 0).take(indices2, axis = 1)
        sigma22 = self.sigma.take(indices2, axis = 0).take(indices2, axis = 1)

        mudiff = xadd(mu1, -mu2)
        sigmadiff = fixSigma(sigma11 + sigma22 - sigma21 - sigma12)
        return MultivariateGaussianDistribution(mudiff, sigmadiff)

    def corners(self, p, xadd = op.add, noZ = False):
        # Generate points along each major axis
        pts = []
        if noZ:
            smallSigma = self.sigma.take([0,1,3], axis=0).take([0,1,3], axis=1)
            (eigVals, eigVecs) = linalg.eigh(smallSigma)
            for (val, vec) in zip(eigVals, eigVecs.T):
                # Just do it for X, Y, Th;  no variation in Z 
                off3 = math.sqrt(val) * p * vec.T / linalg.norm(vec)
                offset = mat([off3[0,0], off3[1, 0], 0, off3[2, 0]]).T
                pts.append(xadd(self.mu, offset))
                pts.append(xadd(self.mu,  -offset))
        else:
            (eigVals, eigVecs) = linalg.eigh(self.sigma)
            for (val, vec) in zip(eigVals, eigVecs.T):
                offset = math.sqrt(val) * p * vec.T / linalg.norm(vec)
                pts.append(xadd(self.mu, offset))
                pts.append(xadd(self.mu,  -offset))
        return pts

    def mean(self):
        return self.mu

    def meanTuple(self):
        return tuple(self.mu.tolist()[0])

    def mode(self):
        return self.mu

    def variance(self):
        return self.sigma

    def modeTuple(self):
        return tuple(self.mode().tolist()[0])

    def modeVar(self):
        return (self.modeTuple(), self.varTuple())

    # if diagonal
    def varTuple(self):
        return tuple([self.sigma[i,i] for i in range(self.sigma.shape[0])])

    def pnm(self, deltas):
        # Amount of probability mass within delta of mean. Treating
        # the dimensions independently; deltas is a vector; returns
        # a vector of results
        return [gaussPNM(math.sqrt(self.sigma[i,i]), deltas[i]) \
                for i in range(len(deltas))]

    def pn(self, value, deltas):
        # Amount of probability mass within delta of value
        # Value is a column vector of same dim as mu; so is delta
        return [gaussPN(value[i], deltas[i], float(self.mu[i]), 
                        math.sqrt(self.sigma[i,i])) \
                for i in range(len(deltas))]

    # Special hack for when we know this is a vector of poses
    def pnPoses(self, value, deltas):
        # Amount of probability mass within delta of value
        # Value is a column vector of same dim as mu; so is delta
        result = []
        for i in range(len(deltas)):
            if mod(i, 4) == 3:
                result.append(gaussPNAngle(value[i], deltas[i],
                                           float(self.mu[i]),
                                           math.sqrt(float(self.sigma[i,i]))))
            else:
                result.append(gaussPN(value[i], deltas[i],
                                      float(self.mu[i]), 
                                      math.sqrt(float(self.sigma[i,i]))))
        return result

    def draw(self):
        return np.random.multivariate_normal(self.mu.flat, self.sigma)

    def drawEllipse(self):
        if len(self.mu) != 2:
            print 'Can only draw 2D distributions'
        points = self.corners(0.9)
        print points
        (eigVals, eigVecs) = linalg.eig(self.sigma)
        e = Ellipse(xy = self.mu,
                    width = 2*math.sqrt(eigVals[0]),
                    height = 2*math.sqrt(eigVals[1]),
                angle = math.atan2(eigVecs[1,0], eigVecs[0,0])* 180 / math.pi)
        e.set_facecolor('blue')
        fig = figure()
        ax = fig.add_subplot(111, aspect='equal')
        ax.add_artist(e)
        ax.plot([float(p[0]) for p in points[0:2]],
                [float(p[1]) for p in points[0:2]], 'ro')
        ax.plot([float(p[0]) for p in points[2:4]],
                [float(p[1]) for p in points[2:4]], 'go')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        fig.show()

    def kalmanTransUpdate(mvg, u, transSigma):
        mu = mvg.mu + u
        sigma = mvg.sigma + transSigma
        return MultivariateGaussianDistribution(mu, sigma)

    def kalmanObsUpdate(mvg, obs, obsSigma):
        innovation = np.transpose(obs - mvg.mu)
        innovation_covariance = mvg.sigma + obsSigma
        kalman_gain = mvg.sigma * np.linalg.inv(innovation_covariance)
        size = mvg.mu.shape[1]
        mu = np.transpose(np.transpose(mvg.mu) + \
                                 kalman_gain * innovation)
        sigma = (np.eye(size)-kalman_gain)*mvg.sigma
        return MultivariateGaussianDistribution(mu, sigma)

    def __str__(self):
        return 'G('+prettyString(self.mu)+','+prettyString(self.sigma)+')'
    __repr__ = __str__
    def __hash__(self):
        return str(self).__hash__()
    def __eq__(self, other):
        return str(self) == str(other)
    def __ne__(self, other):
        return str(self) != str(other)

# A mixture of Gaussians, with implicit "leftover" probability assigned to
# a uniform
class GMU:
    def __init__(self, components, ranges = None):
        # A list of (mvg, p) pairs;  p's sum to <= 1
        # All same dimensionality (d)
        self.components = components
        # A list of d pairs (lo, hi), expressing the range for the uniform
        self.ranges = ranges
        if ranges:
            self.area = prod((hi - lo) for (lo, hi) in ranges)
        self.uniformWeight = 1 - sum(p for (d,p) in components)
        assert ranges == None or self.uniformWeight < 10e-10
        self.mixtureProbs = DDist(dict([(i, p) for (i, (c, p)) in \
                                        enumerate(self.components)] + \
                                       [('u', self.uniformWeight)]))

    def copy(self):
        return GMU([[d.copy(), p] for (d, p) in self.components], self.ranges)

    def kalmanTransUpdate(self, u, transSigma):
        # Side effects
        for i in range(len(self.components)):
            self.components[i][0] = \
                      self.components[i][0].kalmanTransUpdate(u, transSigma)

    # Wrong because it doesn't adjust mixture weights
    def kalmanObsUpdate(self, obs, obsSigma):
        # Side effects
        for i in range(len(self.components)):
            self.components[i][0] = \
                      self.components[i][0].kalmanObsUpdate(obs, obsSigma)

    # Wrong because it doesn't adjust mixture weights
    def kalmanObsUpdateNoSE(self, obs, obsSigma):
        return GMU([[d.kalmanObsUpdate(obs, obsSigma), p] \
                    for (d, p) in self.components])

    def prob(self, x):
        return sum(d.prob(x) * p for (d, p) in self.components) + \
               ((self.uniformWeight / self.area) if self.ranges else 0)

    def mode(self):
        # Just return the mean of the most likely component...even though
        # that might not really be the mode
        c = self.mld()
        if c:
            return c.mu
        else:
            return None

    # Diagonal variance of the most likely component, as a list
    def varTuple(self):
        return self.mld().varTuple()

    def modeTuple(self):
        return self.mld().modeTuple()

    def modeVar(self):
        return self.mld().modeVar()

    def draw(self):
        c = self.mixtureProbs.draw()
        if c == 'u':
            return tuple([random.randint(l, h) for (l, h) in self.ranges])
        else:
            return self.components[c][0].draw()

    # Returns (dist, p) pair
    def mlc(self):
        # Most likely mixture component;  returns None if uniform
        if len(self.components) > 0:
            return miscUtil.argmax(self.components, lambda (d, p): p)
        else:
            return None

    # Returns dist only
    def mld(self):
        # Most likely mixture component;  returns None if uniform
        if len(self.components) > 0:
            return miscUtil.argmax(self.components, lambda (d, p): p)[0]
        else:
            return None
    def __str__(self):
        return 'GMU('+', '.join([prettyString(c) for c in self.components])+')'
    __repr__ = __str__

def fitGaussianToPoses(data):
    # Data is a matrix of vectors (len mod 4 = 0) representing vec of poses
    # Each column is a random variable
    mu = meanPoses(data)
    return MultivariateGaussianDistribution(mu, covPoses(data, mu))

def meanPoses(data):
    # Go by columns
    mu = []
    for i in range(data.shape[1]):
        if mod(i, 4) == 3:
            mu.append(angleMean(data[:,i]))
        else:
            mu.append(data[:,i].mean())
    return mat(mu).T

def angleMean(data):
    d = data.T.tolist()[0]
    n = len(d)
    return math.atan2(sum([math.sin(x) for x in d])/n,
                      sum([math.cos(x) for x in d])/n)

# Rows of data are examples; mu is a column vector
def covPoses(data, mu):
    n = len(mu)
    sigma = mat(zeros([n, n]))
    for x in data:
        # x is a row;  need to do subtraction respecting angles
        delta = util.tangentSpaceAdd(x.T, -mu)
        sigma += delta * delta.T
    return sigma / n

class ProductDistribution:
    def prob(self, vs):
        return reduce(operator.mul, [d.prob(v) for (v, d) in \
                            zip(vs, self.ds)])

    def mean(self):
        return [d.mean() for d in self.ds]

    def mode(self):
        return [d.mode() for d in self.ds]

    def variance(self):
        return [d.variance() for d in self.ds]

    def update(self, obs, variance):
        for (o, v, d) in zip(obs, variance, self.ds):
            d.update(o, v)

    def move(self, obs, variance):
        for (o, v, d) in zip(obs, variance, self.ds):
            d.move(o, v)

    def reset(self, obs, variance):
        for (o, v, d) in zip(obs, variance, self.ds):
            d.reset(o, v)

    def draw(self):
        return tuple([d.draw() for d in self.ds])

class ProductGaussianDistribution(ProductDistribution):
    """
    Product of independent Gaussians
    """
    def __init__(self, means, stdevs):
        self.ds = [GaussianDistribution(m, s) for (m, s) in zip(means, stdevs)]


class ProductUniformDistribution(ProductDistribution):
    def __init__(self, means, stdevs, n):
        self.ds = [CUniformDist(m - n*s, m+n*s) \
                   for (m, s) in zip(means, stdevs)]

        
class CUniformDist:
    """
    Uniform distribution over a given finite one dimensional range
    """
    def __init__(self, xMin, xMax):
        self.xMin = xMin
        self.xMax = xMax
        self.p = 1.0 / (xMax - xMin)

    def prob(self, v):
        if v >= self.xMin and v <= self.xMax:
            return self.p
        else:
            return 0

    def draw(self):
        return self.xMin + random.random() * (self.xMax - self.xMin) 


    
######################################################################
#   Utilities

# Multinomial distribution.  k items, probability p, independent, that
# each one will flip.  How many flips?

def binomialDist(n, p):
    """
    Binomial distribution on C{n} items with probability C{p}.
    """
    return DDist(dict([(k, binCoeff(n, k) * p**k) for k in range(0, n+1)]))

def binCoeff(n, k):
    """
    n choose k  (the binomial coefficient)
    """
    if k < n/2.0:
        return binCoeff(n, n-k)
    else:
        return reduce(operator.mul, [j for j in range(k+1, n+1)], 1)

def cartesianProduct(domains):
    """
    @param domains: list of lists
    @returns: list of elements in the cartesian product of the domains
    (all ways of selecting one element from each of the lists)
    """
    if len(domains) == 0:
        return ((),)
    else:
        return tuple([(v1,) + rest for v1 in domains[0] \
                    for rest in cartesianProduct(domains[1:])])

def ensureList(x):
    """
    If C{x} is a string, put it in a list, otherwise return the
    argument unchanged
    """
    if isinstance(x, type('')):
        return [x]
    else:
        return x


######################################################################
#   Utilities


def removeElt(items, i):
    """
    non-destructively remove the element at index i from a list;
    returns a copy;  if the result is a list of length 1, just return
    the element  
    """
    result = items[:i] + items[i+1:]
    if len(result) == 1:
        return result[0]
    else:
        return result

def incrDictEntry(d, k, v):
    """
    If dictionary C{d} has key C{k}, then increment C{d[k]} by C{v}.
    Else set C{d[k] = v}.
    
    @param d: dictionary
    @param k: legal dictionary key (doesn't have to be in C{d})
    @param v: numeric value
    """
    if d.has_key(k):
        d[k] += v
    else:
        d[k] = v

############################ Regression

# If we want the 1-pnm(delta) after an observation with obsSigma to be
# < epsr, then what does the 1-pnm have to be before the update?

import scipy.special as ss
import scipy.stats as stats

def regressGaussianPNM(epsr, obsSigma, delta):
    # based on observation
    pnmr = 1 - epsr
    ei2 = ss.erfinv(pnmr)**2
    part2 = (delta**2)/(2 * obsSigma**2)
    if ei2 < part2:
        return .99999999
    if pnmr > .99999999:
        raw_input("Erfinv argument too big")
    return 1 - ss.erf(math.sqrt(ei2 - part2))

def regressGaussianPNMTransition(epsr, transSigma, delta):
    # return epsr * 0.8
    # erfinv(1 - epsr) = delta / sqrt(2 * resultVar)
    # 2 * resultVar * erfinv(1 - epsr)**2 = delta**2
    # resultVar = delta**2 / (2 * erfinv(1 - epsr)**2)
    # prevVar + transVar = resultVar
    # So, if resultVar < transVar this is impossible
    
    denom = (2 * ss.erfinv(1-epsr)**2)
    if denom <= 0:
        print "Error in erf calculation, epsr=", epsr
        return 1.0
    
    resultVar = (delta**2) / denom
    prevVar =  resultVar - transSigma**2
    # print 'RegressGaussianPNMTransition'
    # print '    epsr', epsr
    # print '    delta', delta
    # print '    transSigma', transSigma
    # print '    ss.erfinv(1-epsr)', ss.erfinv(1-epsr)
    # print '    ((delta**2) / (2 * ss.erfinv(1-epsr)**2))', \
    #       ((delta**2) / (2 * ss.erfinv(1-epsr)**2))
    if prevVar <= 0:
    #    print '   failed'
        return None
    # print '    delta / math.sqrt(2*prevVar)', delta / math.sqrt(2*prevVar)
    # print '    result', 1 - ss.erf(delta / math.sqrt(2*prevVar))
    return 1 - ss.erf(delta / math.sqrt(2*prevVar))

# Amount of probability mass within delta of mean, given
def gaussPNM(sigma, delta):
    return ss.erf(delta / (math.sqrt(2) * sigma))

# Amount of probability mass within delta of value, given a Gaussian
def gaussPN(value, delta, mu, sigma):
    rv = stats.norm(mu, sigma)
    return rv.cdf(value + delta) - rv.cdf(value - delta)

def gaussPNAngle(value, delta, mu, sigma):
    limit1 = util.fixAnglePlusMinusPi(value - mu - delta)
    limit2 = util.fixAnglePlusMinusPi(value - mu + delta)
    upper = max(limit1, limit2)
    lower = min(limit1, limit2)
    rv = stats.norm(0, sigma)
    return rv.cdf(upper) - rv.cdf(lower)

# Gauss CDF
def Phi(x):
    return 0.5 + ss.erf(x / math.sqrt(2.0)) / 2.0

def probModeMoved(delta, var, obsVar):
    p = 1 - ss.erf(delta * (var + obsVar) / (math.sqrt(2.0 * obsVar) * var))
    return p

#chiSq = (0.71, 1.06, 1.65, 2.20, 3.36, 4.88, 5.99, 7.78, 9.49, 13.28, 18.47)
#pValue = (0.95, 0.90, 0.80, 0.70, 0.50, 0.30, 0.20, 0.10, 0.05, 0.01, 0.001)

pValue = (0.995, 0.975, 0.20, 0.10, 0.05, 0.025, 0.02, 0.01, 0.005, 0.002, 0.001)
chiSqTables = {
    1: (0.0000393, 0.000982, 1.642, 2.706, 3.841, 5.024, 5.412, 6.635, 7.879, 9.550, 10.828),
    2: (0.0100, 0.0506, 3.219, 4.605, 5.991, 7.378, 7.824, 9.210, 10.597, 12.429, 13.816),
    3: (0.0717, 0.216, 4.642, 6.251, 7.815, 9.348, 9.837, 11.345, 12.838, 14.796, 16.266),
    4: (0.207, 0.484, 5.989, 7.779, 9.488, 11.143, 11.668, 13.277, 14.860, 16.924, 18.467),
    5: (0.412, 0.831, 7.289, 9.236, 11.070, 12.833, 13.388, 15.086, 16.750, 18.907, 20.515),
    6: (0.676, 1.237, 8.558, 10.645, 12.592, 14.449, 15.033, 16.812, 18.548, 20.791, 22.458)
    }

# Given p value find chiSq

def chiSqFromP(p, nDof):
    chiSq = chiSqTables[nDof]
    for i in range(len(pValue)):
        if p >= pValue[i]:
            if i == 0:
                slope = (chiSq[i+1] - chiSq[i])/(pValue[i+1] - pValue[i])
                return (p - pValue[i])*slope + chiSq[i]
            else:
                slope = (chiSq[i] - chiSq[i-1])/(pValue[i] - pValue[i-1])
                return (p - pValue[i-1])*slope + chiSq[i-1]
    slope = (chiSq[-2] - chiSq[-1])/(pValue[-2] - pValue[-1])
    return (p - pValue[-1])*slope + chiSq[-1]

def tangentSpaceAdd(a, b):
    res = a + b
    for i in range(3, len(res), 4):
        res[i, 0] = fixAnglePlusMinusPi(res[i, 0])
    return res

def fixAnglePlusMinusPi(a):
    """
    A is an angle in radians;  return an equivalent angle between plus
    and minus pi
    """
    pi2 = 2.0* math.pi
    i = 0
    while abs(a) > math.pi:
        if a > math.pi:
            a = a - pi2
        elif a < -math.pi:
            a = a + pi2
        i += 1
        if i > 10: break                # loop found
    return a

def clip(v, vMin, vMax):
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

def gaussian(x, mu, sigma):
    return math.exp(-((x-mu)**2 / (2*sigma**2))) /(sigma*math.sqrt(2*math.pi))




### All much too specific to 2D.  Fix.

def confDist(c1, c2):
    return math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)

def gauss(mean, var):
    if not type(mean) == np.ndarray:
        mean = np.array([mean[0], mean[1]])
    if not type(var) == np.ndarray:
        var = np.array([[var[0], 0.0],[0.0,var[1]]])
    return MultivariateGaussianDistribution(mean, var)

def composeVariance(var1, var2):
    return var1 + var2

def invComposeVariance(addedVar, resultVar):
    # Assumes np 
    return resultVar - addedVar

def invComposeVarianceLists(addedVar, resultVar):
    # Assumes np 
    return makeDiag(resultVar) - makeDiag(addedVar)

def moveVariance(conf1, conf2):
    dist = confDist(conf1, conf2)
    var = math.ceil(dist)*0.001
    moveVar = makeDiag((var, var))
    return moveVar

# Assume H  is identity (transforms state into obs)
def varBeforeObs(obsVar, varAfterObs):
    # S = VB + VO
    # K = VB * S^-1
    # VA = (I - K) VB
    # VA = (I - VB (VB + VO)^-1) * VB

    # VA = VB - VB (VB + VO)^{-1} VB

    # VA*VB^{-1} = I - VB (VB + VO)^{-1}
    # VB^{-1}*VA*VB^{-1} = VB^{-1} - (VB + VO)^{-1}
    # (VB + VO)^{-1} = VB^{-1}*VA*VB^{-1} - VB^{-1} 
    # (VB + VO)^{-1} = VB^{-1}*(VA*VB^{-1} - I)
    # (VB + VO) = (VA*VB^{-1} - I)^{-1} * VB

    # Urgh.  Hard to invert in general.
    # Assume it's diagonal, and therefore separable
    # So, for a single entry, we have
    # VA = 1 / (1/VO + 1 / VB)
    # VB = VA VO / (VO - VA)

    # ov = undiag(obsVar)
    result = [(x * y / (x - y) if x > y else 1.0)
              for (x, y) in zip(obsVar, varAfterObs)]
    return tuple(result)






