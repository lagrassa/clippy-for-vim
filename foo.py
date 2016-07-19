import operator

# So we can take a product of a list of numbers
def prod(iterable):
    return reduce(operator.mul, iterable, 1)

# Given D rolls a d, and the opponent has offset a, what's the
# probability that D will win?
def f(d, a):
    return 0 if d <= a else (d - a - 1) / 100.0

# Given a list of offsets of the opponents D is facing, what is the
# probability he will win?
def dwin(alphas):
    return sum([.01 * prod([f(d, a) for a in alphas]) for d in range(1, 101)])

    
