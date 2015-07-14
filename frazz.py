def gen(x):
    for i in range(x):
        floo(i)
        yield i

def floo(i):
    if i < 5:
        print i
    else:
        raise Exception, 'Error inside the generator'

def bar(g):
    for thing in g:
        print 'This is a thing in g:', thing

def baz():
    print 'Another layer of function call, just for fun'
    bar(gen(10))

baz()

