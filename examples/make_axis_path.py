import numpy

#takes a list of nonnegative integers to a path representation
#as an axis path
#e.g. [0,1,0] -> [[0,0],[1,0],[1,1],[2,1]]
def ex(a):
    n=max(a)+1
    out = [[0.0]*n]
    for i in a:
        v = out[-1][:]
        v[i]=v[i]+1
        out.append(v)
    return numpy.array(out)
