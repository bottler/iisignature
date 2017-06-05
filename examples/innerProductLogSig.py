#This file provides a function getMatrices which returns the dot products of logsignature
#basis elements as a matrix for each level.
#This allows you to find the L2 dot product of log signatures in tensor space without
#ever expressing them in tensor space.
#This might be useful if you have to calculate such a dot product very many times.
#The demo() function provides an example.

#Don't use for dimension bigger than 9, because it uses the basis() function

import numpy as np, numpy, sys, os, scipy.linalg
from six import print_
 
#add the parent directory, so we find our iisignature build if it was built --inplace
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import iisignature

#the following 3 functions for interpreting basis strings
#are copied from the tests

#text, index -> (either number or [res, res]), newIndex
def parseBracketedExpression(text,index):
    if(text[index]=='['):
        left, m = parseBracketedExpression(text,index+1)
        right, n = parseBracketedExpression(text,m+1)
        return [left,right],n+1
    else:
        n = 0
        while(n<len(text) and text[index+n] in ['1','2','3','4','5','6','7','8','9']): #this should always just happen once if input is a bracketed expression of letters
            n = n + 1
        return int(text[index:index+n]), index+n

#print (parseBracketedExpression("[23,[2,[[22,1],2]]]",0))

#bracketed expression, dim -> numpy array of its value, depth
def multiplyOut(expn, dim):
    if isinstance(expn,list):
        left, leftDepth  = multiplyOut(expn[0],dim)
        right,rightDepth = multiplyOut(expn[1],dim)
        a = numpy.outer(left,right).flatten()
        b = numpy.outer(right,left).flatten()
        return a-b, leftDepth+rightDepth
    else:
        a = numpy.zeros(dim)
        a[expn-1]=1
        return a,1

#string of bracketed expression, dim -> numpy array of its value, depth
#for example:
#valueOfBracket("[1,2]",2) is ([0,1,-1,0],2)
def valueOfBracket(text,dim):
    return multiplyOut(parseBracketedExpression(text,0)[0],dim)

def getMatrices(s):
    info=iisignature.info(s)
    m=info["level"]
    d=info["dimension"]
    values=[[] for i in range(m)]
    for exp in iisignature.basis(s):
        vec,level = valueOfBracket(exp,d)
        values[level-1].append(vec)
    out=[]
    for v in values:
        def f(x,y): return numpy.inner(v[x],v[y])
        out.append(numpy.fromfunction(
            numpy.vectorize(f),(len(v),len(v)),dtype=int))
#        store=[]
#        for x1 in v:
#            store.append([])
#            for x2 in v:
#                store[-1].append(numpy.inner(x1,x2))
#        out.append(numpy.array(store))
    return out

def demo():
    #1. generate some dummy data
    d=3
    m=6
    path1=np.random.uniform(size=(20,d))
    path2=np.random.uniform(size=(20,d))
    paths = (path1,path2)
    s=iisignature.prepare(d,m)

    #2. print the dot product of the log signatures in tensor space
    #(The "x" means that the log signature is returned in tensor space.)
    expandedLogSig1,expandedLogSig2=(iisignature.logsig(i,s,"x") for i in paths)
    target = np.inner(expandedLogSig1,expandedLogSig2)
    print_ ("Target:", float(target))

    #3. use getMatrices to act on the log signatures expressed in a basis
    #    and get the same answer.
    matrices=getMatrices(s)   
    logsig1,logsig2=(iisignature.logsig(i,s) for i in paths)
    adjustment=scipy.linalg.block_diag(*matrices)
    #print(np.dot(adjustment,logsig2).shape)
    print_ ("We get:", float(np.inner(logsig1,np.dot(adjustment,logsig2))))
    
if __name__=="__main__":
    demo()
