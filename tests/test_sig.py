import iisignature, unittest, numpy

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


#inputs are values in the tensor algebra given as lists of levels (from 1 to level), assumed 0 in level 0.
#returns their concatenation product
def multiplyTensor(a,b):
    level = len(a)
    dim = len(a[0])
    sum = [numpy.zeros(dim**m) for m in range(1,level+1)]
    for leftLevel in range(1,level):
        for rightLevel in range(1, 1+level-leftLevel):
            sum[leftLevel+rightLevel-1]+=numpy.outer(a[leftLevel-1],b[rightLevel-1]).flatten()
    return sum

#input is a value in the tensor algebra given as lists of levels (from 1 to level), assumed 0 in level 0.
#returns its exp - assumed 1 in level 0
def exponentiateTensor(a):
    out = [i.copy() for i in a]
    level = len(a)
    products = [out]
    for i in range(2,level+1):
        m=multiplyTensor(a,products[-1])
        for j in m:
            j *= (1.0/i)
        products.append(m)
        
    return [numpy.sum([p[i] for p in products],0) for i in range(level)]


#This test checks that basis, logsig and sig are compatible with each other by calculating a signature both using sig
#and using logsig and checking they are equal 
class A(unittest.TestCase):
    def testa(self):
        #numpy.random.seed(21)
        dim=3
        level = 4
        s = iisignature.prepare(dim,level)
        path = numpy.random.uniform(size=(10,dim))
        basis = iisignature.basis(s)
        logsig = iisignature.logsig(path,s)
        sig = iisignature.sig(path,level)
        expanded_logsig = [numpy.zeros(dim**m) for m in range(1,level+1)]
        for coeff, expression in zip(logsig,basis):
            values, depth  = valueOfBracket(expression,dim)
            expanded_logsig[depth-1]+=values*coeff
        mysig = numpy.concatenate(exponentiateTensor(expanded_logsig))
        diff = numpy.max(numpy.abs(sig-mysig))
        self.assertTrue(diff<0.001)
