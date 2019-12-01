if __name__ != "__main__":
    import iisignature

import unittest
import numpy
import sys
import math
import time
numpy.set_printoptions(suppress=True,linewidth=150)
output_timings = False

#text, index -> (either number or [res, res]), newIndex
def parseBracketedExpression(text,index):
    if(text[index] == '['):
        left, m = parseBracketedExpression(text,index + 1)
        right, n = parseBracketedExpression(text,m + 1)
        return [left,right],n + 1
    else:
        n = 0
        while(n < len(text) and text[index + n] in ['1','2','3','4','5','6','7','8','9']): #this should always just happen once if input is a bracketed expression of
                                                                                           #letters
            n = n + 1
        return int(text[index:index + n]), index + n

#print (parseBracketedExpression("[23,[2,[[22,1],2]]]",0))

#bracketed expression, dim -> numpy array of its value, depth
def multiplyOut(expn, dim):
    if isinstance(expn,list):
        left, leftDepth = multiplyOut(expn[0],dim)
        right,rightDepth = multiplyOut(expn[1],dim)
        a = numpy.outer(left,right).flatten()
        b = numpy.outer(right,left).flatten()
        return a - b, leftDepth + rightDepth
    else:
        a = numpy.zeros(dim)
        a[expn - 1] = 1
        return a,1

#string of bracketed expression, dim -> numpy array of its value, depth
#for example:
#valueOfBracket("[1,2]",2) is ([0,1,-1,0],2)
def valueOfBracket(text,dim):
    return multiplyOut(parseBracketedExpression(text,0)[0],dim)


#inputs are values in the tensor algebra given as lists of levels (from 1 to
#level), assumed 0 in level 0.
#returns their concatenation product
def multiplyTensor(a,b):
    level = len(a)
    dim = len(a[0])
    sum = [numpy.zeros(dim ** m) for m in range(1,level + 1)]
    for leftLevel in range(1,level):
        for rightLevel in range(1, 1 + level - leftLevel):
            sum[leftLevel + rightLevel - 1]+=numpy.outer(a[leftLevel - 1],b[rightLevel - 1]).flatten()
    return sum

#input is a value in the tensor algebra given as lists of levels (from 1 to
#level), assumed 0 in level 0.
#returns its exp - assumed 1 in level 0
#exp(x)-1 = x+x^2/2 +x^3/6 +x^4/24 + ...
def exponentiateTensor(a):
    out = [i.copy() for i in a]
    level = len(a)
    products = [out]
    for m in range(2,level + 1):
        t = multiplyTensor(a,products[-1])
        for j in t:
            j *= (1.0 / m)
        products.append(t)
    return [numpy.sum([p[i] for p in products],0) for i in range(level)]

#input is a value in the tensor algebra given as lists of levels (from 1 to
#level), assumed 1 in level 0.
#returns its log - assumed 0 in level 0
#log(1+x) = x -x^2/2 +x^3/3 -x^4/4 + ...
def logTensor(a):
    out = [i.copy() for i in a]
    level = len(a)
    products = [out]
    for m in range(2,level + 1):
        t = multiplyTensor(a,products[-1])
        products.append(t)
    neg = True
    for m in range(2,level + 1):
        for j in products[m - 1]:
            if neg:
                j *= (-1.0 / m)
            else:
                j *= (1.0 / m)
        neg = not neg
    return [numpy.sum([p[i] for p in products],0) for i in range(level)]

#given a tensor as a concatenated 1D array, return it as a list of levels
def splitConcatenatedTensor(a, dim, level):
    start = 0
    out = []
    for m in range(1,level + 1):
        levelLength = dim ** m
        out.append(a[start:(start + levelLength)])
        start = start + levelLength
    assert(start == a.shape[0])
    return out

#returns the signature of a straight line as a list of levels, and the number
#of multiplications used
def sigOfSegment(displacement, level):
    d = displacement.shape[0]
    sig = [displacement]
    mults = 0
    for m in range(2,level + 1):
        other = sig[-1]
        mults += 2 * other.shape[0] * d
        sig.append(numpy.outer(other,displacement).flatten() * (1.0 / m))
    return sig, mults
    
#inputs are values in the tensor algebra given as lists of levels (from 1 to
#level), assumed 1 in level 0.
#returns their concatenation product, and also the number of multiplications
#used
#c.f.  multiplyTensor
def chen(a,b):
    level = len(a)
    dim = len(a[0])
    mults = 0
    sum = [a[m] + b[m] for m in range(level)]
    for leftLevel in range(1,level):
        for rightLevel in range(1, 1 + level - leftLevel):
            sum[leftLevel + rightLevel - 1]+=numpy.outer(a[leftLevel - 1],b[rightLevel - 1]).flatten()
            mults += a[leftLevel - 1].shape[0] * b[rightLevel - 1].shape[0]
    return sum, mults

def diff(a,b):
    return numpy.max(numpy.abs(a - b))

#Finite difference derivative
#estimate bump dot (the derivative of np.sum(f(X)) wrt X at X=x)
# i.e. the change in np.sum(f(X)) caused by bump 
# with a finite difference approximation
def fdDeriv(f,x,bump,order,nosum=False):
    if order==0:#SIMPLE
        o=f(x+bump)-f(x)
    elif order ==2:#2nd order central
        o=0.5 * (f(x+bump)-f(x-bump))
    elif order ==4:#4th order central
        o=(8*f(x+bump)-8*f(x-bump)+f(x-2*bump)-f(x+2*bump))/12
    elif order ==6:#6th order central
        o=(45*f(x+bump)-45*f(x-bump)+9*f(x-2*bump)-9*f(x+2*bump)+f(x+3*bump)-f(x-3*bump))/60
    if nosum:
        return o
    return numpy.sum(o)

#fn takes an array to a scalar. This manually calculates its derivatives at init.
def allSensitivities(init, fn):
    z=numpy.zeros_like(init, dtype="float64")
    #y=fn(init)
    bump=0.01
    o=numpy.zeros_like(init, dtype="float64")
    for i in range(len(z)):
        z[i]=bump
        #o[i]=(fn(init+z)-y)/bump
        #o[i]=(fn(init+z)-fn(init-z))/(2*bump)
        o[i]=fdDeriv(fn,init,z,6)/bump
        z[i]=0
    return o

class TestCase(unittest.TestCase):
    if sys.hexversion < 0x2070000:
        def assertLess(self,a,b,msg=None):
            self.assertTrue(a < b,msg)

    if output_timings:
        @classmethod
        def setUpClass(cls):
            cls.startTime = time.time()

        @classmethod
        def tearDownClass(cls):
            print ("\n%s.%s: %.3f" % (cls.__module__, cls.__name__, time.time() - cls.startTime))

if "stack" in dir(numpy):
    stack = numpy.stack
else:#Old numpy may not have stack, which we only need with axis=0
    def stack(arr):
        return numpy.vstack([i[numpy.newaxis] for i in arr])

#numpy's lstsq has a different default in different versions
#I think this function agrees with newer versions (1.14+) 
#and with scipy.
#This doesn't really affect the tests anyway.
def lstsq(a,b):
    rcond_to_use=max(a.shape)*numpy.finfo(float).eps
    return numpy.linalg.lstsq(a,b,rcond=rcond_to_use)

#This test checks that basis, logsig and sig are compatible with each other by
#calculating a signature both using sig
#and using logsig and checking they are equal
class A(TestCase):
    def consistency(self, coropa, dim, level):
        #numpy.random.seed(21)
        s = iisignature.prepare(dim,level,"coshx" if coropa else "cosx")
        myinfo = {"level":level, "dimension":dim,
                  "methods": ("COSAX" if level <= 2 else "COSX"),
                  "basis":("Standard Hall" if coropa else "Lyndon"),
                  "logsigtosig_supported" : False}
        self.assertEqual(iisignature.info(s),myinfo)
        path = numpy.random.uniform(size=(10,dim))
        basis = iisignature.basis(s)
        logsig = iisignature.logsig(path,s)
        sig = iisignature.sig(path,level)

        #check lengths
        self.assertEqual(len(basis),iisignature.logsiglength(dim,level))
        self.assertEqual((len(basis),),logsig.shape)
        self.assertEqual(sig.shape,(iisignature.siglength(dim,level),))

        #calculate a signature from logsig
        expanded_logsig = [numpy.zeros(dim ** m) for m in range(1,level + 1)]
        for coeff, expression in zip(logsig,basis):
            values, depth = valueOfBracket(expression,dim)
            expanded_logsig[depth - 1]+=values * coeff
        calculated_sig = numpy.concatenate(exponentiateTensor(expanded_logsig))
        self.assertLess(diff(sig,calculated_sig),0.00001)

        #calculate a log signature from sig
        fullLogSig = numpy.concatenate(logTensor(splitConcatenatedTensor(sig,dim,level)))
        fullLogSigLib = iisignature.logsig(path,s,"x")
        diff1 = numpy.max(numpy.abs(fullLogSigLib - fullLogSig))
        #print
        #(numpy.vstack([fullLogSig,fullLogSigLib,numpy.abs(fullLogSigLib-fullLogSig)]).transpose())
        self.assertLess(diff1,0.00001)

        basisMatrix = []
        zeros = [numpy.zeros(dim ** m) for m in range(1,level + 1)]
        for expression in basis:
            values, depth = valueOfBracket(expression, dim)
            temp = zeros[depth - 1]
            zeros[depth - 1] = values
            basisMatrix.append(numpy.concatenate(zeros))
            zeros[depth - 1] = temp
        calculatedLogSig = lstsq(numpy.transpose(basisMatrix),fullLogSig)[0]
        diff2 = numpy.max(numpy.abs(logsig - calculatedLogSig))
        self.assertLess(diff2,0.00001)

        #check consistency of methods
        slowLogSig = iisignature.logsig(path,s,"o")
        diffs = numpy.max(numpy.abs(slowLogSig - calculatedLogSig))
        self.assertLess(diffs,0.00001)

        sigLogSig = iisignature.logsig(path,s,"s")
        diffs = numpy.max(numpy.abs(sigLogSig - calculatedLogSig))
        self.assertLess(diffs,0.00001)

        if level < 3:
            areaLogSig = iisignature.logsig(path,s,"a")
            diffs = numpy.max(numpy.abs(areaLogSig - calculatedLogSig))
            self.assertLess(diffs,0.00001)

    def testConsistency(self):
        self.consistency(False, 3, 6)
        self.consistency(False, 3, 2)

    def testCoropa(self):
        self.consistency(True, 5, 2)

    #test sigjoin is compatible with sig and also its deriv
    def testjoining(self):
        numberToDo = 1
        dim = 2
        level = 2
        siglength = iisignature.siglength(dim,level)
        for fixedPoint, inputDim, fixed in [(float('nan'),dim,False),(0.1,dim - 1,True)]:
            pathLength = 10
            def makePath():
                p = numpy.random.uniform(size=(pathLength,dim))
                if fixed:
                    p[:,-1] = fixedPoint * numpy.arange(pathLength)
                return p
            paths = [makePath() for i in range(numberToDo)]
            sig = numpy.vstack([iisignature.sig(path,level) for path in paths])

            joinee = numpy.zeros((numberToDo,siglength))
            for i in range(1,pathLength):
                displacements = [path[i:(i + 1),:] - path[(i - 1):i,:] for path in paths]
                displacement = numpy.vstack(displacements)
                if fixed:
                    displacement = displacement[:,:-1]
                joinee = iisignature.sigjoin(joinee,displacement,level,fixedPoint)
            self.assertLess(diff(sig,joinee),0.0001,"fullSig matches sig" + (" with fixed Dim" if fixed else ""))

            extra = numpy.random.uniform(size=(numberToDo,inputDim))
            bumpedExtra = 1.001 * extra
            bumpedJoinee = 1.001 * joinee
            base = numpy.sum(iisignature.sigjoin(joinee,extra,level,fixedPoint))
            bump1 = numpy.sum(iisignature.sigjoin(bumpedJoinee,extra,level,fixedPoint))
            bump2 = numpy.sum(iisignature.sigjoin(joinee,bumpedExtra,level,fixedPoint))
            derivsOfSum = numpy.ones((numberToDo,siglength))
            calculated = iisignature.sigjoinbackprop(derivsOfSum,joinee,extra,
                                                     level,fixedPoint)
            self.assertEqual(len(calculated),3 if fixed else 2)
            diff1 = (bump1 - base) - numpy.sum(calculated[0] * (bumpedJoinee - joinee))
            diff2 = (bump2 - base) - numpy.sum(calculated[1] * (bumpedExtra - extra))
            #print ("\n",bump1,bump2,base,diff1,diff2)
            self.assertLess(numpy.abs(diff1),0.000001,"diff1 as expected " + (" with fixed Dim" if fixed else ""))
            self.assertLess(numpy.abs(diff2),0.00001,"diff2 as expected " + (" with fixed Dim" if fixed else ""))
            if fixed:
                bumpedFixedPoint = fixedPoint * 1.01
                bump3 = numpy.sum(iisignature.sigjoin(joinee,extra, level, bumpedFixedPoint))
                diff3 = (bump3-base - numpy.sum(calculated[2] * (bumpedFixedPoint-fixedPoint)))
                #print("\n",bump3,base, fixedPoint, bumpedFixedPoint, calculated[2])
                self.assertLess(numpy.abs(diff3),0.00001, "diff3")

    #test sigcombine is compatible with sig and also its deriv
    def testcombining(self):
        dim = 2
        level = 2
        siglength = iisignature.siglength(dim,level)
        pathLength = 20
        halfPathLength=10
        numberToDo=4
        path = numpy.random.uniform(size=(numberToDo,pathLength,dim))
        sig = iisignature.sig(path,level)
        sig1 = iisignature.sig(path[:,:halfPathLength],level)
        sig2 = iisignature.sig(path[:,(halfPathLength-1):],level)
        combined = iisignature.sigcombine(sig1,sig2,dim,level)
        self.assertLess(diff(sig,combined),0.0001)

        extra = numpy.random.uniform(size=(siglength,))
        bumpedsig1 = 1.001 * sig1
        bumpedsig2 = 1.001 * sig2
        base = numpy.sum(iisignature.sigcombine(sig1,sig2,dim,level))
        bump1 = numpy.sum(iisignature.sigcombine(bumpedsig1,sig2,dim,level))
        bump2 = numpy.sum(iisignature.sigcombine(sig1,bumpedsig2,dim,level))
        derivsOfSum = numpy.ones((numberToDo,siglength))
        calculated = iisignature.sigcombinebackprop(derivsOfSum,sig1,sig2,dim,level)
        self.assertEqual(len(calculated), 2)
        diff1 = (bump1 - base) - numpy.sum(calculated[0] * (bumpedsig1 - sig1))
        diff2 = (bump2 - base) - numpy.sum(calculated[1] * (bumpedsig2 - sig2))
        #print ("\n",bump1,bump2,base,diff1,diff2)
        self.assertLess(numpy.abs(diff1),0.000001)
        self.assertLess(numpy.abs(diff2),0.00001)


class LogSig2Sig(TestCase):
    def doL2S(self, coropa):
        numpy.random.seed(212)
        d=3
        m=6
        path = numpy.random.uniform(size=(12,d))
        sig = iisignature.sig(path,m)
        s = iisignature.prepare(d, m, "S2H" if coropa else "S2")
        self.assertTrue(iisignature.info(s)["logsigtosig_supported"])
        logsig = iisignature.logsig(path,s)
        sig_ = iisignature.logsigtosig(logsig,s)
        self.assertEqual(sig.shape,sig_.shape)
        self.assertTrue(numpy.allclose(sig,sig_))

        #Like the other iisig functions, we check that derivatives
        #of logsigtosig allow sum(logsigtosig) to be backproped correctly
        #This is a boring thing to calculate, because after the first level
        #each of the log signature elements is a lie bracket and so
        #contributes a net total of 0 to the signature
        derivsOfSum = numpy.ones((sig.shape[0],),dtype="float64")
        bumpedLogSig = 1.01*logsig
        calculated  = iisignature.logsigtosigbackprop(derivsOfSum, logsig, s)
        #wantedbackprop = allSensitivities(logsig, lambda l: iisignature.logsigtosig(l,s).sum())
        manualChange = fdDeriv(lambda x:iisignature.logsigtosig(x,s),
                                logsig, bumpedLogSig-logsig,6)
        calculatedChange = numpy.sum((bumpedLogSig-logsig)*calculated)
        self.assertLess(numpy.abs(manualChange-calculatedChange),0.00001)
        #beyond the first level, all zero
        if m>1:
            self.assertLess(numpy.max(numpy.abs(calculated[d:])),0.00001)
        self.assertEqual(calculated.shape, logsig.shape)

        #Now for a better test, we backprop sum(random*logsigtosig)
        #specifically calculate the change in it caused by bump two ways
        random = numpy.random.uniform(size=sig.shape[0],)
        derivsOfSum = random
        calculated  = iisignature.logsigtosigbackprop(derivsOfSum, logsig, s)
        manualChange = fdDeriv(lambda x: iisignature.logsigtosig(x,s)*random,
                                logsig, bumpedLogSig-logsig,4)
        calculatedChange = numpy.sum((bumpedLogSig-logsig)*calculated)
        self.assertLess(numpy.abs(manualChange-calculatedChange),0.00001)
        self.assertEqual(calculated.shape, logsig.shape)

    def testLyndon(self):
        self.doL2S(False)

    def testCoropa(self):
        self.doL2S(True)


class Deriv(TestCase):
    def testSig(self):
        #test that sigjacobian and sigbackprop compatible with sig
        numpy.random.seed(291)
        d = 3
        m = 5
        pathLength = 10
        path = numpy.random.uniform(size=(pathLength,d))
        path = numpy.cumsum(2 * (path - 0.5),0)#makes it more random-walk-ish, less like a scribble
        increment = 0.01 * numpy.random.uniform(size=(pathLength,d))
        base_sig = iisignature.sig(path,m)

        target = fdDeriv(lambda x:iisignature.sig(x,m),path,increment,2, nosum=True)
        
        gradient = iisignature.sigjacobian(path,m)
        calculated = numpy.tensordot(increment,gradient)

        diffs = numpy.max(numpy.abs(calculated - target))
        niceOnes = numpy.abs(calculated) > 1.e-4
        niceOnes2 = numpy.abs(calculated) < numpy.abs(base_sig)
        diffs1 = numpy.max(numpy.abs((calculated[niceOnes2] - target[niceOnes2]) / base_sig[niceOnes2]))
        diffs2 = numpy.max(numpy.abs(calculated[1 - niceOnes2] - target[1 - niceOnes2]))
        ratioDiffs = numpy.max(numpy.abs(calculated[niceOnes] / target[niceOnes] - 1))

        #numpy.set_printoptions(suppress=True,linewidth=os.popen('stty size',
        #'r').read().split()[1] #LINUX
        #numpy.set_printoptions(suppress=True,linewidth=150)
        #print ("")
        #print (path)
        #print
        #(numpy.vstack([range(len(base_sig)),base_sig,calculated,target,(calculated-target)/base_sig,calculated/target-1]).transpose())
        #print (diffs, ratioDiffs, diffs1, diffs2)
        #print(numpy.argmax(numpy.abs(calculated[niceOnes]/target[niceOnes]-1)),numpy.argmax(numpy.abs((calculated-target)/base_sig)))

        self.assertLess(diffs,0.00001)
        self.assertLess(ratioDiffs,0.01)
        self.assertLess(diffs1,0.001) 
        self.assertLess(diffs2,0.00001) 

        #compatibility between sigbackprop and sigjacobian is strong
        dFdSig = numpy.random.uniform(size=(iisignature.siglength(d,m),))
        backProp = iisignature.sigbackprop(dFdSig,path,m)
        manualCalcBackProp = numpy.dot(gradient,dFdSig)
        backDiffs = numpy.max(numpy.abs(backProp - manualCalcBackProp))
        if 0: # to investigate the compile logic problem I used this and
              # (d,m,pathLength)=(1,2,2)
            print("")
            print(dFdSig)
            print(path)
            print(backProp)
            print(manualCalcBackProp)
        self.assertLess(backDiffs,0.000001)

    def logSig(self, type, m=7):
        numpy.random.seed(291)
        d=2
        pathLength=10
        s=iisignature.prepare(d,m,type)
        path = numpy.random.uniform(size=(pathLength,d))
        path = numpy.cumsum(2 * (path - 0.5),0)#makes it more random-walk-ish, less like a scribble
        increment = 0.01*path
        increment = 0.1*numpy.random.uniform(size=(pathLength,d))

        manualChange = fdDeriv(lambda x:iisignature.logsig(x,s,type),path,increment,4)
        
        dFdlogSig = numpy.ones(iisignature.siglength(d,m) if "X"==type else iisignature.logsiglength(d,m))
        calculatedChange = numpy.sum(increment*iisignature.logsigbackprop(dFdlogSig,path,s,type))
        #print(manualChange, calculatedChange)
        self.assertLess(numpy.abs(manualChange-calculatedChange),0.0001)

    def testLogSig_expanded(self):
        self.logSig("X")
    def testLogSig_lyndon(self):
        self.logSig("S")
    def testLogSig_hall(self):
        self.logSig("H")
    def testLogSig_area(self):
        self.logSig("A",2)

    def test_logsigbackwards_can_augment_s(self):
        numpy.random.seed(291)
        d=2
        m=7
        pathLength=3
        path = numpy.random.uniform(size=(pathLength,d))
        increment = 0.1*numpy.random.uniform(size=(pathLength,d))
        dFdlogSig = numpy.ones(iisignature.logsiglength(d,m))
        for types in (("x","o","s"),("xh","oh","sh")):
            ss=[iisignature.prepare(d,m,t) for t in types]
            backs=[iisignature.logsigbackprop(dFdlogSig,path,s) for s in ss]
            self.assertTrue(numpy.allclose(backs[0],backs[2]),types[0])
            self.assertTrue(numpy.allclose(backs[1],backs[2]),types[1])
            fwds=[iisignature.logsig(path,s,"s") for s in ss]
            self.assertTrue(numpy.allclose(fwds[0],fwds[2]),types[0])
            self.assertTrue(numpy.allclose(fwds[1],fwds[2]),types[1])


class Counts(TestCase):
    #check sigmultcount, and also that sig matches a manual signature
    #calculation
    def testa(self):
        numpy.random.seed(2141)
        d = 5
        m = 5
        pathLength = 4
        displacement = numpy.random.uniform(size=(d))
        sig,mults = sigOfSegment(displacement,m)
        path = [numpy.zeros(d),displacement]
        for x in range(pathLength):
            displacement1 = numpy.random.uniform(size=(d))
            path.append(path[-1] + displacement1)
            sig1, mults1 = sigOfSegment(displacement1, m)
            sig, mults2 = chen(sig,sig1)
            mults+=mults1 + mults2
        #print (path)
        path = numpy.vstack(path)
        isig = iisignature.sig(path,m,1)
        for i in range(m):
            self.assertLess(diff(isig[i],sig[i]),0.00001)
        self.assertEqual(mults,iisignature.sigmultcount(path,m))

class SimpleCases(TestCase):
    def testFewPoints(self):
        # check sanity of paths with less than 3 points
        path1=[[4.3,0.8]]
        path2=numpy.array([[1,2],[2,4]])
        m=4
        d=2
        s=iisignature.prepare(d,m,"cosx")
        s_a=iisignature.prepare(d,2,"cosx")
        length=iisignature.siglength(d,m)
        loglength=iisignature.logsiglength(d,m)
        loglength_a=iisignature.logsiglength(d,2)
        blankLogSig=numpy.zeros(loglength)
        blankLogSig_a=numpy.zeros(loglength_a)
        blankSig=numpy.zeros(length)
        self.assertLess(diff(iisignature.sig(path1,m),blankSig),0.000000001)
        self.assertTrue(numpy.array_equal(iisignature.sig(path1,m,2),numpy.zeros([0,length])))
        self.assertLess(diff(iisignature.logsig(path1,s,"C"),blankLogSig),0.000000001)
        self.assertLess(diff(iisignature.logsig(path1,s,"O"),blankLogSig),0.000000001)
        self.assertLess(diff(iisignature.logsig(path1,s,"S"),blankLogSig),0.000000001)
        self.assertLess(diff(iisignature.logsig(path1,s,"X"),blankSig),0.000000001)
        self.assertLess(diff(iisignature.logsig(path1,s_a,"A"),blankLogSig_a),0.000000001)
        blankLogSig[:d]=path2[1]-path2[0]
        blankLogSig_a[:d]=path2[1]-path2[0]
        blankSig[:d]=path2[1]-path2[0]
        self.assertLess(diff(iisignature.logsig(path2,s,"C"),blankLogSig),0.000001)
        self.assertLess(diff(iisignature.logsig(path2,s,"O"),blankLogSig),0.000001)
        self.assertLess(diff(iisignature.logsig(path2,s,"S"),blankLogSig),0.000001)
        self.assertLess(diff(iisignature.logsig(path2,s,"X"),blankSig),0.000001)
        self.assertLess(diff(iisignature.logsig(path2,s_a,"A"),blankLogSig_a),0.000001)

    def testLevel1(self):
        m=1
        d=2
        path=numpy.random.uniform(size=(10,d))
        rightSig = path[-1,:]-path[0,:]
        s=iisignature.prepare(d,m,"cosx2")
        self.assertLess(diff(iisignature.sig(path,m),rightSig),0.0000001)
        for type_ in ("C","O","S","X","A"):
            self.assertLess(diff(iisignature.logsig(path,s,type_),rightSig),0.0000001,type_)
        self.assertLess(diff(rightSig,iisignature.logsigtosig(rightSig,s)),0.000001)
        derivs=numpy.array([2.1,3.2])
        pathderivs=numpy.zeros_like(path)
        pathderivs[-1]=derivs
        pathderivs[0]=-derivs
        self.assertLess(diff(iisignature.logsigbackprop(derivs,path,s),pathderivs),0.00001)
        self.assertLess(diff(iisignature.logsigbackprop(derivs,path,s,"X"),pathderivs),0.00001)
        self.assertLess(diff(iisignature.sigbackprop(derivs,path,m),pathderivs),0.00001)

    def testHighDim(self):
        for m in [1,2]:
            d=1000
            path = numpy.random.rand(10,d)
            s=iisignature.prepare(d,m)
            iisignature.logsig(path,s,"A")
            #not testing result, just that it calculates something
        
    def testCumulative(self):
        m=3
        d=2
        length = 10
        path=numpy.random.uniform(size=(length,d))
        cumul = iisignature.sig(path,m,2)
        expected = numpy.array([iisignature.sig(path[:(i+1)],m) for i in range(1,length)])
        self.assertTrue(numpy.allclose(expected, cumul))
        path=numpy.random.uniform(size=(3,2,length,d))
        cumul = iisignature.sig(path,m,2)
        #expected = numpy.stack([iisignature.sig(path[:,:,:(i+1)],m) for i in range(1,length)],-2)
        expected = numpy.rollaxis(stack([iisignature.sig(path[:,:,:(i+1)],m) for i in range(1,length)]),0,3)
        self.assertTrue(numpy.allclose(expected, cumul))
        
    def testSquareLevel2(self):
        path = numpy.array([[0,0],[1,0],[1,1],[0,1],[0,0]])
        m=2
        sig = iisignature.sig(path, m)
        self.assertTrue(numpy.allclose(sig, [0,0,0,1,-1,0]))

    def testEllLevel2(self):
        path = numpy.array([[0,0],[1,0],[1,1]])
        m=2
        sig = iisignature.sig(path, m)
        self.assertTrue(numpy.allclose(sig, [1,1,0.5,1,0,0.5]))

    def test1d(self):
        path=numpy.array([[0],[1],[3]])
        m=3
        sig=iisignature.sig(path,m)
        self.assertTrue(numpy.allclose(sig, [3,4.5,4.5]))


class Scales(TestCase):
    #check sigscale and its derivatives
    def testa(self):
        numpy.random.seed(775)
        d = 3
        m = 5
        pathLength = 5
        numberToDo = 2
        paths = numpy.random.uniform(size=(numberToDo,pathLength,d))
        sigs = numpy.vstack([iisignature.sig(i,m) for i in paths])
        scales = numpy.random.uniform(0.5,0.97,size=(numberToDo,d))
        scaledPaths = paths * scales[:,numpy.newaxis,:]
        scaledSigs = numpy.vstack([iisignature.sig(i,m) for i in scaledPaths])
        scaledSigsCalc = iisignature.sigscale(sigs,scales,m)
        self.assertEqual(scaledSigs.shape,scaledSigsCalc.shape)
        self.assertLess(diff(scaledSigs,scaledSigsCalc),0.0000001)

        bumpedScales = 1.001 * scales
        bumpedSigs = 1.001 * sigs
        base = numpy.sum(scaledSigsCalc)
        bump1 = numpy.sum(iisignature.sigscale(bumpedSigs,scales,m))
        bump2 = numpy.sum(iisignature.sigscale(sigs,bumpedScales,m))
        derivsOfSum = numpy.ones_like(scaledSigsCalc)
        calculated = iisignature.sigscalebackprop(derivsOfSum,sigs,scales,m)
        diff1 = (bump1 - base) - numpy.sum(calculated[0] * (bumpedSigs - sigs))
        diff2 = (bump2 - base) - numpy.sum(calculated[1] * (bumpedScales - scales))
        #print(calculated[1].shape,bumpedScales.shape,scales.shape)
        #print(calculated[1][0,0],bump2,base,bumpedScales[0,0],scales[0,0])
        #print (bump1,bump2,base,diff1,diff2)
        self.assertLess(numpy.abs(diff1),0.0000001)
        self.assertLess(numpy.abs(diff2),0.0000001)

class Bases(TestCase):
    #Check that the Lyndon basis consists of Lyndon words.
    def testLyndon(self):
        d=2
        m=5
        s=iisignature.prepare(d,m,"O")
        for expression in iisignature.basis(s):
            word = ''.join(c for c in expression if c not in '[,]')
            if len(word) > 1:
                for prefixLength in range(1,len(word)):
                    self.assertLess(word[:prefixLength],word[prefixLength:])

class Batching(TestCase):
    def test_batch(self):
        numpy.random.seed(734)
        d=2
        m=2
        n=15
        paths = [numpy.random.uniform(-1,1,size=(6,d)) for i in range(n)]
        pathArray15=stack(paths)
        pathArray1315=numpy.reshape(pathArray15,(1,3,1,5,6,d))
        sigs = [iisignature.sig(i,m) for i in paths]
        sigArray=stack(sigs)
        sigArray15=iisignature.sig(pathArray15,m)
        sigArray1315=iisignature.sig(pathArray1315,m)
        siglength=iisignature.siglength(d,m)
        self.assertEqual(sigArray1315.shape,(1,3,1,5,siglength))
        self.assertTrue(numpy.allclose(sigArray1315.reshape(n,siglength),sigs))
        self.assertEqual(sigArray15.shape,(15,siglength))
        self.assertTrue(numpy.allclose(sigArray15,sigs))

        backsigs=[iisignature.sigbackprop(i,j,m) for i,j in zip(sigs,paths)]
        backsigArray = stack(backsigs)
        backsigs1315=iisignature.sigbackprop(sigArray1315,pathArray1315,m)
        self.assertEqual(backsigs1315.shape,(1,3,1,5,6,d))
        self.assertTrue(numpy.allclose(backsigs1315.reshape(n,6,2),backsigArray))

        data=[numpy.random.uniform(size=(d,)) for i in range(n)]
        dataArray1315=stack(data).reshape((1,3,1,5,d))
        joined=[iisignature.sigjoin(i,j,m) for i,j in zip(sigs,data)]
        joined1315=iisignature.sigjoin(sigArray1315,dataArray1315,m)
        self.assertEqual(joined1315.shape,(1,3,1,5,siglength))
        self.assertTrue(numpy.allclose(joined1315.reshape(n,-1),stack(joined)))
        backjoined=[iisignature.sigjoinbackprop(i,j,k,m) for i,j,k in zip(joined,sigs,data)]
        backjoinedArrays=[stack([i[j] for i in backjoined]) for j in range(2)]
        backjoined1315=iisignature.sigjoinbackprop(joined1315,sigArray1315,dataArray1315,m)
        self.assertEqual(backjoined1315[0].shape,sigArray1315.shape)
        self.assertEqual(backjoined1315[1].shape,dataArray1315.shape)
        self.assertTrue(numpy.allclose(backjoined1315[0].reshape(n,-1),backjoinedArrays[0]))
        self.assertTrue(numpy.allclose(backjoined1315[1].reshape(n,-1),backjoinedArrays[1]))

        dataAsSigs=[iisignature.sig(numpy.row_stack([numpy.zeros((d,)),i]),m) for i in data]
        dataArray13151=dataArray1315[:,:,:,:,None,:]
        dataArray13151=numpy.repeat(dataArray13151,2,4)*[[0.0],[1.0]]
        dataArrayAsSigs1315=iisignature.sig(dataArray13151,m)
        combined1315=iisignature.sigcombine(sigArray1315,dataArrayAsSigs1315,d,m)
        self.assertEqual(joined1315.shape,combined1315.shape)
        self.assertTrue(numpy.allclose(joined1315,combined1315))
        backcombined1315=iisignature.sigcombinebackprop(joined1315,sigArray1315,dataArrayAsSigs1315,d,m)
        backcombined=[iisignature.sigcombinebackprop(i,j,k,d,m) for i,j,k in zip(joined,sigs,dataAsSigs)]
        backcombinedArrays=[stack([i[j] for i in backcombined]) for j in range(2)]
        self.assertEqual(backcombined1315[0].shape,sigArray1315.shape)
        self.assertEqual(backcombined1315[1].shape,sigArray1315.shape)
        self.assertTrue(numpy.allclose(backjoined1315[0],backcombined1315[0]))
        self.assertTrue(numpy.allclose(backcombined1315[0].reshape(n,-1),backcombinedArrays[0]))
        self.assertTrue(numpy.allclose(backcombined1315[1].reshape(n,-1),backcombinedArrays[1]))
        
        scaled=[iisignature.sigscale(i,j,m) for i,j in zip(sigs,data)]
        scaled1315=iisignature.sigscale(sigArray1315,dataArray1315,m)
        self.assertEqual(scaled1315.shape,(1,3,1,5,siglength))
        self.assertTrue(numpy.allclose(scaled1315.reshape(n,-1),stack(scaled)))
        backscaled=[iisignature.sigscalebackprop(i,j,k,m) for i,j,k in zip(scaled,sigs,data)]
        backscaledArrays=[stack([i[j] for i in backscaled]) for j in range(2)]
        backscaled1315=iisignature.sigscalebackprop(scaled1315,sigArray1315,dataArray1315,m)
        self.assertEqual(backscaled1315[0].shape,sigArray1315.shape)
        self.assertEqual(backscaled1315[1].shape,dataArray1315.shape)
        self.assertTrue(numpy.allclose(backscaled1315[0].reshape(n,-1),backscaledArrays[0]))
        self.assertTrue(numpy.allclose(backscaled1315[1].reshape(n,-1),backscaledArrays[1]))

        s_s=(iisignature.prepare(d,m,"cosax2"),iisignature.prepare(d,m,"cosahx2"))
        for type in ("c","o","s","x","a","ch","oh","sh","ah"):
            s=s_s[1 if "h" in type else 0]
            logsigs = [iisignature.logsig(i,s,type) for i in paths]
            logsigArray=stack(logsigs)
            logsigArray1315=iisignature.logsig(pathArray1315,s,type)
            self.assertEqual(logsigArray1315.shape,(1,3,1,5,logsigs[0].shape[0]),type)
            self.assertTrue(numpy.allclose(logsigArray1315.reshape(n,-1),logsigArray),type)

            if type in ("s","x","sh"):
                backlogs = stack([iisignature.logsigbackprop(i,j,s,type) for i,j in zip(logsigs,paths)])
                backlogs1315 = iisignature.logsigbackprop(logsigArray1315,pathArray1315,s,type)
                self.assertEqual(backlogs1315.shape,backsigs1315.shape)
                self.assertTrue(numpy.allclose(backlogs1315.reshape(n,6,d),backlogs),type)

        for s in s_s:
            logsigs1315 = iisignature.logsig(pathArray1315,s)
            logsigs=[iisignature.logsig(p,s) for p in paths]
            sigsFromLogSigs1315 = iisignature.logsigtosig(logsigs1315,s)
            self.assertEqual(sigsFromLogSigs1315.shape, sigArray1315.shape)
            self.assertTrue(numpy.allclose(sigsFromLogSigs1315, sigArray1315))

            backls2s = stack([iisignature.logsigtosigbackprop(i,j,s) for i,j in zip (sigs,logsigs)])
            backls2s1315 = iisignature.logsigtosigbackprop(sigArray1315,logsigs1315,s)
            self.assertEqual(backls2s1315.shape, logsigs1315.shape)
            self.assertTrue(numpy.allclose(backls2s1315.reshape(-1,logsigs[0].shape[0]),
                                             backls2s))

        a=iisignature.rotinv2dprepare(m,"a")
        rots=stack([iisignature.rotinv2d(i,a) for i in paths])
        rots1315=iisignature.rotinv2d(pathArray1315,a)
        self.assertEqual(rots1315.shape,(1,3,1,5,rots.shape[1]))
        self.assertTrue(numpy.allclose(rots1315.reshape(n,-1),rots))

#sum (2i choose i) for i in 1 to n
# which is the number of linear rotational invariants up to level 2n
def sumCentralBinomialCoefficient(n):
    f = math.factorial
    return sum(f(2 * i) / (f(i) ** 2) for i in range(1,n + 1))

#can run just this with
#python setup.py test -s tests.test_sig.RotInv2d
class RotInv2d(TestCase):
    def dotest(self,type):
        m = 8
        nPaths = 95
        nAngles = 348
        numpy.random.seed(775)
        s = iisignature.rotinv2dprepare(m,type)
        coeffs = iisignature.rotinv2dcoeffs(s)
        angles = numpy.random.uniform(0,math.pi * 2,size=nAngles + 1)
        angles[0] = 0
        rotationMatrices = [numpy.array([[math.cos(i),math.sin(i)],[-math.sin(i),math.cos(i)]]) for i in angles]
        paths = [numpy.random.uniform(-1,1,size=(32,2)) for i in range(nPaths)]
        samePathRotInvs = [iisignature.rotinv2d(numpy.dot(paths[0],mtx),s) for mtx in rotationMatrices]

        #check the length matches
        (length,) = samePathRotInvs[0].shape
        self.assertEqual(length,sum(i.shape[0] for i in coeffs))
        self.assertEqual(length,iisignature.rotinv2dlength(s))
        if type == "a":
            self.assertEqual(length,sumCentralBinomialCoefficient(m // 2))

        self.assertLess(length,nAngles)#sanity check on the test itself

        #check that the invariants are invariant
        if 0:
            print("\n",numpy.column_stack(samePathRotInvs[0:7]))
        for i in range(nAngles):
            if 0 and diff(samePathRotInvs[0],samePathRotInvs[1 + i]) > 0.01:
                print(i)
                print(samePathRotInvs[0] - samePathRotInvs[1 + i])
                print(diff(samePathRotInvs[0],samePathRotInvs[1 + i]))
            self.assertLess(diff(samePathRotInvs[0],samePathRotInvs[1 + i]),0.01)

        #check that the invariants match the coefficients
        if 1:
            sigLevel=iisignature.sig(paths[0],m)[iisignature.siglength(2,m-1):]
            lowerRotinvs = 0 if 2==m else iisignature.rotinv2dlength(iisignature.rotinv2dprepare(m-2,type))
            #print("\n",numpy.dot(coeffs[-1],sigLevel),"\n",samePathRotInvs[0][lowerRotinvs:])
            #print(numpy.dot(coeffs[-1],sigLevel)-samePathRotInvs[0][lowerRotinvs:])
            self.assertTrue(numpy.allclose(numpy.dot(coeffs[-1],sigLevel),samePathRotInvs[0][lowerRotinvs:],atol=0.000001))

        #check that we are not missing invariants
        if type == "a":
            #print("\nrotinvlength=",length,"
            #siglength=",iisignature.siglength(2,m))
            sigOffsets = []
            for path in paths:
                samePathSigs = [iisignature.sig(numpy.dot(path,mtx),m) for mtx in rotationMatrices[1:70]]
                samePathSigsOffsets = [i - samePathSigs[0] for i in samePathSigs[1:]]
                sigOffsets.extend(samePathSigsOffsets)
            #print(numpy.linalg.svd(numpy.row_stack(sigOffsets))[1])
            def split(a, dim, level):
                start = 0
                out = []
                for m in range(1,level + 1):
                    levelLength = dim ** m
                    out.append(a[:,start:(start + levelLength)])
                    start = start + levelLength
                assert(start == a.shape[1])
                return out
            allOffsets = numpy.row_stack(sigOffsets)
            #print (allOffsets.shape)
            splits = split(allOffsets,2,m)
            #print()
            rank_tolerance = 0.01 # this is hackish
            #print
                                             #([numpy.linalg.matrix_rank(i.astype("float64"),rank_tolerance)
                                                                              #for
                                                                              #i in splits])
                                                                              #print ([i.shape for i in splits])
                                                                              #print(numpy.linalg.svd(splits[-1])[1])

            #sanity check on the test
            self.assertLess(splits[-1].shape[1],splits[0].shape[0])
            totalUnspannedDimensions = sum(i.shape[1] - numpy.linalg.matrix_rank(i,rank_tolerance) for i in splits)
            self.assertEqual(totalUnspannedDimensions,length)

        if 0: #This doesn't work - the rank of the whole thing is less than
        #sigLength-totalUnspannedDimensions, which suggests that there are
        #inter-level dependencies,
        #even though the shuffle product dependencies aren't linear. 
        #I don't know why this is.
            sigLength = iisignature.siglength(2,m)
            numNonInvariant = numpy.linalg.matrix_rank(numpy.row_stack(sigOffsets))

            predictedNumberInvariant = sigLength - numNonInvariant
            print(sigLength,length,numNonInvariant)
            self.assertLess(sigLength,nAngles)
            self.assertEqual(predictedNumberInvariant,length)
    def test_a(self):
        self.dotest("a")
    def test_k(self):
        self.dotest("k")
    def test_q(self):
        self.dotest("q")
    def test_s(self):
        self.dotest("s")

    def testConsistencyOfBases(self):
        m = 6
        sa = iisignature.rotinv2dprepare(m,"a")
        sk = iisignature.rotinv2dprepare(m,"k")
        ca = iisignature.rotinv2dcoeffs(sa)[-1]
        ck = iisignature.rotinv2dcoeffs(sk)[-1]
        
        #every row of ck should be in the span of the rows of ca
        #i.e.  every column of ck.T should be in the span of the columns of
        #ca.T
        #i.e.  there's a matrix b s.t.  ca.T b = ck.T
        residuals = lstsq(ca.T,ck.T)[1]
        self.assertLess(numpy.max(numpy.abs(residuals)),0.000001)

        sq = iisignature.rotinv2dprepare(m, "q")
        cq = iisignature.rotinv2dcoeffs(sq)[-1]
        ss = iisignature.rotinv2dprepare(m, "s")
        cs = iisignature.rotinv2dcoeffs(ss)[-1]
        # every row of cs and cq should be in the span of the rows of ca
        residuals2 = lstsq(ca.T, cs.T)[1]
        self.assertLess(numpy.max(numpy.abs(residuals2)), 0.000001)
        residuals2 = lstsq(ca.T, cq.T)[1]
        self.assertLess(numpy.max(numpy.abs(residuals2)), 0.000001)

        self.assertEqual(cq.shape, cs.shape)

        #check that the invariants are linearly independent (not for k)
        for c, name in ((cs, "s"), (ca, "a"), (cq, "q")):
            self.assertEqual(numpy.linalg.matrix_rank(c),c.shape[0],name)

        #check that rows with nonzeros in evil columns are all before
        #rows with nonzeros in odious columns
        #print ((numpy.abs(ca)>0.00000001).astype("int8"))
        for c, name in ((cs, "s"), (ck, "k"), (ca, "a"), (cq, "q")):
            evilRows = []
            odiousRows = []
            for i in range(c.shape[0]):
                evil = 0
                odious = 0
                for j in range(c.shape[1]):
                    if numpy.abs(c[i, j]) > 0.00001:
                        if bin(j).count("1") % 2:
                            odious = odious + 1
                        else:
                            evil = evil + 1
                if evil > 0:
                    evilRows.append(i)
                if odious > 0:
                    odiousRows.append(i)
            #print (evilRows, odiousRows)
            self.assertLess(numpy.max(evil),numpy.min(odious),"bad order of rows in " + name)
        




if __name__ == "__main__":
    sys.path.append("..")
    import iisignature

    #This is convenient for running some tests just by running this file,
    #but only works with python3 (you may need to have built inplace first)
    a = Scales().testa()
