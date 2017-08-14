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
    denominator = 1
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


#This test checks that basis, logsig and sig are compatible with each other by
#calculating a signature both using sig
#and using logsig and checking they are equal
class A(TestCase):
    def consistency(self, coropa, dim, level):
        #numpy.random.seed(21)
        s = iisignature.prepare(dim,level,"coshx" if coropa else "cosx")
        myinfo = {"level":level,"dimension":dim,"methods":"COSX",
                  "basis":("Standard Hall" if coropa else "Lyndon")}
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
        calculatedLogSig = numpy.linalg.lstsq(numpy.transpose(basisMatrix),fullLogSig)[0]
        diff2 = numpy.max(numpy.abs(logsig - calculatedLogSig))
        self.assertLess(diff2,0.00001)

        #check consistency of methods
        slowLogSig = iisignature.logsig(path,s,"o")
        diffs = numpy.max(numpy.abs(slowLogSig - calculatedLogSig))
        self.assertLess(diffs,0.00001)

        sigLogSig = iisignature.logsig(path,s,"s")
        diffs = numpy.max(numpy.abs(sigLogSig - calculatedLogSig))
        self.assertLess(diffs,0.00001)

    def testConsistency(self):
        self.consistency(False, 3, 6)

    def testCoropa(self):
        self.consistency(True, 2, 2)

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

#test that sigjacobian and sigbackprop compatible with sig
class Deriv(TestCase):
    def testa(self):
        numpy.random.seed(291)
        d = 3
        m = 5
        pathLength = 10
        path = numpy.random.uniform(size=(pathLength,d))
        path = numpy.cumsum(2 * (path - 0.5),0)#makes it more random-walk-ish, less like a scribble
        increment = 0.01 * numpy.random.uniform(size=(pathLength,d))
        base_sig = iisignature.sig(path,m)

        bumped_sig = iisignature.sig(path + increment,m)
        target = bumped_sig - base_sig
        
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
        #print (diffs, diffs1, diffs2, ratioDiffs,
        #numpy.argmax(numpy.abs(calculated[niceOnes]/target[niceOnes]-1)),numpy.argmax(numpy.abs((calculated-target)/base_sig)))

        #These assertions are pretty weak, the small answers are a bit volatile
        self.assertLess(diffs,0.0001)
        self.assertLess(ratioDiffs,0.2)
        self.assertLess(diffs1,0.05) 
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
        loglength=iisignature.logsiglength(d,m)
        blankLogSig=numpy.zeros(loglength)
        blankSig=numpy.zeros(iisignature.siglength(d,m))
        self.assertLess(diff(iisignature.sig(path1,m),blankSig),0.000000001)
        self.assertLess(diff(iisignature.logsig(path1,s,"C"),blankLogSig),0.000000001)
        self.assertLess(diff(iisignature.logsig(path1,s,"O"),blankLogSig),0.000000001)
        self.assertLess(diff(iisignature.logsig(path1,s,"S"),blankLogSig),0.000000001)
        self.assertLess(diff(iisignature.logsig(path1,s,"X"),blankSig),0.000000001)
        blankLogSig[:d]=path2[1]-path2[0]
        blankSig[:d]=path2[1]-path2[0]
        self.assertLess(diff(iisignature.logsig(path2,s,"C"),blankLogSig),0.000001)
        self.assertLess(diff(iisignature.logsig(path2,s,"O"),blankLogSig),0.000001)
        self.assertLess(diff(iisignature.logsig(path2,s,"S"),blankLogSig),0.000001)
        self.assertLess(diff(iisignature.logsig(path2,s,"X"),blankSig),0.000001)

    def testLevel1(self):
        m=1
        d=2
        path=numpy.random.uniform(size=(10,d))
        rightSig = path[-1,:]-path[0,:]
        s=iisignature.prepare(d,m,"cosx")
        self.assertLess(diff(iisignature.sig(path,m),rightSig),0.0000001)
        for type in ("C","O","S","X"):
            self.assertLess(diff(iisignature.logsig(path,s,type),rightSig),0.0000001,type)

        
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
        pathArray15=numpy.stack(paths)
        pathArray1315=numpy.reshape(pathArray15,(1,3,1,5,6,d))
        sigs = [iisignature.sig(i,m) for i in paths]
        sigArray=numpy.stack(sigs)
        sigArray15=iisignature.sig(pathArray15,m)
        sigArray1315=iisignature.sig(pathArray1315,m)
        siglength=iisignature.siglength(d,m)
        self.assertEqual(sigArray1315.shape,(1,3,1,5,siglength))
        self.assertTrue(numpy.allclose(sigArray1315.reshape(n,siglength),sigs))
        self.assertEqual(sigArray15.shape,(15,siglength))
        self.assertTrue(numpy.allclose(sigArray15,sigs))

        backsigs=[iisignature.sigbackprop(i,j,m) for i,j in zip(sigs,paths)]
        backsigArray = numpy.stack(backsigs)
        backsigs1315=iisignature.sigbackprop(sigArray1315,pathArray1315,m)
        self.assertEqual(backsigs1315.shape,(1,3,1,5,6,d))
        self.assertTrue(numpy.allclose(backsigs1315.reshape(n,6,2),backsigArray))

        data=[numpy.random.uniform(size=(d,)) for i in range(n)]
        dataArray1315=numpy.stack(data).reshape((1,3,1,5,d))
        joined=[iisignature.sigjoin(i,j,m) for i,j in zip(sigs,data)]
        joined1315=iisignature.sigjoin(sigArray1315,dataArray1315,m)
        self.assertEqual(joined1315.shape,(1,3,1,5,siglength))
        self.assertTrue(numpy.allclose(joined1315.reshape(n,-1),numpy.stack(joined)))
        backjoined=[iisignature.sigjoinbackprop(i,j,k,m) for i,j,k in zip(joined,sigs,data)]
        backjoinedArrays=[numpy.stack([i[j] for i in backjoined]) for j in range(2)]
        backjoined1315=iisignature.sigjoinbackprop(joined1315,sigArray1315,dataArray1315,m)
        self.assertEqual(backjoined1315[0].shape,sigArray1315.shape)
        self.assertEqual(backjoined1315[1].shape,dataArray1315.shape)
        self.assertTrue(numpy.allclose(backjoined1315[0].reshape(n,-1),backjoinedArrays[0]))
        self.assertTrue(numpy.allclose(backjoined1315[1].reshape(n,-1),backjoinedArrays[1]))

        scaled=[iisignature.sigscale(i,j,m) for i,j in zip(sigs,data)]
        scaled1315=iisignature.sigscale(sigArray1315,dataArray1315,m)
        self.assertEqual(scaled1315.shape,(1,3,1,5,siglength))
        self.assertTrue(numpy.allclose(scaled1315.reshape(n,-1),numpy.stack(scaled)))
        backscaled=[iisignature.sigscalebackprop(i,j,k,m) for i,j,k in zip(scaled,sigs,data)]
        backscaledArrays=[numpy.stack([i[j] for i in backscaled]) for j in range(2)]
        backscaled1315=iisignature.sigscalebackprop(scaled1315,sigArray1315,dataArray1315,m)
        self.assertEqual(backscaled1315[0].shape,sigArray1315.shape)
        self.assertEqual(backscaled1315[1].shape,dataArray1315.shape)
        self.assertTrue(numpy.allclose(backscaled1315[0].reshape(n,-1),backscaledArrays[0]))
        self.assertTrue(numpy.allclose(backscaled1315[1].reshape(n,-1),backscaledArrays[1]))

        s=iisignature.prepare(d,m,"cosx")
        for type in ("c","o","s","x"):
            logsigs = [iisignature.logsig(i,s,type) for i in paths]
            logsigArray=numpy.stack(logsigs)
            logsigArray1315=iisignature.logsig(pathArray1315,s,type)
            self.assertEqual(logsigArray1315.shape,(1,3,1,5,logsigs[0].shape[0]),type)
            self.assertTrue(numpy.allclose(logsigArray1315.reshape(n,-1),logsigArray),type)


#sum (2i choose i) for i in 1 to n
# which is the number of linear rotational invariants up to level 2n
def sumCentralBinomialCoefficient(n):
    f = math.factorial
    return sum(f(2 * i) / (f(i) ** 2) for i in range(1,n + 1))

#can run just this with
#python setup.py test -s tests.test_sig.RotInv2d
class RotInv2d(TestCase):
    def dotest(self,type):
        m = 6
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

        #check that the invariants are not repeated
        if type != "k":
            self.assertEqual(length,numpy.linalg.matrix_rank(numpy.column_stack(samePathRotInvs)))

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
        m = 8
        sa = iisignature.rotinv2dprepare(m,"a")
        sk = iisignature.rotinv2dprepare(m,"k")
        ca = iisignature.rotinv2dcoeffs(sa)[-1]
        ck = iisignature.rotinv2dcoeffs(sk)[-1]
        
        #every row of ck should be in the span of the rows of ca
        #i.e.  every column of ck.T should be in the span of the columns of
        #ca.T
        #i.e.  there's a matrix b s.t.  ca.T b = ck.T
        residuals = numpy.linalg.lstsq(ca.T,ck.T)[1]
        self.assertLess(numpy.max(numpy.abs(residuals)),0.000001)

        sq = iisignature.rotinv2dprepare(m, "q")
        cq = iisignature.rotinv2dcoeffs(sq)[-1]
        ss = iisignature.rotinv2dprepare(m, "s")
        cs = iisignature.rotinv2dcoeffs(ss)[-1]
        # every row of cs and cq should be in the span of the rows of ca
        residuals2 = numpy.linalg.lstsq(ca.T, cs.T)[1]
        self.assertLess(numpy.max(numpy.abs(residuals2)), 0.000001)
        residuals2 = numpy.linalg.lstsq(ca.T, cq.T)[1]
        self.assertLess(numpy.max(numpy.abs(residuals2)), 0.000001)

        self.assertEqual(cq.shape, cs.shape)

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
