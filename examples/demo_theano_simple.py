#In this simple example, we demonstrate using the signature functionality from iisignature
#by attempting to learn the points of a path
#from its signature via gradient descent.
#The starting point is another path whose first point matches the target

import os
#os.environ["THEANO_FLAGS"]="floatX=float32,device=cpu,optimizer=fast_compile"
#os.environ["THEANO_FLAGS"]="floatX=float32,device=cpu,mode=DebugMode"
os.environ["THEANO_FLAGS"]="floatX=float32,device=cpu"
import theano, numpy, sys
import six.moves
import theano.tensor as T

#add the parent directory, so we find our iisignature build if it was built --inplace
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import iisignature

from iisignature_theano import LogSig, Sig, LogSigGrad


#1: SETUP VARIABLES
dim=2
level=3
pathlength=4
timedim=False
useLogSig = True
s=iisignature.prepare(dim,level)

numpy.random.seed(51)
target = numpy.random.uniform(size=(pathlength,dim)).astype("float32")

a=T.dmatrix("a")
b=T.dmatrix("b")

#c=LogSigGrad(b,a,s)
#f=theano.function([a,b],c)
#print (f(target,target))

c=LogSig(b,s)
grad=theano.grad(theano.tensor.sum(c),b)
f=theano.function([b],[c,grad])
o=f(target)

print (o[0])
print (o[1])

