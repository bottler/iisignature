#This boring example just shows that using SigJoin and Sig
#you get the same signatures and derivatives.

import os
#os.environ["THEANO_FLAGS"]="floatX=float32,device=cpu,optimizer=fast_compile"
#os.environ["THEANO_FLAGS"]="floatX=float32,device=cpu,mode=DebugMode"
os.environ["THEANO_FLAGS"]="floatX=float32,device=cpu"

import theano, numpy, sys
import six.moves

#add the parent directory, so we find our iisignature build if it was built --inplace
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import iisignature

from iisignature_theano import Sig, SigJoin
import theano.tensor as T


#1: SETUP VARIABLES
dim=3
level=6
pathlength=4
fixed = float("nan")
fixed = 0.1

numpy.random.seed(51)
start = numpy.random.uniform(size=(pathlength,dim)).astype("float32")

#2: DEFINE THEANO STUFF

path = theano.shared(start, "path")
if not numpy.isnan(fixed):
    fixedT = theano.shared(fixed, "fixedT")
    path=theano.tensor.horizontal_stack(path[:,:-1],fixedT*T.shape_padright(T.arange(pathlength)))
cost1 = theano.tensor.mean(theano.tensor.sqr(Sig(path,level)))
grad1 = theano.grad(cost1,path)

signature = numpy.zeros((1,iisignature.siglength(dim,level))).astype("float32")
for i in six.moves.xrange(1,pathlength):
    if not numpy.isnan(fixed):
        displacement = path[i:(i+1),:-1]-path[(i-1):i,:-1]
        signature = SigJoin(signature,displacement,level,fixedT)
    else:
        displacement = path[i:(i+1),:]-path[(i-1):i,:]
        signature = SigJoin(signature,displacement,level)

cost2 = theano.tensor.mean(theano.tensor.sqr(signature))
grad2 = theano.grad(cost2,path)

#theano.printing.pydotprint(grad2,outfile="a.png")

ff = theano.function([],[grad1,grad2])
#theano.printing.pydotprint(ff,outfile="b.png")

#3: GO

numpy.set_printoptions(suppress=True)
f = ff()
if numpy.isnan(fixed):
    print (f[0]-f[1])
    print (f[0])
    print (f[1])
else:
    print ((f[0]-f[1])[:,:-1])
    print (f[0][:,:-1])
    print (f[1][:,:-1])
    ff2 = theano.function([],[theano.grad(cost1,fixedT),theano.grad(cost2,fixedT)])()
    print(ff2)



