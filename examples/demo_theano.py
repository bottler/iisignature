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

#add the parent directory, so we find our iisignature build if it was built --inplace
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import iisignature

from iisignature_theano import LogSig, Sig


#1: SETUP VARIABLES
dim=2
level=3
pathlength=4
timedim=False
useLogSig = True
s=iisignature.prepare(dim,level)

numpy.random.seed(51)
target = numpy.random.uniform(size=(pathlength,dim)).astype("float32")
#target = numpy.cumsum(2*(target-0.5),0)#makes it more random-walk-ish, less like a scribble

targetSig = iisignature.logsig(target,s) if useLogSig else iisignature.sig(target,level)
start = numpy.random.uniform(size=(pathlength,dim)).astype("float32")
start[0,:] = target[0,:]
learnable_mask = numpy.ones((pathlength,dim)).astype("float32")
learnable_mask[0,:]=0 #to stop the starting point being modified
if timedim:
    for i in six.moves.xrange(pathlength):
        target[i,0]=start[i,0]=i*0.2
        learnable_mask[i,0]=0
learning_rate_decay = 1.003
initial_learning_rate = 1.2
momentum = 0.95

#2: DEFINE THEANO STUFF

learning_rate = theano.shared(numpy.float32(initial_learning_rate),"learning_rate")
path = theano.shared(start, "path")
grad_avg = theano.shared(numpy.zeros_like(start,dtype="float32"),"grad_avg")
sig = LogSig(path,s) if useLogSig else Sig(path,level)
cost = theano.tensor.mean(theano.tensor.sqr(sig-targetSig))
new_grad_avg = (momentum*grad_avg)+(1-momentum)*theano.grad(cost,path)
use_old=False
grad_to_use=grad_avg if use_old else new_grad_avg
ff = theano.function([],[cost],updates=[(path,path-learning_rate*learnable_mask*grad_to_use),
                                        (grad_avg,new_grad_avg),
                                        (learning_rate,learning_rate*learning_rate_decay)])

#3: GO

numpy.set_printoptions(suppress=True)
print ("target:")
print (target)
print ("start:")
print (start)
for i in six.moves.xrange(581): 
    stepcost = ff() #one step of gradient descent
    if i%18 == 0:
        print ("step " + str(i)  + " cost: " + str(ff()[0].item()))
print ("end:")
print (path.get_value())
#a=iisignature.sig(start_.get_value(),level)
#print (numpy.vstack([a,targetSig,a-targetSig]).transpose())

##http://deeplearning.net/software/theano/extending/extending_theano_gpu.html
##http://theano.readthedocs.io/en/rel-0.6rc3/tutorial/extending_theano.html
