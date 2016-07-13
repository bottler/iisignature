import os
#os.environ["THEANO_FLAGS"]="floatX=float32,device=cpu,optimizer=fast_compile"
#os.environ["THEANO_FLAGS"]="floatX=float32,device=cpu,mode=DebugMode"
os.environ["THEANO_FLAGS"]="floatX=float32,device=cpu"
import theano, numpy, sys
import six.moves

#add the parent directory, so we find our iisignature build if it was built --inplace
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import iisignature

from iisignature_theano import Sig
numpy.set_printoptions(suppress=True)

dim=2
level=3
pathlength=4
timedim=False

#In this simple example, we demonstrate attempting to learn the points of a path
#from its signature via gradient descent.
#The starting point is another path whose first point matches the target
numpy.random.seed(51)
target = numpy.random.uniform(size=(pathlength,dim)).astype("float32")
#target = numpy.cumsum(2*(target-0.5),0)#makes it more random-walk-ish, less like a scribble

targetSig = iisignature.sig(target,level)
start = numpy.random.uniform(size=(pathlength,dim)).astype("float32")
start[0,:] = target[0,:]
learnable_mask = numpy.ones((pathlength,dim)).astype("float32")
learnable_mask[0,:]=0 #to stop the starting point being modified
learning_rate = 0.5
if timedim:
    for i in six.moves.xrange(pathlength):
        target[i,0]=start[i,0]=i*0.2
        learnable_mask[i,0]=0

path = theano.shared(start, "path")
cost = theano.tensor.mean(theano.tensor.sqr(Sig(path,level)-targetSig))
ff = theano.function([],[cost],updates=[(path,path-learning_rate*learnable_mask*theano.grad(cost,path))])
print ("target:")
print (target)
print ("start:")
print (start)
for i in six.moves.xrange(181): 
    stepcost = ff() #one step of gradient descent
    if i%18 == 0:
        print ("step " + str(i)  + " cost: " + str(ff()[0].item()))
print ("end:")
print (path.get_value())
#a=iisignature.sig(start_.get_value(),level)
#print (numpy.vstack([a,targetSig,a-targetSig]).transpose())

##http://deeplearning.net/software/theano/extending/extending_theano_gpu.html
##http://theano.readthedocs.io/en/rel-0.6rc3/tutorial/extending_theano.html
