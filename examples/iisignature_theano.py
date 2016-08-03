#This module defines a theano op called Sig,
# which just does iisignature.sig .

import theano, numpy
import iisignature

#This is a theano Op which wraps iisignature.siglength .
#It is used to implement shape inference of Sig.
class SigLength_op(theano.Op):
    __props__=()
    def make_node(self,d,m):
        d = theano.tensor.as_tensor_variable(d)
        m = theano.tensor.as_tensor_variable(m)
        return theano.Apply(self,[d,m],[theano.tensor.iscalar()])
    def perform(self,node,inp,out):
        #do I really need to create an array here?
        out[0][0]=numpy.array(iisignature.siglength(inp[0],inp[1]),dtype="int32")
    def infer_shape(self,node,shapes):
        return [[]]
SigLength=SigLength_op()

#This is a theano Op which wraps sigbackprop .
#It is used in the grad method of Sig.
#Ideally we would have an optimization to sum a variable with this in place
# - like backward in torch
class SigGrad_op(theano.Op):
    __props__=()
    def infer_shape(self,node,shapes):
        return [shapes[0]]
    def make_node(self,x,m,s):
        x=theano.tensor.as_tensor_variable(x)
        m=theano.tensor.as_tensor_variable(m)
        s=theano.tensor.as_tensor_variable(s)
        return theano.Apply(self,inputs=[x,m,s],
                            outputs=[theano.tensor.fmatrix()])
    def perform(self,node,inputs_storage,out):
        x=inputs_storage[0]
        m=inputs_storage[1]
        s=inputs_storage[2]
        out[0][0]=iisignature.sigbackprop(x,m,s)
SigGrad=SigGrad_op()

#This is a theano Op which wraps iisignature.sig
class Sig_op(theano.Op):
    __props__=()
    def infer_shape(self,node,shapes):
        return [[SigLength(shapes[0][1],node.inputs[1])]]
    def make_node(self,x,m):
        x_=theano.tensor.as_tensor_variable(x)
        m=theano.tensor.as_tensor_variable(m)
        return theano.Apply(self,inputs=[x_,m],
                            outputs=[theano.tensor.fvector()])
    def perform(self,node,inputs_storage,outputs_storage):
        x=inputs_storage[0]
        m=inputs_storage[1]
        outputs_storage[0][0]=iisignature.sig(x,m)
    def grad(self,inputs,g):
        return [SigGrad(inputs[0],inputs[1],g[0]),theano.gof.null_type.NullType()()]
#The variable Sig is the only thing in this module intended for external use.
Sig = Sig_op()

#http://deeplearning.net/software/theano/extending/extending_theano_gpu.html
#http://theano.readthedocs.io/en/rel-0.6rc3/tutorial/extending_theano.html
