#This module defines a theano op called Sig,
# which just does iisignature.sig,
#one called LogSig,
# which just does iisignature.logsig,
#one called SigJoin,
# which just does iisignature.sigjoin,
#and one called SigScale,
# which just does iisignature.sigscale.

import theano, numpy as np
import iisignature

#Todo: you can use x.ndim in make_node to change behaviour based on shapes of the input.
#could make these functions optionally batched

#Turn on to throw an exception on nan
nancheck = False
def contains_nan(x):
    return np.isnan(x).any()

#Turn on to throw exceptions in some places when called on a non-contiguous array
#At the moment, iisignature functions will do extra copying when this happens.
#If it happens a lot, we could adapt the code.
report_contig=False
if report_contig:
    def iscontig(arr):
        return arr.flags["C_CONTIGUOUS"]
    def contig_check(a):
        for i,aa in enumerate(a):
            if not iscontig(aa):
                raise RuntimeError(str(i+1)+"th argument is not contiguous")      

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
        out[0][0]=np.array(iisignature.siglength(inp[0],inp[1]),dtype="int32")
    def infer_shape(self,node,shapes):
        return [[]]
SigLength=SigLength_op()

#This is a theano Op which wraps iisignature.logsiglength .
#It could be used to implement shape inference of LogSig, but isn't.
class LogSigLength_op(theano.Op):
    __props__=()
    def make_node(self,d,m):
        d = theano.tensor.as_tensor_variable(d)
        m = theano.tensor.as_tensor_variable(m)
        return theano.Apply(self,[d,m],[theano.tensor.iscalar()])
    def perform(self,node,inp,out):
        out[0][0]=np.array(iisignature.logsiglength(inp[0],inp[1]),dtype="int32")
    def infer_shape(self,node,shapes):
        return [[]]
LogSigLength=LogSigLength_op()

#This is a theano Op which wraps iisignature.sigbackprop .
#It is used in the grad method of Sig.
#Ideally we would have an optimization to sum a variable with this in place
# - like backward in torch
class SigGrad_op(theano.Op):
    __props__=()
    def infer_shape(self,node,shapes):
        return [shapes[1]]
    def make_node(self,s,x,m):
        s=theano.tensor.as_tensor_variable(s)
        x=theano.tensor.as_tensor_variable(x)
        m=theano.tensor.as_tensor_variable(m)
        outT=theano.tensor.TensorType("float32",[False]*x.ndim)()
        return theano.Apply(self,inputs=[s,x,m],
                            outputs=[outT])
    def perform(self,node,inputs_storage,out):
        s=inputs_storage[0]
        x=inputs_storage[1]
        m=inputs_storage[2]
        out[0][0]=iisignature.sigbackprop(s,x,m)
SigGrad=SigGrad_op()

#This is a theano Op which wraps iisignature.sig .
class Sig_op(theano.Op):
    __props__=()
    def infer_shape(self,node,shapes):
        return [shapes[0][:-2]+(SigLength(shapes[0][-1],node.inputs[1]),)]
    def make_node(self,x,m):
        x_=theano.tensor.as_tensor_variable(x)
        m=theano.tensor.as_tensor_variable(m)
        outT=theano.tensor.TensorType("float32",[False]*(x.ndim-1))()
        return theano.Apply(self,inputs=[x_,m],
                            outputs=[outT])
    def perform(self,node,inputs_storage,outputs_storage):
        x=inputs_storage[0]
        m=inputs_storage[1]
        outputs_storage[0][0]=iisignature.sig(x,m)
    def grad(self,inputs,g):
        return [SigGrad(g[0],inputs[0],inputs[1]),
                theano.gradient.grad_undefined(self,1,inputs[1])]
Sig = Sig_op()

#There is no smooth way to pass information which is known in one instance of the
#Op, like the prepared object in LogSig, to the Op's grad() method. node is
#not one of the parameters of grad. Thus we store the prepared objects in
#this global variable, and have its index as a constant input to the apply node.
_prepared_obeject_store=[]

#This is a theano Op which wraps iisignature.logsigbackprop .
#The arguments are a bit weird: if you want to use this yourself
#you need to append the prepared object and method as a tuple to
#_prepared_obeject_store and pass the index as an input.
class LogSigGrad_op(theano.Op):
    __props__=()
    def infer_shape(self,node,shapes):
        return [shapes[1]]
    def make_node(self,grad_output,x,index):
        grad_output=theano.tensor.as_tensor_variable(grad_output)
        x=theano.tensor.as_tensor_variable(x)
        outT=theano.tensor.TensorType("float32",[False]*x.ndim)()
        return theano.Apply(self,inputs=[grad_output,x,index],outputs=[outT])
    def perform(self,node,inputs_storage,out):
        grad_output=inputs_storage[0]
        x=inputs_storage[1]
        s,method=_prepared_obeject_store[inputs_storage[2]]
        out[0][0]=iisignature.logsigbackprop(grad_output,x,s,method)
LogSigGrad=LogSigGrad_op()

#This is a theano Op which wraps iisignature.logsig
class LogSig_op(theano.Op):
    __props__=()
    def infer_shape(self,node,shapes):
        #d=shapes[0][-1]
        s,method=_prepared_obeject_store[node.inputs[1].get_value()]
        m=iisignature.info(s)["level"]
        d=iisignature.info(s)["dimension"]
        if "x" in method or "X" in method:
            length=iisignature.siglength(d,m)
        else:
            length=iisignature.logsiglength(d,m)
        return [shapes[0][:-2]+(length,)]
    def make_node(self,x,s,method=""):
        x_=theano.tensor.as_tensor_variable(x)
        outT=theano.tensor.TensorType("float64",[False]*(x.ndim-1))()
        index=theano.shared(len(_prepared_obeject_store))
        _prepared_obeject_store.append((s,method))
        return theano.Apply(self,inputs=[x_,index], outputs=[outT])
    def perform(self,node,inputs_storage,outputs_storage):
        x=inputs_storage[0]
        s,method=_prepared_obeject_store[inputs_storage[1]]
        outputs_storage[0][0]=iisignature.logsig(x,s,method)
    def grad(self,inputs,g):
        return [LogSigGrad(g[0],inputs[0],inputs[1]),
                theano.gradient.grad_undefined(self,1,inputs[1])]
LogSig = LogSig_op()

#This is a theano Op which wraps iisignature.sigjoinbackprop .
#It has multiple outputs.
class SigJoinGrad_op(theano.Op):
    __props__=()
    def infer_shape(self,node,shapes):
        return [shapes[1],shapes[2], None]
    def make_node(self,s,x,y,m,fixed):
        s=theano.tensor.as_tensor_variable(s)
        x=theano.tensor.as_tensor_variable(x)
        y=theano.tensor.as_tensor_variable(y)
        m=theano.tensor.as_tensor_variable(m)
        fixed=theano.tensor.as_tensor_variable(fixed)
        outT=theano.tensor.TensorType("float32",[False]*(x.ndim))
        return theano.Apply(self,inputs=[s,x,y,m,fixed],
                            outputs=[outT(), outT(), theano.tensor.dscalar()])
    def perform(self,node,inputs_storage,out):
        s=inputs_storage[0]
        x=inputs_storage[1]
        y=inputs_storage[2]
        m=inputs_storage[3]
        fixed=inputs_storage[4]
        if report_contig:
            contig_check([s,x,y])
        o=iisignature.sigjoinbackprop(s,x,y,m,fixed)
        if(nancheck):
            if contains_nan(s):
                raise RuntimeError("nan in s")
            if contains_nan(x):
                raise RuntimeError("nan in x")
            if contains_nan(y):
                raise RuntimeError("nan in y")
            if contains_nan(o[0]) or contains_nan(o[1]):
                raise RuntimeError("nan in output")
        out[0][0]=o[0]
        out[1][0]=o[1]
        out[2][0]=np.array(0.0 if np.isnan(fixed) else o[2], dtype="float64")
SigJoinGrad = SigJoinGrad_op()

#This is a theano Op which wraps iisignature.sigjoin .
class SigJoin_op(theano.Op):
    __props__=()
    def infer_shape(self,node,shapes):
        return [shapes[0]]
    def make_node(self,x,y,m,fixed=float("nan")):
        x=theano.tensor.as_tensor_variable(x)
        y=theano.tensor.as_tensor_variable(y)
        m=theano.tensor.as_tensor_variable(m)
        fixed=theano.tensor.as_tensor_variable(fixed)
        outT=theano.tensor.TensorType("float32",[False]*(x.ndim))
        return theano.Apply(self,inputs=[x,y,m,fixed],
                            outputs=[outT()])
    def perform(self,node,inputs_storage,outputs_storage):
        x=inputs_storage[0]
        y=inputs_storage[1]
        m=inputs_storage[2]
        fixed = inputs_storage[3]
        outputs_storage[0][0]=iisignature.sigjoin(x,y,m,fixed)
        if(nancheck):
            if contains_nan(x):
                raise RuntimeError("nan in x")
            if contains_nan(y) :
                raise RuntimeError("nan in y")
            if contains_nan(outputs_storage[0][0]) :
                raise RuntimeError("nan in output")
    def grad(self,inputs,g):
        gg = SigJoinGrad(g[0],inputs[0],inputs[1],inputs[2],inputs[3])
        return [gg[0],gg[1],
                theano.gradient.grad_undefined(self,2,inputs[2]),
                gg[2]]
SigJoin = SigJoin_op()

#This is a theano Op which wraps iisignature.sigscalebackprop .
#It has two outputs.
class SigScaleGrad_op(theano.Op):
    __props__=()
    def infer_shape(self,node,shapes):
        return [shapes[1],shapes[2]]
    def make_node(self,s,x,y,m):
        s=theano.tensor.as_tensor_variable(s)
        x=theano.tensor.as_tensor_variable(x)
        y=theano.tensor.as_tensor_variable(y)
        m=theano.tensor.as_tensor_variable(m)
        outT=theano.tensor.TensorType("float32",[False]*(x.ndim))
        return theano.Apply(self,inputs=[s,x,y,m],
                            outputs=[outT(), outT()])
    def perform(self,node,inputs_storage,out):
        s=inputs_storage[0]
        x=inputs_storage[1]
        y=inputs_storage[2]
        m=inputs_storage[3]
        if report_contig:
            contig_check([s,x,y])
        o=iisignature.sigscalebackprop(s,x,y,m)
        if(nancheck):
            if contains_nan(s):
                raise RuntimeError("nan in s")
            if contains_nan(x):
                raise RuntimeError("nan in x")
            if contains_nan(y):
                raise RuntimeError("nan in y")
            if contains_nan(o[0]) or contains_nan(o[1]):
                raise RuntimeError("nan in output")
        out[0][0]=o[0]
        out[1][0]=o[1]
SigScaleGrad = SigScaleGrad_op()

#This is a theano Op which wraps iisignature.sigscale .
class SigScale_op(theano.Op):
    __props__=()
    def infer_shape(self,node,shapes):
        return [shapes[0]]
    def make_node(self,x,y,m):
        x=theano.tensor.as_tensor_variable(x)
        y=theano.tensor.as_tensor_variable(y)
        m=theano.tensor.as_tensor_variable(m)
        outT=theano.tensor.TensorType("float32",[False]*(x.ndim))
        return theano.Apply(self,inputs=[x,y,m],
                            outputs=[outT()])
    def perform(self,node,inputs_storage,outputs_storage):
        x=inputs_storage[0]
        y=inputs_storage[1]
        m=inputs_storage[2]
        outputs_storage[0][0]=iisignature.sigscale(x,y,m)
        if(nancheck):
            if contains_nan(x):
                raise RuntimeError("nan in x")
            if contains_nan(y) :
                raise RuntimeError("nan in y")
            if contains_nan(outputs_storage[0][0]) :
                raise RuntimeError("nan in output")
    def grad(self,inputs,g):
        gg = SigScaleGrad(g[0],inputs[0],inputs[1],inputs[2])
        return [gg[0],gg[1],
                theano.gradient.grad_undefined(self,2,inputs[2])]
SigScale = SigScale_op()


#http://deeplearning.net/software/theano/extending/extending_theano_gpu.html
#http://theano.readthedocs.io/en/rel-0.6rc3/tutorial/extending_theano.html
