import iisignature
import tensorflow as tf
import numpy as np

#Provides Sig, SigJoin and SigScale as tensorflow operations
#to match sig, sigjoin and sigscale from iisignature.
#Unlike for theano, there's no obvious reason to provide an
#op for siglength.

#Sig also allows a batch dimension

_zero=np.array(0.0,dtype="float32")

def _sigGradImp(g,x,m):
    o=iisignature.sigbackprop(g,x,m)
    return o, _zero

def _sigScaleGradImp(g,x,y,m):
    o= iisignature.sigscalebackprop(g,x,y,m)
    return o[0],o[1],_zero

def _sigJoinGradImp(g,x,y,m):
    o= iisignature.sigjoinbackprop(g,x,y,m)
    return o[0],o[1],_zero

def _sigGrad(op, grad):
    return tf.py_func(_sigGradImp,[grad]+list(op.inputs),
                      [tf.float32]*2, name="SigGrad")

def _sigScaleGrad(op, grad):
    return tf.py_func(_sigScaleGradImp,[grad]+list(op.inputs),
                      [tf.float32]*3, name="SigScaleGrad")

def _sigJoinGrad(op, grad):
    return tf.py_func(_sigJoinGradImp,[grad]+list(op.inputs),
                      [tf.float32]*3, name="SigJoinGrad")

def _sigImp(x,m):
    return iisignature.sig(x,m)

def _sigScaleImp(x,y,m):
    return iisignature.sigscale(x,y,m)

def _sigJoinImp(x,y,m):
    return iisignature.sigjoin(x,y,m)

def Sig(x, m):
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
    tf.RegisterGradient(rnd_name)(_sigGrad)
    g=tf.get_default_graph()
    with g.gradient_override_map({"PyFunc":rnd_name}):
        return tf.py_func(_sigImp, [x,m], tf.float32, name="Sig") 

def SigScale(x, y, m):
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
    tf.RegisterGradient(rnd_name)(_sigScaleGrad)
    g=tf.get_default_graph()
    with g.gradient_override_map({"PyFunc":rnd_name}):
        return tf.py_func(_sigScaleImp, [x,y,m], tf.float32, name="SigScale") 
    
def SigJoin(x,y,m):
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
    tf.RegisterGradient(rnd_name)(_sigJoinGrad)
    g=tf.get_default_graph()
    with g.gradient_override_map({"PyFunc":rnd_name}):
        return tf.py_func(_sigJoinImp, [x,y,m], tf.float32, name="SigJoin")

#https://stackoverflow.com/questions/37924071/tensorflow-writing-an-op-in-python
#https://gist.github.com/harpone/3453185b41d8d985356cbe5e57d67342
