import iisignature
import tensorflow as tf
import numpy as np

#Provides Sig, LogSig, SigJoin and SigScale as tensorflow operations
#to match sig, logsig, sigjoin and sigscale from iisignature.
#Unlike for theano, there's no obvious reason to provide an
#op for siglength.

_zero=np.array(0.0,dtype="float32")

#SECTION 1: implementation of each gradient
def _sigGradImp(g,x,m):
    o=iisignature.sigbackprop(g,x,m)
    return o, _zero

class _logSigGradImp:
    def __init__(self,s, method):
        self.s=s
        self.method=method
    def __call__(self,g,x):
        return iisignature.logsigbackprop(g,x,self.s,self.method)

def _sigScaleGradImp(g,x,y,m):
    o= iisignature.sigscalebackprop(g,x,y,m)
    return o[0],o[1],_zero

def _sigJoinGradImp(g,x,y,m):
    o= iisignature.sigjoinbackprop(g,x,y,m)
    return o[0],o[1],_zero

def _sigJoinGradFixedImp(g,x,y,m,fixedlast):
    o= iisignature.sigjoinbackprop(g,x,y,m,fixedlast)
    return o[0],o[1],_zero,np.array(o[2],dtype="float32")

#SECTION 2: op for each gradient
def _sigGrad(op, grad):
    return tf.py_func(_sigGradImp,[grad]+list(op.inputs),
                      [tf.float32]*2, name="SigGrad")

class _logSigGrad:
    def __init__(self,s, method):
        self.s=s
        self.method=method
    def __call__(self,op,grad):
        fn = _logSigGradImp(self.s,self.method)
        return tf.py_func(fn,[grad]+list(op.inputs),
                      [tf.float32], name="LogSigGrad")

def _sigScaleGrad(op, grad):
    return tf.py_func(_sigScaleGradImp,[grad]+list(op.inputs),
                      [tf.float32]*3, name="SigScaleGrad")

def _sigJoinGrad(op, grad):
    return tf.py_func(_sigJoinGradImp,[grad]+list(op.inputs),
                      [tf.float32]*3, name="SigJoinGrad")

def _sigJoinGradFixed(op, grad):
    return tf.py_func(_sigJoinGradFixedImp,[grad]+list(op.inputs),
                      [tf.float32]*4, name="SigJoinGradFixed")

#SECTION 3: implementation of each forward operation
def _sigImp(x,m):
    return iisignature.sig(x,m)

class _logSigImp:
    def __init__(self,s, method):
        self.s=s
        self.method=method
    def __call__(self,x):
        return iisignature.logsig(x,self.s,self.method)

def _sigScaleImp(x,y,m):
    return iisignature.sigscale(x,y,m)

def _sigJoinImp(x,y,m):
    return iisignature.sigjoin(x,y,m)
def _sigJoinFixedImp(x,y,m,fixedLast):
    return iisignature.sigjoin(x,y,m,fixedLast)

#SECTION 4: op for each forward operation
def Sig(x, m):
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
    tf.RegisterGradient(rnd_name)(_sigGrad)
    g=tf.get_default_graph()
    with g.gradient_override_map({"PyFunc":rnd_name}):
        return tf.py_func(_sigImp, [x,m], tf.float32, name="Sig") 

def LogSig(x, s, method=""):
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
    tf.RegisterGradient(rnd_name)(_logSigGrad(s,method))
    g=tf.get_default_graph()
    with g.gradient_override_map({"PyFunc":rnd_name}):
        return tf.py_func(_logSigImp(s,method), [x], tf.float64, name="LogSig") 

def SigScale(x, y, m):
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
    tf.RegisterGradient(rnd_name)(_sigScaleGrad)
    g=tf.get_default_graph()
    with g.gradient_override_map({"PyFunc":rnd_name}):
        return tf.py_func(_sigScaleImp, [x,y,m], tf.float32, name="SigScale") 
    
def SigJoin(x,y,m,fixedLast=None):
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
    if fixedLast is None:
        tf.RegisterGradient(rnd_name)(_sigJoinGrad)
        g=tf.get_default_graph()
        with g.gradient_override_map({"PyFunc":rnd_name}):
            return tf.py_func(_sigJoinImp, [x,y,m], tf.float32, name="SigJoin")
    else:
        tf.RegisterGradient(rnd_name)(_sigJoinGradFixed)
        g=tf.get_default_graph()
        with g.gradient_override_map({"PyFunc":rnd_name}):
            return tf.py_func(_sigJoinFixedImp, [x,y,m,fixedLast], tf.float32, name="SigJoin")

#https://stackoverflow.com/questions/37924071/tensorflow-writing-an-op-in-python
#https://gist.github.com/harpone/3453185b41d8d985356cbe5e57d67342
