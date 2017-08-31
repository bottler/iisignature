#This module defines a PyTorch function called Sig,
# which just does iisignature.sig,
#one called LogSig,
# which just does iisignature.logsig,
#one called SigJoin,
# which just does iisignature.sigjoin,
#and one called SigScale,
# which just does iisignature.sigscale.
import torch
from torch.autograd import Function

import iisignature

class SigFn(Function):
    def __init__(self, m):
        super(SigFn, self).__init__()
        self.m = m
    def forward(self,X):
        result=iisignature.sig(X.numpy(), self.m)
        self.save_for_backward(X)
        return torch.FloatTensor(result)
    def backward(self, grad_output):
        (X,) = self.saved_tensors
        result = iisignature.sigbackprop(grad_output.numpy(),X.numpy(),self.m)
        return torch.FloatTensor(result)

class LogSigFn(Function):
    def __init__(self, s, method):
        super(LogSigFn, self).__init__()
        self.s = s
        self.method = method
    def forward(self,X):
        result=iisignature.logsig(X.numpy(), self.s, self.method)
        self.save_for_backward(X)
        return torch.FloatTensor(result)
    def backward(self, grad_output):
        (X,) = self.saved_tensors
        g=grad_output.numpy()
        result = iisignature.logsigbackprop(g,X.numpy(),self.s,self.method)
        return torch.FloatTensor(result)
    
class SigJoinFn(Function):
    def __init__(self, m):
        super(SigJoinFn, self).__init__()
        self.m = m
    def forward(self,X,Y):
        result=iisignature.sigjoin(X.numpy(),Y.numpy(), self.m)
        self.save_for_backward(X,Y)
        return torch.FloatTensor(result)
    def backward(self, grad_output):
        X,Y = self.saved_tensors
        result = iisignature.sigjoinbackprop(grad_output.numpy(),X.numpy(),Y.numpy(),self.m)
        return torch.FloatTensor(result[0]),torch.FloatTensor(result[1])
class SigJoinFixedFn(Function):
    def __init__(self, m):
        super(SigJoinFixedFn, self).__init__()
        self.m = m
    def forward(self,X,Y,fixed):
        result=iisignature.sigjoin(X.numpy(),Y.numpy(), self.m, fixed.numpy())
        self.save_for_backward(X,Y, fixed)
        return torch.FloatTensor(result)
    def backward(self, grad_output):
        X,Y,fixed = self.saved_tensors
        result = iisignature.sigjoinbackprop(grad_output.numpy(),X.numpy(),Y.numpy(),self.m, fixed.numpy())
        return torch.FloatTensor(result[0]),torch.FloatTensor(result[1]),torch.FloatTensor([result[2]])

class SigScaleFn(Function):
    def __init__(self, m):
        super(SigScaleFn, self).__init__()
        self.m = m
    def forward(self,X,Y):
        result=iisignature.sigscale(X.numpy(),Y.numpy(), self.m)
        self.save_for_backward(X,Y)
        return torch.FloatTensor(result)
    def backward(self, grad_output):
        X,Y = self.saved_tensors
        result = iisignature.sigscalebackprop(grad_output.numpy(),X.numpy(),Y.numpy(),self.m)
        return tuple(torch.FloatTensor(i) for i in result)

def Sig(X,m):
    return SigFn(m)(X)

def LogSig(X,s,method=""):
    return LogSigFn(s,method)(X)
    
def SigJoin(X,y,m,fixed=None):
    if fixed is not None:
        return SigJoinFixedFn(m)(X,y,fixed)
    return SigJoinFn(m)(X,y)
    
def SigScale(X,y,m):
    return SigScaleFn(m)(X,y)
