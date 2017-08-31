import torch
from torch.autograd import Variable
#from torch.nn.modules.module import Module

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import iisignature

from iisignature_torch import Sig, LogSig, SigJoin, SigScale

def trySig():
    inp = Variable(torch.randn(8, 2), requires_grad=True)
    result = Sig(inp,2)
    print(result.data)
    result.backward(torch.randn(result.size()))
    print(inp.grad)

def tryLogSig():
    s=iisignature.prepare(2,2)
    for expanded in (False, True):
        inp = Variable(torch.randn(8, 2), requires_grad=True)
        if expanded:
            result = LogSig(inp,s,"x")
        else:
            result = LogSig(inp, s)
        print(result.data.numpy())
        result.backward(torch.randn(result.size()))
        print(inp.grad.data)
tryLogSig()
    
def trySigScale():
    inp1 = Variable(torch.randn(12,6), requires_grad=True)
    inp2 = Variable(torch.randn(12,2), requires_grad=True)
    result = SigScale(inp1,inp2,2)
    print(result.data)
    result.backward(torch.randn(result.size()))
    print(inp1.grad)
    print(inp2.grad)
#trySigScale()

def trySigJoin():
    inp1 = Variable(torch.randn(12,6), requires_grad=True)
    inp2 = Variable(torch.randn(12,2), requires_grad=True)
    result = SigJoin(inp1,inp2,2)
    print(result.data)
    result.backward(torch.randn(result.size()))
    print(inp1.grad)
    print(inp2.grad)
#trySigJoin()

def trySigJoin2():
    inp1 = Variable(torch.randn(12,12), requires_grad=True)
    inp2 = Variable(torch.randn(12,2), requires_grad=True)
    inp3 = Variable(torch.randn(1),requires_grad=True)
    result = SigJoin(inp1,inp2,2,inp3)
    print(result.data)
    result.backward(torch.randn(result.size()))
    print(inp1.grad)
    print(inp2.grad)
    print(inp3.grad)
#trySigJoin2()
