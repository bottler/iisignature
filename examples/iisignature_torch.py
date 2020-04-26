# This module defines a PyTorch function called Sig,
#  which just does iisignature.sig,
# one called LogSig,
#  which just does iisignature.logsig,
# one called SigJoin,
#  which just does iisignature.sigjoin,
# and one called SigScale,
#  which just does iisignature.sigscale.
import torch
from torch.autograd import Function

import iisignature


class SigFn(Function):
    @staticmethod
    def forward(ctx, X, m):
        result = iisignature.sig(X.detach().numpy(), m)
        ctx.save_for_backward(X)
        ctx.m = m
        return torch.FloatTensor(result)

    @staticmethod
    def backward(ctx, grad_output):
        (X,) = ctx.saved_tensors
        m = ctx.m
        result = iisignature.sigbackprop(grad_output.numpy(), X.detach().numpy(), m)
        return torch.FloatTensor(result), None


class LogSigFn(Function):
    @staticmethod
    def forward(ctx, X, s, method):
        result = iisignature.logsig(X.detach().numpy(), s, method)
        ctx.save_for_backward(X)
        ctx.s = s
        ctx.method = method
        return torch.FloatTensor(result)

    @staticmethod
    def backward(ctx, grad_output):
        (X,) = ctx.saved_tensors
        s = ctx.s
        method = ctx.method
        g = grad_output.numpy()
        result = iisignature.logsigbackprop(g, X.detach().numpy(), s, method)
        return torch.FloatTensor(result), None, None


class SigJoinFn(Function):
    @staticmethod
    def forward(ctx, X, Y, m):
        result = iisignature.sigjoin(X.detach().numpy(), Y.detach().numpy(), m)
        ctx.save_for_backward(X, Y)
        ctx.m = m
        return torch.FloatTensor(result)

    @staticmethod
    def backward(ctx, grad_output):
        X, Y = ctx.saved_tensors
        m = ctx.m
        result = iisignature.sigjoinbackprop(
            grad_output.numpy(), X.detach().numpy(), Y.detach().numpy(), m
        )
        return torch.FloatTensor(result[0]), torch.FloatTensor(result[1]), None


class SigJoinFixedFn(Function):
    @staticmethod
    def forward(ctx, X, Y, m, fixed):
        result = iisignature.sigjoin(
            X.detach().numpy(), Y.detach().numpy(), m, fixed.detach().numpy()
        )
        ctx.save_for_backward(X, Y, fixed)
        ctx.m = m
        return torch.FloatTensor(result)

    @staticmethod
    def backward(ctx, grad_output):
        X, Y, fixed = ctx.saved_tensors
        m = ctx.m
        result = iisignature.sigjoinbackprop(
            grad_output.numpy(),
            X.detach().numpy(),
            Y.detach().numpy(),
            m,
            fixed.detach().numpy(),
        )
        return (
            torch.FloatTensor(result[0]),
            torch.FloatTensor(result[1]),
            None,
            torch.FloatTensor([result[2]]),
        )


class SigScaleFn(Function):
    @staticmethod
    def forward(ctx, X, Y, m):
        result = iisignature.sigscale(X.detach().numpy(), Y.detach().numpy(), m)
        ctx.save_for_backward(X, Y)
        ctx.m = m
        return torch.FloatTensor(result)

    @staticmethod
    def backward(ctx, grad_output):
        X, Y = ctx.saved_tensors
        m = ctx.m
        result = iisignature.sigscalebackprop(
            grad_output.numpy(), X.detach().numpy(), Y.detach().numpy(), m
        )
        return torch.FloatTensor(result[0]), torch.FloatTensor(result[1]), None


def Sig(X, m):
    return SigFn.apply(X, m)


def LogSig(X, s, method=""):
    return LogSigFn.apply(X, s, method)


def SigJoin(X, y, m, fixed=None):
    if fixed is not None:
        return SigJoinFixedFn.apply(X, y, m, fixed)
    return SigJoinFn.apply(X, y, m)


def SigScale(X, y, m):
    return SigScaleFn.apply(X, y, m)
