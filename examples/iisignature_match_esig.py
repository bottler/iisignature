#This file provides a wrapper around iisignature which provides the
#same interface as esig.tosig.

import iisignature
import numpy as np
from six.moves import range

_prep={}
_prepLyndon={}
def _getPrep(d,m, lyndon=False):
    hash2use = _prepLyndon if lyndon else _prep
    key = (d,m)
    if key in hash2use:
        return hash2use[key]
    ans = iisignature.prepare(d,m, "D" if lyndon else "DH")
    hash2use[key]=ans
    return ans

def sigdim(d,m):
    return 1 + iisignature.siglength(d,m)

logsigdim = iisignature.logsiglength

def sigkeys(d,m):
    from itertools import chain, product
    alphabet="123456789"[:d]
    it=chain.from_iterable(product(alphabet, repeat=r) for r in range(m+1))
    return " "+" ".join("("+",".join(j for j in i)+")" for i in it)

def logsigkeys(d,m):
    s=_getPrep(d,m)
    return " "+" ".join(iisignature.basis(s))

def stream2logsig(path, m):
    d=path.shape[1]
    s=_getPrep(d,m)
    return iisignature.logsig(path,s).astype("float64")

def stream2logsig_Lyndon(path, m):
    d=path.shape[1]
    s=_getPrep(d,m,True)
    return iisignature.logsig(path,s).astype("float64")

def stream2sig(path, m):
    return np.hstack([1.0, iisignature.sig(path,m)])
