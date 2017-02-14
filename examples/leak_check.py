import os, sys, numpy as np
from six.moves import range

#This program just continually runs some of the important functionality in
#iisignature on small data, so that you can check that memory usage
#does not grow. Even a tiny memory leak becomes very visible e.g. in htop.

#add the parent directory, so we find our iisignature build if it was built --inplace
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import iisignature

length = 20
dim=3
level=2
npaths=3
paths_ = np.random.uniform(size=(npaths,length,dim))
scale_ = np.random.uniform(size=(npaths,dim))
initialsigs_ = np.random.uniform(size=(npaths,iisignature.siglength(dim,level)))
p=iisignature.prepare(dim,level,"cosx")
while 0:
    iisignature.sig(paths[0],level)
for i in range(10**10):
    #copy major parts of the input data, in case we are leaking references to it
    paths=paths_[:]
    increment=scale=scale_[:]
    initialsigs=initialsigs_[:]
    iisignature.sigjoin(initialsigs,scale,level)
    iisignature.sigscale(initialsigs,scale,level)
    iisignature.sigjoinbackprop(initialsigs,initialsigs,scale,level)
    iisignature.sigscalebackprop(initialsigs,initialsigs,scale,level)
    iisignature.sig(paths[0,:,:],level)
    iisignature.sigbackprop(initialsigs[0,:],paths[0,:,:],level)
    #iisignature.sigjacobian(paths[0,:,:],level)
    #iisignature.prepare(dim,level,"cosx")#much slower than other functions
    iisignature.logsig(paths[0,:,:],p,"c")
    iisignature.logsig(paths[0,:,:],p,"o")
    iisignature.logsig(paths[0,:,:],p,"s")                       
    if i%10000==0:
        print (i)
 
