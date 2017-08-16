import os, sys, numpy as np
#add the parent directory, so we find our iisignature build if it was built --inplace
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import iisignature_match_esig as tosig

d=3
m=4
path=np.random.uniform(size=(10,d))

print(tosig.stream2sig(path,m))
print(tosig.stream2logsig(path,m))
print(tosig.sigkeys(d,m))
print(tosig.logsigkeys(d,m))
print(tosig.sigdim(d,m))
print(tosig.logsigdim(d,m))

