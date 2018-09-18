#This shows paths which are made of traversing a figure of eight,
#you seem to be able to get arbitrarily (Fibonaccily?) many levels of the signature to be 0.

import sys, os, numpy as np, math
from math import cos,sin,pi

#add the parent directory, so we find our iisignature build if it was built --inplace
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import iisignature

np.set_printoptions(suppress=True, precision=6)


fractions = np.arange(1,101)/100.0

#This is a unit circle centred on [0.5,0]
positivePetal=np.column_stack([1-np.cos(fractions*pi*2),
                               np.sin(fractions*pi*2)])

petals=[i*positivePetal for i in [[1,1],[1,-1],[-1,1],[-1,-1]]]
         #   2<-   #                     #    ->0   #
      #               #              #                  #

   #                     #        #                        #
                       
#                           #   #                             #
                             # #

#                             O

#                            # #                              #
                            #   #

  #                        #       #                         #
 
      #               #              #                   #
         #   3<- #                       #    ->1    #


#USER: leave uncommented the version of seq which you care about.
#circle - nonzero level 2
seq=[1]

#figure of eight - nonzero level 3
seq=[1,3]
seq=[2,0]

#nonzero level 5
seq=[1,3,0,2]
#seq=[3,1,2,0]
#seq=[2,0,3,1]
#seq=[0,2,1,3]

#nonzero level 8
seq=[1,3,0,2,2,0,3,1]
#seq=[3,1,2,0,0,2,1,3]
#seq=[2,0,3,1,1,3,0,2]
#seq=[0,2,1,3,3,1,2,0]

#nonzero level 13
seq=[1,3,0,2,2,0,3,1,3,1,2,0,0,2,1,3]
#seq=[3,1,2,0,0,2,1,3,1,3,0,2,2,0,3,1]
#seq=[0,2,1,3,3,1,2,0,2,0,3,1,1,3,0,2]
#seq=[2,0,3,1,1,3,0,2,0,2,1,3,3,1,2,0]

#zero up to 
#seq=[1,3,0,2,2,0,3,1,3,1,2,0,0,2,1,3,0,2,1,3,3,1,2,0,2,0,3,1,1,3,0,2]

#The pattern from one to the next seems to be:
#  0 -> 2 0
#  1 -> 1 3
#  2 -> 3 1
#  3 -> 0 2

total=[[(0,0)]]
for i in range(1,len(seq)):
    s={seq[i-1],seq[i]}
    if s in [{0,1},{2,3}]:
        raise RuntimeError("treelike")
for i in seq:
    offset=np.array([0,0])#change this to [2,0] to display petals separately
    startpoint=total[-1][-1]+offset
    if i in [0,1,2,3]:
        total.append(startpoint+petals[i])
    else:
        raise "whoops"

path=np.vstack(total)
print("Maximum absolute value of sig elements in each level:")
for i,j in enumerate(iisignature.sig(path,14,1)):
    print("level {:2}:".format(i+1),np.max(np.abs(j)))

if 0:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    #colors = np.arange(path.
    #plt.plot(path[:,0],path[:,1])
    plt.scatter(path[:,0],path[:,1],c=np.arange(path.shape[0]),
                norm=mpl.colors.Normalize(vmin=0,vmax=path.shape[0]))
    plt.show()




#A sort of Thue Morse chain of fig eights

