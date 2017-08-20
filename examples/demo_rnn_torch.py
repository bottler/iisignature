#A trivial demonstration of the RecurrentSig layer from iisignature_recurrent_torch.py
#No assertion is made that this model is a good idea, or that this code is idiomatic pytorch.

import numpy as np, sys, os, itertools
import torch
from torch.autograd import Variable
import torch.nn as nn

#add the parent directory, so we find our iisignature build if it was built --inplace
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from iisignature_recurrent_torch import RecurrentSig

criterion = nn.MSELoss()

#The task here for the network to learn is very easy - the average of two of the inputs
x = np.random.uniform(size=(2311,5,3))
y = (x[:,1,1] + x[:,3,2])/2 # The output is a number between 0 and 1, so matches sigmoid activation of the top layer
testx = x[2000:,:,:]
testy = y[2000:]
x=x[:2000,:,:]
y=y[:2000]

rnn=RecurrentSig(3,5,sig_level=2,use_signatures = False, output_signatures = False, train_time_lapse=False)
finalLayer=nn.Linear(5,1)

optimizer=torch.optim.Adam(itertools.chain(rnn.parameters(),finalLayer.parameters()),lr=0.0001)

minibatch_size = 32

def train(x_batch, y_batch):
    minibatch_size=y_batch.shape[0]
    hidden = rnn.initHidden(minibatch_size)
    optimizer.zero_grad()
    x_batch=Variable(torch.FloatTensor(x_batch))
    for i in range(5):
        output,hidden = rnn(x_batch[:,i,:], hidden)
        
    output=finalLayer(output)
    loss=criterion(output,Variable(torch.FloatTensor(y_batch)))
    loss.backward()
    optimizer.step()
    return output, loss.data[0]

def predict(x):
    hidden = rnn.initHidden(x.shape[0])
    x=Variable(torch.FloatTensor(x))
    for i in range(5):
        output,hidden = rnn(x[:,i,:], hidden)
    output=finalLayer(output)
    return output
def evaluate(x,y):
    loss=criterion(predict(x),Variable(torch.FloatTensor(y)))
    return loss
    

nb_epoch=2

for i in range(nb_epoch*x.shape[0]//minibatch_size):
    indices = np.random.randint(testx.shape[0],size=minibatch_size)
    output,loss=train(x[indices],y[indices])
    #print (loss)

#a=np.random.uniform(size=(3,5,3))
#print (a, predict(a).data)

print (evaluate(testx,testy).data)
