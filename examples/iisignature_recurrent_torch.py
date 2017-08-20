import torch
from torch.autograd import Variable
import torch.nn as nn
#from torch.nn.modules.module import Module

from iisignature_torch import SigJoin
import iisignature
import math

#This is a pytorch example of a recurrent layer which uses Signature internally.
#It is not identical to the example in iisignature_recurrent_keras, e.g. in that there are
#more bias parameters, no activation, and less control over the initialisation.
#No assertion is made that this model is a good idea, or that this code is idiomatic pytorch.

class RecurrentSig(nn.Module):
    def __init__(self, input_size, n_units, sig_level=2, train_time_lapse=True,
                 initial_time_lapse=0.1, output_signatures=False,
                 use_signatures=True):
        super(RecurrentSig, self).__init__()
        self.sig_level=sig_level
        self.sigsize=iisignature.siglength(2,sig_level)
        self.n_units=n_units
        self.units=n_units*(self.sigsize+1) if output_signatures else n_units
        self.train_time_lapse = train_time_lapse
        #self.initial_time_lapse = initial_time_lapse
        self.output_signatures = output_signatures
        self.use_signatures = use_signatures

        saved_data_length = self.n_units * (1+self.sigsize if self.use_signatures else 1)
        self.W=nn.Linear(input_size,self.n_units)
        self.U=nn.Linear(saved_data_length, self.n_units)
        if self.train_time_lapse:
            self.log_timelapse=nn.parameter.Parameter(torch.FloatTensor([math.log(initial_time_lapse)]))
        else:
            self.time_lapse = Variable(torch.FloatTensor([initial_time_lapse]))
    def initHidden(self, batch_size):
        lengths = (self.n_units*self.sigsize,self.n_units)
        a=Variable(torch.zeros(batch_size, lengths[0]))
        b=Variable(torch.zeros(batch_size, lengths[1]))
        return a,b
    def forward(self, input, states):
        prev_sigs=states[0]#(batch,n_units*sigsize)
        prev_sigs_=prev_sigs.view(-1, self.sigsize)#(batch*n_units,sigsize)
        prev_states=states[1]
        h = self.W(input)
        prev_states_as_memory = (torch.cat([prev_sigs,prev_states],1) if
                                 self.use_signatures else prev_states)
        
        raw_output = h+self.U(prev_states_as_memory)

        displacements=(raw_output-prev_states).view(-1,1)
        time_lapse = torch.exp(self.log_timelapse) if self.train_time_lapse else self.time_lapse 
        sigs = SigJoin(prev_sigs_,displacements,self.sig_level,time_lapse)
        sigs = sigs.view(-1,self.n_units*self.sigsize)
        if self.output_signatures:
            output = torch.cat([sigs,raw_output],1)
        else:
            output = raw_output
        return output,[sigs,raw_output]

if __name__=="__main__":
    #Just a sanity check
    r=RecurrentSig(1,1)
    hidden=r.initHidden(3)
    inp = Variable(torch.randn(3, 1))
    result,hidden = r(inp,hidden)
    print(result)
