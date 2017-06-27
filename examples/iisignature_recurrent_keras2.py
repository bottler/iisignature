import keras.layers.recurrent
from keras.engine import InputSpec
from keras import backend as K
from keras import initializers, activations
import iisignature
from iisignature_theano import SigJoin
import math

#Consider initializing so that state is constant initially or something.
#-perhaps a highway
#add dropout, regularizers

class RecurrentSig(keras.layers.recurrent.Recurrent):
    '''
    A recurrent layer like keras's SimpleRNN or LSTM, which includes the signatures of the graph
    of each state against time.

    n_units: the number of units, this is not the number of outputs if output_signatures is Triue

    sig_level: the depth of the signature

    initial_time_lapse: the distance in the time dimension between previous values when 
    calculating a 2d signature

    train_time_lapse: whether to train the time lapse, this doesn't work yet because
    SigJoinGrad doesn't return a gradient wrt the time lapse.

    output_signatures: whether the signatures as well as the current hidden state are 
    output to succeeding layers
   
    use_signatures: whether the new state can depend on the value of the signature. 
    Note that this layer is just like a simple RNN if use_signatures 
    and output_signatures are both false.

    kernel_initializer: the initialisation of the map from input to state

    recurrent_initializer: the initialisation of the matrix from state to state

    activation: the activation applied to the output values of the state.
    '''
    def __init__(self, n_units,
                 sig_level=2,
                 train_time_lapse =False,#doesn't work yet - no gradient produ
                 initial_time_lapse=0.1,
                 output_signatures = False,#whether the output includes the signatures
                 use_signatures = True, #whether each new state can depend on the signature
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='he_normal',#could be good to use 'orthogonal' if not use_signatures
                 activation='tanh',#not applied to signature elements
                 **kwargs):
        self.sig_level = sig_level
        self.sigsize = iisignature.siglength(2,sig_level)
        self.n_units = n_units #like output_dim
        self.units = n_units*(self.sigsize+1) if output_signatures else n_units
        self.train_time_lapse = train_time_lapse
        self.initial_time_lapse = initial_time_lapse
        self.output_signatures = output_signatures
        self.use_signatures = use_signatures
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.activation = activations.get(activation)
        super(RecurrentSig, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.input_dim = input_shape[2]
        self.states = [None,None] #?
        self.W = self.add_weight((self.input_dim, self.n_units),
                                 name='W', initializer=self.kernel_initializer)
        saved_data_length = self.n_units * (
                                   1+self.sigsize if self.use_signatures else 1)
        self.U = self.add_weight((saved_data_length, self.n_units),
                                 name='U',initializer=self.recurrent_initializer)
        self.b = K.zeros((self.n_units,), name='{}_b'.format(self.name))
        
        self.trainable_weights = [self.W, self.U, self.b]
        if self.train_time_lapse:
            log_timelapse = K.variable(math.log(self.initial_time_lapse),
                                       name='{}_log_time_lapse'.format(self.name))
            self.trainable_weights.append(log_timelapse)
            self.time_lapse = K.exp(log_timelapse)
        else:
            self.time_lapse = self.initial_time_lapse

    def step(self, x, states):
        prev_sigs=states[0]#(batch,n_units*sigsize)
        prev_sigs_=K.reshape(prev_sigs,(-1,self.sigsize))#(batch*n_units,sigsize)
        prev_states=states[1]
        h = K.dot(x, self.W) + self.b
        prev_states_as_memory = (K.concatenate([prev_sigs,prev_states],1) if
                                 self.use_signatures else prev_states)
        raw_output = self.activation(h + K.dot(prev_states_as_memory, self.U))
        displacements=K.reshape(raw_output-prev_states,(-1,1))
        sigs = SigJoin(prev_sigs_,displacements,self.sig_level,self.time_lapse)
        sigs = K.reshape(sigs,(-1,self.n_units*self.sigsize))
        activated_output_units = self.activation(raw_output)
        if self.output_signatures:
            output = K.concatenate([sigs,activated_output_units],1)
        else:
            output = activated_output_units
        return output,[sigs,raw_output]

    def get_initial_states(self, x):
        # x has shape (samples, timesteps, input_dim)
        # build all-zero tensors of shape (samples, whatever)
        initial_state = K.zeros_like(x)  
        initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
        initial_state = K.expand_dims(initial_state)  # (samples, 1)
        lengths = (self.n_units*self.sigsize,self.n_units)
        initial_states = [K.tile(initial_state, [1, i]) for i in lengths]  # (samples, i)
        return initial_states
    
    def get_config(self):
        config = {'sigsize': self.sigsize,
                  'sig_level': self.sig_level,
                  'train_time_lapse': self.train_time_lapse,
                  'initial_time_lapse': self.initial_time_lapse,
                  'n_units': self.n_units,
                  'units': self.units,
                  'output_signatures' : self.output_signatures,
                  'use_signatures' : self.use_signatures,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'activation': self.activation.__name__}
        base_config = super(RecurrentSig, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
