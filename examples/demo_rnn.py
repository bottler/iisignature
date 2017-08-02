#A trivial demonstration of the RecurrentSig layer from iisignature_recurrent_keras.py
#relies on keras 2

import os
#os.environ["THEANO_FLAGS"]="floatX=float32,device=cpu,optimizer=fast_compile"
#os.environ["THEANO_FLAGS"]="floatX=float32,device=cpu,mode=DebugMode"
#os.environ["THEANO_FLAGS"]="floatX=float32,device=gpu0,force_device=True,cxx=g++-4.8,nvcc.flags=-D_FORCE_INLINES,nvcc.compiler_bindir=/usr/bin/g++-4.8"
#os.environ["THEANO_FLAGS"]="floatX=float32,device=gpu0,force_device=True,cxx=g++-4.8,nvcc.flags=-D_FORCE_INLINES,nvcc.compiler_bindir=/usr/bin/g++-4.8,base_compiledir=/run/user/1001/theano"
os.environ["THEANO_FLAGS"]="floatX=float32,device=cpu,force_device=True,mode=DebugMode,DebugMode.check_finite=False"
os.environ["THEANO_FLAGS"]="floatX=float32,device=cpu,force_device=True"
#os.environ["THEANO_FLAGS"]="floatX=float32,device=cpu,force_device=True,mode=NanGuardMode,exception_verbosity=high,NanGuardMode.inf_is_error=False,NanGuardMode.big_is_error=False,NanGuardMode.action=warn,optimizer=fast_compile"
os.environ["KERAS_BACKEND"]="theano"
os.environ["KERAS_BACKEND"]="tensorflow"

import numpy, sys
 
#add the parent directory, so we find our iisignature build if it was built --inplace
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from iisignature_recurrent_keras import RecurrentSig
import keras.models, keras.layers.recurrent, keras.layers.core
from keras.layers.recurrent import SimpleRNN, LSTM


m=keras.models.Sequential()
#a few possible networks here.
#using relu with RecurrentSig sometimes gets Nans

m.add(RecurrentSig(5,sig_level=2,input_shape=(None,3),return_sequences=False, use_signatures = True, output_signatures = False, activation="tanh",train_time_lapse=True))

#m.add(RecurrentSig(5,input_shape=(5,3),return_sequences=True, use_signatures = True, output_signatures = False, activation="relu"))
#m.add(RecurrentSig(6,return_sequences=False,activation="relu"))

#m.add(LSTM(5,input_shape=(5,3),return_sequences=False))

#m.add(LSTM(5,input_shape=(5,3),return_sequences=True))
#m.add(LSTM(6,return_sequences=False))

#m.add(keras.layers.core.Flatten(input_shape=(5,3)))
#m.add(keras.layers.core.Dense(1000,activation="relu"))

m.add(keras.layers.core.Dense(1, activation="sigmoid"))
op = keras.optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
m.compile(loss='mse', optimizer=op)#metrics = accuracy
m.summary()

#The task here for the network to learn is very easy - the average of two of the inputs
x = numpy.random.uniform(size=(2311,5,3))
y = (x[:,1,1] + x[:,3,2])/2 # The output is a number between 0 and 1, so matches sigmoid activation of the top layer
testx = x[2000:,:,:]
testy = y[2000:]
x=x[:2000,:,:]
y=y[:2000]

#a=numpy.random.uniform(size=(3,5,3))
#print (m.predict(a))
m.fit(x,y,epochs=10,shuffle=0)
print (m.evaluate(testx,testy,verbose=0))
