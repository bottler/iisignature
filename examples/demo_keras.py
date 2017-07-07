#This example shows a very simple Keras model which does nothing interesting,
#but has a layer which uses Sig from iisignature.
#It trains, giving some confidence in the derivatives.

import os, sys
#os.environ["KERAS_BACKEND"]="tensorflow"
#os.environ["THEANO_FLAGS"]="mode=DebugMode,device=cpu,optimizer_excluding=local_shape_to_shape_i"
import numpy as np, keras
import keras.models, keras.layers.recurrent, keras.layers.core
import keras.backend as K
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
if K.backend() == "theano":
    from iisignature_theano import Sig
if K.backend() == "tensorflow":
    from iisignature_tensorflow import Sig
import iisignature

sig_level=2
siglen = iisignature.siglength(2,sig_level)
nSamples=21
xDim=55

X=np.zeros((nSamples,xDim),dtype="float32")
Y=np.ones((nSamples,),dtype="float32")

m=keras.models.Sequential()

m.add(keras.layers.core.Dense(106, input_shape=(xDim,)))
m.add(keras.layers.core.Reshape((53,2)))
m.add(keras.layers.core.Lambda(lambda x:Sig(x,2),output_shape=(siglen,)))
m.add(keras.layers.core.Reshape((siglen,1)))
m.add(keras.layers.pooling.AveragePooling1D(siglen))
m.add(keras.layers.core.Flatten())
m.compile(loss='mse',optimizer="sgd")
m.summary()

m.fit(X,Y, epochs=1000)

#https://github.com/tensorflow/tensorflow/issues/3388
if K.backend() == "tensorflow":
    K.clear_session()
