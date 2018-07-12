from keras.layers  import CuDNNGRU
from keras.layers import Bidirectional
from keras.layers import CuDNNLSTM, TimeDistributed, Dropout
from keras import Sequential
from data_generator import vis_train_features
from IPython.display import Markdown, display
from data_generator import vis_train_features, plot_raw_audio
from IPython.display import Audio
import numpy as np
from data_generator import AudioGenerator
from keras import backend as K
from utils import int_sequence_to_text
from IPython.display import Audio
from keras.models import load_model
import pickle

from keras.models import load_model
import pickle as pickle

from keras.models import Model
from keras.layers import (Input, Lambda)
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint   
import os
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, Conv2D, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM)
from continue_train import train_model


def  bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    drop=0.4  #0.7
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    layer1 = Bidirectional(CuDNNLSTM(units=units,return_sequences=True), merge_mode='concat')(input_data)
    drop1 = Dropout(drop)(layer1)
    bn_rnn1 =BatchNormalization()(drop1)
    #drop1 = Dropout(0.7)(bn_rnn1)
    layer2 = Bidirectional(CuDNNLSTM(units=units, return_sequences= True), merge_mode='concat')(bn_rnn1)
    drop2 = Dropout(drop)(layer2)
    bn_rnn2 =BatchNormalization()(drop2)
    #drop2 = Dropout(0.7)(bn_rnn2)
    layer3 = Bidirectional(CuDNNLSTM(units=units, return_sequences= True), merge_mode='concat')(bn_rnn2)
    drop3 = Dropout(drop)(layer3)
    bn_rnn3 =BatchNormalization()(drop3)
    layer4 = Bidirectional(CuDNNLSTM(units=units, return_sequences= True), merge_mode='concat')(bn_rnn3)
    drop4 = Dropout(drop)(layer4)
    bn_rnn4 =BatchNormalization()(drop4)
    layer5 = Bidirectional(CuDNNLSTM(units=units, return_sequences= True), merge_mode='concat')(bn_rnn4)
    drop5 = Dropout(drop)(layer5)
    bn_rnn5 =BatchNormalization()(drop5)
    layer6 = Bidirectional(CuDNNLSTM(units=units, return_sequences= True), merge_mode='concat')(bn_rnn5)
    drop6 = Dropout(drop)(layer6)
    bn_rnn6 =BatchNormalization()(drop6)
    layer7 = Bidirectional(CuDNNLSTM(units=units, return_sequences= True), merge_mode='concat')(bn_rnn6)
    drop7 = Dropout(drop)(layer7)
    bn_rnn7 =BatchNormalization()(drop7)
    layer8 = Bidirectional(CuDNNLSTM(units=units, return_sequences= True), merge_mode='concat')(bn_rnn7)
    drop8 = Dropout(drop)(layer8)
    bn_rnn8 =BatchNormalization()(drop8)
    
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn8)
   
    # Specify the model
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


model = bidirectional_rnn_model(input_dim=161, # change to 13 if you would like to use MFCC features
                                  units=512+32)

print('load Model')
model.load_weights('results/model_20.h5')
data_gen = AudioGenerator()
print("Load file")    
audio_path = 'output.wav'
data_point = data_gen.normalize(data_gen.featurize(audio_path))

print("Start prediction")

#input_to_softmax.load_weights(model_path)
prediction = model.predict(np.expand_dims(data_point, axis=0), batch_size=1)
output_length = [model.output_length(data_point.shape[0])] 
pred_ints = (K.eval(K.ctc_decode( prediction, output_length)[0][0])+1).flatten().tolist()

print(prediction)
print(output_length)


print(pred_ints) 
print('Predicted transcription:\n' + '\n' + ''.join(int_sequence_to_text(pred_ints)))

