# -*- coding: utf-8 -*-


from __future__ import print_function


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
set_session(tf.Session(config=config))


import codecs
from keras.layers import Input, Dense, Embedding, LSTM, Bidirectional, Dropout
from keras.models import Model
import numpy as np
import keras
import os

class LossHistory(keras.callbacks.Callback):
    def __init__(self, logpath, modelpath):
        super().__init__()
        self.logpath = logpath
        self.modelpath = modelpath
        if not os.path.exists(modelpath):
            os.mkdir(modelpath)
    def set_model(self, model):
        self.model = model
        self.writer = open(self.logpath, "w")
    def on_epoch_end(self, epoch, logs=None):
        #if epoch == 0 or (epoch + 1) % 5 == 0:
        self.model.save("{}/model_{}.h5".format(self.modelpath, epoch+1))
        # self.writer.write("epoch {}, loss:{},  valid_loss:{}\n".format(epoch+1, logs['loss'], logs['val_loss']))
        self.writer.write("epoch {}, loss:{}\n".format(epoch + 1, logs['loss']))

# read file by line
def read_byline(filepath):
    lines = []
    with codecs.open(filepath, 'r', encoding='utf-8') as reader:
        for line in reader:
            lines.append(line.strip())
    return lines

def generate_arrays_from_file(path,batch_size):
    while 1:
        with codecs.open(path,'r') as f:
            cnt=0
            Q=[]
            R=[]
            Y=[]
            for line in f:
                parts=line.strip().split("\t")
                Q.append(parts[0].split(' '))
                R.append(parts[1].split(" "))
                Y.append(parts[2].strip())
                cnt+=1
                if cnt==batch_size:
                    cnt=0
                    yield ([np.array(Q,dtype='int32'),np.array(R,dtype='int32')],np.array(Y,dtype="int32"))
                    Q=[]
                    R=[]
                    Y=[]

# Question
# embedding_matrix_q=np.loadtxt("../word2vec/embedding_maxtrix.txt")
embedding_matrix = np.loadtxt("../../data/glove_test.txt", dtype=np.float32)
print(embedding_matrix.shape)
EMBEDDING_DIM_Q=300
MAX_SEQUENCE_LENGTH_Q=24

# define model
sequence_input_q=Input(shape=(MAX_SEQUENCE_LENGTH_Q,), dtype='int32')
embedding_layer_q=Embedding(input_dim=62957,
                            output_dim=EMBEDDING_DIM_Q,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH_Q,
                            trainable=False)
embedded_sequences_q=embedding_layer_q(sequence_input_q)
#RNN_Q=recurrent.SimpleRNN
#rnn_q=RNN_Q(100)(embedded_sequences_q)
#q_bilstm=Bidirectional(LSTM(100))(embedded_sequences_q)
q_bigru=Bidirectional(LSTM(100))(embedded_sequences_q)
# Relation

# embedding_matrix_r=np.loadtxt("relation_words_embedding_maxtrix.txt")
# print(embedding_matrix_r.shape)
EMBEDDING_DIM_R = 300
MAX_SEQUENCE_LENGTH_R = 17

# define model
sequence_input_r=Input(shape=(MAX_SEQUENCE_LENGTH_R,), dtype='int32')
embedding_layer_r=Embedding(input_dim=62957,
                            output_dim=EMBEDDING_DIM_R,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH_R,
                            trainable=False)
embedded_sequences_r=embedding_layer_r(sequence_input_r)

r_bigru=Bidirectional(LSTM(100))(embedded_sequences_r)
from keras.layers import concatenate
concatenatecon_layer=concatenate([q_bigru,r_bigru],axis=-1)
dropout1=Dropout(0.1)(concatenatecon_layer)
dense1=Dense(50,activation="sigmoid")(dropout1)
output=Dense(1,activation="sigmoid")(dense1)
model=Model(inputs=[sequence_input_q,sequence_input_r],outputs=output)
model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])
print(model.summary())

BATCH_SIZE = 100
data_path="train_data2.txt"
import datetime
start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
history = LossHistory("log_lstm_binary.txt", "binary_lstm_model")
model.fit_generator(generate_arrays_from_file(data_path,batch_size=BATCH_SIZE),
                    epochs=60,
                    steps_per_epoch=732712 // BATCH_SIZE,
                    callbacks=[history])
endTime=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("startTime:"+start_time)
print("endTime:"+endTime)