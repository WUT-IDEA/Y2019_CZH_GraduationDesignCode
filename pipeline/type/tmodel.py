# -*- coding: utf-8 -*-


from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import keras
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))

from keras.layers import Input, Dense, Embedding, LSTM, Bidirectional
from keras.models import Model
import numpy as np
import datetime


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
    q = []
    t = []
    y = []
    with open(filepath, 'r') as reader:
        for line in reader:
            parts = line.strip().split("\t")
            q.append(parts[0].split(" "))
            t.append(parts[1].split(" "))
            y.append(parts[2].strip())
    q = np.asarray(q, dtype='int32')
    y = np.asarray(y, dtype="int32")
    t = np.asarray(t, dtype='int32')
    return q, t, y

# Question
embedding_matrix_q = np.loadtxt("../../data/glove_test.txt", dtype=np.float32)

print(embedding_matrix_q.shape)
EMBEDDING_DIM_Q=300
MAX_SEQUENCE_LENGTH_Q=24

# define model  58968
sequence_input_q=Input(shape=(MAX_SEQUENCE_LENGTH_Q,), dtype='int32')
embedding_layer_q=Embedding(input_dim=62957,
                            output_dim=EMBEDDING_DIM_Q,
                            weights=[embedding_matrix_q],
                            input_length=MAX_SEQUENCE_LENGTH_Q,
                            trainable=False)
embedded_sequences_q=embedding_layer_q(sequence_input_q)
q_bilstm=Bidirectional(LSTM(100))(embedded_sequences_q)

# subject type

embedding_matrix_t=np.loadtxt("data/glove_type.txt")
print(embedding_matrix_t.shape)
EMBEDDING_DIM_T=300
MAX_SEQUENCE_LENGTH_T=6

# define model
sequence_input_t=Input(shape=(MAX_SEQUENCE_LENGTH_T,), dtype='int32')
embedding_layer_t=Embedding(input_dim=1053,
                            output_dim=EMBEDDING_DIM_T,
                            weights=[embedding_matrix_t],
                            input_length=MAX_SEQUENCE_LENGTH_T,
                            mask_zero=True,
                            trainable=False)
embedded_sequences_t=embedding_layer_t(sequence_input_t)

t_lstm=Bidirectional(LSTM(100))(embedded_sequences_t)
from keras.layers import concatenate
concatenatecon_layer=concatenate([q_bilstm, t_lstm],axis=-1)
dense1=Dense(100,activation="sigmoid")(concatenatecon_layer)
output=Dense(1,activation="sigmoid")(dense1)
# output=Dense(1,activation="sigmoid")(concatenatecon_layer)
model=Model(inputs=[sequence_input_q,sequence_input_t],outputs=output)
model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])
print(model.summary())
input_q, input_t, y = read_byline("training_data/train_data.txt")

BATCH_SIZE=100
EPOCHS=60
history = LossHistory("log_t_binary.txt", "t_binary_model")
start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
model.fit(x=[input_q,input_t],
          y=y,
          batch_size=BATCH_SIZE,
          callbacks=[history],
          epochs=EPOCHS)
model.save("e_model.h5")
endTime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("startTime:"+start_time)
print("endTime:"+endTime)