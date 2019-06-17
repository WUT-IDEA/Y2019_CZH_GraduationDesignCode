# coding: utf-8




import time
import keras
from keras.layers import *
from keras.models import Model
import numpy as np
from DataGenerator import data_generator_w
from losses import rank_hinge_loss
from config import MineBasicConfig
from Customhistory import LossHistory
import numpy as np


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


config = MineBasicConfig()

# word level
question_winput = Input(shape=(config.sentence_len, ), dtype=np.int32)
subject_winput = Input(shape=(config.sub_len, ), dtype=np.int32)
rel_winput = Input(shape=(config.rel_len, ), dtype=np.int32)
embedding_maxtrix = np.loadtxt("../../data/mini_glove.txt", dtype=np.float32)
word_embedding = Embedding(input_dim=config.nums_words,
                           output_dim=config.wordembdim,
                           mask_zero=True,
                           weights=[embedding_maxtrix],
                           trainable=False)

qw_embed = word_embedding(question_winput)
sw_embed = word_embedding(subject_winput)
rw_embed = word_embedding(rel_winput)

wordlstm = LSTM(units=config.encdim)
resp_question = wordlstm(qw_embed)
resp_subject = wordlstm(sw_embed)
resp_rel = wordlstm(rw_embed)

dropout_rate = 0.1

d1_question = Dense(300, activation='relu')(resp_question)
resp_d_question = Dropout(rate=dropout_rate)(d1_question)

d1_subject = Dense(300, activation='relu')(resp_subject)
resp_d_subject = Dropout(rate=dropout_rate)(d1_subject)

d1_rel = Dense(300, activation='relu')(resp_rel)
resp_d_rel = Dropout(rate=dropout_rate)(d1_rel)

score_qs_word = Dot(axes=[1, 1], normalize=True)([resp_d_question, resp_d_subject])
score_qr_word = Dot(axes=[1, 1], normalize=True)([resp_d_question, resp_d_rel])

score_qs_word = Dense(units=1)(score_qs_word)
score_qr_word = Dense(units=1)(score_qr_word)


model = Model(inputs=[question_winput, subject_winput, rel_winput],
              outputs=[score_qs_word, score_qr_word])
qs_model = Model(inputs=[question_winput, subject_winput],
                 output=score_qs_word)
qr_model = Model(inputs=[question_winput, rel_winput],
                 output=score_qr_word)
model.compile(optimizer='adam',
              loss=rank_hinge_loss)
print(model.summary())

batch_size = 512
steps = 75504 * 20 * 2 // batch_size
v_steps = 10778 * 2 * 2 // batch_size

logpath = "logs_wordbase_100.txt"
modelpath = "wordbase_100"
history = LossHistory(logpath, modelpath)
history.set_model(model)


qwi = "../train/qw2.txt"
swi = "../train/sw2.txt"
rwi = "../train/rw2.txt"

# validation

v_qwi = "../valid/qw2.txt"
v_swi = "../valid/sw2.txt"
v_rwi = "../valid/rw2.txt"

start_time = time.strftime('%Y.%m.%d  %H:%M:%S', time.localtime(time.time()))

model.fit_generator(generator=data_generator_w(qwi, swi, rwi, batch_size),
                    steps_per_epoch=steps,
                    validation_data = data_generator_w(v_qwi, v_swi, v_rwi, batch_size),
                    validation_steps = v_steps,
                    callbacks=[history],
                    epochs=50,
                    shuffle=False)
qs_model.save(modelpath + "/qs.h5")
qr_model.save(modelpath + "/qr.h5")
end_time = time.strftime('%Y.%m.%d  %H:%M:%S', time.localtime(time.time()))
print("start time: {}".format(start_time))
print("end time: {}".format(end_time))
