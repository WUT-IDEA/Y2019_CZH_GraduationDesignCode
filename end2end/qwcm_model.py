# coding: utf-8





import keras
import time
from keras.layers import *
from keras.models import Model
import numpy as np
from DataGenerator import data_generator_y3
from losses import rank_hinge_loss
from config import MineBasicConfig
from Customhistory import LossHistory

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


config = MineBasicConfig()

# word level
question_winput = Input(shape=(config.sentence_len, ), dtype=np.int32)
subject_winput = Input(shape=(config.sub_len, ), dtype=np.int32)
rel_winput = Input(shape=(config.rel_len, ), dtype=np.int32)

# character level
qc_input_layer = Input(shape=(config.qu_maxlen, config.word_len), dtype=np.int32)
sc_input_layer = Input(shape=(config.sub_len, config.word_len), dtype=np.int32)
rc_input_layer = Input(shape=(config.rel_len, config.word_len), dtype=np.int32)


embedding_maxtrix = np.loadtxt("../../data/mini_glove.txt", dtype=np.float32)
word_embedding = Embedding(input_dim=config.nums_words,
                           output_dim=config.wordembdim,
                           mask_zero=True,
                           weights=[embedding_maxtrix],
                           trainable=False)

qw_embedding = word_embedding(question_winput)
sw_embedding = word_embedding(subject_winput)
rw_embedding = word_embedding(rel_winput)

q_char_embedding = Embedding(input_dim=config.nums_chars,
                    output_dim=50,
                    mask_zero=True,
                    trainable=True
                    )
s_char_embedding = Embedding(input_dim=config.nums_chars,
                    output_dim=50,
                    mask_zero=True,
                    trainable=True
                    )
r_char_embedding = Embedding(input_dim=config.nums_chars,
                    output_dim=50,
                    mask_zero=True,
                    trainable=True
                    )
qc_embd_layer = q_char_embedding(qc_input_layer)
sc_embd_layer = s_char_embedding(sc_input_layer)
rc_embd_layer = r_char_embedding(rc_input_layer)

max_word_len = config.qu_maxlen
char_dict_len = config.word_len
char_hidden_layer = GRU(units=100,
                        input_shape=(max_word_len, char_dict_len),
                        return_sequences=False,
                        return_state=False)

# char_embd_layer = TimeDistributed(layer=char_hidden_layer, name='Embedding_Char')
qc_embd_layer = TimeDistributed(layer=char_hidden_layer, name='question_c_embedding_char')(qc_embd_layer)
sc_embd_layer = TimeDistributed(layer=char_hidden_layer, name='subject_c_embedding_char')(sc_embd_layer)
rc_embd_layer = TimeDistributed(layer=char_hidden_layer, name='relation_c_embedding_char')(rc_embd_layer)


q_embd_layer = Concatenate(name='q_embedding')([qw_embedding, qc_embd_layer])
s_embd_layer = Concatenate(name='s_embedding')([sw_embedding, sc_embd_layer])
r_embd_layer = Concatenate(name='r_embedding')([rw_embedding, rc_embd_layer])
print(qc_embd_layer)
print(s_embd_layer)
print(r_embd_layer)

wordlstm = LSTM(units=config.encdim)
resp_question = wordlstm(q_embd_layer)
resp_subject = wordlstm(s_embd_layer)
resp_rel = wordlstm(r_embd_layer)

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

# whole
# score = Concatenate(axis=1)([score_qs_word, score_qr_word])
# output = Dense(1)(score)

model = Model(inputs=[question_winput, subject_winput, rel_winput, qc_input_layer, sc_input_layer, rc_input_layer],
              outputs=[score_qs_word, score_qr_word])

qs_model = Model(inputs=[question_winput, subject_winput, qc_input_layer, sc_input_layer],
                 output=score_qs_word)
qr_model = Model(inputs=[question_winput, rel_winput, qc_input_layer, rc_input_layer],
                 output=score_qr_word)

model.compile(optimizer='adam',
              loss=rank_hinge_loss)
print(model.summary())
batch_size = 512
steps = 75504 * 10 * 2 // batch_size
v_steps = 10778 * 2 * 2 // batch_size

logpath = "logs_multi3.txt"
modelpath = "model_multi3"
history = LossHistory(logpath, modelpath)
history.set_model(model)


qwi = "../train/qw2.txt"
swi = "../train/sw2.txt"
rwi = "../train/rw2.txt"
qwci = "../train/qwc2.txt"
swci = "../train/swc2.txt"
rwci = "../train/rwc2.txt"
# validation

v_qwi = "../valid/qw2.txt"
v_swi = "../valid/sw2.txt"
v_rwi = "../valid/rw2.txt"

v_qwci = "../valid/qwc2.txt"
v_swci = "../valid/swc2.txt"
v_rwci = "../valid/rwc2.txt"

start_time = time.strftime('%Y.%m.%d  %H:%M:%S', time.localtime(time.time()))
model.fit_generator(generator=data_generator_y3(qwi, swi, rwi, qwci, swci, rwci, batch_size),
                    steps_per_epoch=steps,
                    validation_data = data_generator_y3(v_qwi, v_swi, v_rwi, v_qwci, v_swci, v_rwci, batch_size),
                    validation_steps = v_steps,
                    callbacks=[history],
                    epochs=40,
                    shuffle=False)
qs_model.save(modelpath+"/qs_model.h5")
qr_model.save(modelpath+"/qr_model.h5")
end_time = time.strftime('%Y.%m.%d  %H:%M:%S', time.localtime(time.time()))
print("start time: {}".format(start_time))
print("end time: {}".format(end_time))

