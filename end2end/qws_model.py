# coding: utf-8




import time
import keras
from keras.layers import *
from keras.models import Model
import numpy as np
from DataGenerator import data_generator_base
from losses import rank_hinge_loss
from config import MineBasicConfig
from Customhistory import LossHistory
import numpy as np
from seq_self_attention import SeqSelfAttention


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


config = MineBasicConfig()

# 全部采用self attention
class CustomFlatten(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(CustomFlatten, self).__init__(**kwargs)

    def compute_mask(self, inputs, mask=None):
        if mask==None:
            return mask
        return K.batch_flatten(mask)

    def call(self, inputs, mask=None):
        return K.batch_flatten(inputs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], np.prod(input_shape[1:]))


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

wordlstm = LSTM(units=config.encdim, return_state=True, return_sequences=True)
qhs, qh, qc = wordlstm(qw_embed)
shs, sh, sc = wordlstm(sw_embed)
rhs, rh, rc = wordlstm(rw_embed)
print("qhs:")
print(qhs)
attention = SeqSelfAttention(attention_activation='sigmoid')
at_q = attention(qhs)
print(at_q)
resp_question = CustomFlatten()(at_q)
print(resp_question)

at_s = SeqSelfAttention(attention_activation='sigmoid')(shs)
resp_subject = CustomFlatten()(at_s)
at_r = SeqSelfAttention(attention_activation='sigmoid')(rhs)
resp_rel = CustomFlatten()(at_r)

dropout_rate = 0.1

d1_question = Dense(300, activation='relu')(resp_question)
resp_d_question = Dropout(rate=dropout_rate)(d1_question)

d1_subject = Dense(300, activation='relu')(resp_subject)
resp_d_subject = Dropout(rate=dropout_rate)(d1_subject)

d1_rel = Dense(300, activation='relu')(resp_rel)
resp_d_rel = Dropout(rate=dropout_rate)(d1_rel)

score_qs_word = Dot(axes=[1, 1], normalize=True)([resp_d_question, resp_d_subject])
score_qr_word = Dot(axes=[1, 1], normalize=True)([resp_d_question, resp_d_rel])

# whole
score = Concatenate(axis=1)([score_qs_word, score_qr_word])
output = Dense(1)(score)

model = Model(inputs=[question_winput, subject_winput, rel_winput],
              output=output)
qs_model = Model(inputs=[question_winput, subject_winput],
                 output=score_qs_word)
qr_model = Model(inputs=[question_winput, rel_winput],
                 output=score_qr_word)
model.compile(optimizer='adam',
              loss=rank_hinge_loss)
print(model.summary())

batch_size = 512
steps = 75504 * 10 * 2 // batch_size
v_steps = 10778 * 2 * 2 // batch_size

logpath = "logs_baseat.txt"
modelpath = "baseat"
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

model.fit_generator(generator=data_generator_base(qwi, swi, rwi, batch_size),
                    steps_per_epoch=steps,
                    validation_data = data_generator_base(v_qwi, v_swi, v_rwi, batch_size),
                    validation_steps = v_steps,
                    callbacks=[history],
                    epochs=40,
                    shuffle=False)
qs_model.save(modelpath + "/qs.h5")
qr_model.save(modelpath + "/qr.h5")
end_time = time.strftime('%Y.%m.%d  %H:%M:%S', time.localtime(time.time()))
print("start time: {}".format(start_time))
print("end time: {}".format(end_time))

keras.losses.categorical_crossentropy()