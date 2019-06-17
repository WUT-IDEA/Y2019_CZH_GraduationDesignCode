# coding: utf-8

from keras.layers import Input, Embedding, Dense, Bidirectional, Dropout, Dot, Lambda, Concatenate, SimpleRNN, LSTM, GRU
from keras.models import Model
from keras.losses import *

def rank_hinge_loss(y_true, y_pred):
    margin = 1.
    # output_shape = K.int_shape(y_pred)
    y_pos = Lambda(lambda a: a[::2, :], output_shape=(1,))(y_pred)
    y_neg = Lambda(lambda a: a[1::2, :], output_shape=(1,))(y_pred)
    loss = K.maximum(0., margin + y_neg - y_pos)
    return K.mean(loss)

class RelationMatch(object):
    def __init__(self, config):
        self.q_len = config.q_len
        self.r_len = config.r_len
        self.n_vocab = config.n_vocab
        self.n_embed = config.n_embed
        self.embed_mat = config.embed_mat
        self.keep_prob = config.keep_prob
        self.keep_prob_rnn = config.keep_prob_rnn
        self.category_rnn = config.category_rnn
        self.n_rnn = config.n_rnn
        self.category_loss = config.category_loss
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.model = None

    def build(self):
        input_q = Input(shape=(self.q_len, ), dtype='int32')
        input_r = Input(shape=(self.r_len, ), dtype='int32')
        if self.embed_mat is None:
            word_embedding = Embedding(input_dim=self.n_vocab,
                                       output_dim=self.n_embed,
                                       mask_zero=True,
                                       trainable=True)
        else:
            print("using glove embedding~")
            word_embedding = Embedding(input_dim=self.n_vocab,
                                       output_dim=self.n_embed,
                                       weights=[self.embed_mat],
                                       mask_zero=True,
                                       trainable=False)
        # dropout = Dropout(self.keep_prob)(word_embedding)
        q_embedding = word_embedding(input_q)
        r_embedding = word_embedding(input_r)
        if self.category_rnn == "rnn":
            q_rep = Bidirectional(SimpleRNN(units=self.n_rnn,
                                            dropout=self.keep_prob_rnn,
                                            recurrent_dropout=self.keep_prob_rnn))(q_embedding)
            r_rep = Bidirectional(SimpleRNN(units=self.n_rnn,
                                            dropout=self.keep_prob_rnn,
                                            recurrent_dropout=self.keep_prob_rnn))(r_embedding)
            print("rnn == rnn")
        elif self.category_rnn == "lstm":
            q_rep = Bidirectional(LSTM(units=self.n_rnn,
                                       dropout=self.keep_prob_rnn,
                                       recurrent_dropout=self.keep_prob_rnn))(q_embedding)
            r_rep = Bidirectional(LSTM(units=self.n_rnn,
                                       dropout=self.keep_prob_rnn,
                                       recurrent_dropout=self.keep_prob_rnn))(r_embedding)
            print("rnn == lstm")
        elif self.category_rnn == "gru":
            q_rep = Bidirectional(GRU(units=self.n_rnn,
                                      dropout=self.keep_prob_rnn,
                                      recurrent_dropout=self.keep_prob_rnn))(q_embedding)
            r_rep = Bidirectional(GRU(units=self.n_rnn,
                                      dropout=self.keep_prob_rnn,
                                      recurrent_dropout=self.keep_prob_rnn))(r_embedding)
            print("rnn == gru")

        else:
            print("not find! error!")
            exit()
        if self.category_loss == 'ranking':
            # qd = Dropout(self.keep_prob)(q_rep)
            # rd = Dropout(self.keep_prob)(r_rep)
            q_rep = Dense(units=200, activation='relu')(q_rep)
            r_rep = Dense(units=200, activation='relu')(r_rep)
            score = Dot(axes=[1, 1], normalize=True)([q_rep, r_rep])
            output = Dense(units=1)(score)
            self.model = Model([input_q, input_r], output)
            self.model.compile(optimizer='adam', loss=rank_hinge_loss)

        elif self.category_loss == 'initial':
            whole_rep = Concatenate(axis=1)([q_rep, r_rep])
            wd = Dropout(self.keep_prob)(whole_rep)
            d = Dense(units=self.n_rnn, activation='relu')(wd)
            output = Dense(units=1, activation='sigmoid')(d)
            self.model = Model([input_q, input_r], output)
            self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        elif self.category_loss == 'full_connection':
            whole_rep = Concatenate(axis=1)([q_rep, r_rep])
            wd = Dropout(self.keep_prob)(whole_rep)
            output = Dense(units=1, activation='sigmoid')(wd)
            self.model = Model([input_q, input_r], output)
            self.model.compile(optimizer='adam', loss=rank_hinge_loss)