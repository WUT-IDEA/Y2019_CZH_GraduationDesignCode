# coding: utf-8

from keras.layers import Input, Embedding, Dense, Bidirectional, Dropout, SimpleRNN, LSTM, GRU
from keras_contrib.layers import CRF
from keras.models import Model

class BiRnnNerModel():
    def __init__(self, config):
        self.n_input = config.n_input
        self.n_vocab = config.n_vocab
        self.n_embed = config.n_embed
        self.embed_mat = config.embed_mat
        self.keep_prob = config.keep_prob
        self.category_rnn = config.category_rnn
        self.keep_prob_rnn = config.keep_prob_rnn
        self.n_rnn = config.n_rnn
        self.n_entity = config.n_entity
        self.use_crf = config.use_crf
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.model = None

    def build(self):
        input_words = Input(shape=(self.n_input, ), dtype='int32')
        if self.embed_mat is None:
            word_embedding = Embedding(input_dim=self.n_vocab,
                                       output_dim=self.n_embed,
                                       mask_zero=True,
                                       trainable=True)(input_words)
        else:
            print("using glove embedding~")
            word_embedding = Embedding(input_dim=self.n_vocab,
                                       output_dim=self.n_embed,
                                       weights=[self.embed_mat],
                                       mask_zero=True,
                                       trainable=False)(input_words)
        dropout = Dropout(self.keep_prob)(word_embedding)
        if self.category_rnn == "rnn":
            rnn = Bidirectional(SimpleRNN(units=self.n_rnn,
                                          dropout=self.keep_prob_rnn,
                                          recurrent_dropout=self.keep_prob_rnn,
                                          return_sequences=True))(dropout)
            print("rnn == rnn")
        elif self.category_rnn == "lstm":
            rnn = Bidirectional(LSTM(units=self.n_rnn,
                                     dropout=self.keep_prob_rnn,
                                     recurrent_dropout=self.keep_prob_rnn,
                                     return_sequences=True))(dropout)
            print("rnn == lstm")
        elif self.category_rnn == "gru":
            rnn = Bidirectional(GRU(units=self.n_rnn,
                                    dropout=self.keep_prob_rnn,
                                    recurrent_dropout=self.keep_prob_rnn,
                                    return_sequences=True))(dropout)
            print("rnn == gru")
        else:
            print("not find! error!")
            exit()
        # encoded_text = TimeDistributed(Dropout(self.keep_prob))(rnn)
        encoded_text = Dropout(self.keep_prob)(rnn)

        if self.use_crf == "True":
            print("use_crf!")
            crf = CRF(self.n_entity, test_mode="viterbi", sparse_target=False)
            output = crf(encoded_text)
            self.model = Model(input_words, output)
            self.model.compile(optimizer='adam', loss=crf.loss_function, metrics=[crf.accuracy])
        else:
            print("not use crf!")
            output = Dense(self.n_entity, activation='softmax')(encoded_text)
            self.model = Model(input_words, output)
            self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])