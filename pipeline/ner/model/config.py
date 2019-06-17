# coding: utf-8

class Config(object):
    def __init__(self):
        self.n_input = 24
        self.n_vocab = 62957
        self.n_embed = 300
        self.embed_mat = None
        self.keep_prob = 0.2  #0.5
        self.category_rnn = "lstm"
        self.keep_prob_rnn = 0.2  #0.7
        self.n_rnn = 300
        self.n_entity = 3
        self.use_crf = "False"
        self.batch_size = 64
        self.epochs = 60