# coding: utf-8

class Config(object):
    def __init__(self):
        self.q_len = 24
        self.r_len = 17
        self.n_vocab = 62957
        self.n_embed = 300
        self.embed_mat = None
        self.keep_prob = 0.5
        self.category_rnn = "lstm"
        self.keep_prob_rnn = 0.2
        self.n_rnn = 300
        self.category_loss = 'ranking'
        self.batch_size = 128
        self.epochs = 60