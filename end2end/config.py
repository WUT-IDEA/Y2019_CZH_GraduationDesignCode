# coding: utf-8




class BasicConfig(object):
    def __init__(self):
        self.nums_words = 62957
        self.nums_chars = 1767
        self.wordembdim = 300
        self.charembdim = 50
        self.embdim = 200
        self.encdim = 400
        self.batch_size = 540
        self.gradnorm = 5.0
        self.sentence_len = 36
        self.word_len = 24
        self.sub_len = 24
        self.rel_len = 17
        # char level
        self.quc_len = 105
        self.subc_len = 60
        self.relc_len = 108
        self.qu_maxlen = 36

class MineBasicConfig(object):
    def __init__(self):
        self.nums_words = 62957
        self.nums_chars = 1767
        self.wordembdim = 300
        self.charembdim = 50
        self.embdim = 200
        self.encdim = 400
        self.batch_size = 540
        self.gradnorm = 5.0
        self.sentence_len = 24
        self.word_len = 50
        self.sub_len = 10
        self.rel_len = 17
        # char level
        self.quc_len = 105
        self.subc_len = 52
        self.relc_len = 108
        self.qu_maxlen = 24
        '''
                self.nums_words = 62957
        self.nums_chars = 1767
        self.wordembdim = 300
        self.charembdim = 50
        self.embdim = 200
        self.encdim = 400
        self.batch_size = 540
        self.gradnorm = 5.0
        self.sentence_len = 24
        self.word_len = 24
        self.sub_len = 10
        self.rel_len = 17
        # char level
        self.quc_len = 105
        self.subc_len = 55
        self.relc_len = 108
        self.qu_maxlen = 24
        '''

