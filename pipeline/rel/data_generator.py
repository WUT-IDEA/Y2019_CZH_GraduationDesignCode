# coding: utf-8

from model.config import Config
import pickle
import json
import numpy as np

class DataGenerator(object):
    def __init__(self):
        self.config = Config()
        self.category = 'ranking'
        # vocab
        self.vocab_list = json.load(open("../../data/vocab_test.pt", "r"))
        # the format is : [question, golden_subid, golden_rel, golden_name]
        self.source = pickle.load(open("../train/source/train.pkl", "rb"))
        self.ent_rel_dict = pickle.load(open("../../data/fb2m_ent_rel.pkl", "rb"))
        self.w_index_dict = dict((self.vocab_list[i], str(i)) for i in range(len(self.vocab_list)))
        self.q_dict = dict()
        self.r_dict = dict()
        for u in self.source:
            self.q_dict[u[0]] = self.generate_words_indexseq(u[0], self.config.q_len)
        self.all_rels = json.load(open("../../data/wordsrel_list.pt"))
        for r in self.all_rels:
            self.r_dict[r] = self.generate_words_indexseq(r, self.config.r_len)
        self.q_seqs = None
        self.r_seqs = None
        self.generate_data()

    def generate_words_indexseq(self, seq, max_n):
        words = seq.split(" ")
        seq_len = len(words)
        index_words = [1 if self.w_index_dict.get(w) is None else self.w_index_dict.get(w) for w in words]
        if seq_len > max_n:
            index_words = index_words[0:max_n]
        elif seq_len < max_n:
            for i in range(max_n - seq_len):
                index_words.append(0)
        return index_words

    def generate_data(self):
        self.q_seqs = []
        self.r_seqs = []
        rel_neg_dict = pickle.load(open("../../data/rel_neg_dict.pkl", "rb"))
        cur_rel_neg_list = []
        for u in self.source:
            related_rels = self.ent_rel_dict.get(u[2])
            if related_rels is None:
                related_rels = []
            if u[2] in related_rels:
                related_rels.remove(u[2])
            n = len(related_rels)
            if n < 50:
                for neg_rel in rel_neg_dict.get(u[2]):
                    if neg_rel not in related_rels:
                        related_rels.append(neg_rel)
                    if len(related_rels) == 50:
                        break
            elif n > 50:
                related_rels = related_rels[0:50]
            cur_rel_neg_list.append(related_rels)
        for i in range(50):
            j = 0
            for u in self.source:
                self.q_seqs.append(u[0])
                self.q_seqs.append(u[0])
                self.r_seqs.append(u[2])
                self.r_seqs.append(cur_rel_neg_list[j][i])
                j += 1

    def generator_initial(self, ):
        pass

    def generator_ranking(self, batch_size):
        while True:
            q_seq = []
            r_seq = []
            count = 0
            for i in range(len(self.q_seqs)):
                q_seq.append(self.q_dict.get(self.q_seqs[i]))
                r_seq.append(self.r_dict.get(self.r_seqs[i]))
                count += 1
                if count == batch_size:
                    count = 0
                    label = np.zeros((batch_size,), dtype=np.int32)
                    label[::2] = 1
                    yield ([np.asarray(q_seq, dtype=np.int32), np.asarray(r_seq, dtype=np.int32)], label)
                    q_seq = []
                    r_seq = []