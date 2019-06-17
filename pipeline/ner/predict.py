# coding: utf-8


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
set_session(tf.Session(config=config))

from model.config import Config
from model.ner_model import BiRnnNerModel
import numpy as np
import sys

def load_label(path):
    labels = []
    with open(path, 'r') as f:
        for line in f:
            tags = line.strip().split(" ")
            tmp = []
            for tag in tags:
                if tag == 'e':
                    tmp.append(1)
                else:
                    tmp.append(0)
            labels.append(tmp)
    return labels

if __name__ == '__main__':
    model_path = sys.argv[1]
    usr_crf = sys.argv[2]
    rnn = sys.argv[3]
    result_name = sys.argv[4]
    config = Config()
    config.use_crf = usr_crf
    config.category_rnn = rnn
    model = BiRnnNerModel(config)
    model.build()
    print(model.model.summary())
    model.model.load_weights(model_path)
    q = np.loadtxt("test/qw.txt", dtype="int32")
    label_trues = load_label("test/test_tag.txt")
    tag_dict = {0: 'c', 1: 'e', 2: 'o'}
    pre_result = np.argmax(model.model.predict(q), axis=2)
    np.savetxt("ner_result/"+result_name, pre_result)
    n_pretrue = 0
    n_true = 0
    n_pre = 0
    for lab, lab_pred in zip(label_trues, pre_result):
        lab_pred = lab_pred[:len(lab)]
        for a, b in zip(lab, lab_pred):
            if a == 1:
                n_true += 1
            if b == 1:
                n_pre += 1
            if a == 1 and b == 1:
                n_pretrue += 1
    precision = n_pretrue * 1.0 / n_pre
    recall = n_pretrue * 1.0 / n_true
    f1 = (2 * precision * recall) / (precision + recall)
    print("precision: {}".format(precision))
    print("recall: {}".format(recall))
    print("f1: {}".format(f1))