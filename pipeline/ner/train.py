# coding: utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))

from model.config import Config
from model.ner_model2 import BiRnnNerModel
from model.customhistory import LossHistory
import numpy as np
import time
import sys

# read file by line
def read_byline(filepath):
    lines = []
    with open(filepath, 'r', encoding='utf-8') as reader:
        for line in reader:
            lines.append(line.strip())
    return lines

def load_tags(filepath):
    tag_list = read_byline(filepath)
    array = np.zeros(shape=(len(tag_list), 24, 3))
    for line_index in range(len(tag_list)):
        tags = tag_list[line_index].split(" ")
        for index in range(24):
            # 0 --> c  1 --> e  2 --> filling
            tag_index = 2
            if index < len(tags):
                tag = tags[index]
                if tag == 'c':
                    tag_index = 0
                elif tag == 'e':
                    tag_index = 1
                array[line_index][index][tag_index] = 1
            else:
                array[line_index][index][tag_index] = 1
    return array

if __name__ == '__main__':
    logpath = sys.argv[1]
    modelpath = sys.argv[2]
    use_crf = sys.argv[3]
    rnn_str = sys.argv[4]
    epochs = sys.argv[5]
    config = Config()
    config.use_crf = use_crf
    print(config.use_crf)
    config.category_rnn = rnn_str
    config.epochs = int(epochs)
    embedding_maxtrix = np.loadtxt("../../data/mini_glove.txt", dtype=np.float32)
    config.embed_mat = embedding_maxtrix
    # config.use_crf = False
    model = BiRnnNerModel(config)
    model.build()
    print(model.model.summary())
    history = LossHistory(logpath, modelpath)
    history.set_model(model.model)
    x = np.loadtxt("train/qw.txt", dtype="int32")
    y = load_tags("train/train_tag.txt")
    v_x = np.loadtxt("valid/qw.txt", dtype="int32")
    v_y = load_tags("valid/valid_tag.txt")
    start_time = time.strftime('%Y.%m.%d  %H:%M:%S', time.localtime(time.time()))
    model.model.fit(x=x,
                    y=y,
                    validation_data=[v_x, v_y],
                    callbacks=[history],
                    batch_size=model.batch_size,
                    epochs=model.epochs)
    end_time = time.strftime('%Y.%m.%d  %H:%M:%S', time.localtime(time.time()))
    print("start time: {}".format(start_time))
    print("end time: {}".format(end_time))