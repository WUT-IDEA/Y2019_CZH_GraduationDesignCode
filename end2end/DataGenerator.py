# coding: utf-8

import numpy as np

def t(list_):
    return np.asarray(list_, dtype=np.int32)

def t2(list_, r, c, batch_size):
    return np.asarray(list_, dtype=np.int32).reshape((batch_size, r, c))

def data_generator(qwpath, swpath, rwpath, qcpath, scpath, rcpath, batch_size):
    while True:
        qw_reader = open(qwpath, 'r', encoding='ascii')
        sw_reader = open(swpath, 'r', encoding='ascii')
        rw_reader = open(rwpath, 'r', encoding='ascii')
        qc_reader = open(qcpath, 'r', encoding='ascii')
        sc_reader = open(scpath, 'r', encoding='ascii')
        rc_reader = open(rcpath, 'r', encoding='ascii')
        count = 0
        qw = []
        sw = []
        rw = []
        qc = []
        sc = []
        rc = []
        for line in qw_reader:
            qw.append(line.strip().split(" "))
            sw.append(sw_reader.readline().strip().split(" "))
            rw.append(rw_reader.readline().strip().split(" "))
            qc.append(qc_reader.readline().strip().split(" "))
            sc.append(sc_reader.readline().strip().split(" "))
            rc.append(rc_reader.readline().strip().split(" "))
            count += 1
            if count == batch_size:
                count = 0
                label = np.zeros((batch_size,), dtype=np.int32)
                label[::2] = 1
                yield ([t(qw), t(sw), t(rw), t(qc), t(sc), t(rc)], label)
                qw = []
                sw = []
                rw = []
                qc = []
                sc = []
                rc = []

def data_generator_w(qwpath, swpath, rwpath, batch_size):
    while True:
        qw_reader = open(qwpath, 'r', encoding='ascii')
        sw_reader = open(swpath, 'r', encoding='ascii')
        rw_reader = open(rwpath, 'r', encoding='ascii')
        count = 0
        qw = []
        sw = []
        rw = []
        for line in qw_reader:
            qw.append(line.strip().split(" "))
            sw.append(sw_reader.readline().strip().split(" "))
            rw.append(rw_reader.readline().strip().split(" "))
            count += 1
            if count == batch_size:
                count = 0
                label = np.zeros((batch_size,), dtype=np.int32)
                label[::2] = 1
                yield ([t(qw), t(sw), t(rw)], label)
                qw = []
                sw = []
                rw = []

def data_generator_y(qwpath, swpath, rwpath, qwcpath, swcpath, rwcpath, batch_size):
    max_n_words_qu = 24
    max_n_words_s = 10
    max_n_words_r = 17
    max_len_word = 50
    qwc_len = max_n_words_qu * max_len_word
    swc_len = max_n_words_s * max_len_word
    rwc_len = max_n_words_r * max_len_word
    while True:
        qw_reader = open(qwpath, 'r', encoding='ascii')
        rw_reader = open(rwpath, 'r', encoding='ascii')
        sw_reader = open(swpath, 'r', encoding='ascii')
        qwc_reader = open(qwcpath, 'r', encoding="ascii")
        swc_reader = open(swcpath, 'r', encoding="ascii")
        rwc_reader = open(rwcpath, 'r', encoding="ascii")
        count = 0
        qw = []
        rw = []
        sw = []
        qwc = []
        swc = []
        rwc = []
        for line in qw_reader:
            qw.append(line.strip().split(" "))
            rw.append(rw_reader.readline().strip().split(" "))
            sw.append(sw_reader.readline().strip().split(" "))
            qwc_line = qwc_reader.readline().strip().split(" ")
            swc_line = swc_reader.readline().strip().split(" ")
            rwc_line = rwc_reader.readline().strip().split(" ")
            cur_qwc_len = len(qwc_line)
            cur_swc_len = len(swc_line)
            cur_rwc_len = len(rwc_line)
            if cur_qwc_len > qwc_len:
                qwc_line = qwc_line[0:qwc_len]
            else:
                for i in range(qwc_len - cur_qwc_len):
                    qwc_line.append("0")
            qwc.append(qwc_line)
            # subject
            if cur_swc_len > swc_len:
                swc_line = swc_line[0:swc_len]
            elif cur_swc_len < swc_len:
                for i in range(swc_len - cur_swc_len):
                    swc_line.append("0")
            swc.append(swc_line)
            # relation
            if cur_rwc_len > rwc_len:
                rwc_line = rwc_line[0:rwc_len]
            elif cur_rwc_len < rwc_len:
                for i in range(rwc_len - cur_rwc_len):
                    rwc_line.append("0")
            rwc.append(rwc_line)
            count += 1
            if count == batch_size:
                count = 0
                label = np.zeros((batch_size,), dtype=np.int32)
                label[::2] = 1
                yield ([t(qw), t(sw), t(rw), t2(qwc, max_n_words_qu, max_len_word, batch_size),
                       t2(swc, max_n_words_s, max_len_word, batch_size),
                       t2(rwc, max_n_words_r, max_len_word, batch_size)],label)
                qw = []
                rw = []
                sw = []
                qwc = []
                swc = []
                rwc = []

def data_generator_y3(qwpath, swpath, rwpath, qwcpath, swcpath, rwcpath, batch_size):
    max_n_words_qu = 24
    max_n_words_s = 10
    max_n_words_r = 17
    max_len_word = 50
    qwc_len = max_n_words_qu * max_len_word
    swc_len = max_n_words_s * max_len_word
    rwc_len = max_n_words_r * max_len_word
    while True:
        qw_reader = open(qwpath, 'r', encoding='ascii')
        rw_reader = open(rwpath, 'r', encoding='ascii')
        sw_reader = open(swpath, 'r', encoding='ascii')
        qwc_reader = open(qwcpath, 'r', encoding="ascii")
        swc_reader = open(swcpath, 'r', encoding="ascii")
        rwc_reader = open(rwcpath, 'r', encoding="ascii")
        count = 0
        qw = []
        rw = []
        sw = []
        qwc = []
        swc = []
        rwc = []
        for line in qw_reader:
            qw.append(line.strip().split(" "))
            rw.append(rw_reader.readline().strip().split(" "))
            sw.append(sw_reader.readline().strip().split(" "))
            qwc_line = qwc_reader.readline().strip().split(" ")
            swc_line = swc_reader.readline().strip().split(" ")
            rwc_line = rwc_reader.readline().strip().split(" ")
            cur_qwc_len = len(qwc_line)
            cur_swc_len = len(swc_line)
            cur_rwc_len = len(rwc_line)
            if cur_qwc_len > qwc_len:
                qwc_line = qwc_line[0:qwc_len]
            else:
                for i in range(qwc_len - cur_qwc_len):
                    qwc_line.append("0")
            qwc.append(qwc_line)
            # subject
            if cur_swc_len > swc_len:
                swc_line = swc_line[0:swc_len]
            elif cur_swc_len < swc_len:
                for i in range(swc_len - cur_swc_len):
                    swc_line.append("0")
            swc.append(swc_line)
            # relation
            if cur_rwc_len > rwc_len:
                rwc_line = rwc_line[0:rwc_len]
            elif cur_rwc_len < rwc_len:
                for i in range(rwc_len - cur_rwc_len):
                    rwc_line.append("0")
            rwc.append(rwc_line)
            count += 1
            if count == batch_size:
                count = 0
                label = np.zeros((batch_size,), dtype=np.int32)
                label[::2] = 1
                label2 = label.copy()
                yield ([t(qw), t(sw), t(rw), t2(qwc, max_n_words_qu, max_len_word, batch_size),
                       t2(swc, max_n_words_s, max_len_word, batch_size),
                       t2(rwc, max_n_words_r, max_len_word, batch_size)],[label, label2])
                qw = []
                rw = []
                sw = []
                qwc = []
                swc = []
                rwc = []