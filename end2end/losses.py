# coding: utf-8


from keras.losses import *
from keras.layers import Lambda
from keras.utils.generic_utils import deserialize_keras_object
import tensorflow as tf

def rank_hinge_loss(y_true, y_pred):
    margin = 1.
    # output_shape = K.int_shape(y_pred)
    y_pos = Lambda(lambda a: a[::2, :], output_shape= (1,))(y_pred)
    y_neg = Lambda(lambda a: a[1::2, :], output_shape= (1,))(y_pred)
    loss = K.maximum(0., margin + y_neg - y_pos)
    return K.mean(loss)
