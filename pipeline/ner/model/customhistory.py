# coding: utf-8

import keras
import os

class LossHistory(keras.callbacks.Callback):
    def __init__(self, logpath, modelpath):
        super().__init__()
        self.logpath = logpath
        self.modelpath = modelpath
        if not os.path.exists(modelpath):
            os.mkdir(modelpath)
    def set_model(self, model):
        self.model = model
        self.writer = open(self.logpath, "w")
    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0 or (epoch + 1) % 5 == 0:
            self.model.save("{}/model_{}.h5".format(self.modelpath, epoch+1))
        self.writer.write("epoch {}, loss:{},  valid_loss:{}\n".format(epoch+1, logs['loss'], logs['val_loss']))