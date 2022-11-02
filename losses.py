import tensorflow as tf
from keras import layers
from keras.metrics import mean_squared_error as error

class vgg22(layers.Layer):
    def __init__(self):
        super(vgg22,self).__init__()
        vgg = tf.keras.applications.VGG19(include_top = False,weights = "imagenet")
        vgg.trainable = False
        self.model = vgg

    def call(self,sr,hr):

        for l in self.model.layers[:6]:
            sr = l(sr)
            hr = l(hr)
        return error(sr,hr)



class vgg54(layers.Layer):
    def __init__(self):
        super(vgg54,self).__init__()
        vgg = tf.keras.applications.VGG19(include_top = False,weights = "imagenet")
        vgg.trainable = False
        self.model = vgg

    def call(self,sr,hr):

        for l in self.model.layers[:21]:
            sr = l(sr)
            hr = l(hr)
        return error(sr,hr)
        