import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg19 import VGG19

class DisBlock(tf.keras.layers.Layer):
    def __init__(self,stride=1,bn=True):
        super(DisBlock,self).__init__()
        self.conv_bn_relu = tf.keras.Sequential(name='conv_bn_relu')
        self.conv_bn_relu.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=stride,padding='same'))
        if bn:
            self.conv_bn_relu.add(tf.keras.layers.BatchNormalization())
        self.conv_bn_relu.add(tf.keras.layers.LeakyReLU(0.2))

    def call(self,input_tensor):
        x = self.conv_bn_relu(input_tensor)
        return x

class Discriminator(tf.keras.Model):
    def __init__(self,chain_len=7):
        super(Discriminator,self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,padding='same')
        self.disblock = DisBlock(bn=False)
        self.chain = tf.keras.Sequential([
            *[DisBlock(stride=1+(i%2)) for i in range(1,chain_len+1)]
        ],name=f'conv_bn_relu_chained_{chain_len}')
        self.final = tf.keras.Sequential([
            tf.keras.layers.Flatten(name='flatten'),
            tf.keras.layers.Dense(units=1024,activation=None,name='dense1'),
            tf.keras.layers.LeakyReLU(0.2,name='leakyrelu'),
            tf.keras.layers.Dense(units=1,activation='sigmoid',name='dense2')
        ],name='Flatten_Dense')
    def call(self,input_tensor):
        x = self.conv(input_tensor)
        x = self.disblock(x)
        x = self.chain(x)
        x = self.final(x)
        return x
    def model(self):
        x = tf.keras.layers.Input(shape=(224,224,3))
        return tf.keras.Model(x,self.call(x))

class GenBlock(tf.keras.layers.Layer):
    def __init__(self):
        super(GenBlock,self).__init__()
        self.conv_bn = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,padding='same'),
            tf.keras.layers.BatchNormalization()
        ],name='conv_bn')
        self.prelu = tf.keras.layers.PReLU()
    def call(self,input_tensor):
        x = self.conv_bn(input_tensor)
        x = self.prelu(x)
        x = self.conv_bn(x)
        return tf.keras.layers.add([x,input_tensor])
        
class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator,self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,padding='same',input_shape=(1,224,224,3))
        self.bn = tf.keras.layers.BatchNormalization()
    
    def call(self,input_tensor):
        x = tf.keras.layers.Conv2D(filters=64,kernel_size=9,strides=1,padding='same')(input_tensor)
        x = tf.keras.layers.PReLU()(x)
        skip = x
        for i in range(4):
            x = GenBlock()(x)
        x = self.conv(x)
        x = self.bn(x)
        x = x + skip
        # print(x.shape)
        # x = tf.keras.layers.Conv2D(filters=256,kernel_size=3,strides=1,padding='same')(x)
        # x = tf.nn.depth_to_space(x,block_size=2)
        # x = tf.keras.layers.PReLU()(x)
        # x = tf.keras.layers.Conv2D(filters=256,kernel_size=3,strides=1,padding='same')(x)
        # x = tf.nn.depth_to_space(x,block_size=2)
        # x = tf.keras.layers.PReLU()(x)
        x = self.block(x)
        x = self.block(x)
        x = tf.keras.layers.Conv2D(filters=3,kernel_size=9,strides=1,padding='same')(x)
        return x
    
    def model(self):
        x = tf.keras.layers.Input(shape=(32,32,3))
        return tf.keras.Model(x,self.call(x))
    
    @staticmethod
    def block(input=None,scale_factor=2):
        x = tf.keras.layers.Conv2D(filters=256,kernel_size=3,strides=1,padding='same')(input)
        x = tf.nn.depth_to_space(x,block_size=scale_factor)
        x = tf.keras.layers.PReLU()(x)
        return x
    
class VGGloss():
    def __init__(self):
        vgg = VGG19(weights='imagenet',include_top=False,input_shape=(128,128,3))
        vgg.trainable = False
        # We use the output of block5_conv4 layer, remove all the layers after that
        self.vgg = vgg.layers[:21]
        self.loss = tf.keras.losses.MeanSquaredError()
    def __call__(self,super_res,high_res):
        for layer in self.vgg:  
            super_res = layer(super_res)
            high_res = layer(high_res)
        # After the loop is over super_res and high_res will be the outputs of the block5_conv4 layer, calculate the mse loss
        loss = self.loss(high_res,super_res)
        return loss

class SRGAN(tf.keras.Model):
    def __init__(self,generator,discriminator):
        super(SRGAN,self).__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.vgg_loss = VGGloss()

    def compile(self,d_optim = None,g_optim = None):
        super(SRGAN,self).compile()
        # opimizer for generator
        self.g_optim = g_optim
        # opimizer for discriminator
        self.d_optim = d_optim

    '''-------------------THIS METHOD DOES NOT WORK YET---------------------------------------
       -------------------PLEASE USE A CUSTOM TRAINING LOOP-----------------------------------'''
    def train_step(self, data):
        low_res, high_res = data
        # print(low_res.shape,high_res.shape)
        # self.generator.trainable = False
        # self.discriminator.trainable = True
        print('hey')
        
        super_res = self.generator(low_res)

        with tf.GradientTape() as tape1:
            # tape1.watch(self.discriminator.trainable_weights)
            sr_dis = self.discriminator(super_res)
            hr_dis = self.discriminator(high_res)
            real_score = tf.keras.losses.binary_crossentropy(hr_dis,tf.ones(shape=tf.shape(hr_dis)))
            fake_score = tf.keras.losses.binary_crossentropy(sr_dis,tf.zeros(shape=tf.shape(sr_dis)))
            # print(f"the real score is {tf.get_static_value(real_score)}")
            discriminator_loss = real_score + fake_score
        discriminator_gradients = tape1.gradient(discriminator_loss,self.discriminator.trainable_weights)
        # print(f"The discriminator gradients are {discriminator_gradients}")
        self.d_optim.apply_gradients(zip(discriminator_gradients,self.discriminator.trainable_weights))

        # self.discriminator.trainable = False
        # self.generator.trainable = True

        with tf.GradientTape() as tape2:
            # tape2.watch(self.generator.trainable_weights)
            sr = self.generator(low_res)
            content_loss = self.vgg_loss(sr,high_res)
            sr_dis = self.discriminator(sr)
            adverserial_loss = 1e-3*tf.keras.losses.binary_crossentropy(sr_dis,tf.ones(shape=tf.shape(sr_dis)))
            generator_loss = content_loss + adverserial_loss
        # print(f"the content loss is {tf.get_static_value(content_loss)} and the adverserial loss is {tf.get_static_value(adverserial_loss)}")
        generator_gradients = tape2.gradient(generator_loss,self.generator.trainable_weights)
        # print(f"The generator gradients are {generator_gradients}")
        # print('hey1')
        self.g_optim.apply_gradients(zip(generator_gradients,self.generator.trainable_weights))
        # print('hey2')

        return {'g_loss': generator_loss, 'd_loss': discriminator_loss}