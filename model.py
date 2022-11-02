import tensorflow as tf
from keras import layers

class cnnBlock(layers.Layer):
    def __init__(
        self,
        channels,
        discriminator = False,
        use_activation = True,
        use_bn = True,
        **kwargs,
    ):
        super(cnnBlock,self).__init__()
        self.use_activation = use_activation
        self.cnn = layers.Conv2D(channels,**kwargs,use_bias = not use_bn)

        if(use_bn):
            self.bn = layers.BatchNormalization()
        else:
            self.bn = layers.Layer()

        if(discriminator):
            self.act =  tf.keras.layers.LeakyReLU(0.2)
        else:
            self.act = layers.PReLU()

    def call(self,x,training = False):
        x = self.cnn(x)
        x = self.bn(x,training = training)
        
        if(self.use_activation):
            x = self.act(x)
        
        return x


class upSample(layers.Layer):
    def __init__(self,channels,scale_factor = 2):
        super(upSample,self).__init__()
        self.conv = layers.Conv2D(channels * scale_factor ** 2,kernel_size = 3,padding = "same")
        self.scale_factor = scale_factor
        self.act = layers.PReLU()

        
    def call(self,x,training = False):
        x = self.conv(x,training = training)
        x = tf.nn.depth_to_space(x,self.scale_factor)
        x = self.act(x)
        return x

class residualBlock(layers.Layer):
    def __init__(self,channels):
        super(residualBlock,self).__init__()
        self.block1 = cnnBlock(channels,kernel_size = 3,strides = 1,padding = "same")
        self.block2 = cnnBlock(channels,kernel_size = 3,strides = 1,padding = "same",use_activation = False)

    def call(self,x,training = False):
        y = self.block1(x,training = training)
        y = self.block2(x,training = training)
        return y + x


class generator(layers.Layer):
    def __init__(self,channels = 64,num_block = 16):
        super(generator,self).__init__()
        self.initial = cnnBlock(channels,kernel_size = 9,strides = 1 , padding = "same",use_bn = False)

        self.residual = tf.keras.Sequential(layers = [residualBlock(channels) for _ in range(num_block)])
        self.conv = cnnBlock(channels,kernel_size = 3,strides = 1,padding = "same",use_activation = False)
        self.upsample = tf.keras.Sequential([
                        upSample(channels,scale_factor=2),
                        upSample(channels,scale_factor=2)
                        ])
        self.final = layers.Conv2D(3,kernel_size = 9,strides = 1,padding = "same")

    def call(self,x,training = False):
        initial = self.initial(x,training = training)
        y = self.residual(initial,training = training)
        y = self.conv(y,training = training) + initial
        y = self.upsample(y,training = training)
        y = self.final(y,training = training)
        y = tf.nn.tanh(y)
        return y

    def model(self):
        x = tf.keras.Input(shape = (24,24,3))
        return tf.keras.Model(x,self.call(x))

class discriminator(layers.Layer):
    def __init__(self,features = [64,64,128,128,256,256,512,512]):
        super(discriminator,self).__init__()
        blocks = []
        for idx , feature in enumerate(features):
            blocks.append(cnnBlock(
                            feature,
                            kernel_size = 3,
                            strides = 1 + idx%2,
                            padding = "same",
                            discriminator=True,
                            use_activation=True,
                            use_bn = False if idx == 0 else True
            ))

        self.blocks = tf.keras.Sequential(layers = blocks)
        self.classifier = tf.keras.Sequential([
                                layers.Flatten(),
                                layers.Dense(1024,activation=tf.keras.layers.LeakyReLU(0.2)),
                                layers.Dense(1,activation = "sigmoid")
        ])

    def call(self,x,training):
        x = self.blocks(x,training = training)
        x = self.classifier(x,training = training)
        return x



