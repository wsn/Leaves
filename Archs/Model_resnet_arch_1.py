import tensorflow as tf
import os
import pdb
import numpy as np

class Model_resnet(tf.keras.Model):

    '''
       Model_resnet： Resnet architecture for task 1 
    '''

    def __init__(self, drop_rate):

        super(Model_resnet, self).__init__()
        
        self.img_channels = 3
        self.width = 128
        self.height = 128
        self.drop_rate = 0.5
        self.model= Resnet_50(self.width,self.height,self.img_channels, self.drop_rate) 
    
       
    def call(self, x, training=True):
        x = self.model(x,training)
        return x

class Conv2d_BN(tf.keras.Model):

    def __init__(self, nb_filter, kernel_size,strides=(1, 1), padding='same'):
        
        super(Conv2d_BN, self).__init__()
        
        self.conv1 = tf.keras.layers.Conv2D(nb_filter, kernel_size, padding=padding, strides=strides)
        self.bn1 = tf.keras.layers.BatchNormalization(axis=3,fused=False)
       
    def call(self, x, training=True):
        x = self.conv1(x)
        x = self.bn1(x, training)
       
        return x

class Bottleneck_Block(tf.keras.Model):

    def __init__(self, nb_filters,strides=(1,1)):
        
        super(Bottleneck_Block, self).__init__()
        k1,k2,k3 = nb_filters
        self.conv1 = Conv2d_BN(nb_filter=k1, kernel_size=1, strides=strides, padding='same')
        self.act1 = tf.keras.layers.Activation('relu')
        self.conv2 = Conv2d_BN(nb_filter=k2, kernel_size=3, padding='same')
        self.act2 = tf.keras.layers.Activation('relu')
        self.conv3 = Conv2d_BN(nb_filter=k3, kernel_size=1, padding='same')
        self.act3 = tf.keras.layers.Activation('relu')
        self.shortcut = Conv2d_BN(nb_filter=k3, strides=strides, kernel_size=1)
    
    def call(self, x, training = True,with_conv_shortcut=False):
        org = x
        x=self.conv1(x,training)
        x = self.act1(x)
        x=self.conv2(x,training)
        x = self.act2(x)
        x=self.conv3(x,training)
        if with_conv_shortcut:
            short = self.shortcut(org,training)
            x = short + x
        else:
            x = x + org
        x = self.act3(x)
        return x  

class Resnet_50(tf.keras.Model):

    def __init__(self, width,height,channel,drop_rate,classes=3):
       
        super(Resnet_50, self).__init__()

        self.inpt = tf.keras.layers.Input(shape=(width, height, channel))
        self.zero = tf.keras.layers.ZeroPadding2D((3, 3))
        self.conv1 = Conv2d_BN(nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
        self.act1 = tf.keras.layers.Activation('relu')      
        self.max = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')
       
        self.bottle11 = Bottleneck_Block( nb_filters=[64,64,256])
        self.bottle12 = Bottleneck_Block( nb_filters=[64,64,256])
        self.bottle13 = Bottleneck_Block( nb_filters=[64,64,256])
        
        self.bottle21 = Bottleneck_Block( nb_filters=[128, 128, 512],strides=(2,2))
        self.bottle22 = Bottleneck_Block( nb_filters=[128, 128, 512])
        self.bottle23 = Bottleneck_Block( nb_filters=[128, 128, 512])
        self.bottle24 = Bottleneck_Block( nb_filters=[128, 128, 512])

        self.bottle31 = Bottleneck_Block( nb_filters=[256, 256, 1024],strides=(2,2))
        self.bottle32 = Bottleneck_Block( nb_filters=[256, 256, 1024])
        self.bottle33 = Bottleneck_Block( nb_filters=[256, 256, 1024])
        self.bottle34 = Bottleneck_Block( nb_filters=[256, 256, 1024])
        self.bottle35 = Bottleneck_Block( nb_filters=[256, 256, 1024])
        self.bottle36 = Bottleneck_Block( nb_filters=[256, 256, 1024])

        self.bottle41 = Bottleneck_Block( nb_filters=[512, 512, 2048],strides=(2,2))
        self.bottle42 = Bottleneck_Block( nb_filters=[512, 512, 2048])
        self.bottle43 = Bottleneck_Block( nb_filters=[512, 512, 2048])

        self.av1 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2),padding='same')
        self.fl1 = tf.keras.layers.Flatten()
        self.drop = tf.keras.layers.Dropout(drop_rate)
        self.dense1 = tf.keras.layers.Dense(3, activation=None)

    def call(self, x, training=True):
        
        x = self.zero(x)
        x = self.conv1(x,training)
        x = self.act1(x)
        x = self.max(x)

        x = self.bottle11(x,training,with_conv_shortcut=True)
        x = self.bottle12(x,training)
        x = self.bottle13(x,training)

        x = self.bottle21(x,with_conv_shortcut=True)
        x = self.bottle22(x,training)
        x = self.bottle23(x,training)
        x = self.bottle24(x,training)
        
        x = self.bottle31(x,training,with_conv_shortcut=True)
        x = self.bottle32(x,training)
        x = self.bottle33(x,training)
        x = self.bottle34(x,training)
        x = self.bottle35(x,training)
        x = self.bottle36(x,training)

        x = self.bottle41(x,training,with_conv_shortcut=True)
        x = self.bottle42(x,training)
        x = self.bottle43(x,training)

        x = self.av1(x)
        x = self.fl1(x)
        x = self.drop(x, training)
        x = self.dense1(x)

        return x 

if __name__ == '__main__':

    tf.enable_eager_execution()
    m = Model_resnet(128,128,3)
    x = tf.convert_to_tensor(np.zeros(shape=[2,128,128,3]))
    y = m(x, True)
    pdb.set_trace()




