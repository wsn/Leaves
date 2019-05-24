import tensorflow as tf
import os
import pdb
import numpy as np

from .basic_block import Bottleneck_Block, Conv2d_BN

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
        #self.model= Resnet_50(self.width,self.height,self.img_channels, self.drop_rate)
        self.conv_part = tf.keras.applications.ResNet50(include_top=False, weights=None, pooling='avg')
        self.flatten = tf.keras.layers.Flatten()
        self.drop = tf.keras.layers.Dropout(self.drop_rate)
        self.fc_part = tf.keras.layers.Dense(3, activation=None)
    
       
    def call(self, x, training):
        #x = self.model(x,training)

        x = self.conv_part(x, training)
        x = self.flatten(x)
        x = self.drop(x, training)
        x = self.fc_part(x)

        return x

class Resnet_50(tf.keras.Model):

    def __init__(self, width,height,channel,drop_rate,classes=3):
       
        super(Resnet_50, self).__init__()

        self.zero1 = tf.keras.layers.ZeroPadding2D((3, 3))
        self.conv1 = Conv2d_BN(nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
        self.act1 = tf.keras.layers.Activation('relu')      
        self.zero2 = tf.keras.layers.ZeroPadding2D((1, 1))
        self.max = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))
       
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

        #self.av1 = tf.keras.layers.AveragePooling2D(pool_size=(7, 7),padding='same')
        self.glo1 = tf.keras.layers.GlobalAveragePooling2D()
        self.fl1 = tf.keras.layers.Flatten()
        self.drop = tf.keras.layers.Dropout(drop_rate)
        self.dense1 = tf.keras.layers.Dense(3, activation=None)

    def call(self, x, training):
        
        x = self.zero1(x)
        x = self.conv1(x,training)
        x = self.act1(x)
        x = self.zero2(x)
        x = self.max(x)

        x = self.bottle11(x,training,with_conv_shortcut=True)
        x = self.bottle12(x,training)
        x = self.bottle13(x,training)

        x = self.bottle21(x,training,with_conv_shortcut=True)
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

       # x = self.av1(x)
        x = self.glo1(x)
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




