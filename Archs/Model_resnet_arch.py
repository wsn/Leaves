import tensorflow as tf
import os
import pdb

class Model_resnet(tf.keras.Model):

    '''
       Model_resnet： Resnet architecture for task 1 
    '''

    def __init__(self, drop_rate):

        super(Model_resnet, self).__init__()
        
        self.drop_rate = drop_rate
        
        self.conv_part = tf.keras.applications.ResNet50(include_top=False, weights=None, pooling='avg')
        self.fc_part = tf.keras.layers.Dense(3, activation=None)
        self.flatten = tf.keras.layers.Flatten()
        self.drop = tf.keras.layers.Dropout(self.drop_rate)
       
    def call(self, x, training=True):
        x = self.conv_part(x,training)
        x = self.flatten(x)
        x = self.drop(x, training)
        x = self.fc_part(x)

