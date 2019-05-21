import tensorflow as tf 

#include two architecture FCN8  and FCN32
class Model_fcn8(tf.keras.Model):

    '''
        Model_fcn8 :  Fcn8 Architecture for task 2.
    '''

    def __init__(self,num_features=64):
        
        super(Model_fcn8, self).__init__()

        self.num_features = num_features
        
        self.block1= Block1(self.num_features)
        self.block2= Block2(self.num_features)
        self.block3= Block3(self.num_features)
        
        self.change_score2 = tf.keras.layers.Conv2DTranspose(2, 2, 2, padding='valid', activation=None)

        self.score_pool4 = tf.keras.layers.Convolution2D(2,1, 1, 'same', activation='relu',kernel_initializer='he_normal')
        self.score4 = tf.keras.layers.Conv2DTranspose(2, 2, 2, padding='valid', activation=None)

        self.score_pool3 = tf.keras.layers.Convolution2D(2 ,1, 1, 'same', activation='relu',kernel_initializer='he_normal')
        self.up1 = tf.keras.layers.Conv2DTranspose(2, 8, 8, padding='valid', activation=None)
        self.activ_1 = tf.keras.layers.Activation('softmax')


    def call(self, x, training=True):
        x3 = self.block1(x, training)
        x2 = self.block2(x3, training)
        x1 = self.block3(x2, training)
        
        #x1 = self.fcn32.get_layer('score_fr').output
        x1 = self.change_score2(x1)
        #x2 = self.fcn32.get_layer('max_pool4').output
        x2 = self.score_pool4(x2)
        x4 = x1 + x2
        x4 = self.score4(x4)
        #x3 = self.fcn32.get_layer('max_pool3').output
        x3 =self.score_pool3(x3)
        x = x3 + x4

        x = self.up1(x)
        x = self.activ_1(x)
        
        return x

class Block1(tf.keras.Model):

    '''
        Block1 :  Part of Fcn32 Architecture for task 2.
    '''

    def __init__(self, num_features=64):
        
        super(Block1, self).__init__()

        self.num_features = num_features
        
        self.conv1_1 = tf.keras.layers.Convolution2D(self.num_features, 3, 1, 'same', activation='relu')
        self.conv1_2 = tf.keras.layers.Convolution2D(self.num_features, 3, 1, 'same', activation='relu')
        self.max_pool1 = tf.keras.layers.MaxPooling2D(2, 2, 'same')
        self.conv2_1 = tf.keras.layers.Convolution2D(self.num_features * 2, 3, 1, 'same', activation='relu')
        self.conv2_2 = tf.keras.layers.Convolution2D(self.num_features * 2, 3, 1, 'same', activation='relu')
        self.max_pool2 = tf.keras.layers.MaxPooling2D(2, 2, 'same')
        self.conv3_1 = tf.keras.layers.Convolution2D(self.num_features * 4, 3, 1, 'same', activation='relu')
        self.conv3_2 = tf.keras.layers.Convolution2D(self.num_features * 4, 3, 1, 'same', activation='relu')
        self.conv3_3 = tf.keras.layers.Convolution2D(self.num_features * 4, 3, 1, 'same', activation='relu')
        self.max_pool3 = tf.keras.layers.MaxPooling2D(2, 2, 'same',name='max_pool3')

    def call(self, x, training=True):

        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.max_pool1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.max_pool2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x1 = self.max_pool3(x)
        
        return x1

class Block2(tf.keras.Model):

    '''
        Block2 :  Part of Fcn32 Architecture for task 2.
    '''

    def __init__(self, num_features=64):
        
        super(Block2, self).__init__()

        self.num_features = num_features
        
        self.conv4_1 = tf.keras.layers.Convolution2D(self.num_features * 8, 3, 1, 'same', activation='relu')
        self.conv4_2 = tf.keras.layers.Convolution2D(self.num_features * 8, 3, 1, 'same', activation='relu')
        self.conv4_3 = tf.keras.layers.Convolution2D(self.num_features * 8, 3, 1, 'same', activation='relu')
        self.max_pool4 = tf.keras.layers.MaxPooling2D(2, 2, 'same',name='max_pool4')
        
    def call(self, x, training=True):
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.max_pool4(x)
       
        return x

class Block3(tf.keras.Model):

    '''
        Block3 :  Part of Fcn32 Architecture for task 2.
    '''

    def __init__(self, num_features=64):
        
        super(Block3, self).__init__()

        self.num_features = num_features

        self.conv5_1 = tf.keras.layers.Convolution2D(self.num_features * 8, 3, 1, 'same', activation='relu')
        self.conv5_2 = tf.keras.layers.Convolution2D(self.num_features * 8, 3, 1, 'same', activation='relu')
        self.conv5_3 = tf.keras.layers.Convolution2D(self.num_features * 8, 3, 1, 'same', activation='relu')
        self.max_pool5 = tf.keras.layers.MaxPooling2D(2, 2, 'same')      
        self.fc6 = tf.keras.layers.Convolution2D(4096, 7, 1, 'same', activation='relu')
        self.dropout_1=tf.keras.layers.Dropout(0.5)
        self.fc7 = tf.keras.layers.Convolution2D(4096, 1, 1, 'same', activation='relu')
        self.dropout_2=tf.keras.layers.Dropout(0.5)
        self.score_fr = tf.keras.layers.Convolution2D(2, 1, 1, 'same', activation='relu',kernel_initializer='he_normal',name = 'score_fr')
        
    def call(self, x, training=True):
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.max_pool5(x)
        x = self.fc6(x)
        x = self.dropout_1(x, training)
        x = self.fc7(x)
        x = self.dropout_2(x, training)
        x = self.score_fr(x)
                
        return x

class Model_fcn32(tf.keras.Model):

    '''
        Model_fcn32 :  Fcn32 Architecture for task 2.
    '''

    def __init__(self,num_features=64):
        
        super(Model_fcn32, self).__init__()

        self.num_features = num_features
        
        self.block1= Block1(self.num_features)
        self.block2= Block2(self.num_features)
        self.block3= Block3(self.num_features)
        self.score2 = tf.keras.layers.Conv2DTranspose(2, kernel_size=(32, 32), strides=(32, 32), padding='valid', activation=None)
        self.activ_1 = tf.keras.layers.Activation('softmax')
   
    def get_layer(self, name=None, index=None):
        return super().get_layer(name=name, index=index)

    def call(self, x, training=True):

        x = self.block1(x, training)
        x = self.block2(x, training)
        x = self.block3(x, training)
        x = self.score2(x)
        x = self.activ_1(x)
        
        return x
