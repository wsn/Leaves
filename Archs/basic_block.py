import tensorflow as tf

class ConvBNReLUBlock(tf.keras.Model):

    def __init__(self, k, n, s, act=True, norm=True, initializer='he_normal', weight_decay=0):

        super(ConvBNReLUBlock, self).__init__()

        self.ksize = k
        self.num_filters = n
        self.stride = s

        self.initializer = initializer
        self.regularizer = tf.keras.regularizers.l2(weight_decay)

        self.conv = tf.keras.layers.Convolution2D(self.num_filters, self.ksize, self.stride, 'same', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.bn = tf.keras.layers.BatchNormalization() if norm else None
        self.relu = tf.keras.layers.ReLU() if act else None
    
    def call(self, x, training):

        x = self.conv(x)

        if self.bn:
            x = self.bn(x, training)

        if self.relu:
            x = self.relu(x)
        
        return x
    
class ResidualBlockV1(tf.keras.Model):

    def __init__(self, num_features, initializer, weight_decay):

        super(ResidualBlockV1, self).__init__()

        self.num_features = num_features
        self.initializer = initializer
        self.weight_decay = weight_decay
        self.regularizer = tf.keras.regularizers.l2(self.weight_decay)

        self.conv1 = tf.keras.layers.Convolution2D(self.num_features, 3, 1, 'same', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.conv2 = tf.keras.layers.Convolution2D(self.num_features, 3, 1, 'same', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.relu1 = tf.keras.layers.ReLU()
        self.relu2 = tf.keras.layers.ReLU()
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
    
    def call(self, x, training):

        short = x
        x = self.conv1(x)
        x = self.bn1(x, training)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x, training)
        x = short + x
        x = self.relu2(x)

        return x

class Conv2d_BN(tf.keras.Model):

    def __init__(self, nb_filter, kernel_size,strides=(1, 1), padding='same',initializer='he_normal'):
        
        super(Conv2d_BN, self).__init__()
        
        self.conv1 = tf.keras.layers.Conv2D(nb_filter, kernel_size, padding=padding, strides=strides,kernel_initializer=initializer)
        self.bn1 = tf.keras.layers.BatchNormalization()
       
    def call(self, x, training):
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
    
    def call(self, x, training,with_conv_shortcut=False):
        org = x
        x=self.conv1(x,training)
        x = self.act1(x)
        x=self.conv2(x,training)
        x = self.act2(x)
        x=self.conv3(x,training)
        if with_conv_shortcut:
            short = self.shortcut(org, training)
            x = short + x
        else:
            x = x + org
        x = self.act3(x)
        return x  
