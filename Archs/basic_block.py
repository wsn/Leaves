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
    
    def call(self, x, training=True):

        x = self.conv(x)

        if self.bn:
            x = self.bn(x, training)

        if self.relu:
            x = self.relu(x)
        
        return x