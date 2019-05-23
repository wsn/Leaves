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
    
    def call(self, x, training=True):

        short = x
        x = self.conv1(x)
        x = self.bn1(x, training)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = short + x
        x = self.relu2(x)

        return x

        
