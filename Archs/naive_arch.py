import tensorflow as tf 

class Naive(tf.keras.Model):

    '''
        Naive : Simplified U-Net Architecture for task 2.
    '''

    def __init__(self, num_features, weight_decay, initializer):
        
        super(Naive, self).__init__()

        self.num_features = num_features
        self.weight_decay = weight_decay
        self.initializer = initializer
        self.regularizer = tf.keras.regularizers.l2(self.weight_decay)

        self.conv_in = tf.keras.layers.Convolution2D(self.num_features, 3, 1, 'same', activation='relu', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.conv1_1 = tf.keras.layers.Convolution2D(self.num_features, 3, 1, 'same', activation='relu', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.conv1_2 = tf.keras.layers.Convolution2D(self.num_features, 3, 1, 'same', activation='relu', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.max_pool1 = tf.keras.layers.MaxPooling2D(2, 2, 'same')
        self.conv2_1 = tf.keras.layers.Convolution2D(self.num_features * 2, 3, 1, 'same', activation='relu', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.conv2_2 = tf.keras.layers.Convolution2D(self.num_features * 2, 3, 1, 'same', activation='relu', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.max_pool2 = tf.keras.layers.MaxPooling2D(2, 2, 'same')
        self.conv3_1 = tf.keras.layers.Convolution2D(self.num_features * 4, 3, 1, 'same', activation='relu', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.conv3_2 = tf.keras.layers.Convolution2D(self.num_features * 4, 3, 1, 'same', activation='relu', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.max_pool3 = tf.keras.layers.MaxPooling2D(2, 2, 'same')
        self.conv4_1 = tf.keras.layers.Convolution2D(self.num_features * 4, 3, 1, 'same', activation='relu', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.conv4_2 = tf.keras.layers.Convolution2D(self.num_features * 4, 3, 1, 'same', activation='relu', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.upsample_3 = tf.keras.layers.UpSampling2D(2, interpolation='bilinear')
        self.conv3_3 = tf.keras.layers.Convolution2D(self.num_features * 4, 3, 1, 'same', activation='relu', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.conv3_4 = tf.keras.layers.Convolution2D(self.num_features * 4, 3, 1, 'same', activation='relu', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.upsample_2 = tf.keras.layers.UpSampling2D(2, interpolation='bilinear')
        self.conv2_3 = tf.keras.layers.Convolution2D(self.num_features * 2, 3, 1, 'same', activation='relu', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.conv2_4 = tf.keras.layers.Convolution2D(self.num_features * 2, 3, 1, 'same', activation='relu', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.upsample_1 = tf.keras.layers.UpSampling2D(2, interpolation='bilinear')
        self.conv1_3 = tf.keras.layers.Convolution2D(self.num_features, 3, 1, 'same', activation='relu', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.conv1_4 = tf.keras.layers.Convolution2D(self.num_features, 3, 1, 'same', activation='relu', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.conv_out = tf.keras.layers.Convolution2D(1, 1, 1, 'same', activation='softmax', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)

    def call(self, x, training=True):

        x = self.conv_in(x)
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x1 = self.max_pool1(x)
        x1 = self.conv2_1(x1)
        x1 = self.conv2_2(x1)
        x2 = self.max_pool2(x1)
        x2 = self.conv3_1(x2)
        x2 = self.conv3_2(x2)
        x3 = self.max_pool3(x3)
        x3 = self.conv4_1(x3)
        x3 = self.conv4_2(x3)
        x3 = self.upsample_3(x3)
        x2 = tf.keras.layers.concatenate([x2, x3])
        x2 = self.conv3_3(x2)
        x2 = self.conv3_4(x2)
        x2 = self.upsample_2(x2)
        x1 = tf.keras.layers.concatenate([x1, x2])
        x1 = self.conv2_3(x1)
        x1 = self.conv2_4(x1)
        x1 = self.upsample_1(x1)
        x = tf.keras.layers.concatenate([x, x1])
        x = self.conv1_3(x)
        x = self.conv1_4(x)
        x = self.conv_out(x)

        return x
