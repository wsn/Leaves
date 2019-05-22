import tensorflow as tf 

from .basic_block import ResidualBlockV1, ConvBNReLUBlock

class Hourglass4(tf.keras.Model):

    def __init__(self, num_features, initializer, weight_decay):

        super(Hourglass4, self).__init__()

        self.num_features = num_features
        self.initializer = initializer
        self.weight_decay = weight_decay
        self.regularizer = tf.keras.regularizers.l2(self.weight_decay)

        self.ds256to128 = tf.keras.layers.Convolution2D(self.num_features, 3, 2, 'same', activation='relu', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.ds128to64 = tf.keras.layers.Convolution2D(self.num_features * 2, 3, 2, 'same', activation='relu', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.ds64to32 = tf.keras.layers.Convolution2D(self.num_features * 4, 3, 2, 'same', activation='relu', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.ds32to16 = tf.keras.layers.Convolution2D(self.num_features * 8, 3, 2, 'same', activation='relu', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)

        self.us16to32 = tf.keras.layers.Convolution2DTranspose(self.num_features * 4, 3, 2, 'same', activation='relu', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.us32to64 = tf.keras.layers.Convolution2DTranspose(self.num_features * 2, 3, 2, 'same', activation='relu', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.us64to128 = tf.keras.layers.Convolution2DTranspose(self.num_features, 3, 2, 'same', activation='relu', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.us128to256 = tf.keras.layers.Convolution2DTranspose(self.num_features, 3, 2, 'same', activation='relu', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)

        self.res256_1 = ResidualBlockV1(self.num_features, self.initializer, self.weight_decay)
        self.res256_2 = ResidualBlockV1(self.num_features, self.initializer, self.weight_decay)
        self.res256_3 = ResidualBlockV1(self.num_features, self.initializer, self.weight_decay)

        self.res128_1 = ResidualBlockV1(self.num_features, self.initializer, self.weight_decay)
        self.res128_2 = ResidualBlockV1(self.num_features, self.initializer, self.weight_decay)
        self.res128_3 = ResidualBlockV1(self.num_features, self.initializer, self.weight_decay)

        self.res64_1 = ResidualBlockV1(self.num_features * 2, self.initializer, self.weight_decay)
        self.res64_2 = ResidualBlockV1(self.num_features * 2, self.initializer, self.weight_decay)
        self.res64_3 = ResidualBlockV1(self.num_features * 2, self.initializer, self.weight_decay)

        self.res32_1 = ResidualBlockV1(self.num_features * 4, self.initializer, self.weight_decay)
        self.res32_2 = ResidualBlockV1(self.num_features * 4, self.initializer, self.weight_decay)
        self.res32_3 = ResidualBlockV1(self.num_features * 4, self.initializer, self.weight_decay)

        self.res16_1 = ResidualBlockV1(self.num_features * 8, self.initializer, self.weight_decay)
        self.res16_2 = ResidualBlockV1(self.num_features * 8, self.initializer, self.weight_decay)
        self.res16_3 = ResidualBlockV1(self.num_features * 8, self.initializer, self.weight_decay)


    def call(self, x, training):

        short256 = self.res256_3(x, training)
        x = self.res256_1(x, training)
        x = self.ds256to128(x)
        x = self.res128_1(x, training)
        short128 = self.res128_3(x, training)
        x = self.ds128to64(x)
        x = self.res64_1(x, training)
        short64 = self.res64_3(x, training)
        x = self.ds64to32(x)
        x = self.res32_1(x, training)
        short32 = self.res32_1(x, training)
        x = self.ds32to16(x)
        x = self.res16_1(x, training)
        short16 = self.res16_3(x, training)
        x = self.res16_2(x, training)
        x = short16 + x
        x = self.us16to32(x)
        x = self.res32_2(x, training)
        x = short32 + x
        x = self.us32to64(x)
        x = self.res64_2(x, training)
        x = short64 + x
        x = self.us64to128(x)
        x = self.res128_2(x, training)
        x = short128 + x
        x = self.us128to256(x)
        x = self.res256_2(x, training)
        x = short256 + x

        return x

class Hourglass(tf.keras.Model):

    def __init__(self, num_features, initializer, weight_decay):

        super(Hourglass, self).__init__()

        self.num_features = num_features
        self.initializer = initializer
        self.weight_decay = weight_decay
        self.regularizer = tf.keras.regularizers.l2(self.weight_decay)

        self.conv_in = tf.keras.layers.Convolution2D(self.num_features, 3, 1, 'same', activation='relu', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.backbone = Hourglass4(self.num_features, self.initializer, self.weight_decay)
        self.conv_out = tf.keras.layers.Convolution2D(2, 1, 1, 'same', activation='softmax', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
    
    def call(self, x, training):

        x = self.conv_in(x)
        x = self.backbone(x, training)
        x = self.conv_out(x)

        return x