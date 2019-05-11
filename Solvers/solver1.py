import tensorflow as tf

from Archs import create_model

class Solver1(object):

    def __init__(self, opt):

        self.opt = opt
        self.train_opt = opt['solver']
        self.data_opt = opt['datasets']
        self.model_opt = opt['networks']

        # Image Properties
        self.img_size = opt['img_size']
        self.img_channels = opt['img_channels']

        # Training Options
        self.batch_size = self.train_opt['batch_size']
        self.learning_rate_init = self.train_opt['lr_init']
        self.optimizer_type = self.train_opt['optimizer_type'].upper()
        self.learning_rate_decay = self.train_opt['lr_decay']
        self.decay_epochs = self.train_opt['decay_epochs']
        self.save_epochs = self.train_opt['save_epochs']
        self.learning_rate = tf.Variable(self.learning_rate_init, name='learning_rate')
        self.max_steps = self.train_opt['max_steps']

        if self.optimizer_type == 'ADAM':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif self.optimizer_type == 'MOMENTUM':
            self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.95)
        elif self.optimizer_type == 'RMSPROP':
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        elif self.optimizer_type == 'SGD':
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        else:
            raise NotImplementedError('Optimizer [%s] not supported!' % self.optimizer_type)
        
        # Dataset Options
        self.cpu_threads = self.opt['cpu_threads']
        self.train_dir = self.data_opt['train']['dir']
        self.train_dataset = tf.data.TFRecordDataset([self.train_dir], None, None, self.cpu_threads)
        self.train_dataset = self.train_dataset.map(self._parse_function, self.cpu_threads).map(self._augment, self.cpu_threads)
        self.train_dataset = self.train_dataset.repeat(-1).shuffle(5000).batch(self.batch_size).prefetch(self.batch_size)
        self.val_dir = self.data_opt['val']['dir']
        self.val_dataset = tf.data.TFRecordDataset([self.val_dir], None, None, self.cpu_threads)
        self.val_dataset = self.val_dataset.map(self._parse_function, self.cpu_threads).batch(self.batch_size).prefetch(self.batch_size)

        # Model Options
        self.model = create_model(self.model_opt)

    def _parse_function(self, example_proto):
        
        features = {'image': tf.FixedLenFeature((), tf.string, ''), 'label': tf.FixedLenFeature((), tf.int64, -1)}
        
        parsed_features = tf.parse_single_example(example_proto, features)
        
        lbl = parsed_features['label']
        img_raw = parsed_features['image']
        
        image = tf.decode_raw(img_raw, tf.uint8)
        image = tf.reshape(image, [self.img_size, self.img_size, self.img_channels])
        label = tf.cast(lbl, tf.int32)
        
        return image, label
    
    def _normalize_image(self, image_in):

        image_out = image_in / 127.5 - 1

        return image_out

    def _denormalize_image(self, image_in):

        image_out = (image_in + 1) * 127.5
        image_out[image_out > 255.0] = 255.0
        image_out[image_out < 0.0] = 0.0

        return image_out
    
    def _augment(self, batch):

        images, labels = batch
        images = tf.image.random_flip_left_right(images)
        images = tf.image.random_flip_up_down(images)
        images = tf.image.random_brightness(images, 127.5)
        batch = (images, labels)

        return batch

    def _save_checkpoint(self):

        pass

    def train(self):

        pass
    
    def test(self):
        pass