import os
import pdb

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from Archs import create_model

class Solver2(object):

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
        self.decay_steps = self.train_opt['decay_steps']
        self.save_steps = self.train_opt['save_steps']
        self.learning_rate = tf.Variable(self.learning_rate_init, name='learning_rate')
        self.global_step = tf.Variable(1, name='global_step')
        self.max_steps = self.train_opt['max_steps']
        self.resume = self.train_opt['resume']
        self.eval_steps = self.train_opt['eval_steps']

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
        self.model = create_model(self.opt)

    def _dice_loss(self, labels, preds):
        
        eps = 1e-3
        numerator = tf.math.reduce_sum(labels * preds)
        denominator = tf.math.reduce_sum(tf.math.square(labels) + tf.math.square(preds))
        dice_loss = 1.0 - 2.0 * (numerator + eps) / (denominator + eps)

        return dice_loss
    
    def _pixelwise_cross_entropy_loss(self, labels, preds):

        pixelwise_bce = tf.keras.losses.binary_crossentropy(labels, preds)
        pixelwise_bce = tf.math.reduce_mean(pixelwise_bce)
        
        return pixelwise_bce

    def _dice_metric(self, labels, preds):

        #pdb.set_trace()
        eps = 1e-3
        preds = tf.math.sign(preds - 0.5) * 0.5 + 0.5
        numerator = tf.math.reduce_sum(labels * preds)
        denominator = tf.math.reduce_sum(labels + preds)
        dice_metric = 2.0 * (numerator + eps) / (denominator + eps)

        return dice_metric

    def metric(self, labels, preds):

        return self._dice_metric(labels, preds)

    def loss(self, labels, preds):

        # return self._dice_loss(labels, preds)
        return self._pixelwise_cross_entropy_loss(labels, preds)

    def _parse_function(self, example_proto):
        
        features = {'image': tf.FixedLenFeature((), tf.string, ''), 'label': tf.FixedLenFeature((), tf.string, '')}
        
        parsed_features = tf.parse_single_example(example_proto, features)
        
        lbl_raw = parsed_features['label']
        label = tf.decode_raw(lbl_raw, tf.uint8)
        label = tf.reshape(label, [self.img_size, self.img_size, 1])
        label = tf.cast(label, tf.float32)

        img_raw = parsed_features['image']
        image = tf.decode_raw(img_raw, tf.uint8)
        image = tf.reshape(image, [self.img_size, self.img_size, self.img_channels])
        image = tf.cast(image, tf.float32)
        
        return image, label
    
    def _augment(self, images, labels):

        bundle = tf.concat([images, labels], axis=2)
        bundle = tf.image.random_flip_left_right(bundle)
        bundle = tf.image.random_flip_up_down(bundle)
        images = bundle[:,:,0:3]
        labels = bundle[:,:,3:]
        #images = tf.image.random_brightness(images, 50)

        return images, labels
    
    def _normalize_image(self, image_in):

        image_out = image_in / 127.5 - 1

        return image_out

    def _denormalize_image(self, image_in):

        image_out = (image_in + 1) / 2
        image_out = tf.image.convert_image_dtype(image_out, tf.uint8, True)

        return image_out

    def _save_checkpoint(self):
        
        checkpoint_dir = self.train_opt['ckpt_dir'] + '%03d/' % self.train_opt['ckpt_id']
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        print('===> Saving checkpoint to [%s]' % checkpoint_dir)
        ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model, optimizer_step=self.global_step)
        ckpt.save(checkpoint_prefix)
    
    def _load_checkpoint(self):

        checkpoint_dir = self.train_opt['ckpt_dir'] + '%03d/' % self.train_opt['ckpt_id']
        print('===> Loading checkpoint from [%s]' % checkpoint_dir)
        ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model, optimizer_step=self.global_step)
        ckpt.restore(tf.train.latest_checkpoint(checkpoint_dir))

    def train(self):
        
        print('===> Training starts.')

        if self.resume:
            self._load_checkpoint()
        
        for step, train_batch in enumerate(self.train_dataset):
            
            glsp = self.global_step.numpy()
            
            if glsp > self.max_steps:
                break
            
            images, labels = train_batch
            
            #background_labels = labels
            foreground_labels = 1 - labels
            
            images = self._normalize_image(images)            

            with tf.GradientTape() as tape:
                foreground_preds = self.model(images, True)
                #foreground_preds, background_preds = preds[:,:,:,0:1], preds[:,:,:,1:]
                #loss = self.loss(foreground_labels, foreground_preds) + self.loss(background_labels, background_preds)
                loss = self.loss(foreground_labels, foreground_preds)

            train_fg_dice = self.metric(foreground_labels, foreground_preds)
            #train_bg_dice = self.metric(background_labels, background_preds)
            
            grads = tape.gradient(loss, self.model.variables)
            
            self.optimizer.apply_gradients(zip(grads, self.model.variables), global_step=self.global_step)
            
            #print('[Step %d] Training loss = %.4f, Training ForegroundDice = %.2f, Training BackgroundDice = %.2f.' % (self.global_step, loss, train_fg_dice, train_bg_dice), end='\r')
            print('[Step %d] Training loss = %.4f, Training ForegroundDice = %.2f.' % (self.global_step, loss, train_fg_dice), end='\r')

            if glsp % self.eval_steps == 0:
                
                print('\n')
                val_fg_dice = []
                #val_bg_dice = []
                for val_img, val_lbl in self.val_dataset:
                    
                    val_img = self._normalize_image(val_img)
                    val_prd = self.model(val_img, False)
                    val_fg_dice.append(self.metric(1 - val_lbl, val_prd).numpy())
                    #val_bg_dice.append(self.metric(val_lbl, val_prd[:,:,:,1:]).numpy())

                val_fg_dice = np.array(val_fg_dice).mean()
                #val_bg_dice = np.array(val_bg_dice).mean()

                #print('Validation ForegroundDice = %.2f, Validation BackgroundDice = %.2f.' % (val_fg_dice, val_bg_dice))
                print('Validation ForegroundDice = %.2f.' % (val_fg_dice))

            if glsp % self.save_steps == 0:
                self._save_checkpoint()

            self.learning_rate.assign(tf.train.exponential_decay(self.learning_rate_init, self.global_step, self.decay_steps, self.learning_rate_decay, True)())

        print('===> Training ends.')

        
    def _load_raw_image(self, path):

        img = cv2.imread(path)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        oshape = img.shape
        img = tf.convert_to_tensor(img)
        img = tf.reshape(img, [1, self.img_size, self.img_size, self.img_channels])
        img = tf.cast(img, tf.float32)

        return img, oshape
    
    def _load_raw_label(self, path):

        lbl = cv2.imread(path)
        lbl = cv2.resize(lbl, (self.img_size, self.img_size))
        if lbl.shape[2] > 3:
            lbl = cv2.cvtColor(lbl, cv2.COLOR_BGRA2BGR)
        lbl = cv2.cvtColor(lbl, cv2.COLOR_BGR2GRAY)
        lbl[lbl > 127.5] = 255.0
        lbl[lbl < 127.5] = 0.0
        lbl = tf.convert_to_tensor(lbl)
        lbl = tf.reshape(lbl, [1, self.img_size, self.img_size, 1])
        lbl = tf.cast(lbl, tf.float32)

        return lbl

    def _show_image(self, img_tensor):

        n, h, w, c = img_tensor.shape[0], img_tensor.shape[1], img_tensor.shape[2], img_tensor.shape[3]

        img_tensor = tf.reshape(img_tensor, [h, w, c])
        img = img_tensor.numpy().astype(np.uint8)
        if c == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        plt.figure()
        plt.imshow(img)
        plt.show()
    
    def _show_all(self, image_tensor, label_tensor, pred_tensor):

        h, w, c = self.img_size, self.img_size, self.img_channels

        image_tensor = tf.reshape(image_tensor, [h, w, c])
        label_tensor = tf.reshape(label_tensor, [h, w, 1])
        pred_tensor = tf.reshape(pred_tensor, [h, w, 1])

        img = image_tensor.numpy().astype(np.uint8)
        lbl = label_tensor.numpy().astype(np.uint8)
        prd = pred_tensor.numpy().astype(np.uint8)

        lbl = cv2.cvtColor(lbl, cv2.COLOR_GRAY2RGB)
        prd = cv2.cvtColor(prd, cv2.COLOR_GRAY2RGB)

        plt.figure(figsize=(12, 6))
        plt.subplot(131)
        plt.imshow(img)
        plt.subplot(132)
        plt.imshow(prd)
        plt.subplot(133)
        plt.imshow(lbl)
        plt.show()

    def _hard_predict(self, pred):

        hard_map = tf.math.sign(pred - 0.5) * 0.5 + 0.5
        hard_map = tf.image.convert_image_dtype(hard_map, tf.uint8, True)

        return hard_map

    def test(self):

        print('===> Testing starts.')

        self._load_checkpoint()

        test_list_path = self.data_opt['test']['dir']

        with open(test_list_path, 'r') as test_list:
            test_fns = test_list.readlines()
            test_fns = list(map(lambda x: x.strip(), test_fns))
        
        #test_fns = sorted(test_fns)
        test_acc = []
        for idx in range(0, len(test_fns), 2):
            
            img_name = test_fns[idx]
            lbl_name = test_fns[idx + 1]
            
            image, oshape = self._load_raw_image(img_name)
            label = self._load_raw_label(lbl_name)

            image = self._normalize_image(image)
            
            pred = self.model(image, False)
            pred_fg = pred[:,:,:,0:1]

            dice_fg = self.metric(1 - label, pred_fg)
            
            image = self._denormalize_image(image)
            prediction = self._hard_predict(1.0 - pred_fg)
            self._show_all(image, label, prediction)
            
            dice_fg = dice_fg.numpy()

            test_acc.append(dice_fg)

            prog = idx / len(test_fns) * 100
            print('[%dth Sample] Testing Dice = %.2f, Progress = %.2f%%.' % ((idx + 1), dice_fg, prog), end='\r')
        
        test_acc = np.array(test_acc).mean()
        
        print('\n')
        print('Total Dice = %.2f.' % test_acc)
        
        print('===> Testing ends.')
