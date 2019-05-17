import tensorflow as tf 
import argparse
import os
import pdb
import cv2

import numpy as np
from PIL import Image


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int, default=1, help='Specify the task to prepare datasets.')
    parser.add_argument('--size', type=int, default=256, help='Specify the target data width and height.')
    parser.add_argument('--vals', type=int, default=50, help='Specify the number of validation images per class.')
    parser.add_argument('--tests', type=int, default=50, help='Specify the number of test images per class.')
    parser.add_argument('--img_root_dir', type=str, default='./data1/', help='Specify the root directory of raw datasets.')
    parser.add_argument('--result_dir', type=str, default='./results/', help='Specify the output TFRecords directory.')
    parser.add_argument('--crop_strategy', type=str, default='DSCROP', help='Specify the method of cropping patches.')
    args = parser.parse_args()

    task_type = args.task
    root_dir = args.img_root_dir
    out_dir = args.result_dir
    img_size = args.size
    nb_val = args.vals
    nb_test = args.tests
    crps = args.crop_strategy

    if task_type == 1:
        prepare_task1(root_dir, out_dir, img_size, nb_val, nb_test)
    elif task_type == 2:
        prepare_task2(root_dir, out_dir, img_size, nb_val, nb_test, crps)
    else:
        raise NotImplementedError('Task type not supported!')

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_task1(image, label):
  
    feature = {'image': _bytes_feature(image), 'label': _int64_feature(label)}
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def serialize_task2(image, label):
  
    feature = {'image': _bytes_feature(image), 'label': _bytes_feature(label)}
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def prepare_task1(root_dir, out_dir, img_size, nb_val, nb_test):

    '''
        We suppose the raw datasets have structures like this.
        data1
        ├─task1_0
        ├─task1_1
        └─task1_2
    '''

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    train_filename = out_dir + 'train1.tfrecord'
    val_filename = out_dir + 'val1.tfrecord'
    test_filename = out_dir + 'test1.txt'

    train_dataset_writer = tf.python_io.TFRecordWriter(train_filename)
    val_dataset_writer = tf.python_io.TFRecordWriter(val_filename)
    test_dataset_writer = open(test_filename, 'w')
    
    test_prefix = './' + os.getcwd().split('/')[-1] + '/'
    
    print('Task1 preparation starts.')

    classes_dir = os.listdir(root_dir)
    for class_idx, class_dir in enumerate(classes_dir):
        full_prefix = root_dir + class_dir + '/'
        img_names = os.listdir(full_prefix)
        nb_imgs = len(img_names)
        notrain_idcs = np.random.choice(nb_imgs, nb_val + nb_test, False)
        val_idcs = notrain_idcs[0:nb_val]
        test_idcs = notrain_idcs[nb_val:nb_val + nb_test]
        for img_idx, img_name in enumerate(img_names):
            img_name_full = full_prefix + img_name
            img = Image.open(img_name_full)
            img = img.resize((img_size, img_size)).tobytes()
            if img_idx in val_idcs:
                val_dataset_writer.write(serialize_task1(img, class_idx))
            elif img_idx in test_idcs:
                img_name_full = img_name_full.replace('./', test_prefix)
                test_dataset_writer.write(img_name_full + ' %d\n' % class_idx)
            else:
                train_dataset_writer.write(serialize_task1(img, class_idx))
            print('Class[%d] %dth Image done, progress = %.2f%%.' % (class_idx, img_idx, 100 * img_idx / nb_imgs), end='\r')
    
    train_dataset_writer.close()
    val_dataset_writer.close()
    test_dataset_writer.close()

    print('Task1 preparation ends.')

def prepare_task2(root_dir, out_dir, img_size, nb_val, nb_test, crps):
    
    '''
        We suppose the raw datasets have structures like this.
        data2
        ├─leaf
        └─vein
    ''' 

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    train_filename = out_dir + 'train2.tfrecord'
    val_filename = out_dir + 'val2.tfrecord'
    test_filename = out_dir + 'test2.txt'

    train_dataset_writer = tf.python_io.TFRecordWriter(train_filename)
    val_dataset_writer = tf.python_io.TFRecordWriter(val_filename)
    test_dataset_writer = open(test_filename, 'w')

    test_prefix = './' + os.getcwd().split('/')[-1] + '/'
    
    print('Task2 preparation starts.')

    leaf_dir = root_dir + 'leaf/'
    leaves_names = os.listdir(leaf_dir)
    nb_imgs = len(leaves_names)
    notrain_idcs = np.random.choice(nb_imgs, nb_val + nb_test, False)
    val_idcs = notrain_idcs[0:nb_val]
    test_idcs = notrain_idcs[nb_val:nb_val + nb_test]

    for idx, sample_name in enumerate(leaves_names):
        full_name_leaf = leaf_dir + sample_name
        full_name_vein = full_name_leaf.replace('leaf', 'vein')
        img = Image.open(full_name_leaf)
        img = img.resize((img_size, img_size)).tobytes()
        lbl = Image.open(full_name_vein)
        lbl = lbl.resize((img_size, img_size)).convert('RGB')
        lbl = np.array(lbl)
        lbl = cv2.cvtColor(lbl, cv2.COLOR_RGB2GRAY)
        lbl[lbl < 127.5] = 0
        lbl[lbl >= 127.5] = 1
        lbl = lbl.astype(np.uint8).tobytes()
        if idx in val_idcs:
            val_dataset_writer.write(serialize_task2(img, lbl))
        elif idx in test_idcs:
            full_name_leaf = full_name_leaf.replace('./', test_prefix)
            full_name_vein = full_name_vein.replace('./', test_prefix)
            test_dataset_writer.write(full_name_leaf + '\n')
            test_dataset_writer.write(full_name_vein + '\n')
        else:
            train_dataset_writer.write(serialize_task2(img, lbl))
        print('%dth Image done, progress = %.2f%%.' % (idx, 100 * idx / nb_imgs), end='\r')
    
    train_dataset_writer.close()
    val_dataset_writer.close()
    test_dataset_writer.close()
    
    print('Task2 preparation ends.')

def gen_patches_t2(images, labels, patch_size, strategy):

    if strategy == 'DSCROP':
        pass
    elif strategy == 'CHOPCROP':
        pass
    else:
        raise NotImplementedError('Strategy [%s] not supported!' % strategy)

if __name__ == '__main__':
    main()
