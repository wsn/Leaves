import tensorflow as tf 
import pdb
import numpy as np
import cv2

tf.enable_eager_execution()

def _parse_function(example_proto):
    features = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
              "label": tf.FixedLenFeature((), tf.int64, default_value=-1)}
    parsed_features = tf.parse_single_example(example_proto, features)
    lbl = parsed_features['label']
    img_raw = parsed_features['image']
    image = tf.decode_raw(img_raw, tf.uint8)
    image = tf.reshape(image, [128,128,3])
    label = tf.cast(lbl, tf.int32)
    image = tf.cast(image, tf.float32)
    return image, label

tset = tf.data.TFRecordDataset(['./results/train1.tfrecord']).map(_parse_function).batch(4)

for imgs, lbls in tset:
    img = imgs[0].numpy().astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow('test', img)
    cv2.waitKey(0)
    #pdb.set_trace()