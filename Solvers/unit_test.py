import tensorflow as tf 

tf.enable_eager_execution()

lr = tf.Variable(2.0)
opt = tf.train.AdamOptimizer(lr)

for ep in range(10):
    lr.assign(tf.train.exponential_decay(2.0, ep, 2, 0.5, True)())
    print(opt._lr)