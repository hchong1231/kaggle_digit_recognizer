# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
import pandas as pd
import datasets
test_data = datasets.test
train_data = datasets.train
print("Train Num: %d" % train_data.num_examples)
print("Test Num: %d" % test_data.num_examples)
# initial weight parameters
def weight_initializer(shape):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))

# initial biases parameters
def biases_initializer(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

# conv layer
def conv2d(x, w):
    return tf.nn.conv2d(input=x, filter=w, strides=[1, 1, 1, 1], padding='SAME')

#pool layer
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def save_result(data, columns):
    """
    :param data: np.array 
    :param columns: list
    """
    index = []
    for i in range(len(data)):
        index.append(i+1)
    data = np.column_stack((index, data))
    df = pd.DataFrame(data, columns=columns)
    df.to_csv("result.csv", index=False)

x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])
# [batch, height, width, channels]
x_image = tf.reshape(x, shape=[-1, 28, 28, 1])
# Conv 1
# patch [height, width, in_channel, out_channel] out_chane
W_conv1 = weight_initializer([5, 5, 1, 32])
b_conv1 = biases_initializer([32])#[out_channel]
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
# Conv 2
W_conv2 = weight_initializer([5, 5, 32, 64])
b_conv2 = biases_initializer([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
# Full Connected Layer1
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
W_fc1 = weight_initializer([7 * 7 * 64, 256])
b_fc1 = biases_initializer([256])
h_fc1 = tf.matmul(h_pool2_flat, W_fc1)+b_fc1
h_fc1_drop_out = tf.nn.dropout(h_fc1, keep_prob=0.1)
# Full Connected Layer2
W_fc2 = weight_initializer([256, 10])
b_fc2 = biases_initializer([10])
h_fc2 = tf.matmul(h_fc1_drop_out, W_fc2)+b_fc2
y_ = tf.nn.softmax(h_fc2)
cross_entropy = -tf.reduce_sum(y*tf.log(y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# graph generate
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(501):
        images_, labels_ = train_data.next_batch()
        sess.run(train_step, feed_dict={x: images_, y: labels_})
        if i % 100 == 0:
            print("Step:%d" % i)
            print("Train accuracy: %.6lf" % accuracy.eval(feed_dict={x: images_, y: labels_}))

    y_pre = y_.eval(feed_dict={x: test_data.images})
    y_pre = np.argmax(y_pre, 1)
    save_result(y_pre, ['ImageId', 'Label'])

