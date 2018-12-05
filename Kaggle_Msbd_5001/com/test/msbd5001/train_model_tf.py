import math
import numpy as np
import pandas as pd
import os
import csv

import tensorflow as tf

from com.test.msbd5001.train_model_02 import preprocess_data, normalize_data

from com.test.msbd5001.methods import one_hot_encoder, get_train_and_valid_set, k_fold

MAX_ITER = 10001
LEARNING_RATE = 0.05

test_csv_file = pd.read_csv('test.csv')

test_x_axis = preprocess_data(test_csv_file)
test_x_axis = one_hot_encoder(test_x_axis)

# print(len(test_x_axis))

csv_file = pd.read_csv('train.csv')
standard_features = preprocess_data(csv_file)

x_axis = preprocess_data(train_features=csv_file)
x_axis = normalize_data(x_axis, standard_features)
# print(x_axis['time'])

x_axis = one_hot_encoder(x_axis)
# x_axis = [x_axis]

column_count = len(x_axis[0])
# print(column_count)

y_axis = csv_file['time'].values.tolist()
# y_axis = [y_axis]
y_axis = [[y] for y in y_axis]
# y_axis = np.array(y_axis).reshape((len(y_axis), 1))
# y_log_axis = np.log(csv_file['time']).values.tolist()

o_axis = x_axis.copy()

x_axis, y_axis, x_valid, y_valid = get_train_and_valid_set(x_axis, y_axis, 0.2)
# x_axis, y_log_axis, x_test, y_log_test = get_train_and_valid_set(x_axis, y_log_axis, 0)

k = 10

x_k_list = k_fold(x_axis, k)
y_k_list = k_fold(y_axis, k)


# y_log_k_list = k_fold(y_log_axis, k)


def add_layer(inputs, input_size, output_size, activation_func=None, layer_no=''):
    weights = tf.Variable(tf.random_normal([input_size, output_size]), name='weights-' + layer_no)
    # print(self.weights.shape)
    biases = tf.Variable(tf.zeros([1, output_size]) + 0.1, name='biases-' + layer_no)
    # print(self.biases.shape)

    outputs = tf.add(tf.matmul(inputs, weights), biases, name='raw-' + layer_no)
    if activation_func is not None:
        outputs = activation_func(outputs)

    # print(layer_no, outputs.get_shape()[1])

    return outputs

    # print(self.outputs.shape)


def dropout(inputs, layer_no=''):
    drop_out = tf.nn.dropout(inputs, 0.3, name='drop-' + layer_no)

    return drop_out


# x_p = tf.placeholder(tf.float32, [None, column_count], name='x_axis')
x_p = tf.placeholder(tf.float32, [None, column_count], name='x_axis')
y_p = tf.placeholder(tf.float32, [None, 1], name='y_axis')
# x_p = tf.placeholder(tf.float32, name='x_axis')
# y_p = tf.placeholder(tf.float32, name='y_axis')

layer_1 = add_layer(x_p, column_count, 32, tf.nn.relu, '1')
layer_2 = add_layer(layer_1, 32, 16, tf.nn.relu, '2')
# layer_2 = dropout(layer_2, '2')
layer_3 = add_layer(layer_2, 16, 8, tf.nn.relu, '3')
# drop_3 = dropout(layer_3, '3')
layer_4 = add_layer(layer_3, 8, 8, tf.nn.relu, '4')
# layer_4 = dropout(layer_4, '4')
layer_5 = add_layer(layer_4, 8, 4, tf.nn.relu, '5')
# layer_5 = dropout(layer_5, '5')
# layer_6 = add_layer(layer_5, 64, 32, tf.nn.relu, '6')

output_layer = add_layer(layer_5, 4, 1, None, 'output')

loss = tf.reduce_mean(tf.square(y_p - output_layer))
# train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # print(sess.run(layer_1.weights))
    # print(sess.run(layer_2.weights))
    # print(sess.run(layer_3.weights))
    # print()

    saver = tf.train.Saver()

    for i in range(MAX_ITER):

        # print(len(x_k_list[i % k]['test']))
        # print(len(y_k_list[i % k]['test']))

        # sess.run(train_step, feed_dict={x_p: x_axis, y_p: y_axis})

        sess.run(train_step,
                 feed_dict={x_p: x_k_list[int(i / 100) % k]['train'], y_p: y_k_list[int(i / 100) % k]['train']})

        if i % 500 == 0:
            total_loss = sess.run(loss,
                                  feed_dict={x_p: x_k_list[int(i / 100) % k]['test'],
                                             y_p: y_k_list[int(i / 100) % k]['test']})
            # total_loss = sess.run(loss, feed_dict={x_p: x_axis, y_p: y_axis})
            # print(y_axis)
            # y_pred = sess.run(output_layer.outputs, feed_dict={x_p: test_x_axis})
            # y_pred = sess.run(output_layer.outputs, feed_dict={x_p: [[0 for _ in range(column_count)]]})
            # for j, y in enumerate(y_pred):
            #     print(j, y)

            print(i, ':', total_loss)
            if total_loss < 0.5:
                break

            # if not os.path.exists('checkpoints'):
            #     os.mkdir('checkpoints')
            # saver.save(sess, './checkpoints/kaggle_model', global_step=i)

    # saver.save(sess, './checkpoints/kaggle_model', global_step=MAX_ITER)

    # x_p_1 = tf.placeholder(tf.float32, [None, column_count])
    # y_pred = sess.run(output_layer.outputs, feed_dict={x_p: [[0 for _ in range(column_count)]]})
    # y_pred = output_layer.outputs.eval(feed_dict={x_p: test_x_axis})
    print()
    # y_pred = output_layer.outputs.eval(feed_dict={x_p: test_x_axis})
    # y_pred = output_layer.outputs.eval(feed_dict={x_p: o_axis})
    # print(len(y_pred))
    # for i, y in enumerate(y_pred):
    #     print(i, y)

    valid_loss = sess.run(loss, feed_dict={x_p: x_valid, y_p: y_valid})
    print(valid_loss)

    y_pred = output_layer.eval(feed_dict={x_p: test_x_axis})
    y_pred = [0 if y < 0 else y for y in y_pred]
    # for i in range(len(y_pred)):
    #     print(i, y_pred[i])

    y_pred = [[str(i), y_pred[i][0]] for i in range(len(y_pred))]
    # print(y_pred)

    head = ['id', 'time']

    with open('test_label_07.csv', 'w', newline='') as p_file:
        writer = csv.writer(p_file)
        writer.writerow(head)
        writer.writerows(y_pred)

# with tf.Session() as sess:
#     y_pred = sess.run(output_layer.outputs, feed_dict={x_p: test_x_axis})
#     print(len(y_pred))
