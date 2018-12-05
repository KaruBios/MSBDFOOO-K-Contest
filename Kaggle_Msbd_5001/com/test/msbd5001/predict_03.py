import pandas as pd

import tensorflow as tf

from com.test.msbd5001.train_model_02 import preprocess_data
from com.test.msbd5001.methods import one_hot_encoder

test_csv_file = pd.read_csv('test.csv')

test_x_axis = preprocess_data(test_csv_file)
test_x_axis = one_hot_encoder(test_x_axis)

print(len(test_x_axis))

column_count = len(test_x_axis[0])

# print(len(test_x_axis))

with tf.Session() as sess:
    x_p = tf.placeholder(tf.float32, [None, column_count])

    new_saver = tf.train.import_meta_graph('./checkpoints/kaggle_model-15000.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./checkpoints'))
    # print(sess.run('weights-output:0'))

    graph = tf.get_default_graph()
    op = graph.get_tensor_by_name('raw-output:0')
    print(len(sess.run(op, feed_dict={x_p: test_x_axis})))
