import tensorflow as tf
import numpy as np

from gyro_dropout import gyro_dropout

FLAGS = tf.app.flags.FLAGS


class LeNet():
    def __init__(self, num_classes=100):
        self.num_classes = num_classes
        self.fc_size = 1024

        self.gyro_dropout_fc1_mask_list = (np.random.uniform(size=(FLAGS.num_subnets, self.fc_size)) > FLAGS.drop_prob).astype(float)
        self.gyro_dropout_fc2_mask_list = (np.random.uniform(size=(FLAGS.num_subnets, self.fc_size)) > FLAGS.drop_prob).astype(float)

        self.mask_shape = [FLAGS.masks_per_batch, self.fc_size]
        self.gyro_dropout_fc1_mask = None
        self.gyro_dropout_fc2_mask = None

        self.do_conventional_dropout = None
        self.do_gyro_dropout = None

    def get_gyro_dropout_masks(self):
        moment_fc1_mask = self.gyro_dropout_fc1_mask_list[np.random.randint(0, len(self.gyro_dropout_fc1_mask_list), FLAGS.masks_per_batch)]
        moment_fc2_mask = self.gyro_dropout_fc2_mask_list[np.random.randint(0, len(self.gyro_dropout_fc2_mask_list), FLAGS.masks_per_batch)]

        return moment_fc1_mask, moment_fc2_mask

    def build_network(self, train_images, train_labels, eval_images, eval_labels):
        self.do_conventional_dropout = tf.placeholder(tf.bool)
        self.do_gyro_dropout = tf.placeholder(tf.bool)
        self.gyro_dropout_fc1_mask = tf.placeholder(dtype=tf.float32, shape=self.mask_shape)
        self.gyro_dropout_fc2_mask = tf.placeholder(dtype=tf.float32, shape=self.mask_shape)

        init = tf.contrib.layers.xavier_initializer()

        with tf.variable_scope('conv1') as scope:
            with tf.device('/gpu'):
                kernel1 = tf.Variable(init([5, 5, 3, 96]))

            conv1 = tf.nn.relu(tf.nn.conv2d(train_images, kernel1, [1, 1, 1, 1], padding='SAME'))

            conv1_eval = tf.nn.relu(tf.nn.conv2d(eval_images, kernel1, [1, 1, 1, 1], padding='SAME'))

        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                             padding='SAME', name='pool1')
        pool1_eval = tf.nn.max_pool(conv1_eval, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                             padding='SAME', name='pool1')

        with tf.variable_scope('conv2') as scope:
            with tf.device('/gpu'):
                kernel2 = tf.Variable(init([5, 5, 96, 256]))
            conv2 = tf.nn.relu(tf.nn.conv2d(pool1, kernel2, [1, 1, 1, 1], padding='SAME'))

            conv2_eval = tf.nn.relu(tf.nn.conv2d(pool1_eval, kernel2, [1, 1, 1, 1], padding='SAME'))

        with tf.name_scope("fc1"):
            flatten_size = np.prod(conv2.get_shape().as_list()[1:])

            conv2_flatten = tf.reshape(conv2, [-1, flatten_size])
            conv2_flatten_eval = tf.reshape(conv2_eval, [-1, flatten_size])

            with tf.device('/gpu'):
                w1 = tf.Variable(init([flatten_size, self.fc_size]))
            layer1 = tf.nn.relu(tf.matmul(conv2_flatten, w1))
            layer1_drop = tf.case([(self.do_conventional_dropout, lambda: tf.nn.dropout(layer1, keep_prob=(1-FLAGS.drop_prob))),
                                   (self.do_gyro_dropout, lambda: gyro_dropout(layer1, self.gyro_dropout_fc1_mask))],
                                   default=lambda: layer1)

            layer1_eval = tf.nn.relu(tf.matmul(conv2_flatten_eval, w1))

        with tf.name_scope("fc2"):
            with tf.device('/gpu'):
                w2 = tf.Variable(init([self.fc_size, self.fc_size]))
            layer2 = tf.nn.relu(tf.matmul(layer1_drop, w2))
            layer2_drop = tf.case([(self.do_conventional_dropout, lambda: tf.nn.dropout(layer2, keep_prob=(1-FLAGS.drop_prob))),
                                   (self.do_gyro_dropout, lambda: gyro_dropout(layer2, self.gyro_dropout_fc2_mask))],
                                   default=lambda: layer2)

            layer2_eval = tf.nn.relu(tf.matmul(layer1_eval, w2))

        with tf.name_scope("fc3"):
            with tf.device('/gpu'):
                w3 = tf.Variable(init([self.fc_size, self.num_classes]))
            layer3 = tf.matmul(layer2_drop, w3)

            layer3_eval = tf.matmul(layer2_eval, w3)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer3, labels=train_labels))
        is_correct = tf.equal(tf.argmax(layer3_eval, 1), tf.argmax(eval_labels, 1))
        
        return cost, is_correct
