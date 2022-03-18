import tensorflow as tf
import numpy as np

from gyro_dropout import gyro_dropout

FLAGS = tf.app.flags.FLAGS


class MLP():
    def __init__(self, num_classes=100):
        self.num_classes = num_classes
        self.fc_size = 1024

        self.gyro_dropout_fc1_mask_list = (np.random.uniform(size=(FLAGS.num_subnets, self.fc_size)) > FLAGS.drop_prob).astype(float)
        self.gyro_dropout_fc2_mask_list = (np.random.uniform(size=(FLAGS.num_subnets, self.fc_size)) > FLAGS.drop_prob).astype(float)
        self.gyro_dropout_fc3_mask_list = (np.random.uniform(size=(FLAGS.num_subnets, self.fc_size)) > FLAGS.drop_prob).astype(float)

        self.mask_shape = [FLAGS.masks_per_batch, self.fc_size]
        self.gyro_dropout_fc1_mask = None 
        self.gyro_dropout_fc2_mask = None 
        self.gyro_dropout_fc3_mask = None 

        self.do_conventional_dropout = None
        self.do_gyro_dropout = None

    def get_gyro_dropout_masks(self):
        moment_fc1_mask = self.gyro_dropout_fc1_mask_list[np.random.randint(0, len(self.gyro_dropout_fc1_mask_list), FLAGS.masks_per_batch)]
        moment_fc2_mask = self.gyro_dropout_fc2_mask_list[np.random.randint(0, len(self.gyro_dropout_fc2_mask_list), FLAGS.masks_per_batch)]
        moment_fc3_mask = self.gyro_dropout_fc3_mask_list[np.random.randint(0, len(self.gyro_dropout_fc3_mask_list), FLAGS.masks_per_batch)]

        return moment_fc1_mask, moment_fc2_mask, moment_fc3_mask

    def build_network(self, train_images, train_labels, eval_images, eval_labels):
        self.do_conventional_dropout = tf.placeholder(tf.bool)
        self.do_gyro_dropout = tf.placeholder(tf.bool)
        self.gyro_dropout_fc1_mask = tf.placeholder(dtype=tf.float32, shape=self.mask_shape)
        self.gyro_dropout_fc2_mask = tf.placeholder(dtype=tf.float32, shape=self.mask_shape)
        self.gyro_dropout_fc3_mask = tf.placeholder(dtype=tf.float32, shape=self.mask_shape)

        init = tf.contrib.layers.xavier_initializer()

        input_size = np.prod(train_images.get_shape().as_list()[1:])
        print(input_size)
        train_images_flatten = tf.reshape(train_images, [-1, input_size])
        eval_images_flatten = tf.reshape(eval_images, [-1, input_size])
        with tf.variable_scope("fc1"):
            with tf.device('/gpu'):
                w1 = tf.Variable(init([input_size, self.fc_size]))
            layer1 = tf.nn.relu(tf.matmul(train_images_flatten, w1))
            layer1_drop = tf.case([(self.do_conventional_dropout, lambda: tf.nn.dropout(layer1, keep_prob=(1-FLAGS.drop_prob))),
                                   (self.do_gyro_dropout, lambda: gyro_dropout(layer1, self.gyro_dropout_fc1_mask))],
                                   default=lambda: layer1)

            layer1_eval = tf.nn.relu(tf.matmul(eval_images_flatten, w1))

        with tf.variable_scope("fc2"):
            with tf.device('/gpu'):
                w2 = tf.Variable(init([self.fc_size, self.fc_size]))
            layer2 = tf.nn.relu(tf.matmul(layer1_drop, w2))
            layer2_drop = tf.case([(self.do_conventional_dropout, lambda: tf.nn.dropout(layer2, keep_prob=(1-FLAGS.drop_prob))),
                                   (self.do_gyro_dropout, lambda: gyro_dropout(layer2, self.gyro_dropout_fc2_mask))],
                                   default=lambda: layer2)

            layer2_eval = tf.nn.relu(tf.matmul(layer1_eval, w2))

        with tf.variable_scope("fc3"):
            with tf.device('/gpu'):
                w3 = tf.Variable(init([self.fc_size, self.fc_size]))
            layer3 = tf.nn.relu(tf.matmul(layer2_drop, w3))
            layer3_drop = tf.case([(self.do_conventional_dropout, lambda: tf.nn.dropout(layer3, keep_prob=(1-FLAGS.drop_prob))),
                                   (self.do_gyro_dropout, lambda: gyro_dropout(layer3, self.gyro_dropout_fc3_mask))],
                                   default=lambda: layer3)

            layer3_eval = tf.nn.relu(tf.matmul(layer2_eval, w3))

        with tf.name_scope("fc4"):
            with tf.device('/gpu'):
                w4 = tf.Variable(init([self.fc_size, self.num_classes]))
            layer4 = tf.matmul(layer3_drop, w4)
            layer4_eval = tf.matmul(layer3_eval, w4)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer4, labels=train_labels))
        is_correct = tf.equal(tf.argmax(layer4_eval, 1), tf.argmax(eval_labels, 1))

        return cost, is_correct
