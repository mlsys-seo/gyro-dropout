import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from scipy.io import loadmat

import sys
import time
import datetime
import glob

from models import AlexNet, LeNet, MLP

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model', "alexnet",
                          """Model to train.""")
tf.app.flags.DEFINE_string('dataset', "cifar10",
                          """Dataset to train.""")
tf.app.flags.DEFINE_boolean('do_augmentation', False,
                           """If the flag is true, train model with augmented dataset.""")
tf.app.flags.DEFINE_integer('batch_size', 256,
                           """Batch size for training or evaluation.""")

tf.app.flags.DEFINE_boolean('do_conventional_dropout', False,
                           """If the flag is true, train model with conventional dropout.""")
tf.app.flags.DEFINE_boolean('do_gyro_dropout', False,
                           """If the flag is true, train model with gyro dropout.""")
tf.app.flags.DEFINE_integer('num_subnets', 256,
                           """Number of total subnetworks for training with gyro dropout.""")
tf.app.flags.DEFINE_integer('masks_per_batch', 16,
                           """Co-scheduled subnetworks for training with gyro dropout.""")
tf.app.flags.DEFINE_float('drop_prob', 0.5,
                         """Drop ratio for conventional or gyro dropout.""")

tf.set_random_seed(int(time.time()))

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=5000)


def augment_data(train_X, eval_X, crop_size):
    crop_row_size = crop_size[0]
    crop_col_size = crop_size[1]
    crop_channel_size = crop_size[2]

    cropped_train_X = tf.map_fn(lambda img: tf.random_crop(img, [crop_row_size, crop_col_size, crop_channel_size]), train_X)
    flipped_train_X = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), cropped_train_X)
    augmented_train_X = tf.map_fn(lambda img: tf.image.per_image_standardization(img), flipped_train_X)

    resized_eval_X = tf.map_fn(lambda img: tf.image.resize_image_with_crop_or_pad(img, crop_row_size, crop_col_size), eval_X)
    augmented_eval_X = tf.map_fn(lambda img: tf.image.per_image_standardization(img), resized_eval_X)

    return augmented_train_X, augmented_eval_X


def train(model, train_dataset, test_dataset, train_info):
    train_epoch = train_info["epoch"]
    lr = train_info["lr"]
    train_dataset_size = train_info["train_dataset_size"]
    test_dataset_size = train_info["test_dataset_size"]
    img_rows, img_cols, img_channels = train_info["img_size"]
    num_classes = train_info["num_classes"]

    x_train, y_train = train_dataset
    x_test, y_test = test_dataset

    train_iterations_per_epoch = train_dataset_size // FLAGS.batch_size
    test_iterations = test_dataset_size // FLAGS.batch_size
    masks_train_period = (train_iterations_per_epoch * train_epoch) // (FLAGS.num_subnets // FLAGS.masks_per_batch)

    with tf.device("/cpu:0"):
        train_X = tf.placeholder(tf.float32, [None, img_rows, img_cols, img_channels])
        train_Y = tf.placeholder(tf.int64, [None, num_classes])

        eval_X = tf.placeholder(tf.float32, [None, img_rows, img_cols, img_channels])
        eval_Y = tf.placeholder(tf.int64, [None, num_classes])

    if FLAGS.do_augmentation:
        crop_size = (24, 24, img_channels)
        augmented_train_X, augmented_eval_X = augment_data(train_X, eval_X, crop_size)
        cost, is_correct = model.build_network(augmented_train_X, train_Y, augmented_eval_X, eval_Y)
    else:
        standardized_train_X = tf.map_fn(lambda img: tf.image.per_image_standardization(img), train_X)
        standardized_eval_X = tf.map_fn(lambda img: tf.image.per_image_standardization(img), eval_X)
        cost, is_correct = model.build_network(standardized_train_X, train_Y, standardized_eval_X, eval_Y)

    learning_rate = tf.placeholder(tf.float32)
    train_op = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.9, momentum=0.9, epsilon=1.0).minimize(cost)

    max_test_acc = 0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        step = 0
        for epoch in range(train_epoch):
            if (epoch == 40 or epoch == 80):
                lr = lr * 0.1

            avg_cost = 0
            x_train, y_train = shuffle(x_train, y_train)
            for s in range(train_iterations_per_epoch):
                batch_xs = x_train[FLAGS.batch_size*s:FLAGS.batch_size*(s+1)]
                batch_ys = y_train[FLAGS.batch_size*s:FLAGS.batch_size*(s+1)]

                iter_start = time.time()

                if FLAGS.model.lower() == 'mlp':
                    if FLAGS.do_gyro_dropout and step % masks_train_period == 0:
                        gyro_mask1, gyro_mask2, gyro_mask3 = model.get_gyro_dropout_masks()

                    if FLAGS.do_gyro_dropout:
                        c, _ = sess.run([cost, train_op], feed_dict={learning_rate: lr, train_X: batch_xs, train_Y: batch_ys,
                                                            model.gyro_dropout_fc1_mask: gyro_mask1,
                                                            model.gyro_dropout_fc2_mask: gyro_mask2,
                                                            model.gyro_dropout_fc3_mask: gyro_mask3,
                                                            model.do_conventional_dropout: False, model.do_gyro_dropout: True})
                    elif FLAGS.do_conventional_dropout:
                        c, _ = sess.run([cost, train_op], feed_dict={learning_rate: lr, train_X: batch_xs, train_Y: batch_ys,
                                                            model.gyro_dropout_fc1_mask: np.zeros(model.mask_shape), 
                                                            model.gyro_dropout_fc2_mask: np.zeros(model.mask_shape), 
                                                            model.gyro_dropout_fc3_mask: np.zeros(model.mask_shape),
                                                            model.do_conventional_dropout: True, model.do_gyro_dropout: False})
                    else:
                        c, _ = sess.run([cost, train_op], feed_dict={learning_rate: lr, train_X: batch_xs, train_Y: batch_ys,
                                                            model.gyro_dropout_fc1_mask: np.zeros(model.mask_shape),
                                                            model.gyro_dropout_fc2_mask: np.zeros(model.mask_shape),
                                                            model.gyro_dropout_fc3_mask: np.zeros(model.mask_shape),
                                                            model.do_conventional_dropout: False, model.do_gyro_dropout: False})
                else:
                    if FLAGS.do_gyro_dropout and step % masks_train_period == 0:
                        gyro_mask1, gyro_mask2 = model.get_gyro_dropout_masks()

                    if FLAGS.do_gyro_dropout:
                        c, _ = sess.run([cost, train_op], feed_dict={learning_rate: lr, train_X: batch_xs, train_Y: batch_ys,
                                                            model.gyro_dropout_fc1_mask: gyro_mask1,
                                                            model.gyro_dropout_fc2_mask: gyro_mask2,
                                                            model.do_conventional_dropout: False, model.do_gyro_dropout: True})
                    elif FLAGS.do_conventional_dropout:
                        c, _ = sess.run([cost, train_op], feed_dict={learning_rate: lr, train_X: batch_xs, train_Y: batch_ys,
                                                            model.gyro_dropout_fc1_mask: np.zeros(model.mask_shape),
                                                            model.gyro_dropout_fc2_mask: np.zeros(model.mask_shape),
                                                            model.do_conventional_dropout: True, model.do_gyro_dropout: False})
                    else:
                        c, _ = sess.run([cost, train_op], feed_dict={learning_rate: lr, train_X: batch_xs, train_Y: batch_ys,
                                                            model.gyro_dropout_fc1_mask: np.zeros(model.mask_shape),
                                                            model.gyro_dropout_fc2_mask: np.zeros(model.mask_shape),
                                                            model.do_conventional_dropout: False, model.do_gyro_dropout: False})

                avg_cost += c / train_iterations_per_epoch
                step += 1
            print("[Epoch: ", "{:>4d}] Cost: {:.9f}".format(epoch+1, avg_cost), end=' ')

            total_preds = 0        
            for s in range(test_iterations):
                batch_xs = x_test[FLAGS.batch_size*s:FLAGS.batch_size*(s+1)]
                batch_ys = y_test[FLAGS.batch_size*s:FLAGS.batch_size*(s+1)]

                if FLAGS.model.lower() == "mlp":
                    preds = sess.run([is_correct], feed_dict={eval_X: batch_xs, eval_Y: batch_ys,
                                                            model.gyro_dropout_fc1_mask: np.zeros(model.mask_shape),
                                                            model.gyro_dropout_fc2_mask: np.zeros(model.mask_shape),
                                                            model.gyro_dropout_fc3_mask: np.zeros(model.mask_shape),
                                                            model.do_conventional_dropout: False, model.do_gyro_dropout: False})
                else:
                    preds = sess.run([is_correct], feed_dict={eval_X: batch_xs, eval_Y: batch_ys,
                                                            model.gyro_dropout_fc1_mask: np.zeros(model.mask_shape),
                                                            model.gyro_dropout_fc2_mask: np.zeros(model.mask_shape),
                                                            model.do_conventional_dropout: False, model.do_gyro_dropout: False})
                total_preds += np.sum(preds)
            test_acc = total_preds / (test_iterations * FLAGS.batch_size) * 100.
            print("\tTest accuracy: {:.4f}%".format(test_acc))
            
            if max_test_acc < test_acc:
                max_test_acc = test_acc

        print("Best Test accuracy: {:.4f}%".format(max_test_acc))
            

if __name__ == '__main__':
    print("Model :", FLAGS.model)
    print("Dataset :", FLAGS.dataset)
    if FLAGS.do_conventional_dropout:
        print("Dropout : Conventional")
        print("Drop prob :", FLAGS.drop_prob)
        method = "conventional"
    elif FLAGS.do_gyro_dropout:
        print("Dropout : Gyro")
        print("Drop prob :", FLAGS.drop_prob)
        print("Num subnets :", FLAGS.num_subnets)
        print("Masks per batch :", FLAGS.masks_per_batch)
        method = "gyro"
    else:
        print("Dropout : No")
        method = "nodropout"

    if FLAGS.dataset.lower() == "cifar10":
        num_classes = 10

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        x_train = x_train.astype(np.float32)
        y_train = tf.keras.utils.to_categorical(y_train, num_classes, dtype='int64')

        x_test = x_test.astype(np.float32)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes, dtype='int64')
    elif FLAGS.dataset.lower() == "cifar100":
        num_classes = 100

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

        x_train = x_train.astype(np.float32)
        y_train = tf.keras.utils.to_categorical(y_train, num_classes, dtype='int64')

        x_test = x_test.astype(np.float32)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes, dtype='int64')
    else:
        print("Please specify dataset. Exit...")
        exit(1)

    if FLAGS.model.lower() == "alexnet":
        model = AlexNet(num_classes)
    elif FLAGS.model.lower() == "lenet":
        model = LeNet(num_classes)
    elif FLAGS.model.lower() == "mlp":
        model = MLP(num_classes)
    else:
        print("Please specify model. Exit...")
        exit(1)

    train_info = {
        "epoch": 100,
        "lr": 0.007,
        "train_dataset_size": x_train.shape[0],
        "test_dataset_size": x_test.shape[0],
        "img_size": x_train.shape[1:],
        "num_classes": num_classes,
    }
    train_dataset = (x_train, y_train)
    test_dataset = (x_test, y_test)
    train(model, train_dataset, test_dataset, train_info)
