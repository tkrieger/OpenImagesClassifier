"""Prototype Runtime for Small ResNet Model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sqlite3
import os
import numpy as np
from OpenImagesClassifier import small_resnet_model as rs
from OpenImagesClassifier import config

# only for prototyping
import cv2
import time

def select_random_from_db(count, subset):
    X = np.zeros([count, 224, 224, 3]) # only prototype no data augmentation here
    y = np.zeros([count])
    with sqlite3.connect(config.DATABASE['filename']) as conn:
        c = conn.cursor()
        result = c.execute("""SELECT I.ImageID, D.DisplayLabelName, D.ClassNumber FROM Images I
                              INNER JOIN Labels L ON I.ImageID = L.ImageID
                              INNER JOIN Dict D ON L.LabelName = D.LabelName
                              WHERE I.Subset = ? 
                              ORDER BY random() 
                              LIMIT ?""", (subset, count))

        for i, row in enumerate(result.fetchall()):
            path = config.DATA_DIRECTORY + "/Images/{}/{}/{}.{}".format(row[1], subset, row[0], "jpg")
            if os.path.exists(path):
                image = cv2.imread(path)
                resized = cv2.resize(image, dsize=(224,224))
                resized = resized.astype(float)
                X[i] = resized / 255.0
                y[i] = row[2]

    return X, y


def trainable_network(X, y):

    logits = rs.build_small_resnet(X, 10, True)

    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name='loss')
    tf.summary.scalar('loss', loss)

    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)
    return training_op, tf.summary.merge_all()

def inference_network(X):
    logits = rs.build_small_resnet(X, 10, False)
    softmax = tf.nn.softmax(logits)
    return softmax

def train():
    time_1 = time.time()
    X_batch, y_batch = select_random_from_db(256, 'train')
    X = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name='X')
    y = tf.placeholder(tf.int32, shape=(None), name='y')

    train_op, merged = trainable_network(X, y)

    with tf.Session() as sess:

        train_writer = tf.summary.FileWriter(config.SUMMARY_DIR + '/train', sess.graph)
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        saver = tf.train.Saver()

        tf.global_variables_initializer().run()
        checkpoint_path = tf.train.latest_checkpoint(config.MODEL_SAVE_DIR)
        if checkpoint_path is not None:
            saver.restore(sess, checkpoint_path)
            print("Loaded saved Model")

        summary, _ = sess.run([merged, train_op], feed_dict={X: X_batch, y: y_batch}, options=run_options,
                              run_metadata=run_metadata)

        saved_path = saver.save(sess, config.MODEL_SAVE_DIR + '/small_resnet.ckpt')
        print("Saved model to path:", saved_path)
        train_writer.add_run_metadata(run_metadata, tag='runtime-test')
        train_writer.add_summary(summary)
        train_writer.flush()

        delta_time = time.time() - time_1
        print("Delta Time:", delta_time)


def predict():
    X_batch, y_batch = select_random_from_db(10, 'train')
    X = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name='X')

    infer_network = inference_network(X)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        tf.global_variables_initializer().run()
        checkpoint_path = tf.train.latest_checkpoint(config.MODEL_SAVE_DIR)
        if checkpoint_path is not None:
            saver.restore(sess, checkpoint_path)
            print("Loaded saved Model")

        predictions = sess.run([infer_network], feed_dict={X: X_batch})

        print(y_batch)
        print(predictions)

if __name__ == '__main__':
    train()

