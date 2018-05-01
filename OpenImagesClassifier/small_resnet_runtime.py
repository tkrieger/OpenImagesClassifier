"""Runtime for small ResNet Model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from OpenImagesClassifier import data_processing as data
from OpenImagesClassifier import config as config
from OpenImagesClassifier import small_resnet_model as rs

import tensorflow as tf


class AbstractNetwork:
    """Class that defines basic interface for using a model"""
    def __init__(self, batch_size=256, model_path=config.MODEL_SAVE_DIR):
        with tf.variable_scope("counter", reuse=True):
            counter = tf.get_variable('overall_training_cycles', shape=(), dtype=tf.int32,
                                                        trainable=False, initializer=tf.constant_initializer(0))
            self.add_train_cycle = counter + tf.constant(1, dtype=tf.int32, shape=())

        self.batch_size = batch_size
        input_ops, self.init_ops = data.create_reinitializable_iterator(batch_size)
        self.X = input_ops[0]
        self.y = input_ops[1][3]

        self.sess = None
        self.model_path = model_path

    def train(self, cycles):
        """Train Model, after each call the model is saved:
            - cycles: number of training iterations (one batch per iteration)
            returns: dictionary with training accuracy and validation acc
            """
        ## maybe it is best to return a special return object that contains all relevant data / dict
        return 0

    def validate(self, number_of_batches=1, aggregated=True):
        """Validates the model with validation dataset one batch
            - cycles: number of iteration (per iteration one batch is processed)
            returns: (mean) validation accuracy"""
        return 0

    def test(self, number_of_batches=None, aggregated=True):
        """Tests the model with test dataset
            - cycles: number of iteration (per iteration one batch is processed)
            returns: (mean) test accuracy"""
        return 0

    def predict(self, file_list):
        """Predicts classes for images given as filenames/path
            returns: list of predicted classes, in order of file_list"""
        return []

    def __enter__(self):
        return

    def __exit__(self):
        if self.sess is not None:
            self.sess.close()


class SmallResNet(AbstractNetwork):
    """Implementation for Small ResNet Model"""
    def __init__(self, batch_size=256, model_path=config.MODEL_SAVE_DIR):
        super(SmallResNet, self).__init__(batch_size, model_path)
        self.training = tf.placeholder(tf.bool, name='training')
        self.logits = rs.build_small_resnet(self.X, classes_count=len(config.CATEGORIES), training=self.training)

        # add training ops
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits)
        loss = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer()
        self.training_op = optimizer.minimize(loss)

        # add ops for prediction
        self.predictions = tf.nn.softmax(logits=self.logits)
        reset_scope = 'reset_metrics'
        # add metric ops, second return value is used for calculating mean over multiple batches, third resets stats
        self.top_1_accuracy, acc_upd, self.reset_op = create_reset_metric(tf.metrics.accuracy, reset_scope,
                                                                          labels=self.y,
                                                                          predictions=tf.argmax(self.logits))
        # todo test top metric, update and reset (one call for all!)
        self.top_3_accuracy, top_3_upd, _ = create_reset_metric(tf.metrics.mean, reset_scope,
                                                                values=tf.nn.in_top_k(self.logits, self.y, k=3))
        self.top_5_accuracy, top_5_upd, _ = create_reset_metric(tf.metrics.mean, reset_scope,
                                                                values=tf.nn.in_top_k(self.logits, self.y, k=5))

        self.update_metrics = tf.group(acc_upd, top_3_upd, top_5_upd)

        # todo add further metrics
        # saver for model
        self.saver = tf.train.Saver()

        # session management
        self.sess = tf.Session()
        tf.global_variables_initializer().run(session=self.sess)
        self.restore_model()

    def restore_model(self):
        checkpoint_path = tf.train.latest_checkpoint(self.model_path)
        if checkpoint_path is not None:
            self.saver.restore(self.sess, checkpoint_path)
            print("Loaded saved Model")

    def save_model(self):
        self.saver.save(self.sess, self.model_path + 'small_resnet.ckpt')

    def train(self, cycles):
        # enable train dataset
        self.sess.run([self.init_ops['train'], self.reset_op])
        training_hist = []

        for i in range(cycles):
            _, top_1_acc, overall_cycle = self.sess.run([self.training_op, self.top_1_accuracy, self.add_train_cycle],
                                                   feed_dict={self.training: True})
            training_hist.append({'overall_cycle': overall_cycle, 'top_1_acc': top_1_acc})

        self.save_model()

        self.sess.run(self.init_ops['validation'])
        top_1_acc, top_3_acc, top_5_acc = self.sess.run([self.top_1_accuracy, self.top_3_accuracy, self.top_5_accuracy],
                                                   feed_dict={self.training: False})
        return {'training': training_hist,
                'validation': {
                    'top_1_acc': top_1_acc,
                    'top_3_acc': top_3_acc,
                    'top_5_acc': top_5_acc
                }}

    def validate(self, number_of_batches=1, aggregated=True):
        """Runs validation for number_of_batches batches.
            If aggregated is True the metrics are aggregated overall cycles"""
        self.sess.run([self.init_ops['validation'], self.reset_op])
        return self._metrics_run(number_of_batches, aggregated)

    def test(self, number_of_batches=None, aggregated=True):
        self.sess.run([self.init_ops['test'], self.reset_op])
        return self._metrics_run(number_of_batches, aggregated)

    def _metrics_run(self, number_of_batches, aggregated):
        ops = [self.top_1_accuracy, self.top_3_accuracy, self.top_5_accuracy]
        if aggregated:
            ops.append(self.update_metrics)
        result = None
        result_list = []
        for i in range(number_of_batches):
            top_1_acc, top_3_acc, top_5_acc = self.sess.run(ops, feed_dict={self.training: False})

            result = {
                'top_1_acc': top_1_acc,
                'top_3_acc': top_3_acc,
                'top_5_acc': top_5_acc
            }
            if not aggregated:
                result_list.append(result)

        return result if aggregated else result_list


def create_reset_metric(metric, scope='reset_metrics', **metric_args):
    with tf.variable_scope(scope) as scope:
        metric_op, update_op = metric(**metric_args)
        vars = tf.contrib.framework.get_variables(
                     scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
        reset_op = tf.variables_initializer(vars)
    return metric_op, update_op, reset_op


if __name__ == '__main__':
    model = SmallResNet()