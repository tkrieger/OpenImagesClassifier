"""Runtime for small ResNet Model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from OpenImagesClassifier import data_processing as data
from OpenImagesClassifier import config as config
from OpenImagesClassifier import small_resnet_model as rs
from enum import Enum

import tensorflow as tf
import time


class AbstractNetwork:
    """Class that defines basic interface for using a model"""

    def __init__(self, batch_size=256, model_path=config.MODEL_SAVE_DIR):
        with tf.variable_scope("counter", reuse=tf.AUTO_REUSE):
            self.counter = tf.get_variable('overall_training_cycles', shape=(), dtype=tf.int32,
                                           trainable=False, initializer=tf.constant_initializer(0))
            self.add_train_cycle = tf.assign_add(self.counter, 1)

        self.batch_size = batch_size
        input_ops, self.init_ops = data.create_reinitializable_iterator(batch_size)
        self.X = input_ops[0]
        self.y = input_ops[1][3]
        self.display_label = input_ops[1][2]

        # tf.summary.image('test', self.X)
        self.sess = None
        self.model_path = model_path

    def train(self, cycles):
        """Train Model, after each call the model is saved:
            - cycles: number of training iterations (one batch per iteration)
            returns: dictionary with training accuracy and validation acc
            """
        # maybe it is best to return a special return object that contains all relevant data / dict
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


# USE IN WITH BLOCK!

class SmallResNet(AbstractNetwork):
    """Implementation for Small ResNet Model"""

    def __init__(self, batch_size=256, model_path=config.MODEL_SAVE_DIR, summary_dir=config.SUMMARY_DIR):
        super(SmallResNet, self).__init__(batch_size, model_path)
        self.training = tf.placeholder(tf.bool, name='training')
        self.logits = rs.build_small_resnet(self.X, classes_count=len(config.CATEGORIES), training=self.training)

        # add training ops
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits)
        loss = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('loss', loss)
        optimizer = tf.train.AdamOptimizer()
        self.training_op = optimizer.minimize(loss)

        # add ops for prediction
        self.predictions = tf.nn.softmax(logits=self.logits)

        # add metric object manages metrics and holds needed references
        # todo test top metric, update and reset (one call for all!)
        self.metrics = Metrics()
        self.metrics.append_metric('top_1_accuracy', tf.metrics.accuracy, labels=self.y,
                                   predictions=tf.argmax(self.logits, axis=1))
        self.metrics.append_metric('top_3_accuracy', tf.metrics.mean,
                                   values=tf.nn.in_top_k(self.logits, self.y, k=3))
        self.metrics.append_metric('top_5_accuracy', tf.metrics.mean,
                                   values=tf.nn.in_top_k(self.logits, self.y, k=5))

        # todo add further metrics
        # saver for model
        self.saver = tf.train.Saver()

        # session management
        self.sess = tf.Session()
        tf.global_variables_initializer().run(session=self.sess)
        tf.local_variables_initializer().run(session=self.sess)
        self.restore_model()

        self.merged_summaries = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(summary_dir + '/train')
        self.validation_writer = tf.summary.FileWriter(summary_dir + '/validation')
        self.test_writer = tf.summary.FileWriter(summary_dir + '/test')
        self.inference_type = self.InferenceType.VALIDATION

        self.train_writer.add_graph(tf.get_default_graph())

    class InferenceType(Enum):
        VALIDATION = 1
        TEST = 2

    def __exit__(self):
        super(SmallResNet, self).__exit__()
        self.train_writer.close()
        self.validation_writer.close()
        self.test_writer.close()

    def restore_model(self):
        checkpoint_path = tf.train.latest_checkpoint(self.model_path)
        if checkpoint_path is not None:
            # checkpoint found for model, restore
            self.saver.restore(self.sess, checkpoint_path)
            print("Loaded saved Model")

    def save_model(self):
        self.saver.save(self.sess, self.model_path + '/small_resnet.ckpt')

    def train(self, cycles):
        # enable train dataset
        self.sess.run([self.init_ops['train']])
        # get top-1 accuracy from metric object
        metric = self.metrics.get_value_op('top_1_accuracy')
        training_hist = []

        for i in range(cycles):
            # reset metric for this batch
            self.sess.run(self.metrics.reset)
            # run training and update metric - update op also returns metric results
            summary, _, res, overall_cycle = self.sess.run([self.merged_summaries, self.training_op,
                                                            self.metrics.update_ops, self.add_train_cycle],
                                                           feed_dict={self.training: True})
            training_hist.append({'overall_cycle': overall_cycle, 'top_1_acc': res[0]})
            self.train_writer.add_summary(summary, global_step=overall_cycle)

        self.save_model()
        self.train_writer.flush()
        validation_results = self.validate(aggregated=False, write_summary=True)
        return {'training': training_hist,
                'validation': validation_results}

    def validate(self, number_of_batches=1, aggregated=True, write_summary=False):
        """Runs validation for number_of_batches batches.
            If aggregated is True the metrics are aggregated overall cycles
            -> For testing and validation the same actions are needed, only difference is the dataset
            -> summary writing only when aggregated=False"""
        self.sess.run([self.init_ops['validation']])
        self.inference_type = self.InferenceType.VALIDATION
        # _metrics_run processes the validation it self
        return self._metrics_run(number_of_batches, aggregated, write_summary)

    def test(self, number_of_batches=1, aggregated=True, write_summary=False):
        """Runs test for number_of_batches batches.
            If aggregated is True the metrics are aggregated overall cycles
            -> For testing and validation the same actions are needed, only difference is the dataset
            -> summary writing only when aggregated=False"""
        self.sess.run([self.init_ops['test']])
        self.inference_type = self.InferenceType.TEST
        # _metrics_run processes the testing it self
        return self._metrics_run(number_of_batches, aggregated, write_summary)

    def _metrics_run(self, number_of_batches, aggregated, write_summary):
        """triggers metrics run
            -> summary writing only when aggregated=False"""
        metric_ops_dict = self.metrics  # run all available metrics for test/validation
        if aggregated:
            # metrics run with results aggregated overall batches
            if write_summary:
                print('Warning: writing summary only with aggregated = False possible!')
            return self._metrics_run_overall(number_of_batches)
        # metrics run with results for each batch
        return self._metrics_run_batch(number_of_batches, write_summary)

    def _metrics_run_batch(self, number_of_batches, write_summary):
        """Processes metrics run, with results for every single batch"""
        summary_writer = self.test_writer if self.inference_type == self.InferenceType.TEST else self.validation_writer
        result_list = []
        for i in range(number_of_batches):
            # reset metrics for every run
            self.sess.run(self.metrics.reset)
            # update metrics => run network to predict
            results, summary = self.sess.run([self.metrics.update_ops, self.merged_summaries],
                                             feed_dict={self.training: False})

            result_dict = {}
            for i, key in enumerate(self.metrics.names):
                result_dict[key] = results[i]
            result_list.append(result_dict)

            if write_summary:
                summary_writer.add_summary(summary, global_step=self.counter.eval(session=self.sess))

        summary_writer.flush()
        return result_list

    def _metrics_run_overall(self, number_of_batches):
        """Processes metrics run, with aggregated results overall batches"""
        # reset metrics once for running batches
        self.sess.run(self.metrics.reset)

        for i in range(number_of_batches):
            # run metric updates -> run network to predict
            self.sess.run(self.metrics.update_ops, feed_dict={self.training: False})

        # get results from metrics once
        results = self.sess.run(self.metrics.value_ops)
        result_dict = {}
        for i, key in enumerate(self.metrics.names):
            result_dict[key] = results[i]

        return result_dict


class Metrics:
    """Object that holds references to ops for metrics for access by name"""

    def __init__(self, scope='reset_metrics'):
        self.names = {}
        self.update_ops = []
        self.value_ops = []
        self.reset = None
        self._scope = scope
        self._number_of_metrics = 0

    def append_metric(self, name, metric, **metric_args):
        with tf.variable_scope(self._scope) as scope:
            metric_op, update_op = metric(**metric_args)
            vars = tf.contrib.framework.get_variables(
                scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
            self.reset = tf.variables_initializer(vars)
            tf.summary.scalar(name, update_op)

        self.names[name] = self._number_of_metrics
        self.value_ops.append(metric_op)
        self.update_ops.append(update_op)
        self._number_of_metrics = self._number_of_metrics + 1

    def get_update_op(self, name):
        return self.update_ops[self.names[name]]

    def get_value_op(self, name):
        return self.value_ops[self.names[name]]


def testing_2():
    model = SmallResNet(batch_size=10)
    results_single = model.validate(number_of_batches=2, aggregated=False)
    results_aggregated = model.validate(number_of_batches=2, aggregated=True)
    print(results_single)
    print(results_aggregated)


def testing():
    time_1 = time.time()
    model = SmallResNet(batch_size=10)
    time_2 = time.time()
    train_accuracy = model.train(cycles=5)
    print(train_accuracy)
    print("Overall time:", time.time() - time_1)
    print("Cycle time:", time.time() - time_2)


# todo image test ? -> image summary writer


if __name__ == '__main__':
    testing()
