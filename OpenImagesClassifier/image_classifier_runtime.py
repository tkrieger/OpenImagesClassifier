"""Runtime for small ResNet Model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from OpenImagesClassifier import data_processing as data
from OpenImagesClassifier import config as config
from OpenImagesClassifier import small_resnet_model as rs
from enum import Enum

import tensorflow as tf
import tensorflow_hub as hub
import time


class ModelType(Enum):
    """Enum for network model types"""
    SMALL_RESNET = 'SMALL_RESNET'
    TRAINED_MODEL = 'TRAINED_MODEL'


class ImageClassifier:
    """Class for train, validate, test and predict with the classifier.
        Multiple instances have to be used in exclusive graphs (as default graph)!"""

    def __init__(self, model_type=ModelType.SMALL_RESNET, batch_size=256, model_path=config.MODEL_SAVE_DIR,
                 summary_dir=config.SUMMARY_DIR):

        self._init_basics(batch_size, model_path, model_type)
        self._init_data_pipeline(batch_size)

        self.training = tf.placeholder(tf.bool, name='training')
        self.logits = None
        self.current_handle_string = None

        if model_type == ModelType.SMALL_RESNET:
            self._init_small_resnet()
        if model_type == ModelType.TRAINED_MODEL:
            self._init_trained_resnet()

        self._init_training()
        self._init_prediction()
        self._init_runtime(summary_dir)

    def _init_small_resnet(self):
        """Used for initialization of small resnet model"""
        self.logits = rs.build_small_resnet(self.X, classes_count=len(config.CATEGORIES), training=self.training)

    def _init_trained_resnet(self):
        """Used for initialization of trained model"""
        module = hub.Module(config.TRAINED_MODEL['url'])  # with trainable=True the model parameter are also trained
        self.features = module(self.X)

        # add a fully connected layer
        self.logits = tf.layers.dense(inputs=self.features, units=len(config.CATEGORIES), name="Logits")

    def _init_prediction(self):
        """Adds all ops needed for prediction + metrics"""
        self.softmax = tf.nn.softmax(logits=self.logits)
        # add metric object manages metrics and holds needed references
        # todo add further metrics
        self.metrics = Metrics()
        self.metrics.append_metric('top_1_accuracy', tf.metrics.accuracy, labels=self.y,
                                   predictions=tf.argmax(self.logits, axis=1))
        self.metrics.append_metric('top_3_accuracy', tf.metrics.mean,
                                   values=tf.nn.in_top_k(self.logits, self.y, k=3))
        self.metrics.append_metric('top_5_accuracy', tf.metrics.mean,
                                   values=tf.nn.in_top_k(self.logits, self.y, k=5))

    def _init_runtime(self, summary_dir):
        """get model ready for runtime"""
        # saver for model
        self.saver = tf.train.Saver()

        tf.global_variables_initializer().run(session=self.sess)
        tf.local_variables_initializer().run(session=self.sess)
        self.restore_model()

        self.merged_summaries = tf.summary.merge_all()
        summary_dir = summary_dir + '/' + self.model_type.name
        self.train_writer = tf.summary.FileWriter(summary_dir + '/train')
        self.validation_writer = tf.summary.FileWriter(summary_dir + '/validation')
        self.test_writer = tf.summary.FileWriter(summary_dir + '/test')
        self.inference_type = self.InferenceType.VALIDATION

        self.train_writer.add_graph(tf.get_default_graph())

    def _init_training(self):
        """Creates all ops needed for training"""
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits)
        loss = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('loss', loss)
        optimizer = tf.train.AdamOptimizer()
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            self.training_op = optimizer.minimize(loss)

    def _init_basics(self, batch_size, model_path, model_type):
        """Initializes all object attributes that represent basic object state"""
        self.sess = tf.Session()

        with tf.variable_scope("counter", reuse=tf.AUTO_REUSE):
            self.counter = tf.get_variable('overall_training_cycles', shape=(), dtype=tf.int32,
                                           trainable=False, initializer=tf.constant_initializer(0))
            self.add_train_cycle = tf.assign_add(self.counter, 1)
        self.batch_size = batch_size
        self.model_path = model_path + '/' + model_type.name
        self.model_type = model_type

    def _init_data_pipeline(self, batch_size):
        """Initializes all object attributes that belongs to data input pipeline"""
        input_ops, self.handle, self.iterator_handle_strings = data.build_datasets_and_iterators(batch_size, self.sess)
        self.X = tf.placeholder_with_default(input_ops[0], shape=[None, 224, 224, 3], name="X_input")
        self.y = tf.placeholder_with_default(input_ops[1][3], shape=[None], name="y_label")
        # tf.summary.image("train", self.X)

        self.prediction_filename_placeholder = tf.placeholder(tf.string, shape=(), name="prediction_filename")
        self.scale_image = data.load_and_scale_image_ops(self.prediction_filename_placeholder)

    class InferenceType(Enum):
        VALIDATION = 1
        TEST = 2

    def __exit__(self, *oth):
        self.sess.close()
        self.train_writer.close()
        self.validation_writer.close()
        self.test_writer.close()

    def __enter__(self):
        return self

    def restore_model(self):
        checkpoint_path = tf.train.latest_checkpoint(self.model_path)
        if checkpoint_path is not None:
            # checkpoint found for model, restore
            self.saver.restore(self.sess, checkpoint_path)
            print("Loaded saved Model")

    def save_model(self):
        self.saver.save(self.sess, self.model_path + '/model.ckpt')

    def train(self, cycles):
        # enable train dataset
        self.current_handle_string = self.iterator_handle_strings['train']
        training_hist = []

        for i in range(cycles):
            # reset metric for this batch
            self.sess.run(self.metrics.reset)
            # run training and update metric - update op also returns metric results
            summary, _, res, overall_cycle = self.sess.run([self.merged_summaries, self.training_op,
                                                            self.metrics.update_ops, self.add_train_cycle],
                                                           feed_dict={self.training: True,
                                                                      self.handle: self.current_handle_string})
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
        self.current_handle_string = self.iterator_handle_strings['validation']
        self.inference_type = self.InferenceType.VALIDATION
        # _metrics_run processes the validation it self
        return self._metrics_run(number_of_batches, aggregated, write_summary)

    def test(self, number_of_batches=1, aggregated=True, write_summary=False):
        """Runs test for number_of_batches batches.
            If aggregated is True the metrics are aggregated overall cycles
            -> For testing and validation the same actions are needed, only difference is the dataset
            -> summary writing only when aggregated=False"""
        self.current_handle_string = self.iterator_handle_strings['test']
        self.inference_type = self.InferenceType.TEST
        # _metrics_run processes the testing it self
        return self._metrics_run(number_of_batches, aggregated, write_summary)

    def predict(self, file_list):
        """Predicts classes for images given as filenames/path
            returns: dict with filename and predictions"""
        result = []
        # hotfix for running prediction as first element
        self.current_handle_string = self.iterator_handle_strings['test']

        for file_name in file_list:
            image = self.sess.run(self.scale_image, feed_dict={self.prediction_filename_placeholder: file_name})
            prediction = self.sess.run([self.softmax], feed_dict={self.X: image,
                                                                  self.training: False,
                                                                  self.handle: self.current_handle_string})
            result.append({'file_name': file_name, 'prediction': prediction})

        return result

    def _metrics_run(self, number_of_batches, aggregated, write_summary):
        """triggers metrics run
            -> summary writing only when aggregated=False"""
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
                                             feed_dict={self.training: False, self.handle: self.current_handle_string})

            result_dict = {}
            for j, key in enumerate(self.metrics.names):
                result_dict[key] = results[j]
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
            self.sess.run(self.metrics.update_ops, feed_dict={self.training: False,
                                                              self.handle: self.current_handle_string})

        # get results from metrics once
        results = self.sess.run(self.metrics.value_ops)
        result_dict = {}
        for i, key in enumerate(self.metrics.names):
            result_dict[key] = results[i]

        return [result_dict]


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
    model = ImageClassifier(batch_size=10)
    results_single = model.validate(number_of_batches=2, aggregated=False)
    results_aggregated = model.validate(number_of_batches=2, aggregated=True)
    print(results_single)
    print(results_aggregated)
    model.train(cycles=5)


def test_both():
    g1 = tf.Graph()
    with g1.as_default():
        with ImageClassifier(batch_size=10) as model_small:
            test_model(model_small)
    g2 = tf.Graph()
    with g2.as_default():
        with ImageClassifier(model_type=ModelType.TRAINED_MODEL, batch_size=20) as model_trained:
            test_model(model_trained)


def test_model(model):
    time_train = time.time()
    train_acc = model.train(cycles=5)
    time_valid = time.time()
    val_acc = model.validate(number_of_batches=1)
    time_test = time.time()
    test_acc = model.test(number_of_batches=1)
    time_pred = time.time()
    files = ['C:/Users/D065030/Documents/dev/Studienarbeit/OpenImagesClassifier/OpenImagesClassifier'
             '/data/Images/Person/train/000bc1eb7f74adae.jpg',
             'C:/Users/D065030/Documents/dev/Studienarbeit/OpenImagesClassifier/OpenImagesClassifier'
             '/data/Images/Car/train/000efa99e67d6f0c.jpg']
    predictions = model.predict(files)
    time_end = time.time()
    print('Train:', train_acc)
    print('Valid:', val_acc)
    print('Test:', test_acc)
    print('Pred:', predictions)
    print("-Times-------------")
    print('Train:', time_valid - time_train)
    print('Valid:', time_test - time_valid)
    print('Test:', time_pred - time_test)
    print('Pred:', time_end - time_pred)
    print('Overall', time_end - time_train)
    print("-------------------------------------")


if __name__ == '__main__':
    test_both()
