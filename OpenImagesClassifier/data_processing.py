""" Input Queue and static image preprocessing for classification algorithm"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from OpenImagesClassifier import config
from PIL import Image

import tensorflow as tf
import pandas as pd
import numpy as np
import sqlite3
import os


def _int64_feature(value):
    """conversion method for .tfrecords file assembly"""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """conversion method for .tfrecords file assembly"""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def preprocess_image(filename, image_id, label, label_display, label_class):
    """Reads image and scales down to 256x256. As mapping function for tf.data.Dataset used."""
    image_encoded = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_encoded)
    image_resized = tf.image.resize_images(image_decoded, [256, 256], align_corners=True)
    image_color = tf.cond(tf.size(image_resized) < 3 * 256 * 256, lambda: tf.image.grayscale_to_rgb(image_resized),
                    lambda: tf.identity(image_resized))
    # for performance during training it would be great to save the image as float32,
    # but this increases the dataset size 4 times!
    image = tf.cast(image_color, dtype=tf.uint8)
    return image, (image_id, label, label_display, label_class)


def load_and_scale_image_ops(filename_placeholder):
    image_encoded = tf.read_file(filename_placeholder)
    image = tf.image.decode_jpeg(image_encoded)

    image = tf.image.convert_image_dtype(image, tf.float32)
    image_resized = tf.image.resize_images(image, [224, 224], align_corners=True)
    image_color = tf.cond(tf.size(image_resized) < 3 * 224 * 224, lambda: tf.image.grayscale_to_rgb(image_resized),
                    lambda: tf.identity(image_resized))
    return tf.reshape(image_color, shape=[1, 224, 224, 3])

def preprocess(subset):
    """Reads all jpeg images, scales them down to 256 x 256 and saves them to one .tfrecords file.
        Why?
        - omit jpeg decompression during training
        - read one big binary file is much faster than reading many individual image files,
            even if it not fits in memory (syscalls, memory mapped files) + tf.data API supports it
        -> saves the file as config.DATA_DIRECTORY/ImagesRaw/{Subset}.tfrecords
    """

    with sqlite3.connect((config.DATABASE['filename'])) as conn:
        c = conn.cursor()
        result = c.execute("""SELECT I.ImageID, I.PathJPEG, L.LabelName, D.DisplayLabelName, D.ClassNumber 
                              FROM Images I
                              INNER JOIN Labels L ON I.ImageID = L.ImageID
                              INNER JOIN Dict D ON L.LabelName = D.LabelName 
                              WHERE Subset = ?
                              ORDER BY random()
                              """, (subset,))

        # list of paths for all images in train dataset
        df = pd.DataFrame(result.fetchall(), columns=['ImageID', 'Path', 'Label', 'Display_Label', 'LabelClass'])
        dataset = tf.data.Dataset.from_tensor_slices((df['Path'].values, df['ImageID'].values, df['Label'].values,
                                                      df['Display_Label'].values, df['LabelClass'].values))

        dataset = dataset.map(preprocess_image)
        next_image, next_image_id = dataset.make_one_shot_iterator().get_next()

        directory = config.DATA_DIRECTORY + "ImagesRaw"
        if not os.path.exists(directory):
            os.mkdir(directory)

        tfrecords_filename = directory + '/' + subset + '.tfrecords'
        writer = tf.python_io.TFRecordWriter(tfrecords_filename)

        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            dataset_size = len(df)
            processed = 0
            # iterate over whole dataset to get every image
            while True:
                try:
                    image, meta = sess.run([next_image, next_image_id])  # get next element (applies map function)

                    # build record from result
                    feature = {'ImageID': _bytes_feature(meta[0]),
                               'Label': _bytes_feature(meta[1]),
                               'Display_Label': _bytes_feature(meta[2]),
                               'Label_Class': _int64_feature(meta[3]),
                               'data': _bytes_feature(tf.compat.as_bytes(image.tostring()))
                               }
                    example = tf.train.Example(features=tf.train.Features(feature=feature))

                    # write record to .tfrecords file
                    writer.write(example.SerializeToString())

                    processed = processed + 1
                    if processed % 500 == 0:
                        print('{} data processed: {}/{}'.format(subset, processed, dataset_size))

                except tf.errors.OutOfRangeError:
                    break

        writer.close()


def preprocess_all_sets():
    """Preporcesses all datasets (-> .tfrecords)"""
    preprocess('train')
    preprocess('validation')
    preprocess('test')


def test_preprocessing(subset):
    """Only for testing functionality, used during development"""
    tfrecords_filename = build_tfrecords_path(subset)

    dataset = tf.data.TFRecordDataset([tfrecords_filename])
    dataset = dataset.map(parse_record)
    next_element = dataset.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
        for _ in range(512):
            try:
                image, meta = sess.run(next_element)
                directory = config.DATA_DIRECTORY + 'ImagesRaw'
                pil_image = Image.fromarray(image)
                pil_image.save(directory + '/' + meta[0].decode('utf-8') + '.bmp')
                print(meta)
            except tf.errors.OutOfRangeError:
                break


def build_tfrecords_path(subset):
    directory = config.DATA_DIRECTORY + 'ImagesRaw'
    tfrecords_path = directory + '/' + subset + '.tfrecords'
    os.path.abspath(tfrecords_path)
    return tfrecords_path


def parse_record(record):
    """Parses one record (string) from .tfrecords file to image tensor and some metadata.
        Used as map function on datasets.
        Returns:  - image as 256x256x3 Tensor of dtype tf.uint8
                  - metadata tupel: (ImageID - tf.string, Label - tf.string,
                                        Display label - tf.string, Label_Class - tf.int64)"""
    dictionary = {'ImageID': tf.FixedLenFeature(shape=(), dtype=tf.string),
                  'Label': tf.FixedLenFeature(shape=(), dtype=tf.string),
                  'Display_Label': tf.FixedLenFeature(shape=(), dtype=tf.string),
                  'Label_Class': tf.FixedLenFeature(shape=(), dtype=tf.int64),
                  'data': tf.FixedLenFeature(shape=(), dtype=tf.string)}
    parsed = tf.parse_single_example(record, dictionary)
    image = tf.decode_raw(parsed['data'], tf.uint8)
    image = tf.reshape(image, (256, 256, 3))

    return image, (parsed['ImageID'], parsed['Label'], parsed['Display_Label'], parsed['Label_Class'])


def parse_test(image, metadata):
    """Parses image from uint [0,255] to float [0,1]. Used as map function"""
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_images(image, [224, 224], align_corners=True)

    return image, metadata


def parse_train(image, metadata):
    """Map function for data augmentation"""
    image = tf.image.convert_image_dtype(image, tf.float32)
    # flipping random
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    # cropping random
    rand_crop = tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32)
    image = tf.cond(rand_crop > 0.5, lambda: tf.random_crop(image, [224, 224, 3]),
                    lambda: tf.image.resize_images(image, [224, 224], align_corners=True))
    # rotating image
    random_int = tf.random_uniform([], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k=random_int)
    # saturation
    rand_saturation = tf.cast(tf.random_uniform([], minval=0, maxval=2, dtype=tf.int32), tf.bool)
    image = tf.cond(rand_saturation, lambda: tf.image.random_saturation(image, 0.2, 1.8),
                    lambda: tf.identity(image))
    # contrast
    rand_contrast = tf.cast(tf.random_uniform([], minval=0, maxval=2, dtype=tf.int32), tf.bool)
    image = tf.cond(rand_contrast, lambda: tf.image.random_contrast(image, 0.25, 0.8),
                    lambda: tf.identity(image))
    # hue
    rand_hue = tf.cast(tf.random_uniform([], minval=0, maxval=2, dtype=tf.int32), tf.bool)
    image = tf.cond(rand_hue, lambda: tf.image.random_hue(image, 0.1), lambda: tf.identity(image))
    return image, metadata


def create_dataset_for_file_list(file_list):
    """Creates dataset for supplied file list"""
    file_list = np.array(file_list)
    dataset = tf.data.Dataset.from_tensor_slices(file_list)
    dataset = dataset.map(preprocess_image)
    return dataset


def build_dataset(subset, batch_size):
    """Creates dataset for supplied subset, enables shuffling and set batch size as supplied"""
    dataset = tf.data.TFRecordDataset([build_tfrecords_path(subset)]) \
        .map(parse_record)

    if subset == 'train':
        dataset = dataset.map(parse_train)
    else:
        dataset = dataset.map(parse_test)

    dataset = dataset.shuffle(buffer_size=5000) \
        .batch(batch_size) \
        .repeat()
    return dataset


def make_saveable_iterator(dataset, session):
    iterator = dataset.make_one_shot_iterator()
    # saveable = tf.contrib.data.make_saveable_from_iterator(iterator)
    # tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, saveable)

    handle = session.run(iterator.string_handle())
    return handle


def build_datasets_and_iterators(batch_size, session):
    """Creates reinitializable iterator, with initialization operation (init op) for every dataset.
        This allows to change dataset without reloading the model.
        Args:
            - batch_size: number of elements per iterator call
        Returns:
            - next_element: nested structure for dataset elements (image, metadata) -> ops used for model input
            - init_ops: dictionary with keys 'train', 'validation', 'test' contains the init ops for each dataset"""

    train_dataset = build_dataset('train', batch_size)
    validation_dataset = build_dataset('validation', batch_size)
    test_dataset = build_dataset('test', batch_size)

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
    next_element = iterator.get_next()

    handle_strings = {'train': make_saveable_iterator(train_dataset, session),
                      'validation': make_saveable_iterator(validation_dataset, session),
                      'test': make_saveable_iterator(test_dataset, session)}
    return next_element, handle, handle_strings


if __name__ == '__main__':
    preprocess_all_sets()
