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
    image = tf.image.decode_jpeg(image_encoded)

    image_resized = tf.image.resize_images(image, [256, 256], align_corners=True)
    # for performance during training it would be great to save the image as float32,
    # but this increases the dataset size 4 times!
    image = tf.cast(image_resized, dtype=tf.uint8)
    return image, (image_id, label, label_display, label_class)


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
                              WHERE Subset = ?""", (subset,))

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

def parse_to_float(image, metadata):
    """Parses image from uint [0,255] to float [0,1]. Used as map function"""
    return tf.image.convert_image_dtype(image, tf.float32), metadata


def data_augmentation(image, metadata):
    """Map function for data augmentation"""
    # TODO implement augmentation
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
                    .map(parse_record) \
                    .map(parse_to_float) \
                    .shuffle(buffer_size=2000) \
                    .batch(batch_size) \
                    .repeat()
    return dataset


def create_reinitializable_iterator(batch_size):
    """Creates reinitializable iterator, with initialization operation (init op) for every dataset.
        This allows to change dataset without reloading the model.
        Args:
            - batch_size: number of elements per iterator call
        Returns:
            - next_element: nested structure for dataset elements (image, metadata) -> ops used for model input
            - init_ops: dictionary with keys 'train', 'validation', 'test' contains the init ops for each dataset"""

    train_dataset = build_dataset('train', batch_size)
    train_dataset = train_dataset.map(data_augmentation)
    validation_dataset = build_dataset('validation', batch_size)
    test_dataset = build_dataset('test', batch_size)

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

    next_element = iterator.get_next()
    train_init_op = iterator.make_initializer(train_dataset)
    validation_init_op = iterator.make_initializer(validation_dataset)
    test_init_op = iterator.make_initializer(test_dataset)

    init_ops = {'train': train_init_op,
                'validation': validation_init_op,
                'test': test_init_op}
    return next_element, init_ops


if __name__ == '__main__':
    preprocess_all_sets()
