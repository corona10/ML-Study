import tensorflow as tf
import numpy as np
import config

FLAGS = tf.app.flags.FLAGS
DATA_DIR = '/Users/corona10/Downloads/nv_dlcontest_dataset/'
TRAIN_DIR = DATA_DIR + "train/"
TEST_DIR = DATA_DIR + "test/"
META_TRAIN_FILES = DATA_DIR + "/meta/train.txt"
META_TEST_FILES = DATA_DIR + "/meta/test.txt"

FLAGS = tf.app.flags.FLAGS
filenames = []

def read_raw_images(data_set):
    filename = ['./data/' + data_set + '_data.bin']
    filename_queue = tf.train.string_input_producer(filename)

    if data_set is 'train':
        image_bytes = FLAGS.height * FLAGS.width * FLAGS.depth
        record_bytes = image_bytes + 1
        reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
        key, value = reader.read(filename_queue)
        record_bytes = tf.decode_raw(value, tf.uint8)
        label = tf.cast(tf.slice(record_bytes, [0], [1]), tf.int32)
        depth_major = tf.reshape(tf.slice(record_bytes, [1], [image_bytes]),[FLAGS.depth, FLAGS.height, FLAGS.width])
        uint8image = tf.transpose(depth_major, [1, 2, 0])
        return label, uint8image
    elif data_set is 'test':
        image_bytes = FLAGS.height * FLAGS.width * FLAGS.depth
        record_bytes = image_bytes + 1
        reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
        key, value = reader.read(filename_queue)
        record_bytes = tf.decode_raw(value, tf.uint8)
        depth_major = tf.reshape(tf.slice(record_bytes, [0], [image_bytes]),
        [FLAGS.depth, FLAGS.height, FLAGS.width])
        uint8image = tf.transpose(depth_major, [1, 2, 0])
        return uint8image

def generate_image_and_label_batch(label, image, min_queue_examples, batch_size):
    num_preprocess_threads = 4
    if label is not None:
        label_batch, images = tf.train.batch(
            [label, image],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples * batch_size)
        return tf.reshape(label_batch, [batch_size]), images
    else:
        images = tf.train.batch(
            [image],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples * batch_size)
        return images

def distorted_inputs(batch_size):
    label, image = read_raw_images('train')
    reshaped_image = tf.cast(image, tf.float32)

    #distorted_image = tf.random_crop(reshaped_image, [FLAGS.height, FLAGS.width, FLAGS.depth])
    distorted_image = tf.image.random_flip_left_right(reshaped_image)
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
    float_image = tf.image.per_image_whitening(distorted_image)

    return generate_image_and_label_batch(label, float_image, 100, batch_size)

def inputs(batch_size):
    image = read_raw_images('test')
    print image
    reshaped_image = tf.cast(image, tf.float32)

    #resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, FLAGS.height, FLAGS.width)
    float_image = tf.image.per_image_whitening(reshaped_image)

    return generate_image_and_label_batch(None,float_image, 10, batch_size)

def get_data(data_set, batch_size):
    if data_set is 'train':
        return distorted_inputs(batch_size)
    else:
        return inputs(batch_size)

def main(argv = None):
    label, input = get_data('train', 10)
    input_test = get_data('test', 10)
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)
    for i in range(0, 2):
        print i 
        print sess.run(label), sess.run(input)
        #print sess.run(hypo, feed_dict = { input: sess.run(result.eval)})

if __name__ == '__main__':
    tf.app.run()
