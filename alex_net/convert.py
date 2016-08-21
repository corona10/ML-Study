import config
import numpy as np
import tensorflow as tf
import sys
import struct
import csv 

FLAGS = tf.app.flags.FLAGS
BAR_LENGTH = 10

def progress(total, step, datasets):
    completeBarLength = int(round(BAR_LENGTH*step / float(total)))
    percent = round(100 * step / float(total), 1)
    bar = completeBarLength * '=' + (30-completeBarLength) * '-'
    sys.stdout.write('[%s] %s%s cnt: %s  %s \r' %(bar, percent, '%', step, datasets))
    sys.stdout.flush()

def listfiles(text_file):
    file_lists = []
    with open(text_file, 'rb') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=' ', quotechar='\n')
        for row in csvreader:
            file_lists.append(row[0])
    return file_lists

def read_jpeg(filename):
    value = tf.read_file(filename)
    #print filename
    decoded_image = tf.image.decode_jpeg(value, channels = FLAGS.depth)
    resized_image = tf.image.resize_images(decoded_image, FLAGS.height, FLAGS.width)
    resized_image = tf.cast(resized_image, tf.uint8)
    return resized_image

def jpg2binary(sess, datasets, label):
    if datasets is 'train':
        files = listfiles(FLAGS.training_meta)
    elif datasets is 'test':
        files = listfiles(FLAGS.test_meta)
    else:
        raise ValueError("not proper dataset")
    
    with open('./data/' + datasets + '_data.bin', 'wb') as f:
        cnt = 0
        for image in files[:100]:
            if datasets is 'train':
                jpg_dir = FLAGS.training_image_dirs + image+".jpg"
                label_name = image.split('/')[0]
                label_encoding = label[label_name]
            elif datasets is 'test':
                jpg_dir = FLAGS.test_image_dirs + image+".jpg"

            resized_image = read_jpeg(jpg_dir)
            try:
                image = sess.run(resized_image)
                #print image
                progress(len(files), cnt, datasets)
                #print cnt
                cnt = cnt+1
            except Exception as e:
                print e.message
                continue
           
            if datasets is 'train':            
                f.write(chr(label_encoding))
            f.write(image.data)

def get_labels(label_dirs):
    label_dict = dict()
    encoding = 0
    with open(label_dirs, 'r') as f:
        for line in f:
            line = line.rstrip()
            print encoding , line
            label_dict[line] = encoding
            encoding = encoding + 1
    return label_dict

def read_raw_images(sess, data_set):
    filename = ['./data/' + data_set + '_data.bin']
    filename_queue = tf.train.string_input_producer(filename)
    print filename
    record_bytes = (FLAGS.height) * (FLAGS.width) * FLAGS.depth + 1
    image_bytes = (FLAGS.height) * (FLAGS.width) * FLAGS.depth
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    key, value = reader.read(filename_queue)
    record_bytes = tf.decode_raw(value, tf.uint8)
    #record_label = tf.decode_raw(value, tf.int32)
    tf.train.start_queue_runners(sess=sess)
    for i in range(0, 10):
        result = sess.run(record_bytes)
        print i, result[0], len(result)
        image = result[1:len(result)]
        print image
        
def main(argv= None):
    print "make label encoding.."
    label_encoding = get_labels(FLAGS.label)
    with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=4,intra_op_parallelism_threads=4)) as sess:
        jpg2binary(sess, 'train', label_encoding)
        jpg2binary(sess, 'test', label_encoding)
        #read_raw_images(sess, 'train')
if __name__ == "__main__":
    with tf.device('/cpu:0'):
        tf.app.run()
