import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
file_url = '/Users/corona10/Downloads/nv_dlcontest_dataset/train/soondubu_jjigae/355287.jpg'
DATA_DIR = '/Users/corona10/Downloads/nv_dlcontest_dataset'
TRAIN_DIR = DATA_DIR + "/train/"
TEST_DIR = DATA_DIR + "/test/"
META_TRAIN_FILES = DATA_DIR + "/meta/train.txt"
META_TEST_FILES = DATA_DIR + "/meta/test.txt"
META_LABEL_FILES = DATA_DIR + "/meta/labels.txt"

BINARY_FILES_DIR = './data'

tf.app.flags.DEFINE_string('training_meta', META_TRAIN_FILES ,'')
tf.app.flags.DEFINE_string('test_meta', META_TEST_FILES,'')
tf.app.flags.DEFINE_string('training_image_dirs', TRAIN_DIR, '')
tf.app.flags.DEFINE_string('test_image_dirs', TEST_DIR, '')
tf.app.flags.DEFINE_string('data_dir', BINARY_FILES_DIR, '')
tf.app.flags.DEFINE_string('label', META_LABEL_FILES, '')

tf.app.flags.DEFINE_integer('width',  256, '')
tf.app.flags.DEFINE_integer('height', 256, '')
tf.app.flags.DEFINE_integer('depth', 3, '')

