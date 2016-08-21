import tensorflow as tf
import nvidia_input as ni
import config

def main(argv = None):
    test_input = ni.get_data('test', 2)
    label, train_input = ni.get_data('train', 2)
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        tf.train.start_queue_runners(sess=sess)
        for epoch in range(3):
            print epoch
            print sess.run(label)
            print sess.run(train_input)
            print sess.run(test_input)
        #print sess.run(X, feed_dict={x_data: label})
        #print tf.Print(images, [images], message='test')
        #print sess.run(result[0])

if __name__ == '__main__':
    tf.app.run()
