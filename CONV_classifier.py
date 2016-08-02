# coding: utf-8
import tensorflow as tf
import numpy as np
import progress as pg
import setdivider as sd
import onehotencoder as ohe
import batch

mnist_set = np.loadtxt('./data/train.csv', delimiter=',', skiprows=1)

x_data = mnist_set.T[1:].T
x_data = np.reshape(x_data, (len(x_data), 28, 28, 1))
y_data = mnist_set.T[0].T
y_data = np.reshape(y_data, (len(y_data), 1))

x_train, y_train, x_test, y_test = sd.devide(x_data, y_data, 0.)
y_train = ohe.encode(y_train, 10)
y_test = ohe.encode(y_test, 10)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
drop_prob = tf.placeholder(tf.float32)
input_drop_prob = tf.placeholder(tf.float32)

#Layer1
filt1 = tf.Variable(tf.random_normal([5,5,1,6], stddev=0.01, dtype=tf.float32))
L1 = tf.nn.conv2d(X, filt1, strides=[1,1,1,1], padding='SAME')
print L1

#Layer2
filt2 = tf.Variable(tf.random_normal([2,2,1,1], stddev=0.01, dtype=tf.float32))
L2 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
#L2 = tf.nn.dropout(L2, drop_prob)
print L2

#Layer3
filt3 = tf.Variable(tf.random_normal([5,5,6,16], stddev=0.01, dtype=tf.float32))
L3 = tf.nn.conv2d(L2, filt3, strides=[1,1,1,1], padding='VALID')
#L3 = tf.nn.dropout(L3, drop_prob)
print L3

#Layer4
filt4 = tf.Variable(tf.random_normal([2,2,1,1], stddev=0.01, dtype=tf.float32))
L4 = tf.nn.max_pool(L3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
#L4 = tf.nn.dropout(L4, drop_prob)
print L4

#Layer5
filt5 = tf.Variable(tf.random_normal([5,5,16,120], stddev=0.01, dtype=tf.float32))
L5 = tf.nn.conv2d(L4, filt5, strides=[1,1,1,1], padding='VALID')
#L5 = tf.nn.dropout(L5, drop_prob)
print L5

L5toM = tf.reshape(L5, [-1,120])
print L5toM

W1 = tf.get_variable('W1', shape=[120,84], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable('W2', shape=[84,84], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
W3 = tf.get_variable('W3', shape=[84,10], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

b1 = tf.Variable(tf.zeros([84], dtype=tf.float32))
b2 = tf.Variable(tf.zeros([84], dtype=tf.float32))
b3 = tf.Variable(tf.zeros([10], dtype=tf.float32))

NNL1 = tf.nn.relu(tf.matmul(L5toM, W1) + b1)
NNL1 = tf.nn.dropout(NNL1, input_drop_prob)

NNL2 = tf.nn.relu(tf.matmul(NNL1, W2) + b2)
NNL2 = tf.nn.dropout(NNL2, drop_prob)
#hypothesis = tf.nn.softmax(tf.matmul(L1, W2) + b2)
hypothesis = tf.matmul(NNL2, W3) + b3
#cost = -tf.reduce_mean(Y*tf.log(hypothesis + 1e-6) + (1-Y)*tf.log(1-hypothesis + 1e-6))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypothesis, Y))

learningRate = tf.constant(0.001, dtype=tf.float32)
optimizer = tf.train.AdamOptimizer(learningRate)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

LEARNING_COUNT = 10000
BATCH_SIZE = 100

with tf.Session() as sess:
	sess.run(init)
	bat = batch.Batch(x_train, y_train)

	for i in range(LEARNING_COUNT):
		batch_xs, batch_ys = bat.next_batch(BATCH_SIZE)
		sess.run(train, feed_dict={X:batch_xs, Y:batch_ys, drop_prob:0.6, input_drop_prob:0.7})
		pg.progress(LEARNING_COUNT, i, sess.run(cost, feed_dict={X:batch_xs, Y:batch_ys, drop_prob:0.6, input_drop_prob:0.7}))

	pg.complete()

	correct = tf.equal(tf.argmax(y_test, 1), tf.argmax(hypothesis, 1))
	accuracy = tf.reduce_mean(tf.cast(correct, "float"))
	
	print "accuracy: ", sess.run(accuracy, feed_dict={X:x_test, drop_prob:1., input_drop_prob:1.})

	test_set = np.loadtxt('./data/test.csv', delimiter=',', skiprows= 1)

	test_set = np.reshape(test_set, (len(test_set), 28, 28, 1))
	result = sess.run(hypothesis, feed_dict={X:test_set, drop_prob:1., input_drop_prob:1.})
	result = sess.run(tf.argmax(result, 1))
	with open("result.csv", "w") as f:
		f.write("ImageId,Label\n")
		for i in range(len(test_set)):
			f.write("%d,%d\n" %(i+1,result[i]))
