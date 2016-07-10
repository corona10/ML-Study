# coding: utf-8
import tensorflow as tf
import numpy as np

def accuerncy(hypo, test):
    total = float(len(hypo))
    hit = 0.0
    for i in range(len(hypo)):
        if int(hypo[i]) == int(test[i]):
            hit = hit + 1    
    return hit / total

# one-hot encoding 생성
def make_one_hot(y_data):
    y_one_hot = np.zeros([len(y_data), 2])
    for i in range(len(y_data)):
        y_one_hot[i][y_data[i]] = 1
    return y_one_hot

data = np.genfromtxt('horse/horseColicTraining.txt',delimiter= '\t')
data[np.isnan(data)] = 0.0
x_data = data.T[:-1].T
y_data = data.T[-1]
X = tf.placeholder(tf.float32, [None, 21])
Y = tf.placeholder(tf.float32, [None, 2])
Weight = tf.Variable(tf.random_uniform([21,2], -1.0, 1.0))
bias  = tf.Variable(tf.random_uniform([1,2], -1.0, 1.0))

y_one_hot = make_one_hot(y_data)
hypo= tf.sigmoid(tf.nn.softmax(tf.matmul(X, Weight) + bias))
cross_entropy = -tf.reduce_sum(Y*tf.log(hypo))#-tf.reduce_sum(Y*tf.log(hypo))

optimizer = tf.train.AdamOptimizer(0.01)
train = optimizer.minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
rate = 10000
print "cross_entrophy: ",sess.run(cross_entropy, feed_dict={X:x_data, Y:y_one_hot})
for step in xrange(rate):
    sess.run(train, feed_dict={X:x_data, Y:y_one_hot})
    if step % 1000 == 0:
        print "cross_entrophy: ",sess.run(cross_entropy, feed_dict={X:x_data, Y:y_one_hot})
print "cross_entrophy: ",sess.run(cross_entropy, feed_dict={X:x_data, Y:y_one_hot})
print "weight: ", sess.run(Weight)
print "bias: ", sess.run(bias)

data = np.genfromtxt('horse/horseColicTest.txt',delimiter= '\t')
data[np.isnan(data)] = 0.0
x_test = data.T[:-1].T
y_test = data.T[-1]
test_y_one_hot = make_one_hot(y_test)
all = sess.run(hypo, feed_dict={X:x_test, Y:test_y_one_hot})
all= sess.run(tf.arg_max(all,1))

print "accurency: ",accuerncy(all, y_test)
print all

sess.close()


