
# coding: utf-8

# In[380]:

import tensorflow as tf
import numpy as np


# In[381]:

def accuerncy(hypo, test):
    total = float(len(hypo))
    hit = 0.0
    for i in range(len(hypo)):
        if int(hypo[i]) == int(test[i]):
            hit = hit + 1
    
    return hit / total


# In[382]:

def make_one_hot(y_data):
    y_one_hot = np.zeros([len(y_data), 2])
    for i in range(len(y_data)):
        y_one_hot[i][y_data[i]] = 1
    return y_one_hot


# In[383]:

data = np.genfromtxt('horse/horseColicTraining.txt',delimiter= '\t')
data[np.isnan(data)] = 0.0
x_data = data.T[:-1].T
y_data = data.T[-1]
X = tf.placeholder(tf.float32, [None, 21])
Y = tf.placeholder(tf.float32, [None, 2])
Weight = tf.Variable(tf.random_uniform([21,2], -1.0, 1.0))
bias  = tf.Variable(tf.random_uniform([1,2], -1.0, 1.0))


# In[384]:

#one hot encoding 생성
y_one_hot = make_one_hot(y_data)


# In[385]:

hypo= tf.nn.softmax(tf.matmul(X, Weight) + bias) 
cross_entropy = -tf.reduce_sum(Y*tf.log(hypo + 1e-12))#-tf.reduce_sum(Y*tf.log(hypo))


# In[386]:

optimizer = tf.train.AdagradOptimizer(0.05)
train = optimizer.minimize(cross_entropy)


# In[390]:

sess = tf.Session()
sess.run(tf.initialize_all_variables())
#print sess.run(hypo,  feed_dict={X:x_data, Y:y_one_hot})
rate = 5000
print "cross_entrophy: ",sess.run(cross_entropy, feed_dict={X:x_data, Y:y_one_hot})
for step in xrange(rate):
    sess.run(train, feed_dict={X:x_data, Y:y_one_hot})
    if step % 1000 == 0:
        print "cross_entrophy: ",sess.run(cross_entropy, feed_dict={X:x_data, Y:y_one_hot})
print "cross_entrophy: ",sess.run(cross_entropy, feed_dict={X:x_data, Y:y_one_hot})
print "weight: ", sess.run(Weight)
print "bias: ", sess.run(bias)


# In[370]:

data = np.genfromtxt('horse/horseColicTest.txt',delimiter= '\t')
data[np.isnan(data)] = 0.0
x_test = data.T[:-1].T
y_test = data.T[-1]
test_y_one_hot = make_one_hot(y_test)
all = sess.run(hypo, feed_dict={X:x_data, Y:y_one_hot})
all= sess.run(tf.arg_max(all,1))

print "accurency: ",accuerncy(all, y_data)
print all


# In[371]:

sess.close()


# In[ ]:



