import tensorflow as tf
import random
import numpy as np
import sys
# import matplotlib.pyplot as plt

data_file = 'train_data'
labels_file = 'train_labels'

train_data = []
train_labels = []
with open(data_file, 'r') as f:
    counter = 0
    for temp in f:
        counter = counter + 1
        train_data.append([float(x) for x in temp.split(',')[:(len(temp.split(',')) - 1)]])

# with open(labels_file, 'r') as f:
#     for temp in f:
#         train_labels.append(int(temp))

print("Finished Reading Data")

classwise_x = [[] for i in range(12)]
for index in range(30000):
    classwise_x[train_labels[index]].append(train_data[index])

train_data = []
train_labels = []
test_data = []
test_labels = []
for i in range(2400):
	for j in range(12):
		train_data.append(classwise_x[j][i])
		train_labels.append(j)
train_data = np.array(train_data)
train_labels = np.array(train_labels)
one_hot_train = np.zeros((2400*12, 12))
one_hot_train[np.arange(2400*12), train_labels] = 1

for i in range(2400, 2500):
	for j in range(12):
		test_data.append(classwise_x[j][i])
		test_labels.append(j)

test_data = np.array(test_data)
test_labels = np.array(test_labels)
one_hot_test = np.zeros((100*12, 12))
one_hot_test[np.arange(100*12), test_labels] = 1

x = tf.placeholder(tf.float32, [None, 3072])
y_ = tf.placeholder(tf.float32, [None, 12])
alpha = tf.Variable(tf.zeros([3072, 1000]))
alpha_0 = tf.Variable(tf.zeros([1000]))
beta = tf.Variable(tf.zeros([1000,12]))
beta_0 = tf.Variable(tf.zeros([12]))
z = tf.sigmoid(tf.matmul(x, alpha) + alpha_0)
y = tf.nn.softmax(tf.matmul(z, beta) + beta_0)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

print("Variables are set")

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
print("Session starts running")
accuracies = []
for i in range(10000):
	batch_x = train_data[(i%24):((i%24) + 100)]
	batch_y = one_hot_train[(i%24):((i%24) + 100)]
	sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print("Iteration %d     Accuracy %.2f" % (i, sess.run(accuracy, feed_dict={x:test_data, y_:one_hot_test})), end = '\r')

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x:test_data, y_:one_hot_test}))

for x in accuracies:
	print(x)
