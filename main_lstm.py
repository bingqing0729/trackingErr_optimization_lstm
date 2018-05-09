# -*- coding: utf-8 -*-
"""
Created on Mon May  7 16:38:45 2018

@author: hubingqing
"""
from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
import pickle
import random

# import stock data
pkl_file = open('clean_data.pkl','rb')
stock = pickle.load(pkl_file)
stock_train = [i[0:800] for i in stock]
stock_test = [i[800:1000] for i in stock]
del(stock)

# Training Parameters
learning_rate = 0.001
training_steps = 10000
batch_size = 50
display_step = 200

# Network Parameters
num_input = 100
timesteps = 100 # timesteps
future_time = 5
num_hidden = 64 # hidden layer num of features

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
future_return = tf.placeholder("float", [None, future_time, num_input])
index_return = tf.placeholder("float", [None, future_time])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_input]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_input]))
}


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = tf.nn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return [tf.matmul(outputs[-1], weights['out']) + biases['out'],outputs]

output, lstm_out = RNN(X, weights, biases)
prediction = tf.nn.relu(output)
prediction = tf.expand_dims(tf.divide(prediction,tf.expand_dims(tf.reduce_sum(prediction,1),1)),1)
prediction_final = tf.reduce_sum(tf.multiply(future_return,prediction),2)
# Define loss and optimizer
loss_op = tf.nn.l2_loss(tf.subtract(prediction_final,index_return))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, training_steps+1):
        batch_x, batch_y, batch_z = [i[random.sample(range(0,len(stock_train[2])),batch_size)] for i in stock_train]
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, future_return: batch_y, index_return: batch_z})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, pf, op, out= sess.run([loss_op, prediction_final,output,lstm_out], feed_dict={X: batch_x,
                                                  future_return: batch_y,
                                                  index_return: batch_z})
            print("Step " + str(step) + ", Minibatch Loss= " + format(loss))

    print("Optimization Finished!")

    # Calculate accuracy for test set
    test_len = 200
    test_x, test_y, test_z = stock_test

    print("Testing tracking error:", \
        sess.run(loss_op, feed_dict={X: test_x, future_return: test_y, index_return: test_z}))
