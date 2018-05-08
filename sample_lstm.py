# -*- coding: utf-8 -*-
"""
Created on Mon May  7 16:38:45 2018

@author: hubingqing
"""

import tensorflow as tf
from tensorflow.contrib import rnn
import pickle

# import stock data
pkl_file = open('clean_data.pkl','rb')
stock = pickle.load(pkl_file)

# Training Parameters
learning_rate = 0.001
training_steps = 10000
batch_size = 100
display_step = 200

# Network Parameters
num_input = 100
timesteps = 100 # timesteps
future_time = 5
num_hidden = 128 # hidden layer num of features

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, future_time, num_input])
Z = tf.placeholder("float", [None,future_time])
W = tf.placeholder("float", [None,num_input])

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
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

out = RNN(X, weights, biases)
prediction = tf.matmul(Y,out)

# Define loss and optimizer
loss_op = tf.reduce_mean((prediction-Z)**2)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, training_steps+1):
        batch_x, batch_y, batch_z = stock_data.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, Z: batch_z})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss = sess.run([loss_op], feed_dict={X: batch_x,
                                                  Y: batch_y,
                                                  Z: batch_z})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss))

    print("Optimization Finished!")

    # Calculate accuracy for test set
    test_len = 128
    test_x, test_y, test_z = stock_data.test

    print("Testing tracking error:", \
        sess.run(loss_op, feed_dict={X: test_x, Y: test_y, Z: test_z}))
    
