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
import numpy as np



# import stock data
pkl_file = open('clean_data.pkl','rb')
access, _ = pickle.load(pkl_file)

pkl_file = open('000905.pkl','rb')
comp = pickle.load(pkl_file)


def get_chunk(n,timesteps,num_input,future_time):
    x = np.zeros([n,timesteps,num_input])
    y = np.zeros([n,future_time,num_input])
    i = 0
    while i < n:
        samples = np.random.randint(0,len(list(access.index[0:-timesteps-future_time])),1)
        current_comp_index = len(comp[0][comp[0]<access.index[samples][0]])-1
        stocks = random.sample(comp[1][current_comp_index],num_input)
        x[i] = access.loc[access.index[samples][0]:access.index[samples+timesteps-1][0],stocks]
        y[i] = access.loc[access.index[samples+timesteps][0]:access.index[samples+timesteps+future_time-1][0],stocks]
        if sum(sum(np.isnan(x[i])))>0 or sum(sum(np.isnan(y[i])))>0:
            i = i-1
        i = i+1
    return(x,y)

class my_lstm():
    
    def __init__(self, num_input, timesteps, future_time, num_hidden, batch_size,learning_rate,training_steps,display_step):
        
        self.num_input = num_input
        self.timesteps = timesteps
        self.future_time = future_time
        self.batch_size = batch_size
        self.display_step = display_step

        # Hyper parameters
        self.training_steps = training_steps
        self.num_hidden = num_hidden
        self.learning_rate = learning_rate
        
        # Graph related
        self.graph = tf.Graph()
        self.tf_train_samples = None
        self.tf_train_future_return = None
        self.tf_test_samples = None
        self.tf_test_future_return = None


    def define_graph(self):
        
        with self.graph.as_default():
            self.tf_train_samples = tf.placeholder("float", [self.batch_size, self.timesteps, self.num_input])
            self.tf_train_future_return = tf.placeholder("float", [self.batch_size, self.future_time, self.num_input])
            self.tf_test_samples = tf.placeholder("float", [None, self.timesteps, self.num_input])

            weights = {
                'out': tf.Variable(tf.random_normal([self.num_hidden, self.num_input]))
            }
            biases = {
                'out': tf.Variable(tf.random_normal([self.num_input]))
            }

            def model(x,reuse=None):

                x = tf.unstack(x, self.timesteps, 1)

                # Define a lstm cell with tensorflow
                lstm_cell = rnn.BasicLSTMCell(self.num_hidden, forget_bias=1.0,reuse=reuse)

                # Get lstm cell output
                outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

                # Linear activation, using rnn inner loop last output
                return tf.matmul(outputs[-1], weights['out']) + biases['out']

            def cal_loss(output,input_samples,future):
                self.prediction = tf.nn.relu(output)
                self.prediction = tf.expand_dims(tf.divide(self.prediction,tf.expand_dims(tf.reduce_sum(self.prediction,1),1)),1)
                self.prediction_final = tf.reduce_sum(tf.multiply(future,self.prediction),2)
                # Define loss and optimizer
                #self.loss = tf.nn.l2_loss(tf.subtract(self.prediction_final,self.tf_train_index_return))*2/batch_size
                return tf.nn.l2_loss(self.prediction_final)*2/self.batch_size/self.future_time
            
            self.loss = cal_loss(output = model(self.tf_train_samples),input_samples=self.tf_train_samples,future=self.tf_train_future_return)
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate).minimize(self.loss)
            #self.loss_test = cal_loss(output = model(self.tf_test_samples,reuse=True),input_samples=self.tf_test_samples,future=self.tf_test_future_return)

    def run(self):
        
        self.session = tf.Session(graph=self.graph)
        
        with self.session as sess:
            tf.global_variables_initializer().run()

            # training
            print('Start Training')
            for step in range(1, self.training_steps+1):
                batch_x, batch_y = get_chunk(self.batch_size,self.timesteps,self.num_input,self.future_time)
                # Run optimization op (backprop)
                _, l, w, pf = sess.run([self.optimizer,self.loss,self.prediction,self.prediction_final], 
                                        feed_dict={self.tf_train_samples: batch_x, self.tf_train_future_return: batch_y})       
                if step % self.display_step == 0 or step == 1:
                    print("Step " + str(step) + ", Minibatch Loss= " + format(l))

            print("Optimization Finished!")

            # test
            #test_len = len(test[0])
            #test_x, test_y, _ = test

            #print("Testing tracking error:", \
                #sess.run(self.loss_test, feed_dict={self.tf_test_samples: test_x, self.tf_test_future_return: test_y})*self.batch_size/test_len)


if __name__ == '__main__':
    net = my_lstm(num_hidden = 16, num_input = 100, timesteps = 50, \
    future_time = 10, batch_size = 100, learning_rate = 0.0001, training_steps = 10000, display_step = 100)
    net.define_graph()
    net.run()