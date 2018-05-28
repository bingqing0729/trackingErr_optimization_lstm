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
import matplotlib.pyplot as plt
import math


# import stock data
pkl_file = open('clean_data.pkl','rb')
excess, _ = pickle.load(pkl_file)

pkl_file = open('000905_weight.pkl','rb')
weight0 = pickle.load(pkl_file)

def get_chunk(timesteps,num_input,future_time,n=1):
    x = np.zeros([n,timesteps,num_input])
    y = np.zeros([n,future_time,num_input])
    i = 0
    while i < n:
        # pick a day
        samples = np.random.randint(0,len(list(excess.index[0:-timesteps-future_time])),1)
        # current time point of index
        current_comp_index = len(weight0.index[weight0.index<excess.index[samples][0]])-1
        # components and weights
        comp = weight0.iloc[current_comp_index,:][weight0.iloc[current_comp_index,:]>0]
        # pick num_input stocks(loc)
        stocks = random.sample(range(0,499),num_input)
        # stocks and weights
        stocks = comp.index[stocks]
        weights = list(comp[stocks])
        # history excess return
        x[i] = excess.loc[excess.index[samples][0]:excess.index[samples+timesteps-1][0],stocks]
        # future excess return
        y[i] = excess.loc[excess.index[samples+timesteps][0]:excess.index[samples+timesteps+future_time-1][0],stocks]
        
        if sum(sum(np.isnan(x[i])))>0 or sum(sum(np.isnan(y[i])))>0:
            i = i-1
        i = i+1
    return(x,y,weights)

class one_sample_lstm():
    
    def __init__(self, num_input, timesteps, future_time, num_hidden, learning_rate,training_steps,display_step):
        
        self.num_input = num_input
        self.timesteps = timesteps
        self.future_time = future_time
        self.display_step = display_step

        # Hyper parameters
        self.training_steps = training_steps
        self.num_hidden = num_hidden
        self.learning_rate = learning_rate
        
        # Graph related
        self.graph = tf.Graph()
        self.tf_train_samples = None
        self.tf_train_future_return = None



    def define_graph(self):
        
        with self.graph.as_default():
            self.tf_train_samples = tf.placeholder("float", [None, self.timesteps, self.num_input])
            self.tf_train_future_return = tf.placeholder("float", [None, self.future_time, self.num_input])
            self.weight0 = tf.placeholder("float",[None,self.num_input])

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
                outputs, _ = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

                # Linear activation, using rnn inner loop last output
                return tf.matmul(outputs[-1], weights['out']) + biases['out'] + self.weight0

            def cal_loss(output,input_samples):
                self.prediction = tf.nn.relu(output)
                self.prediction = tf.expand_dims(tf.divide(self.prediction,tf.expand_dims(tf.reduce_sum(self.prediction,1),1)),1)
                #self.prediction_final = tf.reduce_sum(tf.multiply(future,self.prediction),2)
                self.prediction_final = tf.reduce_sum(tf.multiply(input_samples,self.prediction),2)
                #return tf.nn.l2_loss(self.prediction_final)*2/self.batch_size/self.future_time
                return tf.nn.l2_loss(self.prediction_final)*2/self.timesteps
            
            output =  model(self.tf_train_samples)
            self.loss = cal_loss(output = output,input_samples=self.tf_train_samples)
            self.future_loss =  cal_loss(output = output, input_samples=self.tf_train_future_return)*self.timesteps/self.future_time
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate).minimize(self.loss)
            

    def run(self):
        
        self.session = tf.Session(graph=self.graph)
        
        with self.session as sess:
            tf.global_variables_initializer().run()

            # training
            print('Start Training')
            training_x, training_y, weight0 = get_chunk(self.timesteps,self.num_input,self.future_time)
            l = np.zeros(self.training_steps)
            fl = np.zeros(self.training_steps)
            w = np.zeros([self.training_steps,self.num_input])
            for step in range(0, self.training_steps):
                # Run optimization op (backprop)
                _, l[step], fl[step], w[step] = sess.run([self.optimizer,self.loss,self.future_loss,self.prediction], 
                                        feed_dict={self.tf_train_samples: training_x, self.tf_train_future_return: training_y, self.weight0: weight0})       
                if step % self.display_step == 0:
                    print("Step " + str(step) + ", Loss= " + format(l[step]) + ", Future Loss= " + format(fl[step]))

            print("Optimization Finished!")

            self.l = l[-1]
            self.fl = fl[-1]
            self.w = w[-1]
            self.fl_1000 = fl[1000]
            plt.figure()
            plt.plot(l, 'r', label='training loss')  
            plt.plot(fl, label='future loss')
            plt.savefig("loss.jpg") 

if __name__ == '__main__':
    num_exp = 100
    num_input = 500
    l = np.zeros(num_exp)
    fl = np.zeros(num_exp)
    fl_1000 = np.zeros(num_exp)
    w = np.zeros([num_exp,num_input])
    net = one_sample_lstm(num_hidden = 8, num_input = num_input, timesteps = 100, \
    future_time = 10, learning_rate = 100, training_steps = 5000, display_step = 100)
    net.define_graph()
    for i in range(0,num_exp):
        print(i)
        net.run()
        l[i] = math.sqrt(net.l*250)
        fl[i] = math.sqrt(net.fl*250)
        fl_1000[i] = math.sqrt(net.fl_1000*250)
        w[i] = net.w
        print(l[i])
    
    print(l)
    print(fl)
    print(fl_1000)
    

    plt.figure()
    plt.hist(l)
    plt.savefig("hist-l.jpg")

    plt.figure()
    plt.hist(fl_1000)
    plt.savefig("hist-fl_1000.jpg")

    plt.figure()
    plt.hist(fl)
    plt.savefig("hist-fl.jpg")

    plt.figure()
    plt.hist(w[1])
    plt.savefig("weight-dist.jpg")

    