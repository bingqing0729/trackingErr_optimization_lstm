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
import sklearn.preprocessing
import pandas as pd

# import stock data
pkl_file = open('clean_data.pkl','rb')
excess, _ = pickle.load(pkl_file)

pkl_file = open('000905_weight.pkl','rb')
weight0 = pickle.load(pkl_file)

pkl_file = open('return_1m.pkl','rb')
factor_mom = pickle.load(pkl_file)
factor = np.array(factor_mom.fillna(0))
factor[factor<-10]=0
factor = sklearn.preprocessing.scale(factor,axis=1)
factor = pd.DataFrame(-factor,index=factor_mom.index,columns=factor_mom.columns)

def get_chunk(timesteps,num_input,future_time,n=1,factor=factor):
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
        stocks = random.sample(range(0,500),num_input)
        # stocks and weights
        stocks = comp.index[stocks]
        weights = list(comp[stocks])
        # history excess return
        x[i] = excess.loc[excess.index[samples][0]:excess.index[samples+timesteps-1][0],stocks]
        # future excess return
        y[i] = excess.loc[excess.index[samples+timesteps][0]:excess.index[samples+timesteps+future_time-1][0],stocks]
        # history factor
        factor_h = factor.loc[excess.index[samples+timesteps-1][0],stocks]
        # future factor
        factor_f = factor.loc[excess.index[samples+timesteps+future_time-1][0],stocks]

        if sum(sum(np.isnan(x[i])))>0 or sum(sum(np.isnan(y[i])))>0:
            i = i-1
        i = i+1
    return(x,y,weights,factor_h,factor_f)

class one_sample_lstm():
    
    def __init__(self, num_input, te_limit, timesteps, future_time, num_hidden, learning_rate,training_steps,display_step):
        
        self.num_input = num_input
        self.te_limit = te_limit
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
            self.weight0 = tf.placeholder("float",[self.num_input])
            self.factor = tf.placeholder('float',self.num_input)
            self.factor_f = tf.placeholder('float',self.num_input)
            self.limit = tf.placeholder('float',None)

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
                return tf.matmul(outputs[-1], weights['out']) + biases['out'] + tf.expand_dims(self.weight0,0)

            def cal_loss(output,input_samples,factor):
                self.prediction = tf.nn.relu(output)
                self.prediction = tf.expand_dims(tf.divide(self.prediction,tf.expand_dims(tf.reduce_sum(self.prediction,1),1)),1)
                #self.prediction_final = tf.reduce_sum(tf.multiply(future,self.prediction),2)
                self.prediction_final = tf.reduce_sum(tf.multiply(input_samples,self.prediction),2)
                factor = tf.expand_dims(factor,0)
                self.factor_total = tf.reduce_sum(tf.reduce_sum(tf.multiply(tf.expand_dims(factor,0),self.prediction),2),1)
                #return tf.nn.l2_loss(self.prediction_final)*2/self.batch_size/self.future_time
                te = tf.sqrt(tf.nn.l2_loss(self.prediction_final)*2/self.timesteps*250)
                loss = 100000*tf.nn.relu(te-self.limit)-self.factor_total
                ex_return = tf.reduce_sum(self.prediction_final)/self.timesteps*250
                return [loss, ex_return, te, self.factor_total]
            
            output =  model(self.tf_train_samples)
            self.loss,self.ex_return, self.te, self.factor_exposure = \
            cal_loss(output=output,input_samples=self.tf_train_samples,factor=self.factor)
            self.future_loss, self.f_ex_return, self.f_te, self.f_factor_exposure =  \
            [i*self.timesteps/self.future_time for i in cal_loss(output=output, input_samples=self.tf_train_future_return,\
            factor=self.factor_f)]
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate).minimize(self.loss)
            

    def run(self):
        
        self.session = tf.Session(graph=self.graph)
        
        with self.session as sess:
            

            # training
            training_x, training_y, weight0, factor_h, factor_f = get_chunk(self.timesteps,self.num_input,self.future_time)
            self.fe_end = np.zeros(len(self.te_limit))
            i = 0
            for limit in list(self.te_limit):
                tf.global_variables_initializer().run()
                print('Start Training:')
                l = np.zeros(self.training_steps)
                ex_return = np.zeros(self.training_steps)
                te = np.zeros(self.training_steps)
                fe = np.zeros(self.training_steps)
                fl = np.zeros(self.training_steps)
                f_ex_return = np.zeros(self.training_steps)
                f_te = np.zeros(self.training_steps)
                f_fe = np.zeros(self.training_steps)
                w = np.zeros([self.training_steps,self.num_input])
                for step in range(0, self.training_steps):
                    # Run optimization op (backprop)
                    _, l[step], ex_return[step], te[step], fe[step], fl[step], f_ex_return[step], f_te[step], f_fe[step], w[step] = \
                    sess.run([self.optimizer,self.loss,self.ex_return,self.te,self.factor_exposure,self.future_loss,self.f_ex_return,\
                    self.f_te,self.f_factor_exposure, self.prediction], 
                                            feed_dict={self.tf_train_samples: training_x, \
                                            self.tf_train_future_return: training_y, self.weight0: weight0, \
                                            self.factor: factor_h, self.factor_f: factor_f, self.limit: limit})       
                    if step % self.display_step == 0:
                        print("Step " + str(step) + ", Loss= " + format(l[step]) + ", excess return= "+ format(ex_return[step]) + \
                        ", te= " + format(te[step])+", factor exposure "+format(fe[step]))

                print("Optimization Finished!")
                
                self.fe_end[i] = fe[-1]
                i = i+1

if __name__ == '__main__':
    num_exp = 1
    num_input = 100
    te_limit = [0.02,0.03,0.04,0.05,0.06]
    net = one_sample_lstm(num_hidden = 5, num_input = num_input, timesteps = 100, \
    future_time = 10, learning_rate = 0.1, training_steps = 1000, display_step = 50, te_limit = te_limit)
    net.define_graph()
    for i in range(0,num_exp):
        print(i)
        net.run()
        fe_end = net.fe_end
        plt.figure()
        plt.plot(te_limit,fe_end)
        output = "te_limit_fe"+str(i)+".jpg"
        plt.savefig(output)





    