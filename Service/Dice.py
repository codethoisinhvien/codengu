import requests
from datetime import datetime
from Model.config import  DiceData
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from pymongo import ASCENDING, DESCENDING
class DiceCoin:

    def __init__(self,api):
        self.i=0
        self.max=0
        self.totalH=0.00015000
        self.totalM=0.00015000
        self.min=0.00015000
        self.api= api
        self.token=2
        self.coin=2
        self.amount=0.00015000
        self.condition=4
        self.guess=True
        self.sum=0
    def coverObject(self):
        if self.guess==True:
               return  {'access_token':self.token,'currency':self.coin,'target':self.condition,'amount':self.amount,'over':self.guess}
        else:
             return  {'access_token':self.token,'currency':self.coin,'target':self.condition,'amount':self.amount,'over':self.guess}


    def crawlData(self):
        
          
            b= DiceData.query.filter()
            data = [{'number': i.number,}for i in b]
           
            states = pd.DataFrame(data)
      
            temperature= np.array(states['number']);
           
            self.i=self.i
            num_periods =1
            f_horizon = 2
            x_train = temperature[:(len(temperature)-(num_periods))]
            x_batches = x_train.reshape(-1, num_periods, 1)

            y_train = temperature[1:(len(temperature)-(num_periods))+f_horizon]
            y_batches = y_train.reshape(-1, num_periods, 1)
            X_test =temperature[-(num_periods + f_horizon):][:1].reshape(-1, num_periods, 1)
            Y_test =temperature[-(num_periods):].reshape(-1, num_periods, 1)
            tf.reset_default_graph()

            rnn_size = 100
            learning_rate=0.0001

            X = tf.placeholder(tf.float32, [None, num_periods, 1])
            Y = tf.placeholder(tf.float32, [None, num_periods, 1])

            rnn_cells=tf.contrib.rnn.BasicRNNCell(num_units=rnn_size, activation=tf.nn.relu)
            rnn_output, states = tf.nn.dynamic_rnn(rnn_cells, X, dtype=tf.float32)

            output=tf.reshape(rnn_output, [-1, rnn_size])
            logit=tf.layers.dense(output, 1, name="softmax")

            outputs=tf.reshape(logit, [-1, num_periods, 1])
            print(logit)

            loss = tf.reduce_sum(tf.square(outputs - Y))

            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logit, 1), tf.cast(Y, tf.int64)), tf.float32))
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_step=optimizer.minimize(loss)
            epochs = 1000

            sess = tf.Session()
            init = tf.global_variables_initializer()
            sess.run(init)
            if self.i%20==0:
              for epoch in range(epochs):
                 train_dict = {X: x_batches, Y: y_batches}
                 sess.run(train_step, feed_dict=train_dict)
              saver = tf.train.Saver()
              save_path = saver.save(sess, "models/model.ckpt")
            self.i+=1
            saver = tf.train.Saver()
            with tf.Session() as sess:
  # Restore variables from disk.
                 saver.restore(sess, "models/model.ckpt")
                 y_pred=sess.run(outputs, feed_dict={X: X_test})
                 print (y_pred)
            if y_pred[0][0][0]>50:
                self.condition=50.49
                self.guess=True
                self.amount=self.totalH
            else:
                self.condition=49.5
                self.guess=False
                self.amount=self.totalM
            if self.max<self.amount:
                 self.max=self.amount
            print(self.max)
            res=requests.post(self.api,self.coverObject())
            if(res.status_code==200):
            
                val = res.json()
                print(val)
                
                if float(val['profit'])>self.min:
                    if self.guess==True:
                         self.totalH= self.min+self.sum*0.01
 

