import requests

import json

from flask import Flask

from flask_mongoalchemy import BaseQuery

from flask_mongoalchemy import MongoAlchemy

from datetime import datetime

import tensorflow as tf

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import math


app = Flask(__name__)

app.config['MONGOALCHEMY_DATABASE'] = 'heroku_cjxxw05p'

app.config['MONGOALCHEMY_CONNECTION_STRING']="mongodb://root:Phongthien1308@ds247637.mlab.com:47637/heroku_cjxxw05p"

db = MongoAlchemy(app)
# var
api='https://www.bitsler.com/api/bet-dice'
token='dfa6a226bbe9d38b695517f7d5012122133813887d1695afa3ca954da2c9b6467bd0cacf19eff0b231d36a894e684e29726ff572c216c076a4284feb94265baa'
times=0;
high_str=0
low_str=0
min_amount=0.1
amount=min_amount




high_total=min_amount
low_total=min_amount
guess=True
target =49.5
max_amount=0
total=0
number=0

id=2451026534948891
page_token='EAAFS8KvsJCoBAIkf6pMsAaS86XwHxR90pewZB6nTVCGhrQaxNNsg1Bxgu67mzdDpRw6fHuft5MPqySfjrjWB2SUkI6ZAPgzNKk7rfFDzqMZBxjgLG18ePmzDGdrxs87GJGU4lL4ZAvhEloZBDx4OoqrZBVzqbq6AjM7gcKspY6S44HKvZCUR8B4'
data_db=None
class DiceData(db.Document):
    query_class= BaseQuery
    number = db.FloatField()
    over   =db.BoolField() 
    time   =db.DateTimeField()
    target = db.FloatField()


#function
def call_api(coin,target,amount,guess):
      data={'access_token':token,'currency':coin,'target':target,'amount':amount,'over':guess}
      res= requests.post(api,data=data)
      print( res.json())
      return res.json()

def save_db(val):
     dice = DiceData(number=val['result'],time=datetime.now(),over=val['over'],target=val['target']);
     dice.save()

def train_data(number_res):
     global times,data_db,low_str

     number=0

     if times==0:
        b= DiceData.query.filter()
        data_db = [{'number': i.number,}for i in b]
     else:
        data_db.append({'number':number_res})
     
     
     states = pd.DataFrame(data_db)
     temperature= np.array(states['number']);
          
            
     num_periods =1
     f_horizon = 1
     x_train = temperature[:(len(temperature)-1)]
     x_batches = x_train.reshape(-1, num_periods, 1)

     y_train = temperature[1:(len(temperature))]
     y_batches = y_train.reshape(-1, num_periods, 1)
     X_test =temperature[-(1000):][:1000].reshape(-1, num_periods, 1)
   
     tf.reset_default_graph()

     rnn_size = 100
     learning_rate=0.001

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
     if low_str>5 or times== 0:
        low_str=0
        for epoch in range(epochs):

            train_dict = {X: x_batches, Y: y_batches}
            sess.run(train_step, feed_dict=train_dict)
            
        saver = tf.train.Saver()
        save_path = saver.save(sess, "models/model.ckpt")
     times=times+1
  
     saver = tf.train.Saver()
     with tf.Session() as sess:
  # Restore variables from disk.
         saver.restore(sess, "models/model.ckpt")
         y_pred=sess.run(outputs, feed_dict={X: X_test})
        
         number=y_pred[999][0][0]
     return number

def update_high(number,guess_number):
    global high_str
    if number>50.5 and guess_number<50.5:
       high_str+=1
    else:
        high_str=0

def update_low(number,guess_number):
    global low_str
    if number<49.5 and guess_number>49.5:
       low_str+=1
    else:
        low_str=0   
def is_true(number,guess_number):
    global low_str
    if number<49.5 and guess_number>49.5:
        
        low_str=low_str+1
    elif  number>50.5 and guess_number <50.5:
        low_str=low_str+1
    elif  number>50.5 and guess_number >50.5:
        low_str=0
    elif number<49.5 and guess_number<49.5:
         low_str=0
    

def reset_amount(amount,guess):
    global min_amount,high_total,low_total,total
    if amount>min_amount and guess==True:
        high_total=min_amount
        low_total=min_amount
  
    if amount>min_amount and guess==False:
        low_total=min_amount
        high_total=min_amount 
      
    if(amount<0):
        high_total=high_total+min_amount*2
        low_total = low_total+min_amount*2
        
      
def is_high_bet(number):
    return number>50.5


def is_low_bet(number):
    return number<49.5 


def send_message(recipient_id, message_text):



    params = {
        "access_token": page_token
    }
    headers = {
        "Content-Type": "application/json"
    }
    data = json.dumps({
        "recipient": {
            "id": recipient_id
        },
        "message": {
            "text": message_text
        }
    })
    r = requests.post("https://graph.facebook.com/v2.6/me/messages", params=params, headers=headers, data=data)
    if r.status_code != 200:
        print(r.json())

def main():
    
    global amount,min_amount,high_total,low_total,guess,target,max_amount,number,low_str,total
    
    val=call_api('doge',target,amount,guess)
    
  
    if val['success']==False:
         send_message(id,"finshed ")

    reset_amount(float(val['profit']),val['over'])
    is_true(val['result'],number)
       
      
    save_db(val)
   
   
    number = train_data(val['result'])
    # print(number)
    
    if is_high_bet(number):
        target=50.49
        guess=True
        amount=high_total+low_total
        high_total=high_total+amount+min_amount
    elif is_low_bet(number):
        target=49.5
        guess=False
        amount= low_total+high_total
        low_total= low_total+amount+min_amount
        
    else:
       amount=min_amount
  
    if amount >max_amount:
         max_amount=amount
    print(max_amount)
    print(total)

for i in range(1000000):  
    main()

