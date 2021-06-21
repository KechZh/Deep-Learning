#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import os
import tensorflow as tf

from tensorflow import keras
from keras import Sequential, Input
from keras.layers import Dense, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


# In[2]:


path = 'data/air_visit_data.csv'
vis = pd.read_csv(path)
path = 'data/air_reserve.csv'
ares = pd.read_csv(path)
path = 'data/hpg_reserve.csv'
hres = pd.read_csv(path)
path = 'data/air_store_info.csv'
astore = pd.read_csv(path)
path = 'data/hpg_store_info.csv'
hstore = pd.read_csv(path)
path = 'data/store_id_relation.csv'
sid = pd.read_csv(path)
path = 'data/date_info.csv'
hol = pd.read_csv(path).rename(columns={'calendar_date':'visit_date'})


# In[3]:


hres = pd.merge(hres, sid, how='inner', on=['hpg_store_id'])
hres.drop('hpg_store_id', axis=1, inplace=True)
ares = ares.append(hres)

ares['visit_datetime'] = pd.to_datetime(ares['visit_datetime'])
ares['visit_date'] = ares['visit_datetime'].dt.date
ares.drop('visit_datetime', axis=1, inplace=True)
ares.drop('reserve_datetime', axis=1, inplace=True)

ares = ares.groupby(['air_store_id','visit_date'], as_index=False).sum().reset_index()
ares = ares.drop(['index'], axis=1)

vis['visit_datetime'] = pd.to_datetime(vis['visit_date'])
vis['visit_date'] = vis['visit_datetime'].dt.date

hol['visit_date'] = pd.to_datetime(hol['visit_date'])
hol['visit_date'] = hol['visit_date'].dt.date


# In[4]:


df = pd.merge(vis, ares, how='left', on=['air_store_id', 'visit_date'])
df = pd.merge(df, astore, how='inner', on='air_store_id')
df = pd.merge(df, hol, how='left', on='visit_date')

df['year'] = df['visit_datetime'].dt.year
df['month'] = df['visit_datetime'].dt.month
df['day'] = df['visit_datetime'].dt.day
df.drop('visit_datetime', axis=1, inplace=True)

features = [col for col in ['air_genre_name', 'air_area_name', 'day_of_week']]
for col in features:
    tmp = pd.get_dummies(pd.Series(df[col]))
    df = pd.concat([df, tmp], axis=1)
    df.drop([col], axis=1, inplace=True)

df.fillna(0, inplace=True)

df['visitors'] = np.log1p(df['visitors'])


# In[5]:


train = df[(df['year'] == 2016)]
train.drop('air_store_id', axis=1, inplace=True)
train.drop('visit_date', axis=1, inplace=True)
test = df[(df['year'] == 2017)]
test.drop('air_store_id', axis=1, inplace=True)
test.drop('visit_date', axis=1, inplace=True)

train_X = train.drop('visitors', axis=1)
train_Y = (train['visitors'])
test_X = test.drop('visitors', axis=1)
test_Y = (test['visitors'])


# In[6]:


model = Sequential()

model.add(Dense(100, activation = 'relu', input_shape = (train_X.shape[1],)))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(20, activation = 'relu'))
model.add(Dense(1, activation = 'linear'))


# In[7]:


model.compile(loss = 'mse', optimizer = Adam(lr = 0.001, decay = 0.0001), metrics = ['mean_squared_logarithmic_error'])


# In[8]:


model.fit(x = train_X, y = train_Y, epochs = 50, batch_size = 128)


# In[9]:


pred = model.evaluate(x = train_X, y = train_Y)
print('train RMSLE : ' + str(pred[1] ** 0.5))
pred = model.evaluate(x = test_X, y = test_Y)
print('test RMSLE : ' + str(pred[1] ** 0.5))


# In[ ]:




