#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import joblib
import pickle
import xgboost as xgb
import os


# In[2]:


def train_model():
    data = pd.read_csv('./data.csv', encoding='GBK')
    train = data.groupby(['asin', 'data_date']).agg({'ordered_units': sum})
    train = train.reset_index()
    train.data_date = pd.to_datetime(train.data_date)
    train = train[(train['data_date'] >= '2019-12-02')
                  & (train['data_date'] <= '2021-07-22')]
    train.loc[:, 'year'] = train.data_date.dt.year
    train.loc[:, 'month'] = train.data_date.dt.month
    train.loc[:, 'day'] = train.data_date.dt.day
    train = train.drop('data_date', axis=1)
    x_train = train.drop('ordered_units', axis=1)
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    x_train = encoder.fit_transform(x_train)
    y_train = train['ordered_units']
    params = {'eta': 0.2, 'max_depth': 4, 'subsample': 0.8, 'colsample_bytree': 1, 'n_jobs': 8, 'verbosity': 0}
    model = xgb.train(params, xgb.DMatrix(x_train, y_train), 200)
    # joblib.dump(encoder, os.path.join('.', 'encoder.pkl'))
    # joblib.dump(model, os.path.join('.', 'regressor.pkl'))


train_model()


def test(asin, start, end):
    encoder = joblib.load(os.path.join('.', 'encoder.pkl'))
    model = joblib.load(os.path.join('.', 'reg_model.pkl'))
    group = pd.DataFrame()
    group.loc[:, 'data_date'] = pd.date_range(start, end)
    group.loc[:, 'asin'] = 'B074NYJL9J'
    group.loc[:, 'year'] = group.data_date.dt.year
    group.loc[:, 'month'] = group.data_date.dt.month
    group.loc[:, 'day'] = group.data_date.dt.day
    group.drop(['data_date'], axis=1, inplace=True)
    x_test = encoder.transform(group)
    print(model.predict(xgb.DMatrix(x_test)))


test('B085W3GN5J', '2019-10-1', '2021-10-1')
