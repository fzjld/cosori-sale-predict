import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


def metrics1(y_true, y_pred):
    return 1 - (abs(y_true.sum() - y_pred.sum()) / y_true.sum())


def metrics2(y_true, y_pred):
    return 1 - (abs(y_true - y_pred) / y_true).mean()


def create_feature(df1):
    df = df1.copy()
    df['year'] = df.data_date.dt.year
    df['month'] = df.data_date.dt.month
    df['day'] = df.data_date.dt.day
    df['weekday'] = df.data_date.dt.weekday
    df['quarter'] = df.data_date.dt.quarter  # 季度
    df['weekofyear'] = df.data_date.dt.isocalendar().week
    df['weight'] = df.data_date.apply(lambda x: (
            x - pd.to_datetime('2019-12-02')).days)
    df['dayofyear'] = df.data_date.dt.day_of_year
    df = df.set_index('data_date')
    return df


def train(asin):
    '''
    :param asin: 产品asin
    :return: 训练后的模型，保存为产品asin+.pkl
    '''
    data = pd.read_csv('./data.csv', encoding='GBK')
    data.data_date = pd.to_datetime(data.data_date)
    train = data[data.asin == asin][[
        'data_date', 'ordered_units']].sort_values('data_date')

    # 异常值处理
    # 删除掉小于0的销量
    train.drop(train[train.ordered_units <= 0].index, inplace=True)

    #
    #     train.drop(train.ordered_units.nlargest(
    #         random.choice([3, 4, 5, 6])).index, inplace=True)

    # 特征构造
    def create_feature(df1):
        df = df1.copy()
        df['year'] = df.data_date.dt.year
        df['month'] = df.data_date.dt.month
        df['day'] = df.data_date.dt.day
        df['weekday'] = df.data_date.dt.weekday
        df['quarter'] = df.data_date.dt.quarter
        df['weekofyear'] = df.data_date.dt.weekofyear
        df['weight'] = df.data_date.apply(lambda x: (
                x - pd.to_datetime('2019-12-02')).days)
        df['dayofyear'] = df.data_date.dt.day_of_year
        df = df.set_index('data_date')
        return df

    train_featured = create_feature(train)

    # 随机网格参数搜索
    xgbr = XGBRegressor(n_estimators=1500,
                        learning_rate=0.1,
                        min_child_weight=0.2,
                        max_depth=5,
                        subsample=0.7,
                        colsample_bytree=1,
                        objective='reg:squarederror',
                        base_score=0.5,
                        gamma=2,
                        n_jobs=8)

    params_dict = {'n_estimators': np.arange(200, 600, 100),
                   'learning_rate': np.arange(0, 1, 0.1),
                   'max_depth': np.arange(2, 9),
                   'subsample': np.arange(0.5, 1.1, 0.1),
                   'colsample_bytree': np.arange(0.5, 1.1, 0.1),
                   'min_child_weight': np.arange(0, 5, 1),
                   'gamma': np.arange(0, 5, 1),
                   }

    timeKF = TimeSeriesSplit(n_splits=5)
    rscv = RandomizedSearchCV(xgbr, params_dict, n_iter=50, cv=timeKF)
    x = train_featured.drop('ordered_units', axis=1)
    y = train_featured.ordered_units
    rscv.fit(x, y)

    # 使用最好的参数进行训练
    xgbr = XGBRegressor(**rscv.best_params_)
    xgbr.fit(x, y, verbose=0)

    # 保存模型
    save_path = os.path.join('model', asin)
    joblib.dump(xgbr, save_path + '.pkl')




# class Prediction(object):
#     def __init__(self, asin, start, end):
#         self.asin = asin
#         self.start = start
#         self.end = end
#
#     def load_data(self):
#         self.data = pd.read_csv('./data.csv', encoding='GBK')
#         self.data.data_date = pd.to_datetime(self.data.data_date)
#         self.alldata = self.data[(self.data.asin == self.asin)][['data_date',
#                                                                  'ordered_units']].sort_values('data_date')
#         return self.alldata
#
#     def data_preprocessing(self, alldata):
#         '''
#             对数据进行预处理，
#             1.添加权重属性
#             2.删掉小于等于0的点
#         '''
#         # 删除小于等于0的异常值
#         self.alldata.drop(
#             self.alldata[self.alldata.ordered_units <= 0].index, inplace=True)
#
#         # 添加权重
#         self.alldata['weight'] = self.alldata.data_date.apply(lambda x: (
#                 x - pd.to_datetime('2019-12-02')).days)
#
#         return self.alldata
#
#     def remove_(self):
#         pass
#
#     def create_features(self, df):
#         df['year'] = df.data_date.dt.year
#         df['month'] = df.data_date.dt.month
#         df['day'] = df.data_date.dt.day
#         df['weekday'] = df.data_date.dt.weekday  # 周几
#         df['quarter'] = df.data_date.dt.quarter  # 季度
#         df['weekofyear'] = df.data_date.dt.weekofyear
#         df.drop('data_date', axis=1, inplace=True)
#
#         return df
#
#     def split_dataset(self, x, y, test_size, random_state):
#         x_train, x_test, y_train, y_test = train_test_split(
#             x, y, test_size=test_size, random_state=random_state)
#         return x_train, x_test, y_train, y_test
#
#     def encode(self, x_train, x_test):
#         weight_cat = list(range(1, 1000))
#         year_category = [2019, 2020, 2021, 2022]
#         month_category = list(range(1, 13))
#         day_category = list(range(1, 32))
#         weekday_cat = list(range(0, 7))
#         quarter_cat = [1, 2, 3, 4]
#         weekofyear_cat = list(range(1, 54))
#
#         encoder = OneHotEncoder(sparse=False,
#                                 categories=[weight_cat, year_category, month_category,
#                                             day_category, weekday_cat, quarter_cat, weekofyear_cat],
#                                 handle_unknown='ignore')
#
#         x_train = encoder.fit_transform(x_train)
#         x_test = encoder.transform(x_test)
#         return pd.DataFrame(x_train, columns=encoder.get_feature_names()), pd.DataFrame(x_test,
#                                                                                         columns=encoder.get_feature_names())
#
#     def train(self, x_train, y_train, x_test, y_test):
#         xgbr = XGBRegressor(n_estimators=200,
#                             learning_rate=0.35,
#                             max_depth=7,
#                             subsample=0.6,
#                             colsample_bytree=0.7,
#                             objective='reg:squarederror',
#                             base_score=0.5,
#                             gamma=1,
#                             n_jobs=8, verbosity=1)
#
#         xgbr.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)],
#                  early_stopping_rounds=20, verbose=True)
#
#         return xgbrclass Prediction(object):
#     def __init__(self, asin, start, end):
#         self.asin = asin
#         self.start = start
#         self.end = end
#
#     def load_data(self):
#         self.data = pd.read_csv('./data.csv', encoding='GBK')
#         self.data.data_date = pd.to_datetime(self.data.data_date)
#         self.alldata = self.data[(self.data.asin == self.asin)][['data_date',
#                                                                  'ordered_units']].sort_values('data_date')
#         return self.alldata
#
#     def data_preprocessing(self, alldata):
#         '''
#             对数据进行预处理，
#             1.添加权重属性
#             2.删掉小于等于0的点
#         '''
#         # 删除小于等于0的异常值
#         self.alldata.drop(
#             self.alldata[self.alldata.ordered_units <= 0].index, inplace=True)
#
#         # 添加权重
#         self.alldata['weight'] = self.alldata.data_date.apply(lambda x: (
#                 x - pd.to_datetime('2019-12-02')).days)
#
#         return self.alldata
#
#     def remove_(self):
#         pass
#
#     def create_features(self, df):
#         df['year'] = df.data_date.dt.year
#         df['month'] = df.data_date.dt.month
#         df['day'] = df.data_date.dt.day
#         df['weekday'] = df.data_date.dt.weekday  # 周几
#         df['quarter'] = df.data_date.dt.quarter  # 季度
#         df['weekofyear'] = df.data_date.dt.weekofyear
#         df.drop('data_date', axis=1, inplace=True)
#
#         return df
#
#     def split_dataset(self, x, y, test_size, random_state):
#         x_train, x_test, y_train, y_test = train_test_split(
#             x, y, test_size=test_size, random_state=random_state)
#         return x_train, x_test, y_train, y_test
#
#     def encode(self, x_train, x_test):
#         weight_cat = list(range(1, 1000))
#         year_category = [2019, 2020, 2021, 2022]
#         month_category = list(range(1, 13))
#         day_category = list(range(1, 32))
#         weekday_cat = list(range(0, 7))
#         quarter_cat = [1, 2, 3, 4]
#         weekofyear_cat = list(range(1, 54))
#
#         encoder = OneHotEncoder(sparse=False,
#                                 categories=[weight_cat, year_category, month_category,
#                                             day_category, weekday_cat, quarter_cat, weekofyear_cat],
#                                 handle_unknown='ignore')
#
#         x_train = encoder.fit_transform(x_train)
#         x_test = encoder.transform(x_test)
#         return pd.DataFrame(x_train, columns=encoder.get_feature_names()), pd.DataFrame(x_test,
#                                                                                         columns=encoder.get_feature_names())
#
#     def train(self, x_train, y_train, x_test, y_test):
#         xgbr = XGBRegressor(n_estimators=200,
#                             learning_rate=0.35,
#                             max_depth=7,
#                             subsample=0.6,
#                             colsample_bytree=0.7,
#                             objective='reg:squarederror',
#                             base_score=0.5,
#                             gamma=1,
#                             n_jobs=8, verbosity=1)
#
#         xgbr.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)],
#                  early_stopping_rounds=20, verbose=True)
#
#         return xgbr
