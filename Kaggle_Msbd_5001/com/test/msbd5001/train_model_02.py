import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from lightgbm import LGBMRegressor

from com.test.msbd5001.methods import one_hot_encoder, get_train_and_valid_set, k_fold, get_train_and_valid_set_02

csv_file = pd.read_csv('train.csv')

small_csv = csv_file[csv_file['time'] <= 5]

big_csv = csv_file[csv_file['time'] > 5]


def normalize_data(train_features, standard_features):
    for column in train_features:
        if train_features[column].dtype != 'object':
            train_features[column] = (train_features[column] - standard_features[column].mean()) / standard_features[
                column].std()

    return train_features


def preprocess_data(train_features):
    for t in ['id', 'time']:
        if t in train_features:
            train_features = train_features.drop(columns=[t])

    # train_features['noise'] = train_features['scale'] * train_features['flip_y']

    # train_features.drop(columns=['l1_ratio'])
    train_features.drop(columns=['alpha'])
    train_features.drop(columns=['random_state'])
    train_features.drop(columns=['n_clusters_per_class'])
    train_features.drop(columns=['flip_y'])
    train_features.drop(columns=['scale'])

    train_features['n_jobs'] = train_features['n_jobs'].map(lambda x: 5 if x < 0 else math.log2(x))

    train_features.loc[train_features['penalty'] == 'l2', 'l1_ratio'] = 0
    train_features.loc[train_features['penalty'] == 'none', 'l1_ratio'] = 0

    # for column in train_features:
    #     if train_features[column].dtype != 'object' and column != 'time':
    #         train_features[column] = (train_features[column] - train_features[column].min()) / (
    #                 train_features[column].max() - train_features[column].min())

    train_features['n_informative / n_features'] = train_features['n_informative'] / train_features['n_features']

    # for column in train_features:
    #     if train_features[column].dtype != 'object':
    #         train_features[column] = (train_features[column] - train_features[column].mean()) / train_features[
    #             column].std()

    # train_features['max_iter_2'] = train_features['max_iter'] ** 2
    # train_features['max_iter_3'] = train_features['max_iter'] ** 3

    # train_features['n_samples_2'] = train_features['n_samples'] ** 2
    # train_features['n_samples_3'] = train_features['n_samples'] ** 3

    # train_features['n_informative_2'] = train_features['n_informative'] ** 2
    # train_features['n_informative_3'] = train_features['n_informative'] ** 3

    # train_features['max_iter x n_samples'] = train_features['max_iter'] * train_features['n_samples']
    # train_features['max_iter x n_features'] = train_features['max_iter'] * train_features['n_features']
    # train_features['n_samples x n_features'] = train_features['n_samples'] * train_features['n_features']

    return train_features


standard_features = preprocess_data(csv_file)

x_axis = preprocess_data(train_features=csv_file)
x_axis = normalize_data(train_features=x_axis, standard_features=standard_features)
# print(x_axis)

x_axis = one_hot_encoder(x_axis)

y_axis = csv_file['time'].values.tolist()
y_log_axis = np.log(csv_file['time']).values.tolist()

# print(x_axis, '\n', y_axis)

k = 5

x_axis, y_axis, x_valid, y_valid = get_train_and_valid_set(x_axis, y_axis, 0.2)
# x_axis, y_log_axis, _, _ = get_train_and_valid_set(x_axis, y_log_axis, 0)

# print(x_axis)

x_k_list = k_fold(x_axis, k)
y_k_list = k_fold(y_axis, k)
y_log_k_list = k_fold(y_log_axis, k)

# lr = LinearRegression()

train_sum_mse = 0
sum_mse = 0
acc_score = 0

for i in range(k):
    # mlpr = MLPRegressor(solver='adam', learning_rate='adaptive', learning_rate_init=0.01,
    #                     hidden_layer_sizes=(1280, 64, 32, 16), random_state=42, max_iter=6000, batch_size=120,
    #                     alpha=0.09)
    # mlpr.fit(x_k_list[i]['train'], y_k_list[i]['train'])

    # # # mlpr.fit(x_k_list[i]['train'], y_log_k_list[i]['train'])
    #
    # mlpr_02 = MLPRegressor(solver='adam', alpha=0.05, learning_rate='adaptive', hidden_layer_sizes=(8, 8, 8, 8, 8),
    #                        random_state=17, max_iter=3000, batch_size=80)
    # # mlpr_02 = RandomForestRegressor(n_estimators=128, max_leaf_nodes=32, max_depth=16, random_state=17)
    # mlpr_02.fit(x_k_list[i]['train'], y_k_list[i]['train'])
    #
    # mlpr_03 = ExtraTreesRegressor(n_estimators=108, max_leaf_nodes=32, max_depth=32, random_state=42)
    # mlpr_03.fit(x_k_list[i]['train'], y_k_list[i]['train'])
    #
    # mlpr_04 = RandomForestRegressor(n_estimators=108, max_leaf_nodes=32, max_depth=32, random_state=17)
    # mlpr_04.fit(x_k_list[i]['train'], y_k_list[i]['train'])
    #
    # mlpr_05 = LinearRegression()
    # mlpr_05.fit(x_k_list[i]['train'], y_k_list[i]['train'])
    #
    # # xgbr = LGBMRegressor(boosting_type='gbdt', n_estimators=500, num_leaves=5, max_depth=7,
    # #                      subsample_for_bin=38, min_split_gain=0, min_child_weight=15,
    # #                      min_child_samples=1, colsample_bytree=0.66, reg_alpha=0, reg_lambda=0,
    # #                      random_state=42)
    # # xgbr.fit(x_k_list[i]['train'], y_k_list[i]['train'])
    #
    # # mlpr_06 = SVR()
    # # mlpr_06.fit(x_k_list[i]['train'], y_k_list[i]['train'])
    #
    # pred_01 = mlpr.predict(x_k_list[i]['train'])
    # pred_02 = mlpr_02.predict(x_k_list[i]['train'])
    # pred_03 = mlpr_03.predict(x_k_list[i]['train'])
    # pred_04 = mlpr_04.predict(x_k_list[i]['train'])
    # pred_05 = mlpr_05.predict(x_k_list[i]['train'])
    # # pred_06 = xgbr.predict(x_k_list[i]['train'])
    # pred_x_axis = list(zip(pred_01, pred_02, pred_03, pred_04, pred_05))
    #
    # mlpr_super = MLPRegressor(solver='lbfgs', alpha=0.05, learning_rate='adaptive',
    #                           hidden_layer_sizes=(1280, 64, 32, 32), random_state=32, max_iter=2500, batch_size=108)
    #
    # mlpr_super.fit(pred_x_axis, y_k_list[i]['train'])
    #
    # y_pred_01 = np.array(mlpr.predict(x_k_list[i]['test']))
    # y_pred = np.array(mlpr.predict(x_k_list[i]['test']))
    # y_pred_02 = np.array(mlpr_02.predict(x_k_list[i]['test']))
    # y_pred_03 = np.array(mlpr_03.predict(x_k_list[i]['test']))
    # y_pred_04 = np.array(mlpr_04.predict(x_k_list[i]['test']))
    # y_pred_05 = np.array(mlpr_05.predict(x_k_list[i]['test']))
    # # y_pred_06 = np.array(xgbr.predict(x_k_list[i]['test']))
    #
    # test_x_axis = list(zip(y_pred_01, y_pred_02, y_pred_03, y_pred_04, y_pred_05))
    #
    # y_pred = np.array(mlpr_super.predict(test_x_axis))
    #
    # # print(y_k_list[i]['test'])
    # # print(y_pred.tolist())
    # # print(y_pred_01)
    # # print('02:', y_pred_02)
    # # y_pred = y_pred_01 * 0.625 + y_pred_02 * 0.25 + y_pred_03 * 0.125
    #
    # y_pred = [0 if y < 0 else y for y in y_pred]
    #
    # y_pred_01_train = np.array(mlpr.predict(x_k_list[i]['train']))
    # # y_pred = np.array(mlpr.predict(x_k_list[i]['test']))
    # y_pred_02_train = np.array(mlpr_02.predict(x_k_list[i]['train']))
    # y_pred_03_train = np.array(mlpr_03.predict(x_k_list[i]['train']))
    # y_pred_04_train = np.array(mlpr_04.predict(x_k_list[i]['train']))
    # y_pred_05_train = np.array(mlpr_05.predict(x_k_list[i]['train']))
    # # y_pred_06_train = np.array(xgbr.predict(x_k_list[i]['train']))
    # train_x_axis = list(
    #     zip(y_pred_01_train, y_pred_02_train, y_pred_03_train, y_pred_04_train, y_pred_05_train))
    #
    # y_pred_train = mlpr_super.predict(train_x_axis)
    #
    # y_pred_train = [0 if y < 0 else y for y in y_pred_train]
    # # # y_pred = np.exp(y_pred)
    #
    # print(y_k_list[i]['test'])
    # # # print(y_k_list[i]['train'])
    # print(y_pred)
    # print()
    # #
    # # # print((np.array(y_k_list[i]['test']) > 5).astype(int).tolist())
    # # # print((np.array(y_pred) > 5).astype(int).tolist())
    # #
    # # # acc_score += accuracy_score((np.array(y_k_list[i]['test']) > 5).astype(int), (np.array(y_pred) > 5).astype(int))
    # # # sum_mse += mean_squared_error(y_k_list[i]['test'], y_pred)
    # sum_mse += mean_squared_error(y_k_list[i]['test'], y_pred)
    # train_sum_mse += mean_squared_error(y_k_list[i]['train'], y_pred_train)

    pass

# print(acc_score / k)
# print(sum_mse / k)
# print(train_sum_mse / k)
# print()
#
# lr = LinearRegression()
# lr.fit(x_axis, y_log_axis)
#
mlpr = MLPRegressor(solver='adam', learning_rate='adaptive', learning_rate_init=0.01,
                    hidden_layer_sizes=(1280, 64, 32, 16), random_state=42, max_iter=6000, batch_size=120,
                    alpha=0.09)
# mlpr = MLPRegressor(solver='adam', learning_rate='adaptive', learning_rate_init=0.01,
#                     hidden_layer_sizes=(1280, 64, 32), random_state=42, max_iter=6000, batch_size=30,
#                     alpha=0.09)
mlpr.fit(x_axis, y_axis)
# mlpr.fit(small_x_axis, small_y_axis)
# mlpr.fit(big_x_axis, big_y_axis)

mlpr_02 = MLPRegressor(solver='adam', alpha=0.05, learning_rate='adaptive', hidden_layer_sizes=(8, 8, 8, 8, 8),
                       random_state=17, max_iter=3000, batch_size=80)
mlpr_02.fit(x_axis, y_axis)

mlpr_03 = ExtraTreesRegressor(n_estimators=108, max_leaf_nodes=32, max_depth=32, random_state=42)
mlpr_03.fit(x_axis, y_axis)

mlpr_04 = RandomForestRegressor(n_estimators=108, max_leaf_nodes=32, max_depth=32, random_state=17)
mlpr_04.fit(x_axis, y_axis)

mlpr_05 = LinearRegression()
mlpr_05.fit(x_axis, y_axis)

pred_01 = np.array(mlpr.predict(x_axis))
pred_02 = np.array(mlpr_02.predict(x_axis))
pred_03 = np.array(mlpr_03.predict(x_axis))
pred_04 = np.array(mlpr_03.predict(x_axis))
pred_05 = np.array(mlpr_03.predict(x_axis))

pred_x_axis = list(zip(pred_01, pred_02, pred_03, pred_04, pred_05))

mlpr_super = MLPRegressor(solver='adam', alpha=0.005, learning_rate='adaptive',
                          hidden_layer_sizes=(1280, 64, 32, 16), random_state=32, max_iter=3200, batch_size=108)

mlpr_super.fit(pred_x_axis, y_axis)

y_pred_01 = np.array(mlpr.predict(x_valid))
y_pred_02 = np.array(mlpr_02.predict(x_valid))
y_pred_03 = np.array(mlpr_03.predict(x_valid))
y_pred_04 = np.array(mlpr_04.predict(x_valid))
y_pred_05 = np.array(mlpr_05.predict(x_valid))

valid_x_axis = list(zip(y_pred_01, y_pred_02, y_pred_03, y_pred_04, y_pred_05))
y_pred = np.array(mlpr_super.predict(valid_x_axis))

y_pred = y_pred_01
# y_pred = y_pred_01 * 0.625 + y_pred_02 * 0.25 + y_pred_03 * 0.125
y_pred = [0 if y < 0 else y for y in y_pred]

# print('01:', y_pred_01)
# print('02:', y_pred_02)

print(y_valid)
print(y_pred)

print(mean_squared_error(y_valid, y_pred))
# print(mean_squared_error(y_valid, y_pred_01))
