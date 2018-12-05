import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from com.test.msbd5001.train_model_02 import preprocess_data, normalize_data
from com.test.msbd5001.methods import one_hot_encoder, get_train_and_valid_set, k_fold

from xgboost import XGBRegressor, XGBClassifier
import xgboost as xgb
from lightgbm import LGBMRegressor

csv_file = pd.read_csv('train.csv')
standard_features = preprocess_data(csv_file)

x_axis = preprocess_data(train_features=csv_file)
x_axis = normalize_data(x_axis, standard_features)
x_axis = one_hot_encoder(x_axis)

y_axis = csv_file['time'].values.tolist()
y_judge_axis = (csv_file['time'] > 5).astype(int).values.tolist()

k = 8

x_axis, y_axis, x_valid, y_valid = get_train_and_valid_set(x_axis, y_axis, 0)
# x_axis, y_judge_axis, x_valid, y_judege_valid = get_train_and_valid_set(x_axis, y_judge_axis, 0)

x_k_list = k_fold(x_axis, k)
y_k_list = k_fold(y_axis, k)
# y_judge_k_list = k_fold(y_judge_axis, k)


boosting_type_list = ['gbdt', 'dart']
num_leaves_list = list(range(3, 8, 1))
max_depth_list = list(range(3, 8, 1))

ne_list = list(range(1000, 2501, 100))
sub_sample_bin_list = [38]
min_split_gain_list = [0]

min_child_weight_list = list(range(0, 150001, 1000))
min_child_sample_list = [1]
sub_sample_list = [0]

col_list = list(range(60, 75, 2))
reg_alpha_list = list(range(40, 81, 5))
reg_lambda_list = [0]

importance_type_list = ['split', 'gain']

sum_mse = 0
acc_score = 0

best = 10000
best_a = 10000
best_b = 10000
best_c = 10000

# sum_mse = 0

# for a in col_list:
#     for b in reg_alpha_list:
#         for c in reg_lambda_list:
# for a in importance_type_list:
sum_mse = 0
for i in range(k):
    # xgbr = XGBRegressor(n_estimators=100, max_depth=5, max_leaf_nodes=5, random_state=42)
    # xgbr = XGBRegressor(n_estimators=100, max_depth=6, max_leaf_nodes=4, subsample=0.88, gamma=0.79,
    #                     reg_alpha=0.4, random_state=42)
    # xgbr = XGBClassifier()
    xgbr = LGBMRegressor(boosting_type='gbdt', n_estimators=500, num_leaves=5, max_depth=7,
                         subsample_for_bin=38, min_split_gain=0, min_child_weight=15,
                         min_child_samples=1, colsample_bytree=0.66, reg_alpha=0, reg_lambda=0,
                         random_state=42)
    xgbr.fit(np.array(x_k_list[i]['train']), np.array(y_k_list[i]['train']))
    # xgbr.fit(np.array(x_k_list[i]['train']), np.array(y_judge_k_list[i]['train']))

    y_pred = np.array(xgbr.predict(np.array(x_k_list[i]['test'])))
    y_pred = [0 if y < 0 else y for y in y_pred]

    print(y_k_list[i]['test'])
    print(y_pred)

    sum_mse += mean_squared_error(y_k_list[i]['test'], y_pred)
    # acc_score += accuracy_score(y_judge_k_list[i]['test'], y_pred)

# print(a / 10, b / 10, c / 10)
print(sum_mse / k)

# print(acc_score / k)
