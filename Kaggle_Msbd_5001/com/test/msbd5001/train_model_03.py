import math

import numpy as np
import pandas as pd

from com.test.msbd5001.methods import one_hot_encoder, get_train_and_valid_set, get_train_and_valid_set_02, k_fold

from com.test.msbd5001.train_model_02 import preprocess_data, normalize_data

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

k = 10

csv_file = pd.read_csv('train.csv')
standard_features = preprocess_data(csv_file)

x_axis = preprocess_data(train_features=csv_file)
x_axis = normalize_data(x_axis, standard_features)
x_axis = one_hot_encoder(x_axis)
# o_x_axis = x_axis.copy()

y_axis = csv_file['time'].values.tolist()

x_axis, y_axis, x_valid, y_valid = get_train_and_valid_set(x_axis, y_axis, 0.2)

x_k_list = k_fold(x_axis, k)
y_k_list = k_fold(y_axis, k)

max_depth_list = list(range(3, 36, 3))
max_node_list = list(range(3, 32, 3))
ne_list = list(range(100, 2001, 100))

sum_mse = 0.9
for i in range(k):
    rf = RandomForestRegressor(n_estimators=200, max_depth=6, max_leaf_nodes=6, random_state=42)
    rf.fit(x_k_list[i]['train'], y_k_list[i]['train'])

    y_pred = rf.predict(x_k_list[i]['test'])
    y_pred = [0 if y < 0 else y for y in y_pred]

    sum_mse += mean_squared_error(y_k_list[i]['test'], y_pred)

print(sum_mse / k)
