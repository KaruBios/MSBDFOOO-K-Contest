import pandas as pd
import numpy as np
import math
import csv
from com.test.msbd5001.methods import one_hot_encoder
# from com.test.msbd5001.train_model_02 import preprocess_data, mlpr, mlpr_02
from com.test.msbd5001.train_model_02 import preprocess_data, mlpr, mlpr_02, mlpr_03, mlpr_04, mlpr_05, mlpr_super

csv_file = pd.read_csv('test.csv')
# x_axis = csv_file.drop(columns=['id'])
# # #
# # # x_axis = x_axis.drop(columns=['random_state'])
# # # x_axis = x_axis.drop(columns=['l1_ratio'])
# # # x_axis = x_axis.drop(columns=['penalty'])
# # # x_axis = x_axis.drop(columns=['alpha'])
# # # x_axis = x_axis.drop(columns=['n_informative'])
# # #
# # # x_axis['n_jobs'] = x_axis['n_jobs'].map(lambda x: np.log2(x_axis['n_jobs']).max() + 1 if x < 0 else math.log2(x))
# # # # x_axis['l1_ratio'] = np.abs(x_axis['l1_ratio'] - 0.4)
# # #
# # # for column in x_axis:
# # #     x_axis[column] = (x_axis[column] - x_axis[column].min()) / (x_axis[column].max() - x_axis[column].min())
# # #
# # # x_axis = one_hot_encoder(x_axis)
# # #
# # # x_axis_right = one_hot_encoder(csv_file['penalty'].to_frame())
# # #
# # # for i in range(len(x_axis)):
# # #     x_axis[i] = x_axis[i] + x_axis_right[i]
# # #
# # # y_pred = lr.predict(x_axis)

test_x_axis = preprocess_data(csv_file)
test_x_axis = one_hot_encoder(test_x_axis)

# y_pred = mlpr.predict(test_x_axis)
# y_pred = [0 if y < 0 else y for y in y_pred]

y_pred_01 = np.array(mlpr.predict(test_x_axis))
y_pred_02 = np.array(mlpr_02.predict(test_x_axis))
y_pred_03 = np.array(mlpr_03.predict(test_x_axis))
y_pred_04 = np.array(mlpr_04.predict(test_x_axis))
y_pred_05 = np.array(mlpr_05.predict(test_x_axis))

pred_x_axis = list(zip(y_pred_01, y_pred_02, y_pred_03, y_pred_04, y_pred_05))
y_pred = mlpr_super.predict(pred_x_axis)

# y_pred = y_pred_01 * 0.625 + y_pred_02 * 0.25 + y_pred_03 * 0.125
# y_pred = y_pred_01
y_pred = [0 if y < 0 else y for y in y_pred]

# print(y_pred)

# y_pred = [math.exp(y) for y in y_pred]

# print(y_pred)

# print(y_pred)

y_pred = [[i, y_pred[i]] for i in range(len(y_pred))]

head = ['id', 'time']

with open('test_label_14.csv', 'w', newline='') as p_file:
    writer = csv.writer(p_file)
    writer.writerow(head)
    writer.writerows(y_pred)
