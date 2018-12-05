from com.test.msbd5001.train_model import rf_list
import pandas as pd
import csv

from com.test.msbd5001.methods import one_hot_encoder

csv_file = pd.read_csv('test.csv')
x_axis = csv_file.drop(columns=['id'])

for column in x_axis:
    if x_axis[column].dtype != 'object':
        x_axis[column] = x_axis[column] / (x_axis[column].max() - x_axis[column].min())

x_axis = one_hot_encoder(x_axis)

pred_list = list()
for rf in rf_list:
    pred_list.append(rf.predict(x_axis))

y_pred = [sum([y[i] for y in pred_list]) / len(pred_list) for i in range(len(x_axis))]
# print(y_pred)
# print(mean_squared_error(y_axis, y_pred))

y_pred = [[i, y_pred[i]] for i in range(len(y_pred))]

# print(y_pred)

head = ['id', 'time']

with open('test_label.csv', 'w', newline='') as p_file:
    writer = csv.writer(p_file)
    writer.writerow(head)
    writer.writerows(y_pred)
