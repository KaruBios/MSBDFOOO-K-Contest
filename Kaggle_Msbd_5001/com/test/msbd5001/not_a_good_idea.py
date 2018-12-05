import pandas as pd
import csv
import numpy as np

# csv_file_list = list()
# csv_file_list.append(pd.read_csv('test_label_03.csv'))
# csv_file_list.append(pd.read_csv('test_label_04.csv'))
# csv_file_list.append(pd.read_csv('test_label_12.csv'))
# # csv_file_list.append(pd.read_csv('test_label_13.csv'))
# csv_file_list.append(pd.read_csv('test_label_14.csv'))
# csv_file_03 = pd.read_csv('test_label_05.csv')

csv_file = pd.read_csv('test_label_25.csv')
csv_file.loc[csv_file['time'] < 3, 'time'] *= 1.002
csv_file.loc[csv_file['time'] > 8, 'time'] *= 0.992
test_label = pd.DataFrame()
test_label['id'] = [str(int(i)) for i in range(100)]
# a_sum = csv_file_list[0]
# for i in range(len(csv_file_list) - 1):
#     a_sum += csv_file_list[i + 1]

# print(a_sum)
test_label = csv_file

test_label['id'] = test_label['id'].astype('int32').astype(str)

# print(test_label)

# print(test_label.values)
heads = ['id', 'time']

with open('test_label_29.csv', 'w', newline='') as p_file:
    writer = csv.writer(p_file)
    writer.writerow(heads)
    writer.writerows(test_label.values)

    pass
# print(test_label)
