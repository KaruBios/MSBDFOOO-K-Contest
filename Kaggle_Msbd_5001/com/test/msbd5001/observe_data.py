import pandas as pd
import numpy as np
from scipy import optimize

import math

from matplotlib.pyplot import show
import matplotlib.pyplot as plt

train_features = pd.read_csv('train.csv')
train_features = train_features.drop(columns=['id'])


# print(train_features)

def f_1(x, A, B):
    return A * x + B


def f_2(x, A, B, C):
    return A * x * x + B * x + C


def f_3(x, A, B, C, D):
    return A * (x ** 3) + B * (x ** 2) + C * x + D


if __name__ == '__main__':
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)

    # for index, row in train_features.iterrows():
    #     for column in train_features:
    #         if train_features[column].dtype != 'object':
    #             if row[column] <= 0:
    #                 train_features.at[index, column] = 0.000000001

    # print(train_features)

    train_features['n_jobs'] = train_features['n_jobs'].map(lambda x: 5 if x < 0 else math.log2(x))

    train_features.loc[train_features['penalty'] == 'l2', 'l1_ratio'] = 0
    train_features.loc[train_features['penalty'] == 'none', 'l1_ratio'] = 0
    print(train_features)

    # for column in train_features:
    #     if train_features[column].dtype != 'object' and column != 'time':
    #         train_features[column] = (train_features[column] - train_features[column].min()) / (
    #                 train_features[column].max() - train_features[column].min())

    for column in train_features:
        if train_features[column].dtype != 'object' and column != 'time':
            train_features[column] = (train_features[column] - train_features[column].mean()) / train_features[
                column].std()

    # print(train_features['n_jobs'])
    # train_features['time'] = np.log(train_features['time'])

    columns = [column for column in train_features]

    # sorted_train_features = train_features['time'].sort_values().reset_index()['time']
    # print(sorted_train_features)

    # sorted_train_features.astype(int).plot(grid=True)
    # show()

    # for i in range(1, len(columns)):
    #     for c in columns[i + 1:-1]:
    #         train_features[columns[i] + ' x ' + c] = train_features[columns[i]] * train_features[c]

    # for i in range(1, len(columns)):
    #     for c in columns[1:-1]:
    #         train_features[columns[i] + ' / ' + c] = (train_features[columns[i]] + 0.01) / (train_features[c] + 0.01)

    # train_features['n_informative / n_features'] = train_features['n_informative'] / (
    #             train_features['n_features'] + 0.01)

    # train_features['max_iter'] = np.log(train_features['max_iter'])

    # train_features['time_01'] = train_features['time']

    # corr_df = train_features.corr()
    #
    # print(corr_df)
    #
    # print(corr_df[abs(corr_df) > 0.1])

    # print(train_features)

    for column in train_features:
        if train_features[column].dtype != 'object':
            # train_features.sort_values(by=column).plot(x=column, y='time', kind='scatter')
            plt.figure(column)
            x0 = train_features[column]
            y0 = train_features['time']

            plt.scatter(x0, y0, c='#DDEECC')

            A1, B1 = optimize.curve_fit(f_1, x0, y0)[0]
            x1 = np.arange(x0.min(), x0.max(), 0.005)
            y1 = A1 * x1 + B1
            plt.plot(x1, y1, c='green')

            A2, B2, C2 = optimize.curve_fit(f_2, x0, y0)[0]
            x2 = np.arange(x0.min(), x0.max(), 0.005)
            y2 = A2 * (x2 ** 2) + B2 * x2 + C2
            plt.plot(x2, y2, c='blue')

            A3, B3, C3, D3 = optimize.curve_fit(f_3, x0, y0)[0]
            x3 = np.arange(x0.min(), x0.max(), 0.005)
            y3 = A3 * (x3 ** 3) + B3 * (x3 ** 2) + C3 * x3 + D3
            plt.plot(x3, y3, c='#DD55EE')

            show()

        else:
            apple = train_features.groupby(by=column)[[column, 'time']].mean()
            print(apple)
            apple.plot(y='time')
            show()
        pass

    # train_features.sort_values('l1_ratio').groupby(by='penalty').plot(x='penalty', y='l1_ratio')
    # show()

    # print(train_features.groupby(by=['penalty', 'l1_ratio'])['l1_ratio'].count())

    # print(train_features)

    # print(train_features.sort_values('time'))

    # print(train_features.groupby(by='penalty')[['penalty', 'time']].mean())

    # show()
