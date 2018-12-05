import pandas as pd
import numpy as np

from com.test.msbd5001.train_model_02 import preprocess_data, normalize_data
from com.test.msbd5001.methods import one_hot_encoder, k_fold, get_train_and_valid_set

from keras.layers import Dense, LeakyReLU, ReLU, ELU
from keras import Sequential
import keras

from sklearn.metrics import mean_squared_error

csv_file = pd.read_csv('train.csv')

standard_features = preprocess_data(csv_file)

x_axis = preprocess_data(train_features=csv_file)
x_axis = normalize_data(train_features=x_axis, standard_features=standard_features)
x_axis = one_hot_encoder(x_axis)

column_count = len(x_axis[0])

y_axis = csv_file['time'].values.tolist()

# x_axis = np.array(x_axis)
# y_axis = np.array(y_axis)
# print(x_axis, y_axis)

x_axis, y_axis, x_valid, y_valid = get_train_and_valid_set(x_axis, y_axis, 0)

x_valid = np.array(x_valid)
y_valid = np.array(y_valid)

k = 5
x_k_list = k_fold(x_axis, k)
y_k_list = k_fold(y_axis, k)

act_layers = [LeakyReLU, ReLU, ELU]

alphas = [0.61]

for alpha in alphas:
    # for act_layer in act_layers:
    sum_mse = 0
    for i in range(k):
        # mini_gap = 100000
        # mini_step = 0

        model = Sequential()
        model.add(Dense(units=16, input_shape=(column_count,)))
        model.add(act_layers[0](alpha / 100))
        model.add(Dense(units=16, input_shape=(16,)))
        model.add(act_layers[0](alpha / 100))
        model.add(Dense(units=16, input_shape=(16,)))
        model.add(act_layers[0](alpha / 100))
        model.add(Dense(units=8, input_shape=(16,)))
        model.add(act_layers[0](alpha / 100))
        model.add(Dense(units=1, input_shape=(1,)))

        model.compile(loss='mse', optimizer='adam')

        for step in range(1001):
            cost = model.train_on_batch(np.array(x_k_list[i]['train']), np.array(y_k_list[i]['train']))
            if step % 100 == 0:
                # print(step)
                # print('train error:', cost)
                # y_pred = model.predict(np.array(x_k_list[i]['test']))
                # y_pred = [0 if y < 0 else y for y in y_pred]
                #
                # valid_cost = mean_squared_error(y_pred, y_k_list[i]['test'])
                # print('valid error:', valid_cost)
                # print()

                # if abs(valid_cost - cost) < mini_gap:
                #     print(step)
                #     print('train error:', cost)
                #     print('valid error:', valid_cost)
                #     print()
                #     mini_gap = abs(valid_cost - cost)
                #     mini_step = step
                pass

        y_pred = model.predict(np.array(x_k_list[i]['test']))
        y_pred = [0 if y < 0 else y for y in y_pred]

        valid_cost = mean_squared_error(y_pred, y_k_list[i]['test'])
        sum_mse += valid_cost

        # print(mini_step, mini_gap)

    print(alpha / 100)
    print(sum_mse / k)
    print()

model = Sequential()
model.add(Dense(units=16, input_shape=(column_count,)))
model.add(act_layers[0](alphas[0] / 100))
model.add(Dense(units=16, input_shape=(16,)))
model.add(act_layers[0](alphas[0] / 100))
model.add(Dense(units=16, input_shape=(16,)))
model.add(act_layers[0](alphas[0] / 100))
model.add(Dense(units=8, input_shape=(16,)))
model.add(act_layers[0](alphas[0] / 100))
model.add(Dense(units=1, input_shape=(1,)))

model.compile(loss='mse', optimizer='adam')

for step in range(1001):
    cost = model.train_on_batch(np.array(x_axis), np.array(y_axis))

y_pred = model.predict(np.array(x_valid))
print(mean_squared_error(y_pred, y_valid))
