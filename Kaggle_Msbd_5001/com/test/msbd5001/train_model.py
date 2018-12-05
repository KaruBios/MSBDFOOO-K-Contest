import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from com.test.msbd5001.methods import one_hot_encoder, get_train_and_valid_set, k_fold

csv_file = pd.read_csv('train.csv')
x_axis = csv_file.drop(columns=['id', 'time'])
# x_axis = x_axis.drop(columns=['penalty'])
# x_axis = x_axis.drop(columns=['alpha', 'n_jobs', 'n_classes', 'n_clusters_per_class', 'n_informative'])

for column in x_axis:
    if x_axis[column].dtype != 'object':
        x_axis[column] = x_axis[column] / (x_axis[column].max() - x_axis[column].min())

x_axis = one_hot_encoder(x_axis)
# print(x_axis)

y_axis = csv_file['time'].values.tolist()

x_train, y_train, x_test, y_test = get_train_and_valid_set(x_axis, y_axis, 0.08, True)

k = 5

x_k_list = k_fold(x_train, k)
y_k_list = k_fold(y_train, k)

lr_list = list()
rf_list = list()
pred_list = list()

for i in range(k):
    # lr = LinearRegression()
    # lr.fit(x_train, y_train)
    # lr.fit(x_k_list[i]['train'], y_k_list[i]['train'])

    rf = RandomForestRegressor(n_estimators=100, max_depth=8, max_leaf_nodes=6)
    rf.fit(x_k_list[i]['train'], y_k_list[i]['train'])
    rf_list.append(rf)

    # lr_list.append(lr)

    # y_pred = [0 if a < 0 else a for a in lr.predict(x_test)]
    # y_pred = [0 if a < 0 else a for a in lr.predict(x_k_list[i]['train'])]
    y_pred = [0 if a < 0 else a for a in rf.predict(x_k_list[i]['train'])]
    # pred_list.append(y_pred)

    # print(y_pred)

    # print(mean_squared_error(y_test, y_pred))
    # print(mean_squared_error(y_k_list[i]['train'], y_pred))

if __name__ == '__main__':

    pred_list = list()
    for rf in rf_list:
        pred_list.append(rf.predict(x_axis))

    y_pred = [sum([y[i] for y in pred_list]) / len(pred_list) for i in range(len(y_axis))]
    print(y_pred)
    print(mean_squared_error(y_axis, y_pred))
