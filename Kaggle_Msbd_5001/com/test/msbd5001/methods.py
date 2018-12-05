import pandas as pd
import random


def k_fold(whole_list, k):
    sub_len = int(len(whole_list) / k)
    k_list = list()
    for i in range(k):
        empty_dict = dict()
        empty_dict['train'] = whole_list[sub_len:]
        empty_dict['test'] = whole_list[:sub_len]
        whole_list = whole_list[sub_len:] + whole_list[:sub_len]
        k_list.append(empty_dict)

    return k_list


def one_hot_encoder(train_features):
    hot_dict = dict()

    for column in train_features:
        if train_features[column].dtype == 'object':
            hot_dict[column] = list(sorted(set(train_features[column])))

    x_axis = list()
    for index, row in train_features.iterrows():
        empty_row = list()
        for column in train_features:
            if train_features[column].dtype == 'object':
                for t in hot_dict[column]:
                    if t == row[column]:
                        empty_row.append(1)
                    else:
                        empty_row.append(0)

            else:
                empty_row.append(row[column])
        x_axis.append(empty_row.copy())

    return x_axis


def get_train_and_valid_set(x_axis, y_axis, valid_size, shuffle=True):
    valid_size = int(valid_size * len(x_axis))

    if shuffle:
        c = list(zip(x_axis, y_axis))
        random.shuffle(c)
        x_axis, y_axis = zip(*c)

    x_train = x_axis[valid_size:]
    y_train = y_axis[valid_size:]

    x_valid = x_axis[:valid_size]
    y_valid = y_axis[:valid_size]

    return x_train, y_train, x_valid, y_valid


def get_train_and_valid_set_02(x_axis, y_axis, another_y_axis, valid_size, shuffle=True):
    valid_size = int(valid_size * len(x_axis))

    if shuffle:
        c = list(zip(x_axis, y_axis, another_y_axis))
        random.shuffle(c)
        x_axis, y_axis, another_y_axis = zip(*c)

    x_train = x_axis[valid_size:]
    y_train = y_axis[valid_size:]
    another_y_train = another_y_axis[valid_size:]

    x_valid = x_axis[:valid_size]
    y_valid = y_axis[:valid_size]
    another_y_valid = another_y_axis[:valid_size]

    return x_train, y_train, another_y_train, x_valid, y_valid, another_y_valid


if __name__ == '__main__':
    a_dict = dict(a=[1, 2, 3, 4, 5], b=['a', 'b', 'c', 'd', 'd'])
    a_dict = pd.DataFrame(a_dict)

    print(a_dict.a.values)

    # print(one_hot_encoder(a_dict))
