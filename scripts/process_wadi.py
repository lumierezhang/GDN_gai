import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler


# max min(0-1)
def norm(train, test):

    normalizer = MinMaxScaler(feature_range=(0, 1))# scale training data to [0,1] range
    # print("normalizer")
    normalizer_test = MinMaxScaler(feature_range=(0, 1))
    # print("normalizer_test")
    train_ret = normalizer.fit_transform(train)
    # print("train_ret")
    test_ret = normalizer_test.fit_transform(test)
    # print("test_ret")

    return train_ret, test_ret


# downsample by 10
def downsample(data, labels, down_len):
    np_data = np.array(data)
    np_labels = np.array(labels)

    orig_len, col_num = np_data.shape

    down_time_len = orig_len // down_len

    np_data = np_data.transpose()
    # print('before downsample', np_data.shape)

    d_data = np_data[:, :down_time_len*down_len].reshape(col_num, -1, down_len)
    d_data = np.median(d_data, axis=2).reshape(col_num, -1)

    d_labels = np_labels[:down_time_len*down_len].reshape(-1, down_len)
    # if exist anomalies, then this sample is abnormal
    d_labels = np.round(np.max(d_labels, axis=1))

    d_data = d_data.transpose()

    # print('after downsample', d_data.shape, d_labels.shape)

    return d_data.tolist(), d_labels.tolist()


def main():

    train = pd.read_csv('../data/wadi/WADI_14days_new.csv', sep=",",index_col=0)  # 128
    test = pd.read_csv('../data/wadi/WADI_attackdataLABLE.csv',sep=",", index_col=0)  # 130
    test.rename(columns={"Attack LABLE (1:No Attack, -1:Attack)":"attack"},inplace=1)
    test.replace({"attack":{1:0,-1:1}},inplace=True)
    # print(train.shape)
    # print(test.shape)
    # print(train)
    # print(test)


    train = train.iloc[:, 2:]
    test = test.iloc[:, 2:]
    print(test.shape)

    train = train.fillna(train.mean())
    test = test.fillna(test.mean())
    train = train.fillna(0)
    test = test.fillna(0)
    # print(test.shape,"after fillna")
    # print(train,"after fillna")

    # trim column names
    train = train.rename(columns=lambda x: x.strip())
    test = test.rename(columns=lambda x: x.strip())
    # print(test.shape,"after rename")
    # print(train,"after rename")

    train_labels = np.zeros(len(train))
    test_labels = test.attack

    test = test.drop(columns=['attack']) # (172801,126)
    # print(test.shape,"after drop")

    # cols = [x[46:] for x in train.columns] # remove column name prefixes
    # train.columns = cols
    # print(train.columns)
    # test.columns = cols
    # print(test.columns)
    print(train,"66666")

    x_train, x_test = norm(train.values, test.values)
    # print(x_train,"x_train")
    # print(train,"111")
    #
    # print(x_train.shape,"x_train shape")
    # print(x_test.shape,"x_test shape")
    # print(train.shape,"after x_train train shape")
    # print(test.shape,"after x_test test shape")
    # print(train)

    for i,col in enumerate(train.columns):
        train.iloc[:, [i]] = x_train[:, i]
        test.iloc[:,[i]] = x_test[:, i]
    # print(train)
    # exit()


    d_train_x, d_train_labels = downsample(train.values, train_labels, 10)
    d_test_x, d_test_labels = downsample(test.values, test_labels, 10)

    train_df = pd.DataFrame(d_train_x, columns = train.columns)
    test_df = pd.DataFrame(d_test_x, columns = test.columns)


    test_df['attack'] = d_test_labels
    train_df['attack'] = d_train_labels

    train_df = train_df.iloc[2160:]

    # exit()
    train_df.to_csv('../data/wadi/WADI_14days_new.csv')
    test_df.to_csv('../data/wadi/WADI_attackdataLABLE.csv')
    print("finish")
    f = open('../data/wadi/list.txt', 'w')
    for col in train.columns:
        f.write(col+'\n')
    f.close()

if __name__ == '__main__':
    main()
