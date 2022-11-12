import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler


# max min(0-1)
def norm(train, test):

    normalizer = MinMaxScaler(feature_range=(0, 1))# scale training data to [0,1] range
    print("normalizer")
    normalizer_test = MinMaxScaler(feature_range=(0, 1))
    print("normalizer_test")
    train_ret = normalizer.fit_transform(train)
    print("train_ret")
    test_ret = normalizer_test.fit_transform(test)
    print("test_ret")

    return train_ret, test_ret

# downsample by 10
def downsample(data, labels, down_len):
    np_data = np.array(data)
    np_labels = np.array(labels)
    print("np_labels")
    # print(np_data)
    # print(np_labels)

    orig_len, col_num = np_data.shape
    # print(orig_len)
    # print(col_num)

    down_time_len = orig_len // down_len
    # print(down_time_len)

    np_data = np_data.transpose()
    # print(np_data)

    d_data = np_data[:, :down_time_len*down_len].reshape(col_num, -1, down_len)
    # print(d_data)
    d_data = np.median(d_data, axis=2).reshape(col_num, -1)
    # print(d_data)
    # print("d_data2")

    d_labels = np_labels[:down_time_len*down_len].reshape(-1, down_len)
    # print(d_labels)

    # print("d_labels1")
    # if exist anomalies, then this sample is abnormal
    # print(type(d_labels[0][0]))

    d_labels=np.array(d_labels)
    # print(d_labels.shape)
    # d_labels.astype(int)
    # print(isinstance()) 是用来判断一个对象的变量类型。
    # d_labels=np.array(d_labels,dtype=int)
    temp=[]
    for i in range(d_labels.shape[0]):
        try:
            temp.append(np.max(d_labels[i]))
        except:
            print(i)
    # print(d_labels[17226])
    # print(d_labels[17230])
    # exit()
    d_labels = np.round(temp)
    # print(d_labels)
    # print("d_labels")


    d_data = d_data.transpose()
    # print(d_data)
    # print("d_data3")

    return d_data.tolist(), d_labels.tolist()


def main():

    test = pd.read_csv('../data/swat/SWaT_Dataset_Attack_v0.csv',sep=';',index_col=0)
    test.rename(columns={"Normal/Attack":"attack"},inplace=1)
    test.replace("Normal",0,inplace=True)
    test.replace("Attack",1,inplace=True)
    test.replace("A ttack",1,inplace=True)
    # print(test)
    train = pd.read_csv('../data/swat/SWaT_Dataset_Normal_v1.csv',sep=',',index_col=0)
    train.rename(columns={"Normal/Attack":"attack"},inplace=1)
    train.replace("Normal",0,inplace=True)
    # print(train)
    # exit(0)

    test = test.iloc[:, 1:]
    train = train.iloc[:, 1:]

    # train = train.fillna(train.mean())  # 缺失值用平均值填充
    # test = test.fillna(test.mean())
    train = train.fillna(0)
    test = test.fillna(0)

    # trim column names
    train = train.rename(columns=lambda x: x.strip())
    test = test.rename(columns=lambda x: x.strip())

    # print(len(test.columns),test.columns)
    # print(len(train.columns),train.columns)


    train_labels = train.attack
    test_labels = test.attack

    train = train.drop(columns=['attack'])
    test = test.drop(columns=['attack'])

    for i in list(train):  # i=49（从每一列标签开始循环）
        train[i] = train[i].apply(lambda x: str(x).replace(",", "."))
    train = train.astype(float)
    # print(train)
    for i in list(test):  # i=49（从每一列标签开始循环）
        test[i] = test[i].apply(lambda x: str(x).replace(",", "."))

    # print("enter norm")
    x_train, x_test = norm(train, test)

    # print("exit norm")
    for i, col in enumerate(train.columns):
        # print(col)
        train.loc[:, col] = x_train[:, i]
        test.loc[:, col] = x_test[:, i]


    d_train_x, d_train_labels = downsample(train.values, train_labels, 10)
    # print("train_downsample")
    d_test_x, d_test_labels = downsample(test.values, test_labels, 10)
    # print("test_downsample")

    train_df = pd.DataFrame(d_train_x, columns = train.columns)
    test_df = pd.DataFrame(d_test_x, columns = test.columns)

    test_df['attack'] = d_test_labels
    train_df['attack'] = d_train_labels

    train_df = train_df.iloc[2160:]



    train_df.to_csv('../data/swat/SWaT_Dataset_Normal_v1.csv')  # data into csv
    test_df.to_csv('../data/swat/SWaT_Dataset_Attack_v0.csv')
    print("finish")
    f = open('../data/swat/list.txt', 'w')
    for col in train.columns:
        f.write(col+'\n')
    f.close()

if __name__ == '__main__':
    main()
