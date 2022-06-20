from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import numpy as np
import torch

def load_data(path, batch_size, shuffle=True):

    data = pd.read_csv(path)

    y = data['sales'].values
    x = data.drop(['sales'], axis = 1).values
    x = data.drop(["date"], axis=1).values

    x, y = to_batches(x, y, batch_size, shuffle)

    x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random_state=0)

    transformer = preprocessing.PowerTransformer()
    y_train = transformer.fit_transform(y_train.reshape(-1,1))
    y_test = transformer.transform(y_test.reshape(-1,1))
    
    y_train = y_train.reshape(y_train.shape[0]//batch_size,batch_size,1)
    y_test = y_test.reshape(y_test.shape[0]//batch_size,batch_size,1)

    x_train = torch.tensor(x_train.astype(np.float32))
    y_train = torch.tensor(y_train.astype(np.float32))

    x_test = torch.tensor(x_test.astype(np.float32))
    y_test = torch.tensor(y_test.astype(np.float32))
    
    return x_train, x_test, y_train, y_test

def to_batches(x, y, batch_size, shuffle=True):

    if shuffle:

        indices = np.random.permutation(x.shape[0])
        x = x[indices]
        y = y[indices]

        n_batches = x.shape[0] // batch_size

        x = x[:n_batches * batch_size].reshape(n_batches, batch_size, x.shape[1])
        y = y[:n_batches * batch_size].reshape(n_batches, batch_size, 1)

    return x, y

# def newdata(path, batch_size, shuffle=True):

#     data = pd.read_csv(path)

#     data.drop(["date"], axis=1).values
#     data1 = data.loc[data['month']]
#     data2 = pd.concat[data, data1]
#     #data['month1'] = data.apply(lambda row: row['month'], axis=1)
#     x = data2.values
#     test_data = testdata(x, batch_size, shuffle)

#     #x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random_state=0)

#     #transformer = preprocessing.PowerTransformer()
#     # y_train = transformer.fit_transform(y_train.reshape(-1,1))
#     # y_test = transformer.transform(y_test.reshape(-1,1))
    
#     # y_train = y_train.reshape(y_train.shape[0]//batch_size,batch_size,1)
#     # y_test = y_test.reshape(y_test.shape[0]//batch_size,batch_size,1)

#     x_train = torch.tensor(test_data.astype(np.float32))
#     # y_train = torch.tensor(y_train.astype(np.float32))

#     # x_test = torch.tensor(x_test.astype(np.float32))
#     # y_test = torch.tensor(y_test.astype(np.float32))
    
#     return x_train

# def testdata(x, batch_size, shuffle=True):

#     if shuffle:

#         indices = np.random.permutation(len(x))
#         x = x[indices]

#         n_batches = len(x) // batch_size

#         x = x[:n_batches * batch_size].reshape(n_batches, batch_size, x.shape[1])

#     return x


#x_train, x_test, y_train, y_test =  load_data("final_train.csv", batch_size=100, shuffle = True)

#print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)