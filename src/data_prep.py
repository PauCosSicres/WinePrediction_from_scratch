import numpy as np
import pandas as pd

def data_prep(path):
    df = pd.read_csv(path)
    data = np.array(df)
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]

    train_size = int(0.7 * data.shape[0])
    train_data = data[:train_size]
    test_data = data[train_size:]

    X_train = train_data[:, :-1] # all rows & cols except label (last one)
    y_train = train_data[:, -1] # last row
    X_test = test_data[:, :-1]
    y_test= test_data[:, -1]

    return X_train, y_train, X_test, y_test
