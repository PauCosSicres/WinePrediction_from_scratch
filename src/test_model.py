import numpy as np
import pandas as pd
from train_model import LogisticReg

data = pd.read_csv('../data/processed/clean_wine.csv')
data = np.array(data)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)

train_size = int(0.7 * len(indices))
train_data = data[:train_size]
test_data = data[train_size:]

X_train = train_data[:, :-1] # all rows & cols except label (last one)
y_train = train_data[:, -1] # last row
X_test = test_data[:, :-1]
y_test= test_data[:, -1]

model = LogisticReg()
model.init_params(X_train)
model.fit(X_train, y_train, 0.01, 100)
y_pred = model.predict(X_test)

accuracy = np.mean(y_pred == y_test)
print(f"Test Accuracy: {accuracy}%")
