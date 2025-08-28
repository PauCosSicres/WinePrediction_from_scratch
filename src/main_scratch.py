import numpy as np
from logisticR_scratch import LogisticReg
from data_prep import data_prep

X_train, y_train, X_test, y_test = data_prep('data/processed/clean_wine.csv')
model = LogisticReg()
model.init_params(X_train)
model.fit(X_train, y_train, 0.01, 500)
y_pred = model.predict(X_test)

accuracy = np.mean(y_pred == y_test)
print(f'Test Data Accuracy: {accuracy}%')
