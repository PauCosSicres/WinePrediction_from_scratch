import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, log_loss, roc_curve, roc_auc_score
from logisticR_scratch import LogisticReg
from data_prep import data_prep

X_train, y_train, X_test, y_test = data_prep('data/processed/clean_wine.csv')

# Scratch Model
scratch = LogisticReg()
scratch.init_params(X_train)
scratch.fit(X_train, y_train, n_iter=1000)
y_pred_scratch= scratch.predict(X_test)
y_prob_scratch = scratch.sigmoid(X_test @ scratch.weights + scratch.bias)

# Library Model
library = LogisticRegression(max_iter=1000)
library.fit(X_train, y_train)
y_pred_lib = library.predict(X_test)
y_prob_lib= library.predict_proba(X_test)[:,1] # take 2n column


# Get Metrics
def compute_metrics(y, y_pred, y_prob):
    accuracy = accuracy_score(y, y_pred)
    loss = log_loss(y, y_prob)
    roc = roc_auc_score(y, y_prob)
    confusionM = confusion_matrix(y, y_pred)

    print(f'Accuracy Score: {accuracy}\n'
          f'Log Loss: {loss}\n'
          f'ROC Score: {roc}\n'
          f'Confusion Matrix:\n{confusionM}\n')


print('|| Model from Scratch Metrics ||\n')
compute_metrics(y_test, y_pred_scratch, y_prob_scratch)

print('|| Model from Library Metrics ||\n')
compute_metrics(y_test, y_pred_lib, y_prob_lib)
