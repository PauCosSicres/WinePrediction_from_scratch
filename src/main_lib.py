from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from data_prep import data_prep

X_train, y_train, X_test, y_test = data_prep('data/processed/clean_wine.csv')

model = LogisticRegression(max_iter= 500)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_train= model.predict(X_train)
acc_train = accuracy_score(y_train, y_pred_train)
probs_train = model.predict_proba(X_train)[:, 1]
loss_train = log_loss(y_train, probs_train)
accuracy = accuracy_score(y_test, y_pred)
probs = model.predict_proba(X_test)[:, 1]
loss = log_loss(y_test, probs)

print(f'Train - Loss: {loss_train:.4f}, Accuracy: {acc_train:.4f}')
print(f'Test  - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')