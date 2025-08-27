
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix, roc_auc_score, roc_curve

from logisticR_scratch import LogisticReg
from data_prep import data_prep

# -----------------------------
# 1. Cargar y preparar datos
# -----------------------------
X_train, y_train, X_test, y_test = data_prep("data/processed/clean_wine.csv")

# Asegurarse de tipo float
X_train, X_test = X_train.astype(float), X_test.astype(float)
y_train, y_test = y_train.astype(float), y_test.astype(float)

# -----------------------------
# 2. Modelo from scratch
# -----------------------------
model_scratch = LogisticReg()
model_scratch.init_params(X_train)
model_scratch.fit(X_train, y_train, lr=0.01, n_iter=200)

y_pred_scratch_proba = model_scratch.sigmoid(X_test @ model_scratch.weights + model_scratch.bias)
y_pred_scratch = (y_pred_scratch_proba >= 0.5).astype(int)

# -----------------------------
# 3. Modelo librería sklearn
# -----------------------------
model_lib = LogisticRegression(max_iter=1000)
model_lib.fit(X_train, y_train)
y_pred_lib_proba = model_lib.predict_proba(X_test)[:, 1]
y_pred_lib = model_lib.predict(X_test)

# -----------------------------
# 4. Calcular métricas
# -----------------------------
def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "log_loss": log_loss(y_true, y_prob),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }

metrics_scratch = compute_metrics(y_test, y_pred_scratch, y_pred_scratch_proba)
metrics_lib = compute_metrics(y_test, y_pred_lib, y_pred_lib_proba)

# -----------------------------
# 5. Imprimir resumen
# -----------------------------
print("\n=== Metrics Scratch ===")
for k, v in metrics_scratch.items():
    if k != "confusion_matrix":
        print(f"{k}: {v:.4f}")
print("Confusion Matrix:\n", metrics_scratch["confusion_matrix"])

print("\n=== Metrics Sklearn ===")
for k, v in metrics_lib.items():
    if k != "confusion_matrix":
        print(f"{k}: {v:.4f}")
print("Confusion Matrix:\n", metrics_lib["confusion_matrix"])

# -----------------------------
# 6. Graficas comparativas
# -----------------------------
# ROC curves
fpr_s, tpr_s, _ = roc_curve(y_test, y_pred_scratch_proba)
fpr_l, tpr_l, _ = roc_curve(y_test, y_pred_lib_proba)

plt.figure(figsize=(8,6))
plt.plot(fpr_s, tpr_s, label="Scratch ROC AUC: %.3f" % metrics_scratch["roc_auc"])
plt.plot(fpr_l, tpr_l, label="Sklearn ROC AUC: %.3f" % metrics_lib["roc_auc"])
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()

# Confusion matrices
fig, axes = plt.subplots(1,2, figsize=(12,5))
sns.heatmap(metrics_scratch["confusion_matrix"], annot=True, fmt="d", ax=axes[0])
axes[0].set_title("Scratch Confusion Matrix")
sns.heatmap(metrics_lib["confusion_matrix"], annot=True, fmt="d", ax=axes[1])
axes[1].set_title("Sklearn Confusion Matrix")
plt.show()

# -----------------------------
# 7. Pesos / coeficientes
# -----------------------------
print("\nWeights Scratch:", model_scratch.weights)
print("Weights Sklearn:", model_lib.coef_)
