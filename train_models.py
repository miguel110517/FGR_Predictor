import pandas as pd
import numpy as np
import os
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from fcmeans import FCM

# Crear carpeta de modelos
os.makedirs("models", exist_ok=True)

# Cargar dataset
df = pd.read_excel("FGR_dataset.xlsx")
X = df[['C' + str(i) for i in range(1, 31)]]
y = df['C31']

# Normalizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División inicial
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


print("\n🔍 Entrenando Regresión Logística...")
log_grid = {
    'C': [0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs']
}
log_model = GridSearchCV(LogisticRegression(max_iter=1000), log_grid, cv=5)
log_model.fit(X_train, y_train)
log_best = log_model.best_estimator_
pickle.dump(log_best, open("models/logistic_model.pkl", "wb"))
print("Mejores parámetros:", log_model.best_params_)


print("\n🔍 Entrenando Red Neuronal (MLP)...")
mlp_grid = {
    'hidden_layer_sizes': [(30, 15), (50, 25), (60, 30)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam'],
    'max_iter': [1000]
}
mlp_model = GridSearchCV(MLPClassifier(), mlp_grid, cv=5)
mlp_model.fit(X_train, y_train)
mlp_best = mlp_model.best_estimator_
pickle.dump(mlp_best, open("models/ann_model.pkl", "wb"))
print("Mejores parámetros:", mlp_model.best_params_)


print("\n🔍 Entrenando SVM...")
svm_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf', 'linear'],
    'gamma': ['scale', 'auto']
}
svm_model = GridSearchCV(SVC(probability=True), svm_grid, cv=5)
svm_model.fit(X_train, y_train)
svm_best = svm_model.best_estimator_
pickle.dump(svm_best, open("models/svm_model.pkl", "wb"))
print("Mejores parámetros:", svm_model.best_params_)


print("\n🔍 Entrenando FCM (Mapa Cognitivo Difuso)...")
fcm = FCM(n_clusters=2)
fcm.fit(X_scaled)
fcm_model = {"centers": fcm.centers, "scaler": scaler}
pickle.dump(fcm_model, open("models/fcm_model.pkl", "wb"))


print("\n📊 Evaluación en conjunto de prueba:")
modelos = {
    "Regresión Logística": log_best,
    "Red Neuronal": mlp_best,
    "SVM": svm_best
}
for nombre, modelo in modelos.items():
    y_pred = modelo.predict(X_test)
    print(f"\n--- {nombre} ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

print("\n✅ Todos los modelos entrenados y guardados con éxito.")
