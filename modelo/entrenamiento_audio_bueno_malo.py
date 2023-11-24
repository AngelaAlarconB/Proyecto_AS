import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


X = np.load("features/features.npy")
X = X[:, list(range(20)) + [55, 56, 57]]
y_encoded = np.load("labels/2labels.npy")

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Inicializar el clasificador RandomForest
classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Entrenar el modelo
classifier.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
predictions = classifier.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, predictions)
print("Precisión del modelo:", accuracy)

# Guardar el modelo entrenado en un archivo
modelo_guardado = "modelos/modelo_random_forest_2clases.pkl"
joblib.dump(classifier, modelo_guardado)