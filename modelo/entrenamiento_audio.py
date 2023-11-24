import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#import matplotlib.pyplot as plt
import joblib


X = np.load("features/features.npy")
X = X[:, list(range(20)) + [55, 56, 57]]
y_encoded = np.load("labels/labels.npy")

# Dividimos en conjunto de entrenamiento y test
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

modelo_guardado = "modelos/modelo_random_forest.pkl"
joblib.dump(classifier, modelo_guardado)

#importancias_caracteristicas = classifier.feature_importances_
#num_mfccs = 55  
#mfcc_feature_names = [f"mfcc{i}" for i in range(1, num_mfccs + 1)]
#other_features = ["energy", "zero crossing", "centroid", "pitch period"]

#todas_caracteristicas = mfcc_feature_names + other_features

#plt.figure(figsize=(10, 6))
#plt.bar(range(len(importancias_caracteristicas)), importancias_caracteristicas, align="center")
#plt.xticks(range(len(importancias_caracteristicas)), list(todas_caracteristicas), rotation='vertical')
#plt.xlabel("Características")
#plt.ylabel("Importancia Relativa")
#plt.title("Importancia de las Características en el Modelo")
#plt.show()
