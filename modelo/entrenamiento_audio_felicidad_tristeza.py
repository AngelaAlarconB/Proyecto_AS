import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

X = np.load("features/features_felicidad_tristeza.npy")
y_encoded = np.load("labels/labels_felicidad_tristeza.npy")

#si queremos entrenar el modelo con los audios en español se comenta lo anterior y se usan las siguientes rutas

#X = np.load("features/features_felicidad_tristeza_espanol.npy")
#y_encoded = np.load("labels/labels_felicidad_tristeza_espanol.npy")

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
modelo_guardado = "modelos/modelo_random_forest_felicidad_tristeza.pkl"

#si queremos entrenar el modelo con los audios en español se comenta lo anterior y se usa la siguiente ruta de guardado

#modelo_guardado = "modelos/modelo_random_forest_felicidad_tristeza_espanol.pkl"
joblib.dump(classifier, modelo_guardado)