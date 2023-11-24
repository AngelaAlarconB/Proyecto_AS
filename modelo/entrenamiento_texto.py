import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

#X = np.load("features/features_text.npy")
#y_encoded = np.load("labels/labels_text.npy")

# si queremos entrenar el modelo con todas las emociones se comenta esto y se usan las rutas anteriores
X = np.load("features/features_text_felicidad_tristeza.npy")
y_encoded = np.load("labels/labels_text_felicidad_tristeza.npy")

vectorizer = TfidfVectorizer()
X_text_vectorized = vectorizer.fit_transform(X)

#joblib.dump(vectorizer, "vectorizer/vectorizer_text.pkl")

# si queremos entrenar el modelo con todas las emociones se comenta esto y se usa la ruta anterior
joblib.dump(vectorizer, "vectorizer/vectorizer_text_felicidad_tristeza.pkl")

X_train, X_test, y_train, y_test = train_test_split(X_text_vectorized, y_encoded, test_size=0.2, random_state=42)

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
#modelo_guardado = "modelos/modelo_random_forest_text.pkl"

# si queremos entrenar el modelo con todas las emociones se comenta esto y se usa las ruta anterior para guardarlo
modelo_guardado = "modelos/modelo_random_forest_text_felicidad_tristeza.pkl"

joblib.dump(classifier, modelo_guardado)

