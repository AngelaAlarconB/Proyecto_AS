import numpy as np
import os
import speech_recognition as sr
from sklearn.preprocessing import LabelEncoder

# Ruta de la carpeta que contiene las subcarpetas con los archivos de audio
carpeta_audio = "audio_files"

# Obtener la lista de subcarpetas (emociones)
#emociones = os.listdir(carpeta_audio) #si queremos entrenar el modelo con todas las emociones
emociones = ["felicidad", "tristeza"]

# Inicializar el reconocedor de voz
recognizer = sr.Recognizer()

transcripciones = []
labels = []

for emocion in emociones:
    archivos_emocion = os.listdir(os.path.join(carpeta_audio, emocion))
    for file_name in archivos_emocion:
        ruta_original = os.path.join(carpeta_audio, emocion, file_name)

        try:
            with sr.AudioFile(ruta_original) as source:
                audio_data = recognizer.record(source)
                transcription = recognizer.recognize_google(audio_data, language="es-ES")

            transcripciones.append(transcription)
            labels.append(emocion)

        except Exception as e:
            print(f"Error al procesar el archivo {ruta_original}: {e}")


X_text = np.array(transcripciones)
y_text = np.array(labels)

# Codificar las etiquetas en números
label_encoder_text = LabelEncoder()
y_encoded_text = label_encoder_text.fit_transform(y_text)

#np.save("features/features_text.npy", X_text)
#np.save("labels/labels_text.npy", y_encoded_text)

#si queremos extraer todas las emociones se comenta lo siguiente y se usan las rutas anteriores de guardado

np.save("features/features_text_felicidad_tristeza.npy", X_text)
np.save("labels/labels_text_felicidad_tristeza.npy", y_encoded_text)