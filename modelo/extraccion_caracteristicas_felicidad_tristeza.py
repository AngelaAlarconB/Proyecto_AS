import os
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder

#Para ejecutar este codigo es necesario disponer de la carpeta Emotions o en su defecto, usamos los audios en español de la carpeta audio_files
carpeta_audio = "Emotions"
#carpeta_audio = "audio_files"

# Obtener la lista de subdirectorios (emociones)
emociones = ['Happy', 'Sad']
#emociones = ['felicidad', 'tristeza'] #si nuestra carpeta_audio = "audio_files"

features = []
labels = []

for emocion in emociones:

    archivos_emocion = os.listdir(os.path.join(carpeta_audio, emocion))
    for file_name in archivos_emocion:
        ruta_original = os.path.join(carpeta_audio, emocion, file_name)

        try:
            audio_data, sample_rate = librosa.load(ruta_original)
            y = librosa.util.normalize(audio_data)

            y_filtered = librosa.effects.preemphasis(y)

            ventana_10ms = int(0.01 * sample_rate)
        
            # Calcular short time energy
            energy = librosa.feature.rms(y=y_filtered, frame_length=ventana_10ms)
            
            # Calcular short time zero-crossing rate
            zero_crossings = librosa.feature.zero_crossing_rate(y=y_filtered, frame_length=ventana_10ms)

            # Calcular pitch period
            #pitches, magnitudes = librosa.core.piptrack(y=y_filtered) 
            pitch_period = np.nanmedian(librosa.core.estimate_tuning(y=y_filtered, n_fft = ventana_10ms))

            # Añadir una nueva dimensión a pitch_period
            pitch_period = np.expand_dims(pitch_period, axis=0)

            centroid = librosa.feature.spectral_centroid(y=y, sr=sample_rate, n_fft=ventana_10ms)

            # Extraer características MFCCs
            mfccs = librosa.feature.mfcc(y=y_filtered, sr=sample_rate, n_mfcc=20) 
            combined_features = np.concatenate([np.mean(mfccs, axis=1), np.mean(energy, axis=1), np.mean(zero_crossings, axis=1), np.mean(centroid, axis=1), pitch_period])

            features.append(combined_features)
            labels.append(emocion)

        except Exception as e:
            print(f"Error al procesar el archivo {ruta_original}: {e}")

# Convertir las listas de características y etiquetas a matrices numpy
X = np.array(features)
y = np.array(labels)

# Codificar las nuevas etiquetas en números
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

np.save("features/features_felicidad_tristeza.npy", X)
np.save("labels/labels_felicidad_tristeza.npy", y_encoded)

#si estamos usando carpeta_audio = "audio_files" se comenta lo anterior y se usan las siguientes rutas de guardado

#np.save("features/features_felicidad_tristeza_espanol.npy", X)
#np.save("labels/labels_felicidad_tristeza_espanol.npy", y_encoded)