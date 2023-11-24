import os
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder

#Para ejecutar este codigo es necesario disponer de la carpeta Emotions
carpeta_audio = "Emotions"

# Obtener la lista de subdirectorios (emociones)
emociones = os.listdir(carpeta_audio)

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
            mfccs = librosa.feature.mfcc(y=y_filtered, sr=sample_rate, n_mfcc=55) 

            # Concatenar las nuevas características a los MFCCs
            combined_features = np.concatenate([np.mean(mfccs, axis=1), np.mean(energy, axis=1), np.mean(zero_crossings, axis=1), np.mean(centroid, axis=1), pitch_period])

            features.append(combined_features)
            labels.append(emocion)

        except Exception as e:
            print(f"Error al procesar el archivo {ruta_original}: {e}")

X = np.array(features)
y = np.array(labels)

# Codificar las etiquetas en números
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

np.save("features/features.npy", X)
np.save("labels/labels.npy", y_encoded)