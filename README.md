# Proyecto_AS
Proyecto Análisis de Señales : Análisis de Emociones en Audio y texto.

Disponemos de dos carpetas: filtros y modelo.

En la carpeta "modelo" se tiene:
- "audio_files" : carpeta con todos los audios dispuestos por emociones.
- "features" : carpeta con diferentes archivos con las características extraídas en función del modelo que se quiera entrenar.
- "labels" : carpeta con diferentes archivos con las etiquetas correspondientes extraídas en función del modelo que se quiera entrenar.
- "modelos" : carpeta con diferentes archivos correspondientes a los distintos modelos entrenados.
- "vectorizer" : carpeta con diferentes archivos correspondientes a las propiedades de texto en función de los distintos modelos.
- 4 códigos "extraccion_caracteristicas_().py" donde () se corresponde con el modelo para el cual se extraen los datos.
- 4 códigos "entrenamiento_().py" donde () se corresponde con el modelo a entrenar.

Hemos intentado que el código sea lo más reproducible posible, proporcionando todas las herramientas para que no sea necesario extraer las caracterísitcas de todos los audios cada vez que se quiere entrenar el modelo, ya que podría resultar costoso (aunque hemos proporcionado también el código de extracción de características). Por ello, los archivos "entrenamiento_(*).py" son ejecutables y nos proporcionan rápidamente los resultados de accuracy obtenidos para cada modelo.

En caso de querer extraer características, se utilizará la carpeta "audio_files" o en su defecto, los archivos del siguiente repositorio de kaggle https://www.kaggle.com/datasets/uldisvalainis/audio-emotions .
