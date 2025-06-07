# filepath: d:\Programación\VALL-E-X\macros.py

# Configuración del modelo VALL-E-X optimizada para inglés únicamente
NUM_LAYERS = 12
NUM_HEAD = 16
N_DIM = 1024
NUM_QUANTIZERS = 8
SAMPLE_RATE = 24000

# Configuración de idioma simplificada para inglés
LANGUAGE = "en"
LANGUAGE_TOKEN = "[EN]"

# Configuración del modelo
NUM_TEXT_TOKENS = 2048
NUM_AUDIO_TOKENS = 1024
NUM_MEL_BINS = 100

# Imports simplificados de sklearn
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split