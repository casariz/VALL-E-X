# VALL-E-X: Zero-Shot Text-to-Speech Synthesis API (English Only)

VALL-E-X es una implementación de síntesis de voz de alta calidad que puede generar habla personalizada con solo una grabación de 3 segundos de un hablante desconocido como prompt acústico. Esta implementación está optimizada específicamente para **síntesis de texto a voz en inglés**.

## 🎯 Características Principales

- **Zero-shot voice cloning**: Clona cualquier voz con solo 3-10 segundos de audio
- **Síntesis en inglés**: Optimizado específicamente para el idioma inglés
- **API REST con FastAPI**: Interfaz HTTP fácil de usar para integración
- **Transcripción automática**: Usa Whisper para extraer texto del audio prompt
- **Compatibilidad multiplataforma**: Funciona en Windows, Linux y macOS con soporte CPU/CUDA/MPS

## 🏗️ Arquitectura del Proyecto

### 📁 Estructura de Directorios

```
VALL-E-X/
├── app.py                 # API principal con FastAPI
├── test_app.py           # Tests unitarios de la API  
├── macros.py             # Configuraciones y constantes globales
├── requirements.txt      # Dependencias del proyecto
├── data/                 # Módulos de procesamiento de datos
│   ├── tokenizer.py     # Tokenización de audio con EnCodec
│   └── collation.py     # Procesamiento de tokens de texto
├── models/              # Arquitectura del modelo neural
│   └── vallex.py       # Modelo principal VALL-E
├── utils/               # Utilidades del sistema
│   └── g2p/            # Conversión grafema a fonema
│       └── bpe_69.json # Tokenizer BPE para fonemas
├── checkpoints/         # Pesos del modelo pre-entrenado (descarga automática)
├── whisper/            # Modelos Whisper para transcripción (descarga automática)
├── output/             # Archivos de audio generados por la API
├── prompts/            # Prompts de audio temporales para procesamiento
└── nltk_data/          # Datos de NLTK (si es necesario)
```

## 🧠 Componentes Principales

### 1. **API Principal (`app.py`)**

El corazón del sistema es una API REST construida con FastAPI que proporciona:

**Endpoints principales:**
- **`/api/infer_audio/`**: Genera voz sintética a partir de un prompt de audio
- **`/api/ping`**: Verificación de estado de la API  
- **`/api/test-audio-access/{filename}`**: Test de acceso a archivos generados
- **`/audio/*`**: Servir archivos de audio estáticos

**Funcionalidades técnicas:**
- **Configuración automática del dispositivo**: CPU, CUDA o MPS (Apple Silicon)
- **Descarga automática de modelos**: VALL-E-X checkpoint y Whisper medium
- **Transcripción con Whisper**: Extracción automática de texto del audio prompt
- **Tokenización dual**: Audio (EnCodec) y texto (BPE fonético)
- **Síntesis VALL-E**: Generación autoregresiva y no-autoregresiva
- **Decodificación Vocos**: Conversión de tokens a audio de alta calidad (24kHz)
- **Gestión de memoria**: Descarga automática de modelos cuando no se usan
- **Parche multiplataforma**: Solución para PathLib en sistemas Unix

**Configuración CORS:**
```python
allow_origins=["https://pronunciapp.me", "https://www.pronunciapp.me", "http://localhost:4200"]
```

### 2. **Configuración Global (`macros.py`)**

Define las constantes y configuraciones esenciales del sistema:

```python
# Arquitectura del modelo
NUM_LAYERS = 12        # Capas del transformer
NUM_HEAD = 16          # Cabezas de atención
N_DIM = 1024          # Dimensión del modelo
NUM_QUANTIZERS = 8     # Niveles de cuantización
SAMPLE_RATE = 24000    # Frecuencia de muestreo

# Configuración simplificada para inglés únicamente
LANGUAGE = "en"
LANGUAGE_TOKEN = "[EN]"
```

### 3. **Funciones Core del Sistema (`app.py`)**

#### **Transcripción de Audio (`transcribe_one`)**
Utiliza Whisper para transcribir audio y detectar idioma:
```python
def transcribe_one(model, audio_path):
    # Carga y procesa el audio
    # Detecta idioma automáticamente
    # Transcribe el contenido
    # Añade puntuación si es necesario
```

#### **Creación de Prompts (`make_prompt`)**
Procesa archivos de audio para crear prompts de voz:
```python
def make_prompt(name, wav, sr, save=True):
    # Normaliza el audio (mono, rango [-1,1])
    # Transcribe con Whisper
    # Añade tokens de idioma
    # Guarda archivos temporales
```

#### **Inferencia Principal (`infer_from_audio`)**
Función central que realiza la síntesis de voz:
```python
def infer_from_audio(text, language, accent, audio_prompt, record_audio_prompt, transcript_content):
    # Procesa el audio de entrada
    # Tokeniza audio y texto
    # Ejecuta inferencia con modelo VALL-E
    # Decodifica con Vocos
    # Gestiona memoria de GPU
```

### 4. **Tests del Sistema (`test_app.py`)**

El proyecto incluye una suite completa de tests unitarios:

```python
def test_ping():                           # Verifica conectividad de la API
def test_infer_audio_no_file():           # Valida manejo de errores
def test_infer_audio_with_dummy_wav():    # Test completo de síntesis
def test_audio_access_endpoint():         # Verifica acceso a archivos generados
```

**Características de testing:**
- Tests automáticos con archivos de audio sintéticos
- Validación de endpoints y manejo de errores
- Verificación de generación y acceso a archivos
- Limpieza automática de archivos temporales

## 🚀 Instalación y Uso

### Requisitos del Sistema
- Python 3.7+
- PyTorch (con soporte CUDA opcional)
- FFmpeg para procesamiento de audio

### Instalación

1. **Clonar el repositorio:**
```bash
git clone <tu-repositorio>
cd VALL-E-X
```

2. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

3. **Crear directorios necesarios:**
```bash
mkdir -p checkpoints whisper output prompts
```

### Uso de la API

1. **Iniciar el servidor:**
```bash
python app.py
# o usar uvicorn directamente:
uvicorn app:app --host 0.0.0.0 --port 8000
```

2. **Hacer una petición de síntesis:**
```bash
curl -X POST "http://localhost:8000/api/infer_audio/" \
  -F "text_input=Hello, this is a test of voice synthesis" \
  -F "upload_audio_prompt=@your_voice_sample.wav"
```

3. **Respuesta esperada:**
```json
{
  "text_output": "text prompt: [detected_text]\nsynthesized text: [EN]Hello, this is a test[EN]",
  "audio_url": "https://your-domain.com/audio/generated_audio.wav",
  "audio_data": "base64_encoded_audio_data"
}
```

## 🧪 Testing

Ejecutar tests unitarios:

```bash
# Ejecutar todos los tests
pytest test_app.py -v

# Tests específicos
pytest test_app.py::test_ping           # Test de conectividad
pytest test_app.py::test_infer_audio    # Test de síntesis
```

## 🔧 Configuración Avanzada

### Variables de Entorno
```python
# Configuración automática de dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.backends.mps.is_available() else device

# Configuración de hilos
thread_count = multiprocessing.cpu_count()
torch.set_num_threads(thread_count)
```

### Descarga Automática de Modelos
- El modelo VALL-E-X se descarga automáticamente desde HuggingFace
- Whisper se descarga al primer uso
- Los modelos se almacenan en `./checkpoints/` y `./whisper/`

## 📊 Flujo de Procesamiento

1. **Input**: Audio prompt (3-10 segundos)
2. **Transcripción**: Whisper extrae texto del audio prompt
3. **Tokenización**: 
   - Texto → Fonemas → Tokens
   - Audio → EnCodec → Tokens cuantizados
4. **Inferencia**: 
   - Modelo AR genera primer quantizer
   - Modelo NAR genera quantizers restantes
5. **Decodificación**: Vocos convierte tokens a audio
6. **Output**: Audio sintético con la voz del prompt

## 🌐 Soporte de Idioma

Esta implementación está optimizada específicamente para **inglés**:

| Idioma | Código | Token | Estado |
|--------|--------|-------|--------|
| Inglés | `en` | `[EN]` | ✅ Soportado |

### Simplificación para Inglés
- Eliminación de detección automática de idioma
- Tokens de idioma simplificados
- Procesamiento optimizado para fonemas en inglés
- Reducción del tamaño del modelo al eliminar embeddings multilingües

## 📈 Rendimiento y Optimizaciones

- **Gestión de memoria**: Descarga automática de modelos de GPU cuando no se usan
- **Cache KV**: Optimización para generación secuencial en modo AR
- **Paralelización**: Soporte para múltiples trabajadores de datos
- **Compatibilidad de hardware**: CPU, CUDA, y Apple Silicon (MPS)
- **Parches de compatibilidad**: Soporte para sistemas Unix/Linux/macOS

## 🔍 Debugging y Monitoreo

### Logs del Sistema
- Información detallada sobre carga de modelos
- Métricas de rendimiento y uso de memoria

### Endpoints de Diagnóstico
- `/api/ping`: Verificación de estado de la API
- `/api/test-audio-access/{filename}`: Verificación de acceso a archivos

## 📝 Estructura de Datos

### Directorios de Trabajo
- `./checkpoints/`: Modelo VALL-E-X pre-entrenado
- `./whisper/`: Modelos Whisper para transcripción
- `./output/`: Audio generado por la API
- `./prompts/`: Archivos temporales de procesamiento

---

## 🙏 Agradecimientos

Este proyecto es una implementación basada en el excelente trabajo de [VALL-E-X](https://github.com/Plachtaa/VALL-E-X) desarrollado por Plachtaa, el cual a su vez está basado en el trabajo de investigación VALL-E-X de Microsoft Research. Se agradece a ambos equipos por su contribución al desarrollo de tecnologías de síntesis de voz de código abierto.

**Nota**: Este proyecto implementa el paper "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers" con mejoras específicas para producción y soporte optimizado para inglés.

