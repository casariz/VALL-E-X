# Usar la imagen base de Python 3.10 slim
FROM python:3.10-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Establecer el directorio de trabajo
WORKDIR /VALL-E-X

# Copiar e instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto de la aplicación
COPY . .

# Crear directorios para los modelos
RUN mkdir -p ./checkpoints ./whisper

# Descargar el modelo vallex-checkpoint.pt
RUN wget -O ./checkpoints/vallex-checkpoint.pt \
    https://huggingface.co/Plachta/VALL-E-X/resolve/main/vallex-checkpoint.pt

# Descargar el modelo Whisper medium.pt
RUN wget -O ./whisper/medium.pt \
    https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt

# Exponer el puerto y ejecutar la aplicación
EXPOSE 8000

CMD python -m uvicorn app:app --host 0.0.0.0 --port 8000