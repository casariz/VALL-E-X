# Usar la imagen base de Python 3.10 slim
FROM python:3.10-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Establecer el directorio de trabajo
WORKDIR /VALL-E-X

# Copiar e instalar dependencias de Python primero para aprovechar el caché de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto de la aplicación
COPY . .

# Crear usuario no root para seguridad
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Exponer el puerto y ejecutar la aplicación
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
