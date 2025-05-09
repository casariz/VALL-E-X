import os
import requests
from pathlib import Path
from tqdm import tqdm

def download_file(url, destination):
    """
    Descarga un archivo desde una URL al destino especificado con una barra de progreso
    """
    print(f"Descargando desde {url} a {destination}")
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # Iniciar la descarga
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Verificar si la descarga fue exitosa
    
    # Obtener el tamaño total del archivo
    total_size = int(response.headers.get('content-length', 0))
    
    # Configurar barra de progreso
    progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)
    
    # Descargar el archivo en bloques
    with open(destination, 'wb') as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
                progress_bar.update(len(chunk))
    
    progress_bar.close()
    print(f"Descarga completada: {destination}")

def main():
    # URLs de los modelos
    vallex_url = "https://huggingface.co/Plachta/VALL-E-X/resolve/main/vallex-checkpoint.pt"
    whisper_url = "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt"
    
    # Rutas para los archivos
    current_dir = Path.cwd()
    vallex_dir = current_dir / "checkpoints"
    whisper_dir = current_dir / "whisper"
    
    vallex_path = vallex_dir / "vallex-checkpoint.pt"
    whisper_path = whisper_dir / "medium.pt"
    
    # Paso 1: Verificar y crear carpeta checkpoints
    if not vallex_dir.exists():
        print("Creando carpeta 'checkpoints'...")
        vallex_dir.mkdir(exist_ok=True)
    else:
        print("La carpeta 'checkpoints' ya existe.")
    
    # Paso 2: Verificar y descargar modelo vallex-checkpoint.pt
    if not vallex_path.exists():
        print("El archivo 'vallex-checkpoint.pt' no existe. Descargando...")
        download_file(vallex_url, vallex_path)
    else:
        print("El archivo 'vallex-checkpoint.pt' ya existe.")
    
    # Paso 3: Verificar y crear carpeta whisper
    if not whisper_dir.exists():
        print("Creando carpeta 'whisper'...")
        whisper_dir.mkdir(exist_ok=True)
    else:
        print("La carpeta 'whisper' ya existe.")
    
    # Paso 4: Verificar y descargar modelo medium.pt
    if not whisper_path.exists():
        print("El archivo 'medium.pt' no existe. Descargando...")
        download_file(whisper_url, whisper_path)
    else:
        print("El archivo 'medium.pt' ya existe.")
    
    print("\nTodos los modelos han sido verificados y descargados correctamente.")

if __name__ == "__main__":
    main()
