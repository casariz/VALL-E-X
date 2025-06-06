# coding: utf-8
import sys
import os
import platform
import multiprocessing
import torch
import nltk
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import PurePath

print(f"default encoding is {sys.getdefaultencoding()},file system encoding is {sys.getfilesystemencoding()}")
print(f"You are using Python version {platform.python_version()}")
if sys.version_info[0] < 3 or sys.version_info[1] < 7:
    print("Python version should be at least 3.7")
    sys.exit(1)

# Configuración de entorno
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Obtener número de CPUs y configurar PyTorch
thread_count = multiprocessing.cpu_count()
torch.set_num_threads(thread_count)
torch.set_num_interop_threads(thread_count)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
torch._C._set_graph_executor_optimize(False)

print("Use", thread_count, "cpu cores for computing")

# Configurar rutas para nltk
nltk.data.path = nltk.data.path + [os.path.join(os.getcwd(), "nltk_data")]

# Imports específicos de VALL-E-X
from data.tokenizer import AudioTokenizer, tokenize_audio
from data.collation import get_text_token_collater
from models.vallex import VALLE
from utils.g2p import PhonemeBpeTokenizer
from macros import *

# Inicializar tokenizers
text_tokenizer = PhonemeBpeTokenizer(tokenizer_path="./utils/g2p/bpe_69.json")
text_collater = get_text_token_collater()

# Configuración del dispositivo
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available, using GPU")
else:
    print("CUDA not available, using CPU")

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS is available, using Apple Silicon GPU")

# Verificar y cargar el modelo VALL-E-X
if not os.path.exists("./checkpoints/"): 
    os.makedirs("./checkpoints/")
    
if not os.path.exists(os.path.join("./checkpoints/", "vallex-checkpoint.pt")):
    print("Downloading VALL-E-X checkpoint...")
    # Código para descargar el modelo

# Función simplificada para transcripción (solo inglés)
def transcribe_one(model, audio_path):
    """
    Transcribe audio file using Whisper - English only
    """
    # ...código de transcripción...

# Inicializar el modelo VALL-E-X
model = VALLE(
    N_DIM,
    NUM_HEAD,
    NUM_LAYERS,
    norm_first=True,
    add_prenet=False,
    prefix_mode=PREFIX_MODE,
    share_embedding=True,
    nar_scale_factor=1.0,
    prepend_bos=True,
    num_quantizers=NUM_QUANTIZERS,
)

# === AGREGAR PARCHES PARA SISTEMAS UNIX ===
import pathlib
import sys

# Monkey-patch para resolver error de WindowsPath en Linux/Mac
if sys.platform != "win32":
    pathlib.WindowsPath = pathlib.PosixPath
# =========================================

checkpoint_path = PurePath("./checkpoints/vallex-checkpoint.pt")
checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
missing_keys, unexpected_keys = model.load_state_dict(
    checkpoint["model"], strict=True
)
assert not missing_keys
model.eval()

# Inicializar el tokenizador de audio
audio_tokenizer = AudioTokenizer(device)

# Cargar vocos para decodificación
from vocos import Vocos
vocos = Vocos.from_pretrained('charactr/vocos-encodec-24khz').to(device)

# Cargar modelo Whisper para transcripción
if not os.path.exists("./whisper/"): os.mkdir("./whisper/")
try:
    whisper_model = whisper.load_model("medium", download_root=os.path.join(os.getcwd(), "whisper")).cpu()
except Exception as e:
    logging.info(e)
    raise Exception(
        "\n Whisper download failed or damaged, please go to "
        "'https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt'"
        "\n manually download model and put it to {} .".format(os.getcwd() + "\whisper"))

# Inicializar FastAPI
app = FastAPI()

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://pronunciapp.me", "https://www.pronunciapp.me", "http://localhost:4200"],  # Include your production URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Endpoint simple para prueba de conexión y CORS
@app.get("/api/ping")
async def ping():
    return {"status": "ok", "message": "API is running and CORS is configured correctly"}

# Directorio para archivos de salida - usa una ruta absoluta
output_dir = os.path.abspath("output")
os.makedirs(output_dir, exist_ok=True)
print(f"Directorio de salida: {output_dir}")  # Para depuración

# Montar directorio de salida para servir archivos estáticos
app.mount("/audio", StaticFiles(directory=output_dir), name="audio")

# Función para transcribir audio con Whisper
def transcribe_one(model, audio_path):
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")
    lang = max(probs, key=probs.get)
    options = whisper.DecodingOptions(temperature=1.0, best_of=5, fp16=False if device == torch.device("cpu") else True, sample_len=150)
    result = whisper.decode(model, mel, options)
    print(result.text)
    text_pr = result.text
    if text_pr.strip(" ")[-1] not in "?!.,。，？！。、":
        text_pr += "."
    return lang, text_pr

def make_prompt(name, wav, sr, save=True):
    global whisper_model
    whisper_model.to(device)
    if not isinstance(wav, torch.FloatTensor):
        wav = torch.tensor(wav)
    if wav.abs().max() > 1:
        wav /= wav.abs().max()
    if wav.size(-1) == 2:
        wav = wav.mean(-1, keepdim=False)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    assert wav.ndim and wav.size(0) == 1
    torchaudio.save(f"./prompts/{name}.wav", wav, sr)
    lang, text = transcribe_one(whisper_model, f"prompts/{name}.wav")
    lang_warning = None  # <-- Añadido para advertencia
    if lang not in lang2token:
        print(f"Idioma detectado no soportado: {lang}, usando 'en' por defecto.")
        lang = "en"
    lang_token = lang2token[lang]
    text = lang_token + text + lang_token
    with open(f"./prompts/{name}.txt", 'w', encoding='utf-8') as f:
        f.write(text)
    if not save:
        os.remove(f"./prompts/{name}.wav")
        os.remove(f"./prompts/{name}.txt")

    whisper_model.cpu()
    torch.cuda.empty_cache()
    return text, lang, lang_warning 

# Función principal de inferencia desde audio
@torch.no_grad()
def infer_from_audio(text, language, accent, audio_prompt, record_audio_prompt, transcript_content):
    global model, text_collater, text_tokenizer, audio_tokenizer
    audio_prompt = audio_prompt if audio_prompt is not None else record_audio_prompt
    sr, wav_pr = audio_prompt
    if not isinstance(wav_pr, torch.FloatTensor):
        wav_pr = torch.FloatTensor(wav_pr)
    if wav_pr.abs().max() > 1:
        wav_pr /= wav_pr.abs().max()
    if wav_pr.size(-1) == 2:
        wav_pr = wav_pr[:, 0]
    if wav_pr.ndim == 1:
        wav_pr = wav_pr.unsqueeze(0)
    assert wav_pr.ndim and wav.size(0) == 1

    lang_warning = None  # <-- Añadido para advertencia

    if transcript_content == "":
        text_pr, lang_pr, lang_warning = make_prompt('dummy', wav_pr, sr, save=False)
    else:
        lang_pr = langid.classify(str(transcript_content))[0]
        lang_token = lang2token[lang_pr]
        text_pr = f"{lang_token}{str(transcript_content)}{lang_token}"

    if language == 'auto-detect':
        lang_token = lang2token[langid.classify(text)[0]]
    else:
        lang_token = langdropdown2token[language]
    lang = token2lang[lang_token]
    text = lang_token + text + lang_token

    # onload model
    model.to(device)

    # tokenize audio
    encoded_frames = tokenize_audio(audio_tokenizer, (wav_pr, sr))
    audio_prompts = encoded_frames[0][0].transpose(2, 1).to(device)

    # tokenize text
    logging.info(f"synthesize text: {text}")
    phone_tokens, langs = text_tokenizer.tokenize(text=f"_{text_pr}".strip())
    text_tokens, text_tokens_lens = text_collater(
        [
            phone_tokens
        ]
    )

    enroll_x_lens = None
    if text_pr:
        text_prompts, _ = text_tokenizer.tokenize(text=f"{text_pr}".strip())
        text_prompts, enroll_x_lens = text_collater(
            [
                text_prompts
            ]
        )
    text_tokens = torch.cat([text_prompts, text_tokens], dim=-1)
    text_tokens_lens += enroll_x_lens
    lang = lang if accent == "no-accent" else token2lang[langdropdown2token[accent]]
    encoded_frames = model.inference(
        text_tokens.to(device),
        text_tokens_lens.to(device),
        audio_prompts,
        enroll_x_lens=enroll_x_lens,
        top_k=-100,
        temperature=1,
        prompt_language=lang_pr,
        text_language=langs if accent == "no-accent" else lang,
        best_of=5,
    )
    # Decode with Vocos
    frames = encoded_frames.permute(2,0,1)
    features = vocos.codes_to_features(frames)
    samples = vocos.decode(features, bandwidth_id=torch.tensor([2], device=device))

    # offload model
    model.to('cpu')
    torch.cuda.empty_cache()

    message = f"text prompt: {text_pr}\nsythesized text: {text}"
    return message, (24000, samples.squeeze(0).cpu().numpy()), lang_warning  # <-- Devuelve advertencia

@app.post("/api/infer_audio/")
async def infer_audio_endpoint(
    text_input: str = Form(...), # Added text_input as a required form field
    upload_audio_prompt: UploadFile = File(None)
):
    try:
        language = "English"
        accent = "no-accent"
        transcript_content = ""
        
        if upload_audio_prompt is None:
            return JSONResponse(content={"error": "No se proporcionó ningún archivo de audio."}, status_code=400)
        
        # Guardar el archivo subido en un archivo temporal
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            content = await upload_audio_prompt.read()
            tmp.write(content)
            tmp.flush()
            temp_path = tmp.name
        logging.info(f"Se creó archivo temporal de entrada: {temp_path}")
        
        # Leer el archivo de audio con soundfile
        wav, sr = sf.read(temp_path, dtype="float32")
        if wav.ndim == 2 and wav.shape[1] > 1:
            wav = wav[:, 0]
        audio_tuple = (sr, wav)
        
        # Eliminar archivo temporal
        os.remove(temp_path)
        logging.info("Archivo temporal de entrada eliminado.")
        
        # Llamar a la función de inferencia
        text_output, audio_out_tuple, lang_warning = infer_from_audio(
            "",  # Texto vacío - el modelo generará una respuesta basada en el audio
            language,
            accent,
            audio_tuple,
            None,
            transcript_content
        )
        out_sr, out_audio = audio_out_tuple
        
        # Guardar el audio generado - usa una ruta absoluta explícita
        timestamp = int(time.time())
        # Sanitize text_input for filename (optional but recommended)
        sanitized_text_input = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in text_input).rstrip()
        if not sanitized_text_input: # Handle empty or fully sanitized string
            sanitized_text_input = "audio"
        output_filename = f"{sanitized_text_input}_{timestamp}.wav"
        output_filepath = os.path.join(output_dir, output_filename)
        
        # Imprime la ruta para depuración
        print(f"Guardando audio en: {output_filepath}")
        
        sf.write(output_filepath, out_audio, out_sr)
        
         # Verifica que el archivo se haya creado correctamente
        if os.path.exists(output_filepath):
            print(f"✅ Archivo creado correctamente: {output_filepath}")
            print(f"Tamaño del archivo: {os.path.getsize(output_filepath)} bytes")
        else:
            print(f"❌ ERROR: El archivo no se creó: {output_filepath}")
        
        # Convertir el audio a base64 para enviarlo directamente en la respuesta
        import base64
        import io
        
        # Crear un buffer en memoria para guardar el audio en formato WAV
        buffer = io.BytesIO()
        sf.write(buffer, out_audio, out_sr, format='WAV')
        buffer.seek(0)  # Regresar al inicio del buffer
        
        # Codificar el contenido del buffer a base64
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        # Devuelve URL absoluta (con dominio) para el audio
        base_url = "https://pronunciapp.me"
        audio_url = f"{base_url}/audio/{output_filename}"
        
        response_content = {
            "text_output": text_output,
            "audio_url": audio_url,  # URL absoluta
            "audio_data": audio_base64
        }
        if lang_warning:
            response_content["warning"] = lang_warning  # <-- Añade advertencia si existe
        
        return JSONResponse(content=response_content)
        
    except Exception as e:
        logging.exception("Error durante la inferencia de audio:")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/test-audio-access/{filename}")
async def test_audio_access(filename: str):
    """Endpoint de prueba para verificar el acceso a archivos de audio"""
    filepath = os.path.join(output_dir, filename)
    if os.path.exists(filepath):
        return {"status": "exists", "path": filepath, "size": os.path.getsize(filepath)}
    else:
        return {"status": "not_found", "path": filepath}
# # Punto de entrada para ejecutar la aplicación
# if __name__ == "__main__":
#     formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
#     logging.basicConfig(format=formatter, level=logging.INFO)
    
#     # Ejecutar la aplicación con uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)
