import os
import io
import pytest
from fastapi.testclient import TestClient
from app import app, output_dir

client = TestClient(app)

def test_ping():
    resp = client.get("/api/ping")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

def test_infer_audio_no_file():
    resp = client.post("/api/infer_audio/", data={"text_input": "test"})
    assert resp.status_code == 400
    assert "error" in resp.json()

def test_infer_audio_with_dummy_wav(tmp_path):
    # Crea un archivo wav de 1 segundo de silencio (24kHz, mono)
    import numpy as np
    import soundfile as sf
    wav = np.zeros(24000, dtype=np.float32)
    wav_bytes = io.BytesIO()
    sf.write(wav_bytes, wav, 24000, format='WAV')
    wav_bytes.seek(0)
    files = {"upload_audio_prompt": ("dummy.wav", wav_bytes, "audio/wav")}
    data = {"text_input": "pytest"}
    resp = client.post("/api/infer_audio/", data=data, files=files)
    assert resp.status_code == 200
    j = resp.json()
    assert "audio_url" in j
    assert "audio_data" in j
    # Verifica que el archivo se haya creado
    filename = j["audio_url"].split("/")[-1]
    filepath = os.path.join(output_dir, filename)
    assert os.path.exists(filepath)
    # Limpia el archivo generado
    os.remove(filepath)

def test_audio_access_endpoint(tmp_path):
    # Crea un archivo de prueba en output_dir
    testfile = os.path.join(output_dir, "testfile.wav")
    with open(testfile, "wb") as f:
        f.write(b"dummy")
    resp = client.get(f"/api/test-audio-access/testfile.wav")
    assert resp.status_code == 200
    assert resp.json()["status"] == "exists"
    os.remove(testfile)

def test_audio_access_not_found():
    resp = client.get("/api/test-audio-access/nonexistent.wav")
    assert resp.status_code == 200
    assert resp.json()["status"] == "not_found"
