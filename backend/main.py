import io   
import os
import uvicorn
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path
from fastapi import File, UploadFile, FastAPI
from fastapi.middleware.cors import CORSMiddleware


# membuat instance FastAPI
app = FastAPI()

# izinkan request dari Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# PERBAIKAN: Cari model di beberapa lokasi yang mungkin
BASE_DIR = Path(__file__).resolve().parent
POSSIBLE_MODEL_PATHS = [
    # BASE_DIR.parent / 'model' / 'best_transfer.h5',  # Path original
    BASE_DIR / 'model' / 'best_transfer.h5',         # Model di subfolder
    BASE_DIR / 'best_transfer.h5',                   # Model di folder yang sama
    BASE_DIR / 'best_fine_tuned.h5',                 # Model fine-tuned
    BASE_DIR / 'final_model.h5',                     # Model final
]

# Cek semua kemungkinan lokasi model
model = None
for model_path in POSSIBLE_MODEL_PATHS:
    if os.path.exists(model_path):
        print(f"Model ditemukan di: {model_path}")
        print(tf.__version__)
        model = tf.keras.models.load_model(str(model_path))
        break

if model is None:
    # Jika masih belum menemukan model, coba cari di semua subfolder
    print("Mencari model di semua subfolder...")
    model_files = list(BASE_DIR.glob('**/best*.h5')) + list(BASE_DIR.glob('**/final*.h5'))
    if model_files:
        print(f"Model ditemukan di: {model_files[0]}")
        model = tf.keras.models.load_model(str(model_files[0]))
    else:
        raise FileNotFoundError(f"Model tidak ditemukan! Pastikan file model ada di salah satu lokasi berikut: {POSSIBLE_MODEL_PATHS}")

# mapping indeks : label
labels = ["paper", "rock", "scissors"]
UNKNOWN_LABEL = "unknown"

# threshold confidence minimum
THRESHOLD = 0.6

def preprocess_pipeline(image: Image.Image, IMG_SIZE = (224, 224)) -> np.ndarray:
    """
    Fungsi untuk melakukan preprocessing pada gambar input.
    - Melakukan resize gambar ke IMG_SIZE.
    - Mengubah gambar menjadi array bertipe float32.
    - Melakukan rescaling pixel dari [0,255] ke [0,1].
    """
    # Resize gambar ke ukuran yang sesuai
    image_resized = image.resize(IMG_SIZE)
    
    # Konversi gambar ke array numpy
    arr = np.array(image_resized)
    
    # Ubah tipe data menjadi float32 dan lakukan normalisasi [0,255] ke [0,1]
    arr = arr.astype(np.float32) / 255.0
    
    return arr  # mengembalikan array hasil preprocessing

# endpoint untuk menerima input dan menghasilkan prediksi
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Preprocess gambar
    x = preprocess_pipeline(image)
    
    # Tambahkan dimensi batch (model memerlukan input 4D)
    x = np.expand_dims(x, axis=0)
    
    # Lakukan prediksi
    predictions = model.predict(x, verbose=0)
    
    # Dapatkan indeks kelas dengan probabilitas tertinggi
    best_index = int(np.argmax(predictions[0]))
    
    # Dapatkan nilai probabilitas tertinggi (confidence)
    confidence = float(predictions[0][best_index])
    
    # Tentukan label berdasarkan confidence
    if confidence < THRESHOLD:
        label = UNKNOWN_LABEL
    else:
        label = labels[best_index]
    
    # Kembalikan hasil prediksi
    return {
        "label": label, 
        "confidence": confidence,
        "probabilities": [float(p) for p in predictions[0]]  # Mengembalikan semua probabilitas
    }

# Endpoint tambahan untuk health check
@app.get("/")
def health_check():
    return {"status": "ok", "model_loaded": True, "model_summary": str(model.output_shape)}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)