import streamlit as st
import requests
from PIL import Image
import io
import numpy as np

# URL API backend
API_URL = "http://localhost:8000/predict/"

# Konfigurasi halaman
st.set_page_config(
    page_title="Rock-Paper-Scissors Classifier",
    page_icon="✂️",
    layout="centered"
)

# Tampilan header
st.title("🪨📃✂️ Rock-Paper-Scissors Classifier")
st.write("Upload gambar atau gunakan kamera untuk mengklasifikasikan gerakan batu, kertas, atau gunting")

# Menampilkan status koneksi ke API
try:
    health_check = requests.get("http://localhost:8000/")
    if health_check.ok:
        st.success("✅ Terhubung ke API")
    else:
        st.error("❌ Tidak dapat terhubung ke API")
except requests.exceptions.ConnectionError:
    st.error("❌ API server tidak berjalan. Pastikan FastAPI backend aktif di port 8000")

# Pilihan input: kamera atau file upload
input_method = st.radio("Pilih sumber input:", ("Upload Gambar", "Kamera"))

img = None

if input_method == "Upload Gambar":
    uploaded = st.file_uploader("Pilih file gambar", type=["jpg","png","jpeg"])
    if uploaded:
        img = Image.open(uploaded)
        
elif input_method == "Kamera":
    camera_img = st.camera_input("Ambil foto dengan kamera")
    if camera_img:
        img = Image.open(camera_img)

# Jika ada gambar, tampilkan & tombol Predict
if img:
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img, caption="Gambar input", use_column_width=True)
    
    with col2:
        # Informasi tentang preprocessing
        st.info("Preprocessing gambar:\n"
                "- Resize ke (224, 224)\n"
                "- Normalisasi nilai pixel dari [0-255] ke [0-1]")
    
    if st.button("Prediksi", type="primary"):
        with st.spinner("Memproses..."):
            try:
                # Siapkan file untuk dikirim ke backend
                buf = io.BytesIO()
                img.save(buf, format="JPEG")
                buf.seek(0)
                files = {"file": ("image.jpg", buf.getvalue(), "image/jpeg")}
                
                # Kirim request ke API backend
                response = requests.post(API_URL, files=files)
                
                if response.ok:
                    data = response.json()
                    
                    # Tampilkan hasil prediksi
                    if data['label'] == "unknown":
                        st.warning(f"⚠️ Tidak dapat mengenali gambar (confidence: {data['confidence']*100:.1f}%)")
                    else:
                        # Menampilkan emoji sesuai prediksi
                        emoji = "🪨" if data['label'] == "rock" else "📃" if data['label'] == "paper" else "✂️"
                        
                        # Warna untuk level confidence
                        color = "green" if data['confidence'] > 0.85 else "orange"
                        
                        st.markdown(f"### Prediksi: {emoji} **{data['label'].upper()}**")
                        st.markdown(f"<p style='color:{color}'>Confidence: {data['confidence']*100:.1f}%</p>", unsafe_allow_html=True)
                        
                        # Tampilkan distribusi probabilitas
                        if 'probabilities' in data:
                            probs = data['probabilities']
                            st.bar_chart({label: prob for label, prob in zip(["Paper", "Rock", "Scissors"], probs)})
                else:
                    st.error(f"Terjadi kesalahan pada server: {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Tidak dapat terhubung ke API server. Pastikan backend sedang berjalan.")

# Tambahkan informasi bantuan
with st.expander("ℹ️ Bantuan"):
    st.markdown("""
    **Cara menggunakan aplikasi ini:**
    1. Pilih sumber input (Upload Gambar atau Kamera)
    2. Jika memilih Upload, pilih file gambar dari perangkat
    3. Jika memilih Kamera, berikan izin kamera dan ambil foto
    4. Tekan tombol "Prediksi" untuk melihat hasil klasifikasi
    
    **Tips untuk hasil terbaik:**
    - Pastikan gambar menampilkan tangan dengan jelas
    - Gunakan latar belakang kontras dengan tangan
    - Posisikan gesture (batu/kertas/gunting) di tengah frame
    """)

# Tambahkan footer
st.markdown("---")
st.caption("Rock-Paper-Scissors Classifier powered by TensorFlow & MobileNetV2")