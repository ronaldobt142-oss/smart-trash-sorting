
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import io

# Konfigurasi Halaman Browser
st.set_page_config(
    page_title="Smart Trash Sorting AI",
    page_icon="♻️",
    layout="wide"
)

# KUSTOMISASI CSS (FIX BAD HEADER PUTIH & FIX KONTRAS WARNA FONT & FIX TOMBOL UNDUH)
st.markdown("""
    <style>
        /* 1. Mengubah background utama dan jenis font */
        .stApp {
            background-color: #1E293B;
            color: #F8FAFC;
            font-family: 'Segoe UI', Roboto, sans-serif;
        }
        
        /* 2. FIX BUG PUTIH DI ATAS */
        header { background-color: transparent !important; }
        .stHeader { background-color: transparent !important; }

        /* 3. Gaya Judul Glowing Hijau Daun */
        h1 { color: #4ADE80 !important; font-weight: 700; }
        h3, h4 { color: #38BDF8 !important; }

        /* 4. FIX KONTRAS FONT EDUKASI */
        .stAlert div, .stAlert p, .stAlert li { color: #FFFFFF !important; }
        .stAlert { background-color: #0F172A !important; border: 1px solid #334155; border-radius: 8px; }

        /* 5. FIX UX TOMBOL UNDUH (PRIMARY BUTTON): Maksa warna background hijau segar & teks putih */
        div.stDownloadButton > button:first-child {
            background-color: #22C55E !important;
            color: #FFFFFF !important;
            font-weight: 600 !important;
            border: none !important;
            border-radius: 8px !important;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1) !important;
        }
        div.stDownloadButton > button:first-child:hover {
            background-color: #166534 !important;
            color: #FFFFFF !important;
            border: none !important;
        }
        
        /* 6. Perbaikan Warna Tombol Upload Biasa */
        div.stFileUploader button {
            background-color: #334155 !important;
            color: #F1F5F9 !important;
            border: 1px solid #475569 !important;
        }
        div.stFileUploader button:hover {
            background-color: #475569 !important;
        }

        /* 7. Perbaikan Teks Petunjuk Upload agar tidak terlalu silau */
        .uploadedFile p, div.stFileUploader > label {
            color: #94A3B8 !important;
        }
    </style>
""", unsafe_allow_html=True)

# SIDEBAR (MENU SAMPING DENGAN LIMITASI SISTEM)
with st.sidebar:
    st.markdown("### 📊 Sistem AI")
    st.markdown("**Engine:** YOLOv8 Nano")
    st.markdown("**Status Server:** Online ✅")
    st.markdown("---")
    st.markdown("### 🏷️ Target Deteksi:")
    st.write("📦 Cardboard | 🥛 Glass | Metal")
    st.write("Organic | 📄 Paper | Plastic")
    st.markdown("---")
    st.markdown("### ⚠️ Keterbatasan Sistem")
    st.caption("Model AI dioptimalkan untuk objek sampah tunggal dengan pencahayaan cukup. Akurasi dapat menurun pada objek bertumpuk atau pencahayaan minim.")

# KONTEN UTAMA (HEADER)
st.title("♻️ Smart Trash Sorting Dashboard")
st.markdown("<p style='font-size:16px; color: #94A3B8;'>Sistem Deteksi, Pemilah, dan Edukasi Dampak Lingkungan Berbasis Komputer Visi</p>", unsafe_allow_html=True)
st.markdown("---")

# Load model AI
@st.cache_resource
def load_model():
    return YOLO('/content/runs/detect/train/weights/best.pt')

try:
    model = load_model()
except:
    st.error("Model 'best.pt' tidak ditemukan. Pastikan training selesai!")

# AREA UPLOAD & PANDUAN
st.markdown("### 📸 Langkah Pengujian")
col_guide, col_upload = st.columns([1, 2])

with col_guide:
    st.write("1. Siapkan foto objek sampah.")
    st.write("2. Format file wajib **JPG, JPEG, atau PNG**.")
    st.write("3. Pastikan pencahayaan cukup terang agar deteksi akurat.")

with col_upload:
    uploaded_file = st.file_uploader("Pilih foto sampah:", type=["jpg", "jpeg", "png"])

st.markdown("---")

# PROSES DETEKSI & DISPLAY SIDE-BY-SIDE
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    with st.spinner("🔄 AI menganalisis gambar..."):
        results = model.predict(img_array)
        res_plotted = results[0].plot()
    
    # Tampilkan Gambar Sebelum vs Sesudah
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h4 style='text-align: center;'>📸 Gambar Asli</h4>", unsafe_allow_html=True)
        st.image(image, use_container_width=True)
    with col2:
        st.markdown("<h4 style='text-align: center;'>🤖 Hasil Deteksi AI</h4>", unsafe_allow_html=True)
        st.image(res_plotted, use_container_width=True)
        
        # PROSES PEMBUATAN TOMBOL DOWNLOAD
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        res_pil = Image.fromarray(res_rgb)
        buf = io.BytesIO()
        res_pil.save(buf, format="JPEG")
        byte_im = buf.getvalue()
        
        st.markdown("<br>", unsafe_allow_html=True)
        # INI DIA TOMBOLNYA: st.download_button bawaan, CSS yang ngerubah gaya
        st.download_button(
            label="📥 Unduh Gambar Hasil Deteksi",
            data=byte_im,
            file_name="hasil_deteksi_sampah.jpg",
            mime="image/jpeg",
            use_container_width=True
        )
        
    st.markdown("---")
    st.success("✅ Analisis Selesai! Pilah sampah berdasarkan hasil visualisasi AI di atas.")
    st.markdown("---")

# FITUR EDUKASI BAHAYA TIAP JENIS SAMPAH
st.markdown("### ⚠️ Edukasi Dampak & Bahaya Sampah")
st.write("Pilih jenis sampah untuk melihat bahayanya bagi kelangsungan makhluk hidup:")

pilihan_sampah = st.selectbox(
    "Pilih Kategori Sampah:",
    ["-- Pilih Sampah --", "Plastik (Plastic)", "Organik (Organic)", "Kaca (Glass)", "Logam/Kaleng (Metal)", "Kertas & Karton (Paper/Cardboard)"]
)

if pilihan_sampah == "Plastik (Plastic)":
    st.info("""
    **🚨 Bahaya Utama Sampah Plastik:**
    * **Sangat Sulit Terurai:** Butuh waktu **100 hingga 500 tahun** agar plastik hancur alami di tanah.
    * **Ancaman Mikroplastik:** Partikel mencemari air, termakan ikan, masuk tubuh manusia, memicu kanker.
    * **Merusak Ekosistem:** Jutaan hewan laut mati setiap tahun karena memakan sampah plastik terapung.
    """)
elif pilihan_sampah == "Organik (Organic)":
    st.info("""
    **🚨 Bahaya Sampah Organik (Jika Menumpuk):**
    * **Gas Metana & Efek Rumah Kaca:** Pembusukan tanpa oksigen menghasilkan **Gas Metana ($CH_4$)**, pemicu global warming.
    * **Ledakan TPA:** Penumpukan gas metana di dalam gunung sampah berisiko memicu ledakan besar.
    * **Sumber Penyakit:** Sarang berkembang biaknya lalat, tikus, bakteri pembawa diare serta kolera.
    """)
elif pilihan_sampah == "Kaca (Glass)":
    st.info("""
    **🚨 Bahaya Sampah Kaca:**
    * **Tidak Bisa Terurai:** Kaca secara alami **tidak akan pernah hancur** di tanah (butuh hingga 1 juta tahun).
    * **Bahaya Fisik:** Pecahan sharp di alam liar sangat rawan melukai fisik hewan dan manusia.
    * **Pemicu Kebakaran:** Serpihan kaca bening bertindak seperti lensa yang memusatkan panas matahari ke rumput kering.
    """)
elif pilihan_sampah == "Logam/Kaleng (Metal)":
    st.info("""
    **🚨 Bahaya Sampah Logam & Kaleng:**
    * **Sarang Nyamuk DBD:** Kaleng terbuka terisi air hujan jadi tempat nyamuk *Aedes aegypti* bertelur.
    * **Pencemaran Logam Berat:** Karat larut ke air tanah, meracuni air sumur warga dengan besi atau timbal.
    """)
elif pilihan_sampah == "Kertas & Karton (Paper/Cardboard)":
    st.info("""
    **🚨 Dampak Sampah Kertas & Karton:**
    * **Eksploitasi Hutan:** Semakin banyak kertas dibuang, semakin banyak pohon hutan ditebang untuk pulp baru.
    * **Limbah Kimia Klorin:** Proses pemutihan kertas menyisakan zat kimia berbahaya bagi biota sungai jika dibuang sembarangan.
    """)
