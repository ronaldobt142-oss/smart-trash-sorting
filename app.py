import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import io

# 1. KONFIGURASI HALAMAN BROWSER (UI PREMIUM)
st.set_page_config(
    page_title="TrashAI // Advanced Waste Sorting Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. SUNTIKAN CSS PREMIUM (GLASSMORPHISM & NEON GLOW STYLE)
st.markdown("""
    <style>
        /* Import Font Modern */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');
        
        /* Tema Dasar Server */
        .stApp {
            background: radial-gradient(circle at top right, #0F172A, #020617);
            color: #F8FAFC;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }
        
        /* Sembunyikan Header Bawaan */
        header { background-color: transparent !important; }
        .stHeader { background-color: transparent !important; }
        
        /* Hero Title & Badge */
        .hero-title {
            font-size: 2.8rem;
            font-weight: 800;
            background: linear-gradient(135deg, #34D399 0%, #06B6D4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.05em;
            margin-bottom: 5px;
        }
        .hero-tagline {
            color: #94A3B8;
            font-size: 1.1rem;
            margin-bottom: 25px;
        }
        .portfolio-badge {
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: #10B981;
            padding: 4px 12px;
            border-radius: 99s9px;
            font-size: 0.8rem;
            font-weight: 600;
            display: inline-block;
            margin-bottom: 15px;
        }

        /* Kartu Efek Kaca (Glassmorphism Container) */
        div[data-testid="stVerticalBlock"] > div.element-container:has(.glass-card),
        .glass-card {
            background: rgba(30, 41, 59, 0.4) !important;
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 20px;
        }

        /* Kustomisasi Tab Super Elegan */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: rgba(15, 23, 42, 0.6);
            padding: 6px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        .stTabs [data-baseweb="tab"] {
            background-color: transparent !important;
            color: #64748B !important;
            border-radius: 8px !important;
            padding: 8px 16px !important;
            font-weight: 600 !important;
            border: none !important;
            transition: all 0.3s ease;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
            color: #020617 !important;
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
        }

        /* Tombol Upload & Kamera Premium */
        div.stFileUploader button, div.stCameraInput button {
            background: rgba(51, 65, 85, 0.5) !important;
            color: #F1F5F9 !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            transition: all 0.2s ease;
        }
        div.stFileUploader button:hover, div.stCameraInput button:hover {
            background: rgba(71, 85, 105, 0.8) !important;
            border-color: #38BDF8 !important;
        }

        /* TOMBOL DOWNLOAD (HIJAU GLOWING) */
        div.stDownloadButton > button:first-child {
            background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
            color: #020617 !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em;
            border: none !important;
            border-radius: 10px !important;
            padding: 12px 24px !important;
            box-shadow: 0 4px 20px rgba(16, 185, 129, 0.2) !important;
            transition: all 0.3s ease !important;
        }
        div.stDownloadButton > button:first-child:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 24px rgba(16, 185, 129, 0.4) !important;
            color: #020617 !important;
        }

        /* Desain Sidebar */
        .css-163ttbj, [data-testid="stSidebar"] {
            background-color: #0B1329 !important;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        /* Alert Info Box */
        .stAlert {
            background: rgba(15, 23, 42, 0.6) !important;
            border: 1px solid rgba(56, 189, 248, 0.2) !important;
            border-radius: 12px !important;
        }
        .stAlert div, .stAlert p, .stAlert li { color: #E2E8F0 !important; }
    </style>
""", unsafe_allow_html=True)

# 3. SIDEBAR (SPEK PORTFOLIO ENGINE)
with st.sidebar:
    st.markdown("<div style='text-align: center; padding: 10px 0;'><h2 style='color:#10B981; margin:0;'>⚡ TrashAI</h2><p style='color:#64748B; font-size:0.85rem;'>Production v1.2.0</p></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🖥️ Core Architecture")
    st.markdown("• **Model Engine:** `YOLOv8 Nano` (Ultralytics)\n• **Framework:** `Streamlit Cloud` / `OpenCV`\n• **Inference Speed:** `~0.12s / image`\n• **Status:** `Active Operational` 🟢")
    st.markdown("---")
    st.markdown("### 🏷️ Target Classes")
    st.code("📦 Cardboard\n🥛 Glass\n🥫 Metal\n🍎 Organic\n📄 Paper\n🥤 Plastic", language="text")
    st.markdown("---")
    st.markdown("### 👨‍💻 Developer Project")
    st.caption("Aplikasi ini dibuat sebagai bagian dari Portofolio Sistem Komputer Visi Tingkat Lanjut untuk optimalisasi manajemen pemilahan limbah pintar.")

# 4. KONTEN UTAMA (HERO SECTION)
st.markdown("<div class='portfolio-badge'>✨ Production-Ready Portfolio Project</div>", unsafe_allow_html=True)
st.markdown("<h1 class='hero-title'>Smart Trash Sorting Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p class='hero-tagline'>Sistem klasifikasi otomatis objek sampah berbasis Deep Learning untuk kelestarian lingkungan hulu.</p>", unsafe_allow_html=True)

# Memuat Otak AI
@st.cache_resource
def load_model():
    return YOLO('best.pt')

try:
    model = load_model()
except:
    st.error("Model 'best.pt' gagal dimuat secara otomatis dari sistem repository.")

# 5. AREA INTERAKSI UTAMA (DUAL INTERACTIVE INPUT)
st.markdown("### 🛠️ Pengujian Real-Time Engine")

tab1, tab2 = st.tabs(["📁 Upload Image File", "📷 Live Device Camera"])
source_img = None

with tab1:
    uploaded_file = st.file_uploader("Seret atau pilih file gambar dari perangkat kamu:", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        source_img = Image.open(uploaded_file)

with tab2:
    cam_file = st.camera_input("Izinkan akses kamera untuk menangkap objek sampah secara instan:")
    if cam_file is not None:
        source_img = Image.open(cam_file)

st.markdown("<br>", unsafe_allow_html=True)

# 6. PIPELINE PROSES DETEKSI & RENDERING HASIL VISUAL
if source_img is not None:
    img_array = np.array(source_img)
    
    with st.spinner("🔄 Deep Learning Engine sedang melakukan analisis segmentasi..."):
        results = model.predict(img_array)
        res_plotted = results[0].plot()
    
    # Grid Tampilan Output
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='glass-card'><h4 style='color:#38BDF8; margin-top:0;'>📸 Input Visual</h4>", unsafe_allow_html=True)
        st.image(source_img, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='glass-card'><h4 style='color:#4ADE80; margin-top:0;'>🤖 YOLOv8 Bounding-Box Matrix</h4>", unsafe_allow_html=True)
        st.image(res_plotted, use_container_width=True)
        
        # Konversi Gambar ke Bytes untuk Tombol Download
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        res_pil = Image.fromarray(res_rgb)
        buf = io.BytesIO()
        res_pil.save(buf, format="JPEG")
        byte_im = buf.getvalue()
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="📥 Export Result Image (Cloud)",
            data=byte_im,
            file_name="trashai_detection_result.jpg",
            mime="image/jpeg",
            use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
    st.success("🎯 Analisis Berhasil! Objek terpetakan dengan akurat dalam koordinat Bounding Box.")
    st.markdown("---")

# 7. SEKSI EDUKASI & ANALISIS DAMPAK LINGKUNGAN
st.markdown("### ⚠️ Environmental Education Module")
st.write("Modul integrasi peninjauan dampak polusi material terhadap biosfer bumi:")

pilihan_sampah = st.selectbox(
    "Pilih Jenis Material:",
    ["-- Pilih Material Sampah --", "Plastik (Plastic)", "Organik (Organic)", "Kaca (Glass)", "Logam/Kaleng (Metal)", "Kertas & Karton (Paper/Cardboard)"]
)

if pilihan_sampah == "Plastik (Plastic)":
    st.info("""
    **🚨 Dampak Ekologis Material Plastik:**
    * **Resistensi Dekomposisi:** Membutuhkan rentang waktu **100 - 500 tahun** untuk hancur secara struktural.
    * **Krisis Mikroplastik:** Memecah menjadi partikel mikron, mengontaminasi rantai makanan air, memicu risiko karsinogenik.
    * **Degradasi Biota:** Mengancam ratusan spesies fauna laut akibat tertelan material polimer makro.
    """)
elif pilihan_sampah == "Organik (Organic)":
    st.info("""
    **🚨 Dampak Ekologis Sampah Organik:**
    * **Emisi Gas Metana:** Proses dekomposisi anaerobik menghasilkan senyawa **Metana ($CH_4$)**, katalisator pemanasan global.
    * **Instabilitas Area TPA:** Akumulasi volume gas metana di bawah permukaan tanah berpotensi tinggi memicu bahaya ledakan termal.
    * **Vektor Patogen:** Menjadi episentrum proliferasi bakteri berbahaya serta lalat penular penyakit pencernaan.
    """)
elif pilihan_sampah == "Kaca (Glass)":
    st.info("""
    **🚨 Dampak Ekologis Material Kaca:**
    * **Daya Tahan Abadi:** Secara geologis tidak dapat terurai alami di lapisan tanah bumi (memerlukan jutaan tahun).
    * **Ancaman Cedera Fisik:** Pecahan tajam di area alam bebas sangat rawan melukai jaringan fisik makhluk hidup.
    * **Efek Lensa Termal:** Pecahan kaca bening bertindak sebagai lensa cembung alami yang mampu memicu kebakaran hutan.
    """)
elif pilihan_sampah == "Logam/Kaleng (Metal)":
    st.info("""
    **🚨 Dampak Ekologis Sampah Logam & Kaleng:**
    * **Episentrum Vektor Nyamuk:** Rongga kaleng terbuka yang tergenang air hujan memicu perkembangbiakan vektor demam berdarah.
    * **Leaching Logam Berat:** Oksidasi karat zat besi memicu pelarutan unsur logam berat beracun ke dalam akuifer air tanah lokal.
    """)
elif pilihan_sampah == "Kertas & Karton (Paper/Cardboard)":
    st.info("""
    **🚨 Dampak Eksploitasi Kertas:**
    * **Deforestasi Masif:** Peningkatan limbah selulosa berkorelasi langsung dengan laju penebangan pohon hutan primer sebagai bahan baku pulp.
    * **Limbah Toksik Klorin:** Proses klorinasi industri kertas menghasilkan polutan organoklorin berbahaya bagi ekosistem perairan hulu.
    """)
