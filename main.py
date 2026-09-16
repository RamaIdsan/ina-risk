import os
import math
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import rasterio
from pyproj import Transformer
from huggingface_hub import snapshot_download

# --- CONFIG HUGGING FACE DATASET ---
# Ganti dengan path dataset Hugging Face kamu
HF_DATASET_REPO = "Aquari5/inarisk-geotiff" 
EXTRACT_DIR = "data_geotiff"

def sync_geotiff_data():
    """Mengunduh dan menyinkronkan data GeoTIFF dari Hugging Face Datasets."""
    print("🔄 Memeriksa pembaruan data GeoTIFF dari Hugging Face...")
    try:
        snapshot_download(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            local_dir=EXTRACT_DIR,
            local_dir_use_symlinks=False
        )
        print("✅ Sinkronisasi data GeoTIFF selesai!")
    except Exception as e:
        print(f"⚠️ Gagal sinkronisasi dari HF: {e}. Menggunakan data lokal yang tersedia.")

# Jalankan sinkronisasi data saat server booting/startup
sync_geotiff_data()

# --- INITIALIZE FASTAPI ---
app = FastAPI(title="API Analisis Spasial InaRISK")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LocationInput(BaseModel):
    kode: str = "TEST01"
    nama_toko: str = "Toko Uji Coba"
    inisial: str = "TC"
    category: str = "BRANCH"
    latitude: float = -6.200000
    longitude: float = 106.816666

# --- FUNGSI SPASIAL & PENCARIAN REKURSIF ---

def get_kategori(nilai: float, tipe_indeks: str) -> str:
    if nilai is None or math.isnan(nilai) or nilai <= 0:
        return "Aman / Luar Area"
    
    if tipe_indeks == '1': # Indeks Kategorikal (1, 2, 3)
        mapping = {1: 'Rendah', 2: 'Sedang', 3: 'Tinggi'}
        return mapping.get(round(nilai), "Aman / Luar Area")
        
    elif tipe_indeks == '2': # Indeks Desimal InaRISK (0.0 - 1.0)
        if nilai <= 0.333:
            return 'Rendah'
        elif nilai <= 0.666:
            return 'Sedang'
        else:
            return 'Tinggi'
            
    return "Tidak Terdefinisi"

def get_all_tif_files_recursive(folder_path: str) -> list:
    """Mencari semua file .tif / .tiff secara rekursif hingga ke subfolder terdalam."""
    tif_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.tif', '.tiff')) and not file.endswith('.xml'):
                tif_files.append(os.path.join(root, file))
    return tif_files

def get_value_from_folder(folder_path: str, lat: float, lon: float) -> float:
    if not os.path.exists(folder_path):
        return 0.0
        
    # Ambil semua file pecahan TIF di folder maupun subfolder
    file_list = get_all_tif_files_recursive(folder_path)
    
    for file_tif in file_list:
        try:
            with rasterio.open(file_tif) as src:
                bounds = src.bounds
                transformer_to_raster = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                x, y = transformer_to_raster.transform(lon, lat)
                
                # Filter Bounding Box: Hanya baca piksel jika koordinat berada di dalam area TIF
                if (x >= bounds.left) and (x <= bounds.right) and (y >= bounds.bottom) and (y <= bounds.top):
                    row, col = src.index(x, y)
                    val = float(src.read(1)[row, col])
                    if val > -9000 and val > 0:
                        return val
        except IndexError:
            continue
        except Exception as e:
            print(f"Error membaca {file_tif}: {e}")
            continue
            
    return 0.0

def get_disaster_folders(base_dir: str) -> dict:
    """Mendeteksi subfolder utama bencana secara dinamis di dalam data_geotiff/."""
    disasters = {}
    if not os.path.exists(base_dir):
        return disasters
        
    for entry in os.listdir(base_dir):
        full_path = os.path.join(base_dir, entry)
        if os.path.isdir(full_path):
            # Nama subfolder utama (misal: "banjir", "gempa") menjadi kunci bencana
            disasters[entry] = {
                "folder": full_path,
                "tipe_indeks": "2"
            }
    return disasters

# --- ENDPOINT UTAMA ---

@app.get("/")
def home():
    return {"status": "API InaRISK Aktif!"}

@app.post("/api/process-location")
def process_location(data: LocationInput):
    # Pindai folder bencana secara dinamis
    disasters = get_disaster_folders(EXTRACT_DIR)
    
    results = {}
    for key, config in disasters.items():
        val = get_value_from_folder(config["folder"], data.latitude, data.longitude)
        status = get_kategori(val, config["tipe_indeks"])
        
        results[key] = {
            "nilai": round(val, 6) if val > 0 else 0.0,
            "status": status
        }
        
    return {
        "status": "success",
        "input": data.model_dump(),
        "disaster_analysis": results
    }