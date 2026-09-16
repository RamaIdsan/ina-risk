import os
import math
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import rasterio
from rasterio.windows import Window
from pyproj import Transformer
from huggingface_hub import snapshot_download

# --- CONFIG HUGGING FACE DATASET ---
HF_DATASET_REPO = "Aquari5/inarisk-geotiff"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXTRACT_DIR = os.path.join(BASE_DIR, "data_geotiff")

# Folder lokal sebagai cadangan jika data_geotiff kosong (HF belum tersedia)
LOCAL_FALLBACK = {
    "Banjirfix": "banjir",
    "gelmobang dan abrasi": "gelombang dan abrasi",
    "letusan duar": "letusan gunung api",
    "tanah longsor fix": "tanah longsor",
    "tsunami": "tsunami",
}

# Cache agar tidak dibuat ulang setiap request
_transformer_cache = {}
_tif_cache = {}


def sync_geotiff_data():
    """Mengunduh dan menyinkronkan data GeoTIFF dari Hugging Face Datasets."""
    print("🔄 Memeriksa pembaruan data GeoTIFF dari Hugging Face...")
    try:
        snapshot_download(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            local_dir=EXTRACT_DIR,
        )
        _tif_cache.clear()
        print("✅ Sinkronisasi data GeoTIFF selesai!")
    except Exception as e:
        print(f"⚠️ Gagal sinkronisasi dari HF: {e}. Menggunakan data lokal yang tersedia.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Sinkronisasi hanya saat server start, bukan saat import module
    sync_geotiff_data()
    yield


# --- INITIALIZE FASTAPI ---
app = FastAPI(title="API Analisis Spasial InaRISK", lifespan=lifespan)

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

    if tipe_indeks == '1':  # Indeks Kategorikal (1, 2, 3)
        mapping = {1: 'Rendah', 2: 'Sedang', 3: 'Tinggi'}
        return mapping.get(round(nilai), "Aman / Luar Area")

    elif tipe_indeks == '2':  # Indeks Desimal InaRISK (0.0 - 1.0)
        if nilai <= 0.333:
            return 'Rendah'
        elif nilai <= 0.666:
            return 'Sedang'
        else:
            return 'Tinggi'

    return "Tidak Terdefinisi"


def get_transformer(src_crs):
    """Transformer di-cache per CRS karena semua raster biasanya CRS-nya sama."""
    key = src_crs.to_string() if src_crs else None
    transformer = _transformer_cache.get(key)
    if transformer is None:
        transformer = Transformer.from_crs("EPSG:4326", src_crs, always_xy=True)
        _transformer_cache[key] = transformer
    return transformer


def get_all_tif_files_recursive(folder_path: str) -> list:
    """Mencari semua file .tif / .tiff secara rekursif (hasil di-cache)."""
    cached = _tif_cache.get(folder_path)
    if cached is not None:
        return cached

    tif_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.tif', '.tiff')):
                tif_files.append(os.path.join(root, file))

    _tif_cache[folder_path] = tif_files
    return tif_files


def get_value_from_folder(folder_path: str, lat: float, lon: float) -> float:
    if not os.path.exists(folder_path):
        return 0.0

    for file_tif in get_all_tif_files_recursive(folder_path):
        try:
            with rasterio.open(file_tif) as src:
                bounds = src.bounds
                x, y = get_transformer(src.crs).transform(lon, lat)

                # Filter Bounding Box: lanjut hanya jika koordinat di dalam area TIF
                if (x >= bounds.left) and (x <= bounds.right) and (y >= bounds.bottom) and (y <= bounds.top):
                    row, col = src.index(x, y)
                    # Baca hanya 1 piksel (window 1x1), bukan seluruh band
                    val = float(src.read(1, window=Window(col, row, 1, 1))[0, 0])
                    if val > 0:
                        return val
        except IndexError:
            continue
        except Exception as e:
            print(f"Error membaca {file_tif}: {e}")
            continue

    return 0.0


def get_disaster_folders(base_dir: str) -> dict:
    """Deteksi subfolder bencana di data_geotiff/, fallback ke folder lokal."""
    disasters = {}

    if os.path.isdir(base_dir):
        for entry in os.listdir(base_dir):
            full_path = os.path.join(base_dir, entry)
            if os.path.isdir(full_path):
                disasters[entry] = {"folder": full_path, "tipe_indeks": "2"}

    if disasters:
        return disasters

    # Fallback: gunakan folder sumber lokal bila data_geotiff belum terisi
    for local_name, label in LOCAL_FALLBACK.items():
        full_path = os.path.join(BASE_DIR, local_name)
        if os.path.isdir(full_path):
            disasters[label] = {"folder": full_path, "tipe_indeks": "2"}

    return disasters


# --- ENDPOINT UTAMA ---

@app.get("/")
def home():
    return {"status": "API InaRISK Aktif!"}


@app.post("/api/process-location")
def process_location(data: LocationInput):
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
