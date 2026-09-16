import os
import math
import secrets
import urllib.parse
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import rasterio
from rasterio.windows import Window
from pyproj import Transformer
from huggingface_hub import snapshot_download, HfApi

# --- CONFIG HUGGING FACE DATASET ---
HF_DATASET_REPO = "Aquari5/inarisk-geotiff"
HF_REPO_TYPE = "dataset"
HF_DATA_PREFIX = "data_geotiff"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- CONFIG MODE DATA ---
# 'vsicurl'  (default) -> baca 1 piksel langsung dari Hugging Face lewat HTTP
#                         Range pakai GDAL /vsicurl/. Cocok untuk serverless
#                         (Vercel) karena TIDAK mengunduh GeoTIFF.
# 'download' -> unduh seluruh GeoTIFF ke disk lokal lalu baca dari disk. Cocok
#               untuk dev lokal atau container dengan disk besar (HF Spaces).
DATA_MODE = os.environ.get("DATA_MODE", "vsicurl").strip().lower()

# Folder unduhan untuk mode 'download'. Di Vercel filesystem read-only kecuali
# /tmp, jadi arahkan lewat env: HF_EXTRACT_DIR=/tmp/data_geotiff
EXTRACT_DIR = os.environ.get("HF_EXTRACT_DIR", os.path.join(BASE_DIR, "data_geotiff"))

# --- CONFIG KEAMANAN API ---
# Bila INARISK_API_KEY diisi di environment, setiap request ke
# /api/process-location wajib menyertakan header X-API-Key yang sama.
# Biarkan kosong untuk menonaktifkan (mis. saat pengujian lokal).
API_KEY = os.environ.get("INARISK_API_KEY", "").strip()

# GDAL: jangan "list directory" di server HF dan batasi extension yang boleh
# dibuka lewat /vsicurl/, supaya hanya byte yang perlu yang diunduh.
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff")
os.environ.setdefault("VSI_CACHE", "TRUE")
os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")

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
_local_tif_cache = {}
_remote_index = {"folders": None}
_file_meta_cache = {}


# --- HELPER HUGGING FACE (MODE vsicurl) ---

def hf_resolve_url(repo_path: str) -> str:
    """URL 'resolve' HF untuk sebuah file di dataset repo."""
    quoted = urllib.parse.quote(repo_path)
    return f"https://huggingface.co/datasets/{HF_DATASET_REPO}/resolve/main/{quoted}"


def vsicurl_path(repo_path: str) -> str:
    """Path GDAL /vsicurl/ untuk sebuah file di dataset repo HF."""
    return "/vsicurl/" + hf_resolve_url(repo_path)


def list_remote_hazard_files() -> dict:
    """
    Mengambil daftar file .tif dari HF API (tanpa mengunduh isinya) dan
    mengelompokkannya per folder bencana di bawah data_geotiff/.
    Hasil di-cache selama instance hidup.
    """
    if _remote_index["folders"] is not None:
        return _remote_index["folders"]

    folders = {}
    try:
        api = HfApi()
        files = api.list_repo_files(repo_id=HF_DATASET_REPO, repo_type=HF_REPO_TYPE)
    except Exception as e:
        print(f"WARN: gagal membaca daftar file HF: {e}")
        _remote_index["folders"] = folders
        return folders

    prefix = HF_DATA_PREFIX + "/"
    for path in files:
        if not path.startswith(prefix):
            continue
        if not path.lower().endswith((".tif", ".tiff")):
            continue

        relative = path[len(prefix):]
        if "/" not in relative:
            continue
        hazard = relative.split("/", 1)[0]
        folders.setdefault(hazard, []).append(path)

    _remote_index["folders"] = folders
    print(f"OK: Index HF berisi {len(folders)} bencana, "
          f"{sum(len(v) for v in folders.values())} file GeoTIFF.")
    return folders


def get_remote_file_meta(repo_path: str):
    """Cache bounds+CRS tiap file remote (1x buka header, bukan isi raster)."""
    url = vsicurl_path(repo_path)
    meta = _file_meta_cache.get(url)
    if meta is not None:
        return meta

    try:
        with rasterio.open(url) as src:
            meta = {"bounds": src.bounds, "crs": src.crs}
        _file_meta_cache[url] = meta
    except Exception as e:
        print(f"Error membaca metadata {repo_path}: {e}")
        meta = None
    return meta


def get_value_from_remote_files(files: list, lat: float, lon: float) -> float:
    for repo_path in files:
        meta = get_remote_file_meta(repo_path)
        if not meta:
            continue

        bounds = meta["bounds"]
        x, y = get_transformer(meta["crs"]).transform(lon, lat)

        if not (x >= bounds.left and x <= bounds.right and y >= bounds.bottom and y <= bounds.top):
            continue

        try:
            with rasterio.open(vsicurl_path(repo_path)) as src:
                row, col = src.index(x, y)
                val = float(src.read(1, window=Window(col, row, 1, 1))[0, 0])
                if val > 0:
                    return val
        except IndexError:
            continue
        except Exception as e:
            print(f"Error membaca piksel {repo_path}: {e}")
            continue

    return 0.0


# --- MODE download (disk lokal) ---

def sync_geotiff_data():
    """Mengunduh dan menyinkronkan data GeoTIFF dari Hugging Face Datasets."""
    print("INFO: memeriksa pembaruan data GeoTIFF dari Hugging Face...")
    try:
        snapshot_download(
            repo_id=HF_DATASET_REPO,
            repo_type=HF_REPO_TYPE,
            local_dir=EXTRACT_DIR,
        )
        _local_tif_cache.clear()
        print("OK: sinkronisasi data GeoTIFF selesai.")
    except Exception as e:
        print(f"WARN: gagal sinkronisasi dari HF: {e}. Menggunakan data lokal yang tersedia.")


def get_all_tif_files_recursive(folder_path: str) -> list:
    """Mencari semua file .tif / .tiff secara rekursif (hasil di-cache)."""
    cached = _local_tif_cache.get(folder_path)
    if cached is not None:
        return cached

    tif_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.tif', '.tiff')):
                tif_files.append(os.path.join(root, file))

    _local_tif_cache[folder_path] = tif_files
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


def get_disaster_sources() -> dict:
    """Mengembalikan sumber data per bencana sesuai DATA_MODE."""
    if DATA_MODE == "download":
        return get_disaster_folders(EXTRACT_DIR)
    return list_remote_hazard_files()


def get_value_from_source(source, lat: float, lon: float) -> float:
    if DATA_MODE == "download":
        return get_value_from_folder(source["folder"], lat, lon)
    return get_value_from_remote_files(source, lat, lon)


# --- LIFESPAN ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Siapkan sumber data saat server start (bukan saat import module).
    if DATA_MODE == "download":
        sync_geotiff_data()
    else:
        try:
            list_remote_hazard_files()
        except Exception as e:
            print(f"WARN: inisialisasi index HF gagal: {e}")
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


# --- FUNGSI SPASIAL ---

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


# --- ENDPOINT UTAMA ---

def verify_api_key(x_api_key: str = Header(default="")):
    """Validasi header X-API-Key bila INARISK_API_KEY diatur di environment."""
    if API_KEY and not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(
            status_code=401,
            detail="API key tidak valid atau tidak disertakan.",
        )


@app.get("/")
def home():
    return {
        "status": "API InaRISK Aktif!",
        "auth_aktif": bool(API_KEY),
        "data_mode": DATA_MODE,
        "jumlah_bencana": len(get_disaster_sources()),
    }


@app.post("/api/process-location", dependencies=[Depends(verify_api_key)])
def process_location(data: LocationInput):
    disasters = get_disaster_sources()

    results = {}
    for key, source in disasters.items():
        val = get_value_from_source(source, data.latitude, data.longitude)
        status = get_kategori(val, "2")

        results[key] = {
            "nilai": round(val, 6) if val > 0 else 0.0,
            "status": status
        }

    return {
        "status": "success",
        "input": data.model_dump(),
        "disaster_analysis": results
    }
