import glob
import os
import re

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import rasterio
from rasterio.windows import Window
from pyproj import Transformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

WGS84 = "EPSG:4326"

FOLDER_ALIAS = {
    "banjirfix": "banjir",
    "gelmobang dan abrasi": "gelombang_laut_dan_abrasi",
    "letusan duar": "letusan_gunung_api",
    "tanah longsor fix": "tanah_longsor",
    "tsunami": "tsunami",
}

_transformer_cache = {}


def get_transformer(crs):
    key = crs.to_string() if hasattr(crs, "to_string") else str(crs)
    if key not in _transformer_cache:
        _transformer_cache[key] = Transformer.from_crs(WGS84, crs, always_xy=True)
    return _transformer_cache[key]


def normalize_key(folder_name):
    alias = FOLDER_ALIAS.get(folder_name.strip().lower())
    if alias:
        return alias
    slug = re.sub(r"[^a-z0-9]+", "_", folder_name.strip().lower()).strip("_")
    return slug or folder_name.strip().lower()


def build_disaster_config():
    config = {}
    for entry in sorted(os.listdir(BASE_DIR)):
        folder_path = os.path.join(BASE_DIR, entry)
        if not os.path.isdir(folder_path):
            continue
        tif_files = glob.glob(os.path.join(folder_path, "*.tif")) + glob.glob(
            os.path.join(folder_path, "*.tiff")
        )
        if not tif_files:
            continue
        config[normalize_key(entry)] = {
            "folder_path": folder_path,
            "tipe_indeks": "2",
        }
    return config


app = FastAPI(title="Disaster Risk API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LocationInput(BaseModel):
    kode: str
    nama_toko: str
    inisial: str
    category: str
    latitude: float
    longitude: float


def get_kategori(nilai, tipe_indeks):
    if nilai is None or nilai <= 0:
        return "Aman / Luar Area"

    if tipe_indeks == "1":
        if nilai == 1:
            return "Rendah"
        if nilai == 2:
            return "Sedang"
        if nilai == 3:
            return "Tinggi"
        return "Tidak Diketahui"

    if nilai <= 0.333:
        return "Rendah"
    if nilai <= 0.666:
        return "Sedang"
    return "Tinggi"


def get_value_from_folder(folder_path, lat, lon):
    tif_files = sorted(
        glob.glob(os.path.join(folder_path, "*.tif"))
        + glob.glob(os.path.join(folder_path, "*.tiff"))
    )

    for tif_path in tif_files:
        try:
            with rasterio.open(tif_path) as src:
                if src.crs is None:
                    continue

                transformer = get_transformer(src.crs)
                x, y = transformer.transform(lon, lat)

                bounds = src.bounds
                if not (
                    bounds.left <= x <= bounds.right
                    and bounds.bottom <= y <= bounds.top
                ):
                    continue

                row, col = src.index(x, y)
                if row < 0 or col < 0 or row >= src.height or col >= src.width:
                    continue

                window = Window(col, row, 1, 1)
                data = src.read(1, window=window)
                if data.size == 0:
                    continue

                value = float(data[0][0])
                if value < 0:
                    continue
                if src.nodata is not None and value == src.nodata:
                    continue

                return value
        except rasterio.errors.RasterioIOError:
            continue

    return None


DISASTER_CONFIG = build_disaster_config()


@app.get("/")
def health_check():
    return {
        "status": "success",
        "message": "Disaster Risk API is running",
        "disasters": list(DISASTER_CONFIG.keys()),
    }


@app.post("/api/process-location")
def process_location(location: LocationInput):
    disaster_analysis = {}

    for nama_bencana, cfg in DISASTER_CONFIG.items():
        nilai = get_value_from_folder(
            cfg["folder_path"], location.latitude, location.longitude
        )
        status = get_kategori(nilai, cfg["tipe_indeks"])

        if status == "Aman / Luar Area" or nilai is None:
            disaster_analysis[nama_bencana] = {
                "nilai": 0.0,
                "status": "Aman / Luar Area",
            }
        else:
            disaster_analysis[nama_bencana] = {
                "nilai": round(float(nilai), 6),
                "status": status,
            }

    return {
        "status": "success",
        "data": location.model_dump(),
        "disaster_analysis": disaster_analysis,
    }
