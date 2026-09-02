"""
config.py — Sentralisasi semua konstanta, path, dan daftar fitur.

Semua magic string / magic number wajib didefinisikan di sini.
Jika ada perubahan fitur, path, atau parameter model,
cukup edit file ini saja.
"""

from pathlib import Path

# =====================================================================
# PATHS
# =====================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "models"

DATA_FILE = DATA_RAW_DIR / "car_used.csv"
MODEL_FILE = MODEL_DIR / "linear_regression_model.pkl"
PREPROCESSOR_FILE = MODEL_DIR / "preprocessor.pkl"

# CSV mapping untuk dropdown nama mobil ↔ brand
CAR_DROPDOWN_CSV = DATA_RAW_DIR / "car_dropdown_list.csv"

# =====================================================================
# TARGET
# =====================================================================
TARGET_COLUMN = "price (Rp)"

# =====================================================================
# FEATURE DEFINITIONS
# =====================================================================
# Fitur yang digunakan model
SELECTED_FEATURES: list[str] = [
    # --- Kategorikal ---
    "car name",
    "brand",
    "transmission",
    "location",
    "plate type",
    # --- Numerik (binary 0/1) ---
    "auto retract mirror",
    "vehicle stability control",
    "auto cruise control",
    "360 camera view",
    "keyless push start",
    # --- Numerik (kontinu) ---
    "year",
    "mileage (km)",
]

# Nama kolom binary untuk sidebar display
BINARY_FEATURES: list[str] = [
    "auto retract mirror",
    "vehicle stability control",
    "auto cruise control",
    "360 camera view",
    "keyless push start",
]

# =====================================================================
# DISPLAY / FORMATTING
# =====================================================================
CURRENCY_LABEL = "Rp"

# Sidebar model info
MODEL_INFO = {
    "algorithm": "Linear Regression",
    "num_features": len(SELECTED_FEATURES),
    "metrics": {
        "MAE": "Rp 16.739.515",
        "RMSE": "Rp 39.447.422",
        "R²": "0.771",
    },
}
