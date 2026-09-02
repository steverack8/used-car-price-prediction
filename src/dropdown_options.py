"""
dropdown_options.py - Sumber kebenaran tunggal untuk semua opsi dropdown UI.

Semua opsi dropdown, range, dan mapping DIAMBIL DARI DATA (CSV).
Tidak ada yang didefinisikan manual - jika data berubah, dropdown ikut berubah.
"""

import csv

import pandas as pd

from src.config import CAR_DROPDOWN_CSV, DATA_FILE, BINARY_FEATURES


# =====================================================================
# HELPER - baca dataset utama
# =====================================================================
_df_dataset: pd.DataFrame = pd.read_csv(DATA_FILE)


def _unique_sorted(column: str) -> list[str]:
    """Ambil nilai unik dari kolom, urutkan abjad."""
    return sorted(_df_dataset[column].dropna().astype(str).unique().tolist())


def _safe_first(values: list[str], preferred: str = "") -> str:
    """Ambil elemen pertama; jika preferred ada di list, prioritaskan itu."""
    if preferred in values:
        return preferred
    return values[0] if values else ""


# =====================================================================
# CAR NAMES & CAR NAME -> BRAND MAPPING - from car_dropdown_list.csv
# =====================================================================
def _load_car_dropdown() -> tuple[list[str], dict[str, str]]:
    """Baca car_dropdown_list.csv -> (CAR_NAMES, CAR_NAME_TO_BRAND)."""
    names: list[str] = []
    mapping: dict[str, str] = {}

    with open(CAR_DROPDOWN_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["car name"].strip()
            brand = row["brand"].strip()
            names.append(name)
            mapping[name] = brand

    seen: set[str] = set()
    unique_sorted: list[str] = []
    for n in sorted(names):
        if n not in seen:
            seen.add(n)
            unique_sorted.append(n)

    return unique_sorted, mapping


CAR_NAMES, CAR_NAME_TO_BRAND = _load_car_dropdown()
CAR_NAME_DEFAULT = CAR_NAMES[0] if CAR_NAMES else ""

# =====================================================================
# BRANDS - from car_dropdown_list.csv
# =====================================================================
BRANDS: list[str] = sorted({b for b in CAR_NAME_TO_BRAND.values()})
BRAND_DEFAULT = "Toyota" if "Toyota" in BRANDS else BRANDS[0]

# =====================================================================
# TRANSMISSIONS - from dataset
# =====================================================================
TRANSMISSIONS: list[str] = _unique_sorted("transmission")
TRANSMISSION_DEFAULT = _safe_first(TRANSMISSIONS, "Automatic")

# =====================================================================
# LOCATIONS - from dataset
# =====================================================================
LOCATIONS: list[str] = _unique_sorted("location")
LOCATION_DEFAULT = _safe_first(LOCATIONS, "Jakarta Selatan")

# =====================================================================
# PLATE TYPES - from dataset
# =====================================================================
PLATE_TYPES: list[str] = _unique_sorted("plate type")
PLATE_TYPE_DEFAULT = _safe_first(PLATE_TYPES, "odd plate")

# =====================================================================
# YEAR RANGE - from dataset
# =====================================================================
_year_min = int(_df_dataset["year"].min())
_year_max = int(_df_dataset["year"].max())
YEAR_MIN = _year_min
YEAR_MAX = _year_max
YEARS: list[int] = list(range(YEAR_MAX, YEAR_MIN - 1, -1))
YEAR_DEFAULT = _safe_first(
    [str(y) for y in YEARS],
    preferred="2021",
)
YEAR_DEFAULT = int(YEAR_DEFAULT)

# =====================================================================
# MILEAGE RANGE - from dataset
# =====================================================================
_mileage_max_raw = float(_df_dataset["mileage (km)"].max())
_mileage_median = float(_df_dataset["mileage (km)"].median())
MILEAGE_MIN = 0
MILEAGE_MAX = int(_mileage_max_raw * 1.2)  # buffer 20% di atas max
MILEAGE_DEFAULT = int(round(_mileage_median))
MILEAGE_STEP = 1

# =====================================================================
# BINARY FEATURES - checkbox untuk fitur-fitur 0/1
# Label di-generate otomatis dari nama kolom
# =====================================================================
BINARY_FEATURES_UI: dict[str, str] = {
    feat: feat.replace("_", " ").title()
    for feat in BINARY_FEATURES
}

BINARY_DEFAULTS: dict[str, bool] = {feat: False for feat in BINARY_FEATURES}
