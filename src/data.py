"""
data.py — Modul untuk memuat dataset dan mengekstrak fitur.

Method yang tersedia:
    load_dataset()   → Muat CSV mentah → DataFrame utuh
    load_features()  → Pisahkan DataFrame → (X, y)
"""

import pandas as pd

from src.config import DATA_FILE, SELECTED_FEATURES, TARGET_COLUMN


# =====================================================================
# LOAD DATASET
# =====================================================================

def load_dataset(path: str | None = None) -> pd.DataFrame:
    """Muat CSV mentah dan kembalikan DataFrame utuh.

    Parameters
    ----------
    path : str atau None
        Path ke file CSV. Jika None, menggunakan DATA_FILE dari config.

    Returns
    -------
    pd.DataFrame
        Seluruh baris dan kolom dari CSV.
    """
    filepath = path or DATA_FILE
    df = pd.read_csv(filepath)
    return df


# =====================================================================
# LOAD FEATURES
# =====================================================================

def load_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Pisahkan DataFrame menjadi X (fitur) dan y (target).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame utuh (harus mengandung kolom SELECTED_FEATURES + TARGET_COLUMN).

    Returns
    -------
    X : pd.DataFrame
        Data fitur sesuai SELECTED_FEATURES.
    y : pd.Series
        Kolom target (price).
    """
    missing_features = set(SELECTED_FEATURES) - set(df.columns)
    if missing_features:
        raise ValueError(
            f"Kolom fitur berikut tidak ditemukan di data: {missing_features}"
        )

    missing_target = TARGET_COLUMN not in df.columns
    if missing_target:
        raise ValueError(f"Kolom target '{TARGET_COLUMN}' tidak ditemukan di data.")

    X = df[SELECTED_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y
