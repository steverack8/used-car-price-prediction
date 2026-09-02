"""
model.py — Modul untuk memuat model dari disk.

Method yang tersedia:
    load_model_artifacts()  → Muat model + preprocessor dari .pkl
"""

import joblib

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

from src.config import MODEL_FILE, PREPROCESSOR_FILE


# =====================================================================
# LOAD MODEL ARTIFACTS
# =====================================================================

def load_model_artifacts() -> tuple[LinearRegression, ColumnTransformer]:
    """Muat model dan preprocessor dari disk (.pkl).

    Returns
    -------
    model : LinearRegression
        Model LinearRegression yang sudah di-training.
    preprocessor : ColumnTransformer
        Pipeline preprocessing (StandardScaler + OneHotEncoder).
    """
    model = joblib.load(MODEL_FILE)
    preprocessor = joblib.load(PREPROCESSOR_FILE)
    return model, preprocessor
