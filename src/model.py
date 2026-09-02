"""
model.py — Modul untuk memuat model dari disk.

Method yang tersedia:
    load_model_artifacts()  → Muat model + preprocessor dari .pkl
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    DATA_FILE,
    MODEL_DIR,
    MODEL_FILE,
    PREPROCESSOR_FILE,
    SELECTED_FEATURES,
    TARGET_COLUMN,
)


# =====================================================================
# TRAIN (hanya jika .pkl belum ada)
# =====================================================================

def _train_and_save() -> tuple[LinearRegression, ColumnTransformer]:
    """Train model dari nol lalu simpan ke .pkl."""
    df = pd.read_csv(DATA_FILE)
    X = df[SELECTED_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()

    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    X_train, X_remaining, y_train, y_remaining = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_remaining, y_remaining, test_size=0.50, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)

    model = LinearRegression()
    model.fit(X_train_processed, y_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(preprocessor, PREPROCESSOR_FILE)

    return model, preprocessor


# =====================================================================
# LOAD MODEL ARTIFACTS
# =====================================================================

def load_model_artifacts() -> tuple[LinearRegression, ColumnTransformer]:
    """Muat model dan preprocessor dari disk (.pkl).

    Jika file belum ada (misalnya pertama kali deploy),
    otomatis training dulu lalu simpan.

    Returns
    -------
    model : LinearRegression
        Model LinearRegression yang sudah di-training.
    preprocessor : ColumnTransformer
        Pipeline preprocessing (StandardScaler + OneHotEncoder).
    """
    if MODEL_FILE.exists() and PREPROCESSOR_FILE.exists():
        model = joblib.load(MODEL_FILE)
        preprocessor = joblib.load(PREPROCESSOR_FILE)
    else:
        model, preprocessor = _train_and_save()
    return model, preprocessor
