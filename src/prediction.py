"""
prediction.py — Modul prediksi harga.

Membungkus logika transformasi input → prediksi
sehingga UI tidak perlu tahu detail preprocessing.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

from src.config import SELECTED_FEATURES


def build_input_dataframe(raw_inputs: dict) -> pd.DataFrame:
    """Bangun DataFrame satu baris dari input mentah UI.

    Parameters
    ----------
    raw_inputs : dict
        Dictionary dengan kunci sesuai nama kolom di SELECTED_FEATURES.

    Returns
    -------
    pd.DataFrame
        Satu baris data dengan kolom urut sesuai SELECTED_FEATURES.
    """
    df = pd.DataFrame([raw_inputs])
    # Pastikan urutan kolom sesuai saat training
    df = df[SELECTED_FEATURES]
    return df


def predict_price(
    input_df: pd.DataFrame,
    model: LinearRegression,
    preprocessor: ColumnTransformer,
) -> float:
    """Prediksi harga dari DataFrame input.

    Parameters
    ----------
    input_df : pd.DataFrame
        Output dari build_input_dataframe().
    model : LinearRegression
    preprocessor : ColumnTransformer

    Returns
    -------
    float
        Harga prediksi dalam Rupiah.
    """
    processed = preprocessor.transform(input_df)
    price = model.predict(processed)[0]
    return float(price)
