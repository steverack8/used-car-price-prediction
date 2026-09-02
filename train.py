"""
train.py — Training script untuk model prediksi harga mobil bekas.

Sesuai pipeline notebook: ColumnTransformer (StandardScaler + OneHotEncoder)
pada SEMUA kolom numerik (termasuk binary), lalu LinearRegression.

Jalankan:  python train.py

Akan menghasilkan:
  - models/linear_regression_model.pkl
  - models/preprocessor.pkl
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    DATA_FILE,
    MODEL_DIR,
    MODEL_FILE,
    PREPROCESSOR_FILE,
    SELECTED_FEATURES,
    TARGET_COLUMN,
)


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    print(f"Loading dataset dari {DATA_FILE} ...")
    df = pd.read_csv(DATA_FILE)
    print(f"  Baris: {len(df)}, Kolom: {len(df.columns)}")

    X = df[SELECTED_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()

    # ------------------------------------------------------------------
    # 2. Definisikan tipe kolom (sesuai notebook)
    #    - Kategorikal: object dtype
    #    - Numerik: int64 / float64 (termasuk binary 0/1)
    # ------------------------------------------------------------------
    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    print(f"\nKategorikal ({len(categorical_cols)}): {categorical_cols}")
    print(f"Numerik     ({len(numeric_cols)}): {numeric_cols}")
    print(f"Total fitur: {len(SELECTED_FEATURES)}")

    # ------------------------------------------------------------------
    # 3. Split: 70% train, 15% val, 15% test (persis seperti notebook)
    # ------------------------------------------------------------------
    X_train, X_remaining, y_train, y_remaining = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_remaining, y_remaining, test_size=0.50, random_state=42
    )

    print(f"\nTrain: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # ------------------------------------------------------------------
    # 4. Preprocessing (ColumnTransformer)
    #    StandardScaler untuk semua numerik, OneHotEncoder untuk kategorikal
    # ------------------------------------------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    print(f"\nProcessed shape: {X_train_processed.shape}")

    # ------------------------------------------------------------------
    # 5. Training Linear Regression (model terbaik dari perbandingan)
    # ------------------------------------------------------------------
    print("\nTraining Linear Regression ...")
    model = LinearRegression()
    model.fit(X_train_processed, y_train)

    # ------------------------------------------------------------------
    # 6. Evaluasi
    # ------------------------------------------------------------------
    # Validation
    y_val_pred = model.predict(X_val_processed)
    mae_val = mean_absolute_error(y_val, y_val_pred)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
    r2_val = r2_score(y_val, y_val_pred)

    # Test
    y_test_pred = model.predict(X_test_processed)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    r2_test = r2_score(y_test, y_test_pred)

    print("\n" + "=" * 50)
    print("VALIDATION SET")
    print("=" * 50)
    print(f"  MAE  : {mae_val:,.0f}")
    print(f"  RMSE : {rmse_val:,.0f}")
    print(f"  R²   : {r2_val:.4f}")

    print("\n" + "=" * 50)
    print("TEST SET")
    print("=" * 50)
    print(f"  MAE  : {mae_test:,.0f}")
    print(f"  RMSE : {rmse_test:,.0f}")
    print(f"  R²   : {r2_test:.4f}")
    print("=" * 50)

    # ------------------------------------------------------------------
    # 7. Save model & preprocessor
    # ------------------------------------------------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(preprocessor, PREPROCESSOR_FILE)

    print(f"\nModel disimpan       : {MODEL_FILE}")
    print(f"Preprocessor disimpan: {PREPROCESSOR_FILE}")
    print("Selesai!")


if __name__ == "__main__":
    main()
