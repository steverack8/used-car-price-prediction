"""
app.py -- Streamlit entry point: Prediksi Harga Mobil Bekas Indonesia.

File ini HANYA berisi kode UI (Streamlit widgets).
Semua logika data, model, dan prediksi di-delegate ke package src/.
Semua opsi dropdown diambil dari src/dropdown_options.py.
"""

import streamlit as st

from src.config import CURRENCY_LABEL, MODEL_INFO
from src.data import load_dataset, load_features
from src.model import load_model_artifacts
from src.prediction import build_input_dataframe, predict_price

from src.dropdown_options import (
    CAR_NAMES, CAR_NAME_DEFAULT,
    CAR_NAME_TO_BRAND,
    TRANSMISSIONS, TRANSMISSION_DEFAULT,
    LOCATIONS, LOCATION_DEFAULT,
    PLATE_TYPES, PLATE_TYPE_DEFAULT,
    YEAR_DEFAULT, YEARS,
    MILEAGE_DEFAULT,
    BINARY_FEATURES_UI, BINARY_DEFAULTS,
)

# -- Page Config ------------------------------------------------------
st.set_page_config(
    page_title="Prediksi Harga Mobil Bekas Indonesia",
    layout="wide",
)

# -- Custom CSS -------------------------------------------------------
st.markdown("""
<style>
    .block-container { padding-top: 2rem !important; }
    section[data-testid="stSidebar"] { display: none !important; }
    [data-testid="stMetric"] {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 12px 16px;
    }
    .stButton > button[kind="primary"] {
        background-color: #4A90D9;
        color: white;
        border: none;
        font-weight: 600;
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        letter-spacing: 0.02em;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #3a7bc8;
    }
    /* Tab hover */
    .stTabs [data-baseweb="tab-list"] button:hover {
        color: #4A90D9;
    }
    /* Tab active text */
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: #4A90D9;
    }
    /* Tab active underline */
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #4A90D9 !important;
    }
    /* Rapikan jarak antar elemen form */
    [data-testid="stVerticalBlockBorderWrapper"] > div:first-child {
        gap: 0.25rem;
    }
    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"] {
        gap: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# -- 1. LOAD MODEL (cached) -----------------------------------------
@st.cache_resource
def _load_model():
    """Muat model + preprocessor dari disk (hanya sekali)."""
    return load_model_artifacts()

# -- 2. LOAD DATASET (cached) ---------------------------------------
@st.cache_data
def _load_data():
    """Muat CSV + ekstrak fitur & target."""
    df = load_dataset()
    X, y = load_features(df)
    return df, X, y

# -- Jalankan load ----------------------------------------------------
model, preprocessor = _load_model()
df, X, y = _load_data()

# -- Header -----------------------------------------------------------
st.title("Prediksi Harga Beli Mobil Bekas Indonesia")

# -- Tabs -------------------------------------------------------------
tab_predict, tab_data, tab_model = st.tabs(["Prediksi", "Data Overview", "Informasi Model"])

# =====================================================================
# TAB 1 : PREDIKSI
# =====================================================================
with tab_predict:

    with st.container(border=True):
        st.subheader("Data Mobil")

        c1, c2 = st.columns(2)
        with c1:
            car_name = st.selectbox("Nama Mobil", CAR_NAMES, index=CAR_NAMES.index(CAR_NAME_DEFAULT))
        with c2:
            brand = CAR_NAME_TO_BRAND.get(car_name, "Unknown")
            st.text_input("Merek", value=brand, disabled=True)

        c3, c4 = st.columns(2)
        with c3:
            year = st.selectbox("Tahun", YEARS, index=YEARS.index(YEAR_DEFAULT))
        with c4:
            location = st.selectbox("Lokasi", LOCATIONS, index=LOCATIONS.index(LOCATION_DEFAULT))

        c5, c6 = st.columns(2)
        with c5:
            transmission = st.selectbox("Transmisi", TRANSMISSIONS,
                                        index=TRANSMISSIONS.index(TRANSMISSION_DEFAULT))
        with c6:
            plate_type = st.selectbox("Plat Nomor", PLATE_TYPES,
                                      index=PLATE_TYPES.index(PLATE_TYPE_DEFAULT))

        # Text input dengan auto-format pemisah ribuan
        if "mileage_str" not in st.session_state:
            st.session_state.mileage_str = f"{MILEAGE_DEFAULT:,}"

        def _format_mileage():
            raw = st.session_state.mileage_input.replace(",", "").strip()
            if raw.isdigit():
                st.session_state.mileage_str = f"{int(raw):,}"
            elif raw == "":
                st.session_state.mileage_str = ""

        mileage_text = st.text_input(
            "Jarak Tempuh (km)",
            value=st.session_state.mileage_str,
            key="mileage_input",
            on_change=_format_mileage,
        )

    st.subheader("Fitur Tambahan")
    binary_values: dict[str, bool] = {}
    cols = st.columns(3)
    for idx, (feat, label) in enumerate(BINARY_FEATURES_UI.items()):
        col = cols[idx % 3]
        with col:
            binary_values[feat] = st.checkbox(label, value=BINARY_DEFAULTS[feat])

    st.markdown("")

    if st.button("Prediksi Harga", type="primary", use_container_width=True):
        # Validasi mileage harus angka
        mileage_clean = mileage_text.replace(",", "").strip()
        if not mileage_clean.isdigit() or int(mileage_clean) < 0:
            st.error("Jarak Tempuh harus berupa angka positif (contoh: 50.000).")
            st.stop()

        # Dataset menyimpan mileage dalam ribuan km,
        # jadi input user (dalam km) dibagi 1000.
        mileage_model = int(mileage_clean) / 1000

        raw_inputs = {
            "car name": car_name, "brand": brand,
            "transmission": transmission, "location": location,
            "plate type": plate_type, "year": int(year),
            "mileage (km)": mileage_model,
        }
        raw_inputs.update({k: int(v) for k, v in binary_values.items()})

        input_df = build_input_dataframe(raw_inputs)
        price = predict_price(input_df, model, preprocessor)

        st.markdown("---")
        res1, res2, res3 = st.columns(3)
        with res2:
            st.metric(label="Estimasi Harga", value=f"{CURRENCY_LABEL} {price:,.0f}")

# =====================================================================
# TAB 2 : DATA OVERVIEW
# =====================================================================
with tab_data:

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Data", f"{len(df):,} baris")
    k2.metric("Rata-rata Harga", f"{CURRENCY_LABEL} {y.mean():,.0f}")
    k3.metric("Minimum", f"{CURRENCY_LABEL} {y.min():,.0f}")
    k4.metric("Maksimum", f"{CURRENCY_LABEL} {y.max():,.0f}")

    st.markdown("")

    with st.container(border=True):
        display_df = df.copy()
        display_df["year"] = display_df["year"].astype(str)
        st.dataframe(display_df, use_container_width=True, height=400)

    st.markdown("")

    g1, g2 = st.columns(2)
    with g1:
        with st.container(border=True):
            st.subheader("Harga Rata-rata per Brand")
            st.bar_chart(df.groupby("brand")["price (Rp)"].mean().sort_values(ascending=False))
    with g2:
        with st.container(border=True):
            st.subheader("Distribusi Tahun")
            st.bar_chart(df["year"].astype(str).value_counts().sort_index())

# =====================================================================
# TAB 3 : INFO MODEL
# =====================================================================
with tab_model:
    st.caption("Algoritma: Linear Regression")
    st.caption(f"Fitur: {MODEL_INFO['num_features']} | MAE: {MODEL_INFO['metrics']['MAE']} | R²: {MODEL_INFO['metrics']['R²']}")
