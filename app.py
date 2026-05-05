import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# Cargar modelo y columnas
# =========================
model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")

st.set_page_config(
    page_title="Predicción de Precio de Casas",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 Predicción de Precio de Casas")
st.write("Ingresa los datos principales de la vivienda para estimar su precio.")

# =========================
# Inputs principales
# =========================
st.subheader("Características principales")

bedrooms = st.number_input("Habitaciones", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Baños", min_value=1.0, max_value=10.0, value=2.0, step=0.5)
sqft_living = st.number_input("Área habitable (sqft)", min_value=200, max_value=10000, value=1800)
sqft_lot = st.number_input("Área del terreno (sqft)", min_value=500, max_value=500000, value=5000)
floors = st.number_input("Pisos", min_value=1.0, max_value=5.0, value=1.0, step=0.5)

st.subheader("Condición y ubicación")

waterfront = st.selectbox("¿Tiene vista al agua?", ["No", "Sí"])
waterfront = 1 if waterfront == "Sí" else 0

view = st.slider("Calidad de vista", 0, 4, 0)
condition = st.slider("Condición de la casa", 1, 5, 3)

city = st.selectbox(
    "Ciudad",
    ["Seattle", "Renton", "Bellevue", "Redmond", "Kent", "Kirkland", "Auburn", "Shoreline", "Other"]
)

st.subheader("Construcción")

yr_built = st.number_input("Año de construcción", min_value=1800, max_value=2025, value=1990)
yr_renovated = st.number_input("Año de renovación (0 si no aplica)", min_value=0, max_value=2025, value=0)

# =========================
# Crear input compatible
# =========================
input_data = {col: 0 for col in columns}

house_age = 2025 - yr_built
was_renovated = 1 if yr_renovated > 0 else 0
total_size = sqft_living + sqft_lot
bath_per_room = bathrooms / (bedrooms + 1)
sqft_per_bedroom = sqft_living / (bedrooms + 1)

manual_values = {
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "sqft_living": sqft_living,
    "sqft_lot": sqft_lot,
    "floors": floors,
    "waterfront": waterfront,
    "view": view,
    "condition": condition,
    "sqft_above": sqft_living,
    "sqft_basement": 0,
    "yr_built": yr_built,
    "yr_renovated": yr_renovated,
    "house_age": house_age,
    "was_renovated": was_renovated,
    "total_size": total_size,
    "bath_per_room": bath_per_room,
    "sqft_per_bedroom": sqft_per_bedroom,
    "basement_ratio": 0,
}

for col, value in manual_values.items():
    if col in input_data:
        input_data[col] = value

city_column = f"city_{city}"
if city_column in input_data:
    input_data[city_column] = 1

input_df = pd.DataFrame([input_data])

# =========================
# Predicción
# =========================
st.divider()

if st.button("Calcular precio 💰", use_container_width=True):
    prediction_log = model.predict(input_df)[0]
    prediction = np.expm1(prediction_log)

    st.markdown(
        f"""
        <div style="
            background-color:#f0f2f6;
            padding:30px;
            border-radius:18px;
            text-align:center;
            margin-top:20px;
            border: 1px solid #ddd;
        ">
            <h3 style="color:#333;">Precio estimado</h3>
            <h1 style="color:#0E7C7B;">${prediction:,.2f}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )