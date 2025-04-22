import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from streamlit_lottie import st_lottie
import requests

# Load animation from URL
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load animations
solar_animation = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_dzyzxlqg.json")
predict_animation = load_lottieurl("https://assets10.lottiefiles.com/private_files/lf30_hc3vpx.json")
sidebar_animation = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_gjmecwii.json")

st.set_page_config(page_title="Solar Power Generation Predictor", page_icon="☀️", layout="wide")

@st.cache_resource
def load_model():
    with open('random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('features.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, scaler, features

model, scaler, features = load_model()

# Header
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("<h1 style='color:black;'>☀️ Solar Power Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:black;'>Predict solar power generation based on environment.</p>", unsafe_allow_html=True)
with col2:
    st_lottie(solar_animation, height=200)

# Inputs
col1, col2 = st.columns(2)
with col1:
    st.subheader("🌤️ Weather")
    noon = st.text_input("Solar Noon Distance", "0.0")
    temp = st.text_input("Temperature (°C)", "25.0")
    wind_spd = st.text_input("Wind Speed (km/h)", "10.0")
    sky = st.text_input("Sky Cover", "2.0")
with col2:
    st.subheader("🌬️ Atmosphere")
    wind_dir = st.text_input("Wind Direction (°)", "180.0")
    vis = st.text_input("Visibility (km)", "10.0")
    humid = st.text_input("Humidity (%)", "50.0")
    press = st.text_input("Pressure (hPa)", "1013.0")

if st.button('🔮 Predict Power Generation', use_container_width=True):
    try:
        data = {
            'distance_to_solar_noon': float(noon),
            'temperature': float(temp),
            'wind_direction': float(wind_dir),
            'wind_speed': float(wind_spd),
            'sky_cover': float(sky),
            'visibility': float(vis),
            'humidity': float(humid),
            'average_pressure': float(press)
        }
        df = pd.DataFrame([data], columns=features + ['power_generated']).assign(power_generated=0)
        pred = np.exp(model.predict(scaler.transform(df))[:, 0])

        st.success(f"Predicted Power: {pred[0]:.2f} MW")
        st.info(f"1 Hour Energy: {pred[0] * 1_000_000 * 3600:,.0f} J")
        st_lottie(predict_animation, height=100)
    except Exception as e:
        st.error(f"Error: {e}")

with st.sidebar:
    st_lottie(sidebar_animation, height=100)
    st.header("ℹ️ About")
    st.markdown("ML model predicting solar power output.")
    st.header("📝 How to Use")
    st.markdown("Enter weather data & click predict.")

st.markdown("<hr>", unsafe_allow_html=True)
