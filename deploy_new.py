import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from streamlit_lottie import st_lottie
import requests

# Function to load Lottie animation from URL
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load Lottie animations
solar_animation = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_dzyzxlqg.json")  # solar panel
predict_animation = load_lottieurl("https://assets10.lottiefiles.com/private_files/lf30_hc3vpx.json")  # forecast
sidebar_animation = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_gjmecwii.json")  # info bulb

# Set page config
st.set_page_config(
    page_title="Solar Power Generation Predictor",
    page_icon="☀️",
    layout="wide"
)

@st.cache_resource
def load_model():
    with open('random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('features.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    return model, scaler, feature_names

model, scaler, feature_names = load_model()

# CSS STYLING (same as before)...
# [Your CSS goes here, unchanged, as already styled for black text and background]

# Header with animation
with st.container():
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("<h1 class='header'>☀️ Solar Power Generation Predictor</h1>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background-color: rgba(255, 255, 255, 0.7); padding: 1rem; border-radius: 10px; margin-bottom: 2rem;">
        Predict solar power generation based on weather and environmental conditions.
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st_lottie(solar_animation, height=200, key="header_anim")

# Input Form
col1, col2 = st.columns(2)
with col1:
    st.markdown("<div class='input-box'>", unsafe_allow_html=True)
    st.subheader("🌤️ Solar & Weather Parameters")
    distance_to_solar_noon = st.text_input("Distance to Solar Noon (degrees)", "0.0")
    temperature = st.text_input("Temperature (°C)", "25.0")
    wind_speed = st.text_input("Wind Speed (km/h)", "10.0")
    sky_cover = st.text_input("Sky Cover (oktas)", "2.0")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='input-box'>", unsafe_allow_html=True)
    st.subheader("🌬️ Atmospheric Conditions")
    wind_direction = st.text_input("Wind Direction (°)", "180.0")
    visibility = st.text_input("Visibility (km)", "10.0")
    humidity = st.text_input("Humidity (%)", "50.0")
    average_pressure = st.text_input("Average Pressure (hPa)", "1013.0")
    st.markdown("</div>", unsafe_allow_html=True)

if st
