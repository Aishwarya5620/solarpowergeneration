#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(
    page_title="Solar Power Predictor",
    page_icon="☀️",
    layout="centered"
)

# Load the saved model, scaler, and feature names
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

# Styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f8ff;
        color: black !important;
    }
    .title {
        font-size: 2.5rem;
        font-weight: 600;
        color: black !important;
        text-align: center;
        margin-bottom: 1rem;
    }
    .section {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        color: black !important;
    }
    .stButton>button {
        background-color: #007acc;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #005f99;
    }
    label, .stTextInput, .stNumberInput, .stMetric, .stSelectbox, .stSlider, .stCheckbox {
        color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Prediction function
def predict_power(input_data):
    input_df = pd.DataFrame([input_data], columns=feature_names)
    dummy_row = input_df.copy()
    dummy_row['power_generated'] = 0
    scaled_data = scaler.transform(dummy_row)
    input_scaled = scaled_data[:, :-1]
    log_prediction = model.predict(input_scaled)
    prediction = np.exp(log_prediction)[0]
    return prediction

# Title
st.markdown("<div class='title'>☀️ Solar Power Predictor</div>", unsafe_allow_html=True)

# Input section
with st.container():
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.subheader("Enter Weather & Atmospheric Data")

    col1, col2 = st.columns(2)

    with col1:
        distance_to_solar_noon = st.number_input("Distance to Solar Noon (degrees)", value=0.0)
        temperature = st.number_input("Temperature (°C)", value=25.0)
        wind_speed = st.number_input("Wind Speed (km/h)", value=10.0)
        sky_cover = st.number_input("Sky Cover (oktas, 0-8)", value=2.0)

    with col2:
        wind_direction = st.number_input("Wind Direction (degrees, 0-360)", value=180.0)
        visibility = st.number_input("Visibility (km)", value=10.0)
        humidity = st.number_input("Humidity (%)", value=50.0)
        average_pressure = st.number_input("Average Pressure (hPa)", value=1013.0)

    st.markdown("</div>", unsafe_allow_html=True)

    # Prediction
    if st.button('🔍 Predict Power Generation'):
        try:
            input_data = {
                'distance_to_solar_noon': distance_to_solar_noon,
                'temperature': temperature,
                'wind_direction': wind_direction,
                'wind_speed': wind_speed,
                'sky_cover': sky_cover,
                'visibility': visibility,
                'humidity': humidity,
                'average_pressure': average_pressure
            }

            prediction = predict_power(input_data)
            energy_joules = prediction * 1_000_000 * 3600

            st.success(f"Predicted Power Generation: {prediction:.2f} MW")
            st.info(f"Estimated Energy for 1 Hour: {energy_joules:,.0f} J")

            # Efficiency gauge
            efficiency = (prediction / 10) * 100
            st.metric("System Efficiency", f"{efficiency:.1f}%")
            st.progress(min(efficiency / 100, 1.0))

        except Exception as e:
            st.error(f"Error: {str(e)}")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write("Predict solar power output using weather-based machine learning model.")
    st.write("Model: Random Forest Regressor")
    st.write("Features: Weather, Atmospheric conditions")

    st.header("📌 Instructions")
    st.markdown("""
    1. Fill in all values
    2. Click Predict
    3. View results and efficiency
    """)#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(
    page_title="Solar Power Predictor",
    page_icon="☀️",
    layout="centered"
)

# Load the saved model, scaler, and feature names
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

# Styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f8ff;
        color: black !important;
    }
    .title {
        font-size: 2.5rem;
        font-weight: 600;
        color: black !important;
        text-align: center;
        margin-bottom: 1rem;
    }
    .section {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        color: black !important;
    }
    .stButton>button {
        background-color: #007acc;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #005f99;
    }
    label, .stTextInput, .stNumberInput, .stMetric, .stSelectbox, .stSlider, .stCheckbox {
        color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Prediction function
def predict_power(input_data):
    input_df = pd.DataFrame([input_data], columns=feature_names)
    dummy_row = input_df.copy()
    dummy_row['power_generated'] = 0
    scaled_data = scaler.transform(dummy_row)
    input_scaled = scaled_data[:, :-1]
    log_prediction = model.predict(input_scaled)
    prediction = np.exp(log_prediction)[0]
    return prediction

# Title
st.markdown("<div class='title'>☀️ Solar Power Predictor</div>", unsafe_allow_html=True)

# Input section
with st.container():
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.subheader("Enter Weather & Atmospheric Data")

    col1, col2 = st.columns(2)

    with col1:
        distance_to_solar_noon = st.number_input("Distance to Solar Noon (degrees)", value=0.0)
        temperature = st.number_input("Temperature (°C)", value=25.0)
        wind_speed = st.number_input("Wind Speed (km/h)", value=10.0)
        sky_cover = st.number_input("Sky Cover (oktas, 0-8)", value=2.0)

    with col2:
        wind_direction = st.number_input("Wind Direction (degrees, 0-360)", value=180.0)
        visibility = st.number_input("Visibility (km)", value=10.0)
        humidity = st.number_input("Humidity (%)", value=50.0)
        average_pressure = st.number_input("Average Pressure (hPa)", value=1013.0)

    st.markdown("</div>", unsafe_allow_html=True)

    # Prediction
    if st.button('🔍 Predict Power Generation'):
        try:
            input_data = {
                'distance_to_solar_noon': distance_to_solar_noon,
                'temperature': temperature,
                'wind_direction': wind_direction,
                'wind_speed': wind_speed,
                'sky_cover': sky_cover,
                'visibility': visibility,
                'humidity': humidity,
                'average_pressure': average_pressure
            }

            import streamlit as st

            prediction = predict_power(input_data)
            energy_joules = prediction * 1_000_000 * 3600
            
            # Displaying predicted power generation with black color
            st.success(f"<p style='color:black;'>Predicted Power Generation: {prediction:.2f} MW</p>", unsafe_allow_html=True)
            st.info(f"<p style='color:black;'>Estimated Energy for 1 Hour: {energy_joules:,.0f} J</p>", unsafe_allow_html=True)
            
            # Efficiency gauge
            efficiency = (prediction / 10) * 100
            st.metric("System Efficiency", f"{efficiency:.1f}%", label_visibility="collapsed")
            
            # Progress bar with black color
            st.markdown(f"<div style='height: 10px; background-color: black;'></div>", unsafe_allow_html=True)
            st.progress(min(efficiency / 100, 1.0))


        except Exception as e:
            st.error(f"Error: {str(e)}")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write("Predict solar power output using weather-based machine learning model.")
    st.write("Model: Random Forest Regressor")
    st.write("Features: Weather, Atmospheric conditions")

    st.header("📌 Instructions")
    st.markdown("""
    1. Fill in all values
    2. Click Predict
    3. View results and efficiency
    """)
