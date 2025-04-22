#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

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

# Page configuration
st.set_page_config(page_title="Solar Power Predictor", page_icon="☀️", layout="centered")

st.title("☀️ Solar Power Generation Predictor")
st.markdown("Enter environmental and weather parameters to estimate solar power output.")

# Input fields
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        distance_to_solar_noon = st.number_input("Distance to Solar Noon (degrees)", value=0.0)
        temperature = st.number_input("Temperature (°C)", value=25.0)
        wind_speed = st.number_input("Wind Speed (km/h)", value=10.0)
        sky_cover = st.number_input("Sky Cover (oktas, 0-8)", min_value=0.0, max_value=8.0, value=2.0)

    with col2:
        wind_direction = st.number_input("Wind Direction (degrees, 0-360)", min_value=0.0, max_value=360.0, value=180.0)
        visibility = st.number_input("Visibility (km)", value=10.0)
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
        average_pressure = st.number_input("Average Pressure (hPa)", value=1013.0)

    submitted = st.form_submit_button("Predict")

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

# Display prediction
if submitted:
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
        st.success(f"Predicted Power Generation: {prediction:.2f} MW")
        st.info(f"Estimated Energy for 1 hour: {prediction * 1_000_000 * 3600:,.0f} J")

        if prediction < 2:
            st.warning("Status: Low Power Generation")
        elif prediction < 5:
            st.info("Status: Moderate Power Generation")
        else:
            st.success("Status: High Power Generation")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Sidebar info
with st.sidebar:
    st.header("About")
    st.write("This app predicts solar power generation using a machine learning model.")
    st.write("Model: Random Forest Regressor")
    st.write("Trained on real environmental and solar generation data.")
