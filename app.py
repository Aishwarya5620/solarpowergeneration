
import streamlit as st
import pickle
import numpy as np

# Load the trained model, scaler, and feature list
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('features.pkl', 'rb') as f:
    features = pickle.load(f)

st.title("☀️ Solar Power Generation Predictor")

st.write("Enter the values for each environmental feature:")

user_input = []
for feat in features:
    val = st.number_input(f"{feat}", value=0.0)
    user_input.append(val)

if st.button("Predict"):
    try:
        input_array = np.array(user_input).reshape(1, -1)
        scaled_input = scaler.transform(input_array)
        prediction = model.predict(scaled_input)
        st.success(f"Predicted Power Generation: {prediction[0]:,.2f} Joules")
    except Exception as e:
        st.error(f"Error: {e}")
