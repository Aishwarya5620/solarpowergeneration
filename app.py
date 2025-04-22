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

# CSS Styling
st.set_page_config(page_title="Solar Power Predictor", layout="centered")

st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #004e92, #000428);
        background-attachment: fixed;
        font-family: 'Segoe UI', sans-serif;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem auto;
        max-width: 700px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    }

    h1, h3 {
        color: #ffffff;
        text-align: center;
        text-shadow: 1px 1px 3px #000;
    }

    .stButton > button {
        background-color: #1e88e5;
        color: white;
        padding: 0.6em 1.5em;
        border-radius: 8px;
        font-weight: bold;
        transition: 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #1565c0;
        transform: scale(1.05);
    }

    .stNumberInput input {
        border-radius: 8px;
        padding: 0.5rem;
        background-color: rgba(255, 255, 255, 0.8);
    }

    .stSuccess {
        background-color: rgba(255, 255, 255, 0.3);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# 👇🏽 Start glass container only from here (after CSS is loaded)
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

# Title
st.markdown("## 🌊 Solar Power Generation Predictor", unsafe_allow_html=True)
st.markdown("### 🔧 Enter environmental feature values below", unsafe_allow_html=True)

# Input layout
user_input = []
cols = st.columns(2)
for i, feat in enumerate(features):
    with cols[i % 2]:
        val = st.number_input(f"**{feat}**", value=0.0, format="%.2f")
        user_input.append(val)

# Predict Button
if st.button("🔵 Predict"):
    try:
        input_array = np.array(user_input).reshape(1, -1)
        scaled_input = scaler.transform(input_array)
        prediction = model.predict(scaled_input)
        st.success(f"⚡ Predicted Power Generation: **{prediction[0]:,.2f} Joules**")
    except Exception as e:
        st.error(f"❌ Error: {e}")

# End glass card
st.markdown("</div>", unsafe_allow_html=True)
