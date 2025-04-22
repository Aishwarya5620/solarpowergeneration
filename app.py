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

# Inject CSS styling
st.set_page_config(page_title="Solar Power Predictor", layout="centered")
st.markdown("""
    <style>
    body {
        background-image: url('https://images.unsplash.com/photo-1603297631958-e930c7e8bb3a?auto=format&fit=crop&w=1950&q=80');
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
        background-position: center;
        font-family: 'Segoe UI', sans-serif;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 3rem auto;
        max-width: 750px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    }
    h1, h3 {
        color: #ffffff;
        text-align: center;
        text-shadow: 2px 2px 4px #000;
    }
    .stButton > button {
        background-color: #ff9800;
        color: white;
        padding: 0.7em 1.6em;
        border-radius: 10px;
        font-weight: bold;
        font-size: 16px;
        transition: 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #e65100;
        transform: scale(1.05);
        cursor: pointer;
    }
    .stNumberInput input {
        border-radius: 10px;
        padding: 0.6rem;
        background-color: rgba(255, 255, 255, 0.9);
        border: 1px solid #ccc;
        transition: 0.3s;
    }
    .stNumberInput input:hover {
        border: 1px solid #ff9800;
    }
    .stSuccess {
        background-color: rgba(255, 255, 255, 0.35);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        font-weight: bold;
        font-size: 18px;
        box-shadow: 0 0 15px rgba(255, 255, 255, 0.1);
    }
    footer {
        color: white;
        text-align: center;
        margin-top: 30px;
        font-size: 14px;
        opacity: 0.7;
    }
    </style>
""", unsafe_allow_html=True)

# Glass card container
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

# Title
st.markdown("## 🌞 Solar Power Generation Predictor", unsafe_allow_html=True)
st.markdown("### Enter environmental feature values below", unsafe_allow_html=True)

# Input layout: 2 columns
user_input = []
cols = st.columns(2)
for i, feat in enumerate(features):
    with cols[i % 2]:
        val = st.number_input(f"**{feat}**", value=0.0, format="%.2f")
        user_input.append(val)

# Prediction
if st.button("🚀 Predict"):
    try:
        input_array = np.array(user_input).reshape(1, -1)
        scaled_input = scaler.transform(input_array)
        prediction = model.predict(scaled_input)
        st.success(f"⚡ Predicted Power Generation: **{prediction[0]:,.2f} Joules**")
    except Exception as e:
        st.error(f"❌ Error: {e}")

# Close glass card
st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<footer>Made with ☀️ by Your Name</footer>", unsafe_allow_html=True)
