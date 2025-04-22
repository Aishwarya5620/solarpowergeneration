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
        background: rgba(255, 255, 255, 0.2);
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
        text-shadow: 1px 1px 2px #000;
    }
    .stButton > button {
        background-color: #ff9800;
        color: white;
        padding: 0.6em 1.5em;
        border-radius: 8px;
        font-weight: bold;
        transition: 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #e65100;
        transform: scale(1.05);
    }
    .stNumberInput input {
        border-radius: 8px;
        padding: 0.5rem;
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

# Title
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("## 🌞 Solar Power Generation Predictor", unsafe_allow_html=True)
st.markdown("### 🔧 Enter environmental feature values below", unsafe_allow_html=True)

# Input layout: 2 columns with 4 rows = 8 features
user_input = []
cols = st.columns(2)  # Two columns

for i, feat in enumerate(features):
    with cols[i % 2]:  # Alternate between col 0 and col 1
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

st.markdown("</div>", unsafe_allow_html=True)
