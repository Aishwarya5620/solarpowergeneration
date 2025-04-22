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

# Custom CSS for background and style
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(to right, #d9a7c7, #fffcdc);
    }
    .main {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 3rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    h1 {
        color: #e65100;
        text-align: center;
        font-family: 'Trebuchet MS', sans-serif;
    }
    .stButton>button {
        background-color: #ff9800;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
        padding: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Container for the app
with st.container():
    st.markdown('<div class="main">', unsafe_allow_html=True)

    st.title("☀️ Solar Power Generation Predictor")
    st.write("### 🔧 Enter the values for each environmental feature below:")

    user_input = []
    for feat in features:
        val = st.number_input(f"**{feat}**", value=0.0, format="%.2f")
        user_input.append(val)

    if st.button("🔍 Predict"):
        try:
            input_array = np.array(user_input).reshape(1, -1)
            scaled_input = scaler.transform(input_array)
            prediction = model.predict(scaled_input)
            st.success(f"⚡ Predicted Power Generation: **{prediction[0]:,.2f} Joules**")
        except Exception as e:
            st.error(f"❌ Error: {e}")

    st.markdown('</div>', unsafe_allow_html=True)
