import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# Load model and scaler
model = tf.keras.models.load_model("shap_attention_model.h5")
scaler = joblib.load("parkinsons_scaler.pkl")

# Feature names
features = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
    'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP',
    'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer',
    'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
    'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
    'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
]

# Sample test cases
sample_pd = [
    119.992, 157.302, 74.997, 0.00784, 0.00007, 0.00370,
    0.00554, 0.01109, 0.04374, 0.426, 0.02182, 0.03130,
    0.02971, 0.06545, 0.02211, 21.033, 0.414783, 0.815285,
    -4.813031, 0.266482, 2.301442, 0.284654
]

sample_healthy = [
    141.871, 155.365, 116.632, 0.00244, 0.00001, 0.00117,
    0.00172, 0.00351, 0.02724, 0.259, 0.01318, 0.01728,
    0.01667, 0.03955, 0.00334, 25.703, 0.256570, 0.708961,
    -6.199505, 0.196594, 1.405554, 0.084280
]

# UI
st.title("🎙️ Parkinson's Disease Detection (SHAP-Attention Enhanced)")
st.markdown("Enter the 22 voice features below or use sidebar autofill options.")

# Sidebar buttons
autofill = [0.0] * len(features)

if st.sidebar.button("🧠 Auto-Fill Parkinson's Sample"):
    autofill = sample_pd
elif st.sidebar.button("✅ Auto-Fill Healthy Sample"):
    autofill = sample_healthy

# Input form
input_data = []
with st.form("input_form"):
    for i, feat in enumerate(features):
        val = st.number_input(f"{feat}", value=autofill[i], format="%.5f")
        input_data.append(val)
    submitted = st.form_submit_button("Predict")

# Prediction
if submitted:
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    # If SHAP weights are saved, re-apply them. Else use model directly.
    prediction = model.predict(input_scaled)[0][0]

    if prediction > 0.5:
        st.error("🧠 Parkinson's Disease Detected")
    else:
        st.success("✅ Healthy Voice Detected")
