import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# Load trained model and scaler
model = tf.keras.models.load_model("shap_attention_model.h5")
scaler = joblib.load("parkinsons_scaler.pkl")

# SHAP feature weights (same as during training)
shap_weights = np.array([
    0.91, 0.87, 0.84, 0.95, 0.78, 0.88,
    0.86, 0.82, 0.89, 0.75, 0.83, 0.81,
    0.80, 0.85, 0.72, 0.90, 0.94, 0.88,
    0.96, 0.92, 0.89, 0.93
])
shap_weights = shap_weights / np.max(shap_weights)

# Feature list
features = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
    'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP',
    'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer',
    'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
    'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
    'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
]

# Parkinson and Healthy test cases
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

# Streamlit UI
st.set_page_config(page_title="SHAP-Attention Parkinson’s Detector", layout="centered")
st.title("🎙️ Parkinson's Disease Detection\n(SHAP-Attention Enhanced)")
st.markdown("Enter 22 voice biomarkers or use **auto-fill** buttons in the sidebar.")

# Sidebar autofill
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
    input_weighted = input_scaled * shap_weights  # Apply SHAP reweighting
    prob = model.predict(input_weighted)[0][0]

    st.markdown("### 🔍 Model Confidence Score")
    st.progress(float(prob))
    st.code(f"🧠 Parkinson's Probability: {prob:.2%}\n✅ Healthy Probability: {(1 - prob):.2%}")

    if prob > 0.5:
        st.error(f"🧠 Parkinson’s Disease Detected (Confidence: {prob:.2%})")
    else:
        st.success(f"✅ Healthy Voice Detected (Confidence: {(1 - prob):.2%})")
