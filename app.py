import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Breast Cancer Detector", layout="wide")

st.title("🩺 Breast Cancer Prediction App")
st.write("Enter patient details below to predict whether the tumor is **Malignant (M)** or **Benign (B)**.")

# Load trained model
model = joblib.load("model.pkl")

# 31 features including 'id'
features = [
    'id',
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
    'fractal_dimension_se', 'radius_worst', 'texture_worst',
    'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst',
    'symmetry_worst', 'fractal_dimension_worst'
]

st.subheader("🔢 Enter Features")

# Split into 2 columns for input (16 + 15)
col1, col2 = st.columns(2)

values = []

with col1:
    st.markdown("### Column 1")
    for feature in features[:16]:
        val = st.number_input(feature.replace("_", " ").title(), value=0.0)
        values.append(val)

with col2:
    st.markdown("### Column 2")
    for feature in features[16:]:
        val = st.number_input(feature.replace("_", " ").title(), value=0.0)
        values.append(val)

# Convert to array
input_data = np.array(values).reshape(1, -1)

# Predict
if st.button("🔍 Detect"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    

    if prediction == 1:
        st.error(f"### Result: **Malignant (M)**\nProbability: {probability:.2f}")
    else:
        st.success(f"### Result: **Benign (B)**\nProbability: {probability:.2f}")
