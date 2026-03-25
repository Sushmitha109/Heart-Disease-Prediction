import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model and scaler
model = pickle.load(open("model/model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

# Title
st.title("❤️ Heart Disease Risk Predictor")
st.markdown("### Enter patient clinical details below")

# Sidebar
st.sidebar.header("About")
st.sidebar.write(
    "This Machine Learning app predicts the risk of heart disease based on patient health parameters."
)

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", value=120)
chol = st.number_input("Cholesterol", value=200)
fbs = st.selectbox("Fasting Blood Sugar >120 (1 = Yes, 0 = No)", [0, 1])
restecg = st.selectbox("Rest ECG (0-2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate", value=150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("Oldpeak", value=1.0)
slope = st.selectbox("Slope (0-2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thal (1-3)", [1, 2, 3])

# Prediction
if st.button("Predict"):

    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict probability
    prob = model.predict_proba(input_scaled)[0][1]

    st.subheader("Result")

    if prob > 0.5:
        st.error(f"⚠️ High Risk of Heart Disease ({round(prob*100,2)}%)")
    else:
        st.success(f"✅ Low Risk of Heart Disease ({round((1-prob)*100,2)}%)")

    # Show confidence
    st.write(f"Prediction Confidence: {round(prob*100,2)}%")

# Disclaimer
st.info("⚠️ This is a predictive model and not a medical diagnosis.")