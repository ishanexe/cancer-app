import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("final_rf_model.pkl")
scaler = joblib.load("final_scaler.pkl")

st.title("🧬 Cancer Risk Prediction")

# Input form
st.sidebar.header("Patient Info")
age = st.sidebar.slider("Age", 20, 90, 30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
bmi = st.sidebar.slider("BMI", 15.0, 45.0, 22.0)
smoking = st.sidebar.selectbox("Smoking Habit", ["No", "Yes"])
genetic = st.sidebar.selectbox("Genetic Disorder", ["No", "Yes"])
activity = st.sidebar.slider("Physical Activity Level (1-10)", 1, 10, 5)
alcohol = st.sidebar.slider("Alcohol Intake Level (1-10)", 1, 10, 3)
cancer_history = st.sidebar.selectbox("Family Cancer History", ["No", "Yes"])

# Convert to model input
input_data = pd.DataFrame([[
    age,
    1 if gender == "Male" else 0,
    bmi,
    1 if smoking == "Yes" else 0,
    1 if genetic == "Yes" else 0,
    activity,
    alcohol,
    1 if cancer_history == "Yes" else 0
]], columns=["Age", "Gender", "BMI", "Smoking", "GeneticDisorder", "PhysicalActivity", "Alcohol", "CancerHistory"])

# Scale input
scaled_input = scaler.transform(input_data)

# Predict
if st.button("Predict Cancer Risk"):
    prediction = model.predict(scaled_input)[0]
    if prediction == 1:
        st.error("⚠️ High Risk of Cancer Detected")
    else:
        st.success("✅ Low Risk of Cancer Detected")
