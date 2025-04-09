code = """
import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("final_rf_model.pkl")
scaler = joblib.load("final_scaler.pkl")

st.title("🧬 Cancer Risk Prediction App")
st.write("This app predicts cancer risk based on health and lifestyle inputs.")

# Input fields
age = st.slider("Age", 20, 80, 30)
gender = st.radio("Gender", ["Male", "Female"])
bmi = st.slider("BMI", 15.0, 45.0, 22.0)
smoking = st.radio("Smoking", ["No", "Yes"])
genetic = st.radio("Genetic Disorder", ["No", "Yes"])
activity = st.slider("Physical Activity (1-10)", 1, 10, 5)
alcohol = st.slider("Alcohol Intake (1-10)", 1, 10, 3)
history = st.radio("Family Cancer History", ["No", "Yes"])

# Encode & transform input
input_data = {
    "Age": age,
    "Gender": 1 if gender == "Male" else 0,
    "BMI": bmi,
    "Smoking": 1 if smoking == "Yes" else 0,
    "GeneticDisorder": 1 if genetic == "Yes" else 0,
    "PhysicalActivity": activity,
    "Alcohol": alcohol,
    "CancerHistory": 1 if history == "Yes" else 0,
}

input_df = pd.DataFrame([input_data])
scaled_input = scaler.transform(input_df)

# Predict
if st.button("Predict"):
    prediction = model.predict(scaled_input)
    if prediction[0] == 1:
        st.error("⚠️ High Risk of Cancer")
    else:
        st.success("✅ Low Risk of Cancer")
"""
with open("app.py", "w") as f:
    f.write(code)
