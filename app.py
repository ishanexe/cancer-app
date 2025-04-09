import streamlit as st
import pandas as pd
import joblib

# Load the scaler and model
scaler = joblib.load("final_scaler.pkl")
model = joblib.load("final_rf_model.pkl")

# Set the page title and description
st.set_page_config(page_title="Cancer Risk Predictor", layout="centered")
st.title("🔬 Cancer Risk Predictor")
st.write("Fill in the patient details below to assess cancer risk using a trained machine learning model.")

# Input form
with st.form("input_form"):
    age = st.slider("Age", 20, 80, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    bmi = st.slider("BMI", 15.0, 45.0, 22.5)
    smoking = st.selectbox("Smoking", ["Yes", "No"])
    genetic = st.selectbox("Genetic Disorder", ["Yes", "No"])
    activity = st.slider("Physical Activity Level (1-10)", 1, 10, 5)
    alcohol = st.slider("Alcohol Intake Level (1-10)", 1, 10, 5)
    cancer_history = st.selectbox("Family History of Cancer", ["Yes", "No"])

    submitted = st.form_submit_button("Predict")

# Predict
if submitted:
    try:
        input_data = pd.DataFrame([[
            age,
            1 if gender == "Male" else 0,
            bmi,
            1 if smoking == "Yes" else 0,
            1 if genetic == "Yes" else 0,
            activity,
            alcohol,
            1 if cancer_history == "Yes" else 0
        ]], columns=['Age', 'Gender', 'BMI', 'Smoking', 'GeneticDisorder', 'PhysicalActivity', 'Alcohol', 'CancerHistory'])

        # Scale and predict
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]

        if prediction == 1:
            st.error("⚠️ High Risk of Cancer Detected")
        else:
            st.success("✅ Low Risk of Cancer Detected")
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
