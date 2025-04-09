import streamlit as st
import pandas as pd
import joblib

# Load the scaler and model
scaler = joblib.load("final_scaler.pkl")
model = joblib.load("final_rf_model.pkl")

st.set_page_config(page_title="Cancer Risk Predictor", layout="centered")
st.title("🔬 Cancer Risk Predictor")
st.write("Fill in the patient details below to assess cancer risk using a trained ML model.")

# Input form
with st.form("input_form"):
    age = st.slider("Age", 20, 80, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    bmi = st.slider("BMI", 15.0, 45.0, 22.5)
    smoking = st.selectbox("Smoking", ["Yes", "No"])
    genetic_risk = st.selectbox("Genetic Risk", ["Yes", "No"])
    activity = st.slider("Physical Activity Level (1-10)", 1, 10, 5)
    alcohol_intake = st.slider("Alcohol Intake Level (1-10)", 1, 10, 5)
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
            1 if genetic_risk == "Yes" else 0,
            activity,
            alcohol_intake,
            1 if cancer_history == "Yes" else 0
        ]], columns=[
            'Age', 'Gender', 'BMI', 'Smoking',
            'GeneticRisk', 'PhysicalActivity',
            'AlcoholIntake', 'FamilyHistory'
        ])

        # Debug prints (optional)
        # st.write("Input Columns:", input_data.columns.tolist())
        # st.write("Input Data:", input_data)

        # Scale and predict
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]

        if prediction == 1:
            st.error("⚠️ High Risk of Cancer Detected")
        else:
            st.success("✅ Low Risk of Cancer Detected")

    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")




# Temporarily print these
st.write("Expected by Scaler:", list(scaler.feature_names_in_))
st.write("Given to Model:", input_data.columns.tolist())
