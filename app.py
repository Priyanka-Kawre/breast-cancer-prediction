import joblib
import pandas as pd
import streamlit as st

# Load the model and scaler
rf_model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit UI
st.title("Breast Cancer 10-Year Mortality Prediction App")

# User Inputs
age = st.number_input("Age at Diagnosis", 20, 100)
tumor_size = st.number_input("Tumor Size (mm)", 0.0, 200.0)
grade = st.selectbox("Neoplasm Histologic Grade", [1, 2, 3])
tumor_stage = st.selectbox("Tumor Stage", ['Stage I', 'Stage II', 'Stage III', 'Stage IV'])
er_status = st.selectbox("ER Status", ["Positive", "Negative"])
pr_status = st.selectbox("PR Status", ["Positive", "Negative"])
her2_status = st.selectbox("HER2 Status", ["Positive", "Negative"])
hormone_therapy = st.selectbox("Hormone Therapy", ["Yes", "No"])
radio_therapy = st.selectbox("Radio Therapy", ["Yes", "No"])

# Map tumor stage to numerical value
tumor_stage_mapping = {
    'Stage I': 1,
    'Stage II': 2,
    'Stage III': 3,
    'Stage IV': 4
}
tumor_stage_numeric = tumor_stage_mapping[tumor_stage]

# Creating input dictionary (Important)
input_dict = {
    'Age at Diagnosis': age,
    'Tumor Size': tumor_size,
    'Neoplasm Histologic Grade': grade,
    'Tumor Stage': tumor_stage_numeric,
    'ER Status_Positive': 1 if er_status == "Positive" else 0,
    'PR Status_Positive': 1 if pr_status == "Positive" else 0,
    'HER2 Status_Positive': 1 if her2_status == "Positive" else 0,
    'Hormone Therapy_Yes': 1 if hormone_therapy == "Yes" else 0,
    'Radio Therapy_Yes': 1 if radio_therapy == "Yes" else 0
}

# Convert input to DataFrame
input_df = pd.DataFrame([input_dict])

# Scale the input
scaled_input = scaler.transform(input_df)

# Prediction button
if st.button("Predict 10-Year Survival"):
    prediction = rf_model.predict(scaled_input)
    if prediction[0] == 1:
        st.success("🎯 Patient Likely to Survive More Than 10 Years")
    else:
        st.error("⚠️ Patient Risk of Mortality Within 10 Years")

