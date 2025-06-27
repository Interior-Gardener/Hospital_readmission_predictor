import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from PIL import Image
import os

# Page config
st.set_page_config(
    page_title="Hospital Readmission Predictor",
    page_icon="🏥",
    layout="centered"
)

# Load model and column info
model = joblib.load("readmission_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# App title
st.title("🏥 Hospital Readmission Predictor")
st.markdown("Use this AI tool to estimate if a diabetic patient is at **risk of hospital readmission**.")

# Sidebar instructions
with st.sidebar:
    st.header("📝 How to Use")
    st.markdown("""
    1. Enter patient info below.
    2. Click **Predict Readmission**.
    3. View result and contributing factors.
    """)

# Patient Input Form
st.subheader("🧍 Patient Information")

# def get_user_input():
#     age = st.selectbox("Age", ['[70-80)', '[60-70)', '[50-60)', '[80-90)', '[40-50)', '[30-40)', '[90-100)', '[20-30)'])
#     gender = st.selectbox("Gender", ['Male', 'Female'])
#     admission_type_id = st.selectbox("Admission Type", ['Emergency', 'Urgent', 'Elective'])
#     time_in_hospital = st.slider("🏥 Time in Hospital (days)", 1, 14, 3)
#     num_lab_procedures = st.slider("🔬 Number of Lab Procedures", 0, 100, 40)

#     user_data = {
#         'age': age,
#         'gender': gender,
#         'admission_type_id': admission_type_id,
#         'time_in_hospital': time_in_hospital,
#         'num_lab_procedures': num_lab_procedures
#     }

def get_user_input():
    age = st.selectbox("Age", ['[70-80)', '[60-70)', '[50-60)', '[80-90)', '[40-50)', '[30-40)', '[90-100)', '[20-30)'])
    gender = st.selectbox("Gender", ['Male', 'Female'])
    admission_type_id = st.selectbox("Admission Type", ['Emergency', 'Urgent', 'Elective'])
    discharge_disposition_id = st.selectbox("Discharge Disposition", ['Home', 'Transferred', 'Expired'])
    admission_source_id = st.selectbox("Admission Source", ['Physician Referral', 'Emergency Room', 'Transfer'])
    time_in_hospital = st.slider("🏥 Time in Hospital (days)", 1, 14, 3)
    num_lab_procedures = st.slider("🔬 Number of Lab Procedures", 0, 100, 40)
    number_inpatient = st.slider("🏥 Prior Inpatient Visits", 0, 20, 0)
    number_diagnoses = st.slider("🧪 Number of Diagnoses", 1, 16, 9)
    max_glu_serum = st.selectbox("Max Glucose Serum", ['None', 'Norm', '>200', '>300'])
    A1Cresult = st.selectbox("A1C Result", ['None', 'Norm', '>7', '>8'])
    change = st.selectbox("Medication Change", ['No', 'Ch'])
    diabetesMed = st.selectbox("Diabetes Medication", ['Yes', 'No'])

    return pd.DataFrame([{
        'age': age,
        'gender': gender,
        'admission_type_id': admission_type_id,
        'discharge_disposition_id': discharge_disposition_id,
        'admission_source_id': admission_source_id,
        'time_in_hospital': time_in_hospital,
        'num_lab_procedures': num_lab_procedures,
        'number_inpatient': number_inpatient,
        'number_diagnoses': number_diagnoses,
        'max_glu_serum': max_glu_serum,
        'A1Cresult': A1Cresult,
        'change': change,
        'diabetesMed': diabetesMed
    }])


    return pd.DataFrame([user_data])

input_df = get_user_input()

# Convert user input to encoded model input
encoded_input = pd.get_dummies(input_df)
encoded_input = encoded_input.reindex(columns=model_columns, fill_value=0)

# Display input summary
st.subheader("📋 Patient Summary")
st.write(input_df)

# Prediction button
if st.button("🔍 Predict Readmission"):
    prediction = model.predict(encoded_input)[0]
    probability = model.predict_proba(encoded_input)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Readmission ({probability:.2%})")
    else:
        st.success(f"✅ Low Risk of Readmission ({(1 - probability):.2%})")

    # Display SHAP summary plot if available
    if os.path.exists("shap_summary_plot.png"):
        st.subheader("📊 Model Explanation (SHAP)")
        image = Image.open("shap_summary_plot.png")
        st.image(image, caption="Top features influencing readmission risk", use_column_width=True)
