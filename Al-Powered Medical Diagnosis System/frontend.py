import streamlit as st
import numpy as np
import pickle
import os

# Define datasets and their respective feature names
datasets = {
    "Diabetes": {
        "file_prefix": "diabetes",
        "features": ["Pregnancies", "Glucose Level (mg/dL)", "Blood Pressure (mm Hg)",
                     "Skin Thickness (mm)", "Insulin Level (µU/mL)", "BMI",
                     "Diabetes Pedigree Function", "Age (years)"]
    },
    "Heart Disease": {
        "file_prefix": "heart_disease",
        "features": ["Age", "Sex", "Chest Pain Type", "Resting Blood Pressure",
                     "Cholesterol", "Fasting Blood Sugar", "Resting ECG",
                     "Max Heart Rate", "Exercise-Induced Angina", "Oldpeak",
                     "Slope", "Number of Major Vessels", "Thal"]
    },
    "Hypothyroid": {
        "file_prefix": "hypothyroid",
        "features": ["Age", "TSH", "T3", "TT4", "T4U", "FTI"]
    },
    "Parkinson": {
        "file_prefix": "parkinson",
        "features": ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)",
                     "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
                     "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3",
                     "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR",
                     "RPDE", "DFA", "Spread1", "Spread2", "D2", "PPE"]
    },
    "Lung Cancer": {
        "file_prefix": "lung_cancer",
        "features": ["Age", "Gender", "Smoking", "Yellow Fingers", "Anxiety",
                     "Peer Pressure", "Chronic Disease", "Fatigue", "Allergy",
                     "Wheezing", "Alcohol Consumption", "Coughing", "Shortness of Breath",
                     "Swallowing Difficulty", "Chest Pain"]
    }
}

# Streamlit UI
st.title("Medical Disease Prediction App")

# Select dataset
dataset_name = st.selectbox("Select Disease Model", list(datasets.keys()))

# Load model and scaler
file_prefix = datasets[dataset_name]["file_prefix"]
model_path = f"{file_prefix}_model.pkl"
scaler_path = f"{file_prefix}_scaler.pkl"

# Check if model and scaler exist
if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    st.error(f"Error: Model files for {dataset_name} not found. Train the model first!")
    st.stop()

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

# Input Fields for Features
st.subheader(f"Enter values for {dataset_name} prediction:")

user_inputs = []
for feature in datasets[dataset_name]["features"]:
    user_input = st.number_input(feature, value=0.0 if "Age" not in feature else 30.0)
    user_inputs.append(user_input)

# Predict Button
if st.button("Predict"):
    # Prepare the input data
    input_data = np.array([user_inputs])

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)

    # Display Result
    if prediction[0] == 1:
        st.error(f"The model predicts that the person **has {dataset_name.lower()}**.")
    else:
        st.success(f"The model predicts that the person **does not have {dataset_name.lower()}**.")
