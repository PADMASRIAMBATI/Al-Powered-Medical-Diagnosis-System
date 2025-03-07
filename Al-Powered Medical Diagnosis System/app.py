from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Define datasets and their respective feature names
datasets = {
    "diabetes": {
        "file_prefix": "diabetes",
        "features": ["Pregnancies", "Glucose Level", "Blood Pressure", "Skin Thickness",
                     "Insulin", "BMI", "Diabetes Pedigree Function", "Age"]
    },
    "heart_disease": {
        "file_prefix": "heart_disease",
        "features": ["Age", "Sex", "Chest Pain Type", "Resting Blood Pressure",
                     "Cholesterol", "Fasting Blood Sugar", "Resting ECG",
                     "Max Heart Rate", "Exercise-Induced Angina", "Oldpeak",
                     "Slope", "Number of Major Vessels", "Thal"]
    },
    "hypothyroid": {
        "file_prefix": "hypothyroid",
        "features": ["Age", "TSH", "T3", "TT4", "T4U", "FTI"]
    },
    "parkinson": {
        "file_prefix": "parkinson",
        "features": ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)",
                     "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
                     "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3",
                     "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR",
                     "RPDE", "DFA", "Spread1", "Spread2", "D2", "PPE"]
    },
    "lung_cancer": {
        "file_prefix": "lung_cancer",
        "features": ["Age", "Gender", "Smoking", "Yellow Fingers", "Anxiety",
                     "Peer Pressure", "Chronic Disease", "Fatigue", "Allergy",
                     "Wheezing", "Alcohol Consumption", "Coughing", "Shortness of Breath",
                     "Swallowing Difficulty", "Chest Pain"]
    }
}

@app.route("/")
def home():
    return "Medical Prediction API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json  # Expecting a JSON payload
        dataset_name = data.get("dataset")  # Disease type (e.g., "diabetes")
        
        if dataset_name not in datasets:
            return jsonify({"error": "Invalid dataset. Choose from: " + ", ".join(datasets.keys())})

        # Load correct model and scaler
        file_prefix = datasets[dataset_name]["file_prefix"]
        model_path = f"{file_prefix}_model.pkl"
        scaler_path = f"{file_prefix}_scaler.pkl"

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return jsonify({"error": f"Model files for {dataset_name} not found. Train the model first!"})

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        # Extract features
        features = np.array(data["features"]).reshape(1, -1)

        # Scale the input features
        scaled_features = scaler.transform(features)

        # Make prediction
        prediction = model.predict(scaled_features)

        return jsonify({"prediction": int(prediction[0])})  # Return JSON response

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
