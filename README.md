# AI-Powered Medical Diagnosis System

## Overview
The AI-Powered Medical Diagnosis System is designed to predict various diseases using machine learning models. It processes patient data, scales features, and applies trained classification models to provide predictions for diseases such as diabetes, heart disease, hypothyroidism, Parkinson's disease, and lung cancer.

## Features
- **Multi-Disease Prediction**: Supports predictions for diabetes, heart disease, hypothyroidism, Parkinson's disease, and lung cancer.
- **Data Preprocessing**: Includes feature scaling and label encoding for categorical data.
- **Machine Learning Models**: Uses Random Forest Classifier for accurate predictions.
- **Flask API**: Provides an easy-to-use API for medical predictions.
- **Model Training & Persistence**: Saves trained models and scalers for future use.
- **Streamlit UI**: Provides an interactive web-based interface for predictions.

## Prerequisites
Before running the project, ensure you have the following installed:
- Python 3.8+
- Flask
- Pandas
- NumPy
- scikit-learn
- pickle
- Streamlit

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/PADMASRIAMBATI/Al-Powered-Medical-Diagnosis-System.git
   ```
2. Navigate to the project directory:
   ```sh
   cd AI-Powered-Medical-Diagnosis-System
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
### Train the Models
Run the training script to process the datasets and train models:
```sh
python train_models.py
```

### Start the Flask API
Launch the API server:
```sh
python app.py
```
The API will be accessible at `http://127.0.0.1:5000/`.

### Run the Streamlit App
Start the Streamlit web interface:
```sh
streamlit run app.py
```

## API Endpoints
- `GET /` : Check if the API is running.
- `POST /predict` : Submit patient data and receive a disease prediction.

## Contributing
Feel free to submit issues or contribute to improving the project!

## License
This project is open-source and available under the MIT License.

