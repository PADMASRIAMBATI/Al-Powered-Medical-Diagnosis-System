import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np

# Define dataset filenames
datasets = {
    "diabetes": "diabetes_data.csv",
    "heart_disease": "heart_disease_data.csv",
    "hypothyroid": "hypothyroid.csv",
    "parkinson": "parkinson_data.csv",
    "lung_cancer": "survey lung cancer.csv"
}

# Loop through each dataset
for name, file in datasets.items():
    try:
        print(f"\nProcessing dataset: {name}")

        # Load dataset
        df = pd.read_csv(file)

        # Encode categorical features
        label_encoders = {}
        for col in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le  # Store encoder for future decoding

        # Split dataset
        X = df.iloc[:, :-1]  # All columns except the last (features)
        y = df.iloc[:, -1]  # Last column (target)

        # Handle continuous target values for Parkinson's dataset
        if name == "parkinson" and y.dtype in [np.float64, np.int64]:
            threshold = y.mean()  # Convert to binary classification
            y = (y >= threshold).astype(int)

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Feature scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train model (Classifier for discrete values, Regressor for continuous)
        if name == "parkinson" and len(set(y_train)) > 2:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)

        model.fit(X_train, y_train)

        # Save model, scaler, and encoders
        with open(f"{name}_model.pkl", "wb") as f:
            pickle.dump(model, f)

        with open(f"{name}_scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

        if label_encoders:
            with open(f"{name}_encoders.pkl", "wb") as f:
                pickle.dump(label_encoders, f)

        print(f"✅ Saved: {name}_model.pkl, {name}_scaler.pkl, {name}_encoders.pkl (if applicable)")

    except Exception as e:
        print(f"❌ Error processing {name}: {e}")

print("\n🎉 All models processed successfully!")
