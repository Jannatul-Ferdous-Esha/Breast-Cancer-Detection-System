# import joblib
# import numpy as np
# import joblib
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.preprocessing import StandardScaler
# # Load the model and scaler
# MODEL_PATH = "final_model.pkl"
# SCALER_PATH = "final_scaler.pkl"

# model = joblib.load(MODEL_PATH)
# scaler = joblib.load(SCALER_PATH)

# # Sample input for a Benign (B) case with ID 854941
# sample_input = np.array([[
#     13.03, 18.42, 82.61, 523.8, 0.08983, 0.03766, 0.02562, 0.02923, 0.1467, 0.05863,
#     0.1839, 2.342, 1.17, 14.16, 0.004352, 0.004899, 0.01343, 0.01164, 0.02671, 0.001777,
#     13.3, 22.81, 84.46, 545.9, 0.09701, 0.04619, 0.04833, 0.05013, 0.1987, 0.06169
# ]])

# # Normalize input using the scaler
# sample_scaled = scaler.transform(sample_input)

# # Make a prediction
# prediction = model.predict(sample_scaled)
# result = "Malignant" if prediction == 1 else "Benign"

# print("Prediction:", result)
#       # If it's a DataFrame
import numpy as np
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dropout, BatchNormalization, Flatten, Dense
from tensorflow.keras.models import load_model

# Load model and scaler
model = joblib.load("final_model.pkl")
scaler = joblib.load("final_scaler.pkl")

# --- CNN Feature Extractor (same as in training) ---
def create_cnn_feature_extractor(input_shape):
    cnn = Sequential([
        Conv1D(128, 3, activation='relu', input_shape=input_shape),
        BatchNormalization(), Dropout(0.5),
        Conv1D(64, 3, activation='relu'),
        BatchNormalization(), Dropout(0.5),
        Conv1D(32, 3, activation='relu'),
        BatchNormalization(), Dropout(0.5),
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(), Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5)
    ])
    return cnn

feature_extractor = create_cnn_feature_extractor((30, 1))

# --- Input Sample ---
sample_input = np.array([[  # Benign case
    13.03, 18.42, 82.61, 523.8, 0.08983, 0.03766, 0.02562, 0.02923, 0.1467, 0.05863,
    0.1839, 2.342, 1.17, 14.16, 0.004352, 0.004899, 0.01343, 0.01164, 0.02671, 0.001777,
    13.3, 22.81, 84.46, 545.9, 0.09701, 0.04619, 0.04833, 0.05013, 0.1987, 0.06169
]])

# --- Preprocessing ---
sample_scaled = scaler.transform(sample_input)  # (1, 30)
sample_reshaped = sample_scaled.reshape(sample_scaled.shape[0], sample_scaled.shape[1], 1)  # (1, 30, 1)

# --- Feature Extraction ---
sample_features = feature_extractor.predict(sample_reshaped)  # (1, 64)

# --- Prediction ---
prediction = model.predict(sample_features)
result = "Malignant" if prediction[0] == 1 else "Benign"

print("Prediction:", result)
