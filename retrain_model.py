# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import classification_report, accuracy_score
# import joblib

# # Load the dataset (ensure the correct path to the data.csv)
# data = pd.read_csv('data.csv')

# # Map 'M' and 'B' to 1 and 0 (Malignant and Benign)
# data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
# data = data.drop(columns=['id', 'Unnamed: 32'], errors='ignore')  # Drop any unnecessary columns

# # Split the features and target
# X = data.drop('diagnosis', axis=1)
# y = data['diagnosis']

# # Standardize the features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Split the data into training and testing sets (80% train, 20% test)
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# # Train the RandomForestClassifier (you can use a different model if needed)
# model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
# model.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = model.predict(X_test)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# report = classification_report(y_test, y_pred)

# print(f"Accuracy: {accuracy}")
# print("Classification Report:")
# print(report)

# # Save the model and scaler to disk
# joblib.dump(model, 'new_model.pkl')
# joblib.dump(scaler, 'new_scaler.pkl')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dropout, BatchNormalization, Flatten, Dense
import optuna
import joblib

# Load and preprocess dataset
df = pd.read_csv("data.csv")
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
df.drop(columns=['id', 'Unnamed: 32'], errors='ignore', inplace=True)

X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)

# CNN-based feature extractor
def create_cnn_feature_extractor(input_shape):
    model = Sequential([
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
    return model

# Reshape for CNN
X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

cnn_model = create_cnn_feature_extractor((X_train.shape[1], 1))
feature_extractor = Sequential(cnn_model.layers[:-1])
X_train_features = feature_extractor.predict(X_train_reshaped)
X_test_features = feature_extractor.predict(X_test_reshaped)

# Hyperparameter optimization with Optuna
def objective(trial):
    xgb_params = {
        'n_estimators': trial.suggest_int('xgb_n_estimators', 50, 200),
        'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
        'learning_rate': trial.suggest_float('xgb_learning_rate', 1e-3, 1e-1)
    }
    rf_params = {
        'n_estimators': trial.suggest_int('rf_n_estimators', 50, 200),
        'max_depth': trial.suggest_int('rf_max_depth', 5, 20)
    }

    xgb = XGBClassifier(**xgb_params)
    rf = RandomForestClassifier(**rf_params)
    lr = LogisticRegression()

    stacking_model = StackingClassifier(
        estimators=[('xgb', xgb), ('rf', rf)],
        final_estimator=lr
    )

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in skf.split(X_train_features, y_train):
        X_tr, X_val = X_train_features[train_idx], X_train_features[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        stacking_model.fit(X_tr, y_tr)
        preds = stacking_model.predict(X_val)
        scores.append(accuracy_score(y_val, preds))

    return np.mean(scores)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=15)

# Final model using best params
best = study.best_params
xgb_final = XGBClassifier(n_estimators=best['xgb_n_estimators'], max_depth=best['xgb_max_depth'], learning_rate=best['xgb_learning_rate'])
rf_final = RandomForestClassifier(n_estimators=best['rf_n_estimators'], max_depth=best['rf_max_depth'])
stacking_final = StackingClassifier(estimators=[('xgb', xgb_final), ('rf', rf_final)], final_estimator=LogisticRegression())

# Train and evaluate
stacking_final.fit(X_train_features, y_train)
preds = stacking_final.predict(X_test_features)
print("Classification Report:\n", classification_report(y_test, preds))
print("Accuracy:", accuracy_score(y_test, preds))

# Save model and scaler
joblib.dump(stacking_final, "final_model.pkl")
joblib.dump(scaler, "final_scaler.pkl")
