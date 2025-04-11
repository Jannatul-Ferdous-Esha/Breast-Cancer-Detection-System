# App_post/management/commands/compare_models.py

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = "Compares the performance of two models"

    def handle(self, *args, **kwargs):
        # Load the old and new models and scalers
        old_model_path = 'path_to_model/model.pkl'  # Replace with the correct path
        new_model_path = 'path_to_model/new_model.pkl'  # Replace with the correct path
        old_scaler_path = 'path_to_model/scaler.pkl'  # Replace with the correct path
        new_scaler_path = 'path_to_model/new_scaler.pkl'  # Replace with the correct path

        # Load the models and scalers
        old_model = joblib.load(old_model_path)
        new_model = joblib.load(new_model_path)
        old_scaler = joblib.load(old_scaler_path)
        new_scaler = joblib.load(new_scaler_path)

        # Load the dataset (make sure you provide the correct path)
        data = pd.read_csv("data.csv")
        data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
        data = data.drop(columns=['id', 'Unnamed: 32'], errors='ignore')

        X = data.drop('diagnosis', axis=1)
        y = data['diagnosis']

        # Preprocess the features with both scalers
        X_scaled_old = old_scaler.transform(X)
        X_scaled_new = new_scaler.transform(X)

        # Evaluate both models
        y_pred_old = old_model.predict(X_scaled_old)
        y_pred_new = new_model.predict(X_scaled_new)

        # Calculate accuracy and classification report for both models
        accuracy_old = accuracy_score(y, y_pred_old)
        accuracy_new = accuracy_score(y, y_pred_new)

        report_old = classification_report(y, y_pred_old)
        report_new = classification_report(y, y_pred_new)

        self.stdout.write(f"Old Model Accuracy: {accuracy_old}")
        self.stdout.write(f"New Model Accuracy: {accuracy_new}")
        self.stdout.write(f"\nOld Model Classification Report:\n{report_old}")
        self.stdout.write(f"\nNew Model Classification Report:\n{report_new}")
