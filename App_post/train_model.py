# # # train_model.py
# # import pandas as pd
# # import numpy as np
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.model_selection import train_test_split
# # from sklearn.preprocessing import StandardScaler
# # from sklearn.metrics import classification_report, accuracy_score
# # import joblib

# # # Load data
# # df = pd.read_csv("data.csv")
# # df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
# # df = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')

# # X = df.drop('diagnosis', axis=1)
# # y = df['diagnosis']

# # # Preprocess
# # scaler = StandardScaler()
# # X_scaled = scaler.fit_transform(X)

# # # Train-test split
# # X_train, X_test, y_train, y_test = train_test_split(
# #     X_scaled, y, test_size=0.2, stratify=y, random_state=42
# # )

# # # Train model
# # model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
# # model.fit(X_train, y_train)

# # # Evaluate
# # y_pred = model.predict(X_test)
# # print("Accuracy:", accuracy_score(y_test, y_pred))
# # print(classification_report(y_test, y_pred))

# # # Save model
# # joblib.dump(model, "model.pkl")
# # joblib.dump(scaler, "scaler.pkl")  # Save scaler for input normalization
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import classification_report, accuracy_score
# import joblib

# # Load data
# df = pd.read_csv("data.csv")
# df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
# df = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')

# X = df.drop('diagnosis', axis=1)
# y = df['diagnosis']

# # Preprocess
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X_scaled, y, test_size=0.2, stratify=y, random_state=42
# )

# # Train model
# model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
# model.fit(X_train, y_train)

# # Evaluate
# y_pred = model.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))

# # Save model and scaler
# joblib.dump(model, "new_model.pkl")
# joblib.dump(scaler, "new_scaler.pkl")  # Save scaler for input normalization
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# Load the old and new models and scalers
old_model_path = 'App_post/model.pkl'  # Replace with your actual model path
new_model_path = 'App_post/new_model.pkl'  # Replace with your actual model path
old_scaler_path = 'App_post/scaler.pkl'  # Replace with your actual scaler path
new_scaler_path = 'App_post/new_scaler.pkl'  # Replace with your actual scaler path

old_model = joblib.load(old_model_path)
new_model = joblib.load(new_model_path)
old_scaler = joblib.load(old_scaler_path)
new_scaler = joblib.load(new_scaler_path)

# Load the dataset (make sure you provide the correct path to the dataset)
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

print("Old Model Accuracy:", accuracy_old)
print("New Model Accuracy:", accuracy_new)
print("\nOld Model Classification Report:\n", report_old)
print("\nNew Model Classification Report:\n", report_new)
