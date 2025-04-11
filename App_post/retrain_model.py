import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load the dataset (ensure the correct path to the data.csv)
data = pd.read_csv('data.csv')

# Map 'M' and 'B' to 1 and 0 (Malignant and Benign)
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
data = data.drop(columns=['id', 'Unnamed: 32'], errors='ignore')  # Drop any unnecessary columns

# Split the features and target
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Train the RandomForestClassifier (you can use a different model if needed)
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)

# Save the model and scaler to disk
joblib.dump(model, 'new_model.pkl')
joblib.dump(scaler, 'new_scaler.pkl')

