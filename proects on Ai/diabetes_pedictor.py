import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset
data = {
    "Age": [25, 30, 28, 22, 35],
    "Income": [50000, 60000, 75000, 40000, 90000],
    "Education": ["Bachelor's", "Master's", "PhD", "Bachelor's", "Master's"],
    "Employed": ["Yes", "Yes", "Yes", "No", "Yes"],
    "Outcome": ["Yes", "Yes", "No", "No", "Yes"]
}

# Create a DataFrame from the dataset
df = pd.DataFrame(data)

# Convert categorical variables into numerical values using one-hot encoding
df = pd.get_dummies(df, columns=["Education", "Employed"])

# Define the features (X) and target (y)
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train a RandomForestClassifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Generate and print a classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)
