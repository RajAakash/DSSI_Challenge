import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the datasets without headers
train_df = pd.read_csv('datasets_1_2/mitbih_train.csv', header=None)
test_df = pd.read_csv('datasets_1_2/mitbih_test.csv', header=None)

# Display the first few rows of each dataset
print("Training Dataset:")
print(train_df.head())

print("\nTesting Dataset:")
print(test_df.head())
# Separate features and labels in the training dataset
X_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]

# Separate features and labels in the testing dataset
X_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]
# Scale the features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, y_pred))