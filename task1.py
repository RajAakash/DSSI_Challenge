import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the datasets without headers
normal_df = pd.read_csv('datasets_1_2/ptbdb_normal.csv', header=None)
abnormal_df = pd.read_csv('datasets_1_2/ptbdb_abnormal.csv', header=None)

# Merge the datasets
df_merged = pd.concat([normal_df, abnormal_df], ignore_index=True)
print(df_merged.head(5))

# Handle missing values
df_merged.fillna(df_merged.mean(), inplace=True)

# Separate features and labels
X = df_merged.iloc[:, :-1]
y = df_merged.iloc[:, -1]

# Scale the features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets with shuffling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

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
