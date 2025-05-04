import pandas as pd
import numpy as np
import joblib
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


df = pd.read_csv("Wednesday-workingHours.pcap_ISCX (1).csv")
print("Dataset shape:", df.shape)


df.columns = df.columns.str.strip()

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
 

benign = df[df['Label'] == 'BENIGN']
attacks = df[df['Label'] != 'BENIGN']
benign_downsampled = benign.sample(n=len(attacks), random_state=42)
df = pd.concat([benign_downsampled, attacks])

X = df.drop("Label", axis=1)
y = df["Label"]


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
joblib.dump(label_encoder, "label_encoder.pkl")


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.pkl")

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
print("All Features:", X.columns.tolist())

# Train initial model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Initial model trained.")
sys.stdout.flush()

# Evaluate
predictions = model.predict(X_test)
print("Initial Model Evaluation:")
print("Classification Report:\n", classification_report(y_test, predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))

# Select Top 20 Features Based on Importance
importances = model.feature_importances_
features = X.columns
feature_importance = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)

top_features = [f[0] for f in feature_importance[:20]]
print("Top 20 Important Features:", top_features)

# Save top features
joblib.dump(top_features, "top_features.pkl")

#  Retrain model with Top 20 Features
X_top = df[top_features]
X_top_scaled = scaler.fit_transform(X_top)
X_train_top, X_test_top, y_train_top, y_test_top = train_test_split(X_top_scaled, y_encoded, test_size=0.2, random_state=42)

model_top = RandomForestClassifier(n_estimators=100, random_state=42)
model_top.fit(X_train_top, y_train_top)

# Save the retrained model and updated scaler
joblib.dump(model_top, "cyber_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Final Evaluation
predictions_top = model_top.predict(X_test_top)
print("Final Model (Top Features) Evaluation:")
print("Classification Report:\n", classification_report(y_test_top, predictions_top))
print("Confusion Matrix:\n", confusion_matrix(y_test_top, predictions_top))

print(" Model retrained and saved with only top 20 important features.")