import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle
import os

# --- CONFIG ---
DATA_PATH = r"C:\Users\ANAGHA\OneDrive\Desktop\AI Project(practice)\credit_card_fraud_dataset (1).csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

print("📂 Loading dataset...")
credit = pd.read_csv(DATA_PATH)
print(credit["TransactionType"].unique())

# --- Encode categorical columns safely ---
cat_cols = ["TransactionType", "Location"]
encoders = {}
for col in cat_cols:
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    credit[[col]] = enc.fit_transform(credit[[col]])
    encoders[col] = enc

# --- Define features and target ---
X = credit.drop(["TransactionDate", "IsFraud"], axis=1, errors="ignore")
y = credit["IsFraud"]
print(X)
# --- Train/test split (70% train / 30% test) with stratify ---
x_train, x_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state=42, stratify=y
)

# --- Scale features ---
scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

# --- Train Logistic Regression model with balanced class weight ---
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(x_train_scaled, y_train)

# --- Evaluate ---
print("✅ Training Accuracy:", model.score(x_train_scaled, y_train))
print("✅ Test Accuracy:", model.score(x_test_scaled, y_test))
print("🔎 Unique predictions in test set:", np.unique(model.predict(x_test_scaled)))

# --- Save model, scaler, feature order, and encoders ---
with open(os.path.join(MODEL_DIR, "fraud_model.pkl"), "wb") as f:
    pickle.dump(model, f)

with open(os.path.join(MODEL_DIR, "fraud_scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

with open(os.path.join(MODEL_DIR, "fraud_feature_order.pkl"), "wb") as f:
    pickle.dump(list(X.columns), f)

for col, enc in encoders.items():
    with open(os.path.join(MODEL_DIR, f"{col.lower()}_encoder.pkl"), "wb") as f:
        pickle.dump(enc, f)

print("🎉 Model, scaler, feature order, and encoders saved successfully!")
