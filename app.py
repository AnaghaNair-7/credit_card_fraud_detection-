from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# --- Load model, scaler, feature order, and encoders ---
MODEL_DIR = "models"

with open(os.path.join(MODEL_DIR, "fraud_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(MODEL_DIR, "fraud_scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(MODEL_DIR, "fraud_feature_order.pkl"), "rb") as f:
    feature_order = pickle.load(f)

with open(os.path.join(MODEL_DIR, "transactiontype_encoder.pkl"), "rb") as f:
    transaction_type_enc = pickle.load(f)

with open(os.path.join(MODEL_DIR, "location_encoder.pkl"), "rb") as f:
    location_enc = pickle.load(f)

# --- Routes ---
@app.route("/")
def welcome():
    return render_template("welcome.html")

@app.route("/input")
def input_page():
    return render_template("input.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect form inputs
        TransactionID = request.form["TransactionID"]
        Amount = float(request.form["Amount"])
        MerchantID = float(request.form["MerchantID"])
        TransactionType = request.form["TransactionType"]
        Location = request.form["Location"]

        # Encode categorical values properly (OrdinalEncoder expects 2D array)
        TransactionType_encoded = transaction_type_enc.transform([[TransactionType]])[0][0]
        Location_encoded = location_enc.transform([[Location]])[0][0]

        # Arrange features in the same order as training
        input_dict = {
            "TransactionID": float(TransactionID),  # <-- Included now
            "Amount": Amount,
            "MerchantID": MerchantID,
            "TransactionType": TransactionType_encoded,
            "Location": Location_encoded
        }

        # Ensure features are in correct order and 2D for scaler/model
        features = [input_dict[feat] for feat in feature_order]
        arr = np.array(features, dtype=float).reshape(1, -1)

        # Scale
        arr_scaled = scaler.transform(arr)

        # Predict
        prediction = model.predict(arr_scaled)[0]
        probability = model.predict_proba(arr_scaled)[0][prediction]

        result = "Fraud (1)" if prediction == 1 else "Not Fraud (0)"

        return render_template(
            "output.html",
            prediction=result,
            probability=f"{probability:.2f}",
            transaction_id=TransactionID
        )

    except Exception as e:
        return f"❌ Error during prediction: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
