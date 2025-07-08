from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("../week5_tuning_deployment_prep/Churn_model.pkl")

# Load column names from training
COLUMNS = model.feature_names_in_

@app.route("/")
def home():
    return "📡 Churn Prediction API is Live!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        df = df.reindex(columns=COLUMNS, fill_value=0)
        prediction = model.predict(df)[0]
        result = "Churn" if prediction == 1 else "No Churn"
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
