from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "carbonwise_model.joblib"

app = Flask(__name__)
CORS(app)

artifacts = None
model = None
FEATURES = [
    "Population",
    "Electricity_MWh_Month",
    "Fuel_Litres_Month",
    "Transport_Vehicle_Km_Month",
]

try:
    artifacts = joblib.load(MODEL_PATH)
    model = artifacts["pipeline"]
    FEATURES = artifacts["feature_names"]
    print("CarbonWise model loaded successfully.")
except Exception as exc:
    print(f"Model loading error: {exc}")


def emission_category(value: float) -> str:
    if value < 50:
        return "Low"
    if value < 150:
        return "Moderate"
    if value < 300:
        return "High"
    return "Very High"


def read_input(data, key):
    if key not in data:
        raise ValueError(f"Missing input: {key}")
    value = float(data[key])
    if not np.isfinite(value):
        raise ValueError(f"{key} must be a finite number")
    return value


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "model": artifacts.get("model_name") if artifacts else None,
        "population_range": [1, 1000],
        "unit": "tonnes CO2e/month",
    })


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model is not loaded. Run python train_model.py first."}), 500

    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "Request body must be JSON."}), 400

        population = read_input(data, "population")
        electricity = read_input(data, "electricity")
        fuel = read_input(data, "fuel")
        transport = read_input(data, "transport")

        if not population.is_integer() or not 1 <= population <= 1000:
            raise ValueError("Population must be a whole number between 1 and 1000.")
        if not 10 <= electricity <= 100000:
            raise ValueError("Electricity must be between 10 and 100000 MWh/month.")
        if not 100 <= fuel <= 50000:
            raise ValueError("Fuel must be between 100 and 50000 litres/month.")
        if not 1000 <= transport <= 5000000:
            raise ValueError("Transport must be between 1000 and 5000000 vehicle-km/month.")

        sample = pd.DataFrame([{
            "Population": int(population),
            "Electricity_MWh_Month": electricity,
            "Fuel_Litres_Month": fuel,
            "Transport_Vehicle_Km_Month": transport,
        }], columns=FEATURES)

        prediction = max(float(model.predict(sample)[0]), 0.0)

        return jsonify({
            "prediction": round(prediction, 2),
            "category": emission_category(prediction),
            "unit": "tonnes CO2e/month",
            "inputs": {
                "population": int(population),
                "electricity": electricity,
                "fuel": fuel,
                "transport": transport,
            },
            "note": "Educational estimate based on synthetic training data; not an official emissions measurement.",
        })

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Prediction error")
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
