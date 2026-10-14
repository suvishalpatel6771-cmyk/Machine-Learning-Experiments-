from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "carbonwise_model.joblib"


def get_number(label, minimum, maximum, integer=False):
    while True:
        try:
            value = float(input(f"{label} ({minimum} - {maximum}): "))
            if integer and not value.is_integer():
                raise ValueError
            if value < minimum or value > maximum:
                raise ValueError
            return int(value) if integer else value
        except ValueError:
            print("Invalid value. Please enter a number within the allowed range.")


def category(value):
    if value < 50:
        return "Low"
    if value < 150:
        return "Moderate"
    if value < 300:
        return "High"
    return "Very High"


def main():
    artifacts = joblib.load(MODEL_PATH)
    model = artifacts["pipeline"]
    features = artifacts["feature_names"]

    population = get_number("Area Population", 1, 1000, integer=True)
    electricity = get_number("Electricity MWh/month", 10, 100000)
    fuel = get_number("Fuel litres/month", 100, 50000)
    transport = get_number("Transport vehicle-km/month", 1000, 5000000)

    sample = pd.DataFrame([{
        "Population": population,
        "Electricity_MWh_Month": electricity,
        "Fuel_Litres_Month": fuel,
        "Transport_Vehicle_Km_Month": transport,
    }], columns=features)

    prediction = float(model.predict(sample)[0])

    print("\n" + "=" * 55)
    print("CARBONWISE PREDICTION")
    print("=" * 55)
    print(f"Population       : {population} people")
    print(f"Electricity      : {electricity:,.2f} MWh/month")
    print(f"Fuel             : {fuel:,.2f} litres/month")
    print(f"Transport        : {transport:,.2f} vehicle-km/month")
    print(f"\nEstimated Carbon : {prediction:,.2f} tonnes CO2e/month")
    print(f"Category         : {category(prediction)}")
    print("=" * 55)
    print("Educational prediction using synthetic training data.")


if __name__ == "__main__":
    main()
