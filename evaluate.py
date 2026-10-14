from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "carbon_emission_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "carbonwise_model.joblib"
OUTPUT_DIR = BASE_DIR / "outputs"
SEED = 42


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    artifacts = joblib.load(MODEL_PATH)
    model = artifacts["pipeline"]
    features = artifacts["feature_names"]
    target = artifacts["target_name"]

    X = df[features]
    y = df[target]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED)
    pred = model.predict(X_test)

    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))

    report = pd.DataFrame([{"R2": r2, "MAE": mae, "RMSE": rmse}])
    report.to_csv(OUTPUT_DIR / "metrics.csv", index=False)

    # Actual vs predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, pred, alpha=0.55)
    low = min(float(y_test.min()), float(pred.min()))
    high = max(float(y_test.max()), float(pred.max()))
    plt.plot([low, high], [low, high], linestyle="--")
    plt.xlabel("Actual tonnes CO2e/month")
    plt.ylabel("Predicted tonnes CO2e/month")
    plt.title("CarbonWise: Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=220)
    plt.close()

    # Residual plot
    residuals = y_test - pred
    plt.figure(figsize=(8, 6))
    plt.scatter(pred, residuals, alpha=0.55)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted tonnes CO2e/month")
    plt.ylabel("Residual")
    plt.title("CarbonWise: Residual Analysis")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "residual_plot.png", dpi=220)
    plt.close()

    # Feature coefficients for the scaled ElasticNet model
    reg = model.named_steps["regressor"]
    coefs = pd.DataFrame({"Feature": features, "Coefficient": reg.coef_})
    coefs["Absolute_Coefficient"] = coefs["Coefficient"].abs()
    coefs = coefs.sort_values("Absolute_Coefficient", ascending=False)
    coefs.to_csv(OUTPUT_DIR / "feature_coefficients.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.bar(coefs["Feature"], coefs["Coefficient"])
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("ElasticNet coefficient")
    plt.title("CarbonWise Feature Coefficients")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_coefficients.png", dpi=220)
    plt.close()

    print("=" * 70)
    print("CARBONWISE MODEL EVALUATION")
    print("=" * 70)
    print(f"R2   : {r2:.6f}")
    print(f"MAE  : {mae:.4f} tonnes CO2e/month")
    print(f"RMSE : {rmse:.4f} tonnes CO2e/month")
    print("Evaluation files saved to outputs/.")


if __name__ == "__main__":
    main()
