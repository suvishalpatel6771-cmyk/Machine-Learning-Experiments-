from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_PATH = DATA_DIR / "carbon_emission_dataset.csv"
MODEL_PATH = MODEL_DIR / "carbonwise_model.joblib"
META_PATH = MODEL_DIR / "model_metadata.json"

SEED = 42
SAMPLES = 5000
FEATURES = ["Population", "Electricity_MWh_Month", "Fuel_Litres_Month", "Transport_Vehicle_Km_Month"]
TARGET = "Carbon_Emissions_Tonnes_CO2e_Month"


def make_dataset(path: Path) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    population = rng.integers(1, 1001, SAMPLES)
    electricity = rng.uniform(10, 100000, SAMPLES)
    fuel = rng.uniform(100, 50000, SAMPLES)
    transport = rng.uniform(1000, 5000000, SAMPLES)

    # Synthetic educational relationship; not official emission factors.
    noise = rng.normal(0, 5, SAMPLES)
    emissions = (
        electricity * 0.70
        + fuel * 0.00231
        + transport * 0.00018
        + population * 0.0007
        + noise
    )
    emissions = np.maximum(emissions, 0)

    df = pd.DataFrame({
        "Population": population,
        "Electricity_MWh_Month": electricity,
        "Fuel_Litres_Month": fuel,
        "Transport_Vehicle_Km_Month": transport,
        TARGET: emissions,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def make_pipeline(estimator):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("regressor", estimator),
    ])


def metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def main():
    DATA_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("CARBONWISE | COMPLETE ML TRAINING PIPELINE")
    print("=" * 70)

    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        print(f"[1] Existing dataset loaded: {DATA_PATH}")
    else:
        df = make_dataset(DATA_PATH)
        print(f"[1] Synthetic educational dataset created: {DATA_PATH}")

    print(f"    Shape: {df.shape}")
    print(f"    Columns: {list(df.columns)}")

    print("\n[2] EDA / data quality")
    print(df.describe().T[["mean", "std", "min", "50%", "max"]].round(3))
    print("Missing values:")
    print(df.isna().sum().to_string())
    print(f"Duplicate rows: {df.duplicated().sum()}")

    df = df.drop_duplicates().copy()
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED
    )
    print(f"\n[3] Split: {len(X_train)} train / {len(X_test)} test")

    print("\n[4] Model comparison")
    candidates = {
        "Linear Regression": make_pipeline(LinearRegression()),
        "Ridge Regression": make_pipeline(Ridge(alpha=1.0)),
        "Lasso Regression": make_pipeline(Lasso(alpha=0.01, max_iter=20000, random_state=SEED)),
        "ElasticNet Regression": make_pipeline(ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=30000, random_state=SEED)),
        "Random Forest Regressor": make_pipeline(RandomForestRegressor(
            n_estimators=250, max_depth=18, min_samples_leaf=2,
            random_state=SEED, n_jobs=-1
        )),
    }

    rows = []
    fitted = {}
    for name, pipeline in candidates.items():
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        m = metrics(y_test, pred)
        rows.append({"Model": name, **m})
        fitted[name] = pipeline
        print(f"    {name:26s} R2={m['R2']:.6f}  MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}")

    comparison = pd.DataFrame(rows).sort_values("R2", ascending=False)
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    # Model comparison chart
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.bar(comparison["Model"], comparison["R2"])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("R² score")
    plt.title("CarbonWise Model Comparison")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=220)
    plt.close()

    # Correlation heatmap
    corr = df[FEATURES + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=25, ha="right")
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center")
    ax.set_title("CarbonWise Feature Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=220)
    plt.close(fig)

    print("\n[5] ElasticNet tuning")
    elastic = make_pipeline(ElasticNet(max_iter=50000, random_state=SEED))
    grid = GridSearchCV(
        elastic,
        {"regressor__alpha": [0.001, 0.003, 0.01, 0.03, 0.1],
         "regressor__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
        scoring="r2",
        cv=KFold(n_splits=5, shuffle=True, random_state=SEED),
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    final_model = grid.best_estimator_
    pred = final_model.predict(X_test)
    final_metrics = metrics(y_test, pred)
    print(f"    Best params: {grid.best_params_}")
    print(f"    CV best R2: {grid.best_score_:.6f}")
    print(f"    Test R2: {final_metrics['R2']:.6f}")
    print(f"    Test MAE: {final_metrics['MAE']:.4f}")
    print(f"    Test RMSE: {final_metrics['RMSE']:.4f}")

    artifacts = {
        "pipeline": final_model,
        "feature_names": FEATURES,
        "target_name": TARGET,
        "model_name": "ElasticNet Regression",
        "best_params": grid.best_params_,
        "random_state": SEED,
        "units": {
            "Population": "people",
            "Electricity_MWh_Month": "MWh/month",
            "Fuel_Litres_Month": "litres/month",
            "Transport_Vehicle_Km_Month": "vehicle-km/month",
            TARGET: "tonnes CO2e/month",
        },
        "data_note": "Synthetic educational dataset; coefficients are illustrative, not official emission factors.",
    }
    joblib.dump(artifacts, MODEL_PATH)

    META_PATH.write_text(json.dumps({
        "model": artifacts["model_name"],
        "features": FEATURES,
        "target": TARGET,
        "best_params": grid.best_params_,
        "cv_r2": grid.best_score_,
        "test_metrics": final_metrics,
        "dataset_rows": len(df),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "seed": SEED,
        "note": artifacts["data_note"],
    }, indent=2), encoding="utf-8")

    pd.DataFrame([{
        "Model": "ElasticNet Regression",
        **final_metrics,
        "CV_R2": grid.best_score_,
    }]).to_csv(OUTPUT_DIR / "metrics.csv", index=False)

    print(f"\n[6] Saved model: {MODEL_PATH}")
    print("[7] Saved metrics and comparison files in outputs/")
    print("Training complete.")


if __name__ == "__main__":
    main()
