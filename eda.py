from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "carbon_emission_dataset.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
FEATURES = ["Population", "Electricity_MWh_Month", "Fuel_Litres_Month", "Transport_Vehicle_Km_Month"]
TARGET = "Carbon_Emissions_Tonnes_CO2e_Month"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    print("Dataset shape:", df.shape)
    print("\nData types:\n", df.dtypes)
    print("\nMissing values:\n", df.isna().sum())
    print("\nSummary:\n", df.describe().T)

    plt.figure(figsize=(8, 5))
    plt.hist(df[TARGET], bins=40)
    plt.xlabel("Tonnes CO2e/month")
    plt.title("Carbon Emissions Distribution")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "emissions_distribution.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 6))
    corr = df[FEATURES + [TARGET]].corr()
    plt.imshow(corr.values, aspect="auto")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=25, ha="right")
    plt.yticks(range(len(corr.columns)), corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            plt.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center")
    plt.title("CarbonWise Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=220)
    plt.close()

    print("EDA plots saved to outputs/.")


if __name__ == "__main__":
    main()
