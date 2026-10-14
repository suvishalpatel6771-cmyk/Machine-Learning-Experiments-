# CarbonWise

## Area-Level Carbon Emission Prediction Using Machine Learning

CarbonWise is an educational end-to-end machine-learning web prototype for estimating monthly area-level carbon emissions from population, electricity consumption, fuel consumption and transport activity.

### Main academic model
- ElasticNet Regression
- StandardScaler
- Median imputation
- 80/20 train-test split
- 5-fold cross-validation for ElasticNet tuning

### Model comparison
The training pipeline also compares:
- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet Regression
- Random Forest Regressor

Metrics:
- R²
- MAE
- RMSE

### Inputs
| Input | Range | Unit |
|---|---:|---|
| Population | 1–1000 | people |
| Electricity | 10–100000 | MWh/month |
| Fuel | 100–50000 | litres/month |
| Transport | 1000–5000000 | vehicle-km/month |

### Output
Estimated carbon emissions in **tonnes CO2e/month**.

### Project structure
```text
CarbonWise/
├── app.py
├── train_model.py
├── evaluate.py
├── eda.py
├── predict.py
├── requirements.txt
├── pyproject.toml
├── README.md
├── .gitignore
├── data/
│   └── carbon_emission_dataset.csv
├── models/
│   ├── carbonwise_model.joblib
│   └── model_metadata.json
├── outputs/
│   ├── metrics.csv
│   ├── model_comparison.csv
│   ├── actual_vs_predicted.png
│   ├── residual_plot.png
│   ├── feature_coefficients.csv
│   └── feature_coefficients.png
└── templates/
    └── index.html
```

### Run

```bash
pip install -r requirements.txt
python train_model.py
python evaluate.py
python eda.py
python predict.py
python app.py
```

For Windows, you can also run `run.bat` after opening a terminal in the project folder.

Open:

`http://127.0.0.1:5000`

### API

`GET /health`

`POST /predict`

Example:

```json
{
  "population": 100,
  "electricity": 500,
  "fuel": 1000,
  "transport": 10000
}
```

### Data transparency
The current CSV is a reproducible **synthetic educational dataset** generated with random seed 42. The target relationship uses illustrative coefficients and noise. These coefficients are not official emission factors, and the predictions must not be presented as certified real-world emissions measurements.

### Future work
- Replace the synthetic dataset with validated local measurements.
- Add documented regional emission factors.
- Add time-series forecasting.
- Add uncertainty intervals and external validation.
- Compare against additional domain-appropriate models.
