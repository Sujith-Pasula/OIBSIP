# House Price Prediction with Linear Regression
**Oasis Infobyte — Data Analytics Internship | Level 2, Task 1**

An end-to-end regression project using the California Housing dataset. It includes EDA, missing-value handling, categorical one-hot encoding when available, an 80/20 train-test split, Linear Regression, MSE/RMSE/R² evaluation, actual-vs-predicted and residual plots, coefficient analysis, and a Ridge/Lasso comparison.

> Dataset note: this dataset describes California census-block groups, not individual Indian homes. Outputs are educational estimates, not real-estate valuations.

## Run in Windows PowerShell / VS Code
Extract this ZIP, open the folder in VS Code, then run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python src\house_price_prediction.py
```

If activation is blocked:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Generated outputs
- `data/housing.csv` — downloaded dataset
- `outputs/charts/` — EDA and model diagnostic charts
- `outputs/reports/eda_report.md` — report with the actual results from your run
- `outputs/metrics.csv` — MSE, RMSE, R² for Linear Regression, Ridge, and Lasso
- `outputs/predictions.csv` — actual, predicted, residual values for test set
- `outputs/coefficients.csv` — coefficient analysis
- `outputs/summary.json` — dataset/run summary

Use the results generated on your own computer in the GitHub README and demo; do not invent or copy metrics. Internet is required for first-time dataset download.
