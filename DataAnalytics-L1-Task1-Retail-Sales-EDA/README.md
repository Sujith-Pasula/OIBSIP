# Retail Sales Exploratory Data Analysis (EDA)

**Program:** Oasis Infobyte OIB-SIP — Data Analytics, Level 1, Task 1  
**Primary tools:** Python, pandas, NumPy, Matplotlib, Seaborn  
**Editor:** VS Code

## Dataset source and honesty note
The included `dataset/retail_store_sales.csv` is **synthetic sample data generated for this runnable project bundle**. It is not the original Kaggle dataset and must not be presented as real retailer data. If Oasis Infobyte assigned or specified a dataset, replace this CSV with that official/assigned file (keep the filename the same) and rerun the analysis.

## Project goals
- Inspect dataset dimensions, columns, types, missing values, duplicates and descriptive statistics.
- Clean common data quality issues and document rows retained/excluded.
- Analyze monthly and quarterly revenue, categories, top products, age groups and gender when fields exist.
- Examine correlations among numeric variables.
- Add payment-method and sales-channel insights.
- Export charts, a cleaned CSV, summary metrics and an automatically generated Markdown report.

## Run in VS Code (Windows PowerShell)
Open this project folder in VS Code, then use Terminal → New Terminal:

```powershell
python -m pip install -r requirements.txt
python Retail_Sales_EDA.py
```

If `python` is not recognized, install Python 3 and select it as the VS Code interpreter. No Jupyter browser is required.

## Folder structure
```text
DataAnalytics-L1-Task1-Retail-Sales-EDA/
├── dataset/
│   └── retail_store_sales.csv
├── outputs/
│   ├── charts/              # PNG charts generated on run
│   ├── reports/
│   │   └── eda_report.md    # generated findings and recommendations
│   ├── retail_store_sales_cleaned.csv
│   └── summary_metrics.csv
├── Retail_Sales_EDA.py
├── requirements.txt
├── README.md
├── START_HERE.txt
└── .gitignore
```

## Outputs
The script generates charts for monthly revenue, quarterly revenue, category revenue, top 10 products, revenue by gender and age group (when present), correlation heatmap, payment method, and sales channel. It also saves a cleaned CSV, summary metrics, and a Markdown report.

## Interpretation
Read `outputs/reports/eda_report.md` after running the script. Its findings are calculated from the data loaded. Business recommendations are starting hypotheses; validate them against the real assigned dataset, profit margins, stock, returns and other business context before presenting them.

## Submission checklist
- [ ] Confirm dataset source and attribution.
- [ ] Run the script from a fresh terminal.
- [ ] Inspect all charts and the generated report.
- [ ] Add your own markdown observations and explain at least three recommendations.
- [ ] Capture screenshots and record a clear project demo.
- [ ] Push to a public GitHub repository named exactly `OIBSIP`, with this project in the required task folder.
