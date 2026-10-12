"""
OIB-SIP Data Analytics — Level 1, Task 1: Retail Sales EDA
Run:
    python -m pip install -r requirements.txt
    python Retail_Sales_EDA.py

The bundled CSV is synthetic sample data for demonstration. See README.md.
"""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset" / "retail_store_sales.csv"
OUT = ROOT / "outputs"
CHARTS = OUT / "charts"
REPORTS = OUT / "reports"
CLEANED = OUT / "retail_store_sales_cleaned.csv"
SUMMARY_CSV = OUT / "summary_metrics.csv"
INSIGHTS_TXT = REPORTS / "eda_report.md"
for folder in (CHARTS, REPORTS):
    folder.mkdir(parents=True, exist_ok=True)

if not DATA.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA}\nPlace retail_store_sales.csv in dataset/.")

raw = pd.read_csv(DATA)
original_rows = len(raw)
raw_shape = list(raw.shape)
raw_missing = raw.isna().sum().to_dict()
raw_duplicates = int(raw.duplicated().sum())

df = raw.copy()
df.columns = df.columns.astype(str).str.strip()
df = df.dropna(how="all").drop_duplicates()

# Normalize text safely.
text_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
for col in text_cols:
    df[col] = df[col].astype("string").str.strip()
    df[col] = df[col].replace("", pd.NA)

# Normalize common categories, but preserve readable labels.
if "Category" in df:
    df["Category"] = df["Category"].str.title()
if "Payment Method" in df:
    df["Payment Method"] = df["Payment Method"].str.title()
if "Location" in df:
    df["Location"] = df["Location"].str.title()

numeric_cols = [c for c in ["Price Per Unit", "Quantity", "Total Spent", "Age"] if c in df.columns]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
if "Transaction Date" in df:
    df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], errors="coerce")

# Remove rows where a valid transaction date or meaningful sales amount is unavailable.
rows_before_validity_filter = len(df)
if "Transaction Date" in df:
    df = df.dropna(subset=["Transaction Date"])
if "Total Spent" in df:
    df = df.dropna(subset=["Total Spent"])
if "Total Spent" in df:
    df = df[df["Total Spent"] >= 0]
for col in ["Price Per Unit", "Quantity"]:
    if col in df:
        df = df[df[col].isna() | (df[col] >= 0)]

# Fill only fields where imputation is defensible; don't fabricate IDs or dates.
for col in ["Price Per Unit", "Quantity", "Age"]:
    if col in df and df[col].isna().any():
        median = df[col].median()
        if pd.notna(median):
            df[col] = df[col].fillna(median)
for col in df.select_dtypes(include=["object", "string"]).columns:
    if col not in ["Transaction ID", "Customer ID"]:
        df[col] = df[col].fillna("Unknown")

# Create analysis features.
if "Transaction Date" in df:
    df["Year"] = df["Transaction Date"].dt.year
    df["Month"] = df["Transaction Date"].dt.to_period("M").astype(str)
    df["Quarter"] = df["Transaction Date"].dt.to_period("Q").astype(str)
    df["Day Of Week"] = df["Transaction Date"].dt.day_name()
if {"Price Per Unit", "Quantity"}.issubset(df.columns):
    df["Calculated Revenue"] = (df["Price Per Unit"] * df["Quantity"]).round(2)
if {"Total Spent", "Calculated Revenue"}.issubset(df.columns):
    df["Revenue Difference"] = (df["Total Spent"] - df["Calculated Revenue"]).round(2)
if "Age" in df:
    df["Age Group"] = pd.cut(df["Age"], bins=[17, 25, 35, 45, 55, 120],
                             labels=["18–25", "26–35", "36–45", "46–55", "56+"], include_lowest=True)

df.to_csv(CLEANED, index=False)

# Display data-quality and descriptive-statistics checks.
print("=" * 72)
print("OIB-SIP | RETAIL SALES EXPLORATORY DATA ANALYSIS")
print("=" * 72)
print(f"Input shape: {raw_shape}; rows after cleaning: {len(df)}")
print(f"Duplicate rows detected in raw data: {raw_duplicates}")
print("\nRaw missing values:\n", pd.Series(raw_missing).to_string())
print("\nCleaned data types:\n", df.dtypes.to_string())
print("\nDescriptive statistics:\n", df.describe(include="all").to_string())
print("\nNumeric summary (mean, median, mode, std):")
for col in numeric_cols:
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if not s.empty:
        mode = s.mode()
        print(f"- {col}: mean={s.mean():.2f}; median={s.median():.2f}; "
              f"mode={(mode.iloc[0] if not mode.empty else 'N/A')}; std={s.std():.2f}")

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 120
metrics = []
findings = []
recommendations = []

def save_chart(name, title):
    plt.title(title, fontsize=14, pad=12)
    plt.tight_layout()
    plt.savefig(CHARTS / name, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Chart saved: outputs/charts/{name}")

def metric(name, value, note=""):
    metrics.append({"Metric": name, "Value": value, "Notes": note})

if "Total Spent" in df:
    total_revenue = float(df["Total Spent"].sum())
    avg_transaction = float(df["Total Spent"].mean())
    median_transaction = float(df["Total Spent"].median())
    metric("Total revenue", round(total_revenue, 2), "Sum of Total Spent after cleaning")
    metric("Average transaction value", round(avg_transaction, 2), "Mean Total Spent per row")
    metric("Median transaction value", round(median_transaction, 2), "Median Total Spent per row")
    findings.append(f"- Total recorded revenue after cleaning: **{total_revenue:,.2f}**.")
    findings.append(f"- Average transaction value: **{avg_transaction:,.2f}**; median transaction value: **{median_transaction:,.2f}**.")

if "Transaction ID" in df:
    metric("Unique transactions", int(df["Transaction ID"].nunique()))
if "Customer ID" in df:
    metric("Unique customers", int(df["Customer ID"].nunique()), "Unique non-null Customer ID values")
if "Quantity" in df:
    metric("Units sold", int(df["Quantity"].sum()))

if {"Month", "Total Spent"}.issubset(df.columns):
    monthly = df.groupby("Month", as_index=False)["Total Spent"].sum().sort_values("Month")
    plt.figure(figsize=(11, 5))
    sns.lineplot(data=monthly, x="Month", y="Total Spent", marker="o", linewidth=2)
    plt.xlabel("Month"); plt.ylabel("Revenue")
    plt.xticks(rotation=35, ha="right")
    save_chart("01_monthly_revenue.png", "Monthly Revenue Trend")
    if not monthly.empty:
        best = monthly.loc[monthly["Total Spent"].idxmax()]
        worst = monthly.loc[monthly["Total Spent"].idxmin()]
        findings.append(f"- Highest-revenue month: **{best['Month']}** ({best['Total Spent']:,.2f}); lowest: **{worst['Month']}** ({worst['Total Spent']:,.2f}).")
        metric("Best revenue month", best["Month"], f"Revenue {best['Total Spent']:.2f}")
        recommendations.append(f"Review the drivers behind {best['Month']}'s stronger sales and test whether its product mix or promotions can be repeated in weaker months such as {worst['Month']}.")

if {"Quarter", "Total Spent"}.issubset(df.columns):
    quarterly = df.groupby("Quarter", as_index=False)["Total Spent"].sum().sort_values("Quarter")
    plt.figure(figsize=(8, 5))
    sns.barplot(data=quarterly, x="Quarter", y="Total Spent", color="steelblue")
    plt.xlabel("Quarter"); plt.ylabel("Revenue")
    save_chart("02_quarterly_revenue.png", "Quarterly Revenue Comparison")

if {"Category", "Total Spent"}.issubset(df.columns):
    category_sales = df.groupby("Category", as_index=False)["Total Spent"].sum().sort_values("Total Spent", ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(data=category_sales, x="Total Spent", y="Category", color="steelblue")
    plt.xlabel("Revenue"); plt.ylabel("Product category")
    save_chart("03_revenue_by_category.png", "Revenue by Product Category")
    if len(category_sales):
        top = category_sales.iloc[0]
        bottom = category_sales.iloc[-1]
        findings.append(f"- Leading category by revenue: **{top['Category']}** ({top['Total Spent']:,.2f}); lowest category: **{bottom['Category']}** ({bottom['Total Spent']:,.2f}).")
        metric("Top revenue category", top["Category"], f"Revenue {top['Total Spent']:.2f}")
        recommendations.append(f"Check stock availability and margins for {top['Category']}, the leading revenue category. Test targeted bundles or promotions for {bottom['Category']} after checking profit margin and demand.")

if {"Item", "Total Spent"}.issubset(df.columns):
    top_items = df.groupby("Item", as_index=False)["Total Spent"].sum().sort_values("Total Spent", ascending=False).head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_items, x="Total Spent", y="Item", color="steelblue")
    plt.xlabel("Revenue"); plt.ylabel("Product")
    save_chart("04_top_10_products.png", "Top 10 Products by Revenue")
    if len(top_items):
        item = top_items.iloc[0]
        findings.append(f"- Top product by recorded revenue: **{item['Item']}** ({item['Total Spent']:,.2f}).")
        metric("Top revenue product", item["Item"], f"Revenue {item['Total Spent']:.2f}")

if {"Gender", "Total Spent"}.issubset(df.columns):
    gender_sales = df.groupby("Gender", as_index=False)["Total Spent"].sum().sort_values("Total Spent", ascending=False)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=gender_sales, x="Gender", y="Total Spent", color="steelblue")
    plt.xlabel("Gender"); plt.ylabel("Revenue")
    save_chart("05_revenue_by_gender.png", "Revenue by Gender")

if {"Age Group", "Total Spent"}.issubset(df.columns):
    age_sales = df.groupby("Age Group", observed=False, as_index=False)["Total Spent"].sum()
    plt.figure(figsize=(8, 5))
    sns.barplot(data=age_sales, x="Age Group", y="Total Spent", color="steelblue")
    plt.xlabel("Age group"); plt.ylabel("Revenue")
    save_chart("06_revenue_by_age_group.png", "Revenue by Age Group")

numeric = df.select_dtypes(include=np.number)
if numeric.shape[1] >= 2:
    plt.figure(figsize=(9, 7))
    sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f", center=0, square=True)
    save_chart("07_correlation_heatmap.png", "Correlation Between Numeric Variables")

if {"Payment Method", "Total Spent"}.issubset(df.columns):
    payment = df.groupby("Payment Method", as_index=False)["Total Spent"].sum().sort_values("Total Spent", ascending=False)
    plt.figure(figsize=(9, 5))
    sns.barplot(data=payment, x="Payment Method", y="Total Spent", color="steelblue")
    plt.xlabel("Payment method"); plt.ylabel("Revenue")
    plt.xticks(rotation=20, ha="right")
    save_chart("08_revenue_by_payment_method.png", "Additional Insight: Revenue by Payment Method")
    if len(payment):
        top_pay = payment.iloc[0]
        findings.append(f"- Highest-revenue payment method: **{top_pay['Payment Method']}** ({top_pay['Total Spent']:,.2f}).")
        recommendations.append(f"Ensure checkout and reconciliation work smoothly for {top_pay['Payment Method']}, the largest revenue-contributing payment method in this dataset.")

if {"Location", "Total Spent"}.issubset(df.columns):
    channel = df.groupby("Location", as_index=False)["Total Spent"].sum().sort_values("Total Spent", ascending=False)
    plt.figure(figsize=(7, 5))
    sns.barplot(data=channel, x="Location", y="Total Spent", color="steelblue")
    plt.xlabel("Sales channel"); plt.ylabel("Revenue")
    save_chart("09_revenue_by_location.png", "Additional Insight: Revenue by Sales Channel")

pd.DataFrame(metrics).to_csv(SUMMARY_CSV, index=False)

# Create an evidence-based report from computed results.
quality_lines = [
    f"- Input rows: {original_rows}; input columns: {len(raw.columns)}.",
    f"- Duplicate rows detected: {raw_duplicates}.",
    f"- Rows retained after cleaning: {len(df)}; rows excluded overall: {original_rows - len(df)}.",
    f"- Raw missing cells: {int(pd.Series(raw_missing).sum())}."
]
report = [
    "# Retail Sales EDA — Results Report",
    "",
    "> **Dataset caveat:** The bundled data is synthetic sample data for demonstration. Treat all figures and findings as a coding exercise, not real retailer performance.",
    "",
    "## 1. Method",
    "Inspected data types, missing values, duplicates and descriptive statistics; normalized text categories; converted numeric and date fields; removed duplicate/all-empty rows and rows without valid transaction dates or revenue; filled selected numeric fields with medians; engineered month, quarter, weekday and age-group features where columns were available.",
    "",
    "## 2. Data quality summary",
    *quality_lines,
    "",
    "## 3. Key findings",
    *(findings if findings else ["- No revenue findings could be generated because the expected sales columns were unavailable."]),
    "",
    "## 4. Actionable recommendations",
    *[f"{i}. {rec}" for i, rec in enumerate(recommendations[:3], start=1)],
    "",
    "## 5. Limitations",
    "- Synthetic data is not representative of a real business or customer population.",
    "- Revenue is not profit; margin, discounts, returns, stock levels and marketing spend are not available.",
    "- Correlation does not establish causation. Confirm patterns with more data before making business decisions.",
    "- If the assigned dataset lacks age or gender fields, those demographic analyses will not be generated.",
    "",
    "## 6. Output files",
    "- Cleaned data: `outputs/retail_store_sales_cleaned.csv`",
    "- Metric table: `outputs/summary_metrics.csv`",
    "- Charts: `outputs/charts/`",
]
INSIGHTS_TXT.write_text("\n".join(report) + "\n", encoding="utf-8")
print("\nCreated report:", INSIGHTS_TXT)
print("Created metrics:", SUMMARY_CSV)
print("Cleaned dataset:", CLEANED)
print("\nRun completed successfully.")
