# Retail Sales EDA — Results Report

> **Dataset caveat:** The bundled data is synthetic sample data for demonstration. Treat all figures and findings as a coding exercise, not real retailer performance.

## 1. Method
Inspected data types, missing values, duplicates and descriptive statistics; normalized text categories; converted numeric and date fields; removed duplicate/all-empty rows and rows without valid transaction dates or revenue; filled selected numeric fields with medians; engineered month, quarter, weekday and age-group features where columns were available.

## 2. Data quality summary
- Input rows: 720; input columns: 12.
- Duplicate rows detected: 0.
- Rows retained after cleaning: 719; rows excluded overall: 1.
- Raw missing cells: 1.

## 3. Key findings
- Total recorded revenue after cleaning: **33,563.04**.
- Average transaction value: **46.68**; median transaction value: **30.90**.
- Highest-revenue month: **2025-08** (3,515.60); lowest: **2025-05** (1,928.26).
- Leading category by revenue: **Clothing** (11,275.89); lowest category: **Books** (2,230.10).
- Top product by recorded revenue: **Jacket** (5,020.02).
- Highest-revenue payment method: **Upi** (10,169.08).

## 4. Actionable recommendations
1. Review the drivers behind 2025-08's stronger sales and test whether its product mix or promotions can be repeated in weaker months such as 2025-05.
2. Check stock availability and margins for Clothing, the leading revenue category. Test targeted bundles or promotions for Books after checking profit margin and demand.
3. Ensure checkout and reconciliation work smoothly for Upi, the largest revenue-contributing payment method in this dataset.

## 5. Limitations
- Synthetic data is not representative of a real business or customer population.
- Revenue is not profit; margin, discounts, returns, stock levels and marketing spend are not available.
- Correlation does not establish causation. Confirm patterns with more data before making business decisions.
- If the assigned dataset lacks age or gender fields, those demographic analyses will not be generated.

## 6. Output files
- Cleaned data: `outputs/retail_store_sales_cleaned.csv`
- Metric table: `outputs/summary_metrics.csv`
- Charts: `outputs/charts/`
