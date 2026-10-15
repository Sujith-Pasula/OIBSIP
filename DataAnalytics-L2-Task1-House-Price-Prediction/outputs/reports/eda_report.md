# House Price Prediction — Run Report

- Rows after cleanup: **20,640**
- Columns: **10**
- Target: `median_house_value`
- Duplicate rows removed: **0**

## Missing values before preprocessing
- `longitude`: 0
- `latitude`: 0
- `housing_median_age`: 0
- `total_rooms`: 0
- `total_bedrooms`: 207
- `population`: 0
- `households`: 0
- `median_income`: 0
- `median_house_value`: 0
- `ocean_proximity`: 0

## Model results (20% test set)

| Model | MSE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 4,908,290,571.346 | 70,059.193 | 0.6254 |
| Lasso | 4,908,290,725.357 | 70,059.194 | 0.6254 |
| Ridge | 4,909,247,315.719 | 70,066.021 | 0.6254 |

Lowest RMSE in this run: **Linear Regression** (RMSE 70,059.193).
These results are calculated by this run and are not a guarantee of future performance.

## Interpretation checklist
- Describe the target distribution and important correlations.
- Does the actual-vs-predicted plot follow the diagonal?
- Are residuals randomly scattered or patterned?
- Compare RMSE and R² across Linear Regression, Ridge, and Lasso.

## Limitations
- This dataset represents California census-block groups, not Indian homes or individual property listings.
- Linear Regression may underfit non-linear relationships.
- Use the metrics generated on your own computer; do not invent or copy results.
