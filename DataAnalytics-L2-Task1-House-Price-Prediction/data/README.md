# Dataset
The script downloads the California Housing dataset on first run.

Primary source (CSV): https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv
Fallback source (Google ML Crash Course): https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv

The primary dataset includes `ocean_proximity` and demonstrates one-hot encoding. The fallback contains numeric fields only, so categorical encoding is skipped when that source is used. Target: `median_house_value`. This is California district-level census data, not Indian property pricing.
