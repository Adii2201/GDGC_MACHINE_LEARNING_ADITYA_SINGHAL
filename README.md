Foodborne Disease Analysis using ML and Data Science Techniques

Author: Aditya Singhal
ID: 23BHI10065

📌 Project Overview

This project leverages Machine Learning (ML) and Data Science techniques to analyze a dataset on foodborne diseases. The primary goal is to answer three key questions:

Are foodborne disease outbreaks increasing or decreasing?
Which contaminant is responsible for the most illnesses, hospitalizations, and deaths?
What location for food preparation poses the greatest risk of foodborne illness?

🛠️ Workflow Steps

1️⃣ Data Loading and Preprocessing
Load the dataset using pandas.
Handle missing values using forward fill (fillna(method='ffill')).

import pandas as pd
data = pd.read_csv('/mnt/data/outbreaks.csv')
data.fillna(method='ffill', inplace=True)

2️⃣ Feature Engineering and Scaling
Extract target and features for model building.
Standardize the features using StandardScaler.

from sklearn.preprocessing import StandardScaler
X = data.drop(columns=['target_column'])  # Replace with actual target column
y = data['target_column']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

3️⃣ Model Selection
Models used:
Random Forest Regressor
XGBoost Regressor

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
rf_model = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=10)
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, max_depth=20)
rf_model.fit(X_scaled, y)
xgb_model.fit(X_scaled, y)

4️⃣ Hyperparameter Tuning
Perform grid search for the Random Forest model.

from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3)
grid_search_rf.fit(X_scaled, y)
print(f"Best Parameters for RF: {grid_search_rf.best_params_}")

5️⃣ Model Evaluation
Evaluate models using R², MSE, and MAE.

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
y_pred_rf = rf_model.predict(X_scaled)
rf_r2 = r2_score(y, y_pred_rf)
rf_mse = mean_squared_error(y, y_pred_rf)
rf_mae = mean_absolute_error(y, y_pred_rf)

print(f"Random Forest - R²: {rf_r2:.4f}, MSE: {rf_mse:.4f}, MAE: {rf_mae:.4f}")

6️⃣ Answering the Questions
Q1: Are foodborne disease outbreaks increasing or decreasing?

Analyze yearly outbreak trends.

data['year'] = pd.to_datetime(data['date_column']).dt.year  # Replace with actual date column
yearly_data = data.groupby('year')['outbreak_column'].sum()  # Replace 'outbreak_column' with actual column
yearly_data.plot(title='Foodborne Disease Outbreaks Over Time', xlabel='Year', ylabel='Number of Outbreaks')
print("Yearly Illnesses Trend:\n", yearly_data)

Q2: Which contaminant is responsible for the most illnesses, hospitalizations, and deaths?

Group data by contaminants and calculate totals.

contaminant_data = data.groupby('contaminant_column')[['illnesses', 'hospitalizations', 'deaths']].sum()
top_contaminants = contaminant_data.sort_values(by='illnesses', ascending=False).head()
print("Top 5 Contaminants:\n", top_contaminants)

Q3: What location for food preparation poses the greatest risk?

Identify high-risk locations.

location_data = data.groupby('location_column')[['illnesses', 'hospitalizations', 'deaths']].sum()
top_locations = location_data.sort_values(by='illnesses', ascending=False).head()
print("Top 5 High-Risk Locations:\n", top_locations)

📊 Results

Yearly Illnesses Trend:

Year
1998    27156
1999    24899
2000    26033
...
2015    15018

Top 5 Contaminants:
                          Illnesses  Hospitalizations  Fatalities
Unknown                      77954             967.0        27.0
Norovirus genogroup I        76406             668.0         2.0
Salmonella enterica          60018            6888.0        82.0
Norovirus genogroup II       38175             518.0         6.0
Clostridium perfringens      28734             106.0        12.0

Top 5 High-Risk Locations:
Location
Restaurant                131970
Unknown                    66015
Catering Service           36044
Private Home/Residence     22564
Prison/Jail                20608

🔑 Key Insights

Outbreaks are generally decreasing over time.
Unknown and Norovirus-related contaminants are the most harmful.
Restaurants pose the highest risk for foodborne illnesses.

