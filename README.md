# Foodborne Disease Analysis using ML and Data Science Techniques  
**Author**: Aditya Singhal | **ID**: 23BHI10065  

---

## Project Overview  
This project leverages **Machine Learning (ML)** and **Data Science** techniques to analyze a dataset on **foodborne diseases**. The analysis aims to answer three key questions:  
1. Are foodborne disease outbreaks increasing or decreasing?  
2. Which contaminant is responsible for the most illnesses, hospitalizations, and deaths?  
3. What location for food preparation poses the greatest risk of foodborne illness?  

---

## Workflow Steps  

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Step 1: Data Loading and Preprocessing
data = pd.read_csv('/mnt/data/outbreaks.csv')
data.fillna(method='ffill', inplace=True)

# Step 2: Feature Engineering and Scaling
X = data.drop(columns=['target_column'])  # Replace 'target_column' with actual column
y = data['target_column']  # Replace 'target_column' with actual column
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Model Selection
rf_model = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=10)
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, max_depth=20)
rf_model.fit(X_scaled, y)
xgb_model.fit(X_scaled, y)

# Step 4: Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3)
grid_search_rf.fit(X_scaled, y)
print(f"Best Parameters for RF: {grid_search_rf.best_params_}")

# Step 5: Model Evaluation
y_pred_rf = rf_model.predict(X_scaled)
y_pred_xgb = xgb_model.predict(X_scaled)

rf_r2 = r2_score(y, y_pred_rf)
rf_mse = mean_squared_error(y, y_pred_rf)
rf_mae = mean_absolute_error(y, y_pred_rf)

xgb_r2 = r2_score(y, y_pred_xgb)
xgb_mse = mean_squared_error(y, y_pred_xgb)
xgb_mae = mean_absolute_error(y, y_pred_xgb)

print(f"Random Forest - R²: {rf_r2:.4f}, MSE: {rf_mse:.4f}, MAE: {rf_mae:.4f}")
print(f"XGBoost - R²: {xgb_r2:.4f}, MSE: {xgb_mse:.4f}, MAE: {xgb_mae:.4f}")

# Step 6: Answering Questions

# Q1: Are foodborne disease outbreaks increasing or decreasing?
data['year'] = pd.to_datetime(data['date_column']).dt.year  # Convert date to year
yearly_data = data.groupby('year')['outbreak_column'].sum()  # Replace 'outbreak_column' with actual column
plt.plot(yearly_data)
plt.title('Foodborne Disease Outbreaks Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Outbreaks')
plt.show()

print("Yearly Illnesses Trend:\n", yearly_data)

# Q2: Which contaminant has been responsible for the most illnesses, hospitalizations, and deaths?
contaminant_data = data.groupby('contaminant_column')[['illnesses', 'hospitalizations', 'deaths']].sum()
print("Top 5 Contaminants:\n", contaminant_data.sort_values(by='illnesses', ascending=False).head())

# Q3: What location for food preparation poses the greatest risk of foodborne illness?
location_data = data.groupby('location_column')[['illnesses', 'hospitalizations', 'deaths']].sum()
print("Top 5 High-Risk Locations:\n", location_data.sort_values(by='illnesses', ascending=False).head())



# Q1: Are foodborne disease outbreaks increasing or decreasing?
print("Yearly Illnesses Trend:")
print("""
Year
1998    27156
1999    24899
2000    26033
2001    25192
2002    24939
2003    23079
2004    29034
2005    19761
2006    28656
2007    20970
2008    23089
2009    13813
2010    15893
2011    14278
2012    14995
2013    13431
2014    13295
2015    15018
Name: Illnesses, dtype: int64
""")

# Q2: Which contaminant has been responsible for the most illnesses, hospitalizations, and deaths?
print("Top 5 Contaminants:")
print("""
                          Illnesses  Hospitalizations  Fatalities
Species                                                         
Unknown                      77954             967.0        27.0
Norovirus genogroup I        76406             668.0         2.0
Salmonella enterica          60018            6888.0        82.0
Norovirus genogroup II       38175             518.0         6.0
Clostridium perfringens      28734             106.0        12.0
""")

# Q3: What location for food preparation poses the greatest risk of foodborne illness?
print("Top 5 High-Risk Locations:")
print("""
Location
Restaurant                131970
Unknown                    66015
Catering Service           36044
Private Home/Residence     22564
Prison/Jail                20608
Name: Illnesses, dtype: int64
""")
