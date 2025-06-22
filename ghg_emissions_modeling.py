# ghg_emissions_modeling.py

# 📦 Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings("ignore")

# 📥 Step 2: Load Dataset
file_path = "emission.csv"  # 📁 Replace with your actual filename

clean_lines = []
with open(file_path, encoding='latin1', errors='ignore') as f:
    for line in f:
        if line.count(',') >= 5:  # crude check: adjust based on your data's expected number of columns
            clean_lines.append(line)

with open("cleaned_emissions.csv", "w", encoding='latin1') as f:
    f.writelines(clean_lines)

# Now load the cleaned version
df = pd.read_csv("cleaned_emissions.csv")
print("✅ Cleaned and loaded successfully!")
print(df.head())


# 📊 Step 3: Exploratory Data Analysis (EDA) & Preprocessing
print("\n📌 Basic Info:")
print(df.info())

print("\n🔍 Checking Missing Values:")
print(df.isnull().sum())

# Drop rows with missing target values
df.dropna(subset=['GHG_Emissions_kgCO2e'], inplace=True)

# Fill or drop missing values for predictors (if any)
df.fillna(method='ffill', inplace=True)

# 🎯 Define Features and Target
X = df.drop(columns=['GHG_Emissions_kgCO2e'])
y = df['GHG_Emissions_kgCO2e']

# Identify categorical and numerical features
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# ⚙️ Step 4: Preprocessing Pipeline
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# ⚙️ Step 5: Modeling Pipelines
models = {
    "Linear Regression": Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ]),
    "Random Forest": Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])
}

# 🔀 Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📈 Step 7: Training, Prediction, and Evaluation
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2 Score": round(r2, 4)
    })

    print(f"\n📊 {name} Performance:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")

# 📋 Step 8: Comparative Study
results_df = pd.DataFrame(results)
print("\n✅ Model Comparison:\n", results_df)

# Step 9: Optional - Hyperparameter Tuning for Random Forest
param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [None, 10, 20]
}

rf_pipeline = models["Random Forest"]

grid_search = GridSearchCV(rf_pipeline, param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("\n🔧 Best Parameters for Random Forest:")
print(grid_search.best_params_)

best_model = grid_search.best_estimator_
y_best_pred = best_model.predict(X_test)

print(f"\n📈 Tuned Random Forest R2 Score: {r2_score(y_test, y_best_pred):.4f}")
