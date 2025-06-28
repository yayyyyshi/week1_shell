# Week 2: Improved version of GHG emissions preprocessing and EDA
# Previous code from Week 1 is retained separately in week1.ipynb
# Week 2: GHG Emissions Data Preprocessing and Exploration
# Week 2: GHG Emissions Data Preprocessing and Exploration (Fixed)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Load the dataset (update path if necessary)
df = pd.read_csv("emission.csv")  

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Drop columns with more than 50% missing values
df = df.dropna(axis=1, thresh=len(df)*0.5)
print("Remaining columns:", df.columns.tolist())

# Basic dataframe info
print("\nDataFrame Info:")
df.info()

print("\nDataFrame Description:")
print(df.describe().T)

print("\nNull Values in Each Column:")
print(df.isnull().sum())

# Visualizing distribution of numerical features
numerical_cols = df.select_dtypes(include=np.number).columns
df[numerical_cols].hist(figsize=(15, 10), bins=30)
plt.tight_layout()
plt.show()

# Checking categorical variables
categorical_cols = df.select_dtypes(include='object').columns.tolist()
print("\nCategorical Columns:", categorical_cols)

# Unit
print("\nUnique values in 'Unit' with count:")
print(df['Unit'].value_counts())

# Substance
print("\nUnique values in 'Substance':")
print(df['Substance'].unique())

# Mapping categorical variables to integers
substance_mapping = {k: v for v, k in enumerate(df['Substance'].unique())}
df['Substance'] = df['Substance'].map(substance_mapping)

unit_mapping = {k: v for v, k in enumerate(df['Unit'].unique())}
df['Unit'] = df['Unit'].map(unit_mapping)

print("\nSubstance Mapping:", substance_mapping)
print("Unit Mapping:", unit_mapping)

# Data types and non-null counts after mapping
df.info()

# Checking uniqueness in 'Industry Code' and 'Industry Name'
print("\nUnique values in 'Industry Code':", df['Industry Code'].nunique())
print("Unique values in 'Industry Name':", df['Industry Name'].nunique())

# Reset index for plotting
df.reset_index(drop=True, inplace=True)

# Top 10 emitting industries by 'Supply Chain Emission Factors with Margins'
top10 = df.groupby('Industry Name')['Supply Chain Emission Factors with Margins'].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=top10.values, y=top10.index)
plt.title("Top 10 Emitting Industries")
plt.xlabel("Total Emissions")
plt.ylabel("Industry Name")
plt.tight_layout()
plt.show()

# Data overview
print("\nDataFrame Head:")
print(df.head())

print("\nDataFrame Shape:", df.shape)

# Feature set and target variable
X = df.drop(columns=['Supply Chain Emission Factors with Margins'])
y = df['Supply Chain Emission Factors with Margins']
print("\nFeatures:", X.columns.tolist())
print("Target Variable: 'Supply Chain Emission Factors with Margins'")

# Count plots for categorical variables
for col in ['Substance', 'Unit']:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=col, data=df)
    plt.title(f"Count Plot for {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()
