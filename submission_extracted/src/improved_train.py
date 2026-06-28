"""
Simple ML script to train a product pricing model and calculate SMAPE.
"""

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import time

# SMAPE calculation function
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

# Load sample data
data_path = 'C:/Users/sruja/OneDrive/Documents/Amazon_ML_Hackathon/student_resource/dataset/sample_test.csv'
ground_truth_path = 'C:/Users/sruja/OneDrive/Documents/Amazon_ML_Hackathon/student_resource/dataset/sample_test_out.csv'

try:
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded data with shape: {df.shape}")
    
    print(f"Loading ground truth from {ground_truth_path}")
    gt_df = pd.read_csv(ground_truth_path)
    print(f"Loaded ground truth with shape: {gt_df.shape}")
    
    # Merge datasets
    merged_df = pd.merge(df, gt_df, on='asin', how='inner')
    print(f"Merged dataset has {merged_df.shape[0]} rows")
    
    # Check if there's enough data for cross-validation
    if merged_df.shape[0] < 10:
        raise ValueError("Not enough data for cross-validation after merging")
        
    df = merged_df
    
except Exception as e:
    print(f"Error loading or merging real data: {e}")
    print("Creating synthetic data for demonstration.")
    np.random.seed(42)
    n_samples = 1000
    
    # Create synthetic features
    df = pd.DataFrame({
        'title': [f"Product {i}" for i in range(n_samples)],
        'description': [f"This is a description for product {i}" for i in range(n_samples)],
        'bullet_points': [f"Feature 1, Feature 2, Feature {i}" for i in range(n_samples)],
        'brand': np.random.choice(['BrandA', 'BrandB', 'BrandC', 'BrandD'], n_samples),
        'category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports'], n_samples),
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples),
        'price': 10 + 100 * np.random.random(n_samples)  # prices between 10 and 110
    })

# Print basic statistics
print("\nBasic statistics of the price column:")
print(df['price'].describe())

# Basic feature engineering
print("\nPerforming feature engineering...")
start_time = time.time()

# Feature columns by type
text_columns = []
categorical_columns = []
numeric_columns = []

for col in df.columns:
    if col in ['asin', 'price', 'index']:  # Skip ID and target columns
        continue
        
    if df[col].dtype == 'object':
        # Check if it's a long text field
        if df[col].str.len().mean() > 20:  # Arbitrary threshold for text vs categorical
            text_columns.append(col)
        else:
            categorical_columns.append(col)
    elif df[col].dtype in ['int64', 'float64']:
        numeric_columns.append(col)

print(f"Text columns: {text_columns}")
print(f"Categorical columns: {categorical_columns}")
print(f"Numeric columns: {numeric_columns}")

# Target variable
y = df['price'].values

# Log transform the target for better prediction
print("Applying log transform to target variable")
y_log = np.log1p(y)

# Create preprocessing pipeline
transformers = []

# Text features
for i, col in enumerate(text_columns):
    transformers.append(
        (f'text_{i}', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='')),
            ('tfidf', TfidfVectorizer(max_features=100))
        ]), [col])
    )

# Categorical features
if categorical_columns:
    transformers.append(
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_columns)
    )

# Numeric features
if numeric_columns:
    transformers.append(
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_columns)
    )

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder='drop'
)

# Create model pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('model', GradientBoostingRegressor(n_estimators=100, random_state=42))
])

print("Feature engineering completed in {:.2f} seconds".format(time.time() - start_time))

# Cross-validation
print("\nRunning cross-validation...")
start_time = time.time()

# Use at least 2 folds for small datasets, more for larger ones
n_splits = min(5, max(2, df.shape[0] // 50))
cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
fold_smapes = []

for i, (train_idx, val_idx) in enumerate(cv.split(df)):
    X_train, X_val = df.iloc[train_idx], df.iloc[val_idx]
    y_train, y_val = y_log[train_idx], y[val_idx]  # Log transform for training, original for evaluation
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions (and reverse log transform)
    y_pred_log = model.predict(X_val)
    y_pred = np.expm1(y_pred_log)  # Reverse the log transform
    
    # Calculate SMAPE
    fold_smape = smape(y_val, y_pred)
    fold_smapes.append(fold_smape)
    
    print(f"Fold {i+1} SMAPE: {fold_smape:.2f}%")

average_smape = np.mean(fold_smapes)
print("\nAverage SMAPE: {:.2f}%".format(average_smape))
print("Cross-validation completed in {:.2f} seconds".format(time.time() - start_time))

# Train final model on all data
print("\nTraining final model on all data...")
model.fit(df, y_log)  # Log transform target for training
print("Final model training completed!")

print("\nProduct Pricing Model - Performance Summary")
print("=" * 40)
print(f"Number of samples: {df.shape[0]}")
print(f"Text features used: {', '.join(text_columns[:3])}{'...' if len(text_columns) > 3 else ''}")
print(f"Categorical features used: {', '.join(categorical_columns[:3])}{'...' if len(categorical_columns) > 3 else ''}")
print(f"Numeric features used: {', '.join(numeric_columns[:3])}{'...' if len(numeric_columns) > 3 else ''}")
print(f"Cross-validated SMAPE: {average_smape:.2f}%")
print("=" * 40)

# Create directory if it doesn't exist
os.makedirs("../models", exist_ok=True)

# Save performance metrics
with open("../models/performance_metrics.txt", "w") as f:
    f.write(f"Cross-validated SMAPE: {average_smape:.2f}%\n")
    for i, smape_val in enumerate(fold_smapes):
        f.write(f"Fold {i+1} SMAPE: {smape_val:.2f}%\n")

print("\nDone!")