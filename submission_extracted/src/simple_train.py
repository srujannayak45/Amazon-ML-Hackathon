"""
Simple ML script to train a product pricing model and calculate SMAPE.
"""

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import time

# SMAPE calculation function
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

# Load sample data or use synthetic data if real data is not available
try:
    # Try to load real data
    data_path = '../student_resource/dataset/train.csv'
    if os.path.exists(data_path):
        print(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        print(f"Loaded data with shape: {df.shape}")
    else:
        # Create synthetic data if real data is not available
        print("Real data not found. Creating synthetic data for demonstration.")
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
        
except Exception as e:
    print(f"Error loading data: {e}")
    print("Creating minimal synthetic data")
    n_samples = 500
    df = pd.DataFrame({
        'text_feature': [f"Text {i}" for i in range(n_samples)],
        'numeric_feature': np.random.randn(n_samples),
        'price': 10 + 100 * np.random.random(n_samples)  # prices between 10 and 110
    })

# Print basic statistics
print("\nBasic statistics of the price column:")
print(df['price'].describe())

# Basic feature engineering
print("\nPerforming feature engineering...")
start_time = time.time()

# Feature extraction for text features
text_columns = [col for col in df.columns if df[col].dtype == 'object' and col != 'price']
if not text_columns:
    text_columns = ['text_feature']  # Use default for synthetic data

print(f"Text columns: {text_columns}")

# Numeric features
numeric_columns = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and col != 'price']
if not numeric_columns:
    numeric_columns = ['numeric_feature']  # Use default for synthetic data

print(f"Numeric columns: {numeric_columns}")

# Target variable
y = df['price'].values

# Create a preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(max_features=100), text_columns[0]),  # Process first text column
        ('num', StandardScaler(), numeric_columns)
    ],
    remainder='drop'
)

# Combine preprocessing with model
model = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

print("Feature engineering completed in {:.2f} seconds".format(time.time() - start_time))

# Cross-validation
print("\nRunning cross-validation...")
start_time = time.time()

cv = KFold(n_splits=5, shuffle=True, random_state=42)
fold_smapes = []

for i, (train_idx, val_idx) in enumerate(cv.split(df)):
    X_train, X_val = df.iloc[train_idx], df.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_val)
    
    # Calculate SMAPE
    fold_smape = smape(y_val, y_pred)
    fold_smapes.append(fold_smape)
    
    print(f"Fold {i+1} SMAPE: {fold_smape:.2f}%")

average_smape = np.mean(fold_smapes)
print("\nAverage SMAPE: {:.2f}%".format(average_smape))
print("Cross-validation completed in {:.2f} seconds".format(time.time() - start_time))

# Train final model on all data
print("\nTraining final model on all data...")
model.fit(df, y)
print("Final model training completed!")

print("\nProduct Pricing Model - Performance Summary")
print("=" * 40)
print(f"Number of samples: {df.shape[0]}")
print(f"Text features used: {', '.join(text_columns[:3])}{'...' if len(text_columns) > 3 else ''}")
print(f"Numeric features used: {', '.join(numeric_columns[:3])}{'...' if len(numeric_columns) > 3 else ''}")
print(f"Cross-validated SMAPE: {average_smape:.2f}%")
print("=" * 40)

# Save performance metrics
with open("../models/performance_metrics.txt", "w") as f:
    f.write(f"Cross-validated SMAPE: {average_smape:.2f}%\n")
    for i, smape_val in enumerate(fold_smapes):
        f.write(f"Fold {i+1} SMAPE: {smape_val:.2f}%\n")

print("\nDone!")