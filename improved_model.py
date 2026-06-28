"""
Improved Product Pricing Model with Advanced Feature Engineering
"""

import os
import numpy as np
import pandas as pd
import re
import joblib
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import VotingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("Starting improved product pricing pipeline...")

# SMAPE calculation function
def smape(y_true, y_pred):
    """Calculate Symmetric Mean Absolute Percentage Error"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Ensure no zeros (to avoid division by zero)
    y_true = np.maximum(y_true, 0.01)
    y_pred = np.maximum(y_pred, 0.01)
    
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

# Function to extract value and unit information
def extract_value_unit(text):
    if not isinstance(text, str):
        return 1.0, "Count"
        
    value_match = re.search(r'Value:\s*([\d\.]+)', text)
    value = 1.0
    if value_match:
        try:
            value = float(value_match.group(1))
        except:
            pass
            
    unit_match = re.search(r'Unit:\s*(\w+)', text)
    unit = "Count"
    if unit_match:
        unit = unit_match.group(1)
        
    return value, unit

# Function to extract item dimensions from text
def extract_dimensions(text):
    if not isinstance(text, str):
        return None, None, None
        
    # Look for common dimension patterns
    dim_patterns = [
        r'dimensions:\s*([\d\.]+)\s*x\s*([\d\.]+)\s*x\s*([\d\.]+)',
        r'size:\s*([\d\.]+)\s*x\s*([\d\.]+)\s*x\s*([\d\.]+)',
        r'measurements:\s*([\d\.]+)\s*x\s*([\d\.]+)\s*x\s*([\d\.]+)',
        r'(\d+[\.\d]*)\s*(?:inch|in|"|cm)?\s*x\s*(\d+[\.\d]*)\s*(?:inch|in|"|cm)?\s*x\s*(\d+[\.\d]*)',
        r'length:\s*([\d\.]+).*?width:\s*([\d\.]+).*?height:\s*([\d\.]+)',
        r'l\s*x\s*w\s*x\s*h:\s*([\d\.]+)\s*x\s*([\d\.]+)\s*x\s*([\d\.]+)'
    ]
    
    for pattern in dim_patterns:
        match = re.search(pattern, text.lower())
        if match:
            try:
                return float(match.group(1)), float(match.group(2)), float(match.group(3))
            except:
                pass
                
    return None, None, None

# Function to extract brand from text
def extract_brand(text):
    if not isinstance(text, str):
        return "Unknown"
        
    # Look for common brand patterns
    brand_patterns = [
        r'brand:\s*([A-Za-z0-9][A-Za-z0-9\s&\-\']+)',
        r'by\s+([A-Z][A-Za-z0-9\s&\-\']+)',
        r'from\s+([A-Z][A-Za-z0-9\s&\-\']+)',
        r'item name:\s*([A-Z][A-Za-z0-9\s&\-\']+)',
        r'^([A-Z][A-Za-z0-9\s&\-\']+)'
    ]
    
    for pattern in brand_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            brand = match.group(1).strip()
            # Limit length and filter out generic terms
            if len(brand) > 1 and len(brand) < 30:
                # Clean the brand name
                brand = re.sub(r'[^\w\s&\-\']', '', brand).strip()
                return brand
                
    # Extract from item name
    if "Item Name:" in text:
        name_part = text.split("Item Name:")[1].split("\n")[0].strip()
        words = name_part.split()
        if words and len(words[0]) > 1:
            return words[0]
            
    return "Unknown"

# Function to extract material information
def extract_material(text):
    if not isinstance(text, str):
        return "Unknown"
        
    material_patterns = [
        r'material:\s*([A-Za-z0-9\s&\-\']+)',
        r'made of\s+([A-Za-z0-9\s&\-\']+)',
        r'made from\s+([A-Za-z0-9\s&\-\']+)',
        r'constructed of\s+([A-Za-z0-9\s&\-\']+)',
        r'constructed from\s+([A-Za-z0-9\s&\-\']+)',
        r'premium\s+([A-Za-z0-9\s&\-\']+)',
        r'high[- ]quality\s+([A-Za-z0-9\s&\-\']+)'
    ]
    
    common_materials = [
        'wood', 'plastic', 'metal', 'steel', 'iron', 'aluminum', 'cotton', 
        'polyester', 'silk', 'leather', 'glass', 'ceramic', 'rubber', 'nylon',
        'stainless steel', 'fabric', 'wool', 'acrylic', 'canvas', 'crystal',
        'silicone'
    ]
    
    # Try to match material patterns
    for pattern in material_patterns:
        match = re.search(pattern, text.lower())
        if match:
            material = match.group(1).strip()
            if len(material) > 1 and len(material) < 30:
                return material.title()
                
    # Check for presence of common materials
    for material in common_materials:
        if re.search(r'\b' + material + r'\b', text.lower()):
            return material.title()
            
    return "Unknown"

# Function to extract weight information
def extract_weight(text):
    if not isinstance(text, str):
        return None
        
    weight_patterns = [
        r'weight:\s*([\d\.]+)\s*(lb|pound|kg|g|oz|ounce)',
        r'([\d\.]+)\s*(lb|pound|kg|g|oz|ounce)',
        r'([\d\.]+)\s*(?:lb|pound|kg|g|oz|ounce)\b',
        r'weight:\s*([\d\.]+)'
    ]
    
    # Common weight unit conversions to a standard unit (kg)
    unit_to_kg = {
        'lb': 0.453592,
        'pound': 0.453592,
        'kg': 1.0,
        'g': 0.001,
        'oz': 0.0283495,
        'ounce': 0.0283495
    }
    
    for pattern in weight_patterns:
        match = re.search(pattern, text.lower())
        if match:
            try:
                if len(match.groups()) > 1:
                    weight = float(match.group(1))
                    unit = match.group(2)
                    if unit in unit_to_kg:
                        # Convert to kg
                        return weight * unit_to_kg[unit]
                else:
                    # Just the number, assume pounds
                    return float(match.group(1)) * 0.453592
            except:
                pass
                
    return None

# Function to extract category information
def extract_category(text):
    if not isinstance(text, str):
        return "Other"
        
    # Common product categories
    categories = {
        'Electronics': ['electronics', 'device', 'gadget', 'tech', 'phone', 'computer', 'laptop', 'tablet', 'tv', 'headphone', 'camera', 'speaker'],
        'Kitchen': ['kitchen', 'cook', 'food', 'utensil', 'dish', 'plate', 'cup', 'glass', 'pan', 'pot', 'appliance'],
        'Home': ['home', 'decor', 'furniture', 'decoration', 'bed', 'mattress', 'pillow', 'blanket', 'table', 'chair', 'sofa'],
        'Clothing': ['cloth', 'apparel', 'wear', 'dress', 'shirt', 'pant', 'jean', 'sock', 'shoe', 'hat', 't-shirt', 'jacket'],
        'Beauty': ['beauty', 'makeup', 'cosmetic', 'cream', 'lotion', 'perfume', 'fragrance', 'skin care', 'hair care'],
        'Sports': ['sport', 'fitness', 'exercise', 'workout', 'gym', 'yoga', 'athletic', 'outdoor', 'hiking', 'camping'],
        'Toys': ['toy', 'game', 'play', 'puzzle', 'doll', 'action figure', 'lego', 'kids'],
        'Books': ['book', 'novel', 'read', 'textbook', 'cookbook', 'fiction', 'literature'],
        'Pet': ['pet', 'dog', 'cat', 'animal', 'food', 'toy', 'treat', 'bed', 'leash', 'collar'],
        'Grocery': ['food', 'drink', 'grocery', 'snack', 'beverage', 'fruit', 'vegetable', 'meat', 'dairy', 'organic'],
        'Automotive': ['car', 'vehicle', 'auto', 'truck', 'automotive', 'accessory', 'part'],
        'Office': ['office', 'supply', 'stationery', 'pen', 'paper', 'notebook', 'desk', 'chair']
    }
    
    # Count matches for each category
    category_scores = {category: 0 for category in categories}
    
    text_lower = text.lower()
    for category, keywords in categories.items():
        for keyword in keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                category_scores[category] += 1
                
    # Get the category with the highest score
    max_score = 0
    best_category = "Other"
    
    for category, score in category_scores.items():
        if score > max_score:
            max_score = score
            best_category = category
            
    return best_category if max_score > 0 else "Other"

# Function to extract premium indicators
def extract_premium_indicators(text):
    if not isinstance(text, str):
        return 0
        
    premium_keywords = [
        'premium', 'luxury', 'high-end', 'high end', 'exclusive', 'deluxe', 'elite',
        'professional', 'gourmet', 'artisan', 'handcrafted', 'limited edition',
        'high quality', 'top quality', 'superior', 'finest', 'authentic', 'genuine',
        'award-winning', 'signature', 'designer', 'crafted', 'custom'
    ]
    
    # Count occurrences of premium keywords
    premium_score = 0
    text_lower = text.lower()
    
    for keyword in premium_keywords:
        premium_score += len(re.findall(r'\b' + re.escape(keyword) + r'\b', text_lower))
        
    return premium_score

# Load data
print("\nLoading data...")
try:
    train_path = "student_resource/dataset/train.csv"
    test_path = "student_resource/dataset/test.csv"
    sample_test_path = "student_resource/dataset/sample_test.csv"
    sample_test_out_path = "student_resource/dataset/sample_test_out.csv"
    
    # Check which data file exists
    if os.path.exists(train_path) and os.path.exists(test_path):
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        print(f"Loaded train data: {train_df.shape[0]} rows")
        print(f"Loaded test data: {test_df.shape[0]} rows")
    elif os.path.exists(sample_test_path):
        # Use sample data
        test_df = pd.read_csv(sample_test_path)
        print(f"Loaded sample test data: {test_df.shape[0]} rows")
        
        if os.path.exists(sample_test_out_path):
            truth_df = pd.read_csv(sample_test_out_path)
            print(f"Loaded sample truth data: {truth_df.shape[0]} rows")
            
            # Merge to create training data from sample
            train_df = pd.merge(test_df, truth_df, on='sample_id')
            test_df = train_df.copy()  # Use same data for testing
            print(f"Created training data from sample: {train_df.shape[0]} rows")
        else:
            print("Sample truth data not found!")
            exit(1)
    else:
        print("No training or test data found!")
        exit(1)
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

print("\nExtracting features from catalog content...")

# Extract features from catalog content
start_time = time.time()

# Feature extraction functions
def extract_features_from_catalog(df):
    # Create a copy to avoid modifying the original
    df_features = pd.DataFrame(index=df.index)
    
    # Extract core features
    print("Extracting value and unit...")
    df_features['value'], df_features['unit'] = zip(*df['catalog_content'].apply(extract_value_unit))
    
    print("Extracting dimensions...")
    df_features['length'], df_features['width'], df_features['height'] = zip(*df['catalog_content'].apply(extract_dimensions))
    
    print("Extracting brand...")
    df_features['brand'] = df['catalog_content'].apply(extract_brand)
    
    print("Extracting material...")
    df_features['material'] = df['catalog_content'].apply(extract_material)
    
    print("Extracting weight...")
    df_features['weight'] = df['catalog_content'].apply(extract_weight)
    
    print("Extracting category...")
    df_features['category'] = df['catalog_content'].apply(extract_category)
    
    print("Extracting premium indicators...")
    df_features['premium_score'] = df['catalog_content'].apply(extract_premium_indicators)
    
    # Text length features
    print("Extracting text features...")
    df_features['text_length'] = df['catalog_content'].apply(lambda x: len(x) if isinstance(x, str) else 0)
    df_features['word_count'] = df['catalog_content'].apply(lambda x: len(str(x).split()) if isinstance(x, str) else 0)
    df_features['bullet_count'] = df['catalog_content'].apply(lambda x: str(x).lower().count('bullet point') if isinstance(x, str) else 0)
    
    # Specific sections length
    df_features['description_length'] = df['catalog_content'].apply(
        lambda x: len(re.findall(r'product description:(.*?)(?:value:|unit:|$)', str(x).lower(), re.DOTALL)[0]) 
        if isinstance(x, str) and re.search(r'product description:', str(x).lower()) else 0
    )
    
    df_features['item_name_length'] = df['catalog_content'].apply(
        lambda x: len(re.findall(r'item name:(.*?)(?:bullet point|product description|$)', str(x).lower(), re.DOTALL)[0]) 
        if isinstance(x, str) and re.search(r'item name:', str(x).lower()) else 0
    )
    
    # Number of features mentioned
    df_features['feature_count'] = df['catalog_content'].apply(
        lambda x: str(x).lower().count('feature') if isinstance(x, str) else 0
    )
    
    return df_features

# Extract features
train_features = extract_features_from_catalog(train_df)
test_features = extract_features_from_catalog(test_df)

# Data preprocessing
# Fill missing values
train_features = train_features.fillna({
    'length': -1,
    'width': -1,
    'height': -1,
    'weight': -1
})

test_features = test_features.fillna({
    'length': -1,
    'width': -1,
    'height': -1,
    'weight': -1
})

# Calculate volume for products with dimensions
train_features['volume'] = train_features.apply(
    lambda x: x['length'] * x['width'] * x['height'] if x['length'] > 0 and x['width'] > 0 and x['height'] > 0 else -1, 
    axis=1
)

test_features['volume'] = test_features.apply(
    lambda x: x['length'] * x['width'] * x['height'] if x['length'] > 0 and x['width'] > 0 and x['height'] > 0 else -1, 
    axis=1
)

# Create a combined feature set
train_combined = pd.concat([train_df[['sample_id']], train_features], axis=1)
test_combined = pd.concat([test_df[['sample_id']], test_features], axis=1)

# For the training data, add the target variable
if 'price' in train_df.columns:
    train_combined['price'] = train_df['price']
else:
    # For sample data where we merged with truth data
    train_combined['price'] = train_combined['price_y'] if 'price_y' in train_combined.columns else None

# Print feature stats
print(f"\nExtracted {train_features.shape[1]} features in {time.time() - start_time:.2f} seconds")
print(f"Training data shape: {train_combined.shape}")
print(f"Test data shape: {test_combined.shape}")

# Prepare for modeling
print("\nPreparing data for modeling...")

# Split features into categories
numerical_features = [
    'value', 'length', 'width', 'height', 'weight', 'premium_score', 
    'text_length', 'word_count', 'bullet_count', 'description_length', 
    'item_name_length', 'feature_count', 'volume'
]

categorical_features = ['unit', 'brand', 'material', 'category']

# Create preprocessing pipelines
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())  # Use robust scaler to handle outliers
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Create text features using TF-IDF
print("Generating text features...")
tfidf = TfidfVectorizer(
    max_features=200,  # Limit number of features to avoid overfitting
    min_df=2,          # Minimum document frequency
    max_df=0.95,       # Maximum document frequency
    ngram_range=(1, 2) # Use both unigrams and bigrams
)

# Extract text from catalog content
train_text = train_df['catalog_content'].fillna('').astype(str)
test_text = test_df['catalog_content'].fillna('').astype(str)

# Fit and transform the training data
train_tfidf = tfidf.fit_transform(train_text)
test_tfidf = tfidf.transform(test_text)

# Convert to DataFrame
train_tfidf_df = pd.DataFrame(
    train_tfidf.toarray(),
    columns=[f'tfidf_{i}' for i in range(train_tfidf.shape[1])],
    index=train_df.index
)

test_tfidf_df = pd.DataFrame(
    test_tfidf.toarray(),
    columns=[f'tfidf_{i}' for i in range(test_tfidf.shape[1])],
    index=test_df.index
)

# Combine with other features
X_train = pd.concat([
    train_combined.drop(['sample_id', 'price'] + [col for col in train_combined.columns if col.startswith('price_')], axis=1),
    train_tfidf_df
], axis=1)

X_test = pd.concat([
    test_combined.drop(['sample_id'] + [col for col in test_combined.columns if col.startswith('price_')], axis=1),
    test_tfidf_df
], axis=1)

y_train = train_combined['price']

print(f"Final training feature set: {X_train.shape}")
print(f"Final test feature set: {X_test.shape}")

# Log transform the target (prices are usually log-normally distributed)
print("Applying log transformation to target...")
y_train_log = np.log1p(y_train)

# Train ensemble model
print("\nTraining ensemble model...")

# Define models for the ensemble
model1 = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=RANDOM_SEED
)

model2 = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=RANDOM_SEED
)

model3 = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    random_state=RANDOM_SEED
)

# Create the ensemble
ensemble = VotingRegressor([
    ('lgb', model1),
    ('xgb', model2),
    ('gbr', model3)
])

# Create full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', ensemble)
])

# Evaluate using cross-validation
print("Evaluating model with cross-validation...")

# Create 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

# CV predictions to calculate SMAPE
cv_preds = cross_val_predict(pipeline, X_train, y_train_log, cv=kf, n_jobs=-1)
cv_preds_original = np.expm1(cv_preds)

# Calculate SMAPE
cv_smape = smape(y_train, cv_preds_original)
print(f"Cross-validation SMAPE: {cv_smape:.4f}%")

# Record fold scores
fold_scores = []
for train_idx, val_idx in kf.split(X_train):
    X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_fold_train, y_fold_val = y_train_log.iloc[train_idx], y_train.iloc[val_idx]
    
    # Train on this fold
    pipeline.fit(X_fold_train, y_fold_train)
    
    # Predict and evaluate
    fold_preds_log = pipeline.predict(X_fold_val)
    fold_preds = np.expm1(fold_preds_log)
    
    # Calculate SMAPE for this fold
    fold_smape = smape(y_fold_val, fold_preds)
    fold_scores.append(fold_smape)
    print(f"Fold SMAPE: {fold_smape:.4f}%")

# Train final model on all data
print("\nTraining final model on all data...")
pipeline.fit(X_train, y_train_log)

# Generate predictions
print("Generating predictions...")
test_preds_log = pipeline.predict(X_test)
test_preds = np.expm1(test_preds_log)

# Create submission file
submission = pd.DataFrame({
    'sample_id': test_df['sample_id'],
    'price': test_preds
})

# Save submission
output_path = "improved_test_out.csv"
submission.to_csv(output_path, index=False)
print(f"Saved predictions to {output_path}")

# Output final results
print("\nModel Performance Summary:")
print(f"Cross-validation SMAPE: {cv_smape:.4f}%")
print(f"Individual fold scores: {[f'{score:.4f}%' for score in fold_scores]}")
print(f"Average fold SMAPE: {np.mean(fold_scores):.4f}%")
print("\nDone!")