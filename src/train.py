"""
Training script for the product pricing model.
Usage: python train.py --config configs/config.yaml
"""

import os
import argparse
import yaml
import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, Any, Tuple, List
from sklearn.model_selection import train_test_split

from utils import setup_logging, load_data, evaluate_predictions
from featurize import FeatureExtractor

# Import model implementations
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_base_models(X_train, y_train, X_valid, y_valid, config):
    """
    Train base models.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_valid: Validation features
        y_valid: Validation targets
        config: Configuration dictionary
        
    Returns:
        Dictionary of trained models
    """
    models = {}
    
    # LightGBM model
    if config.get('models', {}).get('use_lightgbm', True):
        logger.info("Training LightGBM model...")
        lgb_params = config.get('models', {}).get('lightgbm', {})
        
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
        
        model = lgb.train(
            params=lgb_params,
            train_set=lgb_train,
            valid_sets=[lgb_train, lgb_valid],
            num_boost_round=lgb_params.get('num_boost_round', 1000),
            early_stopping_rounds=lgb_params.get('early_stopping_rounds', 50),
            verbose_eval=100
        )
        
        models['lightgbm'] = model
        logger.info(f"LightGBM training completed, best iteration: {model.best_iteration}")
    
    # XGBoost model
    if config.get('models', {}).get('use_xgboost', True):
        logger.info("Training XGBoost model...")
        xgb_params = config.get('models', {}).get('xgboost', {})
        
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            early_stopping_rounds=xgb_params.get('early_stopping_rounds', 50),
            verbose=100
        )
        
        models['xgboost'] = model
        logger.info(f"XGBoost training completed, best iteration: {model.best_iteration}")
    
    # Random Forest model
    if config.get('models', {}).get('use_random_forest', False):
        logger.info("Training Random Forest model...")
        rf_params = config.get('models', {}).get('random_forest', {})
        
        model = RandomForestRegressor(**rf_params)
        model.fit(X_train, y_train)
        
        models['random_forest'] = model
        logger.info("Random Forest training completed")
    
    return models

def train_meta_model(base_preds_train, y_train, base_preds_valid, y_valid, config):
    """
    Train meta model for ensembling.
    
    Args:
        base_preds_train: Base model predictions on training data
        y_train: Training targets
        base_preds_valid: Base model predictions on validation data
        y_valid: Validation targets
        config: Configuration dictionary
        
    Returns:
        Trained meta model
    """
    logger.info("Training meta model...")
    meta_params = config.get('models', {}).get('meta', {})
    
    # Default to Ridge regression
    model = Ridge(
        alpha=meta_params.get('alpha', 1.0),
        random_state=config.get('random_seed', 42)
    )
    
    model.fit(base_preds_train, y_train)
    
    # Evaluate on validation data
    meta_preds = model.predict(base_preds_valid)
    meta_score = evaluate_predictions(y_valid, meta_preds)
    
    logger.info(f"Meta model training completed, validation score: {meta_score:.4f}")
    
    return model

def train_cv_models(X, y, feature_extractor, config):
    """
    Train models using cross-validation.
    
    Args:
        X: Input features
        y: Target values
        feature_extractor: Feature extractor object
        config: Configuration dictionary
        
    Returns:
        List of trained base models and meta model
    """
    # Get CV settings
    n_folds = config.get('training', {}).get('cv_folds', 5)
    random_seed = config.get('random_seed', 42)
    stratify = config.get('training', {}).get('stratify', True)
    
    # Create stratification bins if stratify is True
    if stratify:
        n_bins = config.get('training', {}).get('stratify_bins', 10)
        # Apply log transformation before binning to handle skewed distribution
        y_binned = pd.qcut(np.log1p(y), n_bins, labels=False, duplicates='drop')
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        splits = list(cv.split(X, y_binned))
    else:
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        splits = list(cv.split(X, y))
    
    # Initialize lists to store models and predictions
    base_models = []
    oof_predictions = np.zeros((len(X), 0))  # Out-of-fold predictions
    
    # Train models for each fold
    for fold, (train_idx, valid_idx) in enumerate(splits):
        logger.info(f"Training fold {fold+1}/{n_folds}")
        
        # Split data
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        
        # Extract features
        logger.info("Extracting features...")
        train_features = feature_extractor.transform(X_train, fit=True)
        valid_features = feature_extractor.transform(X_valid, fit=False)
        
        # Train base models
        logger.info("Training base models...")
        fold_models = train_base_models(train_features, y_train, valid_features, y_valid, config)
        
        # Generate out-of-fold predictions
        fold_preds = []
        for name, model in fold_models.items():
            pred = model.predict(valid_features)
            fold_preds.append(pred)
        
        # Stack predictions
        fold_stacked = np.column_stack(fold_preds)
        oof_predictions = np.vstack([
            oof_predictions, 
            np.zeros((oof_predictions.shape[0], fold_stacked.shape[1] - oof_predictions.shape[1]))
        ]) if oof_predictions.shape[1] < fold_stacked.shape[1] else oof_predictions
        
        oof_predictions[valid_idx, :fold_stacked.shape[1]] = fold_stacked
        
        # Add models to list
        base_models.append(fold_models)
        
        # Log fold results
        valid_preds = np.mean(fold_stacked, axis=1)
        valid_score = evaluate_predictions(y_valid, valid_preds)
        logger.info(f"Fold {fold+1} validation score: {valid_score:.4f}")
    
    # Train meta model on out-of-fold predictions
    meta_model = None
    if config.get('models', {}).get('use_meta', True) and oof_predictions.shape[1] > 0:
        # Split data for meta model
        meta_train_idx, meta_valid_idx = train_test_split(
            range(len(X)),
            test_size=0.2,
            random_state=random_seed,
            stratify=y_binned if stratify else None
        )
        
        meta_X_train = oof_predictions[meta_train_idx]
        meta_y_train = y.iloc[meta_train_idx]
        meta_X_valid = oof_predictions[meta_valid_idx]
        meta_y_valid = y.iloc[meta_valid_idx]
        
        meta_model = train_meta_model(meta_X_train, meta_y_train, meta_X_valid, meta_y_valid, config)
    
    return base_models, meta_model, feature_extractor

def save_models(base_models, meta_model, feature_extractor, output_dir):
    """
    Save trained models to disk.
    
    Args:
        base_models: List of base models
        meta_model: Meta model
        feature_extractor: Feature extractor
        output_dir: Directory to save models
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save base models
    joblib.dump(base_models, os.path.join(output_dir, 'base_models.joblib'))
    
    # Save meta model if available
    if meta_model is not None:
        joblib.dump(meta_model, os.path.join(output_dir, 'meta_model.joblib'))
    
    # Save feature extractor components
    if hasattr(feature_extractor, 'text_vectorizer'):
        joblib.dump(feature_extractor.text_vectorizer, os.path.join(output_dir, 'text_preprocessor.joblib'))
    
    if hasattr(feature_extractor, 'category_encoder'):
        joblib.dump(feature_extractor.category_encoder, os.path.join(output_dir, 'category_preprocessor.joblib'))
    
    if hasattr(feature_extractor, 'scaler'):
        joblib.dump(feature_extractor.scaler, os.path.join(output_dir, 'feature_scaler.joblib'))
    
    # Save embedding model config if available
    if hasattr(feature_extractor, 'embedding_model_name'):
        embedding_config = {'model_name': feature_extractor.embedding_model_name}
        joblib.dump(embedding_config, os.path.join(output_dir, 'embedding_config.joblib'))
    
    logger.info(f"Models and preprocessors saved to {output_dir}")

def main():
    """Main function to train the model."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Product Pricing Model Training')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='models/best_model', help='Directory to save models')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set up logging
    setup_logging(config)
    
    logger.info(f"Starting training process with config: {args.config}")
    
    # Load data
    train_df, test_df = load_data(config)
    
    logger.info(f"Loaded training data with shape: {train_df.shape}")
    if test_df is not None:
        logger.info(f"Loaded test data with shape: {test_df.shape}")
    
    # Prepare features and target
    feature_cols = [col for col in train_df.columns if col not in ['price', 'sample_id']]
    target_col = 'price'
    
    X = train_df[feature_cols]
    y = train_df[target_col]
    
    # Log transform target if specified
    if config.get('preprocessing', {}).get('log_transform', True):
        logger.info("Applying log transformation to target")
        y = np.log1p(y)
    
    # Initialize feature extractor
    feature_extractor = FeatureExtractor(config)
    
    # Train models
    base_models, meta_model, feature_extractor = train_cv_models(X, y, feature_extractor, config)
    
    # Save models
    save_models(base_models, meta_model, feature_extractor, args.output_dir)
    
    logger.info("Training process completed successfully")

if __name__ == '__main__':
    main()