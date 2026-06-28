"""
Utility functions for the product pricing model.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Union, List
import yaml

logger = logging.getLogger(__name__)

def setup_logging(config: Dict[str, Any]) -> None:
    """
    Set up logging configuration.
    
    Args:
        config: Configuration dictionary
    """
    log_level = config.get('logging', {}).get('level', 'INFO').upper()
    log_format = config.get('logging', {}).get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create log directory if needed
    log_dir = config.get('paths', {}).get('log_dir', 'logs')
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'training.log')
    else:
        log_file = None
    
    # Configure logging
    handlers = []
    
    # Add file handler if log file is specified
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    # Add console handler
    handlers.append(logging.StreamHandler())
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=handlers
    )
    
    logger.info(f"Logging configured with level {log_level}")

def load_data(config: Dict[str, Any]) -> Tuple[pd.DataFrame, Union[pd.DataFrame, None]]:
    """
    Load data from CSV files.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_df, test_df)
    """
    # Get paths
    train_path = config.get('paths', {}).get('train_data', 'train.csv')
    test_path = config.get('paths', {}).get('test_data')
    
    logger.info(f"Loading training data from {train_path}")
    
    # Load training data
    try:
        train_df = pd.read_csv(train_path)
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        raise
    
    # Load test data if specified
    test_df = None
    if test_path:
        logger.info(f"Loading test data from {test_path}")
        try:
            test_df = pd.read_csv(test_path)
        except Exception as e:
            logger.warning(f"Error loading test data: {e}")
    
    # Apply preprocessing
    train_df = preprocess_data(train_df, config, is_train=True)
    
    if test_df is not None:
        test_df = preprocess_data(test_df, config, is_train=False)
    
    return train_df, test_df

def preprocess_data(df: pd.DataFrame, config: Dict[str, Any], is_train: bool = True) -> pd.DataFrame:
    """
    Preprocess data.
    
    Args:
        df: Input dataframe
        config: Configuration dictionary
        is_train: Whether this is training data
        
    Returns:
        Preprocessed dataframe
    """
    logger.info(f"Preprocessing {'training' if is_train else 'test'} data with shape {df.shape}")
    
    # Copy dataframe to avoid modifying original
    df = df.copy()
    
    # Fill missing values
    text_col = config.get('data', {}).get('text_column', 'catalog_content')
    if text_col in df.columns:
        df[text_col] = df[text_col].fillna("")
    
    # Handle price column for training data
    target_col = config.get('data', {}).get('target_column', 'price')
    if is_train and target_col in df.columns:
        # Remove rows with missing target
        n_missing = df[target_col].isna().sum()
        if n_missing > 0:
            logger.info(f"Removing {n_missing} rows with missing target values")
            df = df.dropna(subset=[target_col])
        
        # Remove outliers if configured
        if config.get('preprocessing', {}).get('remove_outliers', False):
            df = remove_outliers(df, target_col, config)
    
    return df

def remove_outliers(df: pd.DataFrame, target_col: str, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Remove outliers from data.
    
    Args:
        df: Input dataframe
        target_col: Target column name
        config: Configuration dictionary
        
    Returns:
        Dataframe with outliers removed
    """
    method = config.get('preprocessing', {}).get('outlier_method', 'iqr')
    
    logger.info(f"Removing outliers using method: {method}")
    
    # Copy dataframe to avoid modifying original
    df = df.copy()
    
    if method == 'iqr':
        # IQR method
        multiplier = config.get('preprocessing', {}).get('outlier_params', {}).get('iqr_multiplier', 1.5)
        
        q1 = df[target_col].quantile(0.25)
        q3 = df[target_col].quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        logger.info(f"IQR bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        # Filter outliers
        n_before = len(df)
        df = df[(df[target_col] >= lower_bound) & (df[target_col] <= upper_bound)]
        n_removed = n_before - len(df)
        
        logger.info(f"Removed {n_removed} outliers ({n_removed/n_before:.2%} of data)")
        
    elif method == 'percentile':
        # Percentile method
        lower_pct = config.get('preprocessing', {}).get('outlier_params', {}).get('lower_percentile', 1)
        upper_pct = config.get('preprocessing', {}).get('outlier_params', {}).get('upper_percentile', 99)
        
        lower_bound = df[target_col].quantile(lower_pct / 100)
        upper_bound = df[target_col].quantile(upper_pct / 100)
        
        logger.info(f"Percentile bounds ({lower_pct}% - {upper_pct}%): [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        # Filter outliers
        n_before = len(df)
        df = df[(df[target_col] >= lower_bound) & (df[target_col] <= upper_bound)]
        n_removed = n_before - len(df)
        
        logger.info(f"Removed {n_removed} outliers ({n_removed/n_before:.2%} of data)")
    
    return df

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        SMAPE value
    """
    # Ensure predictions are positive
    y_pred = np.maximum(y_pred, 0)
    
    # Calculate SMAPE
    numerator = 2.0 * np.abs(y_pred - y_true)
    denominator = np.abs(y_pred) + np.abs(y_true)
    
    # Handle division by zero
    mask = denominator != 0
    
    if not np.any(mask):
        return 0.0
    
    smape_value = np.mean(numerator[mask] / denominator[mask]) * 100.0
    
    return smape_value

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Evaluate predictions.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Evaluation metric
    """
    # If inputs are log-transformed, exponentiate them
    if np.mean(y_true) < 10 or np.mean(y_pred) < 10:
        y_true_original = np.expm1(y_true)
        y_pred_original = np.expm1(y_pred)
    else:
        y_true_original = y_true
        y_pred_original = y_pred
    
    # Calculate SMAPE
    result = smape(y_true_original, y_pred_original)
    
    return result