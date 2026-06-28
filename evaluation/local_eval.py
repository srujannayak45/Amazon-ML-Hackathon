"""
Local evaluation script for product pricing predictions.
Usage: python local_eval.py --predictions path/to/predictions.csv --ground_truth path/to/ground_truth.csv
"""

import argparse
import pandas as pd
import numpy as np
import sys
import os
import logging
import json
from typing import Dict, Any, Tuple, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        SMAPE value
    """
    # Ensure arrays have the same shape
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    
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

def log_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate SMAPE on log-transformed values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Log SMAPE value
    """
    # Ensure arrays have the same shape
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    
    # Ensure values are positive for log transform
    y_true = np.maximum(y_true, 0.01)
    y_pred = np.maximum(y_pred, 0.01)
    
    # Log transform
    log_y_true = np.log1p(y_true)
    log_y_pred = np.log1p(y_pred)
    
    # Calculate SMAPE on log-transformed values
    numerator = 2.0 * np.abs(log_y_pred - log_y_true)
    denominator = np.abs(log_y_pred) + np.abs(log_y_true)
    
    # Handle division by zero
    mask = denominator != 0
    
    if not np.any(mask):
        return 0.0
    
    smape_value = np.mean(numerator[mask] / denominator[mask]) * 100.0
    
    return smape_value

def evaluate(pred_df: pd.DataFrame, true_df: pd.DataFrame) -> Dict[str, float]:
    """
    Evaluate predictions against ground truth.
    
    Args:
        pred_df: Predictions dataframe
        true_df: Ground truth dataframe
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating predictions...")
    
    # Verify dataframes have required columns
    for df, name in [(pred_df, 'predictions'), (true_df, 'ground_truth')]:
        if 'sample_id' not in df.columns:
            raise ValueError(f"sample_id column missing from {name} dataframe")
        if 'price' not in df.columns:
            raise ValueError(f"price column missing from {name} dataframe")
    
    # Merge on sample_id
    merged = pd.merge(pred_df, true_df, on='sample_id', suffixes=('_pred', '_true'))
    
    # Check if any samples are missing
    n_pred = len(pred_df)
    n_true = len(true_df)
    n_merged = len(merged)
    
    if n_merged < n_true:
        logger.warning(f"{n_true - n_merged} samples from ground truth not found in predictions")
    
    if n_merged < n_pred:
        logger.warning(f"{n_pred - n_merged} predicted samples not found in ground truth")
    
    # Extract true and predicted values
    y_true = merged['price_true'].values
    y_pred = merged['price_pred'].values
    
    # Calculate metrics
    metrics = {
        'smape': smape(y_true, y_pred),
        'log_smape': log_smape(y_true, y_pred),
        'mae': np.mean(np.abs(y_pred - y_true)),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        'rmse': np.sqrt(np.mean((y_pred - y_true) ** 2)),
        'n_samples': n_merged
    }
    
    # Calculate stats by price range
    price_ranges = [
        (0, 10),
        (10, 25),
        (25, 50),
        (50, 100),
        (100, float('inf'))
    ]
    
    for min_price, max_price in price_ranges:
        mask = (y_true >= min_price) & (y_true < max_price)
        if np.any(mask):
            range_name = f"price_{min_price}_{max_price if max_price != float('inf') else 'up'}"
            metrics[f"smape_{range_name}"] = smape(y_true[mask], y_pred[mask])
            metrics[f"n_samples_{range_name}"] = mask.sum()
    
    return metrics

def generate_report(metrics: Dict[str, float]) -> str:
    """
    Generate evaluation report.
    
    Args:
        metrics: Dictionary of evaluation metrics
        
    Returns:
        Formatted report
    """
    report = "\n" + "=" * 50 + "\n"
    report += "PRODUCT PRICING MODEL EVALUATION\n"
    report += "=" * 50 + "\n\n"
    
    report += f"Evaluated on {metrics['n_samples']} samples\n\n"
    
    report += "Overall Metrics:\n"
    report += f"  SMAPE:      {metrics['smape']:.2f}%\n"
    report += f"  Log SMAPE:  {metrics['log_smape']:.2f}%\n"
    report += f"  MAE:        ${metrics['mae']:.2f}\n"
    report += f"  MAPE:       {metrics['mape']:.2f}%\n"
    report += f"  RMSE:       ${metrics['rmse']:.2f}\n\n"
    
    report += "Performance by Price Range:\n"
    price_ranges = [
        (0, 10),
        (10, 25),
        (25, 50),
        (50, 100),
        (100, float('inf'))
    ]
    
    for min_price, max_price in price_ranges:
        range_name = f"price_{min_price}_{max_price if max_price != float('inf') else 'up'}"
        if f"smape_{range_name}" in metrics:
            label = f"${min_price}-${max_price if max_price != float('inf') else '+'}"
            report += f"  {label:<10} SMAPE: {metrics[f'smape_{range_name}']:.2f}%"
            report += f" (n={metrics[f'n_samples_{range_name}']})\n"
    
    report += "\n" + "=" * 50 + "\n"
    report += "TARGET ACHIEVEMENT:\n"
    if metrics['smape'] < 15:
        report += "✅ SUCCESS! SMAPE is below 15%\n"
        report += f"  {metrics['smape']:.2f}% vs 15% target\n"
        report += f"  Improvement: {(15 - metrics['smape']):.2f}% below target\n"
    else:
        report += "❌ TARGET NOT MET. SMAPE is above 15%\n"
        report += f"  {metrics['smape']:.2f}% vs 15% target\n"
        report += f"  Gap: {(metrics['smape'] - 15):.2f}% above target\n"
        
    report += "=" * 50 + "\n"
    
    return report

def main():
    """Main function to evaluate predictions."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Product Pricing Evaluation')
    parser.add_argument('--predictions', type=str, required=True, help='Path to predictions CSV')
    parser.add_argument('--ground_truth', type=str, required=True, help='Path to ground truth CSV')
    parser.add_argument('--output', type=str, default=None, help='Path to save evaluation results')
    args = parser.parse_args()
    
    # Check if files exist
    for path, name in [(args.predictions, 'predictions'), (args.ground_truth, 'ground truth')]:
        if not os.path.exists(path):
            logger.error(f"{name} file not found: {path}")
            sys.exit(1)
    
    # Load data
    try:
        pred_df = pd.read_csv(args.predictions)
        true_df = pd.read_csv(args.ground_truth)
        
        logger.info(f"Loaded predictions with shape: {pred_df.shape}")
        logger.info(f"Loaded ground truth with shape: {true_df.shape}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)
    
    # Evaluate predictions
    try:
        metrics = evaluate(pred_df, true_df)
    except Exception as e:
        logger.error(f"Error evaluating predictions: {e}")
        sys.exit(1)
    
    # Generate report
    report = generate_report(metrics)
    print(report)
    
    # Save evaluation results if output path is specified
    if args.output:
        try:
            with open(args.output, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Evaluation results saved to {args.output}")
            
            # Save report as text file
            report_path = args.output.replace('.json', '.txt')
            if report_path == args.output:
                report_path = args.output + '.txt'
            
            with open(report_path, 'w') as f:
                f.write(report)
            logger.info(f"Evaluation report saved to {report_path}")
        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")

if __name__ == '__main__':
    main()