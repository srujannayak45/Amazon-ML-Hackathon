"""
Inference script for the product pricing model.
Usage: python run_inference.py --input_path path/to/test.csv --output_path predictions.csv
"""

import os
import argparse
import pandas as pd
import numpy as np
import joblib
import logging
from predict_utils import load_models, extract_features, make_predictions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to run inference."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Product Pricing Inference')
    parser.add_argument('--input_path', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save predictions')
    parser.add_argument('--model_dir', type=str, default='../models/best_model', help='Directory containing model files')
    parser.add_argument('--image_dir', type=str, default=None, help='Directory containing product images')
    args = parser.parse_args()

    logger.info(f"Starting inference process on {args.input_path}")
    
    # Load data
    try:
        test_df = pd.read_csv(args.input_path)
        logger.info(f"Loaded test data with shape: {test_df.shape}")
    except Exception as e:
        logger.error(f"Failed to load test data: {e}")
        return
    
    # Load models
    try:
        models, preprocessors = load_models(args.model_dir)
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return
    
    # Extract features
    try:
        test_features = extract_features(
            test_df, 
            preprocessors, 
            image_dir=args.image_dir
        )
        logger.info("Feature extraction completed")
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        return
    
    # Make predictions
    try:
        predictions = make_predictions(test_features, models)
        logger.info(f"Generated predictions with shape: {predictions.shape}")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return
    
    # Create submission file
    submission = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': predictions
    })
    
    # Save predictions
    try:
        submission.to_csv(args.output_path, index=False)
        logger.info(f"Predictions saved to {args.output_path}")
    except Exception as e:
        logger.error(f"Failed to save predictions: {e}")
        return
    
    logger.info("Inference process completed successfully")

if __name__ == '__main__':
    main()