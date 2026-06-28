"""
Utility functions for inference.
"""

import os
import numpy as np
import pandas as pd
import joblib
import cv2
from tqdm import tqdm
import re
from typing import Dict, List, Tuple, Any, Union
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

def load_models(model_dir: str) -> Tuple[Dict, Dict]:
    """
    Load models and preprocessors from disk.
    
    Args:
        model_dir: Directory containing model files
        
    Returns:
        models: Dictionary of loaded models
        preprocessors: Dictionary of preprocessors
    """
    logger.info(f"Loading models from {model_dir}")
    
    models = {}
    preprocessors = {}
    
    # Load base models
    base_models_path = os.path.join(model_dir, 'base_models.joblib')
    if os.path.exists(base_models_path):
        models['base'] = joblib.load(base_models_path)
        logger.info(f"Loaded base models")
    
    # Load meta model
    meta_model_path = os.path.join(model_dir, 'meta_model.joblib')
    if os.path.exists(meta_model_path):
        models['meta'] = joblib.load(meta_model_path)
        logger.info(f"Loaded meta model")
    
    # Load text preprocessor
    text_preprocessor_path = os.path.join(model_dir, 'text_preprocessor.joblib')
    if os.path.exists(text_preprocessor_path):
        preprocessors['text'] = joblib.load(text_preprocessor_path)
        logger.info(f"Loaded text preprocessor")
    
    # Load category preprocessor
    category_preprocessor_path = os.path.join(model_dir, 'category_preprocessor.joblib')
    if os.path.exists(category_preprocessor_path):
        preprocessors['category'] = joblib.load(category_preprocessor_path)
        logger.info(f"Loaded category preprocessor")
    
    # Load feature scaler
    scaler_path = os.path.join(model_dir, 'feature_scaler.joblib')
    if os.path.exists(scaler_path):
        preprocessors['scaler'] = joblib.load(scaler_path)
        logger.info(f"Loaded feature scaler")
    
    # Load embedding model
    embedding_model_path = os.path.join(model_dir, 'embedding_config.joblib')
    if os.path.exists(embedding_model_path):
        embedding_config = joblib.load(embedding_model_path)
        model_name = embedding_config.get('model_name', 'all-MiniLM-L6-v2')
        try:
            preprocessors['embeddings'] = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
        except Exception as e:
            logger.warning(f"Could not load embedding model {model_name}: {e}")
    
    return models, preprocessors

def preprocess_text(text: str) -> str:
    """
    Preprocess text data.
    
    Args:
        text: Input text
        
    Returns:
        Preprocessed text
    """
    if pd.isna(text) or text is None:
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def extract_text_features(df: pd.DataFrame, preprocessors: Dict) -> pd.DataFrame:
    """
    Extract text features from dataframe.
    
    Args:
        df: Input dataframe
        preprocessors: Dictionary of preprocessors
        
    Returns:
        Dataframe with text features
    """
    logger.info("Extracting text features")
    
    text_col = 'catalog_content'
    if text_col not in df.columns:
        logger.warning(f"Text column {text_col} not found in dataframe")
        return pd.DataFrame(index=df.index)
    
    # Clean text
    texts = df[text_col].fillna("").apply(preprocess_text).values
    
    features = {}
    
    # TF-IDF features
    if 'text' in preprocessors and hasattr(preprocessors['text'], 'transform'):
        logger.info("Extracting TF-IDF features")
        tfidf_features = preprocessors['text'].transform(texts)
        
        # Convert sparse matrix to dense if needed for certain models
        if hasattr(tfidf_features, 'toarray'):
            tfidf_features = tfidf_features.toarray()
            
        for i in range(tfidf_features.shape[1]):
            features[f'tfidf_{i}'] = tfidf_features[:, i]
    
    # Extract embeddings
    if 'embeddings' in preprocessors:
        logger.info("Extracting text embeddings")
        try:
            embeddings = preprocessors['embeddings'].encode(
                texts, 
                batch_size=32, 
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            for i in range(embeddings.shape[1]):
                features[f'embedding_{i}'] = embeddings[:, i]
                
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
    
    # Create feature dataframe
    if features:
        features_df = pd.DataFrame(features, index=df.index)
        logger.info(f"Generated {len(features)} text features")
        return features_df
    else:
        logger.warning("No text features were generated")
        return pd.DataFrame(index=df.index)

def extract_image_features(df: pd.DataFrame, image_dir: str) -> pd.DataFrame:
    """
    Extract image features from dataframe.
    
    Args:
        df: Input dataframe
        image_dir: Directory containing images
        
    Returns:
        Dataframe with image features
    """
    if image_dir is None or not os.path.exists(image_dir):
        logger.warning("Image directory not provided or does not exist")
        return pd.DataFrame(index=df.index)
    
    logger.info("Extracting image features")
    
    features = {}
    sample_ids = df['sample_id'].values
    
    for sample_id in tqdm(sample_ids, desc="Processing images"):
        image_path = os.path.join(image_dir, f"{sample_id}.jpg")
        
        # Skip if image doesn't exist
        if not os.path.exists(image_path):
            continue
        
        try:
            # Read and resize image
            img = cv2.imread(image_path)
            if img is None:
                continue
                
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Extract basic color statistics
            for i, color in enumerate(['red', 'green', 'blue']):
                features.setdefault(f'img_{color}_mean', []).append(np.mean(img[:,:,i]))
                features.setdefault(f'img_{color}_std', []).append(np.std(img[:,:,i]))
                
            # Extract brightness and contrast
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            features.setdefault('img_brightness', []).append(np.mean(gray))
            features.setdefault('img_contrast', []).append(np.std(gray))
            
        except Exception as e:
            logger.error(f"Error processing image {sample_id}: {e}")
    
    # Create feature dataframe
    if features:
        # Convert lists to arrays
        for key in features:
            features[key] = np.array(features[key])
            
        # Create dataframe with the correct index
        img_features_df = pd.DataFrame(features)
        
        # Map features back to original dataframe rows
        result = pd.DataFrame(index=df.index)
        for col in img_features_df.columns:
            result[col] = np.nan
            
        # Update only rows that have image features
        img_indices = df[df['sample_id'].isin(img_features_df.index)].index
        result.loc[img_indices, img_features_df.columns] = img_features_df.values
        
        logger.info(f"Generated {len(features)} image features for {len(img_indices)} images")
        return result
    else:
        logger.warning("No image features were generated")
        return pd.DataFrame(index=df.index)

def extract_features(df: pd.DataFrame, preprocessors: Dict, image_dir: str = None) -> pd.DataFrame:
    """
    Extract all features from dataframe.
    
    Args:
        df: Input dataframe
        preprocessors: Dictionary of preprocessors
        image_dir: Directory containing images
        
    Returns:
        Dataframe with all features
    """
    logger.info("Extracting features")
    
    # Extract text features
    text_features = extract_text_features(df, preprocessors)
    
    # Extract image features if image directory is provided
    image_features = extract_image_features(df, image_dir) if image_dir else pd.DataFrame(index=df.index)
    
    # Combine all features
    combined_features = pd.concat([text_features, image_features], axis=1)
    
    # Apply scaling if scaler is available
    if 'scaler' in preprocessors:
        scaler = preprocessors['scaler']
        scaled_features = scaler.transform(combined_features.fillna(0))
        combined_features = pd.DataFrame(scaled_features, index=df.index, columns=combined_features.columns)
    
    logger.info(f"Generated {combined_features.shape[1]} features")
    return combined_features

def make_predictions(features: pd.DataFrame, models: Dict) -> np.ndarray:
    """
    Make predictions using trained models.
    
    Args:
        features: Feature dataframe
        models: Dictionary of models
        
    Returns:
        Array of predictions
    """
    logger.info("Generating predictions")
    
    # Handle empty features
    if features.empty:
        logger.warning("Empty feature set, returning default predictions")
        return np.ones(features.shape[0]) * 19.99  # Default price
    
    base_preds = []
    
    # Generate predictions from base models
    if 'base' in models:
        base_models = models['base']
        logger.info(f"Using {len(base_models)} base models")
        
        for i, model in enumerate(base_models):
            try:
                pred = model.predict(features.fillna(0))
                base_preds.append(pred)
                logger.info(f"Base model {i+1} generated predictions")
            except Exception as e:
                logger.error(f"Error in base model {i+1}: {e}")
    
    # If we have base predictions and a meta model, use the meta model
    if base_preds and 'meta' in models:
        logger.info("Using meta model for final predictions")
        try:
            # Stack base predictions
            stacked_preds = np.column_stack(base_preds)
            
            # Get meta predictions
            meta_preds = models['meta'].predict(stacked_preds)
            
            # Transform predictions back to original scale if needed
            final_preds = np.expm1(meta_preds) if np.any(meta_preds < 10) else meta_preds
            
            # Ensure positive predictions
            final_preds = np.maximum(final_preds, 0.01)
            
            logger.info("Meta model predictions generated")
            return final_preds
        except Exception as e:
            logger.error(f"Error in meta model: {e}")
    
    # If meta model fails or doesn't exist, average base predictions
    if base_preds:
        logger.info("Using averaged base model predictions")
        avg_preds = np.mean(base_preds, axis=0)
        
        # Transform predictions back to original scale if needed
        final_preds = np.expm1(avg_preds) if np.any(avg_preds < 10) else avg_preds
        
        # Ensure positive predictions
        final_preds = np.maximum(final_preds, 0.01)
        
        return final_preds
    
    # Fallback to default predictions
    logger.warning("No valid models found, returning default predictions")
    return np.ones(features.shape[0]) * 19.99  # Default price