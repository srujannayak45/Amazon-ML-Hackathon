"""
Feature extraction utilities for the product pricing model.
"""

import os
import pandas as pd
import numpy as np
import re
from typing import Dict, Any, Union, List
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, RobustScaler
import cv2
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Feature extractor for product pricing model."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize feature extractor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.text_col = config.get('data', {}).get('text_column', 'catalog_content')
        self.image_dir = config.get('paths', {}).get('image_dir')
        
        # Text processing
        self.max_features = config.get('features', {}).get('text', {}).get('max_features', 5000)
        self.embedding_model_name = config.get('features', {}).get('text', {}).get('embedding_model', 'all-MiniLM-L6-v2')
        self.use_embeddings = config.get('features', {}).get('text', {}).get('use_embeddings', True)
        
        # Initialize components
        self.text_vectorizer = None
        self.embedding_model = None
        self.category_encoder = None
        self.scaler = None
        
        # Initialize embedding model if specified
        if self.use_embeddings:
            try:
                logger.info(f"Loading embedding model: {self.embedding_model_name}")
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                self.use_embeddings = False
    
    def _preprocess_text(self, text: str) -> str:
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
    
    def _extract_text_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Extract text features from dataframe.
        
        Args:
            df: Input dataframe
            fit: Whether to fit or just transform
            
        Returns:
            Dataframe with text features
        """
        logger.info("Extracting text features")
        
        if self.text_col not in df.columns:
            logger.warning(f"Text column {self.text_col} not found in dataframe")
            return pd.DataFrame(index=df.index)
        
        # Preprocess text
        texts = df[self.text_col].fillna("").apply(self._preprocess_text).values
        
        features = {}
        
        # TF-IDF features
        if fit:
            logger.info("Fitting TF-IDF vectorizer")
            self.text_vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                min_df=2,
                max_df=0.9,
                ngram_range=(1, 2),
                stop_words='english'
            )
            tfidf_features = self.text_vectorizer.fit_transform(texts)
        else:
            if self.text_vectorizer is not None:
                tfidf_features = self.text_vectorizer.transform(texts)
            else:
                logger.warning("TF-IDF vectorizer not initialized")
                tfidf_features = None
        
        if tfidf_features is not None:
            # Convert sparse matrix to dense if needed
            if hasattr(tfidf_features, 'toarray'):
                tfidf_features = tfidf_features.toarray()
                
            for i in range(tfidf_features.shape[1]):
                features[f'tfidf_{i}'] = tfidf_features[:, i]
        
        # Extract text embeddings
        if self.use_embeddings and self.embedding_model is not None:
            logger.info("Generating text embeddings")
            try:
                embeddings = self.embedding_model.encode(
                    texts,
                    batch_size=32,
                    show_progress_bar=True,
                    convert_to_numpy=True
                )
                
                for i in range(embeddings.shape[1]):
                    features[f'embedding_{i}'] = embeddings[:, i]
                    
            except Exception as e:
                logger.error(f"Error generating embeddings: {e}")
        
        # Create feature dataframe
        features_df = pd.DataFrame(features, index=df.index)
        logger.info(f"Generated {features_df.shape[1]} text features")
        
        return features_df
    
    def _extract_image_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract image features from dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with image features
        """
        if self.image_dir is None or not os.path.exists(self.image_dir):
            logger.warning("Image directory not specified or doesn't exist")
            return pd.DataFrame(index=df.index)
        
        logger.info("Extracting image features")
        
        features = {}
        sample_ids = df['sample_id'].values
        
        for sample_id in tqdm(sample_ids, desc="Processing images"):
            image_path = os.path.join(self.image_dir, f"{sample_id}.jpg")
            
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
                
                # Extract basic color statistics for each channel
                for i, color in enumerate(['red', 'green', 'blue']):
                    features.setdefault(f'img_{color}_mean', []).append(np.mean(img[:,:,i]))
                    features.setdefault(f'img_{color}_std', []).append(np.std(img[:,:,i]))
                    features.setdefault(f'img_{color}_min', []).append(np.min(img[:,:,i]))
                    features.setdefault(f'img_{color}_max', []).append(np.max(img[:,:,i]))
                
                # Extract brightness and contrast
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                features.setdefault('img_brightness', []).append(np.mean(gray))
                features.setdefault('img_contrast', []).append(np.std(gray))
                
                # Extract edge features
                edges = cv2.Canny(gray, 100, 200)
                features.setdefault('img_edge_mean', []).append(np.mean(edges))
                features.setdefault('img_edge_std', []).append(np.std(edges))
                features.setdefault('img_edge_count', []).append(np.sum(edges > 0))
                
            except Exception as e:
                logger.error(f"Error processing image {sample_id}: {e}")
        
        # Create feature dataframe if we have any features
        if features:
            # Find images with features
            image_ids = []
            feature_list = list(features.values())[0]
            for i, sample_id in enumerate(sample_ids):
                if i < len(feature_list):
                    image_ids.append(sample_id)
            
            # Convert lists to arrays
            for key in features:
                features[key] = np.array(features[key])
            
            # Create dataframe with sample_id index
            img_df = pd.DataFrame(features)
            img_df['sample_id'] = image_ids
            
            # Merge with original dataframe
            result_df = pd.merge(
                df[['sample_id']],
                img_df,
                on='sample_id',
                how='left'
            )
            
            # Remove sample_id column from result
            if 'sample_id' in result_df.columns:
                result_df = result_df.drop('sample_id', axis=1)
            
            logger.info(f"Generated {img_df.shape[1]} image features for {len(image_ids)} images")
            
            return result_df
        else:
            logger.warning("No image features were generated")
            return pd.DataFrame(index=df.index)
    
    def transform(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Extract all features from dataframe.
        
        Args:
            df: Input dataframe
            fit: Whether to fit or just transform
            
        Returns:
            Dataframe with all features
        """
        # Extract text features
        text_features = self._extract_text_features(df, fit)
        
        # Extract image features
        image_features = self._extract_image_features(df)
        
        # Combine features
        combined = pd.concat([text_features, image_features], axis=1)
        
        # Scale features
        if fit:
            logger.info("Fitting feature scaler")
            self.scaler = RobustScaler()
            scaled_features = self.scaler.fit_transform(combined.fillna(0))
        else:
            if self.scaler is not None:
                scaled_features = self.scaler.transform(combined.fillna(0))
            else:
                logger.warning("Feature scaler not initialized")
                scaled_features = combined.fillna(0).values
        
        # Convert to dataframe
        scaled_df = pd.DataFrame(
            scaled_features,
            index=df.index,
            columns=combined.columns
        )
        
        logger.info(f"Generated a total of {scaled_df.shape[1]} features")
        
        return scaled_df