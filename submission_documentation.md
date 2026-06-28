# ML Challenge 2025: Smart Product Pricing Solution

**Team Name:** Superbytes  
**Team Members:** Abhishek Kumar Singh
                  Maloth Srujan Nayak
                  Ashish Gupta
                  P Sai Lahari
**Submission Date:** 13/10/2025

---

## 1. Executive Summary

Our solution utilizes a multimodal approach combining text, category, and image features with gradient boosting models to predict product prices. By applying advanced feature extraction techniques and ensemble methods, we achieved a SMAPE of 14.09%, significantly outperforming the baseline and meeting the target improvement threshold of 15%.

---

## 2. Methodology Overview

### 2.1 Problem Analysis

We approached the pricing challenge as a multimodal regression problem where the goal was to accurately predict product prices based on their descriptions, categories, and images. Our EDA revealed strong correlations between pricing and certain textual features (e.g., brand mentions, premium-signaling words) as well as category hierarchies.

**Key Observations:**
- Product prices follow a log-normal distribution with significant skew
- Text descriptions contain valuable pricing signals (brand names, material quality, dimensions)
- Category is a strong predictor with hierarchical importance
- Image features provide complementary information not captured in text
- Data contains outliers that required robust preprocessing techniques

### 2.2 Solution Strategy

We developed a comprehensive multimodal pipeline that extracts features from different data modalities and combines them using ensemble methods. Our approach emphasizes feature engineering quality and model robustness rather than architectural complexity.

**Approach Type:** Ensemble with Cross-Modal Feature Fusion  
**Core Innovation:** Log-space cross-validation with target-encoded categorical features combined with text embeddings and visual feature extraction

---

## 3. Model Architecture

### 3.1 Architecture Overview

Our architecture consists of three main components:
1. Feature extraction module for each modality (text, category, image)
2. Feature fusion layer that combines all extracted features
3. Ensemble of gradient boosting models (LightGBM and XGBoost)

The system performs preprocessing, feature extraction, and model training in a pipeline that ensures consistent handling of both training and inference data.

```
[Input Data] → [Feature Extraction] → [Feature Fusion] → [Ensemble Model] → [Price Prediction]
    │                  │                      │                  │
    ├→ Text Data   →   ├→ Text Features   →   │                  │
    ├→ Category    →   ├→ Cat Features    →   ├→ Fused Vector →  ├→ LightGBM
    └→ Image Data  →   └→ Image Features  →   │                  └→ XGBoost
                                                                  └→ Meta-learner
```

### 3.2 Model Components

**Text Processing Pipeline:**
- Preprocessing steps: [Tokenization, stop-word removal, TF-IDF vectorization]
- Model type: [SentenceTransformer embeddings with TF-IDF features]
- Key parameters: [max_features=10000, ngram_range=(1,3)]

**Category Processing Pipeline:**
- Preprocessing steps: [Target encoding, hierarchical feature extraction]
- Model type: [One-hot encoding + target encoding]
- Key parameters: [smoothing=10, min_samples_leaf=5]

**Image Processing Pipeline:**
- Preprocessing steps: [Resize to 224x224, normalization, data augmentation]
- Model type: [Color histogram features + edge detection + ResNet50 embeddings]
- Key parameters: [bins=32, layer='avg_pool']

**Ensemble Configuration:**
- Base models: LightGBM (n_estimators=500, max_depth=8) and XGBoost (n_estimators=500, max_depth=8)
- Meta-learner: Ridge regression (alpha=1.0)
- Cross-validation: 5-fold stratified by price range

---

## 4. Model Performance

### 4.1 Validation Results
- **SMAPE Score:** 14.09%
- **Other Metrics:** 
  - MAE: 0.1334
  - RMSE: 0.1728
  - R²: 0.9542

**Cross-validation Results:**
| Fold | SMAPE (%) | MAE    |
|------|-----------|--------|
| 1    | 13.95     | 0.1328 |
| 2    | 14.23     | 0.1342 |
| 3    | 14.08     | 0.1335 |
| 4    | 13.87     | 0.1325 |
| 5    | 14.32     | 0.1347 |

**Performance by Category:**
Our model maintains consistent performance across most product categories, with slightly better results in Electronics (13.21% SMAPE) and Home & Kitchen (13.54% SMAPE), likely due to more structured pricing patterns in these categories.

---

## 5. Conclusion

Our multimodal ensemble approach successfully captures pricing signals from text, category, and image data, resulting in accurate price predictions across diverse product types. By applying log transformation, robust feature engineering, and gradient boosting models, we achieved a 14.09% SMAPE, meeting the competition target. Future improvements could include deeper semantic text understanding and more advanced image feature extraction techniques.

---

## Appendix

### A. Code artifacts

The submission package includes:
- `src/` - Main source code with model implementation
- `inference/` - Scripts for inference
- `notebooks/` - Jupyter notebooks with examples
- `models/` - Model files and metadata
- `evaluation/` - Evaluation scripts

### B. Additional Results

**Feature Importance:**
1. Category features (42.3%)
2. Text features (38.7%)
3. Image features (19.0%)

**Model Training Parameters:**
```
lightgbm_params = {
    'learning_rate': 0.05,
    'max_depth': 8,
    'num_leaves': 31,
    'n_estimators': 500
}

xgboost_params = {
    'learning_rate': 0.05,
    'max_depth': 8,
    'n_estimators': 500
}
```

---

**Note:** This submission represents the culmination of our research and development efforts for the ML Challenge 2025. All code and models are original work developed specifically for this competition.