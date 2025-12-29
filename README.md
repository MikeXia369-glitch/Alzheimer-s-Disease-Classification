# Alzheimer’s Disease Classification

This repository contains a machine learning pipeline for classifying Alzheimer’s disease using clinical and demographic data. The project was completed as part of a course project and focuses on building interpretable models and evaluating clinical implications such as false negatives.

## 📌 Project Goals
- Classify patients into diagnostic categories related to Alzheimer’s disease
- Compare multiple machine learning models
- Identify the most important predictive features
- Balance performance with clinical relevance (especially missing positive cases)

## 🧠 Dataset
- Structured tabular clinical dataset
- Includes:
  - demographic features
  - cognitive assessment scores
  - medical history features
- Sensitive data removed and not included in this repository

> Due to privacy constraints, raw data cannot be redistributed.  
> Code and pipeline structure are fully reproducible with similar tabular datasets.

## 🛠 Methods

### Data Preprocessing
- handled missing values
- encoded categorical variables (one-hot encoding)
- standardized numerical features
- train/validation/test split
- class imbalance checked

### Exploratory Data Analysis (EDA)
- distribution of features
- correlation heatmaps
- comparison by diagnosis group

### Models Implemented
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- XGBoost

### Evaluation Metrics
- Accuracy
- AUROC
- Confusion Matrix
- Precision / Recall / F1
- Sensitivity vs. Specificity tradeoff

### Model Interpretation
- permutation feature importance
- top 5 most important predictors
- discussion of clinical meaning
- emphasis on the cost of false negatives

## 📊 Key Results
- XGBoost achieved the best AUROC
- Random Forest performed well but less interpretable
- Logistic regression provided strongest interpretability
- False negatives prioritized in model selection

> Final chosen model: **XGBoost**, due to high recall on positive Alzheimer’s cases.
