# 🧠 Predicting Alzheimer’s Disease Diagnosis Using Machine Learning Models

**Team Members:** Ken Li (cl7073), Mike Xia (zx1622)

---

##  Project Overview

This project develops and evaluates machine learning models to predict **Alzheimer’s disease (AD)** diagnosis using a structured clinical dataset of 2,149 patients.

We:

- preprocessed and explored the dataset  
- performed statistical tests  
- engineered features  
- trained multiple ML models  
- compared their performance  
- analyzed clinically meaningful feature importance  

Our goal is to build:

- an accurate predictive model  
- **and** an interpretable framework for identifying key risk factors associated with AD diagnosis

---

## Dataset

- **Source:** Kaggle  
  `rabieelkharoua/alzheimers-disease-dataset`
- **Samples:** 2,149 patients  
- **Original features:** 35 

### Data processing steps

- removed:
  - PatientID  
  - DoctorInCharge  
- verified missing values → none  
- categorized features into:
  - binary  
  - multiclass categorical  
  - numerical  
- one-hot encoded:
  - Ethnicity  
  - EducationLevel  
- feature count increased **32 → 36**

---

## Methods

### Data Preparation

- train/test split (80/20, stratified, random_state=42)  
- standardized numerical variables using **StandardScaler**
- avoided data leakage (fit on train only)
- PCA visualization (2D)

> PCA explained ≈ **8.06%** of variance  
> → strong overlap between AD and non-AD  
> → poor linear separability in low dimensions

---

## Statistical Analysis

We compared AD vs non-AD groups:

- **Welch t-test (MMSE)**
  - significant difference (*p < 0.05*)
  - AD mean ≈ 12
  - non-AD mean ≈ 16.3

- **Pearson correlation (FunctionalAssessment)**
  - r ≈ −0.36, *p << 0.05*
  - better function → lower AD probability

- **Chi-square tests**
  - Smoking vs Diagnosis → not significant (p ≈ 0.86)
  - Age group vs Diagnosis → not significant (p ≈ 0.32)

---

## 🤖 Machine Learning Models

We trained and evaluated:

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest (baseline RF1)
- Random Forest (tuned RF2)
- XGBoost

### Evaluation metrics

- K-Fold cross-validation
- accuracy
- confusion matrix
- **AUROC (primary metric)**

---

##  Model Performance Summary

| Model | AUROC | Key Observation |
|------|------|----------------|
| Logistic Regression | ~0.89 | interpretable, low FN |
| SVM | ~0.89 | higher false negatives |
| Random Forest (RF1) | 0.9409 | strong performance |
| Random Forest (RF2 – tuned) | **0.9419** | highest AUROC |
| XGBoost | 0.9400 | **fewest false negatives** |

### False negatives (clinical importance)

- Logistic Regression → 16  
- **SVM → 44 (worst)**  
- Random Forest RF1 → 17  
- Random Forest RF2 → 18  
- **XGBoost → 16 (best clinical result)**  

> In Alzheimer’s disease, **false negatives are worse than false positives**  
> because missing a true AD case delays diagnosis and treatment.

---

##  Feature Importance

Most important predictors across models:

- FunctionalAssessment
- Activities of Daily Living (ADL)
- MMSE score
- MemoryComplaints
- BehavioralProblems

These align with clinical understanding:

- functional decline  
- cognitive impairment  
- behavioral symptoms  

---

## Recommendation

We recommend **XGBoost** as the primary model:

- AUROC ≈ 0.94  
- **lowest false negatives**
- robust to nonlinear relationships
- interpretable feature importance

---

##  Future Work

- expand dataset size
- external dataset validation
- include longitudinal follow-ups
- integrate imaging/genomic biomarkers
- cost-sensitive training to penalize false negatives
