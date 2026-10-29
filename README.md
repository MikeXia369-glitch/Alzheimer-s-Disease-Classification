# Predicting Alzheimer's Disease Diagnosis with Machine Learning

This is our final project for 381. We used a Kaggle dataset of 2,149 patients and trained five classifiers to predict whether a patient has Alzheimer's disease (AD). Along the way we found a data leakage problem, and we ran a few checks to understand why the tree models did so well.

## Dataset

- Source: [Alzheimer's Disease Dataset](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset) on Kaggle (by Rabie El Kharoua)
- 2,149 patients, 35 columns: demographics, lifestyle, medical history, clinical measurements, cognitive/functional assessments, and symptoms
- Target: `Diagnosis` (0 = no AD, 1 = AD), about 35% positive
- No missing values

The notebook downloads the data automatically with `kagglehub`. If you already have the CSV, comment out that part and use `pd.read_csv("alzheimers_disease_data.csv")` instead.

## How to run

Open `381_Final_Project_Alzheimer.ipynb` in Google Colab or Jupyter and run all cells. Required packages:

```
pandas numpy matplotlib seaborn scipy scikit-learn xgboost kagglehub
```

## What we did

1. **EDA and statistical tests.** We looked at feature distributions and ran a Welch t-test (MMSE), a Pearson correlation (FunctionalAssessment) and chi-square tests (smoking, age group) to compare AD and non-AD patients.
2. **Preprocessing.** We dropped `PatientID` and `DoctorInCharge`, one-hot encoded `Ethnicity` and `EducationLevel` (32 → 36 features), did a stratified 80/20 train/test split, and fit `StandardScaler` on the training set only.
3. **Models.** Logistic Regression, SVM (RBF), two Random Forests with different hyperparameters, and XGBoost.
4. **Evaluation.** 5-fold stratified CV AUROC on the training set, then test AUROC, accuracy and confusion matrices on the held-out test set. For LR and SVM, scaling happens inside a pipeline so the CV folds don't leak.
5. **Feature importance.** Permutation importance for LR and SVM, and built-in importance for RF and XGBoost.
6. **Leakage and sanity checks.** Explained below.

## Results

| Model | CV AUROC | Test AUROC | Test Accuracy | False Negatives |
|---|---|---|---|---|
| Logistic Regression | 0.903 ± 0.012 | 0.886 | 0.816 | 41 |
| SVM | 0.891 ± 0.017 | 0.893 | 0.823 | 44 |
| Random Forest (rf1) | 0.954 ± 0.010 | 0.941 | 0.942 | 17 |
| Random Forest (rf2) | 0.955 ± 0.010 | 0.942 | 0.940 | 18 |
| **XGBoost** | **0.958 ± 0.011** | **0.944** | **0.947** | **14** |

The test set has 430 patients, 152 of them with AD.

The three tree models are clearly better than LR and SVM, but they're basically tied with each other on AUROC. We picked **XGBoost** as the final model because it misses the fewest real AD cases, and in a screening setting a false negative is the more costly mistake.

## Key findings

**PatientID was leaking information.** In an early version of the notebook we forgot to drop `PatientID`, and the tree models got around 0.98 to 0.99 AUROC, which seemed too good. When we compared CV AUROC with and without the ID, Random Forest went from 0.951 to 0.984 and XGBoost from 0.953 to 0.992. Logistic Regression didn't change. PatientID by itself only gets 0.54 AUROC, so trees seem to be combining ID ranges with the real features to pick up some pattern from how the data was generated. All the final results above are computed without it.

**Five features carry almost all the signal.** FunctionalAssessment, ADL, MMSE, MemoryComplaints and BehavioralProblems are the top five for every model. A Random Forest trained on only these five gets 0.949 AUROC, about the same as with all features. A Random Forest trained on everything *except* these five gets 0.513, which is basically random. Lifestyle factors, blood pressure, cholesterol, age, smoking and so on add nothing in this dataset.

## Limitations

- The dataset looks at least partly synthetic. Age and BMI are almost perfectly uniform, age shows no association with diagnosis, and the label is almost fully explained by a handful of scores. The AUROC numbers here probably wouldn't hold on real clinical data.
- The strongest predictors (functional assessments, MMSE) are part of how Alzheimer's actually gets diagnosed, so the model is closer to summarizing an existing assessment than predicting the disease early.
- 2,149 patients is a fairly small sample.

## Future work

- Validate on a real cohort such as ADNI.
- Tune the XGBoost classification threshold to reduce false negatives further.
- Train without the cognitive/functional scores to see whether lifestyle and medical history alone have any predictive value.
