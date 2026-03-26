# Bank Marketing - ML Classification Pipeline

## Overview
End-to-end machine learning pipeline to predict whether a bank client will subscribe to a term deposit, based on a Portuguese bank marketing campaign dataset (UCI).

## Objectives
1. Predict client subscription to a term deposit (binary classification)
2. Handle class imbalance (22% positive cases)
3. Build a reproducible and automated sklearn pipeline

## Methodology
1. **Data Preparation** - Loading train (45,211 obs.) and test sets, variable typing
2. **EDA** - Univariate and bivariate analysis, correlation matrix (Pearson & Spearman), KDE plots
3. **Preprocessing** - StandardScaler (numerical), OneHotEncoder drop='first' (categorical), ColumnTransformer
4. **Feature Selection** - SelectKBest with Mutual Information (k=15) to detect non-linear relationships
5. **Modeling** - RandomForestClassifier (200 trees, random_state=42)
6. **Evaluation** - Accuracy, precision, recall, F1-score, confusion matrix

## Key Results
- **Validation accuracy:** 90.5%
- **Test accuracy:** 98.2%
- **F1-score (test):** 0.95
- Class imbalance identified and analyzed (22% positives)

## Tools & Libraries
- Python, pandas, NumPy
- scikit-learn (Pipeline, ColumnTransformer, SelectKBest, RandomForestClassifier)
- Seaborn, Mat
