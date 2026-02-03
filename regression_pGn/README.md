# Regression Case Study – High-Dimensional Gene Expression Analysis (p > n)

This case study addresses a **supervised regression problem** in a high-dimensional setting 
where the number of predictors vastly exceeds the number of observations (p >> n). 

The analysis focuses on predicting liver toxicity biomarkers from gene expression data, 
comparing regularization techniques and 
ensemble methods specifically designed to handle **high-dimensional** regression, 
and evaluating model stability through **cross-validation**. 

---

## 1. Problem Statement

The objective is to predict **albumin levels (ALB.g/dL)** in liver tissue based on gene expression profiles. 

This is a **p > n problem** where: 
- n = 64 observations (patient samples) 
- p = 3,116 genes (predictors) 
- Ratio p/n ≈ 48.69

This extreme dimensionality presents unique challenges: 
- risk of overfitting 
- computational complexity 
- need for feature selection or regularization 
- model interpretability in high-dimensional space 

The goal is to identify a model that provides **robust predictions** and 
**interpretable biological insights** while managing the curse of dimensionality. 

---

## 2. Dataset

- Source: Liver toxicity study
  - `liver_clinic.csv` – clinical measurements (including target variable: Albumin) 
  - `liver_gene.csv` – gene expression data (3,116 genes) 
- Observations: 64 patient samples 
- Target variable: Albumin concentration (g/dL), range 4.40 – 5.60 
- Predictors: gene expression levels from microarray data 

The dataset allows for:
- exploration of gene-level predictors of liver function
- comparison of different strategies for handling high-dimensional regression
- identification of biologically relevant genes through feature selection stability analysis

---

## 3. Models Used (and Why)

Multiple regression models are trained and compared, 
specifically chosen for their ability to handle high-dimensional data:

- **Ridge Regression**  
  - L2 regularization 
  - shrinks coefficients but keeps all predictors
  Baseline for regularized linear models.

- **Lasso Regression**  
  - L1 regularization 
  - performs automatic feature selection by driving some coefficients to exactly zero. 
  Ideal for interpretability in high-dimensional settings.

- **Elastic Net**  
  - Combines L1 and L2 penalties
  - balances feature selection (like Lasso) with coefficient stability (like Ridge) 
  Particularly effective when predictors are correlated, as is common in gene expression data..

- **Random Forest Regressor**  
  Ensemble method that:
  - handles non-linear relationships
  - performs implicit feature selection
  - reduces overfitting through bootstrap aggregation

- **XGBoost Regressor**  
  Gradient boosting method that:
  - builds models sequentially to correct errors
  - includes built-in regularization
  - efficient with high-dimensional sparse data

**Feature selection strategy**: Inside each cross-validation fold, the top 100 genes are selected 
based on correlation with the target, 
reducing dimensionality from 3,116 to 100 predictors before model training.

Model selection is based on **cross-validated performance**, ensuring fair comparison and robustness assessment.

---

## 4. Evaluation Metrics

Regression performance is evaluated using:

- **Root Mean Squared Error (RMSE)** 
- **Mean Absolute Error (MAE)** 
- **R² score** 

Cross-validation is used to:
- assess model generalization in small-sample settings 
- evaluate stability across different data splits 
- avoid overfitting in the p >> n regime 
- support informed model comparison 
 
**Feature stability analysis** is performed to: 
- identify genes consistently selected across folds, 
- provide additional confidence in biological relevance. 

---

## 5. Key Results

- Ridge regression uses all 100 selected features, 
  providing stable but less interpretable coefficients.

- Lasso performs automatic feature selection, 
  reducing the model to a sparse subset of genes, enhancing interpretability.

- **Elastic Net achieves the best overall performance**, achieving:
  - better generalization than pure Lasso
  - more interpretable models than pure Ridge
  - stable coefficient estimates with correlated gene expression data.

- Ensemble methods (Random Forest, XGBoost) show competitive performance 
  but with reduced interpretability compared to linear models.

- **Feature stability analysis** identified genes consistently selected across cross-validation folds, 
  with several genes appearing in ≥80% of folds, 
  suggesting robust biological signals.

- Top predictive genes include both positive and negative associations with albumin levels,
  providing biological insights into liver function regulation.

The final selected model is saved and documented in the `results/` folder.

---

## 6. Main Takeaways

- Regularization is essential for regression in high-dimensional settings (p >> n).
- **Elastic Net provides an optimal balance** between 
  feature selection (Lasso) and coefficient stability (Ridge) 
  for gene expression data.
- Feature selection within cross-validation folds 
  prevents information leakage and provides realistic performance estimates.
- Cross-validation is critical for model evaluation 
  when sample size is small relative to predictor space.
- Feature stability analysis across folds helps 
  identify robust biological signals beyond single-model results.

---

## Outputs

The `results/` folder contains:
- cross-validation results (`cv_highdim_regression_results.csv`)
- summary tables comparing model performance (`summary_highdim_regression.csv`)
- the final trained model (`final_highdim_regression_enet.pkl`) 
- the list of selected genes for final model (`selected_features_regression.pkl`)
- the coefficient estimates for each gene in the final model (`gene_coefficients.csv`)
- a Jupyter notebook (`Regression_pGn_cancer.ipynb`) with the full analysis workflow 

---

This case study serves as a reference example of a **well-structured high-dimensional regression analysis** 
emphasizing methodological clarity, model comparison, and interpretability. 