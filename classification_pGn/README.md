# Classification Case Study – High-Dimensional Gene Expression Analysis (p > n)

This case study addresses a **supervised classification problem** in a high-dimensional setting 
where the number of predictors vastly exceeds the number of observations (p >> n). 

The analysis focuses on distinguishing colon tumor tissue from normal tissue based on gene expression profiles, 
comparing regularization techniques and 
ensemble methods specifically designed to handle **high-dimensional** classification, 
and evaluating model stability through **cross-validation**. 

---

## 1. Problem Statement

The objective is to classify tissue samples as **tumor** or **normal** 
based on gene expression data from colon cancer patients. 

This is a **p > n problem** where:
- n = 62 observations (tissue samples) 
- p = 2,000 genes (predictors) 
- Ratio p/n ≈ 32.26 
- Classes: tumor (40 samples) vs. normal (22 samples) 

This extreme dimensionality presents unique challenges:
- high risk of overfitting 
- computational complexity 
- need for feature selection or regularization 
- class imbalance considerations 
- model interpretability in high-dimensional space 

The goal is to identify a model that provides **robust, accurate predictions** and 
**interpretable biological insights** while managing the curse of dimensionality. 

---

## 2. Dataset

- Source: Colon cancer gene expression study
  - `colon_genes.csv` – gene expression data (2,000 genes)
  - `colon_classes.csv` – tissue classification (tumor vs. normal)
- Observations: 62 tissue samples
- Target variable: binary classification (tumor = 1, normal = 0)
- Predictors: gene expression levels from microarray data
- Class distribution: 40 tumor, 22 normal (moderate imbalance)

The dataset allows for:
- identification of gene-level biomarkers for cancer diagnosis
- comparison of different strategies for handling high-dimensional classification
- discovery of biologically relevant genes through feature selection stability analysis
- assessment of model generalization in small-sample settings

---

## 3. Models Used (and Why)

Multiple classification models are trained and compared, 
specifically chosen for their ability to handle high-dimensional data: 

- **Ridge Logistic Regression**  
  - L2-regularized logistic regression 
  - shrinks coefficients but keeps all predictors 
  Baseline for regularized linear classifiers.

- **Lasso Logistic Regression**  
  - L1-regularized logistic regression 
  - performs automatic feature selection by driving some coefficients to exactly zero 
  Ideal for interpretability in high-dimensional settings.

- **Elastic Net Logistic Regression**  
  - Combines L1 and L2 penalties
  - balances feature selection (like Lasso) with coefficient stability (like Ridge). 
  Particularly effective when predictors are correlated, as is common in gene expression data.

- **SVM (Support Vector Machine) with Linear Kernel**  
  - finds optimal separating hyperplane between classes
  Effective in high-dimensional spaces and provides robust decision boundaries.

- **Random Forest Classifier**  
  Ensemble method that:
  - handles non-linear decision boundaries
  - performs implicit feature selection
  - reduces overfitting through bootstrap aggregation
  - provides feature importance estimates

- **XGBoost Classifier**  
  Gradient boosting method that:
  - builds models sequentially to correct classification errors
  - includes built-in regularization
  - efficient with high-dimensional sparse data

- **Naive Bayes**  
  Probabilistic classifier based on Bayes' theorem
  - fast and interpretable baseline
  - assumes feature independence

**Feature selection strategy**: Inside each cross-validation fold, the top 100 genes are selected 
using t-test, reducing dimensionality from 2,000 to 100 predictors before model training.

Model selection is based on **cross-validated performance**, ensuring 
fair comparison, class balance preservation, and robustness assessment. 

---

## 4. Evaluation Metrics

Classification performance is evaluated using:

- **Accuracy**
- **Precision**
- **Recall (Sensitivity)**
- **F1-Score**
- **Specificity**
- **AUC (Area Under ROC Curve)**
- **Confusion Matrix** 

Stratified cross-validation is used to:
- maintain class distribution across folds
- assess model generalization in small-sample settings
- evaluate stability across different data splits
- avoid overfitting in the p >> n regime
- support informed model comparison

**Feature stability analysis** is also performed to:
- identify genes consistently selected across folds
- provide additional confidence in biological relevance.

---

## 5. Key Results

- Ridge logistic regression uses all 100 selected features, 
  providing stable but less interpretable coefficients.

- Lasso performs automatic feature selection, 
  reducing the model to a sparse subset of genes.

- SVM Linear achieved marginally higher cross-validation accuracy
  but Elastic Net was selected for superior interpretability with comparable performance.

- **Elastic Net achieves the best overall performance**, achieving:
  - excellent generalization performance
  - interpretable linear coefficients
  - stable estimates with correlated gene expression data
  - biological insights through coefficient analysis

- Ensemble methods (Random Forest, XGBoost) show strong performance 
  but with reduced interpretability compared to linear models.

- **Feature stability analysis** identified 71 genes consistently selected in ≥80% of folds, 
  with 20 genes appearing in 100% of folds, 
  suggesting extremely robust biological signals.

- Top predictive genes show clear directional associations (tumor-promoting vs. tumor-suppressing), 
  providing biological insights into colon cancer molecular signatures.

The final selected model is saved and documented in the `results/` folder.

---

## 6. Main Takeaways

- Regularization is essential for classification in high-dimensional settings (p >> n).
- **Elastic Net provides an optimal balance** between 
  feature selection (Lasso) and coefficient stability (Ridge) 
  for gene expression classification tasks. 
- Feature selection within cross-validation folds 
  prevents information leakage and provides realistic performance estimates.
- Stratified cross-validation is critical for model evaluation 
  when dealing with imbalanced classes and 
  small sample sizes relative to predictor space.
- High feature stability (71 genes selected in ≥80% of folds) 
  provides strong evidence for robust, reproducible biological signals.
- Gene expression data provides powerful discriminative features 
  for cancer classification when properly regularized and validated.

---

## Outputs

The `results/` folder contains:

- cross-validation results (`cv_highdim_classification_results.csv`)
- summary tables comparing model performance (`summary_highdim_classification.csv`)
- the final trained model (`final_highdim_classification_enet.pkl`) 
- the list of selected genes for final model (`selected_features_classification.pkl`)
- the coefficient estimates for each gene in the final model (`gene_coefficients_classification.csv`)
- a Jupyter notebook (`Classification_pGn_cancer.ipynb`) with the full analysis workflow 

---

This case study serves as a reference example of a **well-structured high-dimensional classification analysis** 
with emphasis on 
- regularization techniques, 
- stratified cross-validation in small-sample contexts, 
- feature stability analysis
- balance between predictive accuracy and biological interpretability.

---
