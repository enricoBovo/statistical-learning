# Regression Case Study – Wine Quality Prediction

This case study focuses on a **supervised regression problem** aimed at predicting 
wine quality based on physicochemical properties. 
The analysis follows a structured statistical learning workflow, with particular 
attention to **model comparison**, **cross-validation**, and **interpretation of results**. 

The objective is not only predictive accuracy, but also understanding 
which modeling choices are most appropriate for this type of data. 

---

## 1. Problem Statement

The goal is to predict **wine quality** (continuous outcome) using a set of 
measured physicochemical features. 

From a modeling perspective, this represents a **classical regression problem** with: 
- a moderate number of observations 
- a limited number of predictors 
- potential non-linear relationships between features and outcome 

The task is approached by comparing multiple regression models with different 
assumptions and levels of flexibility. 

---

## 2. Dataset

- Source: `winequality-white.csv` 
- Observations: white wine samples 
- Target variable: `quality` 
- Predictors: physicochemical properties such as acidity, sugar content, alcohol, etc. 

The dataset is well suited for illustrating: 
- linear vs non-linear modeling 
- bias–variance trade-offs 
- the impact of model complexity in regression 

---

## 3. Models Used (and Why)

Several regression models are trained and compared: 

- **Linear Regression**  
  Baseline model to assess purely linear relationships. 

- **Ridge / Lasso / Elastic Net**  
  Regularized linear models used to: 
  - control coefficient magnitude 
  - assess stability 
  - reduce potential overfitting 

- **Decision Tree Regressor**  
  Introduced to capture non-linear effects and interactions. 

- **Random Forest Regressor**  
  Ensemble method used to: 
  - improve predictive performance 
  - reduce variance compared to single trees 
  - model complex, non-linear relationships without strong parametric assumptions 

- **Gradient Boosting (XGBoost)**  
  Boosting-based ensemble method used to: 
  - sequentially reduce prediction errors 
  - capture complex non-linear relationships 
  - improve performance compared to bagging-based approaches 

- **Gradient Boosting (LightGBM)**  
  Efficient boosting framework designed to: 
  - scale well with data size and feature space 
  - handle non-linearities effectively 
  - provide competitive performance with reduced computational cost 

Model selection is based on **cross-validated performance**, not on a single train/test split.

---

## 4. Evaluation Metrics

Models are evaluated using:

- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**
- **R² score**

Cross-validation is used to ensure: 
- robust performance estimates 
- fair comparison across models 

---

## 5. Key Results

- Linear models provide a useful baseline but are limited in capturing
  complex relationships in the data. 
- Regularization improves stability but does not substantially outperform
  the basic linear model in terms of predictive accuracy. 
- Tree-based models show better performance, highlighting the presence
  of non-linear effects. 
- **Random Forest emerges as the best-performing model** in cross-validation,
  achieving the most favorable bias–variance trade-off. 

The final selected model is saved and documented in the `results/` folder.

---

## 6. Main Takeaways

- Model comparison is essential: performance differences are driven more by
  model assumptions than by minor tuning choices. 
- Non-linear models significantly improve prediction quality for this dataset. 
- Ensemble methods such as Random Forest offer strong performance with
  limited manual feature engineering. 

---

## Outputs

The `results/` folder contains:
- cross-validation results 
- summary tables comparing model performance 
- the final trained model (`final_model_rf.pkl`) 
- a Jupyter notebook (`Regression_wine.ipynb`) with the full analysis workflow 

---

This case study serves as a reference example of a **well-structured regression analysis**,
emphasizing methodological clarity, model comparison, and interpretability. 