# Classification Case Study – Wine Quality Classification

This case study addresses a **supervised classification problem** based on the same
wine dataset used in the regression analysis, reformulated to predict wine quality
as a **categorical outcome**. 

The analysis focuses on comparing different classification models, 
evaluating their performance through cross-validation, and understanding 
the trade-offs between interpretability and predictive power. 

---

## 1. Problem Statement

The objective is to classify wines into quality categories based on their
physicochemical characteristics. 

Compared to the regression setting, this formulation introduces:
- a discrete target variable 
- class separation challenges 
- the need for classification-specific performance metrics 

The goal is to identify a model that provides **robust and interpretable predictions**
rather than optimizing accuracy alone. 

---

## 2. Dataset

- Source: `winequality-white.csv` 
- Observations: white wine samples 
- Target variable: wine quality class 
- Predictors: physicochemical features such as acidity, residual sugar, alcohol, etc. 

The dataset allows for:
- direct comparison with the regression task 
- evaluation of how different modeling assumptions affect classification performance 

---

## 3. Models Used (and Why)

Multiple classification models are trained and compared:

- **Logistic Regression**  
  Baseline linear classifier, used as a reference for performance and interpretability.

- **Regularized Logistic Models**  
  Introduced to improve stability and control overfitting.

- **Decision Tree Classifier**  
  Used to capture non-linear decision boundaries and feature interactions.

- **Random Forest Classifier**  
  Ensemble method chosen to:
  - improve generalization performance
  - reduce variance compared to single trees
  - handle non-linearities without strong parametric assumptions

- **Gradient Boosting (XGBoost)**  
  Boosting-based ensemble method used to:
  - improve classification performance through sequential error reduction
  - capture complex, non-linear decision boundaries
  - perform well with minimal feature engineering

- **Naive Bayes**  
  Probabilistic classifier based on conditional independence assumptions, used as:
  - a simple and fast baseline
  - a reference for comparison with more flexible models

- **Support Vector Machine (Linear Kernel)**  
  Margin-based classifier used to:
  - perform well in high-dimensional feature spaces 
  - provide a strong linear decision boundary with regularization 

- **Support Vector Machine (RBF Kernel)**  
  Non-linear extension of SVM used to: 
  - capture complex class boundaries 
  - model non-linear relationships between predictors 
  - balance flexibility and generalization through kernel parameters 

Model selection is based on **cross-validated results**, ensuring fair and consistent comparison.

---

## 4. Evaluation Metrics

Classification performance is evaluated using:

- **Accuracy**
- **Precision and Recall**
- **F1-score**
- **ROC-AUC**

Cross-validation is used to: 
- assess model stability 
- mitigate dependence on a single data split 
- support informed model comparison 

---

## 5. Key Results

- Linear classifiers provide a strong and interpretable baseline but are limited
  in capturing complex class boundaries. 
- Tree-based models improve classification performance, indicating the presence
  of non-linear relationships between predictors and the target. 
- **Random Forest achieves the best overall performance** across validation metrics,
  offering a favorable balance between accuracy and robustness. 
- Performance gains are achieved without extensive manual feature engineering. 

The final selected model is saved and documented in the `results/` folder. 

---

## 6. Main Takeaways

- Reframing the same dataset as a classification problem highlights
  the impact of modeling assumptions.  
- Ensemble methods outperform simpler classifiers in the presence
  of non-linear decision boundaries. 
- Cross-validation is essential for reliable performance comparison in classification tasks. 

---

## Outputs

The `results/` folder includes:
- cross-validation results for all tested classifiers 
- performance summary tables 
- the final trained model (`final_model_rf.pkl`) 
- a Jupyter notebook (`Classification_Wine.ipynb`) containing the full analysis workflow 

---

This case study complements the regression analysis by demonstrating 
a **classification-oriented statistical learning workflow** on the same data, 
with emphasis on model comparison, evaluation metrics, and interpretability. 