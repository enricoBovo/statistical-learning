# Statistical Learning Portfolio (R & Python)

This repository is a curated portfolio of **statistical learning and machine learning case studies**, 
covering regression and classification problems in both **low-dimensional** and **high-dimensional (p > n)** settings. 

The goal of this project is to demonstrate **practical, applied ML skills**: 
from problem formulation and model selection to evaluation, interpretation, and comparison, 
using **R and Python**. 

---

## Scope of the Portfolio

The portfolio is structured around four core supervised learning scenarios: 

1. **Regression** 
2. **Classification** 
3. **Regression with high-dimensional data (p > n)** 
4. **Classification with high-dimensional data (p > n)** 

Each scenario is implemented as an **independent case study** with: 
- its own dataset 
- dedicated scripts 
- model outputs and results 
- a detailed README explaining assumptions, choices, and conclusions 

---

## Repository Structure

- statistical-learning/ 

    - regression/ 
    - classification/ 
    - regression_pGn/ 
    - classification_pGn/ 
    - cheatSheet_models.pdf 


### Case study folders
Each folder contains: 
- `data/`: datasets used in the analysis  
- `scripts/`: R and Python scripts for modeling and evaluation  
- `results/`: organized outputs (notebooks, selected models, selected features, metrics)  
- `README.md`: problem description, methodology, and key insights  

---

## Methodological Focus

Across the different case studies, the emphasis is on: 

- clear **problem definition** 
- comparison of multiple models rather than a single “best” one 
- proper **train / test separation** and **cross-validation** 
- handling of overfitting, especially in **p > n** settings 
- interpretability and stability of results 

Rather than maximizing performance alone, the analyses focus on 
**understanding model behavior and trade-offs**. 

---

## Methods & Algorithms

### Linear and Generalized Models
- Linear Regression
- Logistic Regression
- Multinomial Logistic Regression
- Regularized models (Ridge, Lasso, Elastic Net)

### Tree-based Models
- Regression Trees
- Decision Trees
- Random Forest
- Gradient Boosting (XGBoost, LightGBM)
- Ranger (fast Random Forest implementation)

### Non-linear and Flexible Models
- MARS
- Projection Pursuit Regression (PPR)
- Support Vector Machines (linear and RBF)

### Probabilistic Models
- Naive Bayes

---

### Tools
- **R**: glm, glmnet, rpart, randomForest, ranger, earth, ppr
- **Python**: scikit-learn, xgboost, lightgbm
- **Model evaluation**: cross-validation, metrics, feature selection, regularization techniques


---

## Cheat Sheet

The file `cheatSheet_models.pdf` provides a concise reference summarizing:
- main regression and classification models
- strengths and weaknesses
- typical use cases
- considerations for low- vs high-dimensional data

It reflects the **conceptual framework** used throughout the project.

---

## Intended Audience

This repository is designed for:
- recruiters and hiring managers evaluating applied ML skills
- technical interviews and portfolio reviews
- anyone interested in practical statistical learning workflows

Each case study can be explored independently, while the repository as a whole
illustrates a **coherent and structured approach to supervised learning**.

---

## How to Navigate

- Start with this README for an overview
- Open any subfolder to explore a specific learning scenario
- Refer to the local README files for detailed explanations and results