# Heart Disease Prediction: Optimized Classification Project

## Project Overview

This repository features a complete Machine Learning (ML) pipeline designed to classify the presence of heart disease based on clinical data from the UCI Heart Disease dataset.

Key steps include advanced data preprocessing, feature engineering through interaction terms, feature selection using Mutual Information, and the implementation of multiple classifiers, culminating in a Stacking Ensemble for optimized performance.

## Technical Implementation Highlights

The core logic is contained within the `mainline.py` and covers the following stages:

### Data Cleaning & Feature Engineering

* **Target Creation:** A binary target (`target`) is created from the original `num` column ($>0$ is disease presence).
* **Missing Value Handling:** Zero values in key variables (`trestbps`, `chol`) are imputed using the median of their respective columns.
* **Feature Engineering:** Multiple multiplicative interaction terms (e.g., `age_chol_interaction`, `oldpeak_thalch_interaction`) were generated to capture non-linear relationships.

### Feature Selection & Preprocessing

* **Mutual Information (MI) Screening:** The Mutual Information scores were calculated for engineered features against the target to quantify predictive relevance.
* **MI Thresholding:** A threshold of $MI \geq 0.05$ was applied to select the most informative interactions, filtering out less significant terms.
    * **Selected Interactions:** `['exang_thalch_interaction', 'oldpeak_thalch_interaction', 'oldpeak_exang_interaction', 'age_chol_interaction', 'age_trestbps_interaction', 'trestbps_thalch_interaction']`.
* **Pipeline Preprocessing:** A `ColumnTransformer` handles the data stream:
    * Numerical features are imputed (median) and scaled (**StandardScaler**).
    * Categorical features are imputed (most frequent) and encoded (**OneHotEncoder**).

### Model Training, Tuning, and Ensembling

* **Hyperparameter Tuning:** All base models undergo rigorous tuning using **GridSearchCV** with a **Repeated Stratified K-Fold** cross-validation strategy.
* **Base Models Evaluated:** Logistic Regression, Decision Tree, Random Forest, XGBoost, and CatBoost.
* **Stacking Ensemble:** A robust Stacking Classifier is implemented, leveraging the best individual models (LR, RF, XGBoost, CatBoost) with a Logistic Regression as the final meta-classifier.

## Performance Results

The models were evaluated on the held-out test set, with a focus on **ROC-AUC** and **F1-score** as primary metrics for imbalanced classification tasks.


### Model Comparison Summary

| Model | ROC-AUC | F1-score | Accuracy | Precision | Recall | Best Params |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CatBoost** | **0.908** | **0.860** | **0.837** | 0.821 | **0.902** | `depth: 6`, `iterations: 100`, `learning_rate: 0.05` |
| Random Forest | 0.904 | 0.831 | 0.810 | 0.819 | 0.843 | `max_depth: 5`, `min_samples_leaf: 2`, `n_estimators: 300` |
| Stacking Ensemble | 0.902 | 0.838 | 0.815 | 0.815 | 0.863 | N/A (Ensemble) |
| XGBoost | 0.901 | 0.853 | 0.832 | 0.826 | 0.882 | `learning_rate: 0.05`, `max_depth: 3`, `n_estimators: 100` |
| Logistic Regression | 0.888 | 0.823 | 0.799 | 0.804 | 0.843 | `C: 0.1`, `penalty: l2`, `solver: lbfgs` |
| Decision Tree | 0.804 | 0.782 | 0.739 | 0.729 | 0.843 | `max_depth: 5`, `min_samples_leaf: 3`, `min_samples_split: 2` |

*(Full metrics are saved in the generated `reports/metrics_summary.csv` file.)*


### Key Conclusion

The **CatBoost Classifier** demonstrated superior performance across the board, achieving the highest **ROC-AUC of 0.908** and an **F1-score of 0.860** on the test set. This robust outcome validates the effectiveness of the pre-processing steps and the benefits of advanced gradient boosting techniques in healthcare predictive modeling.

*(Visual comparison of model performance is available in the generated ROC Curve plot at `reports/roc_curve.png`)*


## Getting Started

To replicate this environment and analysis locally:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/mima-milovanov/heart-disease-prediction-uci
    cd heart-disease-prediction-uci
    ```
2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the main pipeline:**
    ```bash
    python main.py
    ```

