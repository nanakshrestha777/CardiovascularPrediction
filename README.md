# Cardiovascular Disease Prediction

This project develops a machine learning model to predict cardiovascular disease risk based on health indicators.  A voting classifier ensemble method, combining Gradient Boosting, XGBoost, Random Forest, and AdaBoost, achieves robust performance. LIME provides model interpretability.  The model is deployed using Streamlit for interactive use. This project was developed and run entirely within the Kaggle environment, leveraging the provided computational resources and streamlined workflow.



**Dataset:** [Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)

## Notebook
You can view the full notebook on Kaggle: [Cardio Prediction](https://www.kaggle.com/code/nanakshrestha/cardio-prediction)

## 1. Introduction

Cardiovascular disease (CVD) is a leading cause of death worldwide. Early detection and preventive interventions are crucial. This project aims to build a machine learning model for CVD risk prediction using key health indicators.

## 2. Project Objective

Develop a machine learning model that predicts the risk of cardiovascular disease based on indicators like age, blood pressure, cholesterol levels, smoking status, physical activity, and diabetes.

## 3. Data Exploration and Preprocessing

* **Key Insights:** Initial data analysis revealed potential outliers in blood pressure features and the need for feature engineering.  The target variable (Cardio) showed a slight class imbalance.
* **Data Cleaning:** Outliers in blood pressure were handled. Missing values were checked.
* **Feature Engineering:** BMI and blood pressure categories were created to improve model performance.  Age was converted from days to years.
* **Data Splitting:** The data was split into 80% training and 20% testing sets.  Features were scaled using standardization.

## 4. Model Development and Evaluation

* **Model Selection:** Several models were evaluated, including Logistic Regression, Random Forest, Support Vector Machine (SVC), Naive Bayes, Gradient Boosting, LightGBM, CatBoost, Decision Tree, K-Nearest Neighbors, and XGBoost.
* **Cross-Validation:** Stratified 5-fold cross-validation was used for robust performance assessment.
* **Evaluation Metrics:**  Precision, Recall, F1-score, Matthews Correlation Coefficient (MCC), ROC-AUC, and accuracy were used to evaluate model performance.
* **Hyperparameter Tuning:** Optuna was employed to optimize hyperparameters for each model.

## 5. Ensemble Model

A voting classifier, combining Gradient Boosting, XGBoost, Random Forest, and AdaBoost, was used to improve predictive performance. Soft voting was used to combine predicted probabilities.

## 6. Model Interpretation

LIME (Local Interpretable Model-agnostic Explanations) was used to understand feature importance and model predictions.  (Include a brief summary of key insights from LIME – e.g., "Systolic blood pressure and cholesterol levels were key factors influencing predictions.")

## 7. Deployment

The model was deployed using Streamlit, enabling interactive risk prediction based on user inputs.

## 8. Conclusion and Further Improvements (As before)

* Incorporate additional features (lifestyle, genetics)
* Explore alternative ensemble techniques
* Implement SHAP for deeper interpretability
* Analyze non-linear feature dependencies
* Statistically validate gender differences within blood pressure categories


## Usage: (Optional, add instructions if applicable)

```bash
# Example:
pip install -r requirements.txt
streamlit run cardio_prediction.py  # Or however you launch your Streamlit app


