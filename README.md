# Explainable Credit Risk Assessment and Loan Decision Support System Using Machine Learning

## Project Overview

This project is an academic prototype that uses machine learning to classify credit applicants into Good Credit and Bad Credit categories. It provides a web-based interface for entering applicant information, viewing model predictions, and exploring explanations of model outputs using SHAP.

The system is intended for educational and decision-support purposes. It does not automatically approve or reject real loan applications and does not guarantee an applicant's future repayment behaviour.

## Features

- Credit risk classification using Logistic Regression and Random Forest.
- Data preprocessing for numerical and categorical attributes.
- Model evaluation using accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC.
- SHAP-based explanations for model predictions.
- Streamlit web interface for entering applicant details and viewing results.
- Basic input validation.

## Dataset

The project uses the Statlog (German Credit Data) dataset from the UCI Machine Learning Repository.

- Records: 1,000
- Input attributes: 20
- Original target labels: 1 = Good Credit, 2 = Bad Credit
- Target labels used in the project: 0 = Good Credit, 1 = Bad Credit

The dataset is historical and may not represent current lending conditions or all applicant populations.

Dataset reference: Hofmann, H. (1994). *Statlog (German Credit Data)*. UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77

## Model Evaluation

Recorded evaluation results from the current train/test split:

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 78.0% | 76.5% |
| Bad-Credit Precision | 67.0% | 61.0% |
| Bad-Credit Recall | 53.0% | 60.0% |
| Bad-Credit F1-score | 59.0% | 61.0% |
| ROC-AUC | 0.804 | 0.8009 |

These results describe performance on the selected test split. They should not be interpreted as a guarantee of performance on real-world loan applications.

## Technologies

- Python
- pandas
- scikit-learn
- SHAP
- Matplotlib
- Streamlit
- joblib
- Git and GitHub

## Project Files

- `app.py` — Streamlit application.
- `train_model.py` — data preprocessing, model training, evaluation, and model saving.
- `predict.py` — prediction-related code.
- `inspect_dataset.py` — dataset inspection.
- `credit_risk_model.pkl` — saved Logistic Regression pipeline.
- `random_forest_model.pkl` — saved Random Forest pipeline.
- `shap_explanation.png` — saved SHAP explanation visualisation.
- `data/` — dataset files and documentation.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Itsmedua/FYP-Credit-Risk-System.git
cd FYP-Credit-Risk-System
```

### 2. Install the required packages

Create a `requirements.txt` file listing the packages used by the project, then install them with:

```bash
pip install -r requirements.txt
```

The exact dependency versions should be verified against the working development environment.

### 3. Run the application

```bash
streamlit run app.py
```

If the saved model files are missing or need to be regenerated, run the training script after installing its dependencies:

```bash
python train_model.py
```

## Explainability

SHAP is used to help explain which input features contribute to model predictions. These explanations describe model behaviour; they do not establish causation or prove that a model is fair.

## Limitations and Responsible Use

- The dataset contains only 1,000 historical records.
- Model predictions may not generalise to contemporary applicants.
- The model's Bad-Credit score is not a calibrated guarantee of default probability.
- Sensitive attributes and potential bias require careful consideration.
- The system is an academic prototype, not a production banking system.

## Future Work

- Add automated and more comprehensive tests.
- Improve dependency management and reproducibility.
- Further evaluate fairness and model generalisation.
- Improve documentation and deployment readiness.

## Author and Academic Context

Developed as a Bachelor of Science in Computer Science Final Year Project (FYP).
