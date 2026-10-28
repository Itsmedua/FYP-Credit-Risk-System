# Explainable Credit Risk Assessment and Loan Decision Support System Using Machine Learning

## Project Overview

This project is an academic prototype that uses machine learning to classify credit applicants into Good Credit and Bad Credit categories. It provides a web-based interface for entering applicant information, viewing model predictions, and exploring explanations of model outputs using SHAP.

The system is intended for educational and decision-support purposes. It does not automatically approve or reject real loan applications and does not guarantee an applicant's future repayment behaviour.

## Features

- Credit risk classification using Logistic Regression and Random Forest.
- Data preprocessing for numerical and categorical attributes.
- Model evaluation using accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC.
- SHAP-based explanations for individual model predictions.
- Global feature importance visualisation to explore which features influence model predictions.
- Streamlit web interface for entering applicant details and viewing results.
- Basic input validation.

## Dataset

The project uses the Statlog (German Credit Data) dataset from the UCI Machine Learning Repository.

- **Records:** 1,000
- **Input attributes:** 20
- **Original target labels:** 1 = Good Credit, 2 = Bad Credit
- **Target labels used in the project:** 0 = Good Credit, 1 = Bad Credit

The dataset is historical and may not represent current lending conditions or all applicant populations.

**Dataset reference:**

Hofmann, H. (1994). *Statlog (German Credit Data)* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77

## Model Evaluation

The following results were recorded using the current train/test split:

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

- `app.py` — Streamlit web application.
- `train_model.py` — Data preprocessing, model training, evaluation, and model saving.
- `predict.py` — Prediction-related code.
- `inspect_dataset.py` — Dataset inspection.
- `test_validation.py` — Basic input-validation tests.
- `testing_results.md` — Documented testing results.
- `credit_risk_model.pkl` — Saved Logistic Regression pipeline.
- `random_forest_model.pkl` — Saved Random Forest pipeline.
- `shap_explanation.png` — Saved SHAP explanation visualisation.
- `data/` — Dataset files and documentation.
- `requirements.txt` — Python package dependencies.

## Installation and Setup

### 1. Clone the Repository

Open a terminal and run:

```bash
git clone https://github.com/Itsmedua/FYP-Credit-Risk-System.git
cd FYP-Credit-Risk-System
```

### 2. Install the Required Packages

The repository already contains a `requirements.txt` file. Install the listed packages with:

```bash
pip install -r requirements.txt
```

For reproducible results, dependency versions should be verified and pinned against the working development environment.

### 3. Run the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

Streamlit will provide a local URL in the terminal that can be opened in a web browser.

### 4. Train the Models if Needed

If the saved model files are missing or need to be regenerated, run:

```bash
python train_model.py
```

Run this command from the project directory after installing the required dependencies and ensuring that the dataset is available at the expected path.

## Explainability

SHAP is used to help explain which input features contribute to model predictions. Individual explanations provide information about a particular prediction, while global feature importance helps explore feature contributions across the model.

These explanations describe model behaviour. They do not establish causation or, by themselves, prove that a model is fair or reliable.

## Limitations and Responsible Use

- The dataset contains only 1,000 historical records.
- Model predictions may not generalise to contemporary applicants or other populations.
- The model's Bad-Credit score is not a calibrated guarantee of default probability.
- Sensitive attributes and potential bias require careful consideration.
- The evaluation results are based on a selected train/test split.
- The system is an academic prototype, not a production banking system or an automated loan-approval service.

## Future Work

- Add automated and more comprehensive tests.
- Improve dependency management and reproducibility.
- Further evaluate fairness and model generalisation.
- Improve documentation and deployment readiness.

## Author and Academic Context

Developed as a Bachelor of Science in Computer Science Final Year Project (FYP).

The project is intended for academic learning, experimentation, and decision-support research.
