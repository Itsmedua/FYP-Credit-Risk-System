
# Credit Risk Assessment System — Testing Results

## 1. Testing Overview

The application was tested to verify input validation, boundary-value handling, credit-risk prediction, and SHAP explanation display. The tests include manual testing through the Streamlit interface and an isolated automated test for loan-duration validation.

## 2. Test Cases

| Test ID | Test Description | Test Input | Expected Result | Actual Result | Status |
|---|---|---|---|---|---|
| TC-01 | Validate blank Customer ID | Customer ID left blank | Display an error and stop prediction | Expected error displayed | Pass |
| TC-02 | Reject zero credit amount | Credit Amount = 0 | Display a credit-amount validation error and no prediction | Expected error displayed; prediction not displayed | Pass |
| TC-03 | Accept minimum valid credit amount | Credit Amount = 1 | Generate prediction and SHAP explanation | Both displayed successfully | Pass |
| TC-04A | Test minimum loan duration | Duration = 4 months | Generate prediction and SHAP explanation | Both displayed successfully | Pass |
| TC-04B | Test maximum loan duration | Duration = 72 months | Generate prediction and SHAP explanation | Both displayed successfully | Pass |
| TC-05 | Test a valid customer input | Customer ID = CUST006; Credit Amount = 5000; Duration = 24 months | Display prediction, score, and SHAP explanation | All displayed successfully | Pass |
| TC-06A | Automated duration boundary test | Duration = 3 months | Invalid | Invalid | Pass |
| TC-06B | Automated duration boundary test | Duration = 4 months | Valid | Valid | Pass |
| TC-06C | Automated duration boundary test | Duration = 72 months | Valid | Valid | Pass |
| TC-06D | Automated duration boundary test | Duration = 73 months | Invalid | Invalid | Pass |
| TC-07 | Prediction and SHAP display | CUST007; Duration = 24; Amount = 5000; Age = 30 | Display prediction, score, and SHAP explanation | All displayed successfully | Pass |
| TC-08 | Re-test after credit-amount validation fix | CUST007; Duration = 24; Amount = 5000 | Display prediction, score, and SHAP explanation | Bad Credit; score 64.00%; SHAP displayed | Pass |

## 3. Automated Duration Validation

The `test_validation.py` script checks whether the permitted loan duration is between 4 and 72 months, inclusive.

All four assertions passed. This test verifies the duration-validation function independently; it does not represent direct testing of invalid duration values through the Streamlit form.

## 4. Model Evaluation

Both Logistic Regression and Random Forest were trained and evaluated using the same stratified 80:20 train-test split. The test set contained 200 records: 140 Good Credit and 60 Bad Credit cases.

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 78.0% | 76.5% |
| Bad-Credit Precision | 67.0% | 61.0% |
| Bad-Credit Recall | 53.0% | 60.0% |
| Bad-Credit F1-score | 59.0% | 61.0% |
| ROC-AUC | 0.804 | 0.8009 |

Logistic Regression achieved higher accuracy, Bad-Credit precision, and ROC-AUC on this test split. Random Forest achieved higher Bad-Credit recall and F1-score.

Random Forest identified 36 of the 60 actual Bad Credit cases, compared with 32 identified by Logistic Regression. However, Random Forest also classified more Good Credit cases as Bad Credit.

These results describe performance on one test split of the German Credit dataset. They do not establish performance on contemporary real-world loan applicants or determine a universally preferred model.

## 5. Prediction and SHAP Explanation

For TC-07 and TC-08, the application displayed the predicted category, estimated Bad-Credit Model Score, and SHAP explanation for the submitted customer input.

These tests verify that the prediction and explanation are displayed together without an observed application error. They do not independently establish the mathematical correctness of SHAP values.

## 6. Testing Limitations and Remaining Work

The recorded tests cover selected input-validation scenarios, duration boundaries through an isolated function, and successful prediction and explanation display.

The following areas have not been documented as fully tested and should be considered before final submission:

- Behaviour when saved model files are missing or incompatible.
- Broader exception handling and unusual input combinations.
- More comprehensive automated tests of the application workflow.
- Additional validation of explanation behaviour and model generalisation.

The current tests do not prove that all possible inputs are handled correctly or that model predictions are reliable for real-world lending decisions.














