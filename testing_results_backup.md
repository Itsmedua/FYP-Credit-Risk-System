# Credit Risk Assessment System — Testing Results

## 1. Testing Overview

The application was tested to verify input validation, handling of boundary values, successful credit-risk prediction, and generation of SHAP explanations. The tests were performed using the Streamlit web interface.

## 2. Test Cases

| Test ID | Test Description | Test Input | Expected Result | Actual Result | Status |
|---|---|---|---|---|---|
| TC-01 | Validate blank Customer ID | Customer ID left blank | Display an error and stop prediction | Expected error displayed | Pass |
| TC-02 | Reject zero credit amount | Credit Amount = 0 | Display a credit-amount validation error | Expected error displayed | Pass |
| TC-03 | Accept minimum valid credit amount | Credit Amount = 1 | Generate prediction and SHAP explanation | Both displayed successfully | Pass |
| TC-04A | Test minimum loan duration | Duration = 4 months | Generate prediction and SHAP explanation | Both displayed successfully | Pass |
| TC-04B | Test maximum loan duration | Duration = 72 months | Generate prediction and SHAP explanation | Both displayed successfully | Pass |
| TC-05 | Test a valid customer input | Customer ID = CUST006; Credit Amount = 5000; Duration = 24 months | Display prediction, score, and SHAP explanation | All displayed successfully | Pass |

## 3. Summary

Six test executions have been completed successfully. The tests cover blank customer identification, invalid credit amount, minimum valid credit amount, minimum and maximum loan duration, and a normal valid customer input.

These results confirm the observed behaviour of the tested application scenarios. They do not establish that every input combination is valid or that the model's predictions are accurate for real-world loan applicants.

## 4. Remaining Testing

Further testing is required for invalid duration values, missing or incompatible model files, consistency of model predictions, and broader application error handling. Model performance must be evaluated separately using appropriate classification metrics.














## 5. Automated Duration Validation Test

An isolated automated test was created in `test_validation.py` to verify the permitted loan-duration range of 4 to 72 months.

| Test ID | Input | Expected Result | Actual Result | Status |
|---|---:|---|---|---|
| TC-06A | 3 months | Invalid | Invalid | Pass |
| TC-06B | 4 months | Valid | Valid | Pass |
| TC-06C | 72 months | Valid | Valid | Pass |
| TC-06D | 73 months | Invalid | Invalid | Pass |

All four assertions passed. This test verifies the duration validation function independently; it does not represent direct testing of invalid values through the Streamlit form.





Both Logistic Regression and Random Forest were trained and evaluated using the same stratified 80:20 train-test split. The test set contained 200 records: 140 Good Credit and 60 Bad Credit cases.
Logistic Regression
Random Forest
Logistic Regression achieved higher accuracy, Bad Credit precision, and ROC-AUC on this test set. Random Forest achieved higher Bad Credit recall and F1-score.

Random Forest identified 36 of the 60 actual Bad Credit cases, compared with 32 identified by Logistic Regression. However, Random Forest also classified more Good Credit cases as Bad Credit.

These results describe performance on one test split of the German Credit dataset. They do not establish performance on contemporary real-world loan applicants or determine a universally preferred model.





### TC-07 — Prediction and SHAP Consistency

- **Test Input:** Customer ID: CUST007
- **Loan Duration:** 24 months
- **Credit Amount:** 5000
- **Age:** 30
- **Test Objective:** Verify that the application displays the prediction, Bad-Credit Model Score, and SHAP explanation for the same customer input.
- **Expected Result:** Prediction, score, and SHAP explanation should display without errors.
- **Actual Result:** Prediction, Bad-Credit Model Score, and SHAP explanation displayed successfully.
- **Status:** PASS

**Note:** This test verifies that prediction and explanation are displayed together. It does not independently prove the mathematical correctness of the SHAP explanation.







### TC-08 — Re-test After Credit Amount Validation Fix

- **Test Input:** Customer ID: CUST007
- **Loan Duration:** 24 months
- **Credit Amount:** 5000
- **Expected Result:** The application should display a prediction, Bad-Credit Model Score, and SHAP explanation without errors.
- **Actual Result:** Predicted Category: Bad Credit; Estimated Bad-Credit Model Score: 64.00%; SHAP explanation displayed.
- **Status:** PASS