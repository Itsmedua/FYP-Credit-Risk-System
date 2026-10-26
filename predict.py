
import joblib
import pandas as pd

# Trained Logistic Regression pipeline load karna
model = joblib.load("credit_risk_model.pkl")


def predict_credit_risk(customer_data):
    # Customer ki information ko DataFrame mein convert karna
    customer_df = pd.DataFrame([customer_data])

    # Model se prediction lena
    prediction = model.predict(customer_df)[0]

# Bad-credit class (1) ka estimated score
    bad_credit_score = model.predict_proba(customer_df)[0][1]

    # Numeric prediction ko readable category mein badalna
    if prediction == 0:
        result = "Good Credit"
    else:
        result = "Bad Credit"

    return result, round(bad_credit_score * 100, 2)


# Test customer — sirf function test karne ke liye
sample_customer = {
    "checking_status": "A11",
    "duration": 12,
    "credit_history": "A34",
    "purpose": "A43",
    "credit_amount": 2500,
    "savings": "A61",
    "employment": "A73",
    "installment_rate": 2,
    "personal_status_sex": "A93",
    "other_debtors": "A101",
    "residence_duration": 2,
    "property": "A121",
    "age": 30,
    "other_installment_plans": "A143",
    "housing": "A151",
    "existing_credits": 1,
    "job": "A173",
    "liable_persons": 1,
    "telephone": "A191",
    "foreign_worker": "A201"
    
  
}

# Function ko call karna
result, score = predict_credit_risk(sample_customer)


print("Estimated Bad-Credit Score:", score, "%")

print("Credit Risk Prediction:", result)