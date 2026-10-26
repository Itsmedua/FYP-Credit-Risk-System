
from pathlib import Path

import pandas as pd

import matplotlib.pyplot as plt

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# 1. Load dataset
data_path = Path("data/german.data")

column_names = [
    "checking_status", "duration", "credit_history",
    "purpose", "credit_amount", "savings", "employment",
    "installment_rate", "personal_status_sex", "other_debtors",
    "residence_duration", "property", "age",
    "other_installment_plans", "housing", "existing_credits",
    "job", "liable_persons", "telephone", "foreign_worker",
    "target"
]

df = pd.read_csv(
    data_path,
    sep=r"\s+",
    header=None,
    names=column_names
)

# 2. Separate input features and target
X = df.drop(columns=["target"])

# Original: 1 = good credit, 2 = bad credit
# New:      0 = good credit, 1 = bad credit
y = (df["target"] == 2).astype(int)

print("Dataset shape:", df.shape)
print("\nTarget distribution:")
print(y.value_counts().sort_index())

# 3. Identify numerical and categorical columns
numeric_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object", "str", "category"]
).columns.tolist()

print("\nNumerical columns:", numeric_columns)
print("\nCategorical columns:", categorical_columns)

# 4. Define preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_columns
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ]
)

# 5. Create a machine-learning pipeline
model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000))
    ]
)

# 6. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# 7. Train logistic regression
model.fit(X_train, y_train)
print("\nlogistic regression training completed successfully!")







rf_model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)
# Train Random Forest
rf_model.fit(X_train, y_train)
print("\nRandom Forest training completed successfully!")
# 8. logistic regression predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 9. Evaluate the model
print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

print("\nCLASSIFICATION REPORT")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Good credit", "Bad credit"],
        zero_division=0
    )
)

print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))





joblib.dump(model, "credit_risk_model.pkl")
print("\nModel saved successfully!")

# Save Random Forest model for SHAP explanations
joblib.dump(rf_model, "random_forest_model.pkl")
print("Random Forest model saved successfully!")

# 10.Random Forest predictions
rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

print("\n========== RANDOM FOREST RESULTS ==========")

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, rf_pred, labels=[0, 1]))

print("\nCLASSIFICATION REPORT")
print(
    classification_report(
        y_test,
        rf_pred,
        labels=[0, 1],
        target_names=["Good credit", "Bad credit"],
        zero_division=0
    )
)

print("ROC-AUC:", round(roc_auc_score(y_test, rf_prob), 4))



print("\n========== FINAL MODEL COMPARISON ==========")

print("Logistic Regression:")
print("Accuracy:", round((y_pred == y_test).mean(), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))

print("\nRandom Forest:")
print("Accuracy:", round((rf_pred == y_test).mean(), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, rf_prob), 4))






# ==========================================
# RANDOM FOREST FEATURE IMPORTANCE
# ==========================================

# Trained Random Forest classifier ko access karna
rf_classifier = rf_model.named_steps["classifier"]

# Har feature ki importance nikalna
importances = rf_classifier.feature_importances_

# Preprocessing ke baad feature names lena
feature_names = rf_model.named_steps["preprocessing"].get_feature_names_out()

# Importance ko table mein arrange karna
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

# Sab se important features ko pehle dikhana
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== TOP 10 IMPORTANT FEATURES ==========")
print(importance_df.head(10).to_string(index=False))





















# ==========================================
# SHAP: EXPLAIN ONE TEST PREDICTION
# ==========================================

import shap

# Test data ka aik customer select karna
one_customer = X_test.iloc[[0]]

# Preprocessing apply karna
preprocessor = rf_model.named_steps["preprocessing"]
customer_processed = preprocessor.transform(one_customer)

# Trained Random Forest classifier lena
rf_classifier = rf_model.named_steps["classifier"]

# SHAP explainer banana
explainer = shap.TreeExplainer(rf_classifier)

# Selected customer ki explanation nikalna
shap_values = explainer(customer_processed)

print("\n========== SHAP EXPLANATION ==========")
print("Actual target:", y_test.iloc[0])
print("Predicted category:", rf_model.predict(one_customer)[0])

# Bad-credit class (class 1) ki SHAP values
bad_credit_shap = shap_values[0, :, 1]

# Sab se influential features print karna
shap_df = pd.DataFrame({
    "Feature": feature_names,
    "SHAP value": bad_credit_shap.values
})

shap_df["Absolute SHAP"] = shap_df["SHAP value"].abs()
shap_df = shap_df.sort_values(
    by="Absolute SHAP",
    ascending=False
)

print("\nTop 10 features affecting this prediction:")
print(shap_df.head(10).to_string(index=False))
















# ==========================================
# SHAP EXPLANATION VISUALIZATION
# ==========================================

# Top 10 features select karo aur chart ke liye arrange karo
plot_df = shap_df.head(10).sort_values(by="SHAP value")

# Bar chart banao
plt.figure(figsize=(12, 7))
plt.barh(plot_df["Feature"], plot_df["SHAP value"])

# Zero line positive aur negative contributions ko separate karti hai
plt.axvline(x=0, linewidth=1)

plt.title("SHAP Explanation for One Credit Risk Prediction")
plt.xlabel("SHAP Value (impact on Bad Credit prediction)")
plt.ylabel("Feature")

plt.tight_layout()

# Chart ko image file mein save karo
plt.savefig("shap_explanation.png", dpi=300, bbox_inches="tight")

# Chart screen par show karo
plt.show()

print("\nSHAP chart saved as shap_explanation.png")
