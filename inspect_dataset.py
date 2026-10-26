from pathlib import Path
import pandas as pd

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

print("DATA TYPES")
print(df.dtypes)

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nNUMERICAL SUMMARY")
print(df[["duration", "credit_amount", "age"]].describe())









print("\nDATASET SHAPE")
print(df.shape)

print("\nFIRST FIVE ROWS")
print(df.head())

print("\nTARGET DISTRIBUTION")
print(df["target"].value_counts().sort_index())

print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

print("\nUNIQUE TARGET VALUES")
print(sorted(df["target"].unique()))
