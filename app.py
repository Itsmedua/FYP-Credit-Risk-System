import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Credit Risk Assessment System",
    page_icon="🏦",
    layout="wide"
)


@st.cache_resource
def load_rf_model():
    return joblib.load("random_forest_model.pkl")


st.title("🏦 Credit Risk Assessment System")

st.write(
    "Explainable Credit Risk Assessment and "
    "Loan Decision Support System Using Machine Learning"
)

st.info(
    "Enter the customer's information below to assess "
    "their credit risk using the trained Random Forest model."
)


# Load the Random Forest model
try:
    rf_model = load_rf_model()
except Exception as error:
    st.error(f"Random Forest model load nahi ho saka: {error}")
    st.stop()


# Customer information form
with st.form("credit_risk_form"):
    st.subheader("Customer Information")

    customer_id = st.text_input("Customer ID", value="CUST001")

    st.markdown("**Financial Information**")
    col1, col2 = st.columns(2)

    with col1:
        checking_status = st.selectbox(
            "Checking Account Status",
            ["A11", "A12", "A13", "A14"]
        )

        duration = st.number_input(
            "Loan Duration (months)",
            min_value=4,
            max_value=72,
            value=12
        )

        credit_history = st.selectbox(
            "Credit History",
            ["A30", "A31", "A32", "A33", "A34"]
        )

        purpose = st.selectbox(
            "Loan Purpose",
            [
                "A40", "A41", "A42", "A43", "A44",
                "A45", "A46", "A47", "A48", "A49", "A410"
            ]
        )

        credit_amount = st.number_input(
            "Credit Amount",
            min_value=0,
            value=2500,
            step=1
        )

        savings = st.selectbox(
            "Savings Account/Bonds",
            ["A61", "A62", "A63", "A64", "A65"]
        )

        employment = st.selectbox(
            "Employment Since",
            ["A71", "A72", "A73", "A74", "A75"]
        )

        installment_rate = st.number_input(
            "Installment Rate (1–4)",
            min_value=1,
            max_value=4,
            value=2
        )

        personal_status_sex = st.selectbox(
            "Personal Status and Sex",
            ["A91", "A92", "A93", "A94", "A95"]
        )

        other_debtors = st.selectbox(
            "Other Debtors/Guarantors",
            ["A101", "A102", "A103"]
        )

    with col2:
        residence_duration = st.number_input(
            "Years at Current Residence",
            min_value=1,
            max_value=4,
            value=2
        )

        property_type = st.selectbox(
            "Property",
            ["A121", "A122", "A123", "A124"]
        )

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30
        )

        other_installment_plans = st.selectbox(
            "Other Installment Plans",
            ["A141", "A142", "A143"]
        )

        housing = st.selectbox(
            "Housing",
            ["A151", "A152", "A153"]
        )

        existing_credits = st.number_input(
            "Existing Credits at This Bank",
            min_value=1,
            max_value=4,
            value=1
        )

        job = st.selectbox(
            "Job Type",
            ["A171", "A172", "A173", "A174"]
        )

        liable_persons = st.number_input(
            "Number of Liable Persons",
            min_value=1,
            max_value=2,
            value=1
        )

        telephone = st.selectbox(
            "Telephone",
            ["A191", "A192"]
        )

        foreign_worker = st.selectbox(
            "Foreign Worker",
            ["A201", "A202"]
        )

    submitted = st.form_submit_button("Predict Credit Risk")


# Run prediction only after form submission
if submitted:

    # Input validation
    if not customer_id.strip():
        st.error("Please enter a Customer ID.")

    elif not 4 <= duration <= 72:
        st.error("Loan Duration must be between 4 and 72 months.")

    elif credit_amount <= 0:
        st.error("Credit Amount must be greater than 0.")

    else:
        # Prepare customer data
        customer_data = {
            "checking_status": checking_status,
            "duration": duration,
            "credit_history": credit_history,
            "purpose": purpose,
            "credit_amount": credit_amount,
            "savings": savings,
            "employment": employment,
            "installment_rate": installment_rate,
            "personal_status_sex": personal_status_sex,
            "other_debtors": other_debtors,
            "residence_duration": residence_duration,
            "property": property_type,
            "age": age,
            "other_installment_plans": other_installment_plans,
            "housing": housing,
            "existing_credits": existing_credits,
            "job": job,
            "liable_persons": liable_persons,
            "telephone": telephone,
            "foreign_worker": foreign_worker
        }

        customer_df = pd.DataFrame([customer_data])

        try:
            # Predict credit category
            prediction = rf_model.predict(customer_df)[0]

            probabilities = rf_model.predict_proba(customer_df)[0]
            bad_credit_index = list(rf_model.classes_).index(1)
            bad_credit_score = probabilities[bad_credit_index] * 100

            st.divider()
            st.subheader(f"Prediction Result — {customer_id}")

            if prediction == 0:
                st.success("Predicted Category: Good Credit")
            else:
                st.error("Predicted Category: Bad Credit")

            st.metric(
                "Estimated Bad-Credit Model Score",
                f"{bad_credit_score:.2f}%"
            )

            st.caption(
                "This is the model's estimated probability for the "
                "Bad Credit class, not a guaranteed real-world default "
                "probability or an automatic loan decision."
            )

            # Generate SHAP explanation
            preprocessor = rf_model.named_steps["preprocessing"]
            rf_classifier = rf_model.named_steps["classifier"]

            customer_processed = preprocessor.transform(customer_df)

            explainer = shap.TreeExplainer(rf_classifier)
            shap_values = explainer(customer_processed)

            bad_credit_shap = shap_values[0, :, 1]
            feature_names = preprocessor.get_feature_names_out()

            shap_df = pd.DataFrame({
                "Feature": feature_names,
                "SHAP value": bad_credit_shap.values
            })

            shap_df["Absolute SHAP"] = shap_df["SHAP value"].abs()

            shap_df = shap_df.sort_values(
                by="Absolute SHAP",
                ascending=False
            ).head(10)

            # Simplify feature names
            shap_df["Feature"] = (
                shap_df["Feature"]
                .str.replace("num__", "", regex=False)
                .str.replace("cat__", "", regex=False)
            )

            plot_df = shap_df.sort_values(by="SHAP value")

            fig, ax = plt.subplots(figsize=(9, 5))

            bar_colors = [
                "indianred" if value > 0 else "steelblue"
                for value in plot_df["SHAP value"]
            ]

            ax.barh(
                plot_df["Feature"],
                plot_df["SHAP value"],
                color=bar_colors
            )

            ax.axvline(x=0, color="black", linewidth=1)
            ax.set_title("Factors Influencing This Customer's Prediction")
            ax.set_xlabel("Impact on Bad Credit prediction")
            ax.set_ylabel("Customer features")
            ax.grid(axis="x", linestyle="--", alpha=0.3)
            ax.set_axisbelow(True)

            fig.tight_layout()

            st.subheader("Understanding This Customer's Prediction")

            st.write(
                "Red bars push this model's prediction toward Bad Credit. "
                "Blue bars push it away from Bad Credit. Longer bars indicate "
                "a larger contribution to this individual prediction."
            )

            st.pyplot(fig)
            plt.close(fig)

        except Exception as error:
            st.error(f"Prediction or SHAP explanation error: {error}")
