import streamlit as st
import pandas as pd
import joblib
import json
import os

st.set_page_config(page_title="SBA Loan Default Predictor", page_icon="💰", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE_DIR, "rf_model.joblib"))

@st.cache_data
def load_dropdown_options():
    with open(os.path.join(BASE_DIR, "dropdown_options.json")) as f:
        return json.load(f)

model = load_model()
options = load_dropdown_options()

st.title("💰 SBA Loan Default Risk Predictor")
st.write(
    "Estimate the probability that a small business loan will default, "
    "based on a Random Forest model trained on SBA 7(a) loan data."
)

st.divider()

st.subheader("Loan Details")

col1, col2 = st.columns(2)

with col1:
    borr_state = st.selectbox("Borrower State", options["BorrState"])
    project_state = st.selectbox("Project State", options["ProjectState"])
    business_type = st.selectbox("Business Type", options["BusinessType"])
    naics_code = st.selectbox("Industry (2-digit NAICS sector)", options["NaicsCode"])
    delivery_method = st.selectbox("Delivery Method", options["DeliveryMethod"])
    subpgmdesc = st.selectbox("Sub-Program", options["subpgmdesc"])

with col2:
    gross_approval = st.number_input(
        "Loan Amount Approved ($)", min_value=1000, max_value=5_000_000,
        value=250_000, step=1000
    )
    term_in_months = st.number_input(
        "Loan Term (months)", min_value=1, max_value=400, value=120, step=1
    )
    initial_interest_rate = st.number_input(
        "Initial Interest Rate (%)", min_value=0.0, max_value=30.0,
        value=7.5, step=0.1
    )
    approval_fiscal_year = st.number_input(
        "Approval Fiscal Year", min_value=1990, max_value=2030, value=2024, step=1
    )
    jobs_supported = st.number_input(
        "Jobs Supported", min_value=0, max_value=1000, value=5, step=1
    )
    revolver_status = st.selectbox(
        "Revolving Credit?", options=[0, 1],
        format_func=lambda x: "Revolving" if x == 1 else "Non-Revolving"
    )

st.divider()

if st.button("Predict Default Risk", type="primary", use_container_width=True):
    input_df = pd.DataFrame([{
        "BorrState": borr_state,
        "DeliveryMethod": delivery_method,
        "subpgmdesc": subpgmdesc,
        "ProjectState": project_state,
        "BusinessType": business_type,
        "NaicsCode": naics_code,
        "GrossApproval": gross_approval,
        "ApprovalFiscalYear": approval_fiscal_year,
        "InitialInterestRate": initial_interest_rate,
        "TermInMonths": term_in_months,
        "RevolverStatus": revolver_status,
        "JobsSupported": jobs_supported,
    }])

    proba = model.predict_proba(input_df)[0, 1]
    prediction = model.predict(input_df)[0]

    st.subheader("Result")

    risk_pct = proba * 100
    if prediction == 1:
        st.error(f"⚠️ Predicted: **Higher Default Risk** ({risk_pct:.1f}% estimated probability)")
    else:
        st.success(f"✅ Predicted: **Lower Default Risk** ({risk_pct:.1f}% estimated probability)")

    st.progress(min(proba, 1.0))
    st.caption(
        "This is a probability estimate from a statistical model, not a "
        "guarantee or an official credit decision. It is intended for "
        "educational and demonstration purposes only."
    )

st.divider()
st.caption(
    "Model: Random Forest (tuned via RandomizedSearchCV) | "
    "Trained on 347,135 SBA 7(a) loan records | "
    "F1 = 0.5012, PR-AUC = 0.5672 on held-out test data"
)
