import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
CLASSIFIER_PATH = BASE_DIR / "best_classification_model.pkl"
REGRESSOR_PATH = BASE_DIR / "best_regression_model.pkl"
METADATA_PATH = BASE_DIR / "model_metadata.json"


def load_artifacts():
    try:
        classifier = joblib.load(CLASSIFIER_PATH)
        regressor = joblib.load(REGRESSOR_PATH)
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        return classifier, regressor, metadata
    except Exception as exc:
        st.error("Required model artifacts are missing or invalid.")
        st.exception(exc)
        st.stop()


def build_features(form_data: dict) -> pd.DataFrame:
    total_expenses = (
        form_data["school_fees"]
        + form_data["college_fees"]
        + form_data["travel_expenses"]
        + form_data["groceries_utilities"]
        + form_data["other_monthly_expenses"]
        + form_data["current_emi_amount"]
        + form_data["monthly_rent"]
    )
    monthly_salary = max(form_data["monthly_salary"], 1.0)

    form_data["total_expenses"] = total_expenses
    form_data["expense_to_income"] = total_expenses / monthly_salary
    form_data["debt_to_income"] = form_data["current_emi_amount"] / monthly_salary
    form_data["savings_to_income"] = form_data["bank_balance"] / monthly_salary
    form_data["emergency_fund_ratio"] = form_data["emergency_fund"] / monthly_salary

    return pd.DataFrame([form_data])


def label_details(label: str) -> tuple[str, str]:
    normalized = label.strip().lower()
    if normalized == "eligible":
        return "success", "Eligible"
    if normalized == "high_risk":
        return "warning", "High Risk"
    return "error", "Not Eligible"


def format_currency(value: float) -> str:
    return f"Rs. {value:,.0f}"


def render_metric(label: str, value: str):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


classifier, regressor, metadata = load_artifacts()
defaults = metadata["defaults"]
choices = metadata["choices"]
summary = metadata["summary"]

st.set_page_config(page_title="EMIPredict AI", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Manrope', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(12, 122, 108, 0.08), transparent 28%),
            linear-gradient(180deg, #f7f4ee 0%, #efe8dc 100%);
    }

    .block-container {
        max-width: 1160px;
        padding-top: 1.6rem;
        padding-bottom: 2rem;
    }

    [data-testid="stSidebar"] {
        background: #172422;
    }

    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stCaption {
        color: #f7f3ea !important;
    }

    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] div[data-baseweb="select"] > div,
    [data-testid="stSidebar"] div[data-baseweb="select"] * {
        color: #182322 !important;
        background: rgba(255, 251, 244, 0.96) !important;
        border-radius: 12px !important;
    }

    .hero {
        background: rgba(255, 252, 247, 0.82);
        border: 1px solid rgba(31, 43, 42, 0.08);
        border-radius: 26px;
        padding: 1.8rem;
        box-shadow: 0 18px 46px rgba(34, 45, 44, 0.08);
        margin-bottom: 1rem;
    }

    .hero .eyebrow {
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-size: 0.76rem;
        font-weight: 800;
        color: #0c7a6c;
        margin-bottom: 0.4rem;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.6rem;
        color: #1b2625;
    }

    .hero p {
        margin: 0.55rem 0 0 0;
        color: #5f6d69;
    }

    .panel, .metric-card, .result-card {
        background: rgba(255, 252, 247, 0.84);
        border: 1px solid rgba(31, 43, 42, 0.08);
        border-radius: 22px;
        box-shadow: 0 18px 46px rgba(34, 45, 44, 0.08);
    }

    .panel {
        padding: 1.2rem;
    }

    .metric-card {
        padding: 0.95rem 1rem;
        margin-bottom: 0.85rem;
    }

    .metric-label {
        color: #6b7774;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.28rem;
    }

    .metric-value {
        font-size: 1.35rem;
        font-weight: 800;
        color: #1b2625;
    }

    .result-card {
        padding: 1.3rem;
    }

    .result-tag {
        display: inline-block;
        border-radius: 999px;
        padding: 0.35rem 0.7rem;
        font-size: 0.78rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
        background: rgba(12, 122, 108, 0.1);
        color: #0c7a6c;
    }

    .result-title {
        font-size: 2rem;
        font-weight: 800;
        color: #1b2625;
        margin-bottom: 0.25rem;
    }

    .result-subtitle {
        color: #64726e;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Financial Risk Assessment</div>
        <h1>EMIPredict AI</h1>
        <p>Check EMI eligibility and estimate a safe monthly EMI from one clean dashboard.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## Customer Details")

    age = st.number_input("Age", min_value=18, max_value=75, value=int(defaults["age"]))
    gender = st.selectbox("Gender", choices["gender"], index=choices["gender"].index(defaults["gender"]))
    marital_status = st.selectbox(
        "Marital Status",
        choices["marital_status"],
        index=choices["marital_status"].index(defaults["marital_status"]),
    )
    education = st.selectbox(
        "Education",
        choices["education"],
        index=choices["education"].index(defaults["education"]),
    )
    monthly_salary = st.number_input(
        "Monthly Salary (INR)",
        min_value=10000.0,
        max_value=500000.0,
        value=float(defaults["monthly_salary"]),
        step=1000.0,
    )
    employment_type = st.selectbox(
        "Employment Type",
        choices["employment_type"],
        index=choices["employment_type"].index(defaults["employment_type"]),
    )
    years_of_employment = st.number_input(
        "Years of Employment",
        min_value=0.0,
        max_value=40.0,
        value=float(defaults["years_of_employment"]),
        step=0.1,
    )
    company_type = st.selectbox(
        "Company Type",
        choices["company_type"],
        index=choices["company_type"].index(defaults["company_type"]),
    )
    house_type = st.selectbox(
        "House Type",
        choices["house_type"],
        index=choices["house_type"].index(defaults["house_type"]),
    )
    monthly_rent = st.number_input(
        "Monthly Rent",
        min_value=0.0,
        max_value=100000.0,
        value=float(defaults["monthly_rent"]),
        step=500.0,
    )
    family_size = st.number_input("Family Size", min_value=1, max_value=12, value=int(defaults["family_size"]))
    dependents = st.number_input("Dependents", min_value=0, max_value=10, value=int(defaults["dependents"]))
    school_fees = st.number_input(
        "School Fees",
        min_value=0.0,
        max_value=100000.0,
        value=float(defaults["school_fees"]),
        step=500.0,
    )
    college_fees = st.number_input(
        "College Fees",
        min_value=0.0,
        max_value=100000.0,
        value=float(defaults["college_fees"]),
        step=500.0,
    )
    travel_expenses = st.number_input(
        "Travel Expenses",
        min_value=0.0,
        max_value=50000.0,
        value=float(defaults["travel_expenses"]),
        step=500.0,
    )
    groceries_utilities = st.number_input(
        "Groceries and Utilities",
        min_value=0.0,
        max_value=80000.0,
        value=float(defaults["groceries_utilities"]),
        step=500.0,
    )
    other_monthly_expenses = st.number_input(
        "Other Monthly Expenses",
        min_value=0.0,
        max_value=50000.0,
        value=float(defaults["other_monthly_expenses"]),
        step=500.0,
    )
    existing_loans = st.selectbox(
        "Existing Loans",
        choices["existing_loans"],
        index=choices["existing_loans"].index(defaults["existing_loans"]),
    )
    current_emi_amount = st.number_input(
        "Current EMI Amount",
        min_value=0.0,
        max_value=150000.0,
        value=float(defaults["current_emi_amount"]),
        step=500.0,
    )
    credit_score = st.number_input(
        "Credit Score",
        min_value=300.0,
        max_value=900.0,
        value=float(defaults["credit_score"]),
        step=1.0,
    )
    bank_balance = st.number_input(
        "Bank Balance",
        min_value=0.0,
        max_value=2000000.0,
        value=float(defaults["bank_balance"]),
        step=1000.0,
    )
    emergency_fund = st.number_input(
        "Emergency Fund",
        min_value=0.0,
        max_value=1000000.0,
        value=float(defaults["emergency_fund"]),
        step=1000.0,
    )
    emi_scenario = st.selectbox(
        "EMI Scenario",
        choices["emi_scenario"],
        index=choices["emi_scenario"].index(defaults["emi_scenario"]),
    )
    requested_amount = st.number_input(
        "Requested Loan Amount",
        min_value=1000.0,
        max_value=5000000.0,
        value=float(defaults["requested_amount"]),
        step=1000.0,
    )
    requested_tenure = st.number_input(
        "Requested Tenure (Months)",
        min_value=1,
        max_value=360,
        value=int(defaults["requested_tenure"]),
    )

form_data = {
    "age": age,
    "gender": gender,
    "marital_status": marital_status,
    "education": education,
    "monthly_salary": monthly_salary,
    "employment_type": employment_type,
    "years_of_employment": years_of_employment,
    "company_type": company_type,
    "house_type": house_type,
    "monthly_rent": monthly_rent,
    "family_size": family_size,
    "dependents": dependents,
    "school_fees": school_fees,
    "college_fees": college_fees,
    "travel_expenses": travel_expenses,
    "groceries_utilities": groceries_utilities,
    "other_monthly_expenses": other_monthly_expenses,
    "existing_loans": existing_loans,
    "current_emi_amount": current_emi_amount,
    "credit_score": credit_score,
    "bank_balance": bank_balance,
    "emergency_fund": emergency_fund,
    "emi_scenario": emi_scenario,
    "requested_amount": requested_amount,
    "requested_tenure": requested_tenure,
}

input_df = build_features(form_data.copy())
total_expenses = float(input_df.iloc[0]["total_expenses"])
monthly_buffer = max(form_data["monthly_salary"] - total_expenses, 0.0)

left_col, right_col = st.columns([0.8, 1.2], gap="large")

with left_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Snapshot")
    render_metric("Monthly Salary", format_currency(form_data["monthly_salary"]))
    render_metric("Total Expenses", format_currency(total_expenses))
    render_metric("Monthly Buffer", format_currency(monthly_buffer))
    render_metric("Debt To Income", f"{input_df.iloc[0]['debt_to_income']:.2f}")
    render_metric("Credit Score", f"{form_data['credit_score']:.0f}")
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    action_col, model_col, reg_col = st.columns(3, gap="medium")
    with action_col:
        render_metric("Rows Used", f"{summary['dataset_rows_used']:,}")
    with model_col:
        render_metric("Best Classifier", summary["best_classifier"]["name"].replace("_", " ").title())
    with reg_col:
        render_metric("Best Regressor", summary["best_regressor"]["name"].replace("_", " ").title())

    predict_clicked = st.button("Generate Risk Assessment", type="primary", use_container_width=True)

    if predict_clicked:
        predicted_label = classifier.predict(input_df)[0]
        predicted_emi = max(0.0, float(regressor.predict(input_df)[0]))
        status, risk_label = label_details(str(predicted_label))
        st.session_state["prediction"] = {
            "status": status,
            "risk_label": risk_label,
            "predicted_emi": predicted_emi,
        }

    prediction = st.session_state.get("prediction")

    if prediction:
        if prediction["status"] == "success":
            tag = "Approved Profile"
        elif prediction["status"] == "warning":
            tag = "Borderline Profile"
        else:
            tag = "Risky Profile"

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-tag">{tag}</div>
                <div class="result-title">{prediction['risk_label']}</div>
                <div class="result-subtitle">Maximum safe monthly EMI based on the trained model pipeline.</div>
                <div class="metric-value">{format_currency(prediction['predicted_emi'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="result-card">
                <div class="result-tag">Ready</div>
                <div class="result-title">Run the assessment</div>
                <div class="result-subtitle">Use the customer details on the left and click the button to get the prediction.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("")
tab_one, tab_two = st.tabs(["Input Summary", "Engineered Features"])

with tab_one:
    summary_df = pd.DataFrame(
        [
            ("Age", form_data["age"]),
            ("Gender", form_data["gender"]),
            ("Employment Type", form_data["employment_type"]),
            ("House Type", form_data["house_type"]),
            ("Existing Loans", form_data["existing_loans"]),
            ("Requested Amount", format_currency(form_data["requested_amount"])),
            ("Requested Tenure", f"{int(form_data['requested_tenure'])} months"),
        ],
        columns=["Field", "Value"],
    )
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

with tab_two:
    engineered_df = input_df[
        [
            "total_expenses",
            "expense_to_income",
            "debt_to_income",
            "savings_to_income",
            "emergency_fund_ratio",
        ]
    ].T.reset_index()
    engineered_df.columns = ["Feature", "Value"]
    st.dataframe(engineered_df, use_container_width=True, hide_index=True)
