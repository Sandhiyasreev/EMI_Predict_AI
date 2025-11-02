import streamlit as st
import pandas as pd
import joblib

# ---------------------
# ⚙️ Page Setup
# ---------------------
st.set_page_config(page_title="EMIPredictAI", layout="wide")
st.title("💰 EMIPredictAI: EMI Eligibility & Amount Prediction")

# ---------------------
# 📦 Load Model
# ---------------------
MODEL_PATH = "/Users/sands/Desktop/EMIPredictAI/models/best_regression_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# ---------------------
# 🧮 Collect Inputs (26 Features)
# ---------------------
st.header("📋 Enter Applicant Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=70, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married"])
    education = st.selectbox("Education", ["Graduate", "Non-Graduate"])
    monthly_salary = st.number_input("Monthly Salary (₹)", min_value=0)
    employment_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Unemployed"])
    years_of_employment = st.number_input("Years of Employment", min_value=0, max_value=40, value=5)
    company_type = st.selectbox("Company Type", ["Private", "Public", "Government"])

with col2:
    house_type = st.selectbox("House Type", ["Owned", "Rented"])
    monthly_rent = st.number_input("Monthly Rent (₹)", min_value=0)
    family_size = st.number_input("Family Size", min_value=1, max_value=20, value=4)
    dependents = st.number_input("Dependents", min_value=0, max_value=10, value=1)
    school_fees = st.number_input("School Fees (₹)", min_value=0)
    college_fees = st.number_input("College Fees (₹)", min_value=0)
    travel_expenses = st.number_input("Travel Expenses (₹)", min_value=0)
    groceries_utilities = st.number_input("Groceries & Utilities (₹)", min_value=0)
    other_monthly_expenses = st.number_input("Other Monthly Expenses (₹)", min_value=0)

with col3:
    existing_loans = st.number_input("Existing Loans (Count)", min_value=0, max_value=10, value=0)
    current_emi_amount = st.number_input("Current EMI Amount (₹)", min_value=0)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=700)
    bank_balance = st.number_input("Bank Balance (₹)", min_value=0)
    emergency_fund = st.number_input("Emergency Fund (₹)", min_value=0)
    emi_scenario = st.selectbox("EMI Scenario", ["Normal", "Stressed"])
    requested_amount = st.number_input("Requested Loan Amount (₹)", min_value=10000)
    requested_tenure = st.number_input("Requested Tenure (months)", min_value=6, max_value=360, value=60)
    emi_eligibility = st.selectbox("EMI Eligibility", ["Yes", "No"])

# ---------------------
# 🧾 Prepare Input Data
# ---------------------
input_dict = {
    "age": age,
    "gender": 1 if gender == "Male" else 0,
    "marital_status": 1 if marital_status == "Married" else 0,
    "education": 1 if education == "Graduate" else 0,
    "monthly_salary": monthly_salary,
    "employment_type": 1 if employment_type == "Salaried" else 0,
    "years_of_employment": years_of_employment,
    "company_type": 1 if company_type == "Private" else 0,
    "house_type": 1 if house_type == "Owned" else 0,
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
    "emi_scenario": 1 if emi_scenario == "Normal" else 0,
    "requested_amount": requested_amount,
    "requested_tenure": requested_tenure,
    "emi_eligibility": 1 if emi_eligibility == "Yes" else 0,
}

input_df = pd.DataFrame([input_dict])

# ---------------------
# 🔮 Predict EMI
# ---------------------
if st.button("💡 Predict EMI Amount"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"💵 Predicted Maximum Monthly EMI: ₹{prediction:,.2f}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
