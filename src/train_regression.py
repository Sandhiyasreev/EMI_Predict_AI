import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from mlflow_logger import log_model_metrics
import os

# ---------------------
# 📥 Load Data
# ---------------------
data_path = "/Users/sands/Desktop/EMIPredictAI/data/processed_data.csv"
print("📂 Loading dataset...")
df = pd.read_csv(data_path, low_memory=False)
print("✅ Data loaded successfully!")
print("📊 Columns found:", df.columns.tolist())

# ---------------------
# 🎯 Target Column
# ---------------------
target_column = "max_monthly_emi"

# ✅ Use your actual feature columns
selected_features = [
    'age', 'gender', 'marital_status', 'education', 'monthly_salary',
    'employment_type', 'years_of_employment', 'company_type', 'house_type',
    'monthly_rent', 'family_size', 'dependents', 'school_fees',
    'college_fees', 'travel_expenses', 'groceries_utilities',
    'other_monthly_expenses', 'existing_loans', 'current_emi_amount',
    'credit_score', 'bank_balance', 'emergency_fund', 'emi_scenario',
    'requested_amount', 'requested_tenure', 'emi_eligibility'
]

df = df[selected_features + [target_column]]

# ---------------------
# 🧹 Data Cleaning
# ---------------------
print("🧹 Cleaning data...")

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.fillna(df.median(numeric_only=True), inplace=True)

# One-hot encode categorical columns (like gender, employment_type, etc.)
X = df.drop(columns=[target_column])
X = pd.get_dummies(X, drop_first=True)
y = df[target_column]

print(f"✅ Features after encoding: {X.shape[1]} columns")

# ---------------------
# 🔀 Train-Test Split
# ---------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("✅ Data split into train and test sets")

# ---------------------
# 🤖 Train Model
# ---------------------
print("🧠 Training RandomForestRegressor model...")
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
print("✅ Model training complete!")

# ---------------------
# 📈 Evaluate Model
# ---------------------
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📊 Model Performance:\n   MAE = {mae:.2f}\n   R² = {r2:.2f}")

# ---------------------
# 🧠 Log & Save Model
# ---------------------
log_model_metrics("regression_model", {"MAE": mae, "R2": r2})

model_dir = os.path.join(os.path.dirname(__file__), "//Users/sands/Desktop/EMIPredictAI/models")
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "best_regression_model.pkl")
joblib.dump(model, model_path)

print(f"💾 Model saved successfully at: {model_path}")
print("✅ Training pipeline completed successfully!")
