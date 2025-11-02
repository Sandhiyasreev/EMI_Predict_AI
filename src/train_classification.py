import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from mlflow_logger import log_model_metrics

df = pd.read_csv("../data/processed_data.csv")

# Example classification target
X = df.drop(columns=['Loan_Status'])
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

log_model_metrics("classification_model", {"accuracy": acc})
joblib.dump(model, "../models/best_classification_model.pkl")
print(f"✅ Classification Model trained | Accuracy: {acc:.2f}")
