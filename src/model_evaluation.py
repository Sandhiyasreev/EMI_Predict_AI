import joblib
import pandas as pd
from sklearn.metrics import classification_report, mean_squared_error

def evaluate_classification(model_path, X_test, y_test):
    model = joblib.load(model_path)
    preds = model.predict(X_test)
    return classification_report(y_test, preds)

def evaluate_regression(model_path, X_test, y_test):
    model = joblib.load(model_path)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    return {"MSE": mse}
