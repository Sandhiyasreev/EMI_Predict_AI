import mlflow

def log_model_metrics(model_name: str, metrics: dict):
    """Log model metrics using MLflow."""
    with mlflow.start_run(run_name=model_name):
        for key, val in metrics.items():
            mlflow.log_metric(key, val)
        print(f"📈 Logged metrics for {model_name}: {metrics}")
