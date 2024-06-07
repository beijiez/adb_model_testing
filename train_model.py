# Databricks notebook source
dbutils.widgets.dropdown("title", "A", ["A", "B", "C"])
print(dbutils.widgets.get("title"))

# COMMAND ----------

pip install mlflow

# COMMAND ----------
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import time
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error  # Importing mean_squared_error
from sklearn.datasets import fetch_california_housing

# Load dataset
cali = fetch_california_housing()
df = pd.DataFrame(cali.data, columns=cali.feature_names)
df["MEDV"] = cali.target

X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=["MEDV"]), df["MEDV"], test_size=0.2, random_state=42)

# Train a model
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)
mlflow.end_run()

# Log model with MLflow
with mlflow.start_run() as run:
    mlflow.sklearn.log_model(model, "cali_housing_model")
    mlflow.log_params({"n_estimators": 100})
    rmse = mean_squared_error(y_test, model.predict(X_test), squared=False)
    mlflow.log_metric("rmse", rmse)
    model_uri = f"runs:/{run.info.run_id}/cali_housing_model"

print(f"Model logged in run {run.info.run_id} at {model_uri}")

# Register the model
model_name = "CaliHousingModel"
model_uri = f"runs:/{run.info.run_id}/cali_housing_model"

# Register the model
client = MlflowClient()
model_name = "CaliHousingModel"
model_uri = f"runs:/{run.info.run_id}/cali_housing_model"

# Register the model
mlflow.sklearn.log_model(model, "cali_housing_model", input_example=X_train[:1], registered_model_name=model_name)
# model_details = mlflow.register_model(model_uri, model_name)

mlflow.end_run()
