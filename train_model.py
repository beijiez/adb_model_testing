# Databricks notebook source
dbutils.widgets.dropdown("title", "A", ["A", "B", "C"])
print(dbutils.widgets.get("title"))

# COMMAND ----------

pip install mlflow

# COMMAND ----------
# new change

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import time
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error  # Importing mean_squared_error

# Load dataset
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
boston = raw_df.values[1::2, 2]

X_train, X_test, y_train, y_test = train_test_split(data, boston, test_size=0.2, random_state=42)
# print(X_train)

# Train a model
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Log model with MLflow
mlflow.end_run() # terminate prev runs if not done so

with mlflow.start_run() as run:
    mlflow.sklearn.log_model(model, "random_forest_model_b")
    mlflow.log_params({"n_estimators": 100})
    rmse = mean_squared_error(y_test, model.predict(X_test), squared=False)
    mlflow.log_metric("rmse", rmse)
    model_uri = f"runs:/{run.info.run_id}/random_forest_model_b"

print(f"Model logged in run {run.info.run_id} at {model_uri}")

# Register the model
model_name = "RandomForestModelB"

# Register the model
client = MlflowClient()
model_uri = f"runs:/{run.info.run_id}/random_forest_model_b"

# Register the model
mlflow.sklearn.log_model(model, "random_forest_model_b", input_example=X_train[:1], registered_model_name=model_name)
# if we use sklearn.log_model it counts as a register so we do not need the below line
# model_details = mlflow.register_model(model_uri, model_name)

# Wait until the model is ready
# for _ in range(10):
#     model_version_details = client.search_model_versions(f"name='{model_name}'")
#     version_ready = any(mv.current_stage == 'None' for mv in model_version_details if mv.version == model_details.version)
#     if version_ready:
#         print(f"Model version {model_details.version} is ready!!")
#         break
#     time.sleep(5)
mlflow.end_run()
