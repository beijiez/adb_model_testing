# Databricks notebook source
!pip install mlflow


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

## Below are the variables that we could adjust to create a similar but different model. This delta model can then be used to verify A/B serving

model_name = "boston_housing_model_a"
# test_size = 0.2
# state_count = 42
# n_estimators = 100

###################################

model_name = "boston_housing_model_b"
test_size = 0.7
state_count = 50
n_estimators = 60

##

# Load dataset
data_url = "/Workspace/Users/beijiezhang@microsoft.com/adb_model_testing/data/boston.csv"
raw_df = pd.read_csv(data_url, sep="\\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
boston = raw_df.values[1::2, 2]

X_train, X_test, y_train, y_test = train_test_split(data, boston, test_size=test_size, random_state=state_count)
# print(X_train)

# Train a model
model = RandomForestRegressor(n_estimators=n_estimators)
model.fit(X_train, y_train)
mlflow.end_run()

# Log model with MLflow
with mlflow.start_run() as run:
    mlflow.sklearn.log_model(model, "{model_name}")
    mlflow.log_params({"n_estimators": n_estimators})
    rmse = mean_squared_error(y_test, model.predict(X_test), squared=False)
    mlflow.log_metric("rmse", rmse)
    model_uri = f"runs:/{run.info.run_id}/{model_name}"

print(f"Model logged in run {run.info.run_id} at {model_uri}")

# Register the model
client = MlflowClient()
model_uri = f"runs:/{run.info.run_id}/{model_name}"

# Register the model
mlflow.sklearn.log_model(model, model_name, input_example=X_train[:1], registered_model_name=model_name)

mlflow.end_run()

