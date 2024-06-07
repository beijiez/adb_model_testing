# What
This is a repo to host testing files used for Azure Databricks model creation, publishing, and mlflow testing 

# Testing with Azure DataBricks
Link this repo with a Azure DataBricks instance, and run the notebook file

# How to run

```
Step 1: Create a data.json file

data.json:

{
  "inputs": [
    [
      0.04113,
      25,
      4.86,
      0,
      0.426,
      6.727,
      33.5,
      5.4007,
      4,
      281,
      19,
      396.9,
      5.29,
      28,
      0.04462,
      25
    ]
  ]
}

Step 2: export token and curl the endpoints
export DATABRICKS_TOKEN=xxx
curl \
  -u token:$DATABRICKS_TOKEN \
  -X POST \
  -H "Content-Type: application/json" \
  -d@data.json \
  https://adb-3524775348533048.8.azuredatabricks.net/model/boston_housing_model_b/1/invocations
```
