# What
This is a repo to host testing files used for Azure Databricks model creation, publishing, and mlflow testing 

# Testing with Azure DataBricks
Link this repo with a Azure DataBricks instance, and run the notebook file

# How to run
Step 1: Create a data.json file
```
{
  "inputs": [
    [
      0.08221,
      22,
      5.86,
      0,
      0.431,
      6.957,
      6.8,
      8.9067,
      7,
      330,
      19.1,
      386.09,
      3.53,
      29.6,
      0.36894,
      22
    ]
  ]
}
```

Step 2: export token and curl the endpoints
```
export DATABRICKS_TOKEN=xxx                          # dapi6exxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
export DATABRICKS_SERVING_URL=xxx                    # https://adb-3524775348533048.8.azuredatabricks.net/serving-endpoints/boston-housing-ab/invocations

curl \
  -u token:$DATABRICKS_TOKEN \
  -X POST \
  -H "Content-Type: application/json" \
  -d@data.json \
  $DATABRICKS_SERVING_URL
```

Expected prediction output should be one of the following depending on the model it landed on:
```
model a: {"predictions":[5.808200000000007]}
model b: {"predictions":[5.904833333333338]}
```
