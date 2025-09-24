# Build Plan for GitHub Copilot

## Detailed Steps

### 1. Project Setup

- Create a repo with the file structure found in the spec.
- Add `requirements.txt`:

  ``` text
  azure-ai-ml==v2.x.x
  mlflow
  scikit-learn
  pandas
  PyYAML
  ```
  
- Run `pip install -r requirements.txt`

### 2. Data Preparation and Synthetic Data Creation ([data_prep.py])

- Create a script using Python (`data_prep.py`) to **generate synthetic training and test data** matching the schema specified in your spec.
  - Use Python libraries such as `faker` and `numpy` for realistic, well-distributed numerical and categorical values.
- Save the generated datasets as `train.csv` and `test.csv` under `/sample_data`.
- Load these synthetic CSV files for further preprocessing:
  - Encode categorical columns (e.g., label encoding for category, yes/no conversion for previously_purchased).
  - Ensure no overlap between train and test sets.
- Export processed data sets for use in training and evaluation.

### 3. Training ([train.py])

- Read train data.
- Train a scikit-learn classifier (e.g., LogisticRegression).
- Save model with MLFlow (`mlflow.sklearn.log_model()`).

### 4. Registration ([register.py])

- Connect to Azure ML workspace via SDK.
- Register the MLFlow model.
- Store model name and version for deployment.

### 5. Deployment ([deploy.py])

- Define online endpoint using SDK.
- Set deployment name, model reference, environment (conda.yaml), scoring script.
- Create endpoint and deployment; wait for completion.

### 6. Scoring Script ([score.py])

- Implement `init()` and `run()` per Azure ML scoring requirements to load and score model.

### 7. Environment File ([conda.yaml])

- List all Python dependencies for reproducibility.

### 8. Configuration ([config.yaml])

- Store resource group, workspace, endpoint name, deployment name.

### 9. Documentation and Testing

- Add usage documentation in README.md.
- Test flow end-to-end using sample data.
