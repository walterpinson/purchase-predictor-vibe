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

### 3. Training (pipeline/train.py)

- Read train data.
- Train a scikit-learn classifier (e.g., LogisticRegression).
- Save model with MLFlow (`mlflow.sklearn.log_model()`).

### 4. Registration (pipeline/register.py)

- Connect to Azure ML workspace via SDK.
- Register the MLFlow model.
- Store model name and version for deployment.

### 5. Deployment (pipeline/deploy_managed_endpoint.py)

**Primary Azure ML Managed Endpoint Deployment with Archival System**
- Creates managed online endpoints with unique naming and retry logic
- Implements comprehensive archival deployment artifact system
- Supports regional deployment configuration
- Provides advanced error handling and debugging capabilities

**Alternative Deployment Options:**
- `deploy_aci.py`: Cost-optimized ACI-style deployment with archival system
- `deploy_azure_ml.py`: Azure ML integration verification and local server bridge

**Primary Deployment (deploy_managed_endpoint.py):**
- Comprehensive Azure ML managed endpoint deployment
- Archival deployment artifact system
- Regional deployment support
- Advanced error handling and retry logic

**ACI-Style Deployment (deploy_aci.py):**
- Cost-optimized deployment with smaller instances
- Standard_F2s_v2 instances for budget-conscious scenarios  

**Azure ML Integration (deploy_azure_ml.py):**
- Verification of Azure ML model registry integration
- Local server bridge functionality

- Define online endpoint using SDK.
- Set deployment name, model reference, environment (conda.yaml), scoring script.
- Create endpoint and deployment; wait for completion.

### 6. Scoring Script (scripts/score.py)

- Implement `init()` and `run()` per Azure ML scoring requirements to load and score model.

### 7. Environment File ([conda.yaml])

- List all Python dependencies for reproducibility.

### 8. Configuration ([config.yaml])

- Store resource group, workspace, endpoint name, deployment name.

### 9. Documentation and Testing

- Add usage documentation in README.md.
- Test flow end-to-end using sample data.
