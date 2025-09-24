### Product Requirements Document (PRD)

#### Objective

Build a Python-based software tool for training a simple binary classifier and deploying it to Azure ML Studio using the Azure Machine Learning SDK and MLFlow for model packaging.

#### Stakeholders

- Data Scientists
- MLOps Engineers
- DevOps Team

#### Functional Requirements

- Train a binary classifier on structured tabular data (CSV).
- Use Azure ML SDK v2 for all Azure interactions.
- Package and register the trained model in MLFlow format.
- Deploy the registered model to a managed Azure ML online endpoint.
- Allow scoring (prediction) via REST API after deployment.
- Provide Python scripts for training, registration, and deployment.

#### Non-Functional Requirements

- Easy to execute from command line or VS Code.
- Minimal external dependencies (Azure ML SDK, MLFlow, scikit-learn, pandas).
- Configuration/settings via YAML or environment variables.

#### Success Criteria

- Model correctly trains and deploys to Azure ML.
- REST endpoint responds with prediction on sample data.
- Automation of workflow (no manual Studio steps needed except Azure resource setup).
