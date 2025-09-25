# Purchase Predictor - Azure ML Deployment

A complete Python project for training and deploying a binary classifier to Azure Machine Learning Studio using Azure ML SDK v2 and MLFlow.

## Features

- **End-to-end MLOps pipeline** with Azure ML SDK v2 integration
- **Shared preprocessing utilities** for consistent data transformation across training and inference  
- **MLFlow tracking** for experiment management and model lifecycle
- **Secure configuration management** with environment variables and secrets handling
- **Production-ready deployment** with Azure managed online endpoints
- **Synthetic data generation** for development and testing
- **Configurable model selection** (Random Forest, Logistic Regression)

## Overview

This project implements an end-to-end machine learning pipeline that:

- Generates synthetic training data for purchase prediction
- Trains a binary classifier using scikit-learn
- Packages the model with MLFlow
- Registers the model in Azure ML workspace
- Deploys the model to a managed online endpoint
- Provides REST API for real-time predictions

## Preprocessing Architecture

The project uses a **centralized preprocessing approach** to ensure consistency and eliminate code duplication:

### Core Components

- **`src/preprocessing.py`**: Contains the `PurchaseDataPreprocessor` class with standardized methods:
  - `fit_transform_training_data()`: Fits preprocessing pipeline and transforms training data
  - `transform_test_data()`: Applies fitted transformations to test data  
  - `transform_inference_data()`: Transforms new data for real-time predictions
  - `load_fitted_preprocessor()`: Loads saved preprocessing pipeline for inference

### Benefits

- **Consistency**: Identical transformations across training, testing, and inference
- **Maintainability**: Single source of truth for all preprocessing logic
- **Reliability**: Eliminates synchronization issues between duplicate code
- **Scalability**: Easy to add new preprocessing steps in one place

### Integration

All scripts (`utilities/data_prep.py`, `train.py`, `scripts/score.py`) use the shared preprocessor, ensuring the same feature engineering pipeline throughout the ML lifecycle.

## Project Structure

```bash
purchase-predictor-vibe/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── conda.yaml                   # Environment definition
├── run_pipeline.sh              # Complete pipeline execution script
├── .env.local                   # Environment variables (not in git)
├── .gitignore                   # Git ignore file
├── config/                      # Configuration and utilities
│   ├── config.yaml              # Configuration settings
│   ├── config_loader.py         # Shared configuration loader utility (uses piny)
│   └── test_config.py           # Configuration validation and testing script
├── src/                         # Source code
│   ├── train.py                 # Model training script
│   ├── register.py              # Model registration script
│   ├── deploy.py                # Model deployment script
│   ├── modules/                 # Shared modules
│   │   └── preprocessing.py     # Shared preprocessing utilities
│   ├── scripts/                 # Deployment scripts
│   │   └── score.py             # Scoring script for endpoint
│   └── utilities/               # Utility scripts
│       └── data_prep.py         # Data generation and preprocessing
├── context/                     # Project documentation
│   ├── prd.md                   # Product Requirements
│   ├── spec.md                  # Technical Specification
│   └── plan.md                  # Build Plan
├── sample_data/                 # Generated training data
│   ├── train.csv
│   └── test.csv
├── processed_data/              # Preprocessed data
└── models/                      # Model artifacts
    ├── model.pkl
    ├── label_encoder.pkl
    ├── preprocessing_metadata.pkl
    ├── registration_info.yaml
    └── endpoint_info.yaml
```

## Data Schema

The model predicts user purchase preference based on these features:

| Column | Type | Description | Example Values |
|--------|------|-------------|----------------|
| `price` | float | Product price in USD | 9.99, 15.00, 22.49 |
| `user_rating` | integer | User rating (1-5) | 4, 2, 5, 3 |
| `category` | string | Product category | electronics, books, clothes |
| `previously_purchased` | string | Previous purchase history | yes, no |
| `label` | integer | Target: 1=liked, 0=not liked | 1, 0 |

## Setup Instructions

### Prerequisites

1. **Python 3.9+** installed
2. **Azure subscription** with Machine Learning workspace
3. **Azure CLI** installed and authenticated
4. **Appropriate permissions** for Azure ML workspace

### Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd purchase-predictor-vibe
   ```

2. Create and activate conda environment:

   ```bash
   conda env create -f conda.yaml
   conda activate purchase-predictor
   ```

3. Verify installation:

   ```bash
   python --version
   pip list
   ```

4. Configure Azure credentials:

   ```bash
   az login
   az account set --subscription <your-subscription-id>
   ```

5. Update your `.env.local` file with Azure credentials:

   ```bash
   AZURE_SUBSCRIPTION_ID=your-actual-subscription-id
   AZURE_RESOURCE_GROUP=your-actual-resource-group
   AZURE_WORKSPACE_NAME=your-actual-workspace-name
   ```

   **Note**: The `config.yaml` file references these environment variables using `${VARIABLE_NAME}` syntax, and the `piny` library automatically substitutes them at runtime.

## Usage

### Quick Start

Run the complete pipeline with these commands:

```bash
# 1. Generate synthetic data and preprocess
python data_prep.py

# 2. Train the model
python train.py

# 3. Register model in Azure ML
python register.py

# 4. Deploy to online endpoint
python deploy.py
```

### Detailed Steps

#### 1. Data Preparation

```bash
python data_prep.py
```

- Generates 500 synthetic samples with realistic distributions
- Creates `train.csv` and `test.csv` in `sample_data/`
- Preprocesses data and saves to `processed_data/`
- Handles categorical encoding and feature engineering

#### 2. Model Training

```bash
python train.py
```

- Trains a Random Forest classifier (configurable)
- Uses MLFlow for experiment tracking
- Saves model locally and logs to MLFlow
- Evaluates performance on test set
- Outputs model accuracy and classification report

#### 3. Model Registration

```bash
python register.py
```

- Connects to Azure ML workspace
- Registers the MLFlow model
- Creates model version in Azure ML
- Saves registration info for deployment

#### 4. Model Deployment

```bash
python deploy.py
```

- Creates managed online endpoint
- Sets up custom environment
- Deploys model with scoring script
- Configures endpoint settings
- Tests deployment with sample data

### Configuration Options

#### Model Configuration

In `config.yaml`, you can specify:

- `model.type`: "random_forest" or "logistic_regression"
- `model.random_state`: Random seed for reproducibility

#### Deployment Configuration

- `deployment.endpoint_name`: Name for Azure ML endpoint
- `deployment.deployment_name`: Name for model deployment
- `deployment.instance_type`: VM size for endpoint
- `deployment.instance_count`: Number of instances

## Making Predictions

Once deployed, use the REST API to make predictions:

### Using Python

```python
import requests
import json

# Get endpoint URI from models/endpoint_info.yaml
endpoint_uri = "https://your-endpoint-uri.azure.com/score"
headers = {"Content-Type": "application/json"}

# Prepare data
data = {
    "data": [
        [25.99, 4, 1, 1],  # price, rating, category_encoded, prev_purchased_encoded
        [150.00, 2, 0, 0]
    ]
}

# Make request
response = requests.post(endpoint_uri, json=data, headers=headers)
predictions = response.json()
print(predictions)
```

### Using curl

```bash
curl -X POST "https://your-endpoint-uri.azure.com/score" \
     -H "Content-Type: application/json" \
     -d '{"data": [[25.99, 4, 1, 1], [150.00, 2, 0, 0]]}'
```

## Model Features

- **Algorithm**: Random Forest Classifier (default) or Logistic Regression
- **Features**: 4 input features (price, rating, category, purchase history)
- **Output**: Binary classification (0=not liked, 1=liked) with probabilities
- **Performance**: Accuracy typically 70-85% on synthetic data
- **Scalability**: Deployed on managed Azure ML infrastructure

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Solution: Install requirements with `pip install -r requirements.txt`

2. **Azure Authentication Failed**
   - Solution: Run `az login` and verify subscription access

3. **Model Registration Failed**
   - Solution: Check Azure ML workspace permissions and run `train.py` first

4. **Deployment Timeout**
   - Solution: Increase timeout in deployment configuration
   - Check Azure portal for deployment logs

5. **Endpoint Not Responding**
   - Solution: Wait 5-10 minutes after deployment
   - Check endpoint status in Azure ML Studio

### Logs and Debugging

- Training logs: Console output during `python train.py`
- MLFlow tracking: Check local MLFlow UI with `mlflow ui`
- Azure logs: Available in Azure ML Studio under Endpoints
- Local testing: Run `python src/scripts/score.py` for scoring script testing

## Development

### Adding New Features

1. Update data schema in `utilities/data_prep.py`
2. Modify preprocessing in `train.py` and `scripts/score.py`
3. Update configuration in `config.yaml`
4. Test locally before deployment

### Custom Models

To use different algorithms:

1. Import model in `train.py`
2. Add configuration in `create_model()` function
3. Update `config.yaml` model type
4. Retrain and redeploy

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes and test locally
4. Update documentation
5. Submit pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:

- Check the troubleshooting section above
- Review Azure ML documentation
- Open an issue in the repository
- Consult the `context/` folder for detailed specifications
