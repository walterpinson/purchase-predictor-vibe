# Purchase Predictor - Azure ML Deployment

A complete Python project for training and deploying a binary classifier to Azure Machine Learning Studio using Azure ML SDK v2 and MLFlow.

## Features

- **End-to-end MLOps pipeline** with Azure ML SDK v2 integration
- **Advanced configuration management** with flexible data processing options
- **Shared preprocessing utilities** for consistent data transformation across training and inference  
- **MLFlow tracking** with explicit schema support and warning elimination
- **Secure configuration management** with environment variables and secrets handling
- **Production-ready deployment** with Azure managed online endpoints
- **Synthetic data generation** for development and testing
- **Configurable model selection** (Random Forest, Logistic Regression)
- **Robust missing data handling** with configurable strategies
- **MLFlow-optimized data types** for seamless model deployment

## Overview

This project implements an end-to-end machine learning pipeline that:

- Generates synthetic training data for purchase prediction
- Trains a binary classifier using scikit-learn
- Packages the model with MLFlow
- Registers the model in Azure ML workspace
- Deploys the model to a managed online endpoint
- Provides REST API for real-time predictions

## Preprocessing Architecture

The project uses a **centralized, configuration-driven preprocessing approach** to ensure consistency and eliminate code duplication:

### Core Components

- **`src/modules/preprocessing.py`**: Contains the `PurchaseDataPreprocessor` class with configurable methods:
  - `fit_transform_training_data()`: Fits preprocessing pipeline and transforms training data
  - `transform_test_data()`: Applies fitted transformations to test data  
  - `transform_inference_data()`: Transforms new data for real-time predictions
  - `load_fitted_preprocessor()`: Loads saved preprocessing pipeline for inference

### Configuration-Driven Processing

The preprocessor supports flexible configuration via `config.yaml`:

```yaml
data_processing:
  handle_missing: "drop"        # Options: "drop", "impute"
  use_float_types: true         # Use float64 for MLFlow compatibility
  drop_threshold: 0.1           # Drop features with >10% missing values
```

**Key Features:**

- **Missing Data Handling**: Configurable strategy for handling missing values
- **MLFlow Optimization**: Uses float64 types to eliminate MLFlow integer schema warnings
- **Production Robustness**: Handles missing data gracefully at inference time
- **Type Consistency**: Ensures consistent data types across training and inference

### Benefits

- **Consistency**: Identical transformations across training, testing, and inference
- **Maintainability**: Single source of truth for all preprocessing logic
- **Reliability**: Eliminates synchronization issues between duplicate code
- **Scalability**: Easy to add new preprocessing steps in one place
- **Configuration Control**: Easy to modify processing behavior without code changes
- **MLFlow Compatible**: Eliminates common MLFlow warnings and compatibility issues

### Integration

All scripts (`src/utilities/data_prep.py`, `src/pipeline/train.py`, `src/scripts/score.py`) use the shared preprocessor, ensuring the same feature engineering pipeline throughout the ML lifecycle.

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
│   ├── pipeline/                # MLOps pipeline scripts
│   │   ├── train.py             # Model training script
│   │   ├── register.py          # Model registration script
│   │   └── deploy.py            # Model deployment script
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

| Column | Type | Description | Example Values | Processing |
|--------|------|-------------|----------------|------------|
| `price` | float64 | Product price in USD | 9.99, 15.00, 22.49 | Direct use |
| `user_rating` | float64 | User rating (1-5) | 4.0, 2.0, 5.0, 3.0 | Convert to float64 |
| `category` | string → float64 | Product category | electronics, books, clothes | Label encoded to float64 |
| `previously_purchased` | string → float64 | Previous purchase history | yes, no | Binary encoded to 1.0, 0.0 |
| `label` | integer | Target: 1=liked, 0=not liked | 1, 0 | Target variable |

**Note**: All features are converted to float64 for MLFlow compatibility and robust missing value handling.

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

## Configuration

### Data Processing Options

Configure data preprocessing behavior in `config.yaml`:

```yaml
# Data Processing Configuration
data_processing:
  handle_missing: "drop"        # Options: "drop", "impute"
  use_float_types: true         # Use float64 for MLFlow compatibility
  drop_threshold: 0.1           # Drop features with >10% missing values
```

**Options Explained:**

- **`handle_missing`**: Strategy for missing data
  - `"drop"`: Remove rows with any missing values (recommended for clean data)
  - `"impute"`: Fill missing values (future enhancement)
- **`use_float_types`**: Data type configuration
  - `true`: Use float64 for all features (eliminates MLFlow warnings)
  - `false`: Use integer types where possible (more memory efficient)
- **`drop_threshold`**: Feature selection threshold
  - Drop features with more than this fraction of missing values

### Model Configuration

Configure model behavior:

```yaml
# Model Configuration
model:
  type: "random_forest"         # Options: "random_forest", "logistic_regression"
  random_state: 42              # Random seed for reproducibility
```

### MLFlow Configuration

Configure experiment tracking:

```yaml
# MLFlow Configuration
mlflow:
  experiment_name: "purchase_predictor"
  run_name: "training_run"
  registered_model_name: "purchase_predictor_model"
```

## Usage

### Quick Start

Run the complete pipeline with these commands:

```bash
# 1. Generate synthetic data and preprocess
python src/utilities/data_prep.py

# 2. Train the model
python src/pipeline/train.py

# 3. Register model in Azure ML
python src/pipeline/register.py

# 4. Deploy to online endpoint
python src/pipeline/deploy.py
```

### Detailed Steps

#### 1. Data Preparation

```bash
python src/utilities/data_prep.py
```

- Generates 500 synthetic samples with realistic distributions
- Creates `train.csv` and `test.csv` in `sample_data/`
- Applies configurable preprocessing (missing data handling, type conversion)
- Saves processed data to `processed_data/`
- Logs data types and processing decisions for transparency

#### 2. Model Training

```bash
python src/pipeline/train.py
```

- Trains a Random Forest classifier (configurable)
- Uses MLFlow for experiment tracking with explicit schema
- Eliminates MLFlow integer schema warnings
- Saves model locally and logs to MLFlow
- Evaluates performance on test set
- Outputs model accuracy and classification report with clear data size information

#### 3. Model Registration

```bash
python src/pipeline/register.py
```

- Connects to Azure ML workspace
- Registers the MLFlow model
- Creates model version in Azure ML
- Saves registration info for deployment

#### 4. Model Deployment

```bash
python src/pipeline/deploy.py
```

- Creates managed online endpoint
- Sets up custom environment
- Deploys model with scoring script
- Configures endpoint settings
- Tests deployment with sample data

### Advanced Configuration Options

#### Model Settings

In `config.yaml`, you can specify:

- `model.type`: "random_forest" or "logistic_regression"
- `model.random_state`: Random seed for reproducibility

#### Deployment Settings

- `deployment.endpoint_name`: Name for Azure ML endpoint
- `deployment.deployment_name`: Name for model deployment
- `deployment.instance_type`: VM size for endpoint
- `deployment.instance_count`: Number of instances

#### Data Processing Settings

- `data_processing.handle_missing`: Strategy for missing values ("drop" or "impute")
- `data_processing.use_float_types`: Use float64 for MLFlow compatibility
- `data_processing.drop_threshold`: Threshold for dropping features with missing values

**Recommended Settings for Production:**

```yaml
data_processing:
  handle_missing: "drop"        # Clean, consistent data
  use_float_types: true         # Eliminates MLFlow warnings
  drop_threshold: 0.1           # Remove low-quality features
```

## Deployment Options

### Option 1: Azure ML Managed Endpoint (Production)

For production deployments with managed infrastructure:

```bash
python src/pipeline/deploy.py
```

This creates a managed online endpoint with:
- Auto-scaling capabilities
- Built-in monitoring and logging  
- Authentication and security
- High availability

### Option 2: Local Inference Server (Development/Testing)

For development, testing, or when Azure deployment isn't available:

```bash
python src/scripts/local_inference.py
```

This starts a Flask-based local server with:
- REST API endpoints for predictions
- Health checks and model information
- Sample data testing
- Interactive debugging capabilities

**Local Server Endpoints:**
- `GET /health` - Health check and status
- `POST /predict` - Make predictions
- `GET /info` - Model and API information  
- `GET /test` - Test with sample data

**Example Usage:**

```bash
# Start the local server
python src/scripts/local_inference.py

# Test with curl (in another terminal)
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [[25.99, 4, 0, 1], [150.00, 2, 1, 0]]}'

# View model info
curl http://localhost:5000/info

# Quick test with sample data
curl http://localhost:5000/test
```

**When to Use Each Option:**
- **Azure ML Endpoint**: Production workloads, need scaling, enterprise security
- **Local Server**: Development, testing, demos, cost-effective inference, Azure access issues

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

6. **MLFlow Warnings About Integer Schemas**
   - Solution: Ensure `data_processing.use_float_types: true` in config.yaml
   - This eliminates integer schema inference warnings

7. **Missing Data Handling Issues**
   - Solution: Configure `data_processing.handle_missing` appropriately
   - Use `"drop"` for clean synthetic data, consider `"impute"` for real-world data

8. **Configuration Not Loading**
   - Solution: Verify `.env.local` file exists with correct Azure credentials
   - Check that `config.yaml` syntax is valid YAML

### Logs and Debugging

- Training logs: Console output during `python train.py`
- MLFlow tracking: Check local MLFlow UI with `mlflow ui`
- Azure logs: Available in Azure ML Studio under Endpoints
- Local testing: Run `python src/scripts/score.py` for scoring script testing

## Development

### Adding New Features

1. Update data schema in `src/utilities/data_prep.py`
2. Modify preprocessing in `src/pipeline/train.py` and `src/scripts/score.py`
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
