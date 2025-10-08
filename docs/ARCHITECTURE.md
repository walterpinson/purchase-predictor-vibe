# Purchase Predictor Architecture

This document details the technical architecture, preprocessing pipeline, and project structure of the Purchase Predictor system.

## Preprocessing Architecture

The project uses a **centralized, configuration-driven preprocessing approach** to ensure consistency and eliminate code duplication:

### Core Components

- **`src/utilities/preprocessing.py`**: Contains the `PurchaseDataPreprocessor` class with configurable methods:
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

All scripts (`src/pipeline/data_prep.py`, `src/pipeline/train.py`, `src/scripts/score.py`) use the shared preprocessor, ensuring the same feature engineering pipeline throughout the ML lifecycle.

## Detailed Project Structure

```bash
purchase-predictor-vibe/
├── README.md                    # Quick start guide and project overview
├── conda.yaml                   # Environment definition
├── .env.local                   # Environment variables (not in git)
├── .gitignore                   # Git ignore file
├── docs/                        # Documentation
│   ├── DEPLOYMENT_GUIDE.md      # Deployment strategies and troubleshooting
│   ├── REGIONAL_DEPLOYMENT_GUIDE.md  # Regional deployment configurations
│   ├── UNIQUE_NAMING_IMPLEMENTATION.md  # Endpoint naming strategies
│   ├── ARCHITECTURE.md          # This file - technical architecture
│   ├── CONFIGURATION.md         # Configuration reference
│   ├── API_REFERENCE.md         # API documentation
│   ├── TROUBLESHOOTING.md       # Troubleshooting guide
│   └── DATA_SCHEMA.md           # Data schema and model details
├── scripts/                     # Project management scripts
│   ├── run_pipeline.sh          # 🌟 MAIN PIPELINE - Complete workflow
│   ├── run_pipeline_aci.sh      # ACI-style deployment pipeline
│   ├── run_pipeline_local.sh    # Local development pipeline
│   ├── cleanup_endpoint.sh      # Endpoint cleanup utility
│   ├── fix_environment.sh       # Environment setup utility
│   ├── check_azure_quotas.sh    # Azure quota monitoring script
│   └── quota_monitor.py         # Python quota monitoring utility
├── config/                      # Configuration and utilities
│   ├── config.yaml              # Main configuration settings
│   ├── config_loader.py         # Shared configuration loader utility (uses piny)
│   └── test_config.py           # Configuration validation and testing script
├── src/                         # Source code
│   ├── pipeline/                # MLOps pipeline scripts
│   │   ├── data_prep.py         # Data generation and preprocessing (Step 1)
│   │   ├── train.py             # Model training script
│   │   ├── register.py          # Model registration script
│   │   ├── deploy_managed_endpoint.py  # Primary Azure ML managed endpoint deployment (with archival)
│   │   ├── deploy_aci.py               # ACI-style deployment (cost-optimized)
│   │   ├── deploy_azure_ml.py          # Azure ML integration verification
│   ├── scripts/                 # Deployment scripts
│   │   └── score.py             # Scoring script for endpoint
│   └── utilities/               # Shared utilities
│       ├── preprocessing.py     # Shared preprocessing utility class
│       ├── endpoint_naming.py   # Endpoint naming utilities
│       ├── local_inference.py   # Local development server
│       ├── server_manager.py    # Deployment archival management
│       ├── test_regional_config.py  # Regional deployment testing
│       └── debug_config.py      # Configuration loading debugging
├── context/                     # Project documentation
│   ├── prd.md                   # Product Requirements
│   ├── spec.md                  # Technical Specification
│   └── plan.md                  # Build Plan
├── sample_data/                 # Generated training data
│   ├── train.csv
│   └── test.csv
├── processed_data/              # Preprocessed data
├── server/                      # Deployment artifacts (auto-generated)
│   ├── score.py                 # Current deployment scoring script  
│   ├── preprocessing.py         # Current deployment preprocessing
│   ├── deployment_info.json     # Current deployment metadata
│   └── archives/                # Timestamped deployment archives
└── models/                      # Model artifacts
    ├── model.pkl
    ├── label_encoder.pkl
    ├── preprocessing_metadata.pkl
    ├── registration_info.yaml
    └── endpoint_info.yaml
```

## Component Details

### Pipeline Scripts (`src/pipeline/`)

**Data Preparation (`data_prep.py`)**
- Generates synthetic training data with realistic distributions
- Applies configurable preprocessing (missing data handling, type conversion)
- Saves both raw and processed data for transparency
- Uses shared preprocessing utilities for consistency

**Model Training (`train.py`)**
- Trains Random Forest classifier (configurable)
- Uses MLFlow for experiment tracking with explicit schema
- Eliminates MLFlow integer schema warnings
- Evaluates performance and generates classification reports

**Model Registration (`register.py`)**
- Connects to Azure ML workspace
- Registers MLFlow model in Azure ML
- Creates versioned model entries
- Saves registration metadata for deployment

**Deployment Scripts**
- `deploy_managed_endpoint.py`: Primary Azure ML managed endpoint with archival
- `deploy_aci.py`: ACI-style deployment for cost optimization
- `deploy_azure_ml.py`: Azure ML integration verification

### Utility Modules (`src/utilities/`)

**Preprocessing (`preprocessing.py`)**
- Centralized `PurchaseDataPreprocessor` class
- Consistent transformations across training/testing/inference
- MLFlow-compatible data types
- Configurable missing data handling

**Local Development (`local_inference.py`)**
- Flask-based local inference server
- REST API endpoints for development testing
- Health checks and model information endpoints
- Sample data testing capabilities

**Server Management (`server_manager.py`)**
- Deployment archival system
- Timestamped backup management
- Rollback and debugging support

### Configuration System (`config/`)

**Main Configuration (`config.yaml`)**
- Environment variables integration via `piny`
- Data processing settings
- Model configuration
- Deployment parameters

**Configuration Loader (`config_loader.py`)**
- Centralized configuration loading
- Environment variable substitution
- Validation and error handling

## MLOps Workflow

### 1. Data Pipeline
```
Raw Data → Preprocessing → Feature Engineering → Validation → Storage
```

### 2. Training Pipeline
```
Processed Data → Model Training → Evaluation → MLFlow Logging → Artifact Storage
```

### 3. Deployment Pipeline
```
Trained Model → Registration → Environment Setup → Endpoint Creation → Testing
```

### 4. Inference Pipeline
```
Input Data → Preprocessing → Model Prediction → Response Formatting → Output
```

## Deployment Archival System

The system includes an automated deployment archival feature:

**📁 Location**: `/server/archives/`
**📋 Purpose**: Operational intelligence and rollback capability

**Archived Components:**
- Deployment scripts and configurations
- Preprocessing utilities
- Model artifacts and metadata
- Deployment timestamps and version info

**Benefits:**
- Easy rollback to previous deployments
- Debugging deployment issues
- Historical deployment tracking
- Environment consistency verification

## Technology Stack Integration

**Azure ML SDK v2**
- Modern Azure ML integration
- Managed online endpoints
- Experiment tracking
- Model registry

**MLFlow**
- Model packaging and versioning
- Experiment tracking
- Model deployment format
- Schema validation

**Scikit-learn**
- Model implementation
- Preprocessing utilities
- Performance evaluation

**Configuration Management**
- `piny` for environment variable handling
- YAML-based configuration
- Secrets management

## Security Considerations

**Environment Variables**
- Sensitive data in `.env.local` (not committed)
- Runtime substitution via `piny`
- Azure credentials management

**Model Security**
- Azure ML managed authentication
- Endpoint access control
- Model artifact protection

**Development Security**
- Local development isolation
- Test data generation (no real user data)
- Configuration validation

See [CONFIGURATION.md](CONFIGURATION.md) for detailed configuration options and [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for deployment strategies.