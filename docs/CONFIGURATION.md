# Configuration Reference

This document provides comprehensive information about configuring the Purchase Predictor system.

## Overview

The Purchase Predictor uses a centralized configuration system based on:

- **`config/config.yaml`**: Main configuration file
- **`.env.local`**: Environment variables and secrets (not committed to git)
- **`piny` library**: Runtime environment variable substitution

## Environment Setup

### Prerequisites

1. **Python 3.9+** installed
2. **Azure subscription** with Machine Learning workspace
3. **Azure CLI** installed and authenticated
4. **Appropriate permissions** for Azure ML workspace

### Environment Variables

Create a `.env.local` file in the project root:

```bash
# Azure Configuration
AZURE_SUBSCRIPTION_ID=your-actual-subscription-id
AZURE_RESOURCE_GROUP=your-actual-resource-group
AZURE_WORKSPACE_NAME=your-actual-workspace-name

# Optional: Additional Azure Settings
AZURE_LOCATION=eastus
AZURE_TENANT_ID=your-tenant-id
```

**Note**: The `config.yaml` file references these environment variables using `${VARIABLE_NAME}` syntax, and the `piny` library automatically substitutes them at runtime.

## Configuration File Structure

### Complete config.yaml Reference

```yaml
# Azure ML Configuration
azure:
  subscription_id: "${AZURE_SUBSCRIPTION_ID}"
  resource_group: "${AZURE_RESOURCE_GROUP}"
  workspace_name: "${AZURE_WORKSPACE_NAME}"
  location: "${AZURE_LOCATION:-eastus}"

# Data Processing Configuration
data_processing:
  handle_missing: "drop"        # Options: "drop", "impute"
  use_float_types: true         # Use float64 for MLFlow compatibility
  drop_threshold: 0.1           # Drop features with >10% missing values

# Data Paths
data:
  train_path: "sample_data/train.csv"
  test_path: "sample_data/test.csv"
  processed_data_dir: "processed_data"

# Model Configuration
model:
  type: "random_forest"         # Options: "random_forest", "logistic_regression"
  random_state: 42              # Random seed for reproducibility
  
  # Random Forest specific settings
  random_forest:
    n_estimators: 100
    max_depth: null
    min_samples_split: 2
    min_samples_leaf: 1
    random_state: 42
  
  # Logistic Regression specific settings
  logistic_regression:
    random_state: 42
    max_iter: 1000
    solver: "liblinear"

# MLFlow Configuration
mlflow:
  experiment_name: "purchase_predictor"
  run_name: "training_run"
  registered_model_name: "purchase_predictor_model"
  tracking_uri: null            # Uses default local tracking

# Deployment Configuration
deployment:
  endpoint_name: "purchase-predictor-endpoint"
  deployment_name: "purchase-predictor-deployment"
  instance_type: "Standard_DS1_v2"    # Quota-optimized instance
  instance_count: 1
  scoring_timeout: 60000        # 60 seconds
  request_timeout: 90000        # 90 seconds
  
  # Environment configuration
  environment:
    name: "purchase-predictor-env"
    image: "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
    conda_file: "conda.yaml"
```

## Configuration Options Explained

### Data Processing Settings

#### `handle_missing`
Strategy for handling missing data:

- **`"drop"`**: Remove rows with any missing values
  - **Use when**: Clean synthetic data, development environments
  - **Pros**: Simple, ensures clean data
  - **Cons**: May lose data in production

- **`"impute"`**: Fill missing values (future enhancement)
  - **Use when**: Production with real-world data
  - **Pros**: Preserves all data rows
  - **Cons**: May introduce bias

#### `use_float_types`
Data type configuration for MLFlow compatibility:

- **`true`**: Use float64 for all features
  - **Pros**: Eliminates MLFlow warnings, consistent types
  - **Cons**: Slightly higher memory usage
  - **Recommended**: For production deployments

- **`false`**: Use integer types where possible
  - **Pros**: More memory efficient
  - **Cons**: May trigger MLFlow schema warnings

#### `drop_threshold`
Feature selection threshold:

- **Range**: 0.0 to 1.0
- **Meaning**: Drop features with more than this fraction of missing values
- **Default**: 0.1 (10% missing data threshold)
- **Example**: 0.2 = drop features with >20% missing values

### Model Configuration

#### Model Types

**Random Forest (`"random_forest"`)**
```yaml
model:
  type: "random_forest"
  random_forest:
    n_estimators: 100      # Number of trees
    max_depth: null        # Maximum tree depth (null = unlimited)
    min_samples_split: 2   # Minimum samples to split a node
    min_samples_leaf: 1    # Minimum samples in a leaf
    random_state: 42       # Random seed
```

**Logistic Regression (`"logistic_regression"`)**
```yaml
model:
  type: "logistic_regression"
  logistic_regression:
    random_state: 42       # Random seed
    max_iter: 1000         # Maximum iterations
    solver: "liblinear"    # Solver algorithm
```

### Deployment Configuration

#### Instance Types

**Standard Options:**
- `Standard_DS1_v2`: 1 vCPU, 3.5 GB RAM (quota-friendly)
- `Standard_DS2_v2`: 2 vCPUs, 7 GB RAM (higher performance)
- `Standard_DS3_v2`: 4 vCPUs, 14 GB RAM (high-performance)

**Cost Optimization:**
- Use `Standard_DS1_v2` for development and testing
- Scale up to `Standard_DS2_v2` for production workloads

#### Timeout Settings

```yaml
deployment:
  scoring_timeout: 60000     # Time limit for model inference (ms)
  request_timeout: 90000     # Total request timeout (ms)
```

**Guidelines:**
- **Scoring timeout**: Time for model to make predictions
- **Request timeout**: Total time including network and processing
- **Request timeout should be > scoring timeout**

## Advanced Configuration

### Environment-Specific Configs

#### Development Configuration
```yaml
# config/dev.yaml
data_processing:
  handle_missing: "drop"
  use_float_types: true
  drop_threshold: 0.1

deployment:
  instance_type: "Standard_DS1_v2"
  instance_count: 1
```

#### Production Configuration
```yaml
# config/prod.yaml
data_processing:
  handle_missing: "impute"  # Handle real-world missing data
  use_float_types: true
  drop_threshold: 0.05      # Stricter feature quality

deployment:
  instance_type: "Standard_DS2_v2"
  instance_count: 2         # High availability
  scoring_timeout: 30000    # Tighter performance requirements
```

### MLFlow Configuration

#### Local Tracking
```yaml
mlflow:
  tracking_uri: null        # Uses ./mlruns directory
  experiment_name: "purchase_predictor_dev"
```

#### Remote Tracking (Optional)
```yaml
mlflow:
  tracking_uri: "https://your-mlflow-server.com"
  experiment_name: "purchase_predictor_prod"
```

### Regional Deployment Settings

For different Azure regions:

```yaml
azure:
  location: "eastus"        # Primary region
  # location: "westus2"     # Alternative region
  # location: "westeurope"  # European region
```

**Regional Considerations:**
- Choose regions close to your users
- Consider Azure ML service availability
- Check quota limits per region

## Configuration Validation

### Testing Configuration

Run configuration validation:

```bash
python config/test_config.py
```

This script validates:
- Environment variable substitution
- Required fields presence
- Configuration syntax
- Azure connectivity

### Common Configuration Issues

#### Missing Environment Variables
```bash
# Error: Environment variable not found
KeyError: 'AZURE_SUBSCRIPTION_ID'

# Solution: Check .env.local file exists and contains required variables
```

#### Invalid Configuration Syntax
```bash
# Error: YAML syntax error
yaml.scanner.ScannerError: while parsing a block mapping

# Solution: Validate YAML syntax online or with yaml linter
```

#### Azure Connection Issues
```bash
# Error: Azure authentication failed
DefaultAzureCredential failed to retrieve a token

# Solution: Run 'az login' and check subscription access
```

## Best Practices

### Development Settings
```yaml
data_processing:
  handle_missing: "drop"        # Clean, predictable data
  use_float_types: true         # Avoid MLFlow warnings
  drop_threshold: 0.1           # Reasonable feature quality

deployment:
  instance_type: "Standard_DS1_v2"  # Cost-effective
  instance_count: 1                 # Single instance for testing
```

### Production Settings
```yaml
data_processing:
  handle_missing: "impute"      # Handle real-world data
  use_float_types: true         # Type consistency
  drop_threshold: 0.05          # High feature quality

deployment:
  instance_type: "Standard_DS2_v2"  # Better performance
  instance_count: 2                 # High availability
  scoring_timeout: 30000            # Performance requirements
```

### Security Best Practices

1. **Never commit `.env.local`** - add to `.gitignore`
2. **Use Azure Key Vault** for production secrets
3. **Rotate credentials regularly**
4. **Use service principals** for automated deployments
5. **Limit Azure permissions** to minimum required

## Configuration Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed troubleshooting guidance and common configuration issues.

## Related Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - Technical architecture details
- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - Deployment strategies
- [DATA_SCHEMA.md](DATA_SCHEMA.md) - Data and model configuration