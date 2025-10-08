# Troubleshooting Guide

This guide helps resolve common issues when working with the Purchase Predictor project. For additional support, please refer to the [CONFIGURATION.md](CONFIGURATION.md) and [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) documentation.

## Quick Diagnostics

### System Health Check

Run this command to verify your environment:

```bash
python -c "
import pandas as pd
import numpy as np
import sklearn
import mlflow
import azure.ai.ml
print('All dependencies installed successfully')
print(f'Python: {pd.__version__}, {np.__version__}, {sklearn.__version__}')
print(f'MLflow: {mlflow.__version__}')
print('Azure ML SDK v2: OK')
"
```

### Pipeline Health Check

```bash
# Verify configuration
python -c "from config.config_loader import load_config; print('Config loaded:', load_config()['data']['raw_data_path'])"

# Check data files
ls -la sample_data/
ls -la processed_data/

# Verify model artifacts
ls -la models/
```

## Common Issues and Solutions

### Environment and Dependencies

#### Issue: Missing Dependencies

**Error Message:**
```
ModuleNotFoundError: No module named 'azure.ai.ml'
ImportError: No module named 'mlflow'
```

**Solution:**

1. **Install via Conda (Recommended):**
   ```bash
   conda env create -f conda.yaml
   conda activate purchase-predictor
   ```

2. **Install via pip:**
   ```bash
   pip install azure-ai-ml mlflow pandas scikit-learn python-dotenv piny pyyaml
   ```

3. **Verify installation:**
   ```bash
   python -c "import azure.ai.ml, mlflow; print('Dependencies OK')"
   ```

#### Issue: Python Version Compatibility

**Error Message:**
```
ERROR: Package requires a different Python version
SyntaxError: f-strings require Python 3.6+
```

**Solution:**

1. **Check Python version:**
   ```bash
   python --version  # Should be 3.9+
   ```

2. **Use conda to manage Python version:**
   ```bash
   conda create -n purchase-predictor python=3.9
   conda activate purchase-predictor
   conda env create -f conda.yaml
   ```

3. **Alternative pyenv setup:**
   ```bash
   pyenv install 3.9.18
   pyenv local 3.9.18
   pip install -r requirements.txt
   ```

### Configuration Issues

#### Issue: Missing Configuration File

**Error Message:**
```
FileNotFoundError: config.yaml not found
Configuration file does not exist
```

**Solution:**

1. **Verify config.yaml location:**
   ```bash
   ls -la config/config.yaml
   ```

2. **Copy from sample if missing:**
   ```bash
   cp config/config.yaml.example config/config.yaml  # If example exists
   ```

3. **Create minimal config.yaml:**
   ```yaml
   data:
     raw_data_path: "./sample_data/"
     processed_data_path: "./processed_data/"
   
   azure:
     subscription_id: "your-subscription-id"
     resource_group: "your-resource-group"
     workspace_name: "your-workspace"
   ```

#### Issue: Missing Environment Variables

**Error Message:**
```
KeyError: 'AZURE_SUBSCRIPTION_ID'
Environment variable not found
```

**Solution:**

1. **Create .env.local file:**
   ```bash
   cat > .env.local << EOF
   AZURE_SUBSCRIPTION_ID=your-subscription-id
   AZURE_RESOURCE_GROUP=your-resource-group
   AZURE_WORKSPACE_NAME=your-workspace-name
   AZURE_LOCATION=eastus
   EOF
   ```

2. **Verify environment variables:**
   ```bash
   python -c "import os; print('Subscription:', os.getenv('AZURE_SUBSCRIPTION_ID', 'NOT SET'))"
   ```

3. **Alternative: Set in shell:**
   ```bash
   export AZURE_SUBSCRIPTION_ID="your-subscription-id"
   export AZURE_RESOURCE_GROUP="your-resource-group"
   export AZURE_WORKSPACE_NAME="your-workspace-name"
   ```

### Data Processing Issues

#### Issue: Missing Training Data

**Error Message:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'sample_data/train.csv'
Data file not found
```

**Solution:**

1. **Check data directory structure:**
   ```bash
   ls -la sample_data/
   # Expected files: train.csv, test.csv
   ```

2. **Generate sample data if missing:**
   ```bash
   python src/utilities/data_generator.py  # If available
   ```

3. **Verify data format:**
   ```bash
   head -5 sample_data/train.csv
   # Should have columns: price, user_rating, category, previously_purchased, purchased
   ```

#### Issue: Data Format Problems

**Error Message:**
```
ValueError: could not convert string to float
KeyError: 'purchased'
pandas.errors.ParserError: Error tokenizing data
```

**Solution:**

1. **Check data format:**
   ```python
   import pandas as pd
   df = pd.read_csv('sample_data/train.csv')
   print(df.info())
   print(df.head())
   print("Columns:", df.columns.tolist())
   ```

2. **Validate expected format:**
   ```python
   # Expected columns and types
   expected_columns = ['price', 'user_rating', 'category', 'previously_purchased', 'purchased']
   missing_cols = set(expected_columns) - set(df.columns)
   if missing_cols:
       print(f"Missing columns: {missing_cols}")
   ```

3. **Fix common data issues:**
   ```python
   # Remove extra whitespace
   df.columns = df.columns.str.strip()
   
   # Handle missing values
   df = df.dropna()
   
   # Verify data types
   df['price'] = pd.to_numeric(df['price'], errors='coerce')
   df['user_rating'] = pd.to_numeric(df['user_rating'], errors='coerce')
   ```

### Model Training Issues

#### Issue: Training Failures

**Error Message:**
```
ValueError: Input contains NaN, infinity or a value too large
sklearn.exceptions.NotFittedError: This model has not been fitted yet
```

**Solution:**

1. **Check for data quality issues:**
   ```python
   import pandas as pd
   import numpy as np
   
   df = pd.read_csv('processed_data/train_processed.csv')
   print("NaN values:", df.isnull().sum())
   print("Infinite values:", np.isinf(df.select_dtypes(include=[np.number])).sum())
   print("Data shape:", df.shape)
   ```

2. **Verify preprocessed data:**
   ```bash
   python -c "
   import pandas as pd
   df = pd.read_csv('processed_data/train_processed.csv')
   print('Processed data shape:', df.shape)
   print('Columns:', df.columns.tolist())
   print('Data types:', df.dtypes.to_dict())
   "
   ```

3. **Run preprocessing manually:**
   ```bash
   python src/pipeline/data_preprocessing.py
   ```

#### Issue: MLflow Tracking Problems

**Error Message:**
```
mlflow.exceptions.MlflowException: Could not create experiment
ConnectionError: Failed to connect to MLflow tracking server
```

**Solution:**

1. **Check MLflow setup:**
   ```bash
   mlflow server --host 127.0.0.1 --port 5000 &
   export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
   ```

2. **Use local file tracking:**
   ```python
   import mlflow
   mlflow.set_tracking_uri("file:./mlruns")
   ```

3. **Clear tracking cache:**
   ```bash
   rm -rf mlruns/.trash  # Clear MLflow cache
   mlflow gc  # Garbage collect
   ```

### Azure ML Issues

#### Issue: Azure Authentication

**Error Message:**
```
azure.core.exceptions.ClientAuthenticationError: No credentials available
DefaultAzureCredential failed to retrieve a token
```

**Solution:**

1. **Install Azure CLI and login:**
   ```bash
   az login
   az account set --subscription "your-subscription-id"
   ```

2. **Verify authentication:**
   ```bash
   az account show
   az ml workspace list
   ```

3. **Alternative authentication methods:**
   ```bash
   # Service principal
   az login --service-principal -u APP_ID -p PASSWORD --tenant TENANT_ID
   
   # Managed identity (if running on Azure)
   az login --identity
   ```

#### Issue: Workspace Connection

**Error Message:**
```
azure.ai.ml.exceptions.ValidationException: Workspace not found
ServiceRequestError: Could not connect to workspace
```

**Solution:**

1. **Verify workspace exists:**
   ```bash
   az ml workspace show --name "your-workspace" --resource-group "your-rg"
   ```

2. **Create workspace if needed:**
   ```bash
   az ml workspace create --name "your-workspace" --resource-group "your-rg"
   ```

3. **Check config settings:**
   ```python
   from azure.ai.ml import MLClient
   from azure.identity import DefaultAzureCredential
   
   credential = DefaultAzureCredential()
   ml_client = MLClient(
       credential=credential,
       subscription_id="your-subscription-id",
       resource_group_name="your-resource-group",
       workspace_name="your-workspace",
   )
   print("Workspace connected:", ml_client.workspace_name)
   ```

### Deployment Issues

#### Issue: Endpoint Deployment Failures

**Error Message:**
```
DeploymentException: Deployment failed with error
ResourceNotFound: The specified resource does not exist
```

**Solution:**

1. **Check deployment logs:**
   ```bash
   az ml online-endpoint get-logs --name purchase-predictor-endpoint --deployment-name purchase-predictor-deployment
   ```

2. **Verify endpoint configuration:**
   ```python
   from azure.ai.ml import MLClient
   from azure.identity import DefaultAzureCredential
   
   ml_client = MLClient.from_config(credential=DefaultAzureCredential())
   endpoint = ml_client.online_endpoints.get("purchase-predictor-endpoint")
   print("Endpoint status:", endpoint.provisioning_state)
   ```

3. **Check resource quotas:**
   ```bash
   az ml quota show --location eastus
   ```

4. **Redeploy with different instance type:**
   ```python
   # Modify deployment configuration
   deployment.instance_type = "Standard_DS2_v2"  # Smaller instance
   deployment.instance_count = 1
   ```

#### Issue: Scoring Failures

**Error Message:**
```
HTTP 500 Internal Server Error
ValueError: Model prediction failed
```

**Solution:**

1. **Test local inference first:**
   ```bash
   python src/utilities/local_inference.py
   curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"data": [[25.99, 4, "electronics", "yes"]]}'
   ```

2. **Check model format:**
   ```python
   import joblib
   model = joblib.load('models/model.pkl')
   print("Model type:", type(model))
   print("Model features:", getattr(model, 'feature_names_in_', 'Not available'))
   ```

3. **Verify input format:**
   ```python
   # Test prediction locally
   import joblib
   model = joblib.load('models/model.pkl')
   preprocessor = joblib.load('models/preprocessing_metadata.pkl')
   
   sample_data = [[25.99, 4, "electronics", "yes"]]
   processed_data = preprocessor.preprocess(sample_data)
   prediction = model.predict(processed_data)
   print("Local prediction:", prediction)
   ```

### Performance Issues

#### Issue: Slow Training

**Error Message:**
```
Training taking too long
Memory usage too high
```

**Solution:**

1. **Monitor resource usage:**
   ```bash
   top -p $(pgrep -f python)
   # Or use htop for better visualization
   ```

2. **Reduce data size for testing:**
   ```python
   # Modify config.yaml or use environment variable
   export DATA_SAMPLE_SIZE=1000
   ```

3. **Optimize model parameters:**
   ```yaml
   # In config.yaml
   model:
     n_estimators: 50  # Reduce from default 100
     max_depth: 10     # Limit tree depth
     n_jobs: -1        # Use all available cores
   ```

#### Issue: High Memory Usage

**Solution:**

1. **Process data in chunks:**
   ```python
   # Modify data processing to use chunks
   chunk_size = 1000
   for chunk in pd.read_csv('data.csv', chunksize=chunk_size):
       process_chunk(chunk)
   ```

2. **Clear memory after processing:**
   ```python
   import gc
   del large_dataframe
   gc.collect()
   ```

3. **Use more efficient data types:**
   ```python
   df['category'] = df['category'].astype('category')
   df['price'] = pd.to_numeric(df['price'], downcast='float')
   ```

## Development Best Practices

### Debugging Tips

1. **Enable verbose logging:**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Use Python debugger:**
   ```python
   import pdb; pdb.set_trace()
   # Or use breakpoint() in Python 3.7+
   breakpoint()
   ```

3. **Test components individually:**
   ```bash
   # Test data preprocessing only
   python src/pipeline/data_preprocessing.py
   
   # Test model training only
   python src/pipeline/model_training.py
   
   # Test inference only
   python src/utilities/local_inference.py
   ```

### Code Quality

1. **Run type checking:**
   ```bash
   pip install mypy
   mypy src/
   ```

2. **Code formatting:**
   ```bash
   pip install black
   black src/
   ```

3. **Linting:**
   ```bash
   pip install flake8
   flake8 src/
   ```

### Testing

1. **Unit testing:**
   ```bash
   python -m pytest tests/ -v
   ```

2. **Integration testing:**
   ```bash
   python tests/test_pipeline_integration.py
   ```

3. **API testing:**
   ```bash
   python tests/test_api.py
   ```

## Environment Verification Script

Save this script as `verify_environment.py` and run it to check your setup:

```python
#!/usr/bin/env python3
"""
Environment verification script for Purchase Predictor
"""

import sys
import os
import importlib
from pathlib import Path

def check_python_version():
    """Check Python version compatibility"""
    version = sys.version_info
    if version.major != 3 or version.minor < 9:
        print(f"❌ Python version {version.major}.{version.minor} not supported. Requires Python 3.9+")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """Check required dependencies"""
    dependencies = [
        'pandas', 'numpy', 'sklearn', 'mlflow', 
        'azure.ai.ml', 'azure.identity', 'piny', 'yaml'
    ]
    
    missing = []
    for dep in dependencies:
        try:
            importlib.import_module(dep)
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep}")
            missing.append(dep)
    
    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Install with: pip install azure-ai-ml mlflow pandas scikit-learn python-dotenv piny pyyaml")
        return False
    return True

def check_files():
    """Check required files and directories"""
    required_paths = [
        'config/config.yaml',
        'sample_data/train.csv',
        'sample_data/test.csv',
        'src/pipeline/',
        'src/utilities/',
        'scripts/run_pipeline.sh'
    ]
    
    missing = []
    for path in required_paths:
        if not Path(path).exists():
            print(f"❌ {path}")
            missing.append(path)
        else:
            print(f"✅ {path}")
    
    if missing:
        print(f"\nMissing files/directories: {', '.join(missing)}")
        return False
    return True

def check_configuration():
    """Check configuration setup"""
    try:
        from config.config_loader import load_config
        config = load_config()
        print("✅ Configuration loaded successfully")
        
        # Check key configuration sections
        if 'data' in config:
            print("✅ Data configuration present")
        else:
            print("❌ Data configuration missing")
            return False
            
        if 'azure' in config:
            print("✅ Azure configuration present")
        else:
            print("⚠️  Azure configuration missing (optional for local development)")
            
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def check_environment_variables():
    """Check environment variables"""
    optional_vars = [
        'AZURE_SUBSCRIPTION_ID',
        'AZURE_RESOURCE_GROUP', 
        'AZURE_WORKSPACE_NAME'
    ]
    
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}")
        else:
            print(f"⚠️  {var} (optional)")

def main():
    """Run all verification checks"""
    print("🔍 Purchase Predictor Environment Verification\n")
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Files & Directories", check_files),
        ("Configuration", check_configuration),
        ("Environment Variables", check_environment_variables)
    ]
    
    all_passed = True
    for name, check_func in checks:
        print(f"\n📋 {name}:")
        if not check_func():
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 All checks passed! Environment is ready.")
        print("\nNext steps:")
        print("1. Run: bash scripts/run_pipeline.sh")
        print("2. Or run: python src/pipeline/main_pipeline.py")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

## Getting Help

### Support Resources

1. **Documentation:** Check all files in the `docs/` directory
2. **Configuration Reference:** See [CONFIGURATION.md](CONFIGURATION.md)
3. **API Documentation:** See [API_REFERENCE.md](API_REFERENCE.md)
4. **Deployment Guide:** See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

### Common Support Workflow

1. **Run verification script:** `python verify_environment.py`
2. **Check logs:** Look for error messages in terminal output
3. **Test components individually:** Isolate the failing component
4. **Check configuration:** Verify config.yaml and environment variables
5. **Search documentation:** Look for similar issues in this guide

### Reporting Issues

When reporting issues, please include:

1. **Error message:** Full error traceback
2. **Environment:** Python version, OS, installed packages
3. **Configuration:** Relevant config.yaml sections (sanitized)
4. **Steps to reproduce:** Exact commands that cause the issue
5. **Expected vs. actual behavior:** What should happen vs. what happens

## Related Documentation

- [README.md](../README.md) - Quick start guide
- [CONFIGURATION.md](CONFIGURATION.md) - Configuration reference
- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - Deployment options
- [API_REFERENCE.md](API_REFERENCE.md) - API documentation
- [DATA_SCHEMA.md](DATA_SCHEMA.md) - Data format details