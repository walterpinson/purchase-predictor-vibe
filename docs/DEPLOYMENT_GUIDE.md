# Purchase Predictor Deployment Guide

This comprehensive guide covers all deployment approaches for the Purchase Predictor model, from local development to production Azure ML endpoints. Choose the deployment strategy that best fits your needs, subscription capabilities, and operational requirements.

## 🚀 **Deployment Options Overview**

The Purchase Predictor supports multiple deployment strategies to accommodate different use cases:

| Deployment Type | Best For | Features | Requirements |
|-----------------|----------|----------|-------------|
| **Azure ML Managed Endpoint** | Production workloads | Auto-scaling, monitoring, enterprise security | Azure subscription, resource providers |
| **Azure ML + Local Inference** | Development, testing | Azure ML integration + local flexibility | Azure subscription (basic) |
| **Azure Container Instance** | Containerized deployment | Simple container deployment | Azure subscription |
| **Local Development Server** | Development, demos | No Azure requirements | Local Python environment |

## 🎯 **Quick Start Deployment**

### Recommended Approach

For most users, start with the **Azure ML Managed Endpoint** for production or **Local Inference Server** for development:

**Production (Azure ML Hosted):**
```bash
./scripts/run_pipeline.sh
```

**Development/Testing:**
```bash
./scripts/run_pipeline_local.sh
python src/utilities/local_inference.py
```

## 🚀 **Detailed Deployment Options**

### 1. **Azure ML Studio Managed Endpoint** (`run_pipeline.sh`)
**Best for:** Production Azure ML Studio hosted inference server (REQUIRED)

```bash
./scripts/run_pipeline.sh
```

**Features:**
- ✅ **Azure ML Studio hosted inference server** 
- ✅ Fully managed endpoint infrastructure
- ✅ Auto-scaling and load balancing
- ✅ Built-in monitoring and logging in Azure ML Studio
- ✅ Production-ready REST API
- ✅ Azure security and compliance
- ⚠️ Requires proper resource provider registration
- ⚠️ May have subscription tier limitations

**Uses:** `src/pipeline/deploy_managed_endpoint.py`

**This is the REQUIRED approach for Azure ML Studio hosted deployment!**

---

### 2. **Azure ML Integrated + Local Inference** (`run_pipeline_local.sh`)
**Best for:** Azure ML integration without managed endpoint limitations

```bash
./scripts/run_pipeline_local.sh
```

**Features:**
- ✅ Full Azure ML Studio integration
- ✅ Model registered and visible in Azure ML
- ✅ Local inference server (production-ready)
- ✅ Bypasses subscription limitations
- ✅ Same REST API as Azure endpoints
- ✅ Works with all subscription types

**Uses:** `src/pipeline/deploy_azure_ml.py`

**After deployment, start the inference server:**
```bash
python src/utilities/local_inference.py
```

**Test predictions:**
```bash
curl http://localhost:5000/test
```

---

### 3. **Azure Container Instance (ACI)** (`run_pipeline_aci.sh`)
**Best for:** Containerized deployment with simpler infrastructure

```bash
./scripts/run_pipeline_aci.sh
```

**Features:**
- ✅ Container-based deployment
- ✅ Simpler than managed endpoints
- ✅ Good for development/testing
- ✅ Cost-effective for low-traffic scenarios
- ⚠️ Less scalable than managed endpoints

**Uses:** `src/pipeline/deploy_aci.py`

---

### 4. **Local Development Server** (No Azure Required)
**Best for:** Development, testing, demos, cost-effective inference

```bash
# Run complete pipeline locally (no Azure deployment)
python src/utilities/data_prep.py
python src/pipeline/train.py

# Start local inference server
python src/utilities/local_inference.py
```

**Features:**
- ✅ No Azure subscription required
- ✅ Immediate setup and testing
- ✅ Full REST API compatibility
- ✅ Interactive debugging capabilities
- ✅ Cost-effective for development
- ✅ Ideal for demos and prototyping

**Local Server Endpoints:**
- `GET /health` - Health check and status
- `POST /predict` - Make predictions
- `GET /info` - Model and API information
- `GET /test` - Test with sample data

**Example Usage:**
```bash
# Start the local server
python src/utilities/local_inference.py

# Test with curl (in another terminal)
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [[25.99, 4, 0, 1], [150.00, 2, 1, 0]]}'

# View model info
curl http://localhost:5000/info

# Quick test with sample data
curl http://localhost:5000/test
```

---

## 🎯 **Deployment Recommendations**

### **For Production (Azure ML Studio Hosted - REQUIRED):**

- **Primary:** `run_pipeline.sh` (Azure ML Studio managed endpoint)
- **Backup:** `run_pipeline_local.sh` (if resource provider issues occur)

### **For Development/Testing:**

- Use `run_pipeline_local.sh` (fastest and most reliable)
- Use `run_pipeline_aci.sh` (for container testing)

### **For Azure ML Studio Integration:**

- **`run_pipeline.sh`** creates actual Azure ML Studio hosted inference server
- **`run_pipeline_local.sh`** provides Azure ML integration with local server
- **`run_pipeline_aci.sh`** provides containerized deployment in Azure ML

---

## 🔧 **Common Issues & Solutions**

### **Resource Provider Registration Errors**

Azure ML managed endpoints require several resource providers to be registered with your subscription. If you encounter `SubscriptionNotRegistered` errors, you need to register these providers:

#### **Required Resource Providers (Standard):**
```bash
az provider register --namespace Microsoft.MachineLearningServices
az provider register --namespace Microsoft.ContainerInstance  
az provider register --namespace Microsoft.Storage
az provider register --namespace Microsoft.KeyVault
az provider register --namespace Microsoft.ContainerRegistry
az provider register --namespace Microsoft.Compute
az provider register --namespace Microsoft.Network
```

#### **Additional Resource Providers (Discovered During Deployment):**
These additional providers were found to be required in some cases:
```bash
az provider register --namespace Microsoft.PolicyInsights
az provider register --namespace Microsoft.Cdn  # Note: "Cdn" not "Cnd" (typo in some docs)
az provider register --namespace Microsoft.ServiceBus
az provider register --namespace Microsoft.Relay
az provider register --namespace Microsoft.EventHub
```

#### **Check Registration Status:**
```bash
# Check all at once
az provider list --query "[?namespace=='Microsoft.MachineLearningServices' || namespace=='Microsoft.PolicyInsights' || namespace=='Microsoft.Cdn'].{Namespace:namespace, State:registrationState}" -o table

# Check individual provider
az provider show --namespace Microsoft.MachineLearningServices --query registrationState -o tsv
```

#### **Alternative Solution:**
If you continue to see resource provider issues:
- Use `run_pipeline_local.sh` instead
- This bypasses managed endpoint limitations
- Provides Azure ML integration with local inference server

### **Endpoint Naming Issues**
- Endpoint names must be ≤ 32 characters
- Use only alphanumeric and hyphens
- All scripts handle this automatically

### **Conda Environment Issues**
- All scripts use modern conda activation
- Compatible with current conda versions
- Environment created automatically if missing

---

## 📊 **After Deployment**

### **Check Results:**
- **Managed/ACI:** `models/endpoint_info.yaml`
- **Local:** `models/azure_ml_deployment_info.yaml`

### **Test Predictions:**
- **Managed/ACI:** Use scoring URI from endpoint info
- **Local:** `curl http://localhost:5000/test`

### **Monitor Models:**
- All models appear in Azure ML Studio
- Track performance and retrain as needed
- Full MLOps lifecycle supported

---

## 🚀 **Quick Start**

1. **Choose your approach** based on needs
2. **Run the pipeline:** `./scripts/run_pipeline_[approach].sh`
3. **Test predictions** using the provided endpoints
4. **Monitor and iterate** through Azure ML Studio

All approaches provide production-ready deployment with full Azure ML integration! 🎉

---

## 📡 **API Usage and Integration**

### Making Predictions

Once deployed, all endpoints provide the same REST API interface:

#### Using Python

```python
import requests
import json

# Azure ML Endpoint (from models/endpoint_info.yaml)
endpoint_uri = "https://your-endpoint-uri.azure.com/score"
# OR Local Endpoint
# endpoint_uri = "http://localhost:5000/predict"

headers = {"Content-Type": "application/json"}

# Prepare data (both raw and preprocessed formats supported)
data = {
    "data": [
        [25.99, 4, "electronics", "yes"],  # Raw format (recommended)
        [150.00, 2, "books", "no"]
    ]
}

# Make request
try:
    response = requests.post(endpoint_uri, json=data, headers=headers)
    response.raise_for_status()
    predictions = response.json()
    print(json.dumps(predictions, indent=2))
except requests.exceptions.RequestException as e:
    print(f"API request failed: {e}")
```

#### Using curl

```bash
# Azure ML Endpoint
curl -X POST "https://your-endpoint-uri.azure.com/score" \
     -H "Content-Type: application/json" \
     -d '{
       "data": [
         [25.99, 4, "electronics", "yes"],
         [150.00, 2, "books", "no"]
       ]
     }'

# Local Endpoint
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"data": [[25.99, 4, "electronics", "yes"]]}'
```

### Understanding API Responses

The API returns predictions with confidence scores:

```json
{
  "predictions": [1, 1],
  "probabilities": [
    [0.024, 0.976],
    [0.482, 0.518]
  ]
}
```

**Response Fields:**
- **`predictions`**: Binary predictions (0=not purchased, 1=purchased)
- **`probabilities`**: Confidence scores `[prob_not_purchased, prob_purchased]`

For detailed API documentation, see [API_REFERENCE.md](API_REFERENCE.md).

---

## ⚙️ **Advanced Deployment Configuration**

### Deployment Settings

Configure deployment behavior in `config.yaml`:

```yaml
# Deployment Configuration
deployment:
  endpoint_name: "purchase-predictor-endpoint"
  deployment_name: "purchase-predictor-deployment"
  instance_type: "Standard_DS2_v2"        # VM size
  instance_count: 1                       # Number of instances
  traffic_percentage: 100                 # Traffic allocation
  
# Model Settings
model:
  type: "random_forest"                   # Model algorithm
  random_state: 42                       # Reproducibility
  
# Data Processing Settings
data_processing:
  handle_missing: "drop"                  # Missing value strategy
  use_float_types: true                   # MLFlow compatibility
  drop_threshold: 0.1                    # Feature selection threshold
```

### Environment Configuration

Set up Azure credentials in `.env.local`:

```bash
# Required for Azure deployments
AZURE_SUBSCRIPTION_ID=your-subscription-id
AZURE_RESOURCE_GROUP=your-resource-group
AZURE_WORKSPACE_NAME=your-workspace-name
AZURE_LOCATION=eastus

# Optional for enhanced functionality
MLFLOW_TRACKING_URI=file:./mlruns
```

### Instance Type Recommendations

| Workload Type | Instance Type | vCPUs | Memory | Cost | Best For |
|---------------|---------------|-------|--------|------|----------|
| **Development** | Standard_DS1_v2 | 1 | 3.5 GB | $ | Testing, low traffic |
| **Production** | Standard_DS2_v2 | 2 | 7 GB | $$ | Balanced performance |
| **High Traffic** | Standard_DS3_v2 | 4 | 14 GB | $$$ | High throughput |
| **Compute Intensive** | Standard_F2s_v2 | 2 | 4 GB | $$ | Fast inference |

---

## 🗃️ **Deployment Archival System**

All deployment scripts include an **automated archival system** that manages deployment artifacts and provides operational intelligence for debugging and rollback purposes.

### **How It Works**

When you run any deployment script, the system:

1. **Creates a `/server` directory** with clean deployment artifacts
2. **Archives previous deployments** with timestamps before deploying new ones  
3. **Copies deployment files** (`score.py`, `preprocessing.py`) to the server directory
4. **Simplifies imports** for Azure ML container compatibility
5. **Tracks deployment metadata** for operational intelligence

### **Directory Structure**

```bash
server/                              # Active deployment directory
├── score.py                         # Current scoring script
├── preprocessing.py                 # Current preprocessing module  
├── deployment_info.json             # Current deployment metadata
└── archives/                        # Historical deployments
    ├── 2025-10-06_14-30-15/        # Previous deployment archive
    │   ├── score.py                 # Archived scoring script
    │   ├── preprocessing.py         # Archived preprocessing
    │   ├── deployment_info.json     # Archived metadata
    │   └── archive_info.json        # Archive metadata
    └── 2025-10-06_15-45-22/        # Another archive
        └── ...
```

### **Archival Management Commands**

Use the `src/utilities/server_manager.py` utility for archival management:

```bash
# List all deployment archives with metadata
python src/utilities/server_manager.py list

# Show current deployment status and files
python src/utilities/server_manager.py current

# Display complete server directory structure
python src/utilities/server_manager.py structure

# Clean old archives (keeps 5 most recent)
python src/utilities/server_manager.py clean

# Prepare fresh deployment environment
python src/utilities/server_manager.py fresh
```

### **Benefits**

- **🐛 Debugging**: Compare current vs previous deployments when issues arise
- **🔄 Rollback**: Previous deployment files are preserved for emergency rollback  
- **📊 Operational Intelligence**: Track deployment history and changes over time
- **🧹 Dependency Management**: Clean separation between development and deployment files
- **🔍 Troubleshooting**: Easy access to deployment artifacts that were actually used

### **Archive Metadata**

Each archive includes rich metadata for troubleshooting:

```json
{
  "deployed_at": "2025-10-06_14-30-15",
  "deployment_files": ["score.py", "preprocessing.py"],
  "source_info": {
    "score_script_source": "src/scripts/score.py",
    "preprocessing_source": "src/modules/preprocessing.py"
  },
  "deployment_type": "azure_ml_managed_endpoint",
  "archive_location": "server/archives/2025-10-06_14-30-15"
}
```

### **Automatic Cleanup**

- Archives are automatically created before each new deployment
- Use `python src/utilities/server_manager.py clean` to remove old archives  
- Keeps the 5 most recent archives by default
- Manual cleanup preserves deployment history with timestamped archives

### **Import Simplification**

The archival system solves the **Azure ML container dependency issue** by:

- Copying both `score.py` and `preprocessing.py` to the same `/server` directory
- Enabling simple local imports: `from preprocessing import PurchaseDataPreprocessor`
- Eliminating complex `sys.path` manipulation that fails in Azure containers
- Ensuring deployment artifacts are self-contained and portable

### **Troubleshooting with Archives**

When deployments fail or behave unexpectedly:

1. **Check current deployment**: `python src/utilities/server_manager.py current`
2. **Compare with previous**: `python src/utilities/server_manager.py list`
3. **Examine specific archive**: Look in `server/archives/{timestamp}/`
4. **Verify file changes**: Diff current vs archived versions
5. **Rollback if needed**: Copy files from working archive back to `/server`

This archival system ensures **deployment reliability**, **operational visibility**, and **easy troubleshooting** across all deployment approaches!

---

## 📚 **Related Documentation**

For comprehensive information about the Purchase Predictor system:

- **[README.md](../README.md)** - Quick start guide and project overview
- **[CONFIGURATION.md](CONFIGURATION.md)** - Complete configuration reference
- **[API_REFERENCE.md](API_REFERENCE.md)** - Detailed API documentation and integration examples
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Common issues and debugging guide
- **[DATA_SCHEMA.md](DATA_SCHEMA.md)** - Data formats and model specifications
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Technical architecture and system design

### Quick Reference Links

- **Configuration Settings**: [CONFIGURATION.md#deployment-settings](CONFIGURATION.md#deployment-settings)
- **API Integration**: [API_REFERENCE.md#integration-examples](API_REFERENCE.md#integration-examples)
- **Common Issues**: [TROUBLESHOOTING.md#deployment-issues](TROUBLESHOOTING.md#deployment-issues)
- **Data Formats**: [DATA_SCHEMA.md#api-input-format](DATA_SCHEMA.md#api-input-format)

### Support and Troubleshooting

If you encounter deployment issues:

1. **Check the troubleshooting guide**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. **Verify configuration**: [CONFIGURATION.md](CONFIGURATION.md)
3. **Test API locally**: Use local inference server first
4. **Review Azure logs**: Check Azure ML Studio for deployment logs
5. **Check archival system**: Use server management utilities for debugging

For additional support, consult the project's documentation suite or open an issue in the repository.