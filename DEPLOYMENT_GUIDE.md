# Purchase Predictor Deployment Guide

This project provides three different deployment approaches for your purchase predictor model. Choose the one that best fits your needs and subscription capabilities.

## 🚀 **Deployment Options**

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
python src/scripts/local_inference.py
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

## 🎯 **Recommendations**

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
If you see `SubscriptionNotRegistered` errors:
- Use `run_pipeline_local.sh` instead
- This bypasses managed endpoint limitations

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