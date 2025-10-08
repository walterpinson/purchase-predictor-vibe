# Purchase Predictor - Azure ML Deployment

A complete end-to-end machine learning project for training and deploying a binary classifier to Azure Machine Learning Studio. Predict user purchase behavior with a production-ready MLOps pipeline built using Azure ML SDK v2 and MLFlow.

## 🚀 Quick Start

Get the Purchase Predictor running in minutes with our automated pipeline:

### 1. **Set Up Environment**

```bash
# Create conda environment
conda env create -f conda.yaml
conda activate purchase-predictor

# Configure Azure credentials
cp .env.local.example .env.local
# Edit .env.local with your Azure subscription details
```

### 2. **Run the Complete Pipeline** ⭐

```bash
# Run end-to-end ML pipeline (recommended)
bash scripts/run_pipeline.sh
```

**This single command will:**
- ✅ Generate synthetic training data
- ✅ Train a Random Forest classifier
- ✅ Register the model in Azure ML Studio
- ✅ Deploy to a managed online endpoint
- ✅ Test the deployment with sample predictions

### 3. **Make Predictions**

After deployment, test your model:

```bash
# Get endpoint information
cat models/endpoint_info.yaml

# Test with curl (preprocessed format)
curl -X POST "https://your-endpoint-uri.azure.com/score" \
     -H "Content-Type: application/json" \
     -d '{"data": [[25.99, 4, 0, 1]]}'

# Note: Raw format like ["electronics", "yes"] requires the latest scoring script
# For raw input, redeploy with: python src/pipeline/deploy_managed_endpoint.py
```

**That's it!** 🎉 You now have a production-ready ML model deployed on Azure.

### 4. **Alternative: Run Local Inference Server**

If you prefer to test locally without Azure deployment:

```bash
# Run local inference server instead
bash scripts/run_pipeline_local.sh

# Test local server
curl http://localhost:5000/test

# Make predictions locally
curl -X POST "http://localhost:5000/predict" \
     -H "Content-Type: application/json" \
     -d '{"data": [[25.99, 4, "electronics", "yes"]]}'
```

This runs a local Flask server for development and testing without requiring Azure resources.

---

## 🎓 Educational Purpose

This project serves as a **simplified demonstration** of modern MLOps practices, showing:

### **Azure ML SDK v2 Integration**
- Complete MLOps pipeline with modern Azure ML practices
- Model registration and managed endpoint deployment
- Production-ready REST API with monitoring and scaling

### **MLFlow Best Practices** 
- Experiment tracking with explicit schema support
- Model packaging and versioning
- Seamless integration with Azure ML Studio

### **Configuration-Driven Development**
- Flexible data processing with `config.yaml`
- Environment-based credential management
- Configurable model selection and parameters

### **Production Robustness**
- Automated deployment archival system
- Comprehensive error handling and logging
- Local development server for testing

### **Real-World Patterns**
- Shared preprocessing utilities across training and inference
- Missing data handling strategies
- Binary classification with probability scores

**Perfect for:** Data scientists learning MLOps basics, developers exploring Azure ML, and teams prototyping production ML pipelines.

---

## 📁 Project Structure

```
purchase-predictor-vibe/
├── scripts/
│   └── run_pipeline.sh          # 🚀 MAIN ENTRY POINT
├── config/
│   ├── config.yaml              # Configuration settings
│   └── config_loader.py         # Configuration management
├── src/
│   ├── pipeline/                # ML pipeline components
│   ├── utilities/               # Helper utilities and tools
│   └── modules/                 # Shared preprocessing modules
├── docs/                        # 📚 Complete documentation
│   ├── DEPLOYMENT_GUIDE.md      # Deployment strategies
│   ├── API_REFERENCE.md         # API documentation
│   ├── CONFIGURATION.md         # Settings reference
│   ├── TROUBLESHOOTING.md       # Issue resolution
│   ├── DATA_SCHEMA.md           # Data formats
│   └── ARCHITECTURE.md          # Technical design
├── models/                      # Saved models and metadata
├── sample_data/                 # Training and test data
└── processed_data/              # Preprocessed datasets
```

---

## 🔧 Alternative Usage

### Development and Testing

```bash
# Local development without Azure deployment
bash scripts/run_pipeline_local.sh
python src/utilities/local_inference.py

# Test locally
curl http://localhost:5000/test
```

### Azure Container Instance Deployment

```bash
# Deploy using Azure Container Instance
bash scripts/run_pipeline_aci.sh
```

### Step-by-Step Execution

```bash
# Manual pipeline execution
python src/utilities/data_prep.py        # Generate data
python src/pipeline/train.py             # Train model
python src/pipeline/register.py          # Register in Azure ML
python src/pipeline/deploy_managed_endpoint.py  # Deploy
```

---

## 📚 Documentation

### **Quick References**
- **[DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md)** - All deployment options and strategies
- **[API_REFERENCE.md](docs/API_REFERENCE.md)** - REST API usage and integration examples  
- **[TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** - Common issues and solutions

### **Detailed Documentation**
- **[CONFIGURATION.md](docs/CONFIGURATION.md)** - Complete settings and options reference
- **[DATA_SCHEMA.md](docs/DATA_SCHEMA.md)** - Data formats and model specifications
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Technical architecture and design

### **Getting Help**
1. **Quick Issues**: Check [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
2. **Configuration**: See [CONFIGURATION.md](docs/CONFIGURATION.md)
3. **API Integration**: Review [API_REFERENCE.md](docs/API_REFERENCE.md)
4. **Deployment Problems**: Consult [DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md)

---

## ⚙️ Configuration

### Minimal Configuration

Edit `config.yaml` for basic customization:

```yaml
# Model Settings
model:
  type: "random_forest"         # or "logistic_regression"
  random_state: 42

# Deployment Settings  
deployment:
  endpoint_name: "purchase-predictor-endpoint"
  instance_type: "Standard_DS2_v2"
```

### Azure Credentials

Set up `.env.local` with your Azure details:

```bash
AZURE_SUBSCRIPTION_ID=your-subscription-id
AZURE_RESOURCE_GROUP=your-resource-group  
AZURE_WORKSPACE_NAME=your-workspace-name
```

**For complete configuration options, see [CONFIGURATION.md](docs/CONFIGURATION.md).**

---

## 🎯 What This Project Teaches

### **MLOps Fundamentals**
- End-to-end pipeline automation
- Model versioning and registry
- Deployment strategies and monitoring

### **Azure ML SDK v2**
- Modern Azure ML workspace integration
- Managed endpoint deployment
- Model registration and management

### **Production Best Practices**
- Configuration-driven development
- Robust error handling and logging
- Automated testing and validation

### **Data Engineering**
- Feature engineering and preprocessing
- Data validation and quality checks
- Missing data handling strategies

**This project demonstrates basic MLOps patterns that can be extended for production environments.**

---

## 🚀 Next Steps

After completing the quick start:

1. **Explore the API**: See [API_REFERENCE.md](docs/API_REFERENCE.md) for integration examples
2. **Customize the Model**: Modify settings in `config.yaml`
3. **Add Your Data**: Replace synthetic data with real datasets
4. **Scale Deployment**: Configure auto-scaling in Azure ML Studio
5. **Monitor Performance**: Set up model monitoring and alerting

**Ready to build production ML systems?** This project provides the foundation for scalable, maintainable MLOps pipelines.

---

## 📞 Support

- **Documentation**: Complete guides in the `docs/` folder
- **Issues**: Open GitHub issues for bugs or questions
- **Examples**: All code includes comprehensive examples
- **Community**: Share your improvements and extensions

**Happy Machine Learning!** 🤖✨
