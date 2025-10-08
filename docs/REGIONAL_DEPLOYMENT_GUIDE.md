# Regional Deployment Guide for Azure ML Endpoints

## 🌍 **Deploy to East US Without New Workspace**

You can deploy your Azure ML managed online endpoint to the **East US** region without creating a new workspace! Here's how:

### **Configuration**

Your `config/config.yaml` is already updated with regional deployment support:

```yaml
deployment:
  endpoint_name: "purchase-predictor-endpoint-v2"
  deployment_name: "purchase-predictor-deployment-v2"
  environment_name: "purchase-predictor-env"
  instance_type: "Standard_DS2_v2"
  instance_count: 1
  region: "eastus"  # ✅ Deploy to East US region
```

### **How It Works**

🏗️ **Cross-Region Deployment**: Azure ML allows deploying managed endpoints to different regions than your workspace
🌍 **Regional Benefits**: Lower latency for East US users, region-specific compliance, disaster recovery
📊 **Same Workspace**: Model registration and management stay in your current workspace
🔧 **No New Resources**: No need to create new workspace, storage, or compute in East US

### **Supported Regions**

Common regions for Azure ML online endpoints:
- `eastus` - East US (your target)
- `eastus2` - East US 2  
- `westus2` - West US 2
- `centralus` - Central US
- `northeurope` - North Europe
- `westeurope` - West Europe
- `southeastasia` - Southeast Asia
- `australiaeast` - Australia East

### **Usage**

#### **Deploy to East US**:
```bash
# Already configured - just run:
./scripts/run_pipeline.sh
```

#### **Deploy to Workspace Region** (remove regional constraint):
```yaml
# In config.yaml, comment out or remove the region line:
deployment:
  # region: "eastus"  # Commented out = use workspace region
```

#### **Change to Different Region**:
```yaml
deployment:
  region: "westus2"  # Or any supported region
```

### **What You'll See**

The deployment will show regional information:
```
🌍 REGIONAL DEPLOYMENT:
   Target Region: eastus
   Actual Region: eastus  
   Regional Deployment: ✅ Enabled

🌐 Endpoint Name: purchase-predictor-1029-1430-a1b2c3
📡 Scoring URI: https://purchase-predictor-1029-1430-a1b2c3.eastus.inference.ml.azure.com/score
```

### **Benefits of East US Deployment**

✅ **Lower Latency**: Reduced response time for East US users
✅ **Regional Compliance**: Meet data residency requirements  
✅ **Disaster Recovery**: Geographic distribution of endpoints
✅ **Load Distribution**: Spread compute across regions
✅ **Cost Optimization**: Potentially lower costs in East US

### **Considerations**

⚠️ **Quota Requirements**: Ensure you have compute quota in East US region
⚠️ **Instance Availability**: Verify `Standard_DS2_v2` instances available in East US
⚠️ **Network Latency**: Training data/workspace in different region (minimal impact)
⚠️ **Billing**: Compute charges will show East US region usage

### **Troubleshooting**

If deployment fails with regional errors:

1. **Check Quota**: Verify compute quota in East US
2. **Try Different Instance**: Change `instance_type` to `Standard_F2s_v2`
3. **Remove Region**: Comment out region line to use workspace region
4. **Try Alternative**: Use `eastus2` or `centralus` instead

### **Advanced Configuration**

You can also set different regions for different deployment scripts:

```yaml
# For managed endpoints
deployment:
  region: "eastus"

# Could add environment-specific regions
development:
  region: "centralus"  
production:
  region: "eastus"
```

## 🚀 **Ready to Deploy to East US!**

Your configuration is ready - just run:
```bash
./scripts/run_pipeline.sh
```

The deployment will automatically:
1. ✅ Validate East US region is supported
2. ✅ Create unique endpoint name with regional tagging  
3. ✅ Deploy managed infrastructure in East US
4. ✅ Provide East US scoring URI for predictions
5. ✅ Show regional deployment details

**Your model will be hosted in East US while managed from your existing workspace!** 🌍