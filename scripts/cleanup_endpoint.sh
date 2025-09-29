#!/bin/bash

# Cleanup script for Azure ML endpoints
# Run this if you need to manually clean up endpoints that are stuck in bad states

echo "🧹 Azure ML Endpoint Cleanup Script"
echo "=================================="

# Load configuration
ENDPOINT_NAME=$(python -c "
import sys
sys.path.append('.')
from config.config_loader import load_config
config = load_config()
print(config['deployment']['endpoint_name'])
")

echo "Endpoint name: $ENDPOINT_NAME"
echo ""

# Check current endpoint status
echo "📊 Checking current endpoint status..."
az ml online-endpoint show --name $ENDPOINT_NAME --resource-group rg-ml-dojo --workspace-name ws-ml-dojo --query "{Name:name, State:provisioning_state, Traffic:traffic}" --output table 2>/dev/null

if [ $? -eq 0 ]; then
    echo ""
    echo "⚠️  Endpoint exists. Do you want to delete it? (y/N)"
    read -r response
    
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "🗑️  Deleting endpoint: $ENDPOINT_NAME"
        az ml online-endpoint delete --name $ENDPOINT_NAME --resource-group rg-ml-dojo --workspace-name ws-ml-dojo --yes
        
        if [ $? -eq 0 ]; then
            echo "✅ Endpoint deleted successfully"
            echo "🚀 You can now run: python src/pipeline/deploy.py"
        else
            echo "❌ Failed to delete endpoint"
        fi
    else
        echo "👍 Keeping existing endpoint"
    fi
else
    echo "✅ No existing endpoint found - ready for fresh deployment"
    echo "🚀 You can run: python src/pipeline/deploy.py"
fi

echo ""
echo "📋 To manually check endpoints:"
echo "   az ml online-endpoint list --resource-group rg-ml-dojo --workspace-name ws-ml-dojo --output table"