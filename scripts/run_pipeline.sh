#!/bin/bash
# Comprehensive Azure ML Studio hosted endpoint deployment script
# This script handles resource provider registration and troubleshooting

set -e  # Exit on any error

echo "🚀 Starting Azure ML Studio Hosted Endpoint Deployment..."
echo "📋 This will deploy to an actual Azure ML Studio managed online endpoint"
echo ""

# Function to check and register resource providers
check_and_register_providers() {
    echo "🔧 Checking Azure resource provider registrations..."
    
    # Required resource providers for Azure ML managed endpoints
    providers=(
        "Microsoft.MachineLearningServices"
        "Microsoft.ContainerInstance" 
        "Microsoft.Storage"
        "Microsoft.KeyVault"
        "Microsoft.ContainerRegistry"
        "Microsoft.Compute"
        "Microsoft.Network"
        "Microsoft.ServiceBus"
        "Microsoft.Relay"
        "Microsoft.EventHub"
    )
    
    for provider in "${providers[@]}"; do
        echo "Checking $provider..."
        status=$(az provider show --namespace "$provider" --query "registrationState" -o tsv 2>/dev/null || echo "NotFound")
        
        if [ "$status" != "Registered" ]; then
            echo "🔄 Registering $provider..."
            az provider register --namespace "$provider"
            echo "✅ $provider registration initiated"
        else
            echo "✅ $provider already registered"
        fi
    done
    
    echo ""
    echo "⏳ Waiting for provider registrations to complete..."
    sleep 30
    
    echo "📊 Final provider status check:"
    for provider in "${providers[@]}"; do
        status=$(az provider show --namespace "$provider" --query "registrationState" -o tsv 2>/dev/null || echo "NotFound")
        echo "  $provider: $status"
    done
    echo ""
}

# Function to check subscription quotas
check_quotas() {
    echo "📊 Checking Azure ML compute quotas..."
    
    # Get subscription info
    subscription_id=$(az account show --query id -o tsv)
    echo "Subscription ID: $subscription_id"
    
    # Check if we can create managed endpoints
    echo "🔍 Checking managed endpoint quota..."
    
    # Try to list existing endpoints to verify access
    echo "📋 Checking existing endpoints..."
    az ml online-endpoint list --query "[].name" -o table || echo "⚠️ Unable to list endpoints - this may indicate quota or permission issues"
    
    echo ""
}

# Function to check Azure CLI version and login
check_azure_cli() {
    echo "🔧 Checking Azure CLI setup..."
    
    # Check Azure CLI version
    az_version=$(az version --query '["azure-cli"]' -o tsv)
    echo "Azure CLI version: $az_version"
    
    # Check login status
    echo "🔐 Checking Azure login status..."
    account_info=$(az account show --query '{subscription:id,tenant:tenantId,user:user.name}' -o table)
    echo "$account_info"
    
    # Check ML extension
    echo "🔧 Checking Azure ML CLI extension..."
    az extension show --name ml --query version -o tsv || {
        echo "📦 Installing Azure ML CLI extension..."
        az extension add --name ml
    }
    
    echo ""
}

# Main execution
main() {
    echo "🎯 AZURE ML STUDIO HOSTED ENDPOINT DEPLOYMENT"
    echo "=============================================="
    echo ""
    
    # Check Azure CLI setup
    check_azure_cli
    
    # Check and register resource providers
    check_and_register_providers
    
    # Check quotas and permissions
    check_quotas
    
    # Check if conda environment exists
    if ! conda info --envs | grep -q "purchase-predictor-env"; then
        echo "📦 Creating conda environment..."
        conda env create -f ./conda.yaml
    fi

    echo "🔧 Activating conda environment..."
    eval "$(conda shell.bash hook)"
    conda activate purchase-predictor-env

    echo "📊 Generating synthetic data..."
    python src/utilities/data_prep.py

    echo "🤖 Training model..."
    python src/pipeline/train.py

    echo "📝 Registering model with Azure ML..."
    python src/pipeline/register.py

    echo ""
    echo "🚢 DEPLOYING TO AZURE ML STUDIO MANAGED ENDPOINT..."
    echo "=================================================="
    echo "This creates an actual Azure ML Studio hosted inference server"
    echo "⏳ This may take 10-15 minutes for the managed endpoint to provision..."
    echo ""
    
    # Run the managed endpoint deployment
    python src/pipeline/deploy_managed_endpoint.py

    echo ""
    echo "✅ AZURE ML STUDIO HOSTED ENDPOINT DEPLOYMENT COMPLETED!"
    echo "========================================================"
    echo ""
    echo "🔗 Check models/endpoint_info.yaml for your hosted endpoint details"
    echo "📊 Your model is now running on Azure ML Studio managed infrastructure"
    echo "🌐 Use the scoring URI from endpoint_info.yaml to make predictions"
    echo ""
    echo "📋 Next steps:"
    echo "  1. Check Azure ML Studio portal for your endpoint"
    echo "  2. Use the scoring URI for production predictions"
    echo "  3. Monitor endpoint performance in Azure ML Studio"
    echo ""
}

# Check if we're running in the right directory
if [ ! -f "config/config.yaml" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    echo "Current directory: $(pwd)"
    echo "Expected files: config/config.yaml"
    exit 1
fi

# Check if Azure CLI is available
if ! command -v az &> /dev/null; then
    echo "❌ Error: Azure CLI is not installed"
    echo "Please install Azure CLI: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
    exit 1
fi

# Run main function
main