#!/bin/bash
# Complete pipeline script for purchase predictor deployment using ACI approach

set -e  # Exit on any error

echo "🚀 Starting Purchase Predictor Pipeline (ACI Deployment)..."

# Check if conda environment exists
if ! conda info --envs | grep -q "purchase-predictor-env"; then
    echo "📦 Creating conda environment..."
    conda env create -f ./conda.yaml
fi

echo "🔧 Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate purchase-predictor-env

echo "📊 Generating synthetic data..."
python src/pipeline/data_prep.py

echo "🤖 Training model..."
python src/pipeline/train.py

echo "📝 Registering model with Azure ML..."
python src/pipeline/register.py

echo "🚢 Deploying model to ACI endpoint..."
python src/pipeline/deploy_aci.py

echo "✅ Pipeline completed successfully!"
echo ""
echo "🔗 Check models/endpoint_info.yaml for ACI endpoint details"
echo "📊 Use the scoring URI to make predictions via REST API"
echo "🐳 ACI deployment provides containerized inference in Azure"