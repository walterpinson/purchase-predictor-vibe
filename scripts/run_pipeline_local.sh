#!/bin/bash
# Complete pipeline script for purchase predictor deployment with Azure ML integration

set -e  # Exit on any error

echo "🚀 Starting Purchase Predictor Pipeline (Azure ML Integrated)..."

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

echo "🚢 Deploying model to Azure ML..."
python src/pipeline/deploy_azure_ml.py

echo "✅ Pipeline completed successfully!"
echo ""
echo "🔗 Check models/azure_ml_deployment_info.yaml for deployment details"
echo "🚀 Start local inference server: python src/scripts/local_inference.py"
echo "🧪 Test predictions: curl http://localhost:5000/test"