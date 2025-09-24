#!/bin/bash
# Complete pipeline script for purchase predictor deployment

set -e  # Exit on any error

echo "🚀 Starting Purchase Predictor Pipeline..."

# Check if conda environment exists
if ! conda info --envs | grep -q "purchase-predictor"; then
    echo "📦 Creating conda environment..."
    conda env create -f conda.yaml
fi

echo "🔧 Activating conda environment..."
source activate purchase-predictor

echo "📊 Generating synthetic data..."
python data_prep.py

echo "🤖 Training model..."
python train.py

echo "📝 Registering model with Azure ML..."
python register.py

echo "🚢 Deploying model to endpoint..."
python deploy.py

echo "✅ Pipeline completed successfully!"
echo ""
echo "🔗 Check models/endpoint_info.yaml for endpoint details"
echo "📊 Use the scoring URI to make predictions via REST API"