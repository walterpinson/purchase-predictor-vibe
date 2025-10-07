#!/bin/bash
# Script to fix the corrupted conda environment

echo "🔧 Fixing Purchase Predictor Environment"
echo "========================================"

# Deactivate current environment and return to base
echo "0. Deactivating current environment and returning to base..."
conda deactivate 2>/dev/null || true
conda activate base

# Remove the corrupted environment
echo "1. Removing corrupted environment..."
conda env remove -n purchase-predictor-env -y

# Clean conda cache
echo "2. Cleaning conda cache..."
conda clean --all -y

# Recreate the environment
echo "3. Creating fresh environment..."
conda env create -f conda.yaml

echo "4. Activating environment..."
conda activate purchase-predictor-env

echo "✅ Environment fixed!"
echo ""
echo "Next steps:"
echo "1. Run: conda activate purchase-predictor-env"
echo "2. Test: python --version"
echo "3. Test: python -c 'import azure.ai.ml; print(\"Azure ML SDK imported successfully\")'"
echo "4. Run: python src/utilities/local_inference.py (for local testing)"