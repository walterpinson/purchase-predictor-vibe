"""
Model registration script for purchase predictor project.
Registers the trained MLFlow model with Azure ML workspace.
"""

import os
import yaml
import logging
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential
import mlflow
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config_loader import load_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_azure_ml_client(config):
    """Create and return Azure ML client."""
    subscription_id = config['azure']['subscription_id']
    resource_group = config['azure']['resource_group']
    workspace_name = config['azure']['workspace_name']
    
    credential = DefaultAzureCredential()
    
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name
    )
    
    logger.info(f"Connected to Azure ML workspace: {workspace_name}")
    return ml_client

def get_latest_mlflow_run(config):
    """Get the latest MLFlow run ID."""
    # Get run ID file path from config
    run_id_file = config.get('artifacts', {}).get('run_id_file', 'models/run_id.txt')
    
    # Try to read run ID from file first
    if os.path.exists(run_id_file):
        with open(run_id_file, 'r') as f:
            run_id = f.read().strip()
        logger.info(f"Found run ID from file: {run_id}")
        return run_id
    
    # Otherwise, get the latest run from MLFlow
    try:
        experiment_name = config.get('mlflow', {}).get('experiment_name', 'purchase_predictor')
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=1
            )
            if not runs.empty:
                run_id = runs.iloc[0]['run_id']
                logger.info(f"Found latest run ID from MLFlow: {run_id}")
                return run_id
    except Exception as e:
        logger.warning(f"Could not retrieve run from MLFlow: {e}")
    
    raise ValueError("No MLFlow run found. Please run src/pipeline/train.py first.")

def register_model(ml_client, config):
    """Register the MLFlow model with Azure ML."""
    logger.info("Registering model with Azure ML...")
    
    # Get MLFlow run ID
    run_id = get_latest_mlflow_run(config)
    
    # Get model configuration
    model_name = config.get('model_registration', {}).get('name', 'purchase-predictor-model')
    model_description = config.get('model_registration', {}).get('description', 'Binary classifier for purchase prediction')
    artifact_path = config.get('mlflow', {}).get('artifact_path', 'model')
    
    # Create model path (MLFlow format)
    model_path = f"runs:/{run_id}/{artifact_path}"
    
    # Create model entity
    model = Model(
        name=model_name,
        path=model_path,
        description=model_description,
        type="mlflow_model",
        tags={
            "framework": "sklearn",
            "type": "binary_classification",
            "problem": "purchase_prediction"
        }
    )
    
    # Register the model
    registered_model = ml_client.models.create_or_update(model)
    
    logger.info(f"Model registered successfully:")
    logger.info(f"  Name: {registered_model.name}")
    logger.info(f"  Version: {registered_model.version}")
    logger.info(f"  ID: {registered_model.id}")
    
    # Save model registration info for deployment script
    registration_info = {
        'model_name': registered_model.name,
        'model_version': registered_model.version,
        'model_id': registered_model.id,
        'run_id': run_id
    }
    
    # Get registration info file path from config
    registration_info_file = config.get('artifacts', {}).get('registration_info_file', 'models/registration_info.yaml')
    
    with open(registration_info_file, 'w') as f:
        yaml.dump(registration_info, f)
    
    logger.info(f"Registration info saved to {registration_info_file}")
    
    return registered_model

def main():
    """Main registration function."""
    logger.info("Starting model registration process...")
    
    # Load configuration
    config = load_config()
    
    # Get models directory from config
    models_dir = config.get('artifacts', {}).get('models_dir', 'models')
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Get Azure ML client
    ml_client = get_azure_ml_client(config)
    
    # Register model
    registered_model = register_model(ml_client, config)
    
    logger.info("Model registration completed successfully!")
    logger.info(f"Registered model: {registered_model.name} (version {registered_model.version})")

if __name__ == "__main__":
    main()