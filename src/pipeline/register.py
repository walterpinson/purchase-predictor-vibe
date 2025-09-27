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
    
    # Validate that variables are properly substituted
    for name, value in [('subscription_id', subscription_id), ('resource_group', resource_group), ('workspace_name', workspace_name)]:
        if value.startswith('${') and value.endswith('}'):
            raise ValueError(f"Environment variable substitution failed for {name}: {value}")
    
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
    """Get the latest MLFlow run ID from local tracking."""
    # Try to get the latest run from local MLFlow first
    try:
        experiment_name = config.get('mlflow', {}).get('experiment_name', 'purchase_predictor')
        logger.info(f"Looking for latest run in experiment: {experiment_name}")
        
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            logger.info(f"Found experiment: {experiment.experiment_id}")
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=1
            )
            if not runs.empty:
                run_id = runs.iloc[0]['run_id']
                logger.info(f"Found latest run ID from local MLFlow: {run_id}")
                logger.info(f"Run status: {runs.iloc[0]['status']}")
                logger.info(f"Run start time: {runs.iloc[0]['start_time']}")
                return run_id
        else:
            logger.warning(f"Experiment '{experiment_name}' not found in local MLFlow tracking")
    except Exception as e:
        logger.warning(f"Could not retrieve run from local MLFlow: {e}")
    
    # Fallback to reading from file
    run_id_file = config.get('artifacts', {}).get('run_id_file', 'models/run_id.txt')
    if os.path.exists(run_id_file):
        with open(run_id_file, 'r') as f:
            run_id = f.read().strip()
        logger.info(f"Using run ID from file: {run_id}")
        return run_id
    
    raise ValueError("No MLFlow run found. Please run src/pipeline/train.py first to create a new training run.")

def register_model(ml_client, config):
    """Register the MLFlow model with Azure ML."""
    logger.info("Registering model with Azure ML...")
    
    # Get MLFlow run ID from local tracking
    run_id = get_latest_mlflow_run(config)
    
    # Get model configuration
    model_name = config.get('model_registration', {}).get('name', 'purchase-predictor-model')
    model_description = config.get('model_registration', {}).get('description', 'Binary classifier for purchase prediction')
    
    # Use the local MLFlow model path instead of Azure ML runs path
    # This approach uploads the local model to Azure ML
    local_model_file = config.get('artifacts', {}).get('local_model_file', 'models/model.pkl')
    
    if os.path.exists(local_model_file):
        logger.info(f"Using local model file: {local_model_file}")
        model_path = local_model_file
    else:
        # Fallback to MLFlow run path (may not work with Azure ML)
        logger.warning("Local model file not found, trying MLFlow run path")
        model_path = f"runs:/{run_id}/model"
    
    # Create model entity
    model = Model(
        name=model_name,
        path=model_path,
        description=model_description,
        type="custom_model" if local_model_file else "mlflow_model",
        tags={
            "framework": "sklearn",
            "type": "binary_classification",
            "problem": "purchase_prediction",
            "run_id": run_id
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