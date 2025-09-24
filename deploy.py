"""
Model deployment script for purchase predictor project.
Deploys the registered MLFlow model to an Azure ML managed online endpoint.
"""

import os
import yaml
import logging
import time
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint, 
    ManagedOnlineDeployment,
    Environment,
    CodeConfiguration
)
from azure.identity import DefaultAzureCredential
from config_loader import load_config

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

def load_registration_info():
    """Load model registration information."""
    registration_file = 'models/registration_info.yaml'
    if not os.path.exists(registration_file):
        raise FileNotFoundError("Registration info not found. Please run register.py first.")
    
    with open(registration_file, 'r') as f:
        registration_info = yaml.safe_load(f)
    
    logger.info(f"Loaded registration info for model: {registration_info['model_name']} v{registration_info['model_version']}")
    return registration_info

def create_or_update_endpoint(ml_client, config):
    """Create or update the online endpoint."""
    endpoint_name = config['deployment']['endpoint_name']
    
    logger.info(f"Creating/updating endpoint: {endpoint_name}")
    
    # Check if endpoint exists
    try:
        endpoint = ml_client.online_endpoints.get(endpoint_name)
        logger.info(f"Endpoint {endpoint_name} already exists")
        return endpoint
    except Exception:
        logger.info(f"Creating new endpoint: {endpoint_name}")
        
        # Create new endpoint
        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            description="Purchase predictor model endpoint",
            auth_mode="key",
            tags={
                "project": "purchase-predictor",
                "environment": "production"
            }
        )
        
        # Create the endpoint
        endpoint = ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        logger.info(f"Endpoint {endpoint_name} created successfully")
        return endpoint

def create_environment(ml_client, config):
    """Create custom environment for the deployment."""
    environment_name = config['deployment'].get('environment_name', 'purchase-predictor-env')
    
    logger.info(f"Creating environment: {environment_name}")
    
    # Create environment from conda file
    environment = Environment(
        name=environment_name,
        description="Environment for purchase predictor model",
        conda_file="conda.yaml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
    )
    
    # Create the environment
    environment = ml_client.environments.create_or_update(environment)
    logger.info(f"Environment {environment_name} created/updated successfully")
    return environment

def create_deployment(ml_client, config, registration_info, endpoint, environment):
    """Create the online deployment."""
    deployment_name = config['deployment']['deployment_name']
    endpoint_name = config['deployment']['endpoint_name']
    
    logger.info(f"Creating deployment: {deployment_name}")
    
    # Get model reference
    model_reference = f"{registration_info['model_name']}:{registration_info['model_version']}"
    
    # Create deployment
    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=model_reference,
        environment=environment,
        code_configuration=CodeConfiguration(
            code=".",
            scoring_script="score.py"
        ),
        instance_type="Standard_DS2_v2",
        instance_count=1,
        request_settings={
            "request_timeout_ms": 90000,
            "max_concurrent_requests_per_instance": 1,
            "max_queue_wait_ms": 500
        },
        liveness_probe={
            "failure_threshold": 30,
            "success_threshold": 1,
            "timeout": 2,
            "period": 10,
            "initial_delay": 10
        },
        readiness_probe={
            "failure_threshold": 10,
            "success_threshold": 1,
            "timeout": 10,
            "period": 10,
            "initial_delay": 10
        }
    )
    
    # Deploy the model
    logger.info("Starting deployment... This may take several minutes.")
    deployment = ml_client.online_deployments.begin_create_or_update(deployment).result()
    
    logger.info(f"Deployment {deployment_name} created successfully")
    return deployment

def set_traffic_to_deployment(ml_client, config):
    """Set 100% traffic to the deployment."""
    endpoint_name = config['deployment']['endpoint_name']
    deployment_name = config['deployment']['deployment_name']
    
    logger.info(f"Setting traffic to deployment: {deployment_name}")
    
    # Get endpoint and set traffic
    endpoint = ml_client.online_endpoints.get(endpoint_name)
    endpoint.traffic = {deployment_name: 100}
    
    # Update endpoint
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    logger.info("Traffic set to 100% for the deployment")

def get_endpoint_details(ml_client, config):
    """Get and display endpoint details."""
    endpoint_name = config['deployment']['endpoint_name']
    
    endpoint = ml_client.online_endpoints.get(endpoint_name)
    
    logger.info("Endpoint details:")
    logger.info(f"  Name: {endpoint.name}")
    logger.info(f"  Scoring URI: {endpoint.scoring_uri}")
    logger.info(f"  Auth mode: {endpoint.auth_mode}")
    
    # Save endpoint info for later use
    endpoint_info = {
        'endpoint_name': endpoint.name,
        'scoring_uri': endpoint.scoring_uri,
        'auth_mode': endpoint.auth_mode
    }
    
    with open('models/endpoint_info.yaml', 'w') as f:
        yaml.dump(endpoint_info, f)
    
    logger.info("Endpoint info saved to models/endpoint_info.yaml")
    return endpoint

def test_endpoint(ml_client, config):
    """Test the deployed endpoint with sample data."""
    endpoint_name = config['deployment']['endpoint_name']
    
    logger.info("Testing endpoint with sample data...")
    
    # Sample test data
    test_data = {
        "data": [
            [25.99, 4, 1, 1],  # price, user_rating, category_encoded, previously_purchased_encoded
            [150.00, 2, 0, 0]
        ]
    }
    
    try:
        response = ml_client.online_endpoints.invoke(
            endpoint_name=endpoint_name,
            request_file=None,
            deployment_name=config['deployment']['deployment_name']
        )
        logger.info(f"Test response: {response}")
    except Exception as e:
        logger.warning(f"Test failed (this is normal initially): {e}")
        logger.info("You can test the endpoint manually once it's fully ready.")

def main():
    """Main deployment function."""
    logger.info("Starting model deployment process...")
    
    # Load configuration
    config = load_config()
    
    # Load model registration info
    registration_info = load_registration_info()
    
    # Get Azure ML client
    ml_client = get_azure_ml_client(config)
    
    # Create or update endpoint
    endpoint = create_or_update_endpoint(ml_client, config)
    
    # Create environment
    environment = create_environment(ml_client, config)
    
    # Create deployment
    deployment = create_deployment(ml_client, config, registration_info, endpoint, environment)
    
    # Set traffic to the deployment
    set_traffic_to_deployment(ml_client, config)
    
    # Get endpoint details
    endpoint = get_endpoint_details(ml_client, config)
    
    # Test endpoint
    test_endpoint(ml_client, config)
    
    logger.info("Model deployment completed successfully!")
    logger.info(f"Endpoint: {endpoint.name}")
    logger.info(f"Scoring URI: {endpoint.scoring_uri}")
    logger.info("Use the scoring URI to make predictions via REST API")

if __name__ == "__main__":
    main()