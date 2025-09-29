"""
ACI deployment script for purchase predictor project.
Deploys the registered MLFlow model to Azure Container Instance for simpler deployment.
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

def load_registration_info(config):
    """Load model registration information."""
    registration_info_file = config.get('artifacts', {}).get('registration_info_file', 'models/registration_info.yaml')
    
    if not os.path.exists(registration_info_file):
        raise FileNotFoundError(f"Registration info not found at {registration_info_file}. Please run src/pipeline/register.py first.")
    
    with open(registration_info_file, 'r') as f:
        registration_info = yaml.safe_load(f)
    
    logger.info(f"Loaded registration info for model: {registration_info['model_name']} v{registration_info['model_version']}")
    return registration_info

def create_environment(ml_client, config):
    """Create custom environment for the deployment."""
    environment_name = config['deployment'].get('environment_name', 'purchase-predictor-env')
    
    logger.info(f"Creating environment: {environment_name}")
    
    # Create environment from conda file
    environment = Environment(
        name=environment_name,
        description="Environment for purchase predictor model",
        conda_file="conda.yaml",
        image="mcr.microsoft.com/azureml/minimal-ubuntu20.04-py38-cpu-inference:latest"
    )
    
    # Create the environment
    environment = ml_client.environments.create_or_update(environment)
    logger.info(f"Environment {environment_name} created/updated successfully")
    return environment

def deploy_to_aci(ml_client, config, registration_info, environment):
    """Deploy model using managed online endpoint with minimal resources."""
    base_name = "purchase-predictor"  # Shortened base name
    endpoint_name = f"{base_name}-ep"  # purchase-predictor-ep (15 chars)
    deployment_name = f"{base_name}-dep"  # purchase-predictor-dep (16 chars)
    
    logger.info(f"Deploying to simple endpoint: {endpoint_name}")
    
    # Get model reference
    model_name = registration_info['model_name']
    model_version = registration_info['model_version']
    model_reference = f"{model_name}:{model_version}"
    
    try:
        # Create endpoint
        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            description="Simple deployment for purchase predictor",
            auth_mode="key"
        )
        
        logger.info(f"Creating endpoint: {endpoint_name}")
        endpoint = ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        logger.info(f"Endpoint created: {endpoint.name}")
        
        # Create deployment with minimal configuration
        deployment = ManagedOnlineDeployment(
            name=deployment_name,
            endpoint_name=endpoint_name,
            model=model_reference,
            environment=environment,
            code_configuration=CodeConfiguration(
                code=".",
                scoring_script="src/scripts/score.py"
            ),
            instance_type="Standard_F2s_v2",  # Smallest available instance
            instance_count=1
        )
        
        logger.info(f"Creating deployment: {deployment_name}")
        deployment = ml_client.online_deployments.begin_create_or_update(deployment).result()
        logger.info(f"Deployment created: {deployment.name}")
        
        # Set traffic to 100%
        endpoint.traffic = {deployment_name: 100}
        ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        logger.info("Traffic set to 100%")
        
        # Get final endpoint details
        endpoint = ml_client.online_endpoints.get(endpoint_name)
        
        logger.info(f"Simple deployment completed successfully!")
        logger.info(f"Endpoint name: {endpoint.name}")
        logger.info(f"Scoring URI: {endpoint.scoring_uri}")
        
        # Save deployment info
        deployment_info = {
            'endpoint_name': endpoint.name,
            'deployment_name': deployment_name,
            'scoring_uri': endpoint.scoring_uri,
            'deployment_type': 'simple_managed',
            'model_name': model_name,
            'model_version': model_version
        }
        
        deployment_info_file = config.get('artifacts', {}).get('endpoint_info_file', 'models/deployment_info.yaml')
        with open(deployment_info_file, 'w') as f:
            yaml.dump(deployment_info, f)
        
        logger.info(f"Deployment info saved to {deployment_info_file}")
        return endpoint
        
    except Exception as e:
        logger.error(f"Simple deployment failed: {str(e)}")
        raise

def test_aci_service(endpoint, ml_client, deployment_name):
    """Test the deployed endpoint with sample data."""
    logger.info("Testing endpoint with sample data...")
    
    # Sample test data
    test_data = {
        "data": [
            [25.99, 4, 0, 1],  # price, user_rating, category_encoded, previously_purchased_encoded
            [150.00, 2, 1, 0]
        ]
    }
    
    try:
        import json
        result = ml_client.online_endpoints.invoke(
            endpoint_name=endpoint.name,
            request_file=None,
            deployment_name=deployment_name,
            request_data=json.dumps(test_data)
        )
        logger.info(f"✅ Endpoint test successful! Response: {result}")
        return True
    except Exception as e:
        logger.warning(f"Endpoint test failed: {e}")
        logger.info("Endpoint may still be warming up. Try testing manually later.")
        return False

def main():
    """Main simple deployment function."""
    logger.info("Starting simple model deployment process...")
    
    # Load configuration
    config = load_config()
    
    # Load model registration info
    registration_info = load_registration_info(config)
    
    # Get Azure ML client
    ml_client = get_azure_ml_client(config)
    
    # Create environment
    environment = create_environment(ml_client, config)
    
    # Deploy with simple configuration
    endpoint = deploy_to_aci(ml_client, config, registration_info, environment)
    
    # Test the endpoint
    test_aci_service(endpoint, ml_client, "purchase-predictor-dep")
    
    logger.info("Simple model deployment completed successfully!")
    logger.info(f"Endpoint: {endpoint.name}")
    logger.info(f"Scoring URI: {endpoint.scoring_uri}")
    
    print("\n" + "="*60)
    print("🚀 AZURE ML SIMPLE DEPLOYMENT SUCCESSFUL!")
    print("="*60)
    print(f"Endpoint Name: {endpoint.name}")
    print(f"Scoring URI:   {endpoint.scoring_uri}")
    print("\nExample usage:")
    print("curl -X POST \\")
    print(f'  "{endpoint.scoring_uri}" \\')
    print('  -H "Content-Type: application/json" \\')
    print("  -d '{\"data\": [[25.99, 4, 0, 1], [150.00, 2, 1, 0]]}'")
    print("="*60)

if __name__ == "__main__":
    main()