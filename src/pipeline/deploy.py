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
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config_loader import load_config
from src.utilities.endpoint_naming import (
    generate_unique_endpoint_name,
    generate_unique_deployment_name,
    create_endpoint_with_cleanup_retry,
    create_deployment_with_retry,
    validate_azure_ml_name
)

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

def create_or_update_endpoint(ml_client, config):
    """Create or update the online endpoint with unique naming."""
    base_endpoint_name = config['deployment'].get('endpoint_name', 'purchase-predictor-endpoint')
    
    # Generate unique endpoint name
    unique_endpoint_name = generate_unique_endpoint_name(base_endpoint_name.split('-')[0])
    
    # Validate the generated name
    is_valid, error_msg = validate_azure_ml_name(unique_endpoint_name, "endpoint")
    if not is_valid:
        logger.warning(f"Generated name validation failed: {error_msg}")
        # Fallback to a simpler unique name
        unique_endpoint_name = generate_unique_endpoint_name("pp")
    
    logger.info(f"Creating endpoint with unique naming:")
    logger.info(f"   Original config: {base_endpoint_name}")
    logger.info(f"   Generated unique: {unique_endpoint_name}")
    
    # Create endpoint configuration
    endpoint_config = ManagedOnlineEndpoint(
        name=unique_endpoint_name,
        description="Purchase predictor model endpoint with unique naming",
        auth_mode="key",
        tags={
            "project": "purchase-predictor",
            "environment": "production",
            "deployment_type": "managed_endpoint_unique",
            "created": time.strftime("%Y-%m-%d_%H-%M-%S"),
            "original_name": base_endpoint_name
        }
    )
    
    logger.info("⏳ Creating endpoint with cleanup and retry logic...")
    endpoint = create_endpoint_with_cleanup_retry(ml_client, endpoint_config)
    
    # Store the actual endpoint name for later use
    config['deployment']['actual_endpoint_name'] = endpoint.name
    
    return endpoint

def create_new_endpoint(ml_client, endpoint_name):
    """Create a new endpoint with a clean state."""
    logger.info(f"Creating new endpoint: {endpoint_name}")
    
    # Create new endpoint
    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        description="Purchase predictor model endpoint",
        auth_mode="key",
        tags={
            "project": "purchase-predictor",
            "environment": "production",
            "created": time.strftime("%Y-%m-%d_%H-%M-%S")
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
    """Create the online deployment with unique naming."""
    base_deployment_name = config['deployment'].get('deployment_name', 'purchase-predictor-deployment')
    
    # Generate unique deployment name
    unique_deployment_name = generate_unique_deployment_name(base_deployment_name.split('-')[0])
    
    # Validate the generated name
    is_valid, error_msg = validate_azure_ml_name(unique_deployment_name, "deployment")
    if not is_valid:
        logger.warning(f"Generated deployment name validation failed: {error_msg}")
        unique_deployment_name = generate_unique_deployment_name("pp-dep")
    
    logger.info(f"Creating deployment with unique naming:")
    logger.info(f"   Original config: {base_deployment_name}")
    logger.info(f"   Generated unique: {unique_deployment_name}")
    logger.info(f"   Target endpoint: {endpoint.name}")
    
    # Get model reference
    model_reference = f"{registration_info['model_name']}:{registration_info['model_version']}"
    
    # Create deployment configuration with unique naming
    deployment_config = ManagedOnlineDeployment(
        name=unique_deployment_name,
        endpoint_name=endpoint.name,
        model=model_reference,
        environment=environment,
        code_configuration=CodeConfiguration(
            code="src/scripts",
            scoring_script="score.py"
        ),
        instance_type="Standard_DS2_v2",
        instance_count=1,
        tags={
            "model_name": registration_info['model_name'],
            "model_version": registration_info['model_version'],
            "deployment_type": "managed_endpoint_unique",
            "original_name": base_deployment_name
        }
    )
    
    # Deploy the model with retry logic
    logger.info("⏳ Starting deployment with retry logic... This may take several minutes.")
    deployment = create_deployment_with_retry(ml_client, deployment_config)
    
    # Store the actual deployment name for later use
    config['deployment']['actual_deployment_name'] = deployment.name
    
    logger.info(f"✅ Deployment {deployment.name} created successfully")
    return deployment

def set_traffic_to_deployment(ml_client, config):
    """Set 100% traffic to the deployment using actual names."""
    # Use actual names that were created (may be different due to unique naming)
    endpoint_name = config['deployment'].get('actual_endpoint_name')
    deployment_name = config['deployment'].get('actual_deployment_name')
    
    if not endpoint_name or not deployment_name:
        logger.error("Missing actual endpoint or deployment names in config")
        return
    
    logger.info(f"Setting traffic routing:")
    logger.info(f"   Endpoint: {endpoint_name}")
    logger.info(f"   Deployment: {deployment_name}")
    
    # Get endpoint and set traffic
    endpoint = ml_client.online_endpoints.get(endpoint_name)
    endpoint.traffic = {deployment_name: 100}
    
    # Update endpoint
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    logger.info("✅ Traffic set to 100% for the deployment")

def get_endpoint_details(ml_client, config):
    """Get and display endpoint details using actual names."""
    # Use actual names that were created
    endpoint_name = config['deployment'].get('actual_endpoint_name')
    deployment_name = config['deployment'].get('actual_deployment_name')
    original_endpoint = config['deployment'].get('endpoint_name', 'unknown')
    original_deployment = config['deployment'].get('deployment_name', 'unknown')
    
    if not endpoint_name:
        logger.error("Missing actual endpoint name in config")
        return None
    
    endpoint = ml_client.online_endpoints.get(endpoint_name)
    
    logger.info("✅ Endpoint details retrieved:")
    logger.info(f"   Name: {endpoint.name}")
    logger.info(f"   Scoring URI: {endpoint.scoring_uri}")
    logger.info(f"   Auth mode: {endpoint.auth_mode}")
    
    # Save comprehensive endpoint info with unique naming details
    endpoint_info = {
        'deployment_type': 'managed_endpoint_unique',
        'naming_strategy': 'unique_names_with_retry',
        'original_names': {
            'endpoint_name': original_endpoint,
            'deployment_name': original_deployment
        },
        'actual_names': {
            'endpoint_name': endpoint.name,
            'deployment_name': deployment_name or 'unknown'
        },
        'endpoint_details': {
            'scoring_uri': endpoint.scoring_uri,
            'auth_mode': endpoint.auth_mode,
            'traffic': getattr(endpoint, 'traffic', {}),
            'provisioning_state': getattr(endpoint, 'provisioning_state', 'unknown')
        },
        'created': time.strftime("%Y-%m-%d_%H-%M-%S")
    }
    
    # Get endpoint info file path from config
    endpoint_info_file = config.get('artifacts', {}).get('endpoint_info_file', 'models/endpoint_info.yaml')
    
    with open(endpoint_info_file, 'w') as f:
        yaml.dump(endpoint_info, f)
    
    logger.info(f"Endpoint info saved to {endpoint_info_file}")
    
    # Display comprehensive deployment summary
    print("\n" + "="*80)
    print("🎉 AZURE ML MANAGED ENDPOINT DEPLOYED SUCCESSFULLY!")
    print("="*80)
    print(f"🌐 Endpoint Name: {endpoint.name}")
    print(f"📊 Original Config: {original_endpoint}")
    print(f"🔑 Unique Naming: ✅ Enabled")
    if deployment_name:
        print(f"🚢 Deployment Name: {deployment_name}")
        print(f"📊 Original Deployment: {original_deployment}")
    print("")
    print(f"📡 Scoring URI: {endpoint.scoring_uri}")
    print(f"🔐 Auth Mode: {endpoint.auth_mode}")
    if hasattr(endpoint, 'traffic') and endpoint.traffic:
        print(f"🔀 Traffic: {endpoint.traffic}")
    print("")
    print("🚀 Your model is hosted on Azure ML managed infrastructure!")
    print("📱 Use the scoring URI above for production predictions")
    print("="*80)
    
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
        import json
        import tempfile
        
        # Create temporary file with test data (Azure ML SDK expects file input)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_file = f.name
        
        try:
            response = ml_client.online_endpoints.invoke(
                endpoint_name=endpoint_name,
                request_file=temp_file,  # Now using the test data via temp file
                deployment_name=config['deployment']['deployment_name']
            )
            
            logger.info(f"✅ Test successful! Predictions: {response}")
            logger.info("Sample interpretations:")
            logger.info("  [25.99, 4, 1, 1] -> Expected: High purchase probability (low price, good rating, previous customer)")
            logger.info("  [150.00, 2, 0, 0] -> Expected: Low purchase probability (high price, poor rating, new customer)")
            
        finally:
            # Clean up temp file
            os.unlink(temp_file)
            
    except Exception as e:
        logger.warning(f"Endpoint test failed: {e}")
        logger.info("This may be normal if the endpoint is still warming up.")
        logger.info("You can test manually once the endpoint is fully ready.")
        logger.info("Test data format:")
        logger.info(json.dumps(test_data, indent=2))

def main():
    """Main deployment function."""
    logger.info("Starting model deployment process...")
    
    # Load configuration
    config = load_config()
    
    # Load model registration info
    registration_info = load_registration_info(config)
    
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