"""
ACI deployment script for purchase predictor project.
Deploys the registered MLFlow model to Azure Container Instance for simpler deployment.
"""

import os
import yaml
import logging
import time
import datetime
import shutil
import json
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
    
    logger.info(f"📋 Loaded registration info:")
    logger.info(f"   Model: {registration_info['model_name']} v{registration_info['model_version']}")
    return registration_info

def prepare_deployment_artifacts():
    """
    Prepare deployment artifacts with archival system for ACI deployment.
    
    Creates a clean /server directory with current deployment files and archives
    previous deployments by timestamp for debugging and rollback purposes.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    logger.info("🗃️ Preparing ACI deployment artifacts with archival system...")
    
    # Create server directory structure
    os.makedirs('server', exist_ok=True)
    archive_dir = f'server/archives/{timestamp}'
    os.makedirs(archive_dir, exist_ok=True)
    
    # Archive current deployment if it exists
    archived_files = []
    if os.path.exists('server/score.py'):
        shutil.copy2('server/score.py', f'{archive_dir}/score.py')
        archived_files.append('score.py')
    
    if os.path.exists('server/preprocessing.py'):
        shutil.copy2('server/preprocessing.py', f'{archive_dir}/preprocessing.py')
        archived_files.append('preprocessing.py')
    
    if archived_files:
        logger.info(f"📦 Archived previous deployment to {archive_dir}:")
        for file in archived_files:
            logger.info(f"   ├── {file}")
        
        # Create archive metadata
        archive_metadata = {
            'archived_at': timestamp,
            'archived_files': archived_files,
            'archive_reason': 'new_aci_deployment',
            'deployment_type': 'aci_style_unique'
        }
        
        with open(f'{archive_dir}/archive_info.json', 'w') as f:
            json.dump(archive_metadata, f, indent=2)
        
        logger.info(f"   └── archive_info.json")
    
    # Copy new deployment files
    logger.info("📋 Copying fresh deployment artifacts...")
    
    # Copy score.py
    if not os.path.exists('src/scripts/score.py'):
        raise FileNotFoundError("src/scripts/score.py not found")
    shutil.copy2('src/scripts/score.py', 'server/score.py')
    logger.info("✅ Copied src/scripts/score.py -> server/score.py")
    
    # Copy preprocessing.py
    if not os.path.exists('src/modules/preprocessing.py'):
        raise FileNotFoundError("src/modules/preprocessing.py not found")
    shutil.copy2('src/modules/preprocessing.py', 'server/preprocessing.py')
    logger.info("✅ Copied src/modules/preprocessing.py -> server/preprocessing.py")
    
    logger.info("🔧 Score.py already configured with simple imports for deployment")
    
    # Create current deployment metadata
    current_metadata = {
        'deployed_at': timestamp,
        'deployment_files': ['score.py', 'preprocessing.py'],
        'source_info': {
            'score_script_source': 'src/scripts/score.py',
            'preprocessing_source': 'src/modules/preprocessing.py'
        },
        'deployment_type': 'aci_style_unique',
        'instance_type': 'Standard_F2s_v2',
        'archive_location': archive_dir if archived_files else None
    }
    
    with open('server/deployment_info.json', 'w') as f:
        json.dump(current_metadata, f, indent=2)
    
    logger.info("✅ ACI deployment artifacts prepared successfully!")
    logger.info("📁 Server directory structure:")
    logger.info("   server/")
    logger.info("   ├── score.py                 # Azure ML scoring script")
    logger.info("   ├── preprocessing.py         # Preprocessing module")
    logger.info("   ├── deployment_info.json     # Current deployment metadata")
    if archived_files:
        logger.info(f"   └── archives/{timestamp}/    # Previous deployment archive")
        for file in archived_files:
            logger.info(f"       ├── {file}")
        logger.info(f"       └── archive_info.json")
    
    return 'server'

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
    """Deploy model using managed online endpoint with unique naming and retry logic."""
    base_name = "purchase-predictor-aci"
    
    # Generate unique names for ACI deployment
    unique_endpoint_name = generate_unique_endpoint_name(base_name)
    unique_deployment_name = generate_unique_deployment_name(f"{base_name}-dep")
    
    # Validate generated names
    is_valid_ep, error_ep = validate_azure_ml_name(unique_endpoint_name, "endpoint")
    is_valid_dep, error_dep = validate_azure_ml_name(unique_deployment_name, "deployment")
    
    if not is_valid_ep:
        logger.warning(f"Generated endpoint name validation failed: {error_ep}")
        unique_endpoint_name = generate_unique_endpoint_name("pp-aci")
    
    if not is_valid_dep:
        logger.warning(f"Generated deployment name validation failed: {error_dep}")
        unique_deployment_name = generate_unique_deployment_name("pp-aci-dep")
    
    logger.info(f"🐳 Deploying to ACI with unique naming:")
    logger.info(f"   Endpoint: {unique_endpoint_name}")
    logger.info(f"   Deployment: {unique_deployment_name}")
    
    # Get model reference
    model_name = registration_info['model_name']
    model_version = registration_info['model_version']
    model_reference = f"{model_name}:{model_version}"
    
    # Prepare deployment artifacts with archival
    server_dir = prepare_deployment_artifacts()
    
    try:
        # Create endpoint configuration
        endpoint_config = ManagedOnlineEndpoint(
            name=unique_endpoint_name,
            description="ACI-style deployment for purchase predictor with unique naming and archival system",
            auth_mode="key",
            tags={
                "project": "purchase-predictor",
                "deployment_type": "aci_style_unique_archival",
                "created": time.strftime("%Y-%m-%d_%H-%M-%S"),
                "archival_system": "enabled"
            }
        )
        
        logger.info("⏳ Creating ACI endpoint with retry logic...")
        endpoint = create_endpoint_with_cleanup_retry(ml_client, endpoint_config)
        
        # Create deployment configuration with unique naming and archival system
        deployment_config = ManagedOnlineDeployment(
            name=unique_deployment_name,
            endpoint_name=endpoint.name,
            model=model_reference,
            environment=environment,
            code_configuration=CodeConfiguration(
                code=server_dir,  # Use the archival server directory
                scoring_script="score.py"
            ),
            instance_type="Standard_F2s_v2",  # Smallest available instance for ACI-style
            instance_count=1,
            tags={
                "model_name": model_name,
                "model_version": model_version,
                "deployment_type": "aci_style_unique_archival",
                "archival_system": "enabled"
            }
        )
        
        logger.info("⏳ Creating ACI deployment with retry logic and archival system...")
        deployment = create_deployment_with_retry(ml_client, deployment_config)
        
        # Set traffic to 100%
        endpoint_updated = ml_client.online_endpoints.get(endpoint.name)
        endpoint_updated.traffic = {deployment.name: 100}
        ml_client.online_endpoints.begin_create_or_update(endpoint_updated).result()
        logger.info("Traffic set to 100%")
        
        # Get final endpoint details
        final_endpoint = ml_client.online_endpoints.get(endpoint.name)
        
        logger.info(f"✅ ACI-style deployment completed successfully!")
        logger.info(f"   Endpoint name: {final_endpoint.name}")
        logger.info(f"   Deployment name: {deployment.name}")
        logger.info(f"   Scoring URI: {final_endpoint.scoring_uri}")
        
        # Save deployment info with unique naming and archival details
        deployment_info = {
            'deployment_type': 'aci_style_unique_archival',
            'naming_strategy': 'unique_names_with_retry',
            'archival_system': {
                'enabled': True,
                'server_directory': server_dir,
                'artifacts': ['score.py', 'preprocessing.py'],
                'deployment_metadata': 'server/deployment_info.json'
            },
            'endpoint_name': final_endpoint.name,
            'deployment_name': deployment.name,
            'scoring_uri': final_endpoint.scoring_uri,
            'auth_mode': final_endpoint.auth_mode,
            'model_name': model_name,
            'model_version': model_version,
            'instance_type': 'Standard_F2s_v2',
            'created': time.strftime("%Y-%m-%d_%H-%M-%S")
        }
        
        deployment_info_file = config.get('artifacts', {}).get('endpoint_info_file', 'models/endpoint_info.yaml')
        with open(deployment_info_file, 'w') as f:
            yaml.dump(deployment_info, f)
        
        logger.info(f"Deployment info saved to {deployment_info_file}")
        return final_endpoint, deployment.name
        
    except Exception as e:
        logger.error(f"ACI-style deployment failed: {str(e)}")
        logger.error("This may be due to:")
        logger.error("  - Resource quota limitations")
        logger.error("  - Subscription restrictions")
        logger.error("  - Azure service availability")
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
    """Main ACI-style deployment function with unique naming and archival system."""
    logger.info("Starting ACI-style model deployment with unique naming and archival system...")
    
    # Load configuration
    config = load_config()
    
    # Load model registration info
    registration_info = load_registration_info(config)
    
    # Get Azure ML client
    ml_client = get_azure_ml_client(config)
    
    # Create environment
    environment = create_environment(ml_client, config)
    
    # Deploy with ACI-style configuration, unique naming, and archival system
    endpoint, deployment_name = deploy_to_aci(ml_client, config, registration_info, environment)
    
    # Test the endpoint
    test_aci_service(endpoint, ml_client, deployment_name)
    
    logger.info("✅ ACI-style model deployment completed successfully!")
    logger.info(f"   Endpoint: {endpoint.name}")
    logger.info(f"   Deployment: {deployment_name}")
    logger.info(f"   Scoring URI: {endpoint.scoring_uri}")
    
    print("\n" + "="*70)
    print("🚀 AZURE ML ACI-STYLE DEPLOYMENT SUCCESSFUL!")
    print("="*70)
    print(f"🌐 Endpoint Name: {endpoint.name}")
    print(f"🚢 Deployment Name: {deployment_name}")
    print(f"📡 Scoring URI: {endpoint.scoring_uri}")
    print(f"🔑 Unique Naming: ✅ Enabled")
    print(f"🗃️ Archival System: ✅ Enabled")
    print("")
    print("🐳 ACI-style deployment provides containerized inference")
    print("💰 Cost-optimized with Standard_F2s_v2 instances")
    print("📁 Deployment artifacts archived in server/archives/")
    print("📱 Use the scoring URI above for predictions")
    print("🎛️ Monitor in Azure ML Studio portal")
    print("="*70)
    print("\nExample usage:")
    print("curl -X POST \\")
    print(f'  "{endpoint.scoring_uri}" \\')
    print('  -H "Content-Type: application/json" \\')
    print("  -d '{\"data\": [[25.99, 4, 0, 1], [150.00, 2, 1, 0]]}'")
    print("="*60)

if __name__ == "__main__":
    main()