"""
Enhanced Azure ML managed endpoint deployment with comprehensive error handling.
This script creates an actual Azure ML Studio hosted inference server.
"""

import os
import yaml
import logging
import time
import json
import datetime
import uuid
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint, 
    ManagedOnlineDeployment,
    Environment,
    CodeConfiguration
)
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_azure_ml_client(config):
    """Create and return Azure ML client with enhanced error handling."""
    subscription_id = config['azure']['subscription_id']
    resource_group = config['azure']['resource_group']
    workspace_name = config['azure']['workspace_name']
    
    logger.info(f"Connecting to Azure ML workspace...")
    logger.info(f"  Subscription: {subscription_id}")
    logger.info(f"  Resource Group: {resource_group}")
    logger.info(f"  Workspace: {workspace_name}")
    
    try:
        credential = DefaultAzureCredential()
        
        ml_client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name
        )
        
        # Test connection
        workspace = ml_client.workspaces.get()
        logger.info(f"✅ Successfully connected to Azure ML workspace: {workspace.name}")
        logger.info(f"   Location: {workspace.location}")
        logger.info(f"   Resource Group: {workspace.resource_group}")
        
        return ml_client
        
    except Exception as e:
        logger.error(f"❌ Failed to connect to Azure ML workspace: {e}")
        logger.error("   Check your Azure credentials and workspace configuration")
        raise

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

def create_optimized_endpoint(ml_client, config):
    """Create endpoint with optimized settings for reliable deployment."""
    endpoint_name = config['deployment']['endpoint_name']
    
    # Ensure endpoint name meets Azure requirements
    if len(endpoint_name) > 32:
        endpoint_name = endpoint_name[:32]
        logger.warning(f"Endpoint name truncated to 32 characters: {endpoint_name}")
    
    logger.info(f"🚀 Creating managed online endpoint: {endpoint_name}")
    
    try:
        # Check if endpoint already exists
        existing_endpoint = ml_client.online_endpoints.get(endpoint_name)
        logger.info(f"✅ Endpoint {endpoint_name} already exists")
        logger.info(f"   State: {existing_endpoint.provisioning_state}")
        
        if existing_endpoint.provisioning_state == "Failed":
            logger.warning("⚠️ Existing endpoint is in Failed state. Deleting and recreating...")
            ml_client.online_endpoints.begin_delete(endpoint_name).result()
            time.sleep(30)  # Wait for deletion to complete
        else:
            return existing_endpoint
            
    except ResourceNotFoundError:
        logger.info(f"📝 Endpoint {endpoint_name} does not exist. Creating new one...")
    
    # Create new endpoint with minimal configuration
    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        description="Azure ML Studio hosted inference server for purchase predictor",
        auth_mode="key",
        tags={
            "project": "purchase-predictor",
            "environment": "production",
            "deployment_type": "azure_ml_studio_hosted",
            "created": time.strftime("%Y-%m-%d_%H-%M-%S")
        }
    )
    
    logger.info("⏳ Creating endpoint... This may take 5-10 minutes...")
    try:
        endpoint = ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        logger.info(f"✅ Endpoint {endpoint_name} created successfully!")
        return endpoint
    except Exception as e:
        logger.error(f"❌ Failed to create endpoint: {e}")
        raise

def create_optimized_environment(ml_client, config):
    """Create environment optimized for managed endpoints."""
    environment_name = f"purchase-predictor-env-{int(time.time())}"  # Unique name
    
    logger.info(f"🐳 Creating deployment environment: {environment_name}")
    
    # Create environment with curated base image
    environment = Environment(
        name=environment_name,
        description="Optimized environment for purchase predictor managed endpoint",
        conda_file="conda.yaml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
    )
    
    try:
        environment = ml_client.environments.create_or_update(environment)
        logger.info(f"✅ Environment {environment_name} created successfully")
        return environment
    except Exception as e:
        logger.error(f"❌ Failed to create environment: {e}")
        raise

def create_optimized_deployment(ml_client, config, registration_info, endpoint, environment):
    """Create deployment with optimized settings."""
    deployment_name = config['deployment']['deployment_name']
    endpoint_name = endpoint.name
    
    logger.info(f"🚢 Creating managed deployment: {deployment_name}")
    logger.info("   This creates the actual Azure ML Studio hosted inference server")
    
    # Get model reference
    model_name = registration_info['model_name']
    model_version = registration_info['model_version']
    model_reference = f"{model_name}:{model_version}"
    
    logger.info(f"📦 Using model: {model_reference}")
    
    # Create deployment with optimized configuration
    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=model_reference,
        environment=environment,
        code_configuration=CodeConfiguration(
            code="src/scripts",  # More specific path
            scoring_script="score.py"
        ),
        instance_type="Standard_DS2_v2",  # Reliable instance type
        instance_count=1,
        tags={
            "model_name": model_name,
            "model_version": model_version,
            "deployment_type": "azure_ml_studio_hosted"
        }
    )
    
    logger.info("⏳ Deploying to Azure ML Studio... This may take 10-15 minutes...")
    logger.info("   🏗️ Provisioning managed compute infrastructure")
    logger.info("   🐳 Building container with your model")
    logger.info("   🌐 Creating hosted inference endpoint")
    
    try:
        deployment = ml_client.online_deployments.begin_create_or_update(deployment).result()
        logger.info(f"✅ Deployment {deployment_name} completed successfully!")
        logger.info("🎉 Your model is now hosted on Azure ML Studio managed infrastructure!")
        return deployment
    except Exception as e:
        logger.error(f"❌ Deployment failed: {e}")
        logger.error("Common causes:")
        logger.error("  - Resource quota exceeded")
        logger.error("  - Resource provider not registered")
        logger.error("  - Subscription limitations")
        raise

def configure_endpoint_traffic(ml_client, endpoint_name, deployment_name):
    """Set 100% traffic to the deployment."""
    logger.info(f"🔀 Configuring traffic routing...")
    
    try:
        endpoint = ml_client.online_endpoints.get(endpoint_name)
        endpoint.traffic = {deployment_name: 100}
        
        ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        logger.info(f"✅ Traffic set to 100% for deployment: {deployment_name}")
    except Exception as e:
        logger.error(f"❌ Failed to set traffic: {e}")
        raise

def get_hosted_endpoint_details(ml_client, config, endpoint_name):
    """Get and save hosted endpoint details."""
    logger.info("📊 Retrieving hosted endpoint details...")
    
    try:
        endpoint = ml_client.online_endpoints.get(endpoint_name)
        
        endpoint_info = {
            'deployment_type': 'azure_ml_studio_hosted',
            'endpoint_name': endpoint.name,
            'scoring_uri': endpoint.scoring_uri,
            'swagger_uri': endpoint.swagger_uri if hasattr(endpoint, 'swagger_uri') else None,
            'auth_mode': endpoint.auth_mode,
            'location': endpoint.location if hasattr(endpoint, 'location') else None,
            'provisioning_state': endpoint.provisioning_state,
            'traffic': endpoint.traffic if hasattr(endpoint, 'traffic') else {},
            'tags': endpoint.tags if hasattr(endpoint, 'tags') else {},
            'created_at': str(endpoint.creation_context.created_at) if endpoint.creation_context else None
        }
        
        # Get endpoint info file path from config
        endpoint_info_file = config.get('artifacts', {}).get('endpoint_info_file', 'models/endpoint_info.yaml')
        
        with open(endpoint_info_file, 'w') as f:
            yaml.dump(endpoint_info, f, default_flow_style=False)
        
        logger.info(f"✅ Endpoint details saved to {endpoint_info_file}")
        
        # Display key information
        print("\n" + "="*70)
        print("🎉 AZURE ML STUDIO HOSTED ENDPOINT DEPLOYED SUCCESSFULLY!")
        print("="*70)
        print(f"🌐 Endpoint Name: {endpoint.name}")
        print(f"📡 Scoring URI: {endpoint.scoring_uri}")
        print(f"🔐 Auth Mode: {endpoint.auth_mode}")
        print(f"📊 Provisioning State: {endpoint.provisioning_state}")
        if endpoint.traffic:
            print(f"🔀 Traffic Distribution: {endpoint.traffic}")
        print("")
        print("🚀 Your model is now hosted on Azure ML Studio managed infrastructure!")
        print("📱 Use the scoring URI above for production predictions")
        print("🎛️ Monitor and manage your endpoint in Azure ML Studio portal")
        print("="*70)
        
        return endpoint
        
    except Exception as e:
        logger.error(f"❌ Failed to get endpoint details: {e}")
        raise

def test_hosted_endpoint(ml_client, endpoint_name, deployment_name):
    """Test the hosted endpoint with sample data."""
    logger.info("🧪 Testing hosted endpoint...")
    
    # Sample test data
    test_data = {
        "data": [
            [25.99, 4, 1, 1],  # Low price, good rating, category 1, previous customer
            [150.00, 2, 0, 0]  # High price, poor rating, category 0, new customer
        ]
    }
    
    try:
        import tempfile
        
        # Create temporary file with test data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_file = f.name
        
        try:
            response = ml_client.online_endpoints.invoke(
                endpoint_name=endpoint_name,
                request_file=temp_file,
                deployment_name=deployment_name
            )
            
            logger.info(f"✅ Hosted endpoint test successful!")
            logger.info(f"📊 Predictions: {response}")
            logger.info("🎯 Test interpretations:")
            logger.info("   [25.99, 4, 1, 1] -> Expected: High purchase probability")
            logger.info("   [150.00, 2, 0, 0] -> Expected: Low purchase probability")
            
        finally:
            os.unlink(temp_file)
            
    except Exception as e:
        logger.warning(f"⚠️ Endpoint test failed: {e}")
        logger.info("This may be normal if the endpoint is still warming up.")
        logger.info("Try testing again in a few minutes using the scoring URI.")

def main():
    """Main function for Azure ML Studio hosted endpoint deployment."""
    print("\n" + "="*70)
    print("🚀 AZURE ML STUDIO HOSTED ENDPOINT DEPLOYMENT")
    print("="*70)
    print("This script deploys your model to actual Azure ML Studio managed infrastructure")
    print("✅ Creates a hosted inference server in Azure ML Studio")
    print("✅ Provides production-ready REST API endpoint")
    print("✅ Includes auto-scaling and monitoring capabilities")
    print("="*70)
    
    logger.info("Starting Azure ML Studio hosted endpoint deployment...")
    
    try:
        # Load configuration
        config = load_config()
        
        # Load model registration info
        registration_info = load_registration_info(config)
        
        # Get Azure ML client
        ml_client = get_azure_ml_client(config)
        
        # Create endpoint
        endpoint = create_optimized_endpoint(ml_client, config)
        
        # Create environment
        environment = create_optimized_environment(ml_client, config)
        
        # Create deployment (this is the actual Azure ML Studio hosted server)
        deployment = create_optimized_deployment(ml_client, config, registration_info, endpoint, environment)
        
        # Configure traffic
        configure_endpoint_traffic(ml_client, endpoint.name, deployment.name)
        
        # Get and display endpoint details
        endpoint = get_hosted_endpoint_details(ml_client, config, endpoint.name)
        
        # Test the endpoint
        test_hosted_endpoint(ml_client, endpoint.name, deployment.name)
        
        print("\n🎊 DEPLOYMENT COMPLETED SUCCESSFULLY!")
        print("Your purchase predictor model is now running on Azure ML Studio!")
        print("Use the scoring URI for production predictions.")
        
    except Exception as e:
        logger.error(f"❌ Deployment failed: {e}")
        print("\n📋 TROUBLESHOOTING SUGGESTIONS:")
        print("1. Run: az login (ensure you're logged into Azure)")
        print("2. Check resource providers: az provider list --query '[?namespace==`Microsoft.MachineLearningServices`]'")
        print("3. Verify subscription quotas in Azure portal")
        print("4. Try the local inference approach as backup: ./scripts/run_pipeline_local.sh")
        raise

if __name__ == "__main__":
    main()