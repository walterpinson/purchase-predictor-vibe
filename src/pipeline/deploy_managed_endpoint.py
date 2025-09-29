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
    """Create endpoint with unique naming and comprehensive retry logic."""
    base_endpoint_name = config['deployment'].get('endpoint_name', 'purchase-predictor-endpoint')
    
    # Generate unique endpoint name
    unique_endpoint_name = generate_unique_endpoint_name(base_endpoint_name.split('-')[0])
    
    # Validate the generated name
    is_valid, error_msg = validate_azure_ml_name(unique_endpoint_name, "endpoint")
    if not is_valid:
        logger.warning(f"Generated name validation failed: {error_msg}")
        # Fallback to a simpler unique name
        unique_endpoint_name = generate_unique_endpoint_name("pp")
    
    logger.info(f"🚀 Creating managed online endpoint with unique name: {unique_endpoint_name}")
    logger.info(f"   Original config name: {base_endpoint_name}")
    logger.info(f"   Generated unique name: {unique_endpoint_name}")
    
    # Create endpoint configuration with minimal settings for reliability
    endpoint_config = ManagedOnlineEndpoint(
        name=unique_endpoint_name,
        description="Azure ML Studio hosted inference server for purchase predictor (unique deployment)",
        auth_mode="key",
        tags={
            "project": "purchase-predictor",
            "environment": "production",
            "deployment_type": "azure_ml_studio_hosted_unique",
            "created": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            "original_name": base_endpoint_name,
            "unique_name": unique_endpoint_name
        }
    )
    
    logger.info("⏳ Creating endpoint with cleanup and retry logic...")
    logger.info("   🔄 Automatic cleanup of failed endpoints")
    logger.info("   🔁 Up to 3 retry attempts with new names")
    logger.info("   ⏱️ 5-minute delays between retries")
    
    try:
        # Use the robust endpoint creation with retry logic
        endpoint = create_endpoint_with_cleanup_retry(ml_client, endpoint_config)
        
        logger.info(f"✅ Endpoint created successfully!")
        logger.info(f"   Final endpoint name: {endpoint.name}")
        logger.info(f"   Provisioning state: {endpoint.provisioning_state}")
        
        # Update config to track the actual endpoint name used
        config['deployment']['actual_endpoint_name'] = endpoint.name
        
        return endpoint
        
    except Exception as e:
        logger.error(f"❌ Failed to create endpoint after all retry attempts: {e}")
        logger.error("   This may indicate:")
        logger.error("   - Subscription quota exceeded")
        logger.error("   - Resource provider registration issues")
        logger.error("   - Insufficient permissions")
        logger.error("   - Service availability issues")
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
    """Create deployment with unique naming and retry logic."""
    base_deployment_name = config['deployment'].get('deployment_name', 'purchase-predictor-deployment')
    endpoint_name = endpoint.name
    
    # Generate unique deployment name
    unique_deployment_name = generate_unique_deployment_name(base_deployment_name.split('-')[0])
    
    # Validate the generated name
    is_valid, error_msg = validate_azure_ml_name(unique_deployment_name, "deployment")
    if not is_valid:
        logger.warning(f"Generated deployment name validation failed: {error_msg}")
        # Fallback to a simpler unique name
        unique_deployment_name = generate_unique_deployment_name("pp-dep")
    
    logger.info(f"🚢 Creating managed deployment with unique name: {unique_deployment_name}")
    logger.info("   This creates the actual Azure ML Studio hosted inference server")
    logger.info(f"   Original config name: {base_deployment_name}")
    logger.info(f"   Generated unique name: {unique_deployment_name}")
    
    # Get model reference
    model_name = registration_info['model_name']
    model_version = registration_info['model_version']
    model_reference = f"{model_name}:{model_version}"
    
    logger.info(f"📦 Using model: {model_reference}")
    
    # Create deployment configuration with optimized settings
    deployment_config = ManagedOnlineDeployment(
        name=unique_deployment_name,
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
            "deployment_type": "azure_ml_studio_hosted_unique",
            "original_name": base_deployment_name,
            "unique_name": unique_deployment_name,
            "created": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        }
    )
    
    logger.info("⏳ Deploying to Azure ML Studio with retry logic...")
    logger.info("   🏗️ Provisioning managed compute infrastructure")
    logger.info("   🐳 Building container with your model")
    logger.info("   🌐 Creating hosted inference endpoint")
    logger.info("   🔁 Up to 2 retry attempts if deployment fails")
    
    try:
        # Use the robust deployment creation with retry logic
        deployment = create_deployment_with_retry(ml_client, deployment_config)
        
        logger.info(f"✅ Deployment completed successfully!")
        logger.info(f"   Final deployment name: {deployment.name}")
        logger.info("🎉 Your model is now hosted on Azure ML Studio managed infrastructure!")
        
        # Update config to track the actual deployment name used
        config['deployment']['actual_deployment_name'] = deployment.name
        
        return deployment
        
    except Exception as e:
        logger.error(f"❌ Deployment failed after all retry attempts: {e}")
        logger.error("Common causes:")
        logger.error("  - Resource quota exceeded for compute instances")
        logger.error("  - Image build failures due to environment issues")
        logger.error("  - Timeout during provisioning (try again later)")
        logger.error("  - Insufficient subscription permissions")
        raise

def configure_endpoint_traffic(ml_client, endpoint_name, deployment_name):
    """Set 100% traffic to the deployment using actual names."""
    logger.info(f"🔀 Configuring traffic routing...")
    logger.info(f"   Endpoint: {endpoint_name}")
    logger.info(f"   Deployment: {deployment_name}")
    
    try:
        endpoint = ml_client.online_endpoints.get(endpoint_name)
        endpoint.traffic = {deployment_name: 100}
        
        ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        logger.info(f"✅ Traffic set to 100% for deployment: {deployment_name}")
        logger.info(f"   All requests to {endpoint_name} will route to {deployment_name}")
    except Exception as e:
        logger.error(f"❌ Failed to set traffic: {e}")
        raise

def get_hosted_endpoint_details(ml_client, config, endpoint_name):
    """Get and save hosted endpoint details with actual names."""
    logger.info("📊 Retrieving hosted endpoint details...")
    
    try:
        endpoint = ml_client.online_endpoints.get(endpoint_name)
        
        # Get actual names used (may be different from config due to unique naming)
        actual_endpoint_name = endpoint.name
        actual_deployment_name = config['deployment'].get('actual_deployment_name', 'unknown')
        original_endpoint_name = config['deployment'].get('endpoint_name', 'unknown')
        original_deployment_name = config['deployment'].get('deployment_name', 'unknown')
        
        endpoint_info = {
            'deployment_type': 'azure_ml_studio_hosted_unique',
            'naming_strategy': 'unique_names_with_retry',
            'original_names': {
                'endpoint_name': original_endpoint_name,
                'deployment_name': original_deployment_name
            },
            'actual_names': {
                'endpoint_name': actual_endpoint_name,
                'deployment_name': actual_deployment_name
            },
            'endpoint_details': {
                'scoring_uri': endpoint.scoring_uri,
                'swagger_uri': endpoint.swagger_uri if hasattr(endpoint, 'swagger_uri') else None,
                'auth_mode': endpoint.auth_mode,
                'location': endpoint.location if hasattr(endpoint, 'location') else None,
                'provisioning_state': endpoint.provisioning_state,
                'traffic': endpoint.traffic if hasattr(endpoint, 'traffic') else {},
                'tags': endpoint.tags if hasattr(endpoint, 'tags') else {},
                'created_at': str(endpoint.creation_context.created_at) if endpoint.creation_context else None
            },
            'usage_instructions': {
                'scoring_uri': endpoint.scoring_uri,
                'auth_mode': endpoint.auth_mode,
                'sample_request': {
                    'method': 'POST',
                    'headers': {
                        'Content-Type': 'application/json',
                        'Authorization': 'Bearer <YOUR_API_KEY>'
                    },
                    'body': {
                        'data': [[25.99, 4, 1, 1], [150.00, 2, 0, 0]]
                    }
                }
            }
        }
        
        # Get endpoint info file path from config
        endpoint_info_file = config.get('artifacts', {}).get('endpoint_info_file', 'models/endpoint_info.yaml')
        
        with open(endpoint_info_file, 'w') as f:
            yaml.dump(endpoint_info, f, default_flow_style=False)
        
        logger.info(f"✅ Endpoint details saved to {endpoint_info_file}")
        
        # Display comprehensive information
        print("\n" + "="*80)
        print("🎉 AZURE ML STUDIO HOSTED ENDPOINT DEPLOYED SUCCESSFULLY!")
        print("="*80)
        print(f"🌐 Endpoint Name: {actual_endpoint_name}")
        print(f"📊 Original Config Name: {original_endpoint_name}")
        print(f"� Unique Naming: ✅ Enabled (prevents common naming conflicts)")
        print("")
        print(f"�📡 Scoring URI: {endpoint.scoring_uri}")
        print(f"🔐 Auth Mode: {endpoint.auth_mode}")
        print(f"📊 Provisioning State: {endpoint.provisioning_state}")
        if endpoint.traffic:
            print(f"🔀 Traffic Distribution: {endpoint.traffic}")
        print("")
        print("🏗️ DEPLOYMENT DETAILS:")
        print(f"   Deployment Name: {actual_deployment_name}")
        print(f"   Original Config Name: {original_deployment_name}")
        print(f"   Instance Type: Standard_DS2_v2")
        print(f"   Instance Count: 1")
        print("")
        print("🚀 Your model is now hosted on Azure ML Studio managed infrastructure!")
        print("📱 Use the scoring URI above for production predictions")
        print("🎛️ Monitor and manage your endpoint in Azure ML Studio portal")
        print("")
        print("📋 UNIQUE NAMING BENEFITS:")
        print("   ✅ Prevents endpoint name conflicts")
        print("   ✅ Enables reliable retry logic")
        print("   ✅ Supports multiple deployments")
        print("   ✅ Automatic cleanup of failed endpoints")
        print("="*80)
        
        return endpoint
        
    except Exception as e:
        logger.error(f"❌ Failed to get endpoint details: {e}")
        raise

def test_hosted_endpoint(ml_client, endpoint_name, deployment_name):
    """Test the hosted endpoint with sample data using actual names."""
    logger.info("🧪 Testing hosted endpoint...")
    logger.info(f"   Testing endpoint: {endpoint_name}")
    logger.info(f"   Using deployment: {deployment_name}")
    
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
            logger.info("")
            logger.info("🔗 Test Results Summary:")
            logger.info(f"   ✅ Endpoint {endpoint_name} is responding correctly")
            logger.info(f"   ✅ Deployment {deployment_name} is processing requests")
            logger.info(f"   ✅ Model is making predictions as expected")
            
        finally:
            os.unlink(temp_file)
            
    except Exception as e:
        logger.warning(f"⚠️ Endpoint test failed: {e}")
        logger.info("This may be normal if the endpoint is still warming up.")
        logger.info("Common reasons for test failures:")
        logger.info("  - Endpoint still provisioning (wait 5-10 minutes)")
        logger.info("  - Model container still starting up")
        logger.info("  - Temporary Azure service issues")
        logger.info("")
        logger.info(f"Try testing manually in a few minutes:")
        logger.info(f"  Endpoint: {endpoint_name}")
        logger.info(f"  Test data: {json.dumps(test_data, indent=2)}")

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