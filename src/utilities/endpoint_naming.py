"""
Unique endpoint name generation utilities for Azure ML deployments.
Handles endpoint naming best practices and retry logic for robust deployments.
"""

import datetime
import uuid
import time
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

def generate_unique_endpoint_name(base_name="purchase-predictor", max_length=32) -> str:
    """
    Generate a unique endpoint name that complies with Azure ML requirements.
    
    Azure ML endpoint name requirements:
    - Must be 3-32 characters long
    - Can contain only lowercase letters, numbers, and hyphens
    - Must start and end with lowercase letter or number
    - Must be unique within the workspace
    
    Args:
        base_name: Base name for the endpoint (default: "purchase-predictor")
        max_length: Maximum length for the endpoint name (default: 32)
    
    Returns:
        Unique endpoint name following Azure ML naming conventions
    """
    # Ensure base name is compliant
    base_name = base_name.lower().replace("_", "-")
    
    # Generate timestamp and unique ID
    timestamp = datetime.datetime.now().strftime("%m%d-%H%M")  # Shorter format
    unique_id = str(uuid.uuid4())[:6]  # 6 characters for more space
    
    # Construct name with format: base-MMDD-HHMM-uniqueid
    candidate_name = f"{base_name}-{timestamp}-{unique_id}"
    
    # Ensure it fits within max_length
    if len(candidate_name) > max_length:
        # Calculate available space for base name
        suffix_length = len(timestamp) + len(unique_id) + 2  # +2 for hyphens
        available_base_length = max_length - suffix_length
        
        if available_base_length < 3:  # Minimum base name length
            # Use very short base and shorter unique ID
            base_name = "pp"  # purchase-predictor abbreviated
            unique_id = str(uuid.uuid4())[:4]
            candidate_name = f"{base_name}-{timestamp}-{unique_id}"
        else:
            truncated_base = base_name[:available_base_length]
            candidate_name = f"{truncated_base}-{timestamp}-{unique_id}"
    
    logger.info(f"Generated unique endpoint name: {candidate_name}")
    return candidate_name

def generate_unique_deployment_name(base_name="purchase-predictor-deployment", max_length=32) -> str:
    """
    Generate a unique deployment name that complies with Azure ML requirements.
    
    Args:
        base_name: Base name for the deployment
        max_length: Maximum length for the deployment name
    
    Returns:
        Unique deployment name following Azure ML naming conventions
    """
    # Use similar logic but with "dep" suffix to distinguish from endpoint
    base_name = base_name.lower().replace("_", "-")
    
    # Shorter format for deployments
    timestamp = datetime.datetime.now().strftime("%m%d%H%M")  # MMDDHHMM
    unique_id = str(uuid.uuid4())[:4]  # 4 characters
    
    candidate_name = f"{base_name}-{timestamp}-{unique_id}"
    
    # Ensure it fits within max_length
    if len(candidate_name) > max_length:
        # Use abbreviated base name
        base_name = "pp-dep"  # purchase-predictor-deployment abbreviated
        candidate_name = f"{base_name}-{timestamp}-{unique_id}"
        
        if len(candidate_name) > max_length:
            candidate_name = candidate_name[:max_length]
    
    logger.info(f"Generated unique deployment name: {candidate_name}")
    return candidate_name

def create_endpoint_with_cleanup_retry(ml_client, endpoint_config, max_retries=3, retry_delay=300) -> any:
    """
    Create endpoint with comprehensive cleanup and retry logic.
    
    Args:
        ml_client: Azure ML client instance
        endpoint_config: ManagedOnlineEndpoint configuration object
        max_retries: Maximum number of retry attempts (default: 3)
        retry_delay: Delay between retries in seconds (default: 300 = 5 minutes)
    
    Returns:
        Successfully created endpoint object
    
    Raises:
        Exception: If all retry attempts fail
    """
    original_name = endpoint_config.name
    retry_count = 0
    
    while retry_count <= max_retries:
        try:
            logger.info(f"Attempting to create endpoint: {endpoint_config.name} (attempt {retry_count + 1})")
            
            # Try to create the endpoint
            result = ml_client.online_endpoints.begin_create_or_update(endpoint_config).result()
            logger.info(f"✅ Successfully created endpoint: {endpoint_config.name}")
            return result
            
        except Exception as e:
            error_msg = str(e).lower()
            logger.warning(f"❌ Endpoint creation failed (attempt {retry_count + 1}): {e}")
            
            # Check if this is a retryable error
            retryable_errors = [
                "has not been created successfully",
                "endpoint is being deleted",
                "endpoint already exists",
                "provisioning failed",
                "resource conflict",
                "timeout"
            ]
            
            is_retryable = any(err in error_msg for err in retryable_errors)
            
            if not is_retryable or retry_count >= max_retries:
                logger.error(f"Non-retryable error or max retries exceeded: {e}")
                raise e
            
            # Cleanup and retry logic
            logger.info(f"⚠️ Retryable error detected. Initiating cleanup and retry...")
            
            try:
                # Try to cleanup any orphaned endpoint
                logger.info(f"🧹 Attempting to cleanup endpoint: {endpoint_config.name}")
                ml_client.online_endpoints.begin_delete(endpoint_config.name).wait()
                logger.info(f"✅ Cleanup completed for: {endpoint_config.name}")
            except Exception as cleanup_error:
                logger.warning(f"⚠️ Cleanup failed (continuing anyway): {cleanup_error}")
            
            # Wait before retry
            if retry_count < max_retries:
                logger.info(f"⏳ Waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
                
                # Generate new unique name for retry
                retry_suffix = f"retry{retry_count + 1}-{int(time.time() % 10000)}"
                if len(original_name) + len(retry_suffix) + 1 <= 32:
                    endpoint_config.name = f"{original_name}-{retry_suffix}"
                else:
                    # Generate completely new name if too long
                    endpoint_config.name = generate_unique_endpoint_name("pp-retry")
                
                logger.info(f"🔄 Retry with new endpoint name: {endpoint_config.name}")
            
            retry_count += 1
    
    # If we get here, all retries failed
    raise Exception(f"Failed to create endpoint after {max_retries + 1} attempts")

def create_deployment_with_retry(ml_client, deployment_config, max_retries=2, retry_delay=180) -> any:
    """
    Create deployment with retry logic for common deployment failures.
    
    Args:
        ml_client: Azure ML client instance
        deployment_config: ManagedOnlineDeployment configuration object
        max_retries: Maximum number of retry attempts (default: 2)
        retry_delay: Delay between retries in seconds (default: 180 = 3 minutes)
    
    Returns:
        Successfully created deployment object
    """
    original_name = deployment_config.name
    retry_count = 0
    
    while retry_count <= max_retries:
        try:
            logger.info(f"Attempting to create deployment: {deployment_config.name} (attempt {retry_count + 1})")
            
            result = ml_client.online_deployments.begin_create_or_update(deployment_config).result()
            logger.info(f"✅ Successfully created deployment: {deployment_config.name}")
            return result
            
        except Exception as e:
            error_msg = str(e).lower()
            logger.warning(f"❌ Deployment creation failed (attempt {retry_count + 1}): {e}")
            
            retryable_errors = [
                "deployment failed",
                "image build failed",
                "timeout",
                "resource temporarily unavailable",
                "provisioning failed"
            ]
            
            is_retryable = any(err in error_msg for err in retryable_errors)
            
            if not is_retryable or retry_count >= max_retries:
                logger.error(f"Non-retryable error or max retries exceeded: {e}")
                raise e
            
            # Wait and retry with new name
            if retry_count < max_retries:
                logger.info(f"⏳ Waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
                
                # Generate new deployment name
                retry_suffix = f"r{retry_count + 1}-{int(time.time() % 1000)}"
                if len(original_name) + len(retry_suffix) + 1 <= 32:
                    deployment_config.name = f"{original_name}-{retry_suffix}"
                else:
                    deployment_config.name = generate_unique_deployment_name("pp-dep-retry")
                
                logger.info(f"🔄 Retry with new deployment name: {deployment_config.name}")
            
            retry_count += 1
    
    raise Exception(f"Failed to create deployment after {max_retries + 1} attempts")

def validate_azure_ml_name(name: str, name_type: str = "endpoint") -> Tuple[bool, Optional[str]]:
    """
    Validate an Azure ML resource name against Azure requirements.
    
    Args:
        name: Name to validate
        name_type: Type of resource ("endpoint" or "deployment")
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not name:
        return False, f"{name_type} name cannot be empty"
    
    if len(name) < 3:
        return False, f"{name_type} name must be at least 3 characters long"
    
    if len(name) > 32:
        return False, f"{name_type} name must be 32 characters or less"
    
    if not name[0].isalnum() or not name[-1].isalnum():
        return False, f"{name_type} name must start and end with alphanumeric character"
    
    # Check for valid characters (lowercase letters, numbers, hyphens)
    import re
    if not re.match(r'^[a-z0-9-]+$', name):
        return False, f"{name_type} name can only contain lowercase letters, numbers, and hyphens"
    
    # Check for consecutive hyphens
    if '--' in name:
        return False, f"{name_type} name cannot contain consecutive hyphens"
    
    return True, None

def create_regional_endpoint_config(endpoint_name: str, config: dict, description: str = None) -> dict:
    """
    Create endpoint configuration with regional deployment settings.
    
    Args:
        endpoint_name: Name for the endpoint
        config: Configuration dictionary with deployment settings
        description: Optional description for the endpoint
    
    Returns:
        Dictionary with endpoint configuration including regional settings
    """
    target_region = config.get('deployment', {}).get('region', '').strip()
    
    endpoint_config = {
        'name': endpoint_name,
        'description': description or f"Purchase predictor endpoint deployed to {target_region or 'workspace region'}",
        'auth_mode': 'key',
        'tags': {
            'project': 'purchase-predictor',
            'environment': 'production',
            'deployment_type': 'managed_endpoint_regional',
            'created': time.strftime("%Y-%m-%d_%H-%M-%S")
        }
    }
    
    # Add regional configuration if specified
    if target_region:
        endpoint_config['location'] = target_region
        endpoint_config['tags']['target_region'] = target_region
        logger.info(f"🌍 Endpoint will be deployed to region: {target_region}")
    else:
        endpoint_config['tags']['target_region'] = 'workspace_region'
        logger.info("🌍 Endpoint will be deployed to workspace region")
    
    return endpoint_config

def get_supported_regions() -> list:
    """
    Get list of commonly supported regions for Azure ML online endpoints.
    
    Returns:
        List of supported region names
    """
    return [
        'eastus',
        'eastus2', 
        'westus',
        'westus2',
        'westus3',
        'centralus',
        'northcentralus',
        'southcentralus',
        'westcentralus',
        'canadacentral',
        'canadaeast',
        'brazilsouth',
        'northeurope',
        'westeurope',
        'francecentral',
        'germanywestcentral',
        'norwayeast',
        'switzerlandnorth',
        'uksouth',
        'ukwest',
        'southeastasia',
        'eastasia',
        'australiaeast',
        'australiasoutheast',
        'centralindia',
        'southindia',
        'japaneast',
        'japanwest',
        'koreacentral',
        'koreasouth'
    ]

def validate_target_region(region: str) -> Tuple[bool, str]:
    """
    Validate if the target region is supported for Azure ML deployments.
    
    Args:
        region: Region name to validate
    
    Returns:
        Tuple of (is_valid, message)
    """
    if not region:
        return True, "No specific region configured - will use workspace region"
    
    region = region.lower().strip()
    supported_regions = get_supported_regions()
    
    if region in supported_regions:
        return True, f"Region '{region}' is supported for Azure ML deployments"
    else:
        similar_regions = [r for r in supported_regions if region in r or r in region]
        if similar_regions:
            return False, f"Region '{region}' not found. Did you mean: {', '.join(similar_regions[:3])}?"
        else:
            return False, f"Region '{region}' not supported. Supported regions include: {', '.join(supported_regions[:5])}..."