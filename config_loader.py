"""
Configuration loader utility for purchase predictor project.
Handles loading configuration from config.yaml with environment variable support.
"""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv


def load_config(config_file='config.yaml', env_file='.env.local'):
    """
    Load configuration from YAML file with environment variable support.
    
    Args:
        config_file (str): Path to the YAML configuration file
        env_file (str): Path to the environment file (optional)
    
    Returns:
        dict: Configuration dictionary
    """
    # Load environment variables from .env.local if it exists
    env_path = Path(env_file)
    if env_path.exists():
        load_dotenv(env_path)
    
    # Load YAML configuration
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"{config_file} not found. Please create it with your configuration.")
    
    # Replace environment variable placeholders in Azure config
    if 'azure' in config:
        azure_config = config['azure']
        
        # Replace ${VARIABLE_NAME} patterns with actual environment values
        if 'subscription_id' in azure_config:
            subscription_id = azure_config['subscription_id']
            if subscription_id.startswith('${') and subscription_id.endswith('}'):
                var_name = subscription_id[2:-1]  # Remove ${ and }
                azure_config['subscription_id'] = os.getenv(var_name)
                if not azure_config['subscription_id']:
                    raise ValueError(f"Environment variable {var_name} not found")
        
        if 'resource_group' in azure_config:
            resource_group = azure_config['resource_group']
            if resource_group.startswith('${') and resource_group.endswith('}'):
                var_name = resource_group[2:-1]  # Remove ${ and }
                azure_config['resource_group'] = os.getenv(var_name)
                if not azure_config['resource_group']:
                    raise ValueError(f"Environment variable {var_name} not found")
        
        if 'workspace_name' in azure_config:
            workspace_name = azure_config['workspace_name']
            if workspace_name.startswith('${') and workspace_name.endswith('}'):
                var_name = workspace_name[2:-1]  # Remove ${ and }
                azure_config['workspace_name'] = os.getenv(var_name)
                if not azure_config['workspace_name']:
                    raise ValueError(f"Environment variable {var_name} not found")
    
    return config


def validate_azure_config(config):
    """
    Validate that required Azure configuration is present.
    
    Args:
        config (dict): Configuration dictionary
    
    Raises:
        ValueError: If required Azure configuration is missing
    """
    if 'azure' not in config:
        raise ValueError("Azure configuration section missing from config")
    
    azure_config = config['azure']
    required_fields = ['subscription_id', 'resource_group', 'workspace_name']
    
    for field in required_fields:
        if field not in azure_config or not azure_config[field]:
            raise ValueError(f"Required Azure configuration field '{field}' is missing or empty")
        
        if not isinstance(azure_config[field], str) or not azure_config[field].strip():
            raise ValueError(f"Azure configuration field '{field}' must be a non-empty string")