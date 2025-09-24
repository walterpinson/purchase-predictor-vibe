"""
Configuration loader utility for purchase predictor project.
Handles loading configuration from config.yaml with environment variable support using piny.
"""

import os
from pathlib import Path
from piny import YamlLoader
from dotenv import load_dotenv


def load_config(config_file='config.yaml', env_file='.env.local'):
    """
    Load configuration from YAML file with automatic environment variable substitution.
    
    Args:
        config_file (str): Path to the YAML configuration file
        env_file (str): Path to the environment file (optional)
    
    Returns:
        dict: Configuration dictionary with environment variables expanded
    """
    # Load environment variables from .env.local if it exists
    env_path = Path(env_file)
    if env_path.exists():
        load_dotenv(env_path)
    
    try:
        # Use piny to load YAML with automatic environment variable substitution
        config = YamlLoader(path=config_file).load()
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"{config_file} not found. Please create it with your configuration.")
    except Exception as e:
        raise ValueError(f"Error loading configuration from {config_file}: {str(e)}")


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