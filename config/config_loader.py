"""
Configuration loader utility for purchase predictor project.
Handles loading configuration from config.yaml with environment variable support using piny.
"""

import os
from pathlib import Path
from piny import YamlLoader
from dotenv import load_dotenv


def load_config(config_file=None, env_file=None):
    """
    Load configuration from YAML file with automatic environment variable substitution.
    
    Args:
        config_file (str): Path to the YAML configuration file (defaults to config/config.yaml from project root)
        env_file (str): Path to the environment file (defaults to .env.local from project root)
    
    Returns:
        dict: Configuration dictionary with environment variables expanded
    """
    # Find project root (assuming config_loader.py is in config/ directory)
    project_root = Path(__file__).parent.parent
    
    # Set default paths relative to project root
    if config_file is None:
        config_file = project_root / 'config' / 'config.yaml'
    elif not os.path.isabs(config_file):
        config_file = project_root / config_file
    
    if env_file is None:
        env_file = project_root / '.env.local'
    elif not os.path.isabs(env_file):
        env_file = project_root / env_file
    
    # Load environment variables from .env.local if it exists
    if Path(env_file).exists():
        load_dotenv(env_file)
    
    try:
        # Use piny to load YAML with automatic environment variable substitution
        config = YamlLoader(path=str(config_file)).load()
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