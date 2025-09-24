#!/usr/bin/env python3
"""
Test script to verify configuration loading with piny.
Run this to test that your .env.local file and config.yaml are set up correctly.
"""

from config.config_loader import load_config, validate_azure_config


def main():
    """Test configuration loading."""
    print("🧪 Testing configuration loading...")
    
    try:
        # Load configuration
        config = load_config()
        print("✅ Configuration loaded successfully!")
        
        # Validate Azure configuration
        validate_azure_config(config)
        print("✅ Azure configuration validated!")
        
        # Display loaded values (without exposing sensitive data)
        azure_config = config['azure']
        print("\n📋 Configuration Summary:")
        print(f"   Subscription ID: {azure_config['subscription_id'][:8]}...")
        print(f"   Resource Group: {azure_config['resource_group']}")
        print(f"   Workspace Name: {azure_config['workspace_name']}")
        print(f"   Model Type: {config.get('model', {}).get('type', 'Not specified')}")
        print(f"   Endpoint Name: {config.get('deployment', {}).get('endpoint_name', 'Not specified')}")
        
        print("\n🎉 Configuration test completed successfully!")
        
    except FileNotFoundError as e:
        print(f"❌ Configuration file not found: {e}")
        print("   Make sure config.yaml exists in the current directory.")
        
    except ValueError as e:
        print(f"❌ Configuration validation failed: {e}")
        print("   Check your .env.local file and ensure all required variables are set.")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()