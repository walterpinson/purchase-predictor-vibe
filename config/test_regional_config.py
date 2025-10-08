#!/usr/bin/env python3
"""
Test script to verify Azure ML SDK regional deployment behavior.
"""

import json
from config_loader import load_config  # Fixed import - now in same directory
from azure.ai.ml.entities import ManagedOnlineEndpoint

def test_endpoint_config():
    """Test how the ManagedOnlineEndpoint handles location parameter."""
    
    print("🧪 Testing Azure ML SDK Regional Deployment")
    print("="*60)
    
    # Load config
    config = load_config()
    target_region = config['deployment'].get('region', '').strip()
    
    print(f"Target region from config: {target_region}")
    print()
    
    # Test ManagedOnlineEndpoint configuration
    print("🔬 Testing ManagedOnlineEndpoint with location parameter:")
    
    endpoint_config = ManagedOnlineEndpoint(
        name="test-endpoint-name",
        description="Test endpoint",
        auth_mode="key",
        location=target_region,  # This is the key parameter
    )
    
    print(f"   endpoint_config.name: {endpoint_config.name}")
    print(f"   endpoint_config.location: {getattr(endpoint_config, 'location', 'NOT_SET')}")
    print(f"   endpoint_config.auth_mode: {endpoint_config.auth_mode}")
    print()
    
    # Test without location parameter
    print("🔬 Testing ManagedOnlineEndpoint WITHOUT location parameter:")
    
    endpoint_config_no_location = ManagedOnlineEndpoint(
        name="test-endpoint-no-location",
        description="Test endpoint without location",
        auth_mode="key"
        # No location parameter
    )
    
    print(f"   endpoint_config.name: {endpoint_config_no_location.name}")
    print(f"   endpoint_config.location: {getattr(endpoint_config_no_location, 'location', 'NOT_SET')}")
    print()
    
    # Check Azure ML SDK version
    try:
        import azure.ai.ml
        print(f"Azure ML SDK version: {azure.ai.ml.__version__}")
    except:
        print("Could not determine Azure ML SDK version")
    
    print()
    print("💡 Analysis:")
    if hasattr(endpoint_config, 'location') and endpoint_config.location:
        print(f"   ✅ Location parameter is being set: {endpoint_config.location}")
        print("   🤔 The configuration looks correct...")
        print("   🔍 The issue might be:")
        print("      - Azure ML SDK ignoring the location parameter")
        print("      - Workspace region override in Azure")
        print("      - Subscription-level region restrictions") 
        print("      - The location parameter syntax is wrong")
    else:
        print("   ❌ Location parameter is NOT being set properly")
        print("   🔧 This explains why deployment goes to workspace region")

if __name__ == "__main__":
    test_endpoint_config()