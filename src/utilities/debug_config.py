#!/usr/bin/env python3
"""
Debug script to test configuration loading and region settings.
"""

import json
from config.config_loader import load_config

def debug_config_loading():
    """Debug the configuration loading to see what's happening with the region setting."""
    
    print("🐛 DEBUG: Configuration Loading Analysis")
    print("="*60)
    
    try:
        # Load the configuration
        config = load_config()
        
        print("✅ Configuration loaded successfully")
        print()
        
        print("📋 Full configuration structure:")
        print(json.dumps(config, indent=2, default=str))
        print()
        
        print("🎯 Deployment section analysis:")
        deployment_section = config.get('deployment', {})
        print(f"   deployment section exists: {bool(deployment_section)}")
        print(f"   deployment section type: {type(deployment_section)}")
        print(f"   deployment section content: {deployment_section}")
        print()
        
        print("🌍 Region configuration analysis:")
        if 'deployment' in config:
            region_raw = config['deployment'].get('region')
            region_stripped = config['deployment'].get('region', '').strip() if region_raw else ''
            
            print(f"   region key exists: {'region' in config['deployment']}")
            print(f"   region raw value: '{region_raw}'")
            print(f"   region raw type: {type(region_raw)}")
            print(f"   region stripped: '{region_stripped}'")
            print(f"   region is truthy: {bool(region_stripped)}")
            print(f"   region length: {len(region_stripped) if region_stripped else 0}")
        else:
            print("   ❌ deployment section not found!")
        
        print()
        
        print("💡 Diagnosis:")
        if 'deployment' not in config:
            print("   ❌ Missing 'deployment' section in config.yaml")
        elif 'region' not in config['deployment']:
            print("   ❌ Missing 'region' key in deployment section")
        elif not config['deployment'].get('region', '').strip():
            print("   ❌ Region value is empty, None, or whitespace")
            print("   🔧 This explains why deployment goes to workspace region (centralus)")
        else:
            region = config['deployment']['region'].strip()
            print(f"   ✅ Region is properly configured: '{region}'")
            if region.lower() == 'eastus':
                print("   ✅ Region is set to East US as expected")
            else:
                print(f"   ⚠️ Region is set to '{region}' (not eastus)")
        
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_config_loading()