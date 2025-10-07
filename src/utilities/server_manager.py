#!/usr/bin/env python3
"""
Server deployment management utility.
Helps manage deployment artifacts and archives in the /server directory.
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path

def list_deployment_archives():
    """List all archived deployments."""
    archives_dir = Path('server/archives')
    
    if not archives_dir.exists():
        print("📁 No deployment archives found.")
        return
    
    archives = sorted([d for d in archives_dir.iterdir() if d.is_dir()], reverse=True)
    
    if not archives:
        print("📁 No deployment archives found.")
        return
    
    print(f"📚 Found {len(archives)} deployment archive(s):")
    print("="*70)
    
    for archive in archives:
        timestamp = archive.name
        info_file = archive / 'deployment_info.json'
        
        print(f"📦 {timestamp}")
        
        if info_file.exists():
            try:
                with open(info_file, 'r') as f:
                    info = json.load(f)
                
                print(f"   Archived at: {info.get('archived_at', 'Unknown')}")
                print(f"   Files: {', '.join(info.get('archived_files', []))}")
                
                if 'source_info' in info:
                    print(f"   Sources:")
                    for key, value in info['source_info'].items():
                        print(f"     {key}: {value}")
                
            except Exception as e:
                print(f"   ⚠️ Could not read metadata: {e}")
        
        # List files in archive
        files = [f.name for f in archive.iterdir() if f.is_file()]
        print(f"   Files in archive: {', '.join(files)}")
        print()

def show_current_deployment():
    """Show information about the current deployment."""
    server_dir = Path('server')
    
    if not server_dir.exists():
        print("📁 No server directory found. Run deployment first.")
        return
    
    print("🎯 Current Deployment Status:")
    print("="*50)
    
    # Check for current deployment files
    score_file = server_dir / 'score.py'
    preprocessing_file = server_dir / 'preprocessing.py'
    info_file = server_dir / 'deployment_info.json'
    
    print(f"Score script: {'✅ Present' if score_file.exists() else '❌ Missing'}")
    print(f"Preprocessing: {'✅ Present' if preprocessing_file.exists() else '❌ Missing'}")
    
    if info_file.exists():
        try:
            with open(info_file, 'r') as f:
                info = json.load(f)
            
            print(f"\n📋 Deployment Metadata:")
            print(f"   Deployed at: {info.get('deployed_at', 'Unknown')}")
            print(f"   Type: {info.get('deployment_type', 'Unknown')}")
            print(f"   Files: {', '.join(info.get('deployment_files', []))}")
            
            if 'source_info' in info:
                print(f"   Source files:")
                for key, value in info['source_info'].items():
                    print(f"     {key}: {value}")
            
            if info.get('archive_location'):
                print(f"   Previous version archived at: {info['archive_location']}")
                
        except Exception as e:
            print(f"⚠️ Could not read deployment metadata: {e}")
    else:
        print("\n📋 No deployment metadata found")

def clean_old_archives(keep_count=5):
    """Clean old deployment archives, keeping only the most recent ones."""
    archives_dir = Path('server/archives')
    
    if not archives_dir.exists():
        print("📁 No archives directory found.")
        return
    
    archives = sorted([d for d in archives_dir.iterdir() if d.is_dir()], reverse=True)
    
    if len(archives) <= keep_count:
        print(f"📦 Found {len(archives)} archives, all within keep limit of {keep_count}")
        return
    
    to_remove = archives[keep_count:]
    
    print(f"🧹 Cleaning up {len(to_remove)} old archive(s), keeping {keep_count} most recent:")
    
    for archive in to_remove:
        print(f"   Removing: {archive.name}")
        shutil.rmtree(archive)
    
    print("✅ Archive cleanup completed!")

def prepare_fresh_deployment():
    """Prepare a fresh deployment by clearing the server directory."""
    server_dir = Path('server')
    
    if not server_dir.exists():
        print("📁 No server directory found. Nothing to clean.")
        return
    
    print("🧹 Preparing fresh deployment environment...")
    
    # Archive current deployment if it exists
    if (server_dir / 'score.py').exists() or (server_dir / 'preprocessing.py').exists():
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        archive_dir = server_dir / 'archives' / f'manual-cleanup-{timestamp}'
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        for file in ['score.py', 'preprocessing.py', 'deployment_info.json']:
            file_path = server_dir / file
            if file_path.exists():
                shutil.copy2(file_path, archive_dir / file)
                file_path.unlink()
        
        print(f"📦 Current deployment archived to: {archive_dir}")
    
    print("✅ Server directory cleaned and ready for fresh deployment!")

def show_server_structure():
    """Show the complete server directory structure."""
    server_dir = Path('server')
    
    if not server_dir.exists():
        print("📁 No server directory found.")
        return
    
    print("📁 Server Directory Structure:")
    print("="*40)
    
    def print_tree(path, prefix=""):
        items = sorted(path.iterdir())
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            print(f"{prefix}{current_prefix}{item.name}")
            
            if item.is_dir():
                next_prefix = prefix + ("    " if is_last else "│   ")
                print_tree(item, next_prefix)
    
    print_tree(server_dir)

def main():
    """Main CLI interface."""
    import sys
    
    if len(sys.argv) < 2:
        print("🛠️ Server Deployment Management Utility")
        print("="*45)
        print("Usage: python server_manager.py <command>")
        print()
        print("Commands:")
        print("  list       - List all deployment archives")
        print("  current    - Show current deployment status")
        print("  structure  - Show server directory structure")
        print("  clean      - Clean old archives (keep 5 most recent)")
        print("  fresh      - Prepare fresh deployment environment")
        print()
        print("Examples:")
        print("  python server_manager.py list")
        print("  python server_manager.py current")
        return
    
    command = sys.argv[1].lower()
    
    if command == 'list':
        list_deployment_archives()
    elif command == 'current':
        show_current_deployment()
    elif command == 'structure':
        show_server_structure()
    elif command == 'clean':
        clean_old_archives()
    elif command == 'fresh':
        prepare_fresh_deployment()
    else:
        print(f"❌ Unknown command: {command}")
        print("Run without arguments to see available commands.")

if __name__ == "__main__":
    main()