#!/usr/bin/env python3
"""
Test script to check the cleanup_orphaned_files functionality
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from utils.model_manager import ModelManager
    
    print("=== Testing Model Manager Cleanup Function ===\n")
    
    # Create a ModelManager instance
    manager = ModelManager()
    print(f"Models directory: {manager.models_dir}")
    print(f"Registry directory: {manager.registry_dir}")
    
    # Load registry
    registry = manager._load_registry()
    print(f"\nModels in registry: {len(registry['models'])}")
    
    # Get registered files
    registry_files = set()
    for model_id, model_info in registry['models'].items():
        file_path = Path(model_info['file_path'])
        if not file_path.is_absolute():
            file_path = manager.models_dir / file_path
        registry_files.add(str(file_path))
    
    print(f"Registry files ({len(registry_files)}):")
    for f in sorted(registry_files):
        exists = "✓" if Path(f).exists() else "✗"
        print(f"  {exists} {Path(f).name}")
    
    # Get actual .h5 files
    actual_files = set(str(f) for f in manager.models_dir.glob('*.h5'))
    print(f"\nActual .h5 files in models dir ({len(actual_files)}):")
    for f in sorted(actual_files):
        print(f"  {Path(f).name}")
    
    # Find orphans
    orphaned = actual_files - registry_files
    print(f"\nOrphaned files ({len(orphaned)}):")
    for orphan in sorted(orphaned):
        print(f"  {Path(orphan).name}")
    
    # Test the cleanup function (ACTUALLY RUN IT)
    if orphaned:
        print(f"\n=== Running cleanup_orphaned_files ===")
        print("This will DELETE the orphaned files!")
        
        # Uncomment the next lines to actually run cleanup
        # removed_count, removed_files = manager.cleanup_orphaned_files()
        # print(f"Cleanup result: {removed_count} files removed")
        # for file in removed_files:
        #     print(f"  Removed: {Path(file).name}")
        print("(Cleanup disabled in test - uncomment to actually remove files)")
    else:
        print("\n✅ No orphaned files found - cleanup not needed!")
        
except Exception as e:
    print(f"❌ Error testing cleanup function: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Test Complete ===")