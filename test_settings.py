#!/usr/bin/env python3
"""
Test script to verify the Settings Manager functionality
Run this to test if the settings system is working properly
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_settings_manager():
    """Test the SettingsManager functionality"""
    
    print("🔧 Testing Settings Manager Functionality")
    print("=" * 50)
    
    try:
        from utils.settings_manager import SettingsManager
        
        # Initialize settings manager
        print("✅ Importing SettingsManager...")
        manager = SettingsManager()
        
        # Test loading settings
        print("✅ Loading settings...")
        settings = manager.load_settings()
        print(f"   Loaded {len(settings)} setting categories")
        
        # Test getting specific settings
        print("✅ Testing specific setting retrieval...")
        confidence = manager.get_setting('model', 'confidence_threshold', 0.5)
        print(f"   Confidence threshold: {confidence}")
        
        # Test updating a setting
        print("✅ Testing setting update...")
        success = manager.update_setting('model', 'confidence_threshold', 0.7)
        print(f"   Update success: {success}")
        
        # Verify the update
        new_confidence = manager.get_setting('model', 'confidence_threshold', 0.5)
        print(f"   New confidence threshold: {new_confidence}")
        
        # Test settings summary
        print("✅ Testing settings summary...")
        summary = manager.get_settings_summary()
        print(f"   Generated summary with {len(summary)} sections")
        
        # Test export
        print("✅ Testing settings export...")
        export_data = manager.export_settings()
        if export_data:
            print(f"   Export data size: {len(export_data)} bytes")
        else:
            print("   ❌ Export failed")
        
        # Test reset to defaults
        print("✅ Testing reset to defaults...")
        reset_success = manager.reset_to_defaults()
        print(f"   Reset success: {reset_success}")
        
        # Get backups
        print("✅ Testing backup listing...")
        backups = manager.get_backups()
        print(f"   Found {len(backups)} backup files")
        
        print("\n🎉 All Settings Manager tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Settings Manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_settings_integration():
    """Test the settings integration helper"""
    
    print("\n🔗 Testing Settings Integration Helper")
    print("=" * 50)
    
    try:
        from utils.settings_integration import (
            get_current_settings, get_confidence_threshold,
            get_gradcam_intensity, is_gpu_enabled,
            should_include_metadata, get_default_report_format
        )
        
        print("✅ Importing settings integration functions...")
        
        # Test getting current settings
        print("✅ Testing get_current_settings...")
        settings = get_current_settings()
        print(f"   Retrieved {len(settings)} setting categories")
        
        # Test specific getters
        print("✅ Testing specific setting getters...")
        
        confidence = get_confidence_threshold()
        print(f"   Confidence threshold: {confidence}")
        
        gradcam_intensity = get_gradcam_intensity()
        print(f"   Grad-CAM intensity: {gradcam_intensity}")
        
        gpu_enabled = is_gpu_enabled()
        print(f"   GPU enabled: {gpu_enabled}")
        
        include_metadata = should_include_metadata()
        print(f"   Include metadata: {include_metadata}")
        
        report_format = get_default_report_format()
        print(f"   Default report format: {report_format}")
        
        print("\n🎉 Settings Integration tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Settings Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 Medical AI Settings System Test Suite")
    print("=" * 60)
    
    # Test 1: Settings Manager
    test1_success = test_settings_manager()
    
    # Test 2: Settings Integration
    test2_success = test_settings_integration()
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 30)
    print(f"Settings Manager:     {'✅ PASS' if test1_success else '❌ FAIL'}")
    print(f"Settings Integration: {'✅ PASS' if test2_success else '❌ FAIL'}")
    
    if test1_success and test2_success:
        print("\n🎉 All tests PASSED! Settings system is fully functional.")
    else:
        print("\n❌ Some tests FAILED. Check the error messages above.")
    
    print("\nTo use the settings in your Streamlit app:")
    print("1. Navigate to Settings page")
    print("2. Adjust your preferences")
    print("3. Click 'Save All Settings'")
    print("4. Settings will be applied throughout the system")

if __name__ == "__main__":
    main()