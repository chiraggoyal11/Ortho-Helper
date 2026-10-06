"""
Verification script to check Grad-CAM integration across all models
Ensures all function calls have proper parameters
"""

import os
import re
import json

def check_gradcam_calls():
    """Check all generate_gradcam_heatmap calls in app.py"""
    print("=" * 80)
    print("GRAD-CAM INTEGRATION VERIFICATION")
    print("=" * 80)
    
    # Read app.py
    with open('app.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all generate_gradcam_heatmap calls
    pattern = r'generate_gradcam_heatmap\((.*?)\)'
    matches = re.finditer(pattern, content, re.DOTALL)
    
    issues_found = []
    calls_verified = []
    
    for i, match in enumerate(matches, 1):
        call_text = match.group(1)
        line_num = content[:match.start()].count('\n') + 1
        
        # Check if intensity parameter is present
        has_intensity = 'intensity=' in call_text
        
        # Determine model type from context
        context_before = content[max(0, match.start()-500):match.start()]
        
        model_type = "Unknown"
        if "Bone Fracture" in context_before or "bone_fracture" in context_before:
            model_type = "Bone Fracture"
        elif "Pneumonia" in context_before:
            model_type = "Pneumonia"
        elif "Cardiomegaly" in context_before:
            model_type = "Cardiomegaly"
        elif "Arthritis" in context_before:
            model_type = "Arthritis"
        elif "Osteoporosis" in context_before:
            model_type = "Osteoporosis"
        
        call_info = {
            'call_number': i,
            'line': line_num,
            'model_type': model_type,
            'has_intensity': has_intensity,
            'status': '✅' if has_intensity else '❌'
        }
        
        if has_intensity:
            calls_verified.append(call_info)
        else:
            issues_found.append(call_info)
        
        print(f"\nCall #{i} (Line {line_num}):")
        print(f"  Model Type: {model_type}")
        print(f"  Has intensity parameter: {call_info['status']}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total Grad-CAM calls found: {len(calls_verified) + len(issues_found)}")
    print(f"✅ Calls with intensity parameter: {len(calls_verified)}")
    print(f"❌ Calls missing intensity parameter: {len(issues_found)}")
    
    if issues_found:
        print("\n⚠️ ISSUES FOUND:")
        for issue in issues_found:
            print(f"  - Line {issue['line']}: {issue['model_type']} - Missing intensity parameter")
        return False
    else:
        print("\n✅ ALL GRAD-CAM CALLS ARE PROPERLY CONFIGURED!")
        print("\nVerified models:")
        for call in calls_verified:
            print(f"  ✓ {call['model_type']} (Line {call['line']})")
        return True

def check_function_signature():
    """Verify the generate_gradcam_heatmap function signature"""
    print("\n" + "=" * 80)
    print("FUNCTION SIGNATURE VERIFICATION")
    print("=" * 80)
    
    with open('utils/gradcam.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find function definition
    pattern = r'def generate_gradcam_heatmap\((.*?)\):'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        signature = match.group(1)
        print("\nFunction signature found:")
        print(f"  def generate_gradcam_heatmap({signature.strip()}):")
        
        # Check for intensity parameter
        has_intensity = 'intensity' in signature
        print(f"\n  Has intensity parameter: {'✅ Yes' if has_intensity else '❌ No'}")
        
        # Extract default value
        if has_intensity:
            intensity_match = re.search(r'intensity:\s*float\s*=\s*([\d.]+)', signature)
            if intensity_match:
                default_value = intensity_match.group(1)
                print(f"  Default intensity value: {default_value}")
        
        return has_intensity
    else:
        print("❌ Could not find function definition!")
        return False

def check_model_registry():
    """Check which models are active"""
    print("\n" + "=" * 80)
    print("ACTIVE MODELS CHECK")
    print("=" * 80)
    
    try:
        with open('models/registry/model_registry.json', 'r') as f:
            registry = json.load(f)
        
        print("\nActive models in registry:")
        for condition, model_info in registry.items():
            if model_info.get('active_model'):
                model_name = model_info['active_model'].get('name', 'Unknown')
                model_type = model_info['active_model'].get('type', 'Unknown')
                print(f"  ✓ {condition.title()}: {model_name} ({model_type})")
        
        return True
    except Exception as e:
        print(f"❌ Error reading model registry: {e}")
        return False

def main():
    """Run all verification checks"""
    print("\n🔍 Starting Grad-CAM Integration Verification...\n")
    
    # Run checks
    signature_ok = check_function_signature()
    calls_ok = check_gradcam_calls()
    registry_ok = check_model_registry()
    
    # Final report
    print("\n" + "=" * 80)
    print("FINAL VERIFICATION REPORT")
    print("=" * 80)
    
    all_checks = [
        ("Function Signature", signature_ok),
        ("Grad-CAM Calls", calls_ok),
        ("Model Registry", registry_ok)
    ]
    
    for check_name, status in all_checks:
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {check_name}: {'PASSED' if status else 'FAILED'}")
    
    all_passed = all(status for _, status in all_checks)
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL VERIFICATION CHECKS PASSED!")
        print("🎉 Grad-CAM is properly integrated for all models!")
    else:
        print("⚠️ SOME CHECKS FAILED - Please review the issues above")
    print("=" * 80)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
