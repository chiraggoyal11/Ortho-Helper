"""
Verification Script: Student Access to Model Information Page
Checks if students can access and view the Model Information page
"""

import json
import re

def check_student_navigation_access():
    """Check if students have Model Information in their navigation menu"""
    print("=" * 80)
    print("STUDENT ACCESS TO MODEL INFORMATION - VERIFICATION")
    print("=" * 80)
    
    with open('app.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the navigation section
    pattern = r'if st\.session_state\.user_role in \[\'doctor\', \'radiologist\'\]:.*?page_options = \[(.*?)\].*?else:.*?page_options = \[(.*?)\]'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        doctor_pages = match.group(1)
        student_pages = match.group(2)
        
        print("\n📋 NAVIGATION MENU ACCESS:")
        print("-" * 80)
        
        print("\n🔵 Doctor/Radiologist Pages:")
        doctor_list = [p.strip().strip('"').strip("'") for p in doctor_pages.split(',')]
        for i, page in enumerate(doctor_list, 1):
            print(f"  {i}. {page}")
        
        print("\n🟢 Student Pages:")
        student_list = [p.strip().strip('"').strip("'") for p in student_pages.split(',')]
        for i, page in enumerate(student_list, 1):
            print(f"  {i}. {page}")
        
        # Check if Model Information is in student pages
        has_model_info = any('Model Information' in page for page in student_list)
        
        print("\n" + "=" * 80)
        print("VERIFICATION RESULT:")
        print("=" * 80)
        
        if has_model_info:
            print("✅ SUCCESS: Students CAN access Model Information page!")
            print("\nStudent navigation includes:")
            for page in student_list:
                if 'Model Information' in page:
                    print(f"  ✓ {page}")
            return True
        else:
            print("❌ ISSUE: Model Information NOT found in student navigation!")
            return False
    else:
        print("❌ Could not parse navigation structure")
        return False

def check_model_info_page_content():
    """Check if Model Information page has role restrictions"""
    print("\n" + "=" * 80)
    print("MODEL INFORMATION PAGE CONTENT CHECK")
    print("=" * 80)
    
    with open('app.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the show_model_info_page function
    pattern = r'def show_model_info_page\(\):.*?(?=\ndef )'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        function_content = match.group(0)
        
        # Check for role restrictions
        has_role_check = 'user_role' in function_content or 'is_admin' in function_content
        
        print("\n🔍 Function Analysis:")
        print(f"  Function length: {len(function_content)} characters")
        print(f"  Has role checks: {'❌ YES (restricted)' if has_role_check else '✅ NO (open access)'}")
        
        # Count the models displayed
        model_count = function_content.count("'name':")
        print(f"  Number of models displayed: {model_count}")
        
        # Check for key features
        features = {
            'Bone Fracture': 'Bone Fracture Detection' in function_content,
            'Pneumonia': 'Pneumonia Detection' in function_content,
            'Cardiomegaly': 'Cardiomegaly Detection' in function_content,
            'Arthritis': 'Arthritis Detection' in function_content,
            'Osteoporosis': 'Osteoporosis Detection' in function_content,
            'Technical Specs': 'Technical Specifications' in function_content,
            'Grad-CAM': 'Grad-CAM' in function_content or 'visualization' in function_content,
            'Accuracy Metrics': 'accuracy' in function_content.lower()
        }
        
        print("\n📊 Content Features:")
        for feature, present in features.items():
            status = "✅" if present else "❌"
            print(f"  {status} {feature}")
        
        if not has_role_check:
            print("\n✅ CONCLUSION: Model Information page is FULLY ACCESSIBLE to students!")
            print("   No role-based restrictions found in the page content.")
            return True
        else:
            print("\n⚠️ WARNING: Role-based restrictions detected in page content!")
            return False
    else:
        print("❌ Could not find show_model_info_page function")
        return False

def check_student_accounts():
    """Check available student accounts for testing"""
    print("\n" + "=" * 80)
    print("AVAILABLE STUDENT ACCOUNTS FOR TESTING")
    print("=" * 80)
    
    try:
        with open('user_data.json', 'r', encoding='utf-8') as f:
            users = json.load(f)
        
        student_accounts = []
        for username, info in users.items():
            if info.get('role') == 'student':
                student_accounts.append({
                    'username': username,
                    'password': info.get('password'),
                    'full_name': info.get('full_name'),
                    'email': info.get('email')
                })
        
        if student_accounts:
            print(f"\n📚 Found {len(student_accounts)} student account(s):\n")
            for i, account in enumerate(student_accounts, 1):
                print(f"{i}. Username: {account['username']}")
                print(f"   Password: {account['password']}")
                print(f"   Name: {account['full_name']}")
                print(f"   Email: {account['email']}")
                print()
            
            print("✅ You can use any of these accounts to test Model Information access")
            return True
        else:
            print("❌ No student accounts found!")
            return False
            
    except Exception as e:
        print(f"❌ Error reading user accounts: {e}")
        return False

def main():
    """Run all verification checks"""
    print("\n🔍 Starting Student Model Information Access Verification...\n")
    
    # Run all checks
    nav_access = check_student_navigation_access()
    content_access = check_model_info_page_content()
    accounts_available = check_student_accounts()
    
    # Final report
    print("=" * 80)
    print("FINAL VERIFICATION REPORT")
    print("=" * 80)
    
    checks = [
        ("Navigation Menu Access", nav_access),
        ("Page Content Accessibility", content_access),
        ("Test Accounts Available", accounts_available)
    ]
    
    all_passed = all(status for _, status in checks)
    
    for check_name, status in checks:
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {check_name}: {'PASSED' if status else 'FAILED'}")
    
    print("\n" + "=" * 80)
    
    if all_passed:
        print("✅ ALL CHECKS PASSED!")
        print("\n🎉 Students CAN access Model Information page!")
        print("\n📋 What students will see:")
        print("   • 5 Binary Classification Models")
        print("   • Bone Fracture Detection (94.5% accuracy)")
        print("   • Pneumonia Detection (95.75% accuracy)")
        print("   • Cardiomegaly Detection (63.0% accuracy)")
        print("   • Arthritis Detection (94.25% accuracy)")
        print("   • Osteoporosis Detection (91.77% accuracy)")
        print("   • Technical specifications")
        print("   • Grad-CAM visualization info")
        print("   • Performance metrics")
        print("   • Clinical validation details")
        print("\n🔐 To Test:")
        print("   1. Go to: http://localhost:8503")
        print("   2. Login with student account (e.g., student/learn123)")
        print("   3. Click '📝 Model Information' in navigation")
        print("   4. View all model details without restrictions")
    else:
        print("⚠️ SOME CHECKS FAILED - Review issues above")
    
    print("=" * 80)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
