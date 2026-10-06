import utils.authentication as auth

# Test student account authentication
print("Testing Student Account Authentication...")
student_login = auth.authenticate_user("student", "learn123")
print(f"Student login: {'SUCCESS' if student_login else 'FAILED'}")

if student_login:
    print(f"Student info: {student_login['full_name']} - Role: {student_login['role']}")

# Test permissions
permissions = auth.get_user_permissions("student")
print(f"\nStudent Permissions:")
print(f"- Can view results: {permissions.get('can_view_all_results', False)}")
print(f"- Can export reports: {permissions.get('can_export_reports', False)}")
print(f"- Can provide feedback: {permissions.get('can_provide_feedback', False)}")
print(f"- Can access advanced features: {permissions.get('can_access_advanced_features', False)}")
print(f"- Can batch process: {permissions.get('can_batch_process', False)}")
print(f"- Max daily uploads: {permissions.get('max_daily_uploads', 0)}")

# Test admin access
is_admin = auth.is_admin_user("student")
print(f"\nAdmin status: {'YES' if is_admin else 'NO'}")

# Test registration
success, message = auth.register_user("test_student123", "password123", "student")
print(f"\nStudent registration test: {'SUCCESS' if success else 'FAILED'} - {message}")

success, message = auth.register_user("test_doctor123", "password123", "doctor")
print(f"Doctor registration test (should fail): {'SUCCESS' if success else 'FAILED'} - {message}")

print("\nTest completed!")