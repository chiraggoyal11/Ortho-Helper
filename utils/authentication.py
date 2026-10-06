# Authentication Module for Medical X-ray AI System

import streamlit as st
import hashlib
import json
import os
from datetime import datetime, timedelta

# File to store user data
USER_DATA_FILE = "user_data.json"

def load_users():
    """Load users from file or create demo users if file doesn't exist"""
    # Demo user credentials - Admin account (you)
    demo_users = {
        "admin": {
            "password": "admin2025",
            "role": "doctor",
            "full_name": "System Administrator",
            "email": "admin@medicalai.com",
            "is_admin": True
        },
        "doctor": {
            "password": "medical123",
            "role": "doctor",
            "full_name": "Dr. Medical Professional",
            "email": "doctor@hospital.com",
            "created_by": "admin"
        },
        "student": {
            "password": "learn123",
            "role": "student", 
            "full_name": "Medical Student",
            "email": "student@university.edu",
            "created_by": "self_registration"
        },
        "patient": {
            "password": "health123",
            "role": "patient",
            "full_name": "Test Patient",
            "email": "patient@hospital.com",
            "created_by": "self_registration"
        }
    }
    
    if os.path.exists(USER_DATA_FILE):
        try:
            with open(USER_DATA_FILE, 'r') as f:
                loaded_users = json.load(f)
                # Merge with demo users to ensure they always exist
                demo_users.update(loaded_users)
                return demo_users
        except:
            return demo_users
    
    return demo_users

def save_users(users):
    """Save users to file"""
    try:
        with open(USER_DATA_FILE, 'w') as f:
            json.dump(users, f, indent=2)
        return True
    except:
        return False

def register_user(username, password, role="student", admin_code=None):
    """
    Register a new user
    
    Args:
        username (str): Username
        password (str): Plain text password
        role (str): User role - only "student" allowed for public registration
        admin_code (str): Admin code required for doctor/radiologist accounts
        
    Returns:
        tuple: (success: bool, message: str)
    """
    if not username or not password:
        return False, "Username and password are required"
    
    users = load_users()
    
    # Check if username already exists
    if username in users:
        return False, "Username already exists"
    
    # Restrict role registration - only students and patients can register publicly
    if role in ["doctor", "radiologist"]:
        # Admin code required for medical professional accounts
        ADMIN_CODE = "MEDAI2025ADMIN"  # You can change this
        if admin_code != ADMIN_CODE:
            return False, "Doctor/Radiologist accounts require admin authorization. Contact system administrator."
    
    # Force role to student/patient for public registration
    if role not in ["student", "patient", "doctor", "radiologist"]:
        role = "student"
    
    # Add new user
    users[username] = {
        "password": password,
        "role": role,
        "full_name": f"{role.title()} User",
        "email": f"{username}@hospital.com",
        "created_at": datetime.now().isoformat(),
        "created_by": "admin" if role in ["doctor", "radiologist"] else "self_registration"
    }
    
    # Save users
    success = save_users(users)
    if success:
        return True, f"Account created successfully as {role}"
    else:
        return False, "Failed to create account. Please try again."

def authenticate_user(username, password):
    """
    Authenticate user credentials
    
    Args:
        username (str): Username
        password (str): Plain text password
        
    Returns:
        dict: User info if authentication successful, None otherwise
    """
    if not username or not password:
        return None
    
    users = load_users()
    
    # Check credentials
    if username in users and users[username]["password"] == password:
        user_info = users[username].copy()
        user_info["username"] = username
        
        # Store in session state
        st.session_state.user_info = user_info
        st.session_state.login_time = datetime.now()
        
        return user_info
    
    return None

def get_user_info():
    """Get current user information"""
    if hasattr(st.session_state, 'user_info'):
        return st.session_state.user_info
    return None

def is_authenticated():
    """Check if user is authenticated"""
    return getattr(st.session_state, 'authenticated', False)

def logout_user():
    """Logout current user"""
    # Clear session state
    for key in ['authenticated', 'username', 'user_info', 'login_time']:
        if key in st.session_state:
            del st.session_state[key]

def check_session_timeout(timeout_minutes=60):
    """
    Check if user session has timed out
    
    Args:
        timeout_minutes (int): Session timeout in minutes
        
    Returns:
        bool: True if session is valid, False if timed out
    """
    if not is_authenticated():
        return False
    
    if 'login_time' not in st.session_state:
        return False
    
    login_time = st.session_state.login_time
    current_time = datetime.now()
    
    # Check if session has expired
    if current_time - login_time > timedelta(minutes=timeout_minutes):
        logout_user()
        return False
    
    return True

def get_user_permissions(username=None):
    """
    Get user permissions based on role
    
    Args:
        username (str): Username (optional, uses current user if not provided)
        
    Returns:
        dict: User permissions
    """
    if username is None:
        user_info = get_user_info()
        if not user_info:
            return {}
        role = user_info.get('role', 'student')
    else:
        users = load_users()
        if username not in users:
            return {}
        role = users[username]['role']
    
    # Define permissions based on role
    permissions = {
        'doctor': {
            'can_view_all_results': True,
            'can_export_reports': True,
            'can_provide_feedback': True,
            'can_access_advanced_features': True,
            'can_batch_process': True,
            'max_daily_uploads': 100
        },
        'radiologist': {
            'can_view_all_results': True,
            'can_export_reports': True,
            'can_provide_feedback': True,
            'can_access_advanced_features': True,
            'can_batch_process': True,
            'max_daily_uploads': 200
        },
        'student': {
            'can_view_all_results': True,
            'can_export_reports': True,
            'can_provide_feedback': False,
            'can_access_advanced_features': False,
            'can_batch_process': False,
            'max_daily_uploads': 20
        },
        'patient': {
            'can_view_all_results': True,
            'can_export_reports': True,
            'can_provide_feedback': True,
            'can_access_advanced_features': False,
            'can_batch_process': False,
            'max_daily_uploads': 5
        }
    }
    
    return permissions.get(role, permissions['student'])

def create_new_user_admin(username, password, role, full_name, email, created_by_admin=None):
    """
    Create a new user (admin function for creating doctor/radiologist accounts)
    
    Args:
        username (str): Username
        password (str): Plain text password
        role (str): User role
        full_name (str): Full name
        email (str): Email address
        created_by_admin (str): Admin username creating this account
        
    Returns:
        tuple: (success: bool, message: str)
    """
    users = load_users()
    
    if username in users:
        return False, "Username already exists"
    
    # Validate inputs
    if len(password) < 6:
        return False, "Password must be at least 6 characters"
    
    if role not in ['doctor', 'student', 'radiologist', 'patient']:
        return False, "Invalid role"
    
    # Create user
    users[username] = {
        "password": password,
        "role": role,
        "full_name": full_name,
        "email": email,
        "created_at": datetime.now().isoformat(),
        "created_by": created_by_admin or "admin"
    }
    
    # Save users
    success = save_users(users)
    if success:
        return True, f"{role.title()} account created successfully"
    else:
        return False, "Failed to create account"

def is_admin_user(username=None):
    """
    Check if user is an admin (can create doctor accounts)
    
    Args:
        username (str): Username to check, uses current user if None
        
    Returns:
        bool: True if user is admin
    """
    if username is None:
        user_info = get_user_info()
        if not user_info:
            return False
        username = user_info.get('username')
    
    users = load_users()
    if username not in users:
        return False
    
    # Admin user or has admin flag
    return (username == 'admin' or 
            users[username].get('is_admin', False) or 
            users[username].get('role') == 'doctor')

def validate_password_strength(password):
    """
    Validate password strength
    
    Args:
        password (str): Password to validate
        
    Returns:
        tuple: (is_valid, message)
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter"
    
    if not any(c.islower() for c in password):
        return False, "Password must contain at least one lowercase letter"
    
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one number"
    
    return True, "Password is strong"

def log_user_activity(activity_type, details=None):
    """
    Log user activity (for future audit trail)
    
    Args:
        activity_type (str): Type of activity
        details (dict): Additional details
    """
    user_info = get_user_info()
    if not user_info:
        return
    
    activity_log = {
        'timestamp': datetime.now().isoformat(),
        'username': user_info['username'],
        'role': user_info['role'],
        'activity_type': activity_type,
        'details': details or {}
    }
    
    # In production, save to database or log file
    # For now, just store in session state
    if 'activity_log' not in st.session_state:
        st.session_state.activity_log = []
    
    st.session_state.activity_log.append(activity_log)

# Security utilities
def sanitize_input(user_input):
    """Sanitize user input to prevent injection attacks"""
    if not isinstance(user_input, str):
        return str(user_input)
    
    # Remove potentially dangerous characters
    dangerous_chars = ['<', '>', '"', "'", '&', ';', '(', ')', '{', '}', '[', ']']
    for char in dangerous_chars:
        user_input = user_input.replace(char, '')
    
    return user_input.strip()

def generate_session_token():
    """Generate a secure session token"""
    import secrets
    return secrets.token_urlsafe(32)

# Example usage and testing
if __name__ == "__main__":
    # Test authentication
    print("Testing authentication system...")
    
    # Test valid credentials
    print(f"Doctor login: {authenticate_user('doctor', 'medical123')}")
    print(f"Student login: {authenticate_user('student', 'learn123')}")
    
    # Test invalid credentials
    print(f"Invalid login: {authenticate_user('doctor', 'wrong_password')}")
    
    # Test password validation
    print(f"Weak password: {validate_password_strength('123')}")
    print(f"Strong password: {validate_password_strength('StrongPass123')}")