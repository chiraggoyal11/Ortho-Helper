#!/usr/bin/env python3
"""
Connection Troubleshooting Script for Medical X-ray AI System
"""

import subprocess
import requests
import socket
import time
from datetime import datetime

def check_port_available(port):
    """Check if a port is available"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        result = sock.connect_ex(('localhost', port))
        return result == 0

def test_streamlit_connection(port):
    """Test Streamlit connection"""
    try:
        response = requests.get(f'http://localhost:{port}', timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    print("🔍 Medical X-ray AI System - Connection Diagnostics")
    print("=" * 50)
    print(f"Timestamp: {datetime.now()}")
    print()
    
    # Test common ports
    ports_to_test = [8501, 8502, 8503, 8504, 8505]
    
    print("📊 Port Status Check:")
    for port in ports_to_test:
        is_listening = check_port_available(port)
        is_responding = test_streamlit_connection(port) if is_listening else False
        
        status = "✅ Active & Responding" if is_responding else "🔶 Listening" if is_listening else "❌ Not Active"
        print(f"  Port {port}: {status}")
        
        if is_responding:
            print(f"    🌐 Access at: http://localhost:{port}")
    
    print()
    print("🔧 Troubleshooting Steps:")
    print("1. Try accessing: http://localhost:8502")
    print("2. Clear browser cache and cookies")
    print("3. Try incognito/private browsing mode")
    print("4. Disable browser extensions")
    print("5. Check Windows Firewall settings")
    print("6. Try a different browser")
    
    print()
    print("📝 System Information:")
    try:
        # Check if processes are running
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'], 
                              capture_output=True, text=True, shell=True)
        python_processes = len([line for line in result.stdout.split('\n') if 'python.exe' in line])
        print(f"  Python processes running: {python_processes}")
    except:
        print("  Could not check Python processes")
    
    print()
    print("🎯 Quick Fixes:")
    print("• Run: streamlit run app.py --server.port 8503")
    print("• Or try: python -m streamlit run app.py --server.port 8503")
    print("• Access via: http://localhost:8503")

if __name__ == "__main__":
    main()