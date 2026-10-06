#!/usr/bin/env python3
"""
Complete Installation and Deployment Script
Medical X-ray AI Classification System - Advanced Feature Integration
"""

import subprocess
import sys
import os
from pathlib import Path
import json
import shutil
import zipfile
from datetime import datetime

class MedicalAIInstaller:
    """Complete installation and setup manager"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.requirements_file = self.project_root / "requirements.txt"
        self.config_dir = self.project_root / "medical_ai_config"
        self.models_dir = self.project_root / "models"
        self.utils_dir = self.project_root / "utils"
        
    def check_system_requirements(self):
        """Check system requirements and dependencies"""
        print("🔍 Checking System Requirements...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 8):
            print("❌ Python 3.8+ is required")
            return False
        else:
            print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Check available space
        total, used, free = shutil.disk_usage(self.project_root)
        free_gb = free // (1024**3)
        
        if free_gb < 5:
            print(f"⚠️ Warning: Only {free_gb}GB free space available. 5GB+ recommended.")
        else:
            print(f"✅ Sufficient disk space: {free_gb}GB available")
        
        return True
    
    def create_requirements_file(self):
        """Create or update requirements.txt with all necessary dependencies"""
        
        requirements = [
            # Core dependencies
            "streamlit>=1.28.0",
            "numpy>=1.21.0",
            "pandas>=1.3.0",
            "pillow>=8.3.0",
            
            # Machine Learning
            "tensorflow>=2.13.0",
            "scikit-learn>=1.0.0",
            "opencv-python>=4.5.0",
            
            # Visualization
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
            
            # Report Generation
            "reportlab>=3.6.0",
            "fpdf2>=2.5.0",
            
            # Image Processing
            "opencv-contrib-python>=4.5.0",
            "imageio>=2.9.0",
            "scikit-image>=0.18.0",
            
            # Data handling
            "joblib>=1.1.0",
            "h5py>=3.1.0",
            
            # Utilities
            "requests>=2.25.0",
            "python-dotenv>=0.19.0",
            
            # Optional GPU support
            "tensorflow-gpu>=2.13.0; sys_platform != 'darwin'",
            
            # Development tools
            "pytest>=6.2.0",
            "black>=21.0.0",
            "flake8>=3.9.0"
        ]
        
        print("📝 Creating requirements.txt...")
        
        with open(self.requirements_file, 'w') as f:
            for req in requirements:
                f.write(f"{req}\n")
        
        print(f"✅ Requirements file created: {self.requirements_file}")
    
    def install_dependencies(self):
        """Install all required dependencies"""
        print("📦 Installing dependencies...")
        
        try:
            # Upgrade pip first
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
            
            # Install requirements
            if self.requirements_file.exists():
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "-r", str(self.requirements_file)
                ])
            else:
                print("⚠️ Requirements file not found. Creating it first...")
                self.create_requirements_file()
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "-r", str(self.requirements_file)
                ])
            
            print("✅ Dependencies installed successfully!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    
    def setup_project_structure(self):
        """Create necessary directories and initialize configuration"""
        print("📁 Setting up project structure...")
        
        # Create directories
        directories = [
            self.config_dir,
            self.config_dir / "backups",
            self.config_dir / "presets", 
            self.config_dir / "exports",
            self.config_dir / "cache",
            self.config_dir / "logs",
            self.models_dir,
            self.project_root / "Dataset",
            self.project_root / "training_results",
            self.project_root / "analytics_data",
            self.project_root / "preprocessing_presets",
            self.project_root / "reports",
            self.project_root / "temp"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created: {directory}")
        
        # Create initial configuration
        self._create_initial_configuration()
        
        # Create sample data structure info
        self._create_dataset_info()
    
    def _create_initial_configuration(self):
        """Create initial system configuration"""
        config = {
            "system": {
                "version": "2.0.0",
                "installation_date": datetime.now().isoformat(),
                "features_enabled": {
                    "advanced_preprocessing": True,
                    "analytics_dashboard": True,
                    "model_management": True,
                    "experimental_features": True,
                    "configuration_persistence": True
                }
            },
            "model": {
                "confidence_threshold": 0.5,
                "batch_processing": False,
                "gradcam_intensity": 0.4,
                "auto_preprocessing": True,
                "enable_gpu": False,
                "cache_models": True
            },
            "reports": {
                "include_metadata": True,
                "include_preprocessing_info": True,
                "include_gradcam": True,
                "default_format": "PDF",
                "auto_download": False,
                "compress_reports": True
            },
            "advanced": {
                "preprocessing_presets": {},
                "custom_models": {},
                "experimental_features": True,
                "analytics_enabled": True
            }
        }
        
        config_file = self.config_dir / "system_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"✅ Initial configuration created: {config_file}")
    
    def _create_dataset_info(self):
        """Create dataset information file"""
        dataset_info = {
            "datasets": {
                "bone_fracture": {
                    "path": "Dataset/Bone_Fracture_Binary_Classification",
                    "type": "binary",
                    "classes": ["fractured", "normal"],
                    "description": "X-ray images for bone fracture detection"
                },
                "pneumonia": {
                    "path": "Dataset/chest_xray Pneumonia",
                    "type": "binary", 
                    "classes": ["pneumonia", "normal"],
                    "description": "Chest X-ray images for pneumonia detection"
                },
                "cardiomegaly": {
                    "path": "Dataset/cardiomelgy",
                    "type": "binary",
                    "classes": ["cardiomegaly", "normal"],
                    "description": "Chest X-ray images for cardiomegaly detection"
                },
                "arthritis": {
                    "path": "Dataset/Osteoarthritis Knee X-ray",
                    "type": "binary",
                    "classes": ["arthritis", "normal"],
                    "description": "Knee X-ray images for arthritis detection"
                },
                "osteoporosis": {
                    "path": "Dataset/Osteoporosis Knee",
                    "type": "binary",
                    "classes": ["osteoporosis", "normal"],
                    "description": "Knee X-ray images for osteoporosis detection"
                }
            },
            "preprocessing": {
                "default_size": [224, 224],
                "normalization": "0-1 scaling",
                "augmentation_options": [
                    "rotation", "horizontal_flip", "zoom", "brightness"
                ]
            }
        }
        
        dataset_info_file = self.project_root / "dataset_info.json"
        with open(dataset_info_file, 'w') as f:
            json.dump(dataset_info, f, indent=4)
        
        print(f"✅ Dataset information created: {dataset_info_file}")
    
    def verify_installation(self):
        """Verify that installation completed successfully"""
        print("🔍 Verifying installation...")
        
        verification_tests = [
            ("Streamlit", "streamlit", "--version"),
            ("TensorFlow", "python", "-c", "import tensorflow; print(f'TensorFlow {tensorflow.__version__}')"),
            ("OpenCV", "python", "-c", "import cv2; print(f'OpenCV {cv2.__version__}')"),
            ("Pandas", "python", "-c", "import pandas; print(f'Pandas {pandas.__version__}')"),
            ("Matplotlib", "python", "-c", "import matplotlib; print(f'Matplotlib {matplotlib.__version__}')")
        ]
        
        all_passed = True
        
        for test_name, *command in verification_tests:
            try:
                result = subprocess.run(command, capture_output=True, text=True, check=True)
                print(f"✅ {test_name}: {result.stdout.strip()}")
            except subprocess.CalledProcessError:
                print(f"❌ {test_name}: Installation verification failed")
                all_passed = False
            except Exception as e:
                print(f"⚠️ {test_name}: Could not verify - {e}")
        
        # Check file structure
        required_files = [
            "app.py",
            "utils/advanced_preprocessing.py",
            "utils/config_persistence.py",
            "utils/feature_completion.py",
            "utils/settings_manager.py",
            "utils/settings_integration.py"
        ]
        
        for file_path in required_files:
            file_full_path = self.project_root / file_path
            if file_full_path.exists():
                print(f"✅ File exists: {file_path}")
            else:
                print(f"❌ Missing file: {file_path}")
                all_passed = False
        
        return all_passed
    
    def create_startup_scripts(self):
        """Create startup scripts for different platforms"""
        print("📜 Creating startup scripts...")
        
        # Windows batch script
        batch_script = """@echo off
echo Starting Medical X-ray AI Classification System...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if Streamlit is available  
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo Error: Streamlit is not installed
    echo Please run: pip install -r requirements.txt
    pause
    exit /b 1
)

echo All dependencies are available!
echo Starting the application...
echo.
echo The application will open in your default web browser.
echo Close this window to stop the application.
echo.

streamlit run app.py --server.port 8502 --server.address localhost

pause
"""
        
        batch_file = self.project_root / "start_medical_ai_advanced.bat"
        with open(batch_file, 'w') as f:
            f.write(batch_script)
        
        print(f"✅ Windows script created: {batch_file}")
        
        # Unix shell script
        shell_script = """#!/bin/bash
echo "Starting Medical X-ray AI Classification System..."
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if Streamlit is available
if ! python3 -c "import streamlit" &> /dev/null; then
    echo "Error: Streamlit is not installed"
    echo "Please run: pip3 install -r requirements.txt"
    exit 1
fi

echo "All dependencies are available!"
echo "Starting the application..."
echo
echo "The application will open in your default web browser."
echo "Press Ctrl+C to stop the application."
echo

streamlit run app.py --server.port 8502 --server.address localhost
"""
        
        shell_file = self.project_root / "start_medical_ai_advanced.sh"
        with open(shell_file, 'w') as f:
            f.write(shell_script)
        
        # Make shell script executable
        try:
            import stat
            shell_file.chmod(stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
        except:
            pass
        
        print(f"✅ Unix script created: {shell_file}")
    
    def generate_installation_report(self):
        """Generate comprehensive installation report"""
        print("📋 Generating installation report...")
        
        report = {
            "installation_info": {
                "date": datetime.now().isoformat(),
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "platform": sys.platform,
                "project_path": str(self.project_root)
            },
            "features_installed": {
                "advanced_preprocessing": True,
                "configuration_persistence": True,
                "analytics_dashboard": True,
                "model_management": True,
                "experimental_features": True,
                "mobile_optimization": False,  # Future feature
                "cloud_integration": False    # Future feature
            },
            "files_created": [
                "utils/advanced_preprocessing.py",
                "utils/config_persistence.py", 
                "utils/feature_completion.py",
                "medical_ai_config/",
                "requirements.txt",
                "start_medical_ai_advanced.bat",
                "start_medical_ai_advanced.sh"
            ],
            "next_steps": [
                "1. Run 'start_medical_ai_advanced.bat' (Windows) or 'start_medical_ai_advanced.sh' (Unix)",
                "2. Navigate to Settings page to configure your preferences",
                "3. Upload medical images for classification",
                "4. Explore Advanced Features (available to medical professionals)",
                "5. Check Analytics dashboard for usage insights"
            ]
        }
        
        report_file = self.project_root / "installation_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=4)
        
        print(f"✅ Installation report created: {report_file}")
        return report
    
    def run_complete_installation(self):
        """Run complete installation process"""
        print("🚀 Medical X-ray AI System - Advanced Installation")
        print("=" * 60)
        
        steps = [
            ("System Requirements", self.check_system_requirements),
            ("Project Structure", self.setup_project_structure),
            ("Requirements File", self.create_requirements_file),
            ("Dependencies", self.install_dependencies),
            ("Startup Scripts", self.create_startup_scripts),
            ("Installation Verification", self.verify_installation),
            ("Installation Report", self.generate_installation_report)
        ]
        
        for step_name, step_function in steps:
            print(f"\n🔄 {step_name}...")
            try:
                result = step_function()
                if result is False:
                    print(f"❌ {step_name} failed!")
                    return False
                else:
                    print(f"✅ {step_name} completed!")
            except Exception as e:
                print(f"❌ {step_name} failed with error: {e}")
                return False
        
        print("\n" + "=" * 60)
        print("🎉 Installation completed successfully!")
        print("\n📋 Summary of new features:")
        print("  🔬 Advanced Preprocessing Controls")
        print("  💾 Enhanced Configuration Persistence") 
        print("  📊 Advanced Analytics Dashboard")
        print("  🔧 Model Management Tools")
        print("  🧪 Experimental Features")
        print("\n🚀 To start the application:")
        print("  Windows: Double-click 'start_medical_ai_advanced.bat'")
        print("  Unix/Mac: Run './start_medical_ai_advanced.sh'")
        print("  Manual: streamlit run app.py")
        
        return True

def main():
    """Main installation function"""
    installer = MedicalAIInstaller()
    success = installer.run_complete_installation()
    
    if success:
        print("\n✅ Ready to use the enhanced Medical X-ray AI Classification System!")
    else:
        print("\n❌ Installation encountered issues. Please check the errors above.")
    
    return success

if __name__ == "__main__":
    main()