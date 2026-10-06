# Setup script for Medical X-ray AI Classification System

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("🔧 Installing required packages...")
    
    # Core packages (minimal installation for quick setup)
    core_packages = [
        "streamlit>=1.28.0",
        "tensorflow>=2.13.0", 
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.15.0"
    ]
    
    for package in core_packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing {package}: {e}")

def create_placeholder_models():
    """Create placeholder models if they don't exist"""
    print("🤖 Setting up AI models...")
    
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Import after tensorflow is installed
    try:
        import tensorflow as tf
        from tensorflow import keras
        
        model_configs = {
            'bone_fracture_model.h5': {
                'input_shape': (224, 224, 3),
                'classes': 1,  # Binary classification
                'activation': 'sigmoid'
            },
            'chest_conditions_model.h5': {
                'input_shape': (224, 224, 3), 
                'classes': 3,  # Multi-class
                'activation': 'softmax'
            },
            'knee_conditions_model.h5': {
                'input_shape': (224, 224, 3),
                'classes': 3,  # Multi-class
                'activation': 'softmax'
            }
        }
        
        for model_name, config in model_configs.items():
            model_path = os.path.join(models_dir, model_name)
            
            if not os.path.exists(model_path):
                print(f"Creating placeholder model: {model_name}")
                
                # Create simple CNN model
                model = keras.Sequential([
                    keras.layers.Input(shape=config['input_shape']),
                    keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
                    keras.layers.MaxPooling2D(2),
                    keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
                    keras.layers.MaxPooling2D(2),
                    keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
                    keras.layers.GlobalAveragePooling2D(),
                    keras.layers.Dense(64, activation='relu'),
                    keras.layers.Dropout(0.5),
                    keras.layers.Dense(config['classes'], activation=config['activation'])
                ])
                
                model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy' if config['classes'] == 1 else 'categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                model.save(model_path)
                print(f"✅ Placeholder model created: {model_path}")
        
        print("🎯 All models ready!")
        
    except ImportError:
        print("⚠️  TensorFlow not available. Models will be created when first needed.")

def setup_directories():
    """Ensure all necessary directories exist"""
    print("📁 Setting up directories...")
    
    directories = [
        "models",
        "utils", 
        "static",
        "temp"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")

def main():
    """Main setup function"""
    print("🏥 Medical X-ray AI Classification System Setup")
    print("=" * 50)
    
    # Setup directories
    setup_directories()
    
    # Install requirements
    install_requirements()
    
    # Create placeholder models
    create_placeholder_models()
    
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("\nTo run the application:")
    print('1. Make sure you are in the virtual environment')
    print('2. Run: streamlit run app.py')
    print('3. Open http://localhost:8501 in your browser')
    print("\nDemo login credentials:")
    print("👨‍⚕️ Doctor - username: doctor, password: medical123")
    print("🎓 Student - username: student, password: learn123") 

if __name__ == "__main__":
    main()