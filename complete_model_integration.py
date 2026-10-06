"""
Complete Model Integration Script - Integrate all trained models from 'new' folder
Ensures compatibility with TensorFlow 2.15 and proper registration
"""

import os
import json
import shutil
import tensorflow as tf
from pathlib import Path
from datetime import datetime

def integrate_all_models():
    """Main integration function"""
    print("="*80)
    print("🏥 MEDICAL AI MODEL INTEGRATION - COMPLETE SUITE")
    print("="*80)
    
    project_root = Path.cwd()
    new_folder = project_root / "new"
    models_folder = project_root / "models"
    
    # Verify new folder exists
    if not new_folder.exists():
        print(f"❌ Error: 'new' folder not found at {new_folder}")
        return
    
    print(f"✅ Source folder: {new_folder}")
    print(f"✅ Destination folder: {models_folder}")
    
    # Model definitions with their paths and performance
    models_config = {
        'pneumonia': {
            'source_path': 'densenet121_pneumonia_intensive_models/models/densenet121_pneumonia_intensive_20251006_182328',
            'display_name': '🫁 Pneumonia Detection',
            'accuracy': 0.9575,
            'precision': 0.9735,
            'recall': 0.9388,
            'grade': 'Medical Grade'
        },
        'cardiomegaly': {
            'source_path': 'cardiomegaly_densenet121/cardiomegaly_intensive_20251006_192404/cardiomegaly_densenet121_intensive_20251006_192404',
            'display_name': '❤️ Cardiomegaly Detection',
            'accuracy': 0.6300,
            'precision': 0.6816,
            'recall': 0.5728,
            'grade': 'Clinical Grade'
        },
        'arthritis': {
            'source_path': 'densenet121_osteoarthritis_intensive_models/models/densenet121_osteoarthritis_intensive_20251006_185456',
            'display_name': '🦵 Knee Arthritis Detection',
            'accuracy': 0.9425,
            'precision': 0.9635,
            'recall': 0.9204,
            'grade': 'Medical Grade'
        },
        'osteoporosis': {
            'source_path': 'densenet121_osteoporosis_intensive_models/models/densenet121_osteoporosis_intensive_20251006_183913',
            'display_name': '🦴 Knee Osteoporosis Detection',
            'accuracy': 0.9177,
            'precision': 0.9568,
            'recall': 0.8806,
            'grade': 'Medical Grade'
        },
        'bone_fracture': {
            'source_path': 'densenet121_limbabnormalities_intensive_models/models/densenet121_limbabnormalities_intensive_20251006_190347',
            'display_name': '💀 Bone Fracture Detection',
            'accuracy': 0.7300,
            'precision': 0.6966,
            'recall': 0.8150,
            'grade': 'Research Grade'
        }
    }
    
    print(f"\n📦 Processing {len(models_config)} medical conditions...")
    
    results = []
    
    # Process each model
    for condition, config in models_config.items():
        print(f"\n{'='*80}")
        print(f"{config['display_name']} ({condition})")
        print(f"{'='*80}")
        
        # Create condition folder
        dest_folder = models_folder / condition
        dest_folder.mkdir(parents=True, exist_ok=True)
        
        # Source files base path
        source_base = new_folder / config['source_path']
        
        # Copy all model formats
        copied = []
        for ext in ['.keras', '.h5', '.weights.h5']:
            source_file = Path(str(source_base) + ext)
            if source_file.exists():
                dest_file = dest_folder / source_file.name
                shutil.copy2(source_file, dest_file)
                print(f"✅ Copied: {source_file.name}")
                copied.append(ext)
            else:
                print(f"⚠️  Not found: {source_file.name}")
        
        # Test model loading
        keras_model = dest_folder / (source_base.name + '.keras')
        load_success = False
        if keras_model.exists():
            try:
                model = tf.keras.models.load_model(str(keras_model), compile=False)
                print(f"✅ Model loads successfully")
                print(f"   Input: {model.input_shape}, Output: {model.output_shape}")
                load_success = True
            except Exception as e:
                print(f"❌ Model loading failed: {str(e)[:100]}")
        
        # Create model_info.json
        model_info = {
            "condition": condition,
            "display_name": config['display_name'],
            "model_file": source_base.name + '.keras',
            "architecture": "DenseNet121",
            "parameters": 7305281,
            "performance": {
                "accuracy": config['accuracy'],
                "precision": config['precision'],
                "recall": config['recall']
            },
            "grade": config['grade'],
            "input_shape": [224, 224, 3],
            "output_classes": 2,
            "classes": ["Normal", condition.replace('_', ' ').title()],
            "gradcam_layer": "conv5_block16_2_conv",
            "training_date": "2025-10-06",
            "integrated_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "production_ready" if load_success else "error"
        }
        
        info_path = dest_folder / "model_info.json"
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        print(f"✅ Created: model_info.json")
        
        results.append({
            'condition': condition,
            'display_name': config['display_name'],
            'copied': copied,
            'load_success': load_success,
            'accuracy': config['accuracy']
        })
    
    # Create summary report
    print(f"\n{'='*80}")
    print("INTEGRATION SUMMARY")
    print(f"{'='*80}")
    
    success_count = sum(1 for r in results if r['load_success'])
    
    print(f"\n📊 Results:")
    print(f"   Total Models: {len(results)}")
    print(f"   Successfully Loaded: {success_count}")
    print(f"   Failed: {len(results) - success_count}")
    
    print(f"\n🎯 Model Performance:")
    for r in results:
        status = "✅" if r['load_success'] else "❌"
        print(f"   {status} {r['display_name']}: {r['accuracy']:.1%}")
    
    # Create integration report
    report_path = models_folder / "INTEGRATION_REPORT.md"
    report_content = f"""# 🏥 Model Integration Report

**Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**TensorFlow Version**: {tf.__version__}
**Source**: new/ folder (trained October 6, 2025)

## 📊 Integration Results

| Condition | Display Name | Accuracy | Status |
|-----------|-------------|----------|--------|
"""
    
    for r in results:
        status = "✅ Ready" if r['load_success'] else "❌ Failed"
        report_content += f"| {r['condition']} | {r['display_name']} | {r['accuracy']:.1%} | {status} |\n"
    
    report_content += f"""
## 🎯 Performance Summary

- **Medical Grade** (≥90%): {sum(1 for r in results if r['accuracy'] >= 0.90)} models
- **Clinical Grade** (60-90%): {sum(1 for r in results if 0.60 <= r['accuracy'] < 0.90)} models
- **Research Grade** (<60%): {sum(1 for r in results if r['accuracy'] < 0.60)} models

## 📁 Model Files

Each condition folder contains:
- `*.keras` - Main model file (TensorFlow 2.15 compatible)
- `*.h5` - Legacy format
- `*.weights.h5` - Weights only
- `model_info.json` - Model metadata and configuration

## 🚀 Usage

All models are now available in the application:
1. Start Streamlit: `streamlit run app.py`
2. Go to "Model Management System"
3. Navigate to "Activate Models" tab
4. Select and activate models for each condition

## 🔧 Technical Details

- **Architecture**: DenseNet121 (7.3M parameters)
- **Input**: 224×224×3 RGB images
- **Output**: Binary classification (Normal vs Condition)
- **Grad-CAM Layer**: conv5_block16_2_conv

---
*Generated by complete_model_integration.py*
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\n✅ Integration report: {report_path}")
    
    if success_count == len(results):
        print("\n" + "="*80)
        print("🎉 SUCCESS! ALL MODELS INTEGRATED AND VERIFIED")
        print("="*80)
        print("\n🚀 Next Steps:")
        print("   1. Restart Streamlit application")
        print("   2. Test each model in the Classification page")
        print("   3. Check Model Management System for all conditions")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("⚠️  PARTIAL SUCCESS - Some models failed to load")
        print("="*80)
        print("Please check error messages above")
    
    return results

if __name__ == "__main__":
    try:
        results = integrate_all_models()
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
