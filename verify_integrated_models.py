"""
Verify all integrated models are working correctly
"""

import tensorflow as tf
from pathlib import Path
import json

def verify_integrated_models():
    print("="*80)
    print("🔍 VERIFYING INTEGRATED MODELS")
    print("="*80)
    
    models_folder = Path("models")
    
    conditions = ['pneumonia', 'cardiomegaly', 'arthritis', 'osteoporosis', 'bone_fracture']
    
    results = []
    
    for condition in conditions:
        print(f"\n{'='*80}")
        print(f"Testing {condition.upper()}")
        print(f"{'='*80}")
        
        condition_folder = models_folder / condition
        
        # Check model_info.json
        info_path = condition_folder / "model_info.json"
        if info_path.exists():
            with open(info_path, 'r') as f:
                model_info = json.load(f)
            print(f"✅ model_info.json found")
            print(f"   Display Name: {model_info.get('display_name', 'N/A')}")
            print(f"   Accuracy: {model_info['performance']['accuracy']:.1%}")
            print(f"   Grade: {model_info.get('grade', 'N/A')}")
            
            # Try loading the model
            model_file = condition_folder / model_info['model_file']
            if model_file.exists():
                print(f"✅ Model file exists: {model_file.name}")
                
                try:
                    model = tf.keras.models.load_model(str(model_file), compile=False)
                    print(f"✅ Model loads successfully!")
                    print(f"   Input shape: {model.input_shape}")
                    print(f"   Output shape: {model.output_shape}")
                    print(f"   Parameters: {model.count_params():,}")
                    
                    # Compile for inference
                    model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                        loss='binary_crossentropy',
                        metrics=['accuracy']
                    )
                    print(f"✅ Model compiled successfully!")
                    
                    results.append({
                        'condition': condition,
                        'status': 'SUCCESS',
                        'accuracy': model_info['performance']['accuracy']
                    })
                    
                except Exception as e:
                    print(f"❌ Model loading FAILED: {str(e)[:150]}")
                    results.append({
                        'condition': condition,
                        'status': 'FAILED',
                        'error': str(e)[:100]
                    })
            else:
                print(f"❌ Model file NOT found: {model_info['model_file']}")
                results.append({
                    'condition': condition,
                    'status': 'FILE_NOT_FOUND'
                })
        else:
            print(f"❌ model_info.json NOT found")
            results.append({
                'condition': condition,
                'status': 'INFO_MISSING'
            })
    
    # Summary
    print(f"\n{'='*80}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*80}")
    
    success_count = sum(1 for r in results if r.get('status') == 'SUCCESS')
    
    print(f"\n📊 Results:")
    print(f"   Total Conditions: {len(conditions)}")
    print(f"   ✅ Working: {success_count}")
    print(f"   ❌ Failed: {len(conditions) - success_count}")
    
    print(f"\n📋 Detailed Status:")
    for r in results:
        status_icon = "✅" if r.get('status') == 'SUCCESS' else "❌"
        condition = r['condition'].replace('_', ' ').title()
        if r.get('status') == 'SUCCESS':
            print(f"   {status_icon} {condition}: {r.get('accuracy', 0):.1%} accuracy")
        else:
            print(f"   {status_icon} {condition}: {r.get('status', 'UNKNOWN')}")
    
    if success_count == len(conditions):
        print(f"\n{'='*80}")
        print("🎉 ALL MODELS ARE READY!")
        print(f"{'='*80}")
        print("\n✅ All 5 medical conditions are available")
        print("✅ All models load and compile successfully")
        print("✅ Ready for use in Streamlit application")
        print("\n🚀 You can now:")
        print("   1. Restart Streamlit (if running)")
        print("   2. Test classifications with X-ray images")
        print("   3. Use Model Management to activate models")
        print(f"{'='*80}")
    else:
        print(f"\n{'='*80}")
        print("⚠️  SOME MODELS HAVE ISSUES")
        print(f"{'='*80}")
        print("Please check the errors above and re-run integration if needed")
    
    return results

if __name__ == "__main__":
    try:
        results = verify_integrated_models()
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
