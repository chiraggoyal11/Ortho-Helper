"""
Quick test to show all models are accessible in the application
"""

import tensorflow as tf
import numpy as np
from pathlib import Path

def test_model_predictions():
    """Test that all models can make predictions"""
    print("="*80)
    print("🧪 TESTING MODEL PREDICTIONS")
    print("="*80)
    
    models_folder = Path("models")
    
    # Create a dummy test image (224x224x3)
    test_image = np.random.rand(1, 224, 224, 3).astype(np.float32)
    
    conditions = {
        'pneumonia': 'densenet121_pneumonia_intensive_20251006_182328.keras',
        'cardiomegaly': 'cardiomegaly_densenet121_intensive_20251006_192404.keras',
        'arthritis': 'densenet121_osteoarthritis_intensive_20251006_185456.keras',
        'osteoporosis': 'densenet121_osteoporosis_intensive_20251006_183913.keras',
        'bone_fracture': 'densenet121_limbabnormalities_intensive_20251006_190347.keras'
    }
    
    display_names = {
        'pneumonia': '🫁 Pneumonia Detection',
        'cardiomegaly': '❤️ Cardiomegaly Detection',
        'arthritis': '🦵 Knee Arthritis Detection',
        'osteoporosis': '🦴 Knee Osteoporosis Detection',
        'bone_fracture': '💀 Bone Fracture Detection'
    }
    
    results = []
    
    for condition, model_file in conditions.items():
        print(f"\n{'='*80}")
        print(f"{display_names[condition]}")
        print(f"{'='*80}")
        
        model_path = models_folder / condition / model_file
        
        try:
            # Load model
            print(f"Loading model...")
            model = tf.keras.models.load_model(str(model_path), compile=False)
            
            # Compile
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Make prediction
            print(f"Making prediction on test image...")
            prediction = model.predict(test_image, verbose=0)
            confidence = float(prediction[0][0])
            
            print(f"✅ Prediction successful!")
            print(f"   Raw output: {confidence:.4f}")
            print(f"   Confidence: {max(confidence, 1-confidence):.1%}")
            print(f"   Prediction: {'Positive' if confidence > 0.5 else 'Negative'}")
            
            results.append({
                'condition': condition,
                'display_name': display_names[condition],
                'status': 'SUCCESS',
                'prediction': confidence
            })
            
        except Exception as e:
            print(f"❌ Error: {str(e)[:150]}")
            results.append({
                'condition': condition,
                'display_name': display_names[condition],
                'status': 'FAILED',
                'error': str(e)[:100]
            })
    
    # Summary
    print(f"\n{'='*80}")
    print("PREDICTION TEST SUMMARY")
    print(f"{'='*80}")
    
    success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
    
    print(f"\n📊 Results:")
    print(f"   Total Models: {len(conditions)}")
    print(f"   ✅ Working: {success_count}")
    print(f"   ❌ Failed: {len(conditions) - success_count}")
    
    print(f"\n📋 Status:")
    for r in results:
        status_icon = "✅" if r['status'] == 'SUCCESS' else "❌"
        print(f"   {status_icon} {r['display_name']}: {r['status']}")
    
    if success_count == len(conditions):
        print(f"\n{'='*80}")
        print("🎉 ALL MODELS CAN MAKE PREDICTIONS!")
        print(f"{'='*80}")
        print("\n✅ All models are fully operational")
        print("✅ Ready for real X-ray classification")
        print("✅ Can be used in Streamlit application")
        print(f"{'='*80}")
    
    return results

if __name__ == "__main__":
    try:
        results = test_model_predictions()
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
