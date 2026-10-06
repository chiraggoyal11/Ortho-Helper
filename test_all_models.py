"""
Test script to verify all 5 medical models can load successfully
"""
import tensorflow as tf

models_to_test = {
    "Pneumonia": "models/pneumonia/densenet121_pneumonia_intensive_20251006_182328.keras",
    "Bone Fracture (Limb Abnormalities)": "models/bone_fracture/densenet121_limbabnormalities_intensive_20251006_190347.keras",
    "Cardiomegaly": "models/cardiomegaly/cardiomegaly_densenet121_intensive_20251006_192404.keras",
    "Arthritis (Osteoarthritis)": "models/arthritis/densenet121_osteoarthritis_intensive_20251006_185456.keras",
    "Osteoporosis": "models/osteoporosis/densenet121_osteoporosis_intensive_20251006_183913.keras"
}

print("=" * 80)
print("Testing All Medical Models")
print("=" * 80)

success_count = 0
fail_count = 0

for name, path in models_to_test.items():
    print(f"\n🔍 Testing {name}...")
    try:
        model = tf.keras.models.load_model(path, compile=False)
        print(f"   ✅ SUCCESS")
        print(f"   - Input shape: {model.input_shape}")
        print(f"   - Output shape: {model.output_shape}")
        success_count += 1
    except Exception as e:
        print(f"   ❌ FAILED: {str(e)[:100]}")
        fail_count += 1

print("\n" + "=" * 80)
print(f"Results: {success_count}/5 models loaded successfully")
if success_count == 5:
    print("✅ All models are working!")
else:
    print(f"❌ {fail_count} models failed to load")
print("=" * 80)
