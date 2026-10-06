"""
Test script to verify model loading with different methods
"""
import os
import tensorflow as tf

model_path = "models/pneumonia/densenet121_pneumonia_intensive_20251006_182328.keras"
h5_path = "models/pneumonia/densenet121_pneumonia_intensive_20251006_182328.h5"

print("=" * 80)
print("Testing Model Loading Methods")
print("=" * 80)

# Test 1: Load .keras file
print("\n1. Testing .keras file...")
try:
    model = tf.keras.models.load_model(model_path, compile=False)
    print(f"✓ SUCCESS: Loaded from .keras file")
    print(f"   Model type: {type(model)}")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
except Exception as e:
    print(f"✗ FAILED: {str(e)[:150]}")

# Test 2: Load .h5 file
print("\n2. Testing .h5 file...")
try:
    model = tf.keras.models.load_model(h5_path, compile=False)
    print(f"✓ SUCCESS: Loaded from .h5 file")
    print(f"   Model type: {type(model)}")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
except Exception as e:
    print(f"✗ FAILED: {str(e)[:150]}")

print("\n" + "=" * 80)
print("Test Complete")
print("=" * 80)
