"""
Test Grad-CAM Compatibility for All Active Models
Checks if each active model can generate Grad-CAM heatmaps correctly
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import sys

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

print("="*80)
print("GRAD-CAM COMPATIBILITY TEST FOR ALL ACTIVE MODELS")
print("="*80)
print()

# Load registry
registry_path = 'models/registry/model_registry.json'
if not os.path.exists(registry_path):
    print(f"❌ Registry not found: {registry_path}")
    sys.exit(1)

with open(registry_path, 'r', encoding='utf-8') as f:
    registry = json.load(f)

active_models = registry.get('active_models', {})
models_info = registry.get('models', {})

print(f"Found {len(active_models)} active model configurations")
print()

# Test results
results = {
    'total': 0,
    'compatible': 0,
    'incompatible': 0,
    'details': []
}

def find_target_layer(model):
    """Find the best layer for Grad-CAM - handles nested Sequential models"""
    conv_layers = []
    activation_layers = []
    
    # UPDATED: Check if this is a Sequential model with nested base architecture
    nested_base_model = None
    if len(model.layers) > 0:
        first_layer = model.layers[0]
        # Check if first layer is a Functional model (base architecture)
        if (hasattr(first_layer, 'layers') and 
            len(first_layer.layers) > 10 and  # Base models have many layers
            first_layer.__class__.__name__ in ['Functional', 'Model']):
            nested_base_model = first_layer
            layers_to_search = nested_base_model.layers
        else:
            layers_to_search = model.layers
    else:
        layers_to_search = model.layers
    
    for layer in layers_to_search:
        layer_name = layer.name.lower()
        layer_class = layer.__class__.__name__.lower()
        
        # Skip problematic layers
        if any(skip_type in layer_name for skip_type in ['dropout', 'batch', 'flatten', 'dense', 'global']):
            continue
        
        # Look for conv layers
        if any(layer_type in layer_class 
               for layer_type in ['conv2d', 'separableconv2d', 'depthwiseconv2d']):
            try:
                if hasattr(layer, 'output') and len(layer.output.shape) == 4:
                    shape = layer.output.shape
                    if shape[1] is not None and shape[2] is not None and shape[1] > 1 and shape[2] > 1:
                        conv_layers.append((layer.name, shape))
            except:
                pass
        
        # Look for activation layers
        elif ('activation' in layer_class or 'relu' in layer_class):
            try:
                if hasattr(layer, 'output') and len(layer.output.shape) == 4:
                    shape = layer.output.shape
                    if shape[1] is not None and shape[2] is not None and shape[1] > 1 and shape[2] > 1:
                        activation_layers.append((layer.name, shape))
            except:
                pass
    
    # Return best available layers
    return {
        'conv_layers': conv_layers,
        'activation_layers': activation_layers,
        'best_layer': activation_layers[-1] if activation_layers else (conv_layers[-1] if conv_layers else None),
        'nested_model': nested_base_model
    }

def test_gradcam(model, input_shape, model_name):
    """Test if Grad-CAM can work with this model"""
    try:
        # Find suitable layers
        layer_info = find_target_layer(model)
        
        if layer_info['best_layer'] is None:
            return {
                'compatible': False,
                'reason': 'No suitable convolutional or activation layer found',
                'layers_found': {
                    'conv_layers': 0,
                    'activation_layers': 0
                }
            }
        
        best_layer_name, best_layer_shape = layer_info['best_layer']
        nested_model = layer_info.get('nested_model')
        
        # Create a dummy image with correct input shape
        dummy_image = np.random.rand(1, input_shape[0], input_shape[1], input_shape[2]).astype(np.float32)
        
        # Try to create Grad-CAM model - handle nested models
        if nested_model:
            target_layer = nested_model.get_layer(best_layer_name)
        else:
            target_layer = model.get_layer(best_layer_name)
        
        grad_model = keras.Model(
            inputs=model.input,
            outputs=[target_layer.output, model.output]
        )
        
        # Test forward pass
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(dummy_image)
            if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                class_idx = tf.argmax(predictions[0])
                class_channel = predictions[:, class_idx]
            else:
                class_channel = predictions[0]
        
        # Test gradient computation
        grads = tape.gradient(class_channel, conv_outputs)
        
        if grads is None:
            return {
                'compatible': False,
                'reason': 'Gradient computation returned None',
                'best_layer': best_layer_name,
                'layers_found': {
                    'conv_layers': len(layer_info['conv_layers']),
                    'activation_layers': len(layer_info['activation_layers'])
                }
            }
        
        # Success!
        return {
            'compatible': True,
            'best_layer': best_layer_name,
            'best_layer_shape': str(best_layer_shape),
            'layers_found': {
                'conv_layers': len(layer_info['conv_layers']),
                'activation_layers': len(layer_info['activation_layers'])
            },
            'conv_layer_names': [name for name, _ in layer_info['conv_layers'][:3]],  # First 3
            'activation_layer_names': [name for name, _ in layer_info['activation_layers'][:3]]  # First 3
        }
        
    except Exception as e:
        return {
            'compatible': False,
            'reason': str(e)[:200],
            'layers_found': {
                'conv_layers': len(layer_info.get('conv_layers', [])),
                'activation_layers': len(layer_info.get('activation_layers', []))
            }
        }

# Test each active model
for condition, active_model_id in active_models.items():
    if not active_model_id or active_model_id not in models_info:
        continue
    
    results['total'] += 1
    model_info = models_info[active_model_id]
    
    print(f"\n{'='*80}")
    print(f"Testing: {condition.upper()}")
    print(f"{'='*80}")
    print(f"Model ID: {active_model_id}")
    print(f"Architecture: {model_info.get('architecture', 'Unknown')}")
    print(f"Input Shape: {model_info.get('input_shape', [224, 224, 3])}")
    print(f"Accuracy: {model_info.get('accuracy', 'N/A')}")
    
    # Get model path
    model_path = f"models/{model_info['file_path']}"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        results['incompatible'] += 1
        results['details'].append({
            'condition': condition,
            'model_id': active_model_id,
            'compatible': False,
            'reason': 'Model file not found'
        })
        continue
    
    print(f"Model Path: {model_path}")
    print(f"File Size: {model_info.get('file_size', 'Unknown')} MB")
    
    try:
        print("\n🔄 Loading model...")
        model = keras.models.load_model(model_path, compile=False)
        print(f"✅ Model loaded successfully")
        
        # Get input shape
        input_shape = model_info.get('input_shape', [224, 224, 3])
        
        print(f"\n🔍 Testing Grad-CAM compatibility...")
        test_result = test_gradcam(model, input_shape, active_model_id)
        
        if test_result['compatible']:
            print(f"✅ GRAD-CAM COMPATIBLE!")
            print(f"   Best Layer: {test_result['best_layer']}")
            print(f"   Layer Shape: {test_result['best_layer_shape']}")
            print(f"   Conv Layers Found: {test_result['layers_found']['conv_layers']}")
            print(f"   Activation Layers Found: {test_result['layers_found']['activation_layers']}")
            if test_result.get('conv_layer_names'):
                print(f"   Sample Conv Layers: {', '.join(test_result['conv_layer_names'])}")
            if test_result.get('activation_layer_names'):
                print(f"   Sample Activation Layers: {', '.join(test_result['activation_layer_names'])}")
            results['compatible'] += 1
        else:
            print(f"❌ NOT GRAD-CAM COMPATIBLE")
            print(f"   Reason: {test_result['reason']}")
            print(f"   Conv Layers Found: {test_result['layers_found']['conv_layers']}")
            print(f"   Activation Layers Found: {test_result['layers_found']['activation_layers']}")
            results['incompatible'] += 1
        
        results['details'].append({
            'condition': condition,
            'model_id': active_model_id,
            'architecture': model_info.get('architecture'),
            'input_shape': input_shape,
            **test_result
        })
        
        # Clean up
        del model
        tf.keras.backend.clear_session()
        
    except Exception as e:
        print(f"❌ Error loading/testing model: {str(e)[:200]}")
        results['incompatible'] += 1
        results['details'].append({
            'condition': condition,
            'model_id': active_model_id,
            'compatible': False,
            'reason': str(e)[:200]
        })

# Print summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Total Models Tested: {results['total']}")
print(f"✅ Grad-CAM Compatible: {results['compatible']}")
print(f"❌ Not Compatible: {results['incompatible']}")
print(f"Success Rate: {(results['compatible']/results['total']*100):.1f}%" if results['total'] > 0 else "N/A")

print("\n" + "="*80)
print("DETAILED RESULTS BY MODEL")
print("="*80)

for detail in results['details']:
    status = "✅" if detail['compatible'] else "❌"
    print(f"{status} {detail['condition'].upper()}: {detail['model_id']}")
    print(f"   Architecture: {detail.get('architecture', 'Unknown')}")
    print(f"   Input Shape: {detail.get('input_shape', 'Unknown')}")
    if detail['compatible']:
        print(f"   Best Layer: {detail.get('best_layer', 'Unknown')}")
        print(f"   Layer Shape: {detail.get('best_layer_shape', 'Unknown')}")
    else:
        print(f"   Reason: {detail.get('reason', 'Unknown')}")
    print()

# Architecture-based analysis
print("="*80)
print("ANALYSIS BY ARCHITECTURE")
print("="*80)

arch_stats = {}
for detail in results['details']:
    arch = detail.get('architecture', 'Unknown')
    if arch not in arch_stats:
        arch_stats[arch] = {'total': 0, 'compatible': 0}
    arch_stats[arch]['total'] += 1
    if detail['compatible']:
        arch_stats[arch]['compatible'] += 1

for arch, stats in arch_stats.items():
    success_rate = (stats['compatible'] / stats['total'] * 100) if stats['total'] > 0 else 0
    status = "✅" if success_rate == 100 else "⚠️" if success_rate > 0 else "❌"
    print(f"{status} {arch}: {stats['compatible']}/{stats['total']} ({success_rate:.1f}% compatible)")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

if results['compatible'] == results['total']:
    print("✅ All active models are Grad-CAM compatible!")
    print("   You can use Grad-CAM visualizations for all conditions.")
else:
    print("⚠️ Some models are not Grad-CAM compatible.")
    print("   Recommendations:")
    print("   1. For incompatible models, the system will use simple overlay visualization")
    print("   2. Consider retraining incompatible models with proper convolutional layers")
    print("   3. Ensure models have spatial feature maps (not fully flattened)")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
