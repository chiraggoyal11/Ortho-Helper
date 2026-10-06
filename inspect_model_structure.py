"""
Deep Model Structure Inspection
Analyzes the internal structure of models to find Grad-CAM compatible layers
"""

import os
import json
import tensorflow as tf
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

def inspect_model_deeply(model, depth=0, max_depth=3):
    """Recursively inspect model structure"""
    layers_info = []
    
    for layer in model.layers:
        layer_name = layer.name
        layer_class = layer.__class__.__name__
        
        try:
            if hasattr(layer, 'output'):
                output_shape = layer.output.shape
            else:
                output_shape = "No output shape"
        except:
            output_shape = "Error getting shape"
        
        info = {
            'name': layer_name,
            'class': layer_class,
            'output_shape': str(output_shape),
            'depth': depth
        }
        
        layers_info.append(info)
        
        # If this is a Functional model, inspect its layers too
        if depth < max_depth and hasattr(layer, 'layers'):
            try:
                sub_info = inspect_model_deeply(layer, depth + 1, max_depth)
                info['sublayers'] = sub_info
            except:
                pass
    
    return layers_info

# Load registry
registry_path = 'models/registry/model_registry.json'
with open(registry_path, 'r', encoding='utf-8') as f:
    registry = json.load(f)

active_models = registry.get('active_models', {})
models_info = registry.get('models', {})

# Test one model from each architecture
test_models = {
    'pneumonia': 'pneumonia_fast_20251007_015119',
    'arthritis': 'arthritis_new_intensive'
}

for condition, model_id in test_models.items():
    print("\n" + "="*80)
    print(f"INSPECTING: {condition.upper()} - {model_id}")
    print("="*80)
    
    model_info = models_info[model_id]
    model_path = f"models/{model_info['file_path']}"
    
    print(f"Architecture: {model_info.get('architecture')}")
    print(f"Model Path: {model_path}")
    print()
    
    # Load model
    model = keras.models.load_model(model_path, compile=False)
    
    print(f"Model Type: {type(model).__name__}")
    print(f"Number of top-level layers: {len(model.layers)}")
    print()
    
    # Inspect structure
    print("LAYER STRUCTURE:")
    print("-" * 80)
    
    layers_info = inspect_model_deeply(model, max_depth=2)
    
    # Print top-level layers
    for i, info in enumerate(layers_info):
        print(f"{i+1}. {info['class']:25} | {info['name']:40} | Shape: {info['output_shape']}")
        
        # If this layer has sublayers, show some of them
        if 'sublayers' in info:
            sublayers = info['sublayers']
            print(f"   └─ Contains {len(sublayers)} sublayers:")
            
            # Show first 5 and last 5 sublayers
            show_sublayers = sublayers[:5] + (['...'] if len(sublayers) > 10 else []) + sublayers[-5:]
            
            for sub in show_sublayers:
                if sub == '...':
                    print(f"      ... ({len(sublayers) - 10} more layers)")
                else:
                    print(f"      - {sub['class']:23} | {sub['name']:35} | {sub['output_shape']}")
    
    print()
    print("SEARCHING FOR GRAD-CAM COMPATIBLE LAYERS:")
    print("-" * 80)
    
    # Search for convolutional layers at all depths
    def find_conv_layers(layers_list, parent_path=""):
        conv_found = []
        for layer_info in layers_list:
            current_path = f"{parent_path}/{layer_info['name']}" if parent_path else layer_info['name']
            
            # Check if this is a convolutional layer
            if any(conv_type in layer_info['class'].lower() 
                   for conv_type in ['conv2d', 'separable', 'depthwise']):
                # Check if it has spatial dimensions
                shape_str = layer_info['output_shape']
                if 'None' in shape_str and ',' in shape_str:
                    # Likely has batch, height, width, channels
                    conv_found.append({
                        'path': current_path,
                        'class': layer_info['class'],
                        'shape': shape_str
                    })
            
            # Check activation layers
            if 'activation' in layer_info['class'].lower() or 'relu' in layer_info['class'].lower():
                shape_str = layer_info['output_shape']
                if 'None' in shape_str and ',' in shape_str and shape_str.count(',') >= 3:
                    conv_found.append({
                        'path': current_path,
                        'class': layer_info['class'],
                        'shape': shape_str
                    })
            
            # Recursively check sublayers
            if 'sublayers' in layer_info:
                sub_conv = find_conv_layers(layer_info['sublayers'], current_path)
                conv_found.extend(sub_conv)
        
        return conv_found
    
    conv_layers = find_conv_layers(layers_info)
    
    if conv_layers:
        print(f"✅ Found {len(conv_layers)} potential Grad-CAM compatible layers:")
        print()
        for i, conv in enumerate(conv_layers[:10], 1):  # Show first 10
            print(f"{i}. {conv['class']:25} | {conv['path']}")
            print(f"   Shape: {conv['shape']}")
        
        if len(conv_layers) > 10:
            print(f"   ... and {len(conv_layers) - 10} more layers")
        
        print()
        print("RECOMMENDED GRAD-CAM LAYER:")
        # Use the last suitable layer (usually best for Grad-CAM)
        recommended = conv_layers[-1]
        print(f"✅ Layer: {recommended['path']}")
        print(f"   Type: {recommended['class']}")
        print(f"   Shape: {recommended['shape']}")
        
    else:
        print("❌ No convolutional or activation layers found with spatial dimensions")
        print("   This model may not support Grad-CAM visualization")
    
    print()
    
    # Try to access nested model
    if len(model.layers) > 0:
        first_layer = model.layers[0]
        print("FIRST LAYER DETAILS:")
        print(f"  Name: {first_layer.name}")
        print(f"  Type: {type(first_layer).__name__}")
        
        if hasattr(first_layer, 'layers'):
            print(f"  This is a nested model with {len(first_layer.layers)} sublayers")
            print(f"  Can access sublayers: YES ✅")
            
            # Try to get a layer from the nested model
            try:
                nested_model = first_layer
                print(f"\n  Testing direct access to nested layers:")
                
                # Find a suitable layer
                for sublayer in nested_model.layers:
                    sublayer_class = sublayer.__class__.__name__
                    if any(conv_type in sublayer_class.lower() 
                           for conv_type in ['conv2d', 'activation', 'relu']):
                        print(f"  ✅ Can access: model.layers[0].layers[x] -> {sublayer.name} ({sublayer_class})")
                        break
            except Exception as e:
                print(f"  ❌ Error accessing nested layers: {e}")
        else:
            print(f"  Not a nested model")
    
    # Clean up
    del model
    tf.keras.backend.clear_session()
    
    print()

print("\n" + "="*80)
print("INSPECTION COMPLETE")
print("="*80)
print("\nKEY FINDINGS:")
print("- Sequential models wrap base architectures (DenseNet121/MobileNetV2)")
print("- Convolutional layers are inside the nested model (model.layers[0])")
print("- Grad-CAM needs to access these nested layers")
print("- Current Grad-CAM implementation may need to handle nested models")
