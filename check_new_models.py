import tensorflow as tf
import os
from datetime import datetime

print('🔍 CHECKING NEW DENSENET121 CARDIOMEGALY MODELS:')
print('=' * 55)

models_to_check = [
    'models/DenseNet121_cardiomegaly.h5',
    'models/DenseNet121_cardiomegaly_20251120_200558.h5'
]

for model_path in models_to_check:
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            params = model.count_params()
            created_time = datetime.fromtimestamp(os.path.getctime(model_path))
            
            print(f'\n📄 {os.path.basename(model_path)}:')
            print(f'   ├─ Size: {size_mb:.1f} MB')
            print(f'   ├─ Parameters: {params:,}')
            print(f'   ├─ Input Shape: {model.input_shape}')
            print(f'   ├─ Output Classes: {model.output_shape[-1]}')
            print(f'   └─ Created: {created_time.strftime("%Y-%m-%d %H:%M:%S")}')
            
        except Exception as e:
            print(f'\n❌ Error loading {model_path}: {str(e)}')
    else:
        print(f'\n❌ Model not found: {model_path}')

print('\n' + '=' * 55)