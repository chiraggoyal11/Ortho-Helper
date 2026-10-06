# DenseNet121 Cardiomegaly Detection Model

## Model Information
- **Architecture**: DenseNet121
- **Medical Condition**: Cardiomegaly (Heart Enlargement)
- **Configuration**: Intensive
- **Training Date**: 20251006_192404
- **Parameters**: 7,305,281

## Performance Metrics
- **Test Accuracy**: 0.6300 (63.00%)
- **Test Precision**: 0.6816
- **Test Recall**: 0.5728
- **Test Loss**: 0.8918

## Dataset Information
- **Classes**: Normal vs Cardiomegaly
- **Training Samples**: 1600
- **Test Samples**: 400
- **Epochs Trained**: 8

## Grad-CAM Optimization
This model is optimized for superior Grad-CAM visualization:
- **Recommended Layer**: `conv5_block16_2_conv`
- **Architecture Benefits**: Dense connectivity for cardiac imaging
- **Visualization Quality**: Excellent for heart abnormality detection

## Usage
```python
import tensorflow as tf
from utils.gradcam import GradCAM

# Load model
model = tf.keras.models.load_model('cardiomegaly_densenet121_models\cardiomegaly_intensive_20251006_192404\cardiomegaly_densenet121_intensive_20251006_192404.keras')

# Initialize Grad-CAM
gradcam = GradCAM(model, layer_name='conv5_block16_2_conv')

# Generate heatmap
heatmap = gradcam.generate_heatmap(chest_xray_image)
```

Generated: 2025-10-06 19:24:07
