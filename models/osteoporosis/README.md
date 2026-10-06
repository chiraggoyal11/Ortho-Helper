# DenseNet121 Knee Osteoporosis Detection Model

## Model Information
- **Architecture**: DenseNet121
- **Medical Condition**: Knee Osteoporosis Detection
- **Configuration**: Intensive
- **Training Date**: 20251006_183913
- **Parameters**: 7,305,281

## Performance Metrics
- **Test Accuracy**: 0.9177 (91.77%)
- **Test Precision**: 0.9568
- **Test Recall**: 0.8806
- **Training Time**: 12.3 minutes
- **Epochs Trained**: 15/15

## Dataset Information
- **Classes**: Normal vs Osteoporosis
- **Training Data**: Medical imaging dataset
- **Dataset Path**: Dataset/KNEE/Osteoporosis/train
- **Image Size**: 224x224 pixels

## Grad-CAM Optimization
This model is optimized for superior Grad-CAM visualization:
- **Recommended Layer**: `conv5_block16_2_conv`
- **Architecture Benefits**: Dense connectivity for medical imaging
- **Visualization Quality**: Excellent for knee osteoporosis detection
- **Medical Interpretation**: Clear region highlighting

## Usage
```python
import tensorflow as tf
from utils.gradcam import GradCAM

# Load model
model = tf.keras.models.load_model('models/densenet121_osteoporosis_intensive_models/models/densenet121_osteoporosis_intensive_20251006_183913.h5')

# Initialize Grad-CAM
gradcam = GradCAM(model, layer_name='conv5_block16_2_conv')

# Generate heatmap
heatmap = gradcam.generate_heatmap(medical_image)
```

## Clinical Application
- **Primary Use**: Knee Osteoporosis Detection
- **Accuracy Level**: Excellent (91.77%)
- **Deployment Ready**: Medical grade
- **Visualization**: Grad-CAM heatmaps for medical interpretation

Generated: 2025-10-06 19:17:18