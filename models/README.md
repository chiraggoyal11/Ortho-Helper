# Placeholder README for models directory
# This directory contains the trained models for the Medical X-ray AI Classification System

## Model Files

This directory should contain the following trained model files:

### 1. bone_fracture_model.h5
- **Purpose**: Bone fracture detection in X-ray images
- **Architecture**: Convolutional Neural Network (CNN)
- **Input**: 224x224x3 RGB images
- **Output**: Binary classification (Normal/Fracture)
- **Dataset**: FracAtlas dataset from your existing data

### 2. chest_conditions_model.h5
- **Purpose**: Chest condition detection (Pneumonia, Cardiomegaly)
- **Architecture**: Convolutional Neural Network (CNN) 
- **Input**: 224x224x3 RGB images
- **Output**: Multi-class classification (Normal/Pneumonia/Cardiomegaly)
- **Dataset**: Chest X-ray datasets from your existing data

### 3. knee_conditions_model.h5
- **Purpose**: Knee condition detection (Osteoporosis, Arthritis)
- **Architecture**: Convolutional Neural Network (CNN)
- **Input**: 224x224x3 RGB images
- **Output**: Multi-class classification (Normal/Osteoporosis/Arthritis)
- **Dataset**: Knee X-ray datasets from your existing data

## Model Training

To train the models with your actual datasets, use the `model_training.py` script:

```python
from models.model_training import MedicalImageModelTrainer

# Example for bone fracture detection
trainer = MedicalImageModelTrainer('bone_fracture')
model = trainer.create_model('custom_cnn')
trainer.compile_model()

# Train with your dataset
history = trainer.train_model(x_train, y_train, x_val, y_val)

# Save the trained model
trainer.save_model('models/bone_fracture_model.h5')
```

## Placeholder Models

The system will create placeholder models automatically if the trained models are not available. These are lightweight dummy models for demonstration purposes only.

To replace with actual trained models:
1. Prepare your datasets from the existing data folders
2. Run the training scripts in `model_training.py`
3. The trained models will be saved in this directory

## Model Performance

After training with your actual datasets, document the performance metrics here:

### Bone Fracture Model
- Training Accuracy: TBD
- Validation Accuracy: TBD
- Test Accuracy: TBD

### Chest Conditions Model  
- Training Accuracy: TBD
- Validation Accuracy: TBD
- Test Accuracy: TBD

### Knee Conditions Model
- Training Accuracy: TBD
- Validation Accuracy: TBD
- Test Accuracy: TBD

## Notes

- All models expect input images to be preprocessed (resized to 224x224, normalized)
- Models are saved in Keras H5 format for easy loading with TensorFlow
- For production use, consider converting to TensorFlow Lite or ONNX format for better performance