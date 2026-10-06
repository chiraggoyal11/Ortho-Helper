# Model Training Guide for Medical X-ray AI System

## Overview
This guide explains how to train custom AI models using your own medical X-ray datasets. The system supports multiple architectures and provides easy model swapping capabilities.

## 🚀 Quick Start Training

### 1. Prepare Your Dataset
1. **Organize Data**: Place your X-ray images in the `Dataset` folder following the expected structure
2. **Navigate**: Go to "📊 Dataset Overview" in the application
3. **Scan**: Click the scan button to analyze available datasets
4. **Prepare**: Click "Prepare for Training" for each dataset you want to use

### 2. Start Training
1. **Navigate**: Go to "🚀 Model Training" 
2. **Select Architecture**: Choose from:
   - **EfficientNetB0**: Lightweight and efficient
   - **ResNet50**: Deep residual network for complex patterns
   - **DenseNet121**: Dense connections for efficient feature reuse
   - **VGG16**: Classic architecture, good baseline
   - **Custom_CNN**: Lightweight custom architecture
3. **Choose Configuration**:
   - **Quick Test**: 5 epochs (fast validation)
   - **Standard**: 25 epochs (balanced performance)
   - **Intensive**: 50 epochs (maximum performance)
4. **Click Train**: Monitor progress in real-time

### 3. Manage Models
1. **Navigate**: Go to "🔧 Model Management"
2. **View Registry**: See all trained models
3. **Activate Models**: Set which model to use for each condition
4. **Compare Performance**: View accuracy comparisons
5. **Export/Import**: Share models between systems

## 📊 Dataset Structure

### Expected Directory Layout
```
Dataset/
├── ARM/FracAtlas/
│   ├── images/
│   │   ├── Fractured/
│   │   └── Non_fractured/
│   └── dataset.csv
├── CHEST/chest_xray Pneumonia/
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── test/
│   └── val/
└── KNEE/Osteoarthritis Knee X-ray/
    ├── Normal/
    ├── Osteoporosis/
    └── Arthritis/
```

### Supported Formats
- **Images**: `.jpg`, `.jpeg`, `.png`, `.dcm` (DICOM)
- **Directory Structures**: 
  - Direct class folders (e.g., `Normal/`, `Fractured/`)
  - Train/Test/Val splits with class subfolders
  - Mixed structures (automatically detected)

## 🏗️ Model Architectures

### EfficientNetB0
- **Best for**: Balanced performance and efficiency
- **Parameters**: ~5.3M
- **Input**: 224×224×3
- **Training Time**: Fast to medium
- **Memory**: Low

### ResNet50
- **Best for**: Complex pattern recognition
- **Parameters**: ~25.6M  
- **Input**: 224×224×3
- **Training Time**: Medium
- **Memory**: Medium

### DenseNet121
- **Best for**: Feature reuse and efficiency
- **Parameters**: ~8.1M
- **Input**: 224×224×3
- **Training Time**: Medium
- **Memory**: Medium

### Custom CNN
- **Best for**: Quick experiments and limited resources
- **Parameters**: ~2.1M
- **Input**: 224×224×3
- **Training Time**: Fast
- **Memory**: Very low

## ⚙️ Training Configuration

### Quick Test (5 epochs)
- **Use case**: Initial validation and debugging
- **Training time**: 10-30 minutes
- **Expected accuracy**: 60-80%
- **Recommended for**: Testing dataset preparation

### Standard (25 epochs)
- **Use case**: Production-ready models
- **Training time**: 1-3 hours
- **Expected accuracy**: 85-95%
- **Recommended for**: Most use cases

### Intensive (50 epochs)
- **Use case**: Maximum performance
- **Training time**: 3-6 hours
- **Expected accuracy**: 90-97%
- **Recommended for**: Final deployment models

## 💡 Training Tips

### Before Training
1. **Check GPU**: Training is much faster with GPU support
2. **Verify Dataset**: Ensure balanced classes and sufficient samples
3. **Free Space**: Ensure 5-10GB free disk space for checkpoints
4. **Close Apps**: Free up system memory for training

### During Training
1. **Monitor Metrics**: Watch training/validation accuracy
2. **Early Stopping**: Training stops automatically if no improvement
3. **Save Checkpoints**: Best models are automatically saved
4. **Resource Usage**: Monitor CPU/GPU/Memory usage

### After Training
1. **Evaluate Performance**: Review test accuracy and confusion matrix
2. **Activate Model**: Set the trained model as active for inference
3. **Compare Models**: Use model comparison to find the best performer
4. **Export Model**: Save model package for backup or sharing

## 🔧 Model Management

### Model Registry
- **Version Control**: Each model is versioned automatically
- **Metadata**: Stores architecture, performance, and training details
- **Hash Verification**: Ensures model file integrity
- **Search & Filter**: Find models by type, accuracy, or date

### Model Swapping
- **Hot Swap**: Change active models without restarting
- **A/B Testing**: Compare different models on the same data
- **Rollback**: Easily revert to previous models
- **Backup**: Automatic backup of replaced models

### Performance Comparison
- **Side-by-side**: Compare multiple models for the same condition
- **Metrics**: Accuracy, precision, recall, F1-score
- **Ranking**: Automatic sorting by performance
- **Visualization**: Charts and graphs for easy comparison

## 📈 Performance Optimization

### Data Optimization
- **Balance Classes**: Aim for roughly equal samples per class
- **Quality Control**: Remove blurry, corrupted, or mislabeled images
- **Augmentation**: The system applies automatic augmentation during training
- **Size**: Minimum 100 samples per class, ideally 500+

### Training Optimization
- **Batch Size**: Start with 32, reduce if memory issues
- **Learning Rate**: System uses adaptive learning rates
- **Transfer Learning**: Pre-trained weights provide better starting points
- **Callbacks**: Early stopping prevents overfitting

### Hardware Requirements
- **Minimum**: 8GB RAM, 4-core CPU
- **Recommended**: 16GB RAM, 8-core CPU, 4GB+ GPU
- **Intensive**: 32GB RAM, GPU with 8GB+ VRAM
- **Storage**: 50GB+ available space

## 🚨 Troubleshooting

### Common Issues

#### "Dataset not prepared" Error
- **Solution**: Visit Dataset Overview and prepare the dataset first
- **Check**: Ensure dataset folder structure is correct
- **Verify**: Check that images are in supported formats

#### Training Fails to Start
- **Check**: Verify sufficient disk space (5GB+)
- **Memory**: Close other applications to free RAM
- **Dependencies**: Ensure all packages are installed correctly

#### Low Training Accuracy
- **Data**: Check for class imbalance or mislabeled images
- **Architecture**: Try different model architecture
- **Epochs**: Increase training duration with intensive configuration

#### Out of Memory Errors
- **Reduce**: Decrease batch size in advanced options
- **Close**: Close other applications
- **Hardware**: Consider upgrading system memory

### Getting Help
1. **Check Logs**: Training logs are saved in `models/logs/`
2. **Model Info**: View detailed information in Model Registry
3. **System Status**: Check system resources during training
4. **Documentation**: Review this guide and README.md

## 🎯 Best Practices

### Dataset Preparation
1. **Consistent Quality**: Use similar X-ray imaging conditions
2. **Proper Labeling**: Verify labels are correct before training
3. **Sufficient Size**: Aim for 500+ samples per class minimum
4. **Test Set**: Keep 15-20% of data for final testing

### Model Selection
1. **Start Simple**: Begin with Custom_CNN for quick validation
2. **Upgrade Gradually**: Move to EfficientNet for production
3. **Compare Multiple**: Train several architectures and compare
4. **Consider Resources**: Balance performance with available hardware

### Training Strategy
1. **Quick Test First**: Always start with quick test configuration
2. **Monitor Progress**: Watch for overfitting or underfitting
3. **Save Everything**: Keep all trained models for comparison
4. **Document Results**: Record training parameters and results

### Deployment
1. **Validate Thoroughly**: Test on unseen data before deployment
2. **Monitor Performance**: Track accuracy in production use
3. **Update Regularly**: Retrain with new data as available
4. **Backup Models**: Keep copies of production models safe

## 📚 Advanced Features

### Custom Training Configurations
- Modify epochs, batch size, and learning rate in advanced options
- Custom preprocessing and augmentation pipelines
- Learning rate scheduling and optimization strategies

### Model Export/Import
- Export trained models as portable packages
- Share models between different system installations
- Version control and model collaboration

### Performance Analytics
- Detailed training history visualization
- Confusion matrices and classification reports
- ROC curves and precision-recall analysis

---

For technical support or questions about model training, please refer to the main README.md or system documentation.