# Medical X-ray AI Classification System

## Overview
A comprehensive Streamlit-based web application for automated medical X-ray analysis using artificial intelligence. The system provides real-time classification of bone fractures, chest conditions, and knee conditions with explainable AI features, plus advanced model training and management capabilities.

## 🏥 Features

### Core Functionality
- **Multi-modal X-ray Classification**
  - Bone fracture detection with Grad-CAM visualization
  - Chest condition detection (Pneumonia, Cardiomegaly)
  - Knee condition detection (Osteoporosis, Arthritis)

- **Advanced AI Features**
  - Convolutional Neural Networks (CNNs) for image analysis
  - Grad-CAM heatmaps for explainable AI (bone fractures)
  - Confidence scoring and interpretation
  - Real-time image preprocessing

- **Model Training & Management** 🚀 NEW!
  - **Custom Model Training** - Train AI models on your own datasets
  - **Multiple Architectures** - EfficientNet, ResNet50, DenseNet121, VGG16, Custom CNN
  - **Swappable Models** - Easy model switching for optimal performance
  - **Model Registry** - Version control and model management
  - **Performance Comparison** - Compare different models side-by-side
  - **Import/Export** - Share models between systems

- **Dataset Management** 📊 NEW!
  - **Automatic Dataset Discovery** - Scan and analyze available datasets
  - **Dataset Preparation** - Automatic train/validation/test splitting  
  - **Multi-format Support** - JPEG, PNG, DICOM support
  - **Class Balance Analysis** - Understand dataset distribution

- **Professional Reporting**
  - Automated PDF report generation
  - HTML reports for web viewing
  - Clinical recommendations
  - Professional medical formatting

### User Experience
- **Modern Authentication System**
  - Role-based access control (Doctor, Radiologist, Student)
  - Session management
  - User activity tracking

- **Intuitive Interface**
  - Drag-and-drop file upload
  - Real-time image preview
  - Interactive confidence visualization
  - Responsive design

- **Advanced Analytics** 📈 NEW!
  - User feedback collection
  - Performance analytics
  - Model performance tracking
  - System usage statistics

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation

1. **Navigate to the project directory:**
   ```bash
   cd "d:\ortho helper"
   ```

2. **Create and activate virtual environment:**
   ```bash
   # Windows
   python -m venv medical_xray_env
   medical_xray_env\Scripts\activate

   # Linux/Mac
   python -m venv medical_xray_env
   source medical_xray_env/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

5. **Access the application:**
   Open your browser and navigate to `http://localhost:8501`

### Demo Login Credentials
- **Doctor**: username: `doctor`, password: `medical123`
- **Student**: username: `student`, password: `learn123`
- **Radiologist**: username: `radiologist`, password: `radiology456`

## 📁 Project Structure

```
ortho helper/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── 
├── utils/                          # Core utilities
│   ├── __init__.py
│   ├── authentication.py          # User authentication
│   ├── image_preprocessing.py      # Image processing pipeline
│   ├── model_inference.py          # AI model inference
│   ├── gradcam.py                  # Grad-CAM visualization
│   ├── report_generator.py         # Report generation
│   └── feedback_system.py          # Feedback collection
├──
├── models/                         # AI models directory
│   ├── README.md                   # Model documentation
│   ├── model_training.py           # Training scripts
│   ├── bone_fracture_model.h5      # Bone fracture CNN
│   ├── chest_conditions_model.h5   # Chest conditions CNN
│   └── knee_conditions_model.h5    # Knee conditions CNN
├──
└── Dataset/                        # Your existing medical datasets
    ├── ARM/FracAtlas/
    ├── Bone_Fracture_Binary_Classification/
    ├── CHEST/
    ├── DGCN/
    └── KNEE/
```

## 🔧 System Architecture

### AI Pipeline
1. **Image Acquisition** → Upload X-ray images (JPG, PNG, DICOM)
2. **Preprocessing** → Resize, normalize, enhance contrast
3. **Model Inference** → CNN-based classification
4. **Explainability** → Grad-CAM heatmap generation
5. **Report Generation** → PDF/HTML report creation

### Model Architecture
- **Input**: 224×224×3 RGB images
- **Architecture**: Custom CNNs with BatchNormalization and Dropout
- **Output**: Class probabilities with confidence scores
- **Explainability**: Grad-CAM for bone fracture detection

### Technology Stack
- **Frontend**: Streamlit with custom CSS
- **Backend**: Python, TensorFlow/Keras
- **Image Processing**: OpenCV, PIL, NumPy
- **Reports**: ReportLab (PDF), HTML templating
- **Visualization**: Matplotlib, Plotly

## 📊 Model Information

### Bone Fracture Detection
- **Classes**: Normal, Fracture
- **Features**: Grad-CAM visualization, high accuracy
- **Dataset**: FracAtlas from your existing data

### Chest Condition Detection  
- **Classes**: Normal, Pneumonia, Cardiomegaly
- **Features**: Multi-condition detection, robust preprocessing
- **Dataset**: Combined chest X-ray datasets

### Knee Condition Detection
- **Classes**: Normal, Osteoporosis, Arthritis  
- **Features**: Age-related conditions, severity grading
- **Dataset**: Knee osteoarthritis datasets

## 🎯 Usage Guide

### Step 1: Authentication
- Log in with provided credentials
- Access level determined by user role

### Step 2: Upload X-ray
- Drag and drop or browse for X-ray image
- Supported formats: JPG, PNG, DICOM
- Real-time image validation

### Step 3: Select Analysis Type
- Choose from: Bone, Chest, or Knee analysis
- Configure preprocessing options
- Click "Classify X-ray"

### Step 4: Review Results
- View prediction with confidence score
- Examine Grad-CAM heatmap (bone fractures)
- Read clinical interpretation

### Step 5: Generate Reports
- Create PDF or HTML reports
- Include images, results, and recommendations
- Download for clinical use

### Step 6: Provide Feedback
- Rate prediction accuracy
- Select predefined feedback options
- Add additional comments

## ⚙️ Configuration

### Preprocessing Options
- **Resizing**: Auto-resize to optimal dimensions (224×224)
- **Normalization**: Pixel value standardization (0-1 range)
- **Enhancement**: Contrast improvement, noise reduction
- **Augmentation**: Optional data augmentation

### Model Settings
- **Confidence Threshold**: Adjustable prediction threshold
- **Grad-CAM Intensity**: Heatmap visualization intensity
- **Batch Processing**: Multiple image analysis (future feature)

### Report Configuration
- **Format Options**: PDF, HTML, or both
- **Content Control**: Include/exclude metadata, preprocessing details
- **Auto-download**: Automatic report download option

## 🔒 Security & Privacy

### Authentication
- Secure password hashing (SHA-256)
- Session timeout management
- Role-based permissions

### Data Privacy
- No permanent image storage
- Session-based processing only
- Optional feedback data collection

### Medical Compliance
- Educational/research use only
- Clinical disclaimer included
- Professional review recommended

## 🌟 Advanced Features

### Explainable AI
- **Grad-CAM**: Visual explanation for bone fracture detection
- **Focus Area Analysis**: Quantitative heatmap analysis
- **Multi-layer Visualization**: Different CNN layer insights

### Analytics Dashboard
- **Feedback Analytics**: Performance tracking
- **Usage Statistics**: System utilization metrics
- **Trend Analysis**: Performance over time

### Extensibility
- **Plugin Architecture**: Easy model addition
- **API Integration**: RESTful API endpoints (future)
- **Database Support**: Optional data persistence

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py
```

### Cloud Deployment Options

#### Streamlit Cloud
1. Push code to GitHub repository
2. Connect Streamlit Cloud to repository
3. Deploy with automatic CI/CD

#### Heroku
```bash
# Add Procfile and runtime.txt
echo "web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0" > Procfile
echo "python-3.9.16" > runtime.txt

# Deploy
heroku create your-app-name
git push heroku main
```

#### Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## 🛠️ Development

### Adding New Models
1. Create model in `models/` directory
2. Add configuration in `model_inference.py`
3. Update UI options in `app.py`
4. Test thoroughly before deployment

### Custom Preprocessing
1. Extend `image_preprocessing.py`
2. Add new preprocessing functions
3. Update pipeline configurations
4. Document changes

### UI Customization
1. Modify CSS in `app.py`
2. Add new Streamlit components
3. Update layout and styling
4. Test across different devices

## 📈 Performance Optimization

### Model Optimization
- Model quantization for faster inference
- TensorFlow Lite conversion for mobile deployment
- ONNX format for cross-platform compatibility

### Caching Strategy
- `@st.cache_resource` for model loading
- `@st.cache_data` for preprocessing results
- Session state management for user data

### Scalability
- Implement model versioning
- Add load balancing for high traffic
- Database integration for production use

## 🐛 Troubleshooting

### Common Issues

**Models not loading:**
- Check model files exist in `models/` directory
- Verify TensorFlow compatibility
- Check file permissions

**Import errors:**
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt`
- Check Python version compatibility

**Performance issues:**
- Monitor CPU/memory usage
- Reduce image resolution if needed
- Clear Streamlit cache regularly

**Authentication problems:**
- Verify credentials in `authentication.py`
- Check session timeout settings
- Clear browser cookies

### Getting Help
- Check error messages in console
- Review Streamlit documentation
- Consult medical AI best practices
- Contact development team

## 📚 References

### Medical AI Guidelines
- FDA Guidelines for AI/ML-Based Medical Devices
- ACR Guidelines for AI in Radiology
- WHO Ethics and Governance of AI for Health

### Technical References
- TensorFlow/Keras Documentation
- Streamlit Documentation
- Medical Image Processing Best Practices

### Datasets
- FracAtlas: Bone Fracture Classification Dataset
- ChestX-ray8: Chest Pathology Detection
- OAI: Osteoarthritis Initiative Dataset

## 📄 License

This project is developed for educational and research purposes. Please ensure compliance with local medical device regulations before clinical use.

## 👥 Contributors

- Development Team: Medical AI Specialists
- Clinical Advisors: Radiologists and Medical Professionals
- Testing: Medical Students and Practitioners

## 🔄 Version History

- **v1.0.0**: Initial release with core functionality
- **v1.1.0**: Added Grad-CAM visualization
- **v1.2.0**: Enhanced reporting system
- **v1.3.0**: Improved authentication and feedback

---

**⚠️ Important Disclaimer**: This system is designed for educational and research purposes only. Always consult qualified medical professionals for clinical decisions. The AI predictions should supplement, not replace, professional medical judgment.