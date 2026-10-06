# Medical X-ray AI Classifier - Project Setup & Running Guide

**Project:** Medical X-ray AI Classification System  
**Date:** October 8, 2025  
**Status:** Production Ready 🚀

---

## 🚀 Quick Start (For Immediate Use)

### **Prerequisites Check**
✅ Python 3.9+ installed  
✅ Virtual environment activated  
✅ Dependencies installed  
✅ Models trained and available  

### **1. Navigate to Project Directory**
```bash
cd D:\Capstone\mynew\capstoneortho
```

### **2. Activate Virtual Environment**
```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# OR Windows Command Prompt
.venv\Scripts\activate.bat

# OR Direct Python execution (if activated)
D:/Capstone/mynew/capstoneortho/.venv/Scripts/python.exe
```

### **3. Run the Application**
```bash
# Method 1: Direct Streamlit execution
streamlit run app.py

# Method 2: Python module execution (Recommended)
python -m streamlit run app.py

# Method 3: Full path execution
D:/Capstone/mynew/capstoneortho/.venv/Scripts/python.exe -m streamlit run app.py
```

### **4. Access the Application**
- **Local URL:** http://localhost:8503
- **Network URL:** http://192.168.29.181:8503

---

## 🔐 Login Information

### **Ready-to-Use Accounts**

| Role | Username | Password | Access Level |
|------|----------|----------|--------------|
| **Administrator** | `admin` | `admin2025` | Full system access |
| **Doctor** | `doctor` | `medical123` | Medical features |
| **Student** | `student` | `learn123` | Educational features |

### **Additional Test Accounts**
- `student11` / `learn12311`
- `farhaan` / `farhaan11`
- `doc` / `doc123`

---

## 📁 Project Structure

```
capstoneortho/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── user_data.json                 # User accounts database
├── .venv/                         # Virtual environment
├── models/                        # Trained AI models
│   ├── registry/                  # Model registry
│   ├── pneumonia/                 # Pneumonia detection models
│   ├── cardiomegaly/             # Cardiomegaly detection models
│   ├── bone_fracture/            # Bone fracture detection models
│   ├── arthritis/                # Arthritis detection models
│   └── osteoporosis/             # Osteoporosis detection models
├── utils/                         # Utility modules
│   ├── gradcam.py                # Grad-CAM visualization
│   ├── model_inference.py        # Model inference logic
│   ├── authentication.py         # User authentication
│   └── settings_manager.py       # Application settings
├── settings/                      # Configuration files
└── feedback_database.db          # User feedback storage
```

---

## 🔧 Detailed Setup (Fresh Installation)

### **Step 1: Environment Setup**

#### **Create Virtual Environment**
```bash
cd D:\Capstone\mynew\capstoneortho
python -m venv .venv
```

#### **Activate Environment**
```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows Command Prompt
.venv\Scripts\activate
```

#### **Verify Activation**
```bash
# Should show virtual environment path
where python
which python  # On Linux/Mac
```

### **Step 2: Install Dependencies**

#### **Install Requirements**
```bash
pip install -r requirements.txt
```

#### **Key Dependencies**
- `streamlit` - Web application framework
- `tensorflow` - Deep learning models
- `opencv-python` - Image processing
- `Pillow` - Image handling
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `scikit-learn` - Machine learning utilities

### **Step 3: Verify Model Files**

#### **Check Model Registry**
```bash
python -c "
import json
with open('models/registry/model_registry.json', 'r') as f:
    registry = json.load(f)
for condition, info in registry.items():
    if info.get('active_model'):
        print(f'✓ {condition}: {info[\"active_model\"][\"name\"]}')
"
```

#### **Expected Active Models**
1. ✓ pneumonia: pneumonia_fast_20251007_015119
2. ✓ cardiomegaly: cardiomegaly_fast_20251007_015119
3. ✓ bone_fracture: bone_fracture_new_intensive
4. ✓ arthritis: arthritis_new_intensive
5. ✓ osteoporosis: osteoporosis_new_intensive

### **Step 4: Test Model Loading**
```bash
python test_active_model_loading.py
```

**Expected Output:**
```
✓ Successfully loaded bone_fracture model
✓ Successfully loaded pneumonia model
✓ Successfully loaded arthritis model
✓ Successfully loaded osteoporosis model
✓ Successfully loaded cardiomegaly model
```

---

## 🚀 Running the Application

### **Standard Launch**
```bash
streamlit run app.py
```

### **With Environment Variables**
```bash
# Set TensorFlow legacy Keras (if needed)
$env:TF_USE_LEGACY_KERAS='1'
streamlit run app.py
```

### **Background Execution**
```bash
# Run in background (Windows)
Start-Process -WindowStyle Hidden streamlit run app.py

# Run in background (Linux/Mac)
nohup streamlit run app.py &
```

### **Custom Port**
```bash
streamlit run app.py --server.port 8504
```

---

## 🌐 Accessing the Application

### **Local Access**
- **URL:** http://localhost:8503
- **Description:** Access from the same machine

### **Network Access**
- **URL:** http://192.168.29.181:8503
- **Description:** Access from other devices on the same network

### **Firewall Configuration**
If external access needed:
```bash
# Windows Firewall
netsh advfirewall firewall add rule name="Streamlit" dir=in action=allow protocol=TCP localport=8503

# Linux iptables
sudo iptables -A INPUT -p tcp --dport 8503 -j ACCEPT
```

---

## 🔍 Application Features

### **Core Functionality**

#### **1. X-ray Classification**
- Upload medical X-ray images
- Classify across 5 conditions:
  - 🦴 Bone Fracture Detection (94.5% accuracy)
  - 🫁 Pneumonia Detection (95.75% accuracy)
  - ❤️ Cardiomegaly Detection (63.0% accuracy)
  - 🦵 Arthritis Detection (94.25% accuracy)
  - 🦴 Osteoporosis Detection (91.77% accuracy)

#### **2. Grad-CAM Visualization**
- Visual explanations of AI decisions
- User-adjustable intensity (0.0 - 1.0)
- Heatmap overlays showing important regions

#### **3. Role-Based Access**
- **Students:** Classification, Model Info, User Guide
- **Doctors:** Full features + Training, Management
- **Admins:** Complete system access

---

## 🛠 Troubleshooting

### **Common Issues & Solutions**

#### **1. Virtual Environment Issues**
```bash
# Problem: Command not found
# Solution: Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Problem: Wrong Python version
# Solution: Recreate virtual environment
Remove-Item -Recurse -Force .venv
python -m venv .venv
```

#### **2. Model Loading Errors**
```bash
# Problem: Model files not found
# Solution: Check model registry
python -c "
import os
print('Model files exist:')
print('Pneumonia:', os.path.exists('models/pneumonia'))
print('Bone Fracture:', os.path.exists('models/bone_fracture'))
"

# Problem: TensorFlow/Keras compatibility
# Solution: Set environment variable
$env:TF_USE_LEGACY_KERAS='1'
```

#### **3. Port Already in Use**
```bash
# Problem: Port 8503 busy
# Solution: Kill existing processes
taskkill /F /IM streamlit.exe

# Or use different port
streamlit run app.py --server.port 8504
```

#### **4. Import Errors**
```bash
# Problem: Module not found
# Solution: Install missing packages
pip install -r requirements.txt

# Check specific package
pip show streamlit
pip show tensorflow
```

### **Memory Issues**
```bash
# For large model files, increase memory
# Add to startup command:
streamlit run app.py --server.maxUploadSize 1000
```

---

## 📊 System Requirements

### **Minimum Requirements**
- **OS:** Windows 10/11, Linux, macOS
- **Python:** 3.9 or higher
- **RAM:** 8GB (16GB recommended)
- **Storage:** 5GB free space
- **Network:** Internet for initial setup

### **Recommended Specifications**
- **RAM:** 16GB+ for smooth operation
- **CPU:** Multi-core processor
- **GPU:** Optional (NVIDIA with CUDA for faster inference)
- **Storage:** SSD for better performance

---

## 🔄 Development Workflow

### **Starting Development**
```bash
# 1. Navigate to project
cd D:\Capstone\mynew\capstoneortho

# 2. Activate environment
.\.venv\Scripts\Activate.ps1

# 3. Run with auto-reload
streamlit run app.py --server.runOnSave true

# 4. Open in browser
# http://localhost:8503
```

### **Testing Changes**
```bash
# Test model loading
python test_active_model_loading.py

# Test all models
python test_all_models.py

# Verify integrations
python verify_integrated_models.py
```

### **Git Workflow**
```bash
# Check status
git status

# Add changes
git add .

# Commit changes
git commit -m "Description of changes"

# View history
git log --oneline
```

---

## 📝 Production Deployment

### **For Production Use**

#### **1. Server Configuration**
```bash
# Create production config
mkdir config
echo "
[server]
port = 8503
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
" > config/production.toml

# Run with config
streamlit run app.py --config config/production.toml
```

#### **2. Process Management**
```bash
# Using PM2 (Node.js process manager)
npm install -g pm2
pm2 start "streamlit run app.py" --name medical-ai

# Using systemd (Linux)
sudo nano /etc/systemd/system/medical-ai.service
```

#### **3. Reverse Proxy (Nginx)**
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8503;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## 📚 Additional Resources

### **Documentation Files**
- `GRADCAM_COMPATIBILITY_REPORT.md` - Grad-CAM technical details
- `STUDENT_MODEL_INFO_ACCESS_VERIFIED.md` - Access verification
- `NAVIGATION_FIX_COMPLETE.md` - Recent fixes documentation

### **Test Scripts**
- `test_gradcam_compatibility.py` - Grad-CAM testing
- `verify_student_model_info.py` - Student access verification
- `inspect_model_structure.py` - Model architecture analysis

### **Support**
- Check logs in Streamlit output
- Review error messages in browser console
- Test with different browsers if issues occur

---

## ✅ Success Checklist

Before considering the project "running":

- [ ] Virtual environment activated
- [ ] All dependencies installed
- [ ] 5 models loaded successfully
- [ ] Application accessible at http://localhost:8503
- [ ] Login works with test accounts
- [ ] X-ray classification functional
- [ ] Grad-CAM visualization working
- [ ] Model Information page accessible

---

**🎉 Your Medical X-ray AI Classifier is ready for use!**

**Quick Start Command:**
```bash
cd D:\Capstone\mynew\capstoneortho && .\.venv\Scripts\python.exe -m streamlit run app.py
```

**Access URL:** http://localhost:8503  
**Test Login:** student / learn123