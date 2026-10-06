# 🎯 Model Activation Priority System Explained

## ❓ **Your Question:** "If I activate all versions of 1 model, which one will be used?"

---

## ✅ **Answer: Only ONE Model Can Be Active Per Dataset Type**

### 🔧 **How the System Works:**

#### **1. Single Active Model Rule**
- **Registry Structure:** Each dataset type (pneumonia, cardiomegaly, arthritis, etc.) can have only **ONE** active model at a time
- **Registry Format:**
```json
{
  "active_models": {
    "pneumonia": "model_id_123",      // Only one active ID
    "cardiomegaly": "model_id_456",   // Only one active ID
    "arthritis": "model_id_789",      // Only one active ID
    "osteoporosis": null,             // No active model
    "bone_fracture": "model_id_101"   // Only one active ID
  }
}
```

#### **2. Last Activation Wins**
When you activate multiple versions of the same model type:

**Example Scenario:**
1. You have 3 pneumonia models: `pneumonia_v1`, `pneumonia_v2`, `pneumonia_v3`
2. You activate them in this order:
   - Activate `pneumonia_v1` → **Active:** `pneumonia_v1`
   - Activate `pneumonia_v2` → **Active:** `pneumonia_v2` (replaces v1)
   - Activate `pneumonia_v3` → **Active:** `pneumonia_v3` (replaces v2)

**Result:** Only `pneumonia_v3` will be used for inference.

#### **3. Physical Model Replacement**
```python
# When activating a new model, the system:
def activate_model(self, model_id: str):
    # 1. Backs up current active model (if exists)
    current_active = registry['active_models'].get(dataset_type)
    if current_active:
        self._backup_active_model(dataset_type, current_active)
    
    # 2. Copies new model to active directory (overwrites previous)
    active_path = self.active_models_dir / f"{dataset_type}_model.h5"
    shutil.copy2(source_path, active_path)
    
    # 3. Updates registry to point to new model ID
    registry['active_models'][dataset_type] = model_id
```

---

## 🎯 **Practical Example**

### **Scenario:** You have multiple cardiomegaly models:
- `cardiomegaly_v1_standard` (60% accuracy)
- `cardiomegaly_v2_improved` (63% accuracy) 
- `cardiomegaly_v3_intensive` (65% accuracy)

### **Activation Sequence:**
1. **Activate v1:** System uses `cardiomegaly_v1_standard`
2. **Activate v2:** System uses `cardiomegaly_v2_improved` (v1 backed up)
3. **Activate v3:** System uses `cardiomegaly_v3_intensive` (v2 backed up)

### **What X-ray Classification Uses:**
- **Active Model:** `cardiomegaly_v3_intensive` (65% accuracy)
- **Physical Location:** `models/active/cardiomegaly_model.h5`
- **Registry Entry:** `"cardiomegaly": "cardiomegaly_v3_intensive_id"`

---

## 🔍 **How to Check Which Model is Currently Active**

### **Option 1: Model Management Interface**
- Go to **Model Management** → **Performance Comparison**
- Look for models marked as **🟢 Active**
- Only ONE model per condition will show as active

### **Option 2: Check Registry File**
- Location: `models/registry/model_registry.json`
- Look at `active_models` section:
```json
{
  "active_models": {
    "cardiomegaly": "cardiomegaly_densenet121_intensive_20251006_192404"
  }
}
```

### **Option 3: Check Active Directory**
- Physical files: `models/active/`
- Each condition has one file: `cardiomegaly_model.h5`, `pneumonia_model.h5`, etc.

---

## ⚠️ **Important Notes**

### **✅ Benefits of Single Active Model:**
1. **No Confusion:** Clear which model is being used for predictions
2. **Performance:** No overhead from loading multiple models
3. **Consistency:** All predictions use the same model version
4. **Memory Efficient:** Only one model per type loaded in memory

### **🔄 Model Switching:**
- **Instant:** Activation immediately switches active model
- **Backed Up:** Previous active model is safely stored in backups
- **Reversible:** Can reactivate any previous model version

### **💡 Best Practice:**
- **Test First:** Use less accurate model for testing
- **Activate Best:** Once satisfied, activate your highest accuracy model
- **Keep Versions:** All model versions remain available for future activation

---

## 🎯 **Summary**

**Answer:** If you activate multiple versions of the same model type, **the LAST activated model will be used**. The system maintains a "single active model per condition" policy, so each new activation replaces the previous one.

**Current System Status:**
- **Pneumonia:** Uses newest activated pneumonia model (95.8% accuracy)
- **Cardiomegaly:** Uses newest activated cardiomegaly model (63% accuracy)
- **Arthritis:** Uses newest activated arthritis model (94.2% accuracy)
- **Osteoporosis:** Uses newest activated osteoporosis model (91.8% accuracy)
- **Bone Fracture:** Uses newest activated bone fracture model (73% accuracy)

**Recommendation:** Activate your best performing model for each condition to ensure optimal medical diagnosis accuracy.