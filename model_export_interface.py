# 📦 Model Export Interface for Streamlit App

import streamlit as st
import os
import json
from datetime import datetime
import zipfile
import base64

def model_export_page():
    """
    Model Export and Download Interface
    """
    st.title("📦 AI Model Export & Download")
    st.markdown("---")
    
    # Check if export directory exists
    export_dir = "exported_models"
    if not os.path.exists(export_dir):
        st.error("❌ Export directory not found. Please run the model export script first.")
        return
    
    # Load export documentation
    doc_path = os.path.join(export_dir, "MODEL_EXPORT_DOCUMENTATION.json")
    if os.path.exists(doc_path):
        try:
            with open(doc_path, 'r', encoding='utf-8') as f:
                export_doc = json.load(f)
        except Exception as e:
            st.error(f"❌ Could not load export documentation: {str(e)}")
            export_doc = {"models": []}
    else:
        export_doc = {"models": []}
    
    # Display export information
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📊 Total Models", len(export_doc.get("models", [])))
    
    with col2:
        if export_doc.get("export_date"):
            export_date = datetime.fromisoformat(export_doc["export_date"]).strftime("%Y-%m-%d")
            st.metric("📅 Export Date", export_date)
        else:
            st.metric("📅 Export Date", "Unknown")
    
    with col3:
        # Calculate total size
        total_size = 0
        for model in export_doc.get("models", []):
            total_size += model.get("file_size_mb", 0)
        st.metric("💾 Total Size", f"{total_size:.1f} MB")
    
    st.markdown("---")
    
    # Model categories
    st.subheader("🎯 Available Classification Models")
    
    # Organize models by performance
    medical_grade_models = []
    research_grade_models = []
    
    for model in export_doc.get("models", []):
        accuracy = float(model.get("accuracy", "0%").replace("%", ""))
        if accuracy >= 90:
            medical_grade_models.append(model)
        else:
            research_grade_models.append(model)
    
    # Medical Grade Models
    if medical_grade_models:
        st.markdown("### 🏅 Medical Grade Models (≥90% Accuracy)")
        
        for model in medical_grade_models:
            with st.expander(f"🩺 {model['classification_task']} - {model['accuracy']} Accuracy"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Description:** {model['description']}")
                    st.write(f"**Input Type:** {model['input_type']}")
                    st.write(f"**Medical Field:** {model['medical_condition']}")
                    st.write(f"**Classes:** {', '.join(model['classes'])}")
                    st.write(f"**Architecture:** {model['architecture']}")
                    st.write(f"**File Size:** {model['file_size_mb']} MB")
                
                with col2:
                    # Download buttons for both formats
                    h5_file = model['model_name']
                    keras_file = model['model_name'].replace('.h5', '.keras')
                    
                    # H5 format download
                    h5_path = os.path.join(export_dir, h5_file)
                    if os.path.exists(h5_path):
                        with open(h5_path, "rb") as f:
                            h5_bytes = f.read()
                        
                        st.download_button(
                            label="📥 Download .h5",
                            data=h5_bytes,
                            file_name=h5_file,
                            mime="application/octet-stream",
                            key=f"h5_{model['model_name']}"
                        )
                    
                    # Keras format download
                    keras_path = os.path.join(export_dir, keras_file)
                    if os.path.exists(keras_path):
                        with open(keras_path, "rb") as f:
                            keras_bytes = f.read()
                        
                        st.download_button(
                            label="📥 Download .keras",
                            data=keras_bytes,
                            file_name=keras_file,
                            mime="application/octet-stream",
                            key=f"keras_{model['model_name']}"
                        )
                
                # Usage example
                st.code(f"""
# Example usage for {model['classification_task']}
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model
model = tf.keras.models.load_model('{h5_file}')

# Preprocess image
def preprocess_xray(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

# Make prediction
processed_image = preprocess_xray('xray_image.jpg')
prediction = model.predict(processed_image)
class_index = np.argmax(prediction[0])
confidence = np.max(prediction[0])

classes = {model['classes']}
result = classes[class_index]
print(f"Prediction: {{result}} ({{confidence:.2%}} confidence)")
                """, language="python")
    
    # Research Grade Models
    if research_grade_models:
        st.markdown("### 🔬 Research Grade Models (<90% Accuracy)")
        
        for model in research_grade_models:
            with st.expander(f"🧪 {model['classification_task']} - {model['accuracy']} Accuracy"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Description:** {model['description']}")
                    st.write(f"**Input Type:** {model['input_type']}")
                    st.write(f"**Medical Field:** {model['medical_condition']}")
                    st.write(f"**Classes:** {', '.join(model['classes'])}")
                    st.write(f"**Architecture:** {model['architecture']}")
                    st.write(f"**File Size:** {model['file_size_mb']} MB")
                    st.warning("⚠️ This model is in research phase. Use for development and testing purposes.")
                
                with col2:
                    # Download buttons
                    h5_file = model['model_name']
                    keras_file = model['model_name'].replace('.h5', '.keras')
                    
                    h5_path = os.path.join(export_dir, h5_file)
                    if os.path.exists(h5_path):
                        with open(h5_path, "rb") as f:
                            h5_bytes = f.read()
                        
                        st.download_button(
                            label="📥 Download .h5",
                            data=h5_bytes,
                            file_name=h5_file,
                            mime="application/octet-stream",
                            key=f"h5_research_{model['model_name']}"
                        )
                    
                    keras_path = os.path.join(export_dir, keras_file)
                    if os.path.exists(keras_path):
                        with open(keras_path, "rb") as f:
                            keras_bytes = f.read()
                        
                        st.download_button(
                            label="📥 Download .keras",
                            data=keras_bytes,
                            file_name=keras_file,
                            mime="application/octet-stream",
                            key=f"keras_research_{model['model_name']}"
                        )
    
    st.markdown("---")
    
    # Bulk download options
    st.subheader("📦 Bulk Download Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Medical Grade Models Only
        if medical_grade_models:
            if st.button("📥 Download Medical Grade Models", key="medical_bulk"):
                zip_path = create_custom_zip(medical_grade_models, "Medical_Grade_Models")
                if zip_path and os.path.exists(zip_path):
                    with open(zip_path, "rb") as f:
                        zip_bytes = f.read()
                    
                    st.download_button(
                        label="📥 Medical Grade ZIP",
                        data=zip_bytes,
                        file_name="Medical_Grade_AI_Models.zip",
                        mime="application/zip",
                        key="medical_zip_download"
                    )
    
    with col2:
        # All Models
        zip_files = [f for f in os.listdir(export_dir) if f.endswith('.zip')]
        if zip_files:
            latest_zip = sorted(zip_files)[-1]  # Get the latest zip
            zip_path = os.path.join(export_dir, latest_zip)
            
            if os.path.exists(zip_path):
                with open(zip_path, "rb") as f:
                    zip_bytes = f.read()
                
                st.download_button(
                    label="📥 Download All Models",
                    data=zip_bytes,
                    file_name="Complete_Medical_AI_Models.zip",
                    mime="application/zip",
                    key="all_models_zip"
                )
    
    with col3:
        # Documentation
        readme_path = os.path.join(export_dir, "README.md")
        if os.path.exists(readme_path):
            with open(readme_path, "r", encoding='utf-8') as f:
                readme_content = f.read()
            
            st.download_button(
                label="📄 Download Documentation",
                data=readme_content,
                file_name="AI_Models_README.md",
                mime="text/markdown",
                key="readme_download"
            )
    
    # Usage instructions
    st.markdown("---")
    st.subheader("🛠️ System Requirements & Setup")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **System Requirements:**
        - Python 3.8 or higher
        - TensorFlow 2.10.0 or higher
        - NumPy, Pillow, OpenCV (for preprocessing)
        - 4GB RAM minimum
        - 200MB storage per model
        """)
    
    with col2:
        st.markdown("""
        **Quick Setup:**
        ```bash
        pip install tensorflow>=2.10.0
        pip install pillow numpy opencv-python
        ```
        
        **Load and Use:**
        ```python
        import tensorflow as tf
        model = tf.keras.models.load_model('model.h5')
        ```
        """)
    
    # Performance summary
    st.markdown("---")
    st.subheader("📊 Model Performance Summary")
    
    if export_doc.get("models"):
        performance_data = []
        for model in export_doc["models"]:
            accuracy = float(model.get("accuracy", "0%").replace("%", ""))
            performance_data.append({
                "Model": model["classification_task"],
                "Accuracy": f"{accuracy}%",
                "Status": "Medical Grade" if accuracy >= 90 else "Research Grade",
                "Medical Field": model.get("medical_condition", "Unknown"),
                "Input Type": model.get("input_type", "Unknown")
            })
        
        st.table(performance_data)
    
    # Support information
    st.markdown("---")
    st.subheader("💬 Support & Documentation")
    
    st.info("""
    **Need Help?**
    
    - **Model Usage:** Check the code examples provided with each model
    - **Integration:** Refer to the README.md file in the download
    - **Technical Issues:** Ensure your TensorFlow version is compatible
    - **Performance:** Medical Grade models (≥90%) are ready for clinical assistance
    - **Research Models:** Use for development and testing purposes
    
    **Model Formats:**
    - **.h5 format:** Standard Keras/TensorFlow format (recommended)
    - **.keras format:** New TensorFlow SavedModel format (future-proof)
    """)

def create_custom_zip(models, zip_name):
    """Create a custom zip file with selected models"""
    try:
        export_dir = "exported_models"
        zip_path = os.path.join(export_dir, f"{zip_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for model in models:
                h5_file = model['model_name']
                keras_file = model['model_name'].replace('.h5', '.keras')
                
                # Add H5 file
                h5_path = os.path.join(export_dir, h5_file)
                if os.path.exists(h5_path):
                    zipf.write(h5_path, h5_file)
                
                # Add Keras file
                keras_path = os.path.join(export_dir, keras_file)
                if os.path.exists(keras_path):
                    zipf.write(keras_path, keras_file)
            
            # Add documentation
            readme_path = os.path.join(export_dir, "README.md")
            if os.path.exists(readme_path):
                zipf.write(readme_path, "README.md")
        
        return zip_path
    except Exception as e:
        st.error(f"Error creating zip file: {str(e)}")
        return None

# Add this to your main Streamlit app
if __name__ == "__main__":
    model_export_page()