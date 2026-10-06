#!/usr/bin/env python3
"""
Advanced Preprocessing Controls Module
Complete implementation for medical image preprocessing with advanced controls
"""

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, Optional
from datetime import datetime
from utils.settings_integration import get_preprocessing_settings, is_debug_mode_enabled, debug_log

class AdvancedPreprocessor:
    """Advanced image preprocessing with medical-specific optimizations"""
    
    def __init__(self):
        self.preprocessing_history = []
        
    def apply_clahe(self, image: np.ndarray, clip_limit: float = 2.0, grid_size: int = 8) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization"""
        debug_log("Applying CLAHE for contrast enhancement")
        
        if len(image.shape) == 3:
            # Convert to LAB color space for better results
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l_channel, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
            l_channel = clahe.apply(l_channel)
            
            # Merge channels back
            lab = cv2.merge((l_channel, a, b))
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
            enhanced = clahe.apply(image)
            
        self.preprocessing_history.append(f"CLAHE (clip_limit={clip_limit}, grid_size={grid_size})")
        return enhanced
    
    def apply_unsharp_masking(self, image: np.ndarray, amount: float = 1.5, radius: float = 1.0) -> np.ndarray:
        """Apply unsharp masking for edge enhancement"""
        debug_log("Applying unsharp masking for edge enhancement")
        
        if len(image.shape) == 3:
            # Process each channel separately
            enhanced = np.zeros_like(image)
            for i in range(image.shape[2]):
                blurred = cv2.GaussianBlur(image[:,:,i], (0, 0), radius)
                enhanced[:,:,i] = cv2.addWeighted(image[:,:,i], 1 + amount, blurred, -amount, 0)
        else:
            # Grayscale image
            blurred = cv2.GaussianBlur(image, (0, 0), radius)
            enhanced = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
        
        self.preprocessing_history.append(f"Unsharp Mask (amount={amount}, radius={radius})")
        return enhanced
    
    def apply_noise_reduction(self, image: np.ndarray, h: float = 10.0, template_window: int = 7, search_window: int = 21) -> np.ndarray:
        """Apply Non-Local Means denoising"""
        debug_log("Applying noise reduction")
        
        if len(image.shape) == 3:
            denoised = cv2.fastNlMeansDenoisingColored(image, None, h, h, template_window, search_window)
        else:
            denoised = cv2.fastNlMeansDenoising(image, None, h, template_window, search_window)
        
        self.preprocessing_history.append(f"Noise Reduction (h={h})")
        return denoised
    
    def apply_edge_enhancement(self, image: np.ndarray, method: str = "laplacian", strength: float = 0.3) -> np.ndarray:
        """Apply edge enhancement using various methods"""
        debug_log(f"Applying edge enhancement using {method}")
        
        if method == "laplacian":
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian = np.uint8(np.absolute(laplacian))
            
            if len(image.shape) == 3:
                laplacian = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2RGB)
            
            enhanced = cv2.addWeighted(image, 1.0, laplacian, strength, 0)
            
        elif method == "sobel":
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel = np.sqrt(sobelx**2 + sobely**2)
            sobel = np.uint8(sobel / sobel.max() * 255)
            
            if len(image.shape) == 3:
                sobel = cv2.cvtColor(sobel, cv2.COLOR_GRAY2RGB)
            
            enhanced = cv2.addWeighted(image, 1.0, sobel, strength, 0)
            
        else:  # Default to simple sharpening
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * strength
            enhanced = cv2.filter2D(image, -1, kernel)
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        self.preprocessing_history.append(f"Edge Enhancement ({method}, strength={strength})")
        return enhanced
    
    def apply_histogram_equalization(self, image: np.ndarray, method: str = "standard") -> np.ndarray:
        """Apply histogram equalization"""
        debug_log(f"Applying histogram equalization using {method}")
        
        if method == "adaptive":
            return self.apply_clahe(image)
        
        if len(image.shape) == 3:
            # Convert to YUV and equalize Y channel
            yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
            enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
        else:
            enhanced = cv2.equalizeHist(image)
        
        self.preprocessing_history.append(f"Histogram Equalization ({method})")
        return enhanced
    
    def apply_gamma_correction(self, image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """Apply gamma correction"""
        debug_log(f"Applying gamma correction (gamma={gamma})")
        
        # Build lookup table
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        
        # Apply gamma correction using the lookup table
        enhanced = cv2.LUT(image, table)
        
        self.preprocessing_history.append(f"Gamma Correction (gamma={gamma})")
        return enhanced
    
    def apply_morphological_operations(self, image: np.ndarray, operation: str = "opening", kernel_size: int = 3) -> np.ndarray:
        """Apply morphological operations"""
        debug_log(f"Applying morphological {operation}")
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        if operation == "opening":
            processed = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        elif operation == "closing":
            processed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        elif operation == "gradient":
            processed = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        elif operation == "tophat":
            processed = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        elif operation == "blackhat":
            processed = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        else:
            processed = gray
        
        if len(image.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
        
        self.preprocessing_history.append(f"Morphological {operation} (kernel_size={kernel_size})")
        return processed
    
    def get_preprocessing_history(self) -> list:
        """Get list of applied preprocessing steps"""
        return self.preprocessing_history.copy()
    
    def clear_history(self):
        """Clear preprocessing history"""
        self.preprocessing_history = []

def show_advanced_preprocessing_controls(image: np.ndarray, user_role: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Display advanced preprocessing controls for medical professionals"""
    
    if user_role not in ['doctor', 'radiologist']:
        return image, {}
    
    st.markdown("### 🔬 **Advanced Preprocessing Controls**")
    st.info("🏥 **Medical Professional Mode**: Fine-tune preprocessing for optimal diagnostic accuracy")
    
    preprocessor = AdvancedPreprocessor()
    processed_image = image.copy()
    applied_operations = {}
    
    # Create tabs for different preprocessing categories
    contrast_tab, enhancement_tab, noise_tab, morphology_tab, custom_tab = st.tabs([
        "🎨 Contrast", "✨ Enhancement", "🔇 Noise Reduction", "🔄 Morphology", "⚙️ Custom"
    ])
    
    # Contrast Enhancement Tab
    with contrast_tab:
        st.markdown("**Contrast and Brightness Controls**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            apply_clahe = st.checkbox("Apply CLAHE Enhancement", value=False, key="clahe_enabled")
            if apply_clahe:
                clip_limit = st.slider("CLAHE Clip Limit", 1.0, 5.0, 2.0, 0.1, key="clahe_clip")
                grid_size = st.slider("CLAHE Grid Size", 4, 16, 8, 1, key="clahe_grid")
                processed_image = preprocessor.apply_clahe(processed_image, clip_limit, grid_size)
                applied_operations["clahe"] = {"clip_limit": clip_limit, "grid_size": grid_size}
        
        with col2:
            apply_histogram = st.checkbox("Apply Histogram Equalization", value=False, key="hist_enabled")
            if apply_histogram:
                hist_method = st.selectbox("Method", ["standard", "adaptive"], key="hist_method")
                processed_image = preprocessor.apply_histogram_equalization(processed_image, hist_method)
                applied_operations["histogram"] = {"method": hist_method}
        
        # Gamma Correction
        apply_gamma = st.checkbox("Apply Gamma Correction", value=False, key="gamma_enabled")
        if apply_gamma:
            gamma_value = st.slider("Gamma Value", 0.5, 2.5, 1.0, 0.1, key="gamma_value")
            processed_image = preprocessor.apply_gamma_correction(processed_image, gamma_value)
            applied_operations["gamma"] = {"gamma": gamma_value}
    
    # Enhancement Tab
    with enhancement_tab:
        st.markdown("**Edge and Detail Enhancement**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            apply_unsharp = st.checkbox("Apply Unsharp Masking", value=False, key="unsharp_enabled")
            if apply_unsharp:
                unsharp_amount = st.slider("Amount", 0.5, 3.0, 1.5, 0.1, key="unsharp_amount")
                unsharp_radius = st.slider("Radius", 0.5, 3.0, 1.0, 0.1, key="unsharp_radius")
                processed_image = preprocessor.apply_unsharp_masking(processed_image, unsharp_amount, unsharp_radius)
                applied_operations["unsharp"] = {"amount": unsharp_amount, "radius": unsharp_radius}
        
        with col2:
            apply_edge = st.checkbox("Apply Edge Enhancement", value=False, key="edge_enabled")
            if apply_edge:
                edge_method = st.selectbox("Method", ["laplacian", "sobel", "simple"], key="edge_method")
                edge_strength = st.slider("Strength", 0.1, 1.0, 0.3, 0.05, key="edge_strength")
                processed_image = preprocessor.apply_edge_enhancement(processed_image, edge_method, edge_strength)
                applied_operations["edge"] = {"method": edge_method, "strength": edge_strength}
    
    # Noise Reduction Tab
    with noise_tab:
        st.markdown("**Noise Reduction and Smoothing**")
        
        apply_denoise = st.checkbox("Apply Noise Reduction", value=False, key="denoise_enabled")
        if apply_denoise:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                denoise_h = st.slider("Filter Strength", 5.0, 20.0, 10.0, 1.0, key="denoise_h")
            with col2:
                template_window = st.slider("Template Window", 3, 15, 7, 2, key="template_window")
            with col3:
                search_window = st.slider("Search Window", 15, 35, 21, 2, key="search_window")
            
            processed_image = preprocessor.apply_noise_reduction(processed_image, denoise_h, template_window, search_window)
            applied_operations["denoise"] = {
                "h": denoise_h,
                "template_window": template_window, 
                "search_window": search_window
            }
    
    # Morphological Operations Tab
    with morphology_tab:
        st.markdown("**Morphological Operations**")
        
        apply_morphology = st.checkbox("Apply Morphological Operations", value=False, key="morphology_enabled")
        if apply_morphology:
            col1, col2 = st.columns(2)
            
            with col1:
                morph_operation = st.selectbox(
                    "Operation",
                    ["opening", "closing", "gradient", "tophat", "blackhat"],
                    key="morph_operation"
                )
            
            with col2:
                kernel_size = st.slider("Kernel Size", 3, 15, 3, 2, key="kernel_size")
            
            processed_image = preprocessor.apply_morphological_operations(processed_image, morph_operation, kernel_size)
            applied_operations["morphology"] = {"operation": morph_operation, "kernel_size": kernel_size}
    
    # Custom Preprocessing Tab
    with custom_tab:
        st.markdown("**Custom Preprocessing Pipeline**")
        
        # Brightness and Contrast
        col1, col2 = st.columns(2)
        
        with col1:
            brightness = st.slider("Brightness", -50, 50, 0, 1, key="brightness")
            contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1, key="contrast")
        
        with col2:
            saturation = st.slider("Saturation", 0.0, 2.0, 1.0, 0.1, key="saturation")
            sharpness = st.slider("Sharpness", 0.0, 2.0, 1.0, 0.1, key="sharpness")
        
        # Apply custom adjustments
        if brightness != 0 or contrast != 1.0 or saturation != 1.0 or sharpness != 1.0:
            # Convert to PIL for easier manipulation
            pil_image = Image.fromarray(processed_image.astype('uint8'))
            
            # Apply brightness and contrast
            if brightness != 0:
                enhancer = ImageEnhance.Brightness(pil_image)
                pil_image = enhancer.enhance(1 + brightness/50.0)
            
            if contrast != 1.0:
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(contrast)
            
            if saturation != 1.0:
                enhancer = ImageEnhance.Color(pil_image)
                pil_image = enhancer.enhance(saturation)
            
            if sharpness != 1.0:
                enhancer = ImageEnhance.Sharpness(pil_image)
                pil_image = enhancer.enhance(sharpness)
            
            processed_image = np.array(pil_image)
            
            applied_operations["custom"] = {
                "brightness": brightness,
                "contrast": contrast,
                "saturation": saturation,
                "sharpness": sharpness
            }
    
    # Show preprocessing comparison if any operations were applied
    if applied_operations:
        st.markdown("### 📊 **Preprocessing Comparison**")
        
        comp_col1, comp_col2 = st.columns(2)
        
        with comp_col1:
            st.markdown("**Original Image**")
            st.image(image, use_column_width=True)
        
        with comp_col2:
            st.markdown("**Processed Image**")
            st.image(processed_image, use_column_width=True)
        
        # Show applied operations
        st.markdown("**Applied Operations:**")
        history = preprocessor.get_preprocessing_history()
        for i, operation in enumerate(history, 1):
            st.markdown(f"{i}. {operation}")
        
        # Export preprocessing settings
        if st.button("💾 Save Preprocessing Preset", key="save_preset"):
            preset_name = st.text_input("Preset Name", value="Custom Preset", key="preset_name")
            save_preprocessing_preset(preset_name, applied_operations)
            st.success(f"✅ Preset '{preset_name}' saved successfully!")
    
    return processed_image, applied_operations

def save_preprocessing_preset(name: str, operations: Dict[str, Any]):
    """Save preprocessing preset to file"""
    import json
    from pathlib import Path
    
    presets_dir = Path("preprocessing_presets")
    presets_dir.mkdir(exist_ok=True)
    
    preset_file = presets_dir / f"{name.replace(' ', '_').lower()}.json"
    
    preset_data = {
        "name": name,
        "operations": operations,
        "created_date": str(datetime.now()),
        "version": "1.0"
    }
    
    with open(preset_file, 'w') as f:
        json.dump(preset_data, f, indent=4)

def load_preprocessing_presets() -> Dict[str, Dict[str, Any]]:
    """Load saved preprocessing presets"""
    import json
    from pathlib import Path
    
    presets_dir = Path("preprocessing_presets")
    if not presets_dir.exists():
        return {}
    
    presets = {}
    for preset_file in presets_dir.glob("*.json"):
        try:
            with open(preset_file, 'r') as f:
                preset_data = json.load(f)
                presets[preset_data["name"]] = preset_data["operations"]
        except Exception as e:
            debug_log(f"Error loading preset {preset_file}: {e}")
    
    return presets

def get_medical_preprocessing_recommendations(image: np.ndarray) -> Dict[str, Any]:
    """Analyze image and provide preprocessing recommendations"""
    
    # Calculate image statistics
    mean_brightness = np.mean(image)
    std_brightness = np.std(image)
    
    # Check if image is too dark or too bright
    recommendations = {}
    
    if mean_brightness < 50:
        recommendations["brightness"] = "Image appears dark - consider gamma correction or CLAHE"
    elif mean_brightness > 200:
        recommendations["brightness"] = "Image appears bright - consider reducing gamma"
    
    # Check contrast
    if std_brightness < 30:
        recommendations["contrast"] = "Low contrast detected - consider CLAHE or histogram equalization"
    
    # Check for potential noise (high frequency content)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    if laplacian_var < 100:
        recommendations["sharpness"] = "Image may be blurry - consider unsharp masking"
    elif laplacian_var > 1000:
        recommendations["noise"] = "High noise detected - consider denoising"
    
    return recommendations