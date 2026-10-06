# Settings Integration Helper Module
"""
Helper functions to integrate user settings across the Medical X-ray AI System
This module provides easy access to user settings from anywhere in the application
"""

import streamlit as st
from typing import Dict, Any, Optional

def get_current_settings() -> Dict[str, Any]:
    """Get current user settings with fallback to defaults"""
    try:
        if 'settings_manager' in st.session_state:
            return st.session_state.settings_manager.load_settings()
    except:
        pass
    
    # Return comprehensive default settings
    return {
        'model': {
            'confidence_threshold': 0.5,
            'batch_processing': False,
            'gradcam_intensity': 0.4,
            'auto_preprocessing': True,
            'enable_gpu': False,
            'cache_models': True
        },
        'reports': {
            'include_metadata': True,
            'include_preprocessing_info': False,
            'include_gradcam': True,
            'default_format': 'PDF',
            'auto_download': False,
            'compress_reports': True
        },
        'system': {
            'max_image_size_mb': 10,
            'dark_mode': False,
            'show_debug_info': False,
            'compact_layout': False
        },
        'privacy': {
            'auto_delete_images': True,
            'anonymous_feedback': True,
            'usage_analytics': False
        },
        'session': {
            'timeout': '30 minutes',
            'auto_save_settings': True,
            'remember_preferences': True
        }
    }

def get_model_settings() -> Dict[str, Any]:
    """Get model-specific settings"""
    settings = get_current_settings()
    return settings.get('model', {})

def get_report_settings() -> Dict[str, Any]:
    """Get report generation settings"""
    settings = get_current_settings()
    return settings.get('reports', {})

def get_system_settings() -> Dict[str, Any]:
    """Get system and interface settings"""
    settings = get_current_settings()
    return settings.get('system', {})

def get_privacy_settings() -> Dict[str, Any]:
    """Get privacy and security settings"""
    settings = get_current_settings()
    return settings.get('privacy', {})

def get_setting_value(category: str, key: str, default: Any = None) -> Any:
    """Get a specific setting value with fallback to default"""
    settings = get_current_settings()
    return settings.get(category, {}).get(key, default)

def should_include_metadata() -> bool:
    """Check if metadata should be included in reports"""
    return get_setting_value('reports', 'include_metadata', True)

def should_include_preprocessing_info() -> bool:
    """Check if preprocessing info should be included in reports"""
    return get_setting_value('reports', 'include_preprocessing_info', False)

def should_include_gradcam() -> bool:
    """Check if Grad-CAM visualizations should be included"""
    return get_setting_value('reports', 'include_gradcam', True)

def get_confidence_threshold() -> float:
    """Get the current confidence threshold for model predictions"""
    return get_setting_value('model', 'confidence_threshold', 0.5)

def get_gradcam_intensity() -> float:
    """Get the current Grad-CAM intensity setting"""
    return get_setting_value('model', 'gradcam_intensity', 0.4)

def get_show_boundaries() -> bool:
    """Get the current boundary detection setting for Grad-CAM"""
    return get_setting_value('model', 'show_boundaries', True)

def is_gpu_enabled() -> bool:
    """Check if GPU acceleration is enabled"""
    return get_setting_value('model', 'enable_gpu', False)

def is_model_caching_enabled() -> bool:
    """Check if model caching is enabled"""
    return get_setting_value('model', 'cache_models', True)

def should_auto_delete_images() -> bool:
    """Check if images should be automatically deleted after processing"""
    return get_setting_value('privacy', 'auto_delete_images', True)

def is_usage_analytics_enabled() -> bool:
    """Check if usage analytics collection is enabled"""
    return get_setting_value('privacy', 'usage_analytics', False)

def get_max_image_size_mb() -> int:
    """Get maximum allowed image size in MB"""
    return get_setting_value('system', 'max_image_size_mb', 10)

def is_debug_mode_enabled() -> bool:
    """Check if debug information should be displayed"""
    return get_setting_value('system', 'show_debug_info', False)

def get_default_report_format() -> str:
    """Get the default report format (PDF, HTML, or Both)"""
    return get_setting_value('reports', 'default_format', 'PDF')

def should_auto_download_reports() -> bool:
    """Check if reports should be automatically downloaded"""
    return get_setting_value('reports', 'auto_download', False)

def should_compress_reports() -> bool:
    """Check if large reports should be compressed"""
    return get_setting_value('reports', 'compress_reports', True)

def get_session_timeout() -> str:
    """Get the session timeout setting"""
    return get_setting_value('session', 'timeout', '30 minutes')

def is_dark_mode_enabled() -> bool:
    """Check if dark mode is enabled"""
    return get_setting_value('system', 'dark_mode', False)

def apply_gpu_settings():
    """Apply GPU settings to TensorFlow based on user preferences"""
    try:
        import tensorflow as tf
        
        if is_gpu_enabled():
            # Enable GPU memory growth to prevent allocation of all GPU memory
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    if is_debug_mode_enabled():
                        st.info(f"🚀 GPU acceleration enabled for {len(gpus)} GPU(s)")
                except RuntimeError as e:
                    if is_debug_mode_enabled():
                        st.warning(f"GPU setup error: {e}")
            else:
                if is_debug_mode_enabled():
                    st.warning("🖥️ No GPUs found, using CPU")
        else:
            # Force CPU usage
            tf.config.set_visible_devices([], 'GPU')
            if is_debug_mode_enabled():
                st.info("💻 Using CPU for model inference")
    except ImportError:
        pass  # TensorFlow not available

def get_preprocessing_settings() -> Dict[str, Any]:
    """Get preprocessing settings for image processing"""
    model_settings = get_model_settings()
    return {
        'auto_preprocessing': model_settings.get('auto_preprocessing', True),
        'max_image_size_mb': get_max_image_size_mb(),
        'resize_enabled': True,  # Always enabled for consistency
        'normalize_enabled': True,  # Always enabled for model compatibility
    }

def debug_log(message: str):
    """Log debug message if debug mode is enabled"""
    if is_debug_mode_enabled():
        st.write(f"🔍 **Debug**: {message}")

def show_performance_tips():
    """Show performance tips based on current settings"""
    if not is_model_caching_enabled():
        st.info("💡 **Tip**: Enable model caching in Settings for faster repeated classifications")
    
    if get_max_image_size_mb() > 25:
        st.info("⚡ **Tip**: Large image size limit may affect performance. Consider reducing in Settings")
    
    if is_gpu_enabled():
        try:
            import tensorflow as tf
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if not gpus:
                st.warning("⚠️ GPU acceleration is enabled but no GPUs detected. Check your CUDA installation")
        except ImportError:
            pass