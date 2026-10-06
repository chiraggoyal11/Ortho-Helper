# Image Preprocessing Module for Medical X-ray AI System

import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import tensorflow as tf
import streamlit as st
from typing import Tuple, Union, Optional
import io

# Optional imports - these packages may not be installed
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

def load_dicom_image(dicom_file):
    """
    Load DICOM image file
    
    Args:
        dicom_file: DICOM file object
        
    Returns:
        PIL.Image: Converted PIL image
    """
    if not PYDICOM_AVAILABLE:
        st.error("DICOM support not available. Please install pydicom: pip install pydicom")
        return None
        
    try:
        # Read DICOM file
        dicom_data = pydicom.dcmread(dicom_file)
        
        # Get pixel array
        pixel_array = dicom_data.pixel_array
        
        # Normalize to 0-255 range
        pixel_array = ((pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
        
        # Convert to PIL Image
        if len(pixel_array.shape) == 2:  # Grayscale
            image = Image.fromarray(pixel_array, mode='L')
        else:  # RGB
            image = Image.fromarray(pixel_array)
        
        return image
    
    except Exception as e:
        st.error(f"Error loading DICOM file: {str(e)}")
        return None

def preprocess_image(image: Union[Image.Image, np.ndarray], 
                    resize: bool = True, 
                    normalize: bool = True,
                    target_size: Tuple[int, int] = (224, 224),
                    enhance_contrast: bool = False) -> np.ndarray:
    """
    Preprocess image for model inference
    
    Args:
        image: Input image (PIL Image or numpy array)
        resize: Whether to resize image
        normalize: Whether to normalize pixel values
        target_size: Target size for resizing
        enhance_contrast: Whether to enhance contrast
        
    Returns:
        np.ndarray: Preprocessed image array
    """
    try:
        # Convert to PIL Image if numpy array
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Convert to RGB if grayscale
        if image.mode == 'L':
            image = image.convert('RGB')
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Enhance contrast if requested
        if enhance_contrast:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)  # Increase contrast by 50%
        
        # Resize image
        if resize:
            image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        image_array = np.array(image, dtype=np.float32)
        
        # Normalize pixel values
        if normalize:
            image_array = image_array / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    
    except Exception as e:
        st.error(f"Error in image preprocessing: {str(e)}")
        return None

def apply_histogram_equalization(image: Image.Image) -> Image.Image:
    """
    Apply histogram equalization to improve contrast
    
    Args:
        image: Input PIL image
        
    Returns:
        PIL.Image: Enhanced image
    """
    try:
        # Convert to numpy array
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:  # RGB image
            # Convert to LAB color space
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            
            # Convert back to RGB
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:  # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(img_array)
        
        return Image.fromarray(enhanced)
    
    except Exception as e:
        st.error(f"Error in histogram equalization: {str(e)}")
        return image

def apply_denoising(image: Image.Image, method: str = 'gaussian') -> Image.Image:
    """
    Apply denoising to image
    
    Args:
        image: Input PIL image
        method: Denoising method ('gaussian', 'bilateral', 'median')
        
    Returns:
        PIL.Image: Denoised image
    """
    try:
        img_array = np.array(image)
        
        if method == 'gaussian':
            denoised = cv2.GaussianBlur(img_array, (5, 5), 0)
        elif method == 'bilateral':
            denoised = cv2.bilateralFilter(img_array, 9, 75, 75)
        elif method == 'median':
            denoised = cv2.medianBlur(img_array, 5)
        else:
            return image
        
        return Image.fromarray(denoised)
    
    except Exception as e:
        st.error(f"Error in denoising: {str(e)}")
        return image

def apply_augmentation(image_array: np.ndarray, 
                      augmentation_type: str = 'medical') -> np.ndarray:
    """
    Apply data augmentation to image
    
    Args:
        image_array: Input image array
        augmentation_type: Type of augmentation ('medical', 'light', 'heavy')
        
    Returns:
        np.ndarray: Augmented image array
    """
    if not ALBUMENTATIONS_AVAILABLE:
        st.info("Advanced augmentation not available. Using basic augmentation.")
        return _basic_augmentation(image_array)
        
    try:
        # Remove batch dimension for augmentation
        if len(image_array.shape) == 4:
            image_array = image_array[0]
        
        # Convert to uint8 for albumentations
        if image_array.max() <= 1.0:
            image_array = (image_array * 255).astype(np.uint8)
        
        # Define augmentation pipelines
        if augmentation_type == 'medical':
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, p=0.5),
                A.Brightness(limit=0.1, p=0.5),
                A.Contrast(limit=0.1, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            ])
        elif augmentation_type == 'light':
            transform = A.Compose([
                A.HorizontalFlip(p=0.3),
                A.Rotate(limit=5, p=0.3),
                A.Brightness(limit=0.05, p=0.3),
            ])
        elif augmentation_type == 'heavy':
            transform = A.Compose([
                A.HorizontalFlip(p=0.7),
                A.Rotate(limit=15, p=0.7),
                A.Brightness(limit=0.2, p=0.7),
                A.Contrast(limit=0.2, p=0.7),
                A.GaussNoise(var_limit=(10.0, 80.0), p=0.5),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
            ])
        else:
            return np.expand_dims(image_array / 255.0, axis=0)
        
        # Apply transformation
        augmented = transform(image=image_array)['image']
        
        # Normalize and add batch dimension
        augmented = augmented.astype(np.float32) / 255.0
        augmented = np.expand_dims(augmented, axis=0)
        
        return augmented
    
    except Exception as e:
        st.error(f"Error in augmentation: {str(e)}")
        return image_array

def _basic_augmentation(image_array: np.ndarray) -> np.ndarray:
    """
    Basic augmentation without albumentations library
    
    Args:
        image_array: Input image array
        
    Returns:
        np.ndarray: Augmented image array
    """
    try:
        # Remove batch dimension
        if len(image_array.shape) == 4:
            image_array = image_array[0]
        
        # Convert to PIL Image for basic augmentation
        if image_array.max() <= 1.0:
            image_array = (image_array * 255).astype(np.uint8)
        
        image = Image.fromarray(image_array)
        
        # Random horizontal flip
        if np.random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Random rotation
        if np.random.random() > 0.5:
            angle = np.random.uniform(-10, 10)
            image = image.rotate(angle)
        
        # Random brightness
        if np.random.random() > 0.5:
            enhancer = ImageEnhance.Brightness(image)
            factor = np.random.uniform(0.9, 1.1)
            image = enhancer.enhance(factor)
        
        # Convert back to array
        augmented = np.array(image, dtype=np.float32) / 255.0
        augmented = np.expand_dims(augmented, axis=0)
        
        return augmented
        
    except Exception as e:
        st.error(f"Error in basic augmentation: {str(e)}")
        return image_array

def create_preprocessing_pipeline(config: dict):
    """
    Create a preprocessing pipeline based on configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        function: Preprocessing pipeline function
    """
    def pipeline(image):
        # Load DICOM if needed
        if hasattr(image, 'read') and image.name.endswith('.dcm'):
            image = load_dicom_image(image)
        
        # Apply histogram equalization
        if config.get('histogram_equalization', False):
            image = apply_histogram_equalization(image)
        
        # Apply denoising
        if config.get('denoising', False):
            image = apply_denoising(image, config.get('denoising_method', 'gaussian'))
        
        # Standard preprocessing
        processed = preprocess_image(
            image,
            resize=config.get('resize', True),
            normalize=config.get('normalize', True),
            target_size=config.get('target_size', (224, 224)),
            enhance_contrast=config.get('enhance_contrast', False)
        )
        
        # Apply augmentation if specified
        if config.get('augmentation', False):
            processed = apply_augmentation(
                processed,
                config.get('augmentation_type', 'medical')
            )
        
        return processed
    
    return pipeline

def validate_image_quality(image: Image.Image) -> Tuple[bool, str]:
    """
    Validate image quality for medical analysis
    
    Args:
        image: Input PIL image
        
    Returns:
        tuple: (is_valid, message)
    """
    try:
        # Check image size
        width, height = image.size
        if width < 128 or height < 128:
            return False, "Image resolution too low (minimum 128x128)"
        
        # Check if image is too dark or too bright
        img_array = np.array(image.convert('L'))  # Convert to grayscale
        mean_brightness = np.mean(img_array)
        
        if mean_brightness < 20:
            return False, "Image is too dark - consider adjusting exposure"
        elif mean_brightness > 235:
            return False, "Image is too bright - consider reducing exposure"
        
        # Check contrast
        contrast = np.std(img_array)
        if contrast < 10:
            return False, "Image has very low contrast - may affect analysis accuracy"
        
        # Check for completely black or white regions
        black_pixels = np.sum(img_array < 5) / img_array.size
        white_pixels = np.sum(img_array > 250) / img_array.size
        
        if black_pixels > 0.3:
            return False, "Image has too many dark regions (>30%)"
        elif white_pixels > 0.3:
            return False, "Image has too many bright regions (>30%)"
        
        return True, "Image quality is acceptable"
    
    except Exception as e:
        return False, f"Error validating image: {str(e)}"

def get_image_statistics(image: Image.Image) -> dict:
    """
    Get comprehensive image statistics
    
    Args:
        image: Input PIL image
        
    Returns:
        dict: Image statistics
    """
    try:
        img_array = np.array(image.convert('L'))
        
        stats = {
            'dimensions': image.size,
            'mode': image.mode,
            'format': image.format,
            'mean_brightness': float(np.mean(img_array)),
            'std_brightness': float(np.std(img_array)),
            'min_value': int(np.min(img_array)),
            'max_value': int(np.max(img_array)),
            'contrast_ratio': float(np.std(img_array) / np.mean(img_array)) if np.mean(img_array) > 0 else 0,
            'dark_pixel_ratio': float(np.sum(img_array < 50) / img_array.size),
            'bright_pixel_ratio': float(np.sum(img_array > 200) / img_array.size)
        }
        
        return stats
    
    except Exception as e:
        st.error(f"Error calculating image statistics: {str(e)}")
        return {}

def batch_preprocess_images(images: list, config: dict) -> list:
    """
    Batch preprocess multiple images
    
    Args:
        images: List of images to process
        config: Preprocessing configuration
        
    Returns:
        list: List of preprocessed image arrays
    """
    pipeline = create_preprocessing_pipeline(config)
    processed_images = []
    
    progress_bar = st.progress(0)
    
    for i, image in enumerate(images):
        try:
            processed = pipeline(image)
            processed_images.append(processed)
            progress_bar.progress((i + 1) / len(images))
        except Exception as e:
            st.error(f"Error processing image {i+1}: {str(e)}")
            continue
    
    progress_bar.empty()
    return processed_images

# Preprocessing configurations for different model types
PREPROCESSING_CONFIGS = {
    'bone_fracture': {
        'resize': True,
        'target_size': (224, 224),
        'normalize': True,
        'histogram_equalization': True,
        'enhance_contrast': True,
        'denoising': False,
        'augmentation': False
    },
    'chest_conditions': {
        'resize': True,
        'target_size': (224, 224),
        'normalize': True,
        'histogram_equalization': False,
        'enhance_contrast': False,
        'denoising': True,
        'denoising_method': 'bilateral',
        'augmentation': False
    },
    'knee_conditions': {
        'resize': True,
        'target_size': (224, 224),
        'normalize': True,
        'histogram_equalization': True,
        'enhance_contrast': True,
        'denoising': True,
        'denoising_method': 'gaussian',
        'augmentation': False
    }
}

# Example usage
if __name__ == "__main__":
    print("Image preprocessing module loaded successfully!")
    print(f"Available preprocessing configs: {list(PREPROCESSING_CONFIGS.keys())}")