import cv2
import numpy as np
from PIL import Image
import io
from typing import Tuple, Optional


class SketchGenerator:
    """
    Generate sketches from photos using various techniques
    """
    
    @staticmethod
    def image_to_sketch_pencil(image_array: np.ndarray, blur_sigma: int = 5) -> np.ndarray:
        """
        Convert photo to pencil sketch using OpenCV
        Fast, no GAN required
        
        Args:
            image_array: Input image as numpy array (BGR)
            blur_sigma: Blur amount (higher = softer sketch)
        
        Returns:
            Sketch as numpy array (grayscale)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        
        # Invert the image
        inverted = 255 - gray
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(inverted, (21, 21), blur_sigma)
        
        # Invert blurred image
        inverted_blur = 255 - blurred
        
        # Create sketch by dividing
        sketch = cv2.divide(gray, inverted_blur, scale=256.0)
        
        return sketch
    
    @staticmethod
    def image_to_sketch_edge(image_array: np.ndarray, 
                             low_threshold: int = 50, 
                             high_threshold: int = 150) -> np.ndarray:
        """
        Convert photo to sketch using edge detection (Canny)
        Creates sharper, more defined sketches
        
        Args:
            image_array: Input image as numpy array (BGR)
            low_threshold: Lower threshold for edge detection
            high_threshold: Upper threshold for edge detection
        
        Returns:
            Sketch as numpy array (grayscale)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        
        # Reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        
        # Invert (white background, black lines)
        sketch = 255 - edges
        
        return sketch
    
    @staticmethod
    def image_to_sketch_adaptive(image_array: np.ndarray) -> np.ndarray:
        """
        Adaptive sketch generation combining multiple techniques
        Best quality without GAN
        
        Args:
            image_array: Input image as numpy array (BGR)
        
        Returns:
            Sketch as numpy array (grayscale)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Edge detection with multiple scales
        edges1 = cv2.Canny(enhanced, 50, 150)
        edges2 = cv2.Canny(enhanced, 100, 200)
        
        # Combine edges
        edges = cv2.addWeighted(edges1, 0.5, edges2, 0.5, 0)
        
        # Morphological operations to connect edges
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Invert
        sketch = 255 - edges
        
        return sketch
    
    @staticmethod
    def enhance_sketch(sketch: np.ndarray, 
                       sharpen: bool = True,
                       denoise: bool = True) -> np.ndarray:
        """
        Post-process sketch for better quality
        
        Args:
            sketch: Input sketch (grayscale)
            sharpen: Apply sharpening
            denoise: Apply denoising
        
        Returns:
            Enhanced sketch
        """
        result = sketch.copy()
        
        if denoise:
            result = cv2.fastNlMeansDenoising(result, None, 10, 7, 21)
        
        if sharpen:
            kernel = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]])
            result = cv2.filter2D(result, -1, kernel)
        
        return result
    
    @staticmethod
    def preprocess_blurry_image(image_array: np.ndarray) -> np.ndarray:
        """
        Enhanced deblurring for blurry images using multiple techniques
        
        Args:
            image_array: Input image (BGR)
        
        Returns:
            Enhanced image (BGR)
        """
        # Method 1: Unsharp masking for better edge enhancement
        gaussian = cv2.GaussianBlur(image_array, (0, 0), 3.0)
        unsharp_image = cv2.addWeighted(image_array, 1.5, gaussian, -0.5, 0)
        
        # Method 2: Advanced sharpening kernel
        kernel_sharpen = np.array([
            [-1, -1, -1, -1, -1],
            [-1,  2,  2,  2, -1],
            [-1,  2,  8,  2, -1],
            [-1,  2,  2,  2, -1],
            [-1, -1, -1, -1, -1]
        ]) / 8.0
        sharpened = cv2.filter2D(unsharp_image, -1, kernel_sharpen)
        
        # Method 3: Bilateral filter to preserve edges while reducing noise
        bilateral = cv2.bilateralFilter(sharpened, 9, 75, 75)
        
        # Method 4: Enhance contrast using CLAHE on each channel
        lab = cv2.cvtColor(bilateral, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Method 5: Edge enhancement using high-pass filter
        lowpass = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        highpass = cv2.subtract(enhanced, lowpass)
        enhanced = cv2.add(enhanced, highpass)
        
        # Method 6: Detail enhancement
        detail = cv2.detailEnhance(enhanced, sigma_s=10, sigma_r=0.15)
        
        return detail
    
    @staticmethod
    def super_resolution_enhance(image_array: np.ndarray) -> np.ndarray:
        """
        Enhanced upscaling and detail recovery for low-quality images
        Uses advanced interpolation techniques
        
        Args:
            image_array: Input image (BGR)
        
        Returns:
            Enhanced higher resolution image (BGR)
        """
        # Get current dimensions
        h, w = image_array.shape[:2]
        
        # If image is small, upscale it first
        if h < 512 or w < 512:
            scale_factor = max(512 / h, 512 / w)
            new_h = int(h * scale_factor)
            new_w = int(w * scale_factor)
            
            # Use INTER_CUBIC for upscaling (better quality than linear)
            upscaled = cv2.resize(image_array, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        else:
            upscaled = image_array.copy()
        
        # Denoise while preserving details
        denoised = cv2.fastNlMeansDenoisingColored(upscaled, None, 10, 10, 7, 21)
        
        # Enhance sharpness
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        # Blend original and sharpened (to avoid over-sharpening)
        enhanced = cv2.addWeighted(denoised, 0.6, sharpened, 0.4, 0)
        
        return enhanced


class FaceFeatureComposer:
    """
    Build faces from individual features
    """
    
    # Feature positions (relative to 512x512 canvas)
    # Keys use plural form to match feature template names
    FEATURE_POSITIONS = {
        'eyes': (256, 210),      # Center X, Y position
        'noses': (256, 300),
        'mouths': (256, 380),
        'eyebrows': (256, 165),
        'face_shapes': (200, 256),
        'ears': (200, 230),
        'hair': (180, 80),
    }
    
    # Target sizes for each feature type on the 512x512 canvas
    FEATURE_SIZES = {
        'eyes': (300, 80),       # width, height  (pair)
        'noses': (100, 140),
        'mouths': (200, 70),
        'eyebrows': (300, 50),   # pair
        'face_shapes': (400, 512),
        'ears': (400, 120),      # pair, wide to sit on face sides
        'hair': (380, 180),
    }
    
    @staticmethod
    def create_blank_canvas(size: Tuple[int, int] = (512, 512)) -> np.ndarray:
        """Create blank white canvas"""
        return np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
    
    @staticmethod
    def load_feature_library() -> dict:
        """
        Load pre-made facial feature templates
        In production, these would be stored in S3 or local files
        For now, we'll generate basic shapes
        """
        features = {
            'face_shapes': [],
            'eyes': [],
            'noses': [],
            'mouths': [],
            'eyebrows': [],
            'hairstyles': [],
        }
        
        # TODO: Load actual feature images from storage
        # This is a placeholder structure
        
        return features
    
    @staticmethod
    def compose_face(selected_features: dict, 
                     canvas_size: Tuple[int, int] = (512, 512),
                     custom_positions: dict = None) -> np.ndarray:
        """
        Compose face from selected features
        
        Args:
            selected_features: Dict with keys like 'eyes', 'noses', 'mouths'
                              Values are feature image arrays (already loaded)
            canvas_size: Output image size
            custom_positions: Optional dict of {feature_type: {x, y}} for user-placed positions
        
        Returns:
            Composed face as numpy array
        """
        canvas = FaceFeatureComposer.create_blank_canvas(canvas_size)
        
        # Layer order (back to front) - use plural keys to match feature template names
        layer_order = ['face_shapes', 'hair', 'ears', 'eyebrows', 'eyes', 'noses', 'mouths']
        
        for feature_type in layer_order:
            if feature_type not in selected_features:
                continue
            
            feature_img = selected_features[feature_type]
            
            if feature_img is None:
                continue
            
            # If it's a file path, load it
            if isinstance(feature_img, str):
                feature_img = cv2.imread(feature_img, cv2.IMREAD_UNCHANGED)
            
            if feature_img is None:
                continue
            
            # Get custom position if provided
            pos_override = None
            if custom_positions and feature_type in custom_positions:
                p = custom_positions[feature_type]
                pos_override = (int(p.get('x', 256)), int(p.get('y', 256)))
            
            # Position and blend
            canvas = FaceFeatureComposer._blend_feature(
                canvas, 
                feature_img, 
                feature_type,
                pos_override=pos_override
            )
        
        return canvas
    
    @staticmethod
    def _blend_feature(canvas: np.ndarray, 
                       feature: np.ndarray, 
                       feature_type: str,
                       pos_override: tuple = None) -> np.ndarray:
        """
        Blend a feature onto the canvas at the correct position with alpha blending
        """
        if feature is None:
            return canvas
        
        pos = pos_override or FaceFeatureComposer.FEATURE_POSITIONS.get(feature_type, (256, 256))
        
        # Scale feature to target size for the canvas
        target_size = FaceFeatureComposer.FEATURE_SIZES.get(feature_type)
        if target_size is not None:
            feature = cv2.resize(feature, target_size, interpolation=cv2.INTER_AREA)
        
        # Check if feature has alpha channel
        has_alpha = feature.shape[2] == 4 if len(feature.shape) == 3 else False
        
        # Get feature dimensions
        h, w = feature.shape[:2]
        ch, cw = canvas.shape[:2]
        
        # Top-left corner of feature on canvas
        canvas_x = pos[0] - w // 2
        canvas_y = pos[1] - h // 2
        
        # Clamp to canvas bounds
        x1 = max(0, canvas_x)
        y1 = max(0, canvas_y)
        x2 = min(cw, canvas_x + w)
        y2 = min(ch, canvas_y + h)
        
        if x2 <= x1 or y2 <= y1:
            return canvas
        
        # Corresponding region in the feature image
        feat_x1 = x1 - canvas_x
        feat_y1 = y1 - canvas_y
        feat_x2 = feat_x1 + (x2 - x1)
        feat_y2 = feat_y1 + (y2 - y1)
        
        # Extract the region of interest
        roi = canvas[y1:y2, x1:x2]
        feature_region = feature[feat_y1:feat_y2, feat_x1:feat_x2]
        
        if has_alpha:
            # Alpha blending
            alpha = feature_region[:, :, 3:4] / 255.0
            feature_rgb = feature_region[:, :, :3]
            
            # Blend
            blended = (alpha * feature_rgb + (1 - alpha) * roi).astype(np.uint8)
            canvas[y1:y2, x1:x2] = blended
        else:
            # Simple overlay for non-transparent features
            # Only copy non-white pixels (assuming white background)
            if len(feature_region.shape) == 3:
                gray = cv2.cvtColor(feature_region, cv2.COLOR_BGR2GRAY)
            else:
                gray = feature_region
            
            # Create mask: copy where feature is not white
            mask = gray < 250
            mask_3ch = np.stack([mask] * 3, axis=-1)
            
            canvas[y1:y2, x1:x2] = np.where(mask_3ch, feature_region, roi)
        
        return canvas


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV format"""
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    """Convert OpenCV image to PIL format"""
    if len(cv2_image.shape) == 2:  # Grayscale
        return Image.fromarray(cv2_image, mode='L')
    else:  # Color
        return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))


def convert_sketch_to_3channel(sketch: np.ndarray) -> np.ndarray:
    """
    Convert grayscale sketch to 3-channel for face recognition
    Face recognition models expect RGB/BGR images
    """
    if len(sketch.shape) == 2:
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    return sketch