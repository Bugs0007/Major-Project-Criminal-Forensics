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
        Enhance blurry images before sketch conversion
        
        Args:
            image_array: Input image (BGR)
        
        Returns:
            Enhanced image (BGR)
        """
        # Deblur using Wiener filter approximation
        kernel = np.ones((5, 5), np.float32) / 25
        deblurred = cv2.filter2D(image_array, -1, kernel)
        
        # Sharpen
        kernel_sharpen = np.array([[-1, -1, -1],
                                   [-1,  9, -1],
                                   [-1, -1, -1]])
        sharpened = cv2.filter2D(deblurred, -1, kernel_sharpen)
        
        # Enhance contrast
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced


class FaceFeatureComposer:
    """
    Build faces from individual features
    """
    
    # Feature positions (relative to 512x512 canvas)
    FEATURE_POSITIONS = {
        'eyes': (256, 180),      # Center X, Y position
        'nose': (256, 280),
        'mouth': (256, 360),
        'eyebrows': (256, 150),
        'face_shape': (256, 256),
        'hair': (256, 100),
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
                     canvas_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
        """
        Compose face from selected features
        
        Args:
            selected_features: Dict with keys like 'eyes', 'nose', 'mouth'
                              Values are feature image paths or arrays
            canvas_size: Output image size
        
        Returns:
            Composed face as numpy array
        """
        canvas = FaceFeatureComposer.create_blank_canvas(canvas_size)
        
        # Layer order (back to front)
        layer_order = ['face_shape', 'hair', 'eyebrows', 'eyes', 'nose', 'mouth']
        
        for feature_type in layer_order:
            if feature_type not in selected_features:
                continue
            
            feature_img = selected_features[feature_type]
            
            if feature_img is None:
                continue
            
            # If it's a file path, load it
            if isinstance(feature_img, str):
                feature_img = cv2.imread(feature_img, cv2.IMREAD_UNCHANGED)
            
            # Position and blend
            canvas = FaceFeatureComposer._blend_feature(
                canvas, 
                feature_img, 
                feature_type
            )
        
        return canvas
    
    @staticmethod
    def _blend_feature(canvas: np.ndarray, 
                       feature: np.ndarray, 
                       feature_type: str) -> np.ndarray:
        """
        Blend a feature onto the canvas at the correct position
        """
        if feature is None:
            return canvas
        
        pos = FaceFeatureComposer.FEATURE_POSITIONS.get(feature_type, (256, 256))
        
        # Resize feature if needed
        # TODO: Implement proper alpha blending with transparency
        
        # For now, simple paste (will be improved with actual feature images)
        h, w = feature.shape[:2]
        y1 = max(0, pos[1] - h // 2)
        y2 = min(canvas.shape[0], pos[1] + h // 2)
        x1 = max(0, pos[0] - w // 2)
        x2 = min(canvas.shape[1], pos[0] + w // 2)
        
        # Simple overlay (will be replaced with alpha blending)
        canvas[y1:y2, x1:x2] = feature[:y2-y1, :x2-x1]
        
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