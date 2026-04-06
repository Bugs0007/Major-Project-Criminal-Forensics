import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Any

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None


class GANEnhancer:
    """
    GAN-based sketch enhancement
    Uses pre-trained models for photo-to-sketch conversion
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        """
        Initialize GAN enhancer
        
        Args:
            model_path: Path to pre-trained model weights
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device) if torch is not None else 'cpu'
        self.model = None
        self.model_loaded = False
        
        if model_path and Path(model_path).exists():
            self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """Load pre-trained GAN model"""
        if torch is None:
            self.model_loaded = False
            return

        try:
            # This is a placeholder - actual implementation depends on model choice
            # For APDrawingGAN or similar
            self.model = torch.load(model_path, map_location=self.device)
            self.model.eval()
            self.model_loaded = True
        except Exception as e:
            print(f"Failed to load GAN model: {e}")
            self.model_loaded = False
    
    def enhance_sketch(self, sketch: np.ndarray) -> np.ndarray:
        """
        Enhance sketch using GAN
        
        Args:
            sketch: Input sketch (grayscale or BGR)
        
        Returns:
            Enhanced sketch
        """
        if torch is None or not self.model_loaded:
            # Fallback to non-GAN enhancement
            return self._fallback_enhancement(sketch)
        
        try:
            # Preprocess
            input_tensor = self._preprocess(sketch)
            
            # Inference
            with torch.no_grad():
                output = self.model(input_tensor)
            
            # Postprocess
            enhanced = self._postprocess(output)
            
            return enhanced
            
        except Exception as e:
            print(f"GAN enhancement failed: {e}, using fallback")
            return self._fallback_enhancement(sketch)
    
    def _preprocess(self, image: np.ndarray) -> Any:
        """Preprocess image for GAN input"""
        if torch is None:
            raise RuntimeError("torch is not available")

        # Resize to model input size (typically 512x512)
        resized = cv2.resize(image, (512, 512))
        
        # Normalize to [-1, 1]
        normalized = (resized.astype(np.float32) / 127.5) - 1.0
        
        # Convert to tensor [B, C, H, W]
        if len(normalized.shape) == 2:
            normalized = np.expand_dims(normalized, axis=2)
        
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1))
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        
        return tensor.to(self.device)
    
    def _postprocess(self, output: Any) -> np.ndarray:
        """Convert GAN output back to image"""
        # Remove batch dimension
        output = output.squeeze(0)
        
        # Convert to numpy
        output = output.cpu().numpy()
        
        # Transpose to [H, W, C]
        output = output.transpose(1, 2, 0)
        
        # Denormalize from [-1, 1] to [0, 255]
        output = ((output + 1.0) * 127.5).astype(np.uint8)
        
        return output
    
    def _fallback_enhancement(self, sketch: np.ndarray) -> np.ndarray:
        """Fallback enhancement without GAN"""
        # Simple sharpening and contrast adjustment
        if len(sketch.shape) == 2:
            result = sketch
        else:
            result = cv2.cvtColor(sketch, cv2.COLOR_BGR2GRAY)
        
        # Sharpen
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(result, -1, kernel)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(sharpened)
        
        return enhanced


# Singleton instance
_gan_enhancer_instance = None


def get_gan_enhancer(model_path: Optional[str] = None) -> GANEnhancer:
    """Get or create GAN enhancer singleton"""
    global _gan_enhancer_instance
    
    if _gan_enhancer_instance is None:
        device = 'cuda' if (torch is not None and torch.cuda.is_available()) else 'cpu'
        _gan_enhancer_instance = GANEnhancer(model_path, device)
    
    return _gan_enhancer_instance