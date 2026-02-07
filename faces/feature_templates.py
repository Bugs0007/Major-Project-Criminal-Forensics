import cv2
import numpy as np
from typing import Dict, List
import os


class FeatureTemplateGenerator:
    """
    Generate simple feature templates procedurally
    In production, replace with actual drawn/designed features
    """
    
    @staticmethod
    def generate_eye_templates() -> List[np.ndarray]:
        """Generate various eye shapes"""
        eyes = []
        
        # Template 1: Almond eyes
        eye1 = np.ones((60, 80, 4), dtype=np.uint8) * 255
        eye1[:, :, 3] = 0  # Transparent
        cv2.ellipse(eye1, (40, 30), (35, 20), 0, 0, 360, (0, 0, 0, 255), 2)
        cv2.circle(eye1, (40, 30), 8, (0, 0, 0, 255), -1)
        eyes.append(eye1)
        
        # Template 2: Round eyes
        eye2 = np.ones((60, 80, 4), dtype=np.uint8) * 255
        eye2[:, :, 3] = 0
        cv2.circle(eye2, (40, 30), 25, (0, 0, 0, 255), 2)
        cv2.circle(eye2, (40, 30), 10, (0, 0, 0, 255), -1)
        eyes.append(eye2)
        
        # Template 3: Narrow eyes
        eye3 = np.ones((60, 80, 4), dtype=np.uint8) * 255
        eye3[:, :, 3] = 0
        cv2.ellipse(eye3, (40, 30), (35, 12), 0, 0, 360, (0, 0, 0, 255), 2)
        cv2.circle(eye3, (40, 30), 6, (0, 0, 0, 255), -1)
        eyes.append(eye3)
        
        return eyes
    
    @staticmethod
    def generate_nose_templates() -> List[np.ndarray]:
        """Generate various nose shapes"""
        noses = []
        
        # Template 1: Straight nose
        nose1 = np.ones((80, 60, 4), dtype=np.uint8) * 255
        nose1[:, :, 3] = 0
        cv2.line(nose1, (30, 10), (30, 60), (0, 0, 0, 255), 2)
        cv2.ellipse(nose1, (30, 65), (15, 10), 0, 0, 180, (0, 0, 0, 255), 2)
        noses.append(nose1)
        
        # Template 2: Button nose
        nose2 = np.ones((80, 60, 4), dtype=np.uint8) * 255
        nose2[:, :, 3] = 0
        cv2.line(nose2, (30, 10), (30, 55), (0, 0, 0, 255), 2)
        cv2.ellipse(nose2, (30, 60), (12, 8), 0, 0, 180, (0, 0, 0, 255), 2)
        noses.append(nose2)
        
        # Template 3: Aquiline nose
        nose3 = np.ones((80, 60, 4), dtype=np.uint8) * 255
        nose3[:, :, 3] = 0
        pts = np.array([[30, 10], [32, 30], [30, 50], [30, 60]], np.int32)
        cv2.polylines(nose3, [pts], False, (0, 0, 0, 255), 2)
        cv2.ellipse(nose3, (30, 65), (15, 10), 0, 0, 180, (0, 0, 0, 255), 2)
        noses.append(nose3)
        
        return noses
    
    @staticmethod
    def generate_mouth_templates() -> List[np.ndarray]:
        """Generate various mouth shapes"""
        mouths = []
        
        # Template 1: Neutral mouth
        mouth1 = np.ones((40, 100, 4), dtype=np.uint8) * 255
        mouth1[:, :, 3] = 0
        cv2.ellipse(mouth1, (50, 20), (40, 15), 0, 0, 180, (0, 0, 0, 255), 2)
        cv2.line(mouth1, (10, 20), (90, 20), (0, 0, 0, 255), 2)
        mouths.append(mouth1)
        
        # Template 2: Smiling mouth
        mouth2 = np.ones((40, 100, 4), dtype=np.uint8) * 255
        mouth2[:, :, 3] = 0
        cv2.ellipse(mouth2, (50, 10), (40, 20), 0, 0, 180, (0, 0, 0, 255), 2)
        mouths.append(mouth2)
        
        # Template 3: Small mouth
        mouth3 = np.ones((40, 100, 4), dtype=np.uint8) * 255
        mouth3[:, :, 3] = 0
        cv2.ellipse(mouth3, (50, 20), (25, 10), 0, 0, 180, (0, 0, 0, 255), 2)
        mouths.append(mouth3)
        
        return mouths
    
    @staticmethod
    def generate_eyebrow_templates() -> List[np.ndarray]:
        """Generate various eyebrow shapes"""
        eyebrows = []
        
        # Template 1: Straight eyebrows
        brow1 = np.ones((30, 80, 4), dtype=np.uint8) * 255
        brow1[:, :, 3] = 0
        cv2.line(brow1, (10, 15), (70, 15), (0, 0, 0, 255), 3)
        eyebrows.append(brow1)
        
        # Template 2: Arched eyebrows
        brow2 = np.ones((30, 80, 4), dtype=np.uint8) * 255
        brow2[:, :, 3] = 0
        cv2.ellipse(brow2, (40, 25), (35, 15), 0, 180, 360, (0, 0, 0, 255), 3)
        eyebrows.append(brow2)
        
        # Template 3: Angled eyebrows
        brow3 = np.ones((30, 80, 4), dtype=np.uint8) * 255
        brow3[:, :, 3] = 0
        pts = np.array([[10, 20], [30, 10], [70, 15]], np.int32)
        cv2.polylines(brow3, [pts], False, (0, 0, 0, 255), 3)
        eyebrows.append(brow3)
        
        return eyebrows
    
    @staticmethod
    def generate_face_shape_templates() -> List[np.ndarray]:
        """Generate various face shapes (outlines)"""
        faces = []
        
        # Template 1: Oval face
        face1 = np.ones((512, 512, 4), dtype=np.uint8) * 255
        face1[:, :, 3] = 0
        cv2.ellipse(face1, (256, 280), (120, 180), 0, 0, 360, (0, 0, 0, 255), 2)
        faces.append(face1)
        
        # Template 2: Round face
        face2 = np.ones((512, 512, 4), dtype=np.uint8) * 255
        face2[:, :, 3] = 0
        cv2.ellipse(face2, (256, 256), (140, 160), 0, 0, 360, (0, 0, 0, 255), 2)
        faces.append(face2)
        
        # Template 3: Square face
        face3 = np.ones((512, 512, 4), dtype=np.uint8) * 255
        face3[:, :, 3] = 0
        pts = np.array([
            [140, 120], [370, 120],
            [380, 240], [380, 360],
            [256, 420], [132, 360],
            [132, 240], [140, 120]
        ], np.int32)
        cv2.polylines(face3, [pts], True, (0, 0, 0, 255), 2)
        faces.append(face3)
        
        return faces
    
    @staticmethod
    def save_templates_to_disk(output_dir: str):
        """Save all generated templates to disk"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate and save all templates
        templates = {
            'eyes': FeatureTemplateGenerator.generate_eye_templates(),
            'noses': FeatureTemplateGenerator.generate_nose_templates(),
            'mouths': FeatureTemplateGenerator.generate_mouth_templates(),
            'eyebrows': FeatureTemplateGenerator.generate_eyebrow_templates(),
            'face_shapes': FeatureTemplateGenerator.generate_face_shape_templates(),
        }
        
        for feature_type, template_list in templates.items():
            feature_dir = os.path.join(output_dir, feature_type)
            os.makedirs(feature_dir, exist_ok=True)
            
            for idx, template in enumerate(template_list):
                filepath = os.path.join(feature_dir, f"{feature_type}_{idx + 1}.png")
                cv2.imwrite(filepath, template)
        
        print(f"Templates saved to {output_dir}")