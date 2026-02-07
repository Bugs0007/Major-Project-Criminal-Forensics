from django.db import models
import json
import numpy as np


class FaceImage(models.Model):
    """
    Model to store face images and their encodings
    """
    # Image information
    image_url = models.URLField(max_length=500)
    original_filename = models.CharField(max_length=255)
    
    # Face encoding (128-dimensional vector stored as JSON)
    encoding = models.TextField()
    
    # Metadata
    name = models.CharField(max_length=255, blank=True, null=True)
    tags = models.JSONField(default=list, blank=True)
    notes = models.TextField(blank=True, null=True)
    
    # Timestamps
    uploaded_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-uploaded_at']
        verbose_name = 'Face Image'
        verbose_name_plural = 'Face Images'
    
    def __str__(self):
        return f"{self.name or 'Unnamed'} - {self.original_filename}"
    
    def set_encoding(self, encoding_array):
        """Convert numpy array to JSON string for storage"""
        if isinstance(encoding_array, np.ndarray):
            self.encoding = json.dumps(encoding_array.tolist())
        else:
            self.encoding = json.dumps(encoding_array)
    
    def get_encoding(self):
        """Convert JSON string back to numpy array"""
        return np.array(json.loads(self.encoding))