from rest_framework import serializers
from .models import FaceImage


class FaceImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = FaceImage
        fields = ['id', 'image_url', 'original_filename', 'name', 
                  'tags', 'notes', 'uploaded_at']
        read_only_fields = ['id', 'uploaded_at']


class FaceUploadSerializer(serializers.Serializer):
    image = serializers.ImageField(required=True)
    name = serializers.CharField(required=False, allow_blank=True)
    tags = serializers.ListField(
        child=serializers.CharField(),
        required=False,
        allow_empty=True
    )
    notes = serializers.CharField(required=False, allow_blank=True)


class FaceSearchSerializer(serializers.Serializer):
    image = serializers.ImageField(required=True)
    min_similarity = serializers.FloatField(
        required=False, 
        default=0.0,
        min_value=0.0,
        max_value=100.0
    )
    max_results = serializers.IntegerField(
        required=False,
        default=10,
        min_value=1,
        max_value=50
    )