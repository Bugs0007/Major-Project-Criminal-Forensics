from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from django.db.models import Q

from .models import FaceImage
from .serializers import (
    FaceImageSerializer, 
    FaceUploadSerializer, 
    FaceSearchSerializer
)
from .utils import (
    upload_to_s3, 
    get_face_encoding, 
    calculate_similarity,
    resize_image_if_needed
)
from .sketch_utils import (
    SketchGenerator, 
    FaceFeatureComposer,
    pil_to_cv2,
    cv2_to_pil,
    convert_sketch_to_3channel
)
from PIL import Image
from .gan_enhancer import get_gan_enhancer
from django.conf import settings
import io
import os
import uuid


@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def upload_face(request):
    """
    Upload a face image to the database
    """
    serializer = FaceUploadSerializer(data=request.data)
    
    if not serializer.is_valid():
        return Response(
            {'error': serializer.errors},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    image_file = serializer.validated_data['image']
    name = serializer.validated_data.get('name', '')
    tags = serializer.validated_data.get('tags', [])
    notes = serializer.validated_data.get('notes', '')
    
    try:
        # Store original filename before resizing
        original_filename = image_file.name if hasattr(image_file, 'name') else 'image.jpg'
        
        # Resize if needed
        image_file = resize_image_if_needed(image_file)
        
        # Get face encoding
        encoding = get_face_encoding(image_file)
        
        if encoding is None:
            return Response(
                {'error': 'No face detected in the image'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Reset file pointer
        image_file.seek(0)
        
        # Upload to S3
        image_url, filename = upload_to_s3(
            image_file,
            filename=f"faces/{original_filename}"
        )
        
        # Save to database
        face_image = FaceImage(
            image_url=image_url,
            original_filename=original_filename,
            name=name,
            tags=tags,
            notes=notes
        )
        face_image.set_encoding(encoding)
        face_image.save()
        
        return Response(
            {
                'message': 'Face uploaded successfully',
                'data': FaceImageSerializer(face_image).data
            },
            status=status.HTTP_201_CREATED
        )
        
    except Exception as e:
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def search_face(request):
    """
    Search for matching faces in the database
    """
    serializer = FaceSearchSerializer(data=request.data)
    
    if not serializer.is_valid():
        return Response(
            {'error': serializer.errors},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    image_file = serializer.validated_data['image']
    min_similarity = serializer.validated_data.get('min_similarity', 0.0)
    max_results = serializer.validated_data.get('max_results', 10)
    
    try:
        # Resize if needed
        image_file = resize_image_if_needed(image_file)
        
        # Get face encoding from search image
        search_encoding = get_face_encoding(image_file)
        
        if search_encoding is None:
            return Response(
                {'error': 'No face detected in the search image'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Compare with all stored faces
        results = []
        
        for face_image in FaceImage.objects.all():
            stored_encoding = face_image.get_encoding()
            
            # Calculate similarity
            similarity = calculate_similarity(stored_encoding, search_encoding)
            
            # Only include if above minimum threshold
            if similarity >= min_similarity:
                results.append({
                    'id': face_image.id,
                    'image_url': face_image.image_url,
                    'original_filename': face_image.original_filename,
                    'name': face_image.name,
                    'tags': face_image.tags,
                    'notes': face_image.notes,
                    'similarity': round(similarity, 2),
                    'uploaded_at': face_image.uploaded_at
                })
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Limit results
        results = results[:max_results]
        
        return Response(
            {
                'matches_found': len(results),
                'results': results
            },
            status=status.HTTP_200_OK
        )
        
    except Exception as e:
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
def list_faces(request):
    """
    List all faces in the database
    """
    faces = FaceImage.objects.all()
    serializer = FaceImageSerializer(faces, many=True)
    
    return Response(
        {
            'count': faces.count(),
            'faces': serializer.data
        },
        status=status.HTTP_200_OK
    )


@api_view(['DELETE'])
def delete_face(request, face_id):
    """
    Delete a face from the database
    """
    try:
        face = FaceImage.objects.get(id=face_id)
        face.delete()
        
        return Response(
            {'message': 'Face deleted successfully'},
            status=status.HTTP_200_OK
        )
        
    except FaceImage.DoesNotExist:
        return Response(
            {'error': 'Face not found'},
            status=status.HTTP_404_NOT_FOUND
        )


@api_view(['GET'])
def health_check(request):
    """
    Simple health check endpoint
    """
    return Response(
        {
            'status': 'healthy',
            'database': 'connected',
            'total_faces': FaceImage.objects.count()
        },
        status=status.HTTP_200_OK
    )

@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def image_to_sketch(request):
    """
    Convert uploaded image to sketch with enhancement options
    """
    if 'image' not in request.FILES:
        return Response(
            {'error': 'No image provided'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    image_file = request.FILES['image']
    sketch_method = request.data.get('method', 'adaptive')  # pencil, edge, adaptive
    enhance_with_gan = request.data.get('enhance_gan', 'false').lower() == 'true'
    is_blurry = request.data.get('is_blurry', 'false').lower() == 'true'
    use_super_resolution = request.data.get('super_resolution', 'false').lower() == 'true'
    
    try:
        # Load image
        image = Image.open(image_file)
        image_cv = pil_to_cv2(image)
        
        # Apply super-resolution enhancement if requested
        if use_super_resolution:
            image_cv = SketchGenerator.super_resolution_enhance(image_cv)
        
        # Check if image is blurry and preprocess
        if is_blurry:
            image_cv = SketchGenerator.preprocess_blurry_image(image_cv)
        
        # Generate sketch based on method
        if sketch_method == 'pencil':
            sketch = SketchGenerator.image_to_sketch_pencil(image_cv)
        elif sketch_method == 'edge':
            sketch = SketchGenerator.image_to_sketch_edge(image_cv)
        else:  # adaptive (default)
            sketch = SketchGenerator.image_to_sketch_adaptive(image_cv)
        
        # Enhance sketch
        sketch = SketchGenerator.enhance_sketch(sketch)
        
        # Optional GAN enhancement
        if enhance_with_gan:
            gan_enhancer = get_gan_enhancer()
            sketch = gan_enhancer.enhance_sketch(sketch)
        
        # Convert to 3-channel for face recognition compatibility
        sketch_3ch = convert_sketch_to_3channel(sketch)
        
        # Convert to PIL
        sketch_pil = cv2_to_pil(sketch)
        
        # Save to bytes
        img_byte_arr = io.BytesIO()
        sketch_pil.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Get image bytes before creating InMemoryUploadedFile
        img_bytes = img_byte_arr.getvalue()
        img_byte_arr.seek(0)
        
        # Upload to S3
        from django.core.files.uploadedfile import InMemoryUploadedFile
        import uuid
        filename = f"sketch_{uuid.uuid4()}.png"
        
        sketch_file = InMemoryUploadedFile(
            io.BytesIO(img_bytes),
            None,
            filename,
            'image/png',
            len(img_bytes),
            None
        )
        sketch_file.content_type = 'image/png'
        
        sketch_url, _ = upload_to_s3(sketch_file, f"sketches/{filename}")
        
        # Optionally get face encoding for immediate search
        encoding = None
        if request.data.get('get_encoding', 'false').lower() == 'true':
            encoding_array = get_face_encoding(io.BytesIO(img_bytes))
            if encoding_array is not None:
                encoding = encoding_array.tolist()
        
        return Response(
            {
                'sketch_url': sketch_url,
                'original_filename': image_file.name,
                'method': sketch_method,
                'gan_enhanced': enhance_with_gan,
                'super_resolution_applied': use_super_resolution,
                'deblur_applied': is_blurry,
                'encoding': encoding
            },
            status=status.HTTP_200_OK
        )
        
    except Exception as e:
        import traceback
        print(f"Error generating sketch: {str(e)}")
        print(traceback.format_exc())
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
def compose_face_from_features(request):
    """
    Compose face from selected features
    """
    try:
        selected_features = request.data.get('features', {})
        
        if not selected_features:
            return Response(
                {'error': 'No features provided'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Load feature images from templates
        from .feature_templates import FeatureTemplateGenerator
        feature_images = {}
        feature_base_dir = os.path.join(settings.MEDIA_ROOT, 'feature_templates')
        
        # Ensure templates exist
        if not os.path.exists(feature_base_dir):
            # Generate templates on the fly
            FeatureTemplateGenerator.save_templates_to_disk(feature_base_dir)
        
        # Load selected features
        for feature_type, feature_id in selected_features.items():
            if feature_id is None:
                continue
                
            # Load feature from disk or generate on the fly
            feature_img = FeatureTemplateGenerator.load_feature_from_disk(
                feature_type, 
                feature_id, 
                feature_base_dir
            )
            
            if feature_img is not None:
                feature_images[feature_type] = feature_img
        
        if not feature_images:
            return Response(
                {'error': 'Could not load any features'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Compose face
        composed_face = FaceFeatureComposer.compose_face(feature_images)
        
        # Convert to sketch style (optional - can return as-is)
        convert_to_sketch = request.data.get('convert_to_sketch', 'true').lower() == 'true'
        
        if convert_to_sketch:
            # Convert composed face to sketch
            sketch = SketchGenerator.image_to_sketch_adaptive(composed_face)
            
            # Optional GAN enhancement
            enhance_with_gan = request.data.get('enhance_gan', 'false').lower() == 'true'
            if enhance_with_gan:
                gan_enhancer = get_gan_enhancer()
                sketch = gan_enhancer.enhance_sketch(sketch)
            
            final_image = sketch
        else:
            final_image = cv2.cvtColor(composed_face, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL
        if len(final_image.shape) == 2:
            final_pil = Image.fromarray(final_image, mode='L')
        else:
            final_pil = Image.fromarray(final_image)
        
        # Save to bytes
        img_byte_arr = io.BytesIO()
        final_pil.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Get image bytes before uploading
        img_bytes = img_byte_arr.getvalue()
        img_byte_arr.seek(0)
        
        # Upload to S3
        from django.core.files.uploadedfile import InMemoryUploadedFile
        import uuid
        filename = f"composed_{uuid.uuid4()}.png"
        
        composed_file = InMemoryUploadedFile(
            io.BytesIO(img_bytes),
            None,
            filename,
            'image/png',
            len(img_bytes),
            None
        )
        composed_file.content_type = 'image/png'
        
        composed_url, _ = upload_to_s3(composed_file, f"sketches/{filename}")
        
        # Get face encoding
        encoding = None
        if request.data.get('get_encoding', 'false').lower() == 'true':
            # Convert to 3-channel if grayscale for face encoding
            if len(final_image.shape) == 2:
                final_image_3ch = convert_sketch_to_3channel(final_image)
            else:
                final_image_3ch = final_image
            
            final_pil_3ch = cv2_to_pil(final_image_3ch)
            img_byte_arr_3ch = io.BytesIO()
            final_pil_3ch.save(img_byte_arr_3ch, format='PNG')
            img_byte_arr_3ch.seek(0)
            
            encoding_array = get_face_encoding(img_byte_arr_3ch)
            if encoding_array is not None:
                encoding = encoding_array.tolist()
        
        return Response(
            {
                'sketch_url': composed_url,
                'features_used': selected_features,
                'gan_enhanced': request.data.get('enhance_gan', 'false').lower() == 'true',
                'encoding': encoding
            },
            status=status.HTTP_200_OK
        )
        
    except Exception as e:
        import traceback
        print(f"Error composing face: {str(e)}")
        print(traceback.format_exc())
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
def get_feature_library(request):
    """
    Get available facial features for the composer
    """
    try:
        # In production, this would list actual feature files from S3
        # For now, return metadata about available features
        
        feature_library = {
            'face_shapes': [
                {'id': 'face_1', 'name': 'Oval', 'thumbnail': '/features/faces/face_1_thumb.png'},
                {'id': 'face_2', 'name': 'Round', 'thumbnail': '/features/faces/face_2_thumb.png'},
                {'id': 'face_3', 'name': 'Square', 'thumbnail': '/features/faces/face_3_thumb.png'},
            ],
            'eyes': [
                {'id': 'eyes_1', 'name': 'Almond', 'thumbnail': '/features/eyes/eyes_1_thumb.png'},
                {'id': 'eyes_2', 'name': 'Round', 'thumbnail': '/features/eyes/eyes_2_thumb.png'},
                {'id': 'eyes_3', 'name': 'Narrow', 'thumbnail': '/features/eyes/eyes_3_thumb.png'},
            ],
            'noses': [
                {'id': 'nose_1', 'name': 'Straight', 'thumbnail': '/features/noses/nose_1_thumb.png'},
                {'id': 'nose_2', 'name': 'Button', 'thumbnail': '/features/noses/nose_2_thumb.png'},
                {'id': 'nose_3', 'name': 'Aquiline', 'thumbnail': '/features/noses/nose_3_thumb.png'},
            ],
            'mouths': [
                {'id': 'mouth_1', 'name': 'Neutral', 'thumbnail': '/features/mouths/mouth_1_thumb.png'},
                {'id': 'mouth_2', 'name': 'Smiling', 'thumbnail': '/features/mouths/mouth_2_thumb.png'},
                {'id': 'mouth_3', 'name': 'Small', 'thumbnail': '/features/mouths/mouth_3_thumb.png'},
            ],
            'eyebrows': [
                {'id': 'brow_1', 'name': 'Straight', 'thumbnail': '/features/eyebrows/brow_1_thumb.png'},
                {'id': 'brow_2', 'name': 'Arched', 'thumbnail': '/features/eyebrows/brow_2_thumb.png'},
                {'id': 'brow_3', 'name': 'Angled', 'thumbnail': '/features/eyebrows/brow_3_thumb.png'},
            ],
        }
        
        return Response(feature_library, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )