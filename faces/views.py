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