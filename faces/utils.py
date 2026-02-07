import boto3
import face_recognition
import numpy as np
from django.conf import settings
from PIL import Image
import io
import uuid
from botocore.exceptions import ClientError


def upload_to_s3(file, filename=None):
    """
    Upload file to S3 and return public URL
    """
    if filename is None:
        ext = file.name.split('.')[-1]
        filename = f"{uuid.uuid4()}.{ext}"
    
    s3_client = boto3.client(
        's3',
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_S3_REGION_NAME
    )
    
    try:
        s3_client.upload_fileobj(
            file,
            settings.AWS_STORAGE_BUCKET_NAME,
            filename,
            ExtraArgs={
                'ACL': 'public-read',
                'ContentType': file.content_type
            }
        )
        
        url = f"https://{settings.AWS_S3_CUSTOM_DOMAIN}/{filename}"
        return url, filename
        
    except ClientError as e:
        raise Exception(f"S3 upload failed: {str(e)}")


def get_face_encoding(image_file):
    """
    Extract face encoding from image file
    Returns: numpy array of face encoding or None if no face found
    """
    try:
        # Load image
        image = face_recognition.load_image_file(image_file)
        
        # Find face encodings
        encodings = face_recognition.face_encodings(
            image,
            model=settings.FACE_RECOGNITION_MODEL
        )
        
        if len(encodings) == 0:
            return None
        
        # Return first face encoding (in case multiple faces detected)
        return encodings[0]
        
    except Exception as e:
        raise Exception(f"Face encoding failed: {str(e)}")


def calculate_similarity(encoding1, encoding2):
    """
    Calculate similarity percentage between two face encodings
    Returns: similarity score (0-100)
    """
    # face_recognition.face_distance returns distance (0-1)
    # where 0 = identical, 1 = completely different
    distance = face_recognition.face_distance([encoding1], encoding2)[0]
    
    # Convert to similarity percentage
    similarity = (1 - distance) * 100
    
    return max(0, min(100, similarity))  # Clamp between 0-100


def compare_faces(known_encoding, test_encoding, tolerance=None):
    """
    Compare two faces and return if they match
    """
    if tolerance is None:
        tolerance = settings.FACE_RECOGNITION_TOLERANCE
    
    matches = face_recognition.compare_faces(
        [known_encoding],
        test_encoding,
        tolerance=tolerance
    )
    
    return matches[0]


def resize_image_if_needed(image_file, max_size=(1024, 1024)):
    """
    Resize image if it's too large (for faster processing)
    """
    try:
        img = Image.open(image_file)
        
        if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            output = io.BytesIO()
            img.save(output, format=img.format or 'JPEG')
            output.seek(0)
            
            return output
        
        image_file.seek(0)
        return image_file
        
    except Exception as e:
        image_file.seek(0)
        return image_file