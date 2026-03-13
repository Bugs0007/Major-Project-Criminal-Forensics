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
import re
import time

import requests as http_requests
from urllib.parse import quote


def _to_bool(value, default=False):
    """Safely parse booleans from JSON booleans, strings, or missing values."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _extract_http_error(response):
    """Best-effort error extraction for upstream AI provider responses."""
    try:
        payload = response.json()
        if isinstance(payload, dict):
            return payload.get('error') or payload.get('message') or str(payload)
        return str(payload)
    except Exception:
        body = (response.text or '').strip()
        return body[:300] if body else 'unknown upstream error'


def _generate_with_hugging_face(prompt, width=512, height=512):
    """Generate an image via Hugging Face Inference API."""
    hf_token = getattr(settings, 'HF_API_TOKEN', None) or os.getenv('HF_API_TOKEN')
    if not hf_token:
        raise RuntimeError('HF_API_TOKEN is not configured')

    model = getattr(
        settings,
        'AI_IMAGE_HF_MODEL',
        'stabilityai/stable-diffusion-xl-base-1.0'
    )
    endpoint = f"https://router.huggingface.co/hf-inference/models/{model}"
    headers = {
        'Authorization': f'Bearer {hf_token}',
        'Accept': 'image/png',
        'Content-Type': 'application/json',
    }
    payload = {
        'inputs': prompt,
        'parameters': {
            'width': width,
            'height': height,
            'negative_prompt': 'color, watermark, logo, text, blurry, low quality',
        },
        'options': {
            'wait_for_model': True,
        },
    }

    for attempt in range(3):
        response = http_requests.post(endpoint, headers=headers, json=payload, timeout=150)
        content_type = response.headers.get('Content-Type', '').lower()
        if response.status_code == 200 and content_type.startswith('image/') and len(response.content) > 1000:
            return response.content, 'huggingface', model

        if response.status_code in {503, 529} and attempt < 2:
            time.sleep(2 * (attempt + 1))
            continue

        error_details = _extract_http_error(response)
        raise RuntimeError(
            f"Hugging Face generation failed (status {response.status_code}): {error_details}"
        )

    raise RuntimeError('Hugging Face generation did not return an image')


def _generate_with_pollinations(prompt, width=512, height=512):
    """Generate an image via Pollinations.ai."""
    encoded_prompt = quote(prompt, safe='')
    model = getattr(settings, 'AI_IMAGE_POLLINATIONS_MODEL', 'flux')
    poll_url = (
        f"https://image.pollinations.ai/prompt/{encoded_prompt}"
        f"?width={width}&height={height}&nologo=true&model={quote(model, safe='')}&seed={uuid.uuid4().int % 100000}"
    )
    response = http_requests.get(poll_url, timeout=120)
    if response.status_code != 200 or len(response.content) < 1000:
        raise RuntimeError(
            f"Pollinations generation failed (status {response.status_code}): {_extract_http_error(response)}"
        )
    return response.content, 'pollinations', model


def _generate_ai_sketch_image(prompt, width=512, height=512):
    """Generate AI image bytes with provider failover."""
    preferred = str(getattr(settings, 'AI_IMAGE_PROVIDER', 'huggingface')).strip().lower()
    order = ['huggingface', 'pollinations']
    if preferred == 'pollinations':
        order = ['pollinations', 'huggingface']

    errors = []
    for provider in order:
        try:
            if provider == 'huggingface':
                return _generate_with_hugging_face(prompt, width=width, height=height)
            return _generate_with_pollinations(prompt, width=width, height=height)
        except Exception as exc:
            errors.append(f"{provider}: {exc}")

    raise RuntimeError(
        "AI image generation failed with all providers. " + " | ".join(errors)
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

@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def image_to_sketch(request):
    """
    Convert uploaded image to sketch with enhancement options.

    When ``use_ai=True`` the endpoint uses the same AI image generator
    with provider fallback as ``compose_face_from_features``. This is especially
    useful for blurry / low-quality images where traditional OpenCV
    processing produces poor results.  Pass an optional ``description``
    field (e.g. "male, oval face, short dark hair") to guide generation.
    """
    if 'image' not in request.FILES:
        return Response(
            {'error': 'No image provided'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    image_file = request.FILES['image']
    sketch_method = request.data.get('method', 'adaptive')  # pencil, edge, adaptive
    enhance_with_gan = _to_bool(request.data.get('enhance_gan'), default=False)
    is_blurry = _to_bool(request.data.get('is_blurry'), default=False)
    use_super_resolution = _to_bool(request.data.get('super_resolution'), default=False)
    use_ai = _to_bool(request.data.get('use_ai'), default=False)
    description = request.data.get('description', '')
    
    try:
        # ----- AI path -----
        if use_ai:
            if description:
                feature_text = description
            else:
                feature_text = "front-facing portrait"

            prompt = (
                f"Realistic police forensic pencil sketch portrait of a person, "
                f"detailed graphite drawing on white paper, "
                f"front-facing mugshot style, neutral expression, "
                f"with the following description: {feature_text}. "
                f"Black and white pencil sketch, high detail, professional forensic artist style, "
                f"clean white background, no color, no watermark"
            )

            print(f"[AI Sketch] Prompt: {prompt[:120]}...")
            img_bytes, provider, model = _generate_ai_sketch_image(prompt, width=512, height=512)
            print(f"[AI Sketch] Received image from {provider}/{model}: {len(img_bytes)} bytes")
        else:
            # ----- Traditional OpenCV path -----
            image = Image.open(image_file)
            image_cv = pil_to_cv2(image)

            if use_super_resolution:
                image_cv = SketchGenerator.super_resolution_enhance(image_cv)

            if is_blurry:
                image_cv = SketchGenerator.preprocess_blurry_image(image_cv)

            if sketch_method == 'pencil':
                sketch = SketchGenerator.image_to_sketch_pencil(image_cv)
            elif sketch_method == 'edge':
                sketch = SketchGenerator.image_to_sketch_edge(image_cv)
            else:
                sketch = SketchGenerator.image_to_sketch_adaptive(image_cv)

            sketch = SketchGenerator.enhance_sketch(sketch)

            if enhance_with_gan:
                gan_enhancer = get_gan_enhancer()
                sketch = gan_enhancer.enhance_sketch(sketch)

            sketch_pil = cv2_to_pil(sketch)
            buf = io.BytesIO()
            sketch_pil.save(buf, format='PNG')
            img_bytes = buf.getvalue()

        # ----- Base64 representation (always available) -----
        import base64
        sketch_b64 = base64.b64encode(img_bytes).decode('utf-8')

        # ----- Try S3 upload, but don't fail if it's unreachable -----
        sketch_url = None
        try:
            from django.core.files.uploadedfile import InMemoryUploadedFile
            filename = f"sketch_{uuid.uuid4()}.png"

            sketch_file = InMemoryUploadedFile(
                io.BytesIO(img_bytes),
                None,
                filename,
                'image/png',
                len(img_bytes),
                None,
            )
            sketch_file.content_type = 'image/png'

            sketch_url, _ = upload_to_s3(sketch_file, f"sketches/{filename}")
        except Exception as s3_err:
            print(f"[Sketch] S3 upload skipped: {s3_err}")

        # ----- Optionally get face encoding -----
        encoding = None
        if _to_bool(request.data.get('get_encoding'), default=False):
            encoding_array = get_face_encoding(io.BytesIO(img_bytes))
            if encoding_array is not None:
                encoding = encoding_array.tolist()

        return Response(
            {
                'sketch_url': sketch_url,
                'sketch_base64': sketch_b64,
                'original_filename': image_file.name,
                'method': 'ai' if use_ai else sketch_method,
                'ai_generated': use_ai,
                'gan_enhanced': enhance_with_gan and not use_ai,
                'super_resolution_applied': use_super_resolution and not use_ai,
                'deblur_applied': is_blurry and not use_ai,
                'encoding': encoding,
            },
            status=status.HTTP_200_OK,
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
    Compose a face from selected features using AI image generation.
    Builds a descriptive prompt from feature names and generates a
    realistic forensic-style pencil sketch with provider fallback.
    """
    try:
        selected_features = request.data.get('features', {})

        if not selected_features:
            return Response(
                {'error': 'No features provided'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # --- Map feature IDs to human-readable descriptions ----
        display_names = {
            'face_shapes': ['oval', 'round', 'square'],
            'hair': ['short cropped', 'long flowing', 'curly'],
            'ears': ['normal', 'pointed', 'small'],
            'eyes': ['almond-shaped', 'round', 'narrow'],
            'noses': ['straight', 'button / upturned', 'aquiline / hooked'],
            'mouths': ['neutral closed', 'slightly smiling', 'small pursed'],
            'eyebrows': ['straight', 'arched', 'angled'],
        }

        descriptions = []
        for feature_type, feature_id in selected_features.items():
            if feature_id is None:
                continue
            try:
                idx = int(feature_id.split('_')[-1]) - 1
                names = display_names.get(feature_type, [])
                if 0 <= idx < len(names):
                    label = feature_type.replace('_', ' ')
                    descriptions.append(f"{names[idx]} {label}")
            except (ValueError, IndexError):
                pass

        if not descriptions:
            return Response(
                {'error': 'No valid features selected'},
                status=status.HTTP_400_BAD_REQUEST
            )

        feature_text = ', '.join(descriptions)

        # --- Optional person details ---
        person_details = []
        age = request.data.get('age')
        gender = request.data.get('gender')
        ethnicity = request.data.get('ethnicity')
        additional_notes = request.data.get('additional_notes', '')

        if gender:
            person_details.append(f"{gender}")
        if age:
            person_details.append(f"approximately {age} years old" if age.replace('+', '').isdigit() or age.endswith('s') else f"{age}")
        if ethnicity:
            person_details.append(f"{ethnicity} ethnicity")

        person_text = ', '.join(person_details)
        if person_text:
            person_text = f"The person is {person_text}. "
        else:
            person_text = ""

        notes_text = ''
        if additional_notes:
            notes_text = f"Additional distinguishing features: {additional_notes}. "

        prompt = (
            f"Realistic police forensic pencil sketch portrait of a person, "
            f"detailed graphite drawing on white paper, "
            f"front-facing mugshot style, neutral expression, "
            f"{person_text}"
            f"with the following facial features: {feature_text}. "
            f"{notes_text}"
            f"Black and white pencil sketch, high detail, professional forensic artist style, "
            f"clean white background, no color, no watermark"
        )

        print(f"[AI Compose] Prompt: {prompt[:120]}...")
        img_bytes, provider, model = _generate_ai_sketch_image(prompt, width=512, height=512)
        print(f"[AI Compose] Received image from {provider}/{model}: {len(img_bytes)} bytes")

        # --- Upload to S3 ----
        from django.core.files.uploadedfile import InMemoryUploadedFile
        filename = f"composed_{uuid.uuid4()}.png"

        composed_file = InMemoryUploadedFile(
            io.BytesIO(img_bytes),
            None,
            filename,
            'image/png',
            len(img_bytes),
            None,
        )
        composed_file.content_type = 'image/png'

        composed_url, _ = upload_to_s3(composed_file, f"sketches/{filename}")

        # --- Optionally get face encoding ----
        encoding = None
        if _to_bool(request.data.get('get_encoding'), default=False):
            encoding_array = get_face_encoding(io.BytesIO(img_bytes))
            if encoding_array is not None:
                encoding = encoding_array.tolist()

        return Response(
            {
                'sketch_url': composed_url,
                'features_used': selected_features,
                'ai_generated': True,
                'prompt': prompt,
                'encoding': encoding,
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
        from .feature_templates import FeatureTemplateGenerator

        feature_base_dir = os.path.join(settings.MEDIA_ROOT, 'feature_templates')
        feature_types = ['face_shapes', 'hair', 'ears', 'eyes', 'noses', 'mouths', 'eyebrows']

        # Ensure local sketch templates exist.
        if not os.path.exists(feature_base_dir):
            FeatureTemplateGenerator.save_templates_to_disk(feature_base_dir)

        for feature_type in feature_types:
            os.makedirs(os.path.join(feature_base_dir, feature_type), exist_ok=True)

        def _extract_index(filename):
            match = re.search(r'_(\d+)\.png$', filename)
            return int(match.group(1)) if match else 9999

        display_names = {
            'face_shapes': ['Oval', 'Round', 'Square'],
            'hair': ['Short', 'Long', 'Curly'],
            'ears': ['Normal', 'Pointed', 'Small'],
            'eyes': ['Almond', 'Round', 'Narrow'],
            'noses': ['Straight', 'Button', 'Aquiline'],
            'mouths': ['Neutral', 'Smiling', 'Small'],
            'eyebrows': ['Straight', 'Arched', 'Angled'],
        }

        feature_library = {}
        for feature_type in feature_types:
            feature_dir = os.path.join(feature_base_dir, feature_type)
            files = [
                f for f in os.listdir(feature_dir)
                if f.lower().endswith('.png')
            ]

            # Regenerate missing sets on demand.
            if not files:
                FeatureTemplateGenerator.save_templates_to_disk(feature_base_dir)
                files = [
                    f for f in os.listdir(feature_dir)
                    if f.lower().endswith('.png')
                ]

            files.sort(key=_extract_index)

            features = []
            for filename in files:
                idx = _extract_index(filename)
                names = display_names.get(feature_type, [])
                default_name = f"{feature_type[:-1].replace('_', ' ').title()} {idx}"
                feature_name = names[idx - 1] if 0 < idx <= len(names) else default_name
                media_path = f"{settings.MEDIA_URL}feature_templates/{feature_type}/{filename}"
                features.append(
                    {
                        'id': f'{feature_type}_{idx}',
                        'name': feature_name,
                        'thumbnail': request.build_absolute_uri(media_path),
                    }
                )

            feature_library[feature_type] = features
        
        return Response(feature_library, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )