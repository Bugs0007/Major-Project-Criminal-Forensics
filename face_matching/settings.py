import os
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


def _get_list_env(name, default=None):
    value = os.getenv(name)
    if value is None:
        return default or []
    return [item.strip() for item in value.split(',') if item.strip()]

def _database_config_from_env():
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        return {
            'ENGINE': 'django.db.backends.postgresql',
            'NAME': os.getenv('DB_NAME', 'face_matching_db'),
            'USER': os.getenv('DB_USER', 'postgres'),
            'PASSWORD': os.getenv('DB_PASSWORD', 'your_password'),
            'HOST': os.getenv('DB_HOST', 'localhost'),
            'PORT': os.getenv('DB_PORT', '5432'),
        }

    parsed = urlparse(database_url)
    engine_map = {
        'postgres': 'django.db.backends.postgresql',
        'postgresql': 'django.db.backends.postgresql',
        'pgsql': 'django.db.backends.postgresql',
    }

    return {
        'ENGINE': engine_map.get(parsed.scheme, 'django.db.backends.postgresql'),
        'NAME': parsed.path.lstrip('/'),
        'USER': parsed.username or '',
        'PASSWORD': parsed.password or '',
        'HOST': parsed.hostname or '',
        'PORT': str(parsed.port or '5432'),
    }

DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'

SECRET_KEY = os.getenv('SECRET_KEY', 'django-insecure-dev-only-key')
if not DEBUG and SECRET_KEY == 'django-insecure-dev-only-key':
    raise ValueError('SECRET_KEY must be set when DEBUG=False')


ALLOWED_HOSTS = _get_list_env(
    'ALLOWED_HOSTS',
    [
        'localhost',
        '127.0.0.1',
        'major-project-criminal-forensics.onrender.com',
        'major-project-criminal-forensics-face-match-1fmg0f5hp.vercel.app',
    ],
)

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    
    # Third party apps
    'rest_framework',
    'corsheaders',
    
    # Local apps
    'faces',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'corsheaders.middleware.CorsMiddleware',  # Add this
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'face_matching.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'face_matching.wsgi.application'

# Database
DATABASES = {
    'default': _database_config_from_env()
}

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static files
STATIC_URL = 'static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')

# Media files
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# CORS Settings
CORS_ALLOW_ALL_ORIGINS = True

CORS_ALLOWED_ORIGINS = _get_list_env(
    'CORS_ALLOWED_ORIGINS',
    [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://major-project-criminal-forensics.onrender.com",
        "https://major-project-criminal-forensics-face-match-1fmg0f5hp.vercel.app",
    ],
)

CORS_ALLOW_CREDENTIALS = True

CSRF_TRUSTED_ORIGINS = _get_list_env(
    'CSRF_TRUSTED_ORIGINS',
    [
        "https://major-project-criminal-forensics.onrender.com",
        "https://major-project-criminal-forensics-face-match-1fmg0f5hp.vercel.app",
    ],
)

# REST Framework Settings
REST_FRAMEWORK = {
    'DEFAULT_PARSER_CLASSES': [
        'rest_framework.parsers.JSONParser',
        'rest_framework.parsers.MultiPartParser',
        'rest_framework.parsers.FormParser',
    ],
}

# AWS S3 Settings
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_STORAGE_BUCKET_NAME = os.getenv('AWS_STORAGE_BUCKET_NAME')
AWS_S3_REGION_NAME = os.getenv('AWS_S3_REGION_NAME', 'us-east-1')
AWS_S3_CUSTOM_DOMAIN = f'{AWS_STORAGE_BUCKET_NAME}.s3.amazonaws.com'

# Face Recognition Settings
FACE_RECOGNITION_TOLERANCE = 0.6  # Lower = more strict matching
FACE_RECOGNITION_MODEL = 'large'  # 'large' or 'small' (large is more accurate)

# AI image generation settings for sketch compose endpoints.
# Options: "huggingface" or "pollinations"
AI_IMAGE_PROVIDER = os.getenv('AI_IMAGE_PROVIDER', 'huggingface')
HF_API_TOKEN = os.getenv('HF_API_TOKEN')
AI_IMAGE_HF_MODEL = os.getenv(
    'AI_IMAGE_HF_MODEL',
    'stabilityai/stable-diffusion-xl-base-1.0'
)
AI_IMAGE_POLLINATIONS_MODEL = os.getenv('AI_IMAGE_POLLINATIONS_MODEL', 'flux')
