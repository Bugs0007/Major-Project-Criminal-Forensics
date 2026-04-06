import multiprocessing
import os


# Django WSGI entrypoint.
wsgi_app = "face_matching.wsgi:application"

# Bind to PORT when provided by platform (Render/Railway/Heroku), else 8000.
bind = f"0.0.0.0:{os.getenv('PORT', '8000')}"

# Sensible defaults, overridable with environment variables.
workers = int(os.getenv("GUNICORN_WORKERS", multiprocessing.cpu_count() * 2 + 1))
threads = int(os.getenv("GUNICORN_THREADS", "2"))
timeout = int(os.getenv("GUNICORN_TIMEOUT", "120"))
graceful_timeout = int(os.getenv("GUNICORN_GRACEFUL_TIMEOUT", "30"))
keepalive = int(os.getenv("GUNICORN_KEEPALIVE", "5"))

max_requests = int(os.getenv("GUNICORN_MAX_REQUESTS", "1000"))
max_requests_jitter = int(os.getenv("GUNICORN_MAX_REQUESTS_JITTER", "100"))

loglevel = os.getenv("GUNICORN_LOG_LEVEL", "info")
accesslog = "-"
errorlog = "-"

preload_app = os.getenv("GUNICORN_PRELOAD", "false").lower() == "true"
