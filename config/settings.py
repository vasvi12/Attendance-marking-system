"""
Django settings for the Computer Vision Attendance System.

See https://docs.djangoproject.com/en/4.2/topics/settings/ for details on
what each of these settings does.
"""

from pathlib import Path

from decouple import Csv, config

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Core / security
# ---------------------------------------------------------------------------

SECRET_KEY = config(
    "SECRET_KEY",
    default="django-insecure-dev-key-change-me-before-deploying",
)

DEBUG = config("DEBUG", default=True, cast=bool)

ALLOWED_HOSTS = config(
    "ALLOWED_HOSTS",
    default="localhost,127.0.0.1",
    cast=Csv(),
)


# ---------------------------------------------------------------------------
# Application definition
# ---------------------------------------------------------------------------

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "attendance.apps.AttendanceConfig",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / config("DATABASE_NAME", default="db.sqlite3"),
    }
}


# ---------------------------------------------------------------------------
# Password validation
# ---------------------------------------------------------------------------

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]


# ---------------------------------------------------------------------------
# Internationalization
# ---------------------------------------------------------------------------

LANGUAGE_CODE = "en-us"
TIME_ZONE = config("TIME_ZONE", default="UTC")
USE_I18N = True
USE_TZ = True


# ---------------------------------------------------------------------------
# Static & media files
# ---------------------------------------------------------------------------

STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"

MEDIA_URL = "media/"
MEDIA_ROOT = BASE_DIR / "media"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {"class": "logging.StreamHandler"},
    },
    "loggers": {
        "attendance": {
            "handlers": ["console"],
            "level": config("LOG_LEVEL", default="INFO"),
        },
        "django": {
            "handlers": ["console"],
            "level": "WARNING",
        },
    },
}


# ---------------------------------------------------------------------------
# Computer vision / face recognition configuration
#
# These values control the detection + recognition pipeline in
# attendance/services/. They are exposed as environment variables so the
# system can be tuned per-camera/per-classroom without touching code.
# ---------------------------------------------------------------------------

CV_DATA_DIR = BASE_DIR / config("CV_DATA_DIR", default="data")
CV_FACES_DIR = CV_DATA_DIR / "faces"
CV_MODEL_PATH = CV_DATA_DIR / "trained_model.yml"
CV_LABELS_PATH = CV_DATA_DIR / "labels.json"

# Optional higher-accuracy DNN face detector (Caffe SSD ResNet10). These
# files are NOT bundled with the repository (see README "Face Detection
# Model" section) - if absent, the system automatically falls back to
# OpenCV's built-in Haar cascade detector, which ships with opencv-python
# and needs no download.
CV_DNN_MODEL_DIR = CV_DATA_DIR / "models"
CV_DNN_PROTOTXT = CV_DNN_MODEL_DIR / "deploy.prototxt"
CV_DNN_WEIGHTS = CV_DNN_MODEL_DIR / "res10_300x300_ssd_iter_140000.caffemodel"

# Minimum face size (pixels) accepted by the detector. Faces smaller than
# this are treated as "too far away" and ignored, which cuts down on false
# positives from background clutter.
CV_MIN_FACE_SIZE = config("CV_MIN_FACE_SIZE", default=60, cast=int)

# Number of enrollment samples captured per student during registration.
CV_SAMPLES_PER_STUDENT = config("CV_SAMPLES_PER_STUDENT", default=20, cast=int)

# LBPH recognizer parameters. LBPH reports a *distance* (lower = better
# match), not a percentage, so CV_RECOGNITION_DISTANCE_THRESHOLD is the
# cutoff below which a match is trusted. Anything above it is "Unknown".
CV_RECOGNITION_DISTANCE_THRESHOLD = config(
    "CV_RECOGNITION_DISTANCE_THRESHOLD", default=75.0, cast=float
)

# How many consecutive recognitions of the same student (across polled
# frames) are required before attendance is actually written to the
# database. This is a simple temporal-smoothing guard against a single
# lucky/unlucky frame flipping a decision.
CV_CONSECUTIVE_MATCHES_REQUIRED = config(
    "CV_CONSECUTIVE_MATCHES_REQUIRED", default=3, cast=int
)

# Laplacian-variance threshold used to reject blurry/low-quality frames
# before they ever reach the recognizer.
CV_BLUR_VARIANCE_THRESHOLD = config("CV_BLUR_VARIANCE_THRESHOLD", default=40.0, cast=float)

# Default OS camera index used by the standalone scripts (scripts/*.py).
# Not used by the web app, which captures frames in the browser.
CV_CAMERA_INDEX = config("CV_CAMERA_INDEX", default=0, cast=int)
