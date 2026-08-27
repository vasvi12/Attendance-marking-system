"""
Views for the Computer Vision Attendance System.

Views are kept intentionally thin: they parse the HTTP request, delegate to
``attendance.services`` for anything involving OpenCV or attendance rules,
and shape a response. No cv2 calls happen directly in this file.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging

import cv2
import numpy as np
from django.conf import settings
from django.contrib import messages
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_GET, require_POST

from .forms import AttendanceFilterForm, StudentForm
from .models import Attendance, Student
from .services import attendance_service
from .services.face_detection import FaceDetector
from .services.face_recognition import (
    FaceRecognitionService,
    FrameQualityError,
    ModelNotTrainedError,
    save_face_sample,
)

logger = logging.getLogger("attendance")

SESSION_STREAK_KEY = "recognition_streaks"

_detector_singleton: FaceDetector | None = None


def _get_detector() -> FaceDetector:
    """
    Loading the DNN net (when present) is the one moderately expensive part
    of building a FaceDetector, so it is cached at module scope and reused
    across requests within this process instead of rebuilt every call.
    """
    global _detector_singleton
    if _detector_singleton is None:
        _detector_singleton = FaceDetector(min_face_size=settings.CV_MIN_FACE_SIZE)
    return _detector_singleton


def _get_recognizer() -> FaceRecognitionService:
    return FaceRecognitionService(
        model_path=settings.CV_MODEL_PATH,
        labels_path=settings.CV_LABELS_PATH,
        faces_dir=settings.CV_FACES_DIR,
        distance_threshold=settings.CV_RECOGNITION_DISTANCE_THRESHOLD,
        blur_threshold=settings.CV_BLUR_VARIANCE_THRESHOLD,
    )


def _decode_base64_image(data_url: str) -> np.ndarray:
    """
    Decode a ``data:image/jpeg;base64,...`` string (as produced by
    ``canvas.toDataURL()`` in the browser) into a BGR OpenCV image.

    Raises ``ValueError`` on anything malformed - callers turn that into a
    400 response rather than a 500.
    """
    if "," in data_url:
        _, _, data_url = data_url.partition(",")
    try:
        raw_bytes = base64.b64decode(data_url)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid base64 image payload.") from exc

    array = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image data.")
    return image


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


@require_GET
def dashboard(request):
    stats = attendance_service.get_dashboard_stats()
    return render(request, "attendance/dashboard.html", {"stats": stats})


# ---------------------------------------------------------------------------
# Students
# ---------------------------------------------------------------------------


@require_GET
def student_list(request):
    students = Student.objects.all()
    return render(request, "attendance/students.html", {"students": students})


def student_create(request):
    if request.method == "POST":
        form = StudentForm(request.POST)
        if form.is_valid():
            student = form.save()
            messages.success(
                request, f"{student.name} registered. Now capture face samples."
            )
            return redirect("attendance:student_capture", student_id=student.pk)
    else:
        form = StudentForm()
    return render(request, "attendance/student_form.html", {"form": form})


@require_GET
def student_capture(request, student_id):
    student = get_object_or_404(Student, pk=student_id)
    return render(
        request,
        "attendance/capture_faces.html",
        {
            "student": student,
            "samples_target": settings.CV_SAMPLES_PER_STUDENT,
        },
    )


@require_POST
def capture_face_sample(request, student_id):
    """
    Receives one webcam frame (base64 JPEG) from the enrollment page,
    validates that exactly one face is visible, and saves it as the next
    numbered training sample for this student.
    """
    student = get_object_or_404(Student, pk=student_id)

    try:
        payload = json.loads(request.body)
        frame = _decode_base64_image(payload["image"])
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        return JsonResponse({"success": False, "message": str(exc)}, status=400)

    detector = _get_detector()
    faces = detector.detect_faces(frame)

    if not faces:
        return JsonResponse(
            {"success": False, "message": "No face detected. Face the camera directly."}
        )
    if len(faces) > 1:
        return JsonResponse(
            {"success": False, "message": "Multiple faces detected. Only one person at a time."}
        )

    face_box = faces[0]
    next_index = student.sample_count + 1

    try:
        save_face_sample(face_box.crop(frame), student.face_samples_dir, next_index)
    except FrameQualityError as exc:
        return JsonResponse({"success": False, "message": str(exc)})

    sample_count = student.sample_count
    return JsonResponse(
        {
            "success": True,
            "message": f"Sample {sample_count} captured.",
            "sample_count": sample_count,
            "target": settings.CV_SAMPLES_PER_STUDENT,
        }
    )


@require_POST
def train_recognizer(request):
    """Retrain the LBPH model from every enrolled student's saved samples."""
    recognizer = _get_recognizer()
    try:
        stats = recognizer.train_from_directory()
    except FrameQualityError as exc:
        return JsonResponse({"success": False, "message": str(exc)}, status=400)

    return JsonResponse(
        {
            "success": True,
            "message": (
                f"Trained on {stats['total_images']} samples across "
                f"{stats['student_count']} students."
            ),
            **stats,
        }
    )


# ---------------------------------------------------------------------------
# Attendance
# ---------------------------------------------------------------------------


@require_GET
def attendance_list(request):
    form = AttendanceFilterForm(request.GET or None)
    records = Attendance.objects.select_related("student")

    if form.is_valid():
        if form.cleaned_data.get("date"):
            records = records.filter(date=form.cleaned_data["date"])
        if form.cleaned_data.get("student"):
            records = records.filter(student=form.cleaned_data["student"])

    return render(
        request, "attendance/attendance.html", {"form": form, "records": records}
    )


# ---------------------------------------------------------------------------
# Live attendance
# ---------------------------------------------------------------------------


@require_GET
def live_attendance(request):
    recognizer = _get_recognizer()
    return render(
        request,
        "attendance/live_attendance.html",
        {"model_trained": recognizer.is_trained},
    )


@require_POST
def process_live_frame(request):
    """
    Receives one webcam frame (base64 JPEG) from the live-attendance page,
    detects every face in it, attempts to recognize each one, applies a
    consecutive-frame confirmation before writing attendance, and returns a
    JSON description of what was found for the browser to overlay on the
    video.
    """
    try:
        payload = json.loads(request.body)
        frame = _decode_base64_image(payload["image"])
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        return JsonResponse({"success": False, "message": str(exc)}, status=400)

    recognizer = _get_recognizer()
    if not recognizer.is_trained:
        return JsonResponse(
            {
                "success": False,
                "message": "No trained recognition model yet. Enroll students and "
                "train the recognizer first.",
                "faces": [],
            }
        )

    detector = _get_detector()
    faces = detector.detect_faces(frame)

    streaks = request.session.get(SESSION_STREAK_KEY, {})
    seen_this_frame = set()
    results = []

    for box in faces:
        entry = {
            "box": {"x": box.x, "y": box.y, "w": box.w, "h": box.h},
            "name": None,
            "student_id": None,
            "confidence": None,
            "status": "unknown",
            "message": "Unknown Face",
        }

        try:
            result = recognizer.recognize(box.crop(frame))
        except FrameQualityError:
            entry["status"] = "low_quality"
            entry["message"] = "Hold still"
            results.append(entry)
            continue
        except ModelNotTrainedError:
            entry["status"] = "not_trained"
            entry["message"] = "Recognizer not trained"
            results.append(entry)
            continue

        entry["confidence"] = round(result.confidence, 1)

        if not result.is_match:
            results.append(entry)
            continue

        student = Student.objects.filter(student_id=result.student_id, is_active=True).first()
        if student is None:
            results.append(entry)
            continue

        seen_this_frame.add(student.student_id)
        streak = streaks.get(student.student_id, 0) + 1
        streaks[student.student_id] = streak

        entry["name"] = student.name
        entry["student_id"] = student.student_id

        if streak >= settings.CV_CONSECUTIVE_MATCHES_REQUIRED:
            mark_result = attendance_service.mark_attendance(student, confidence=result.confidence)
            entry["status"] = "marked" if mark_result.created else "already_marked"
            entry["message"] = (
                "Attendance Marked" if mark_result.created else "Already Marked Today"
            )
        else:
            entry["status"] = "confirming"
            entry["message"] = f"Recognizing... ({streak}/{settings.CV_CONSECUTIVE_MATCHES_REQUIRED})"

        results.append(entry)

    # Any student not seen in this frame loses their streak, so a match has
    # to be genuinely consecutive rather than accumulated on and off.
    for student_id in list(streaks):
        if student_id not in seen_this_frame:
            del streaks[student_id]
    request.session[SESSION_STREAK_KEY] = streaks

    return JsonResponse({"success": True, "faces": results})
