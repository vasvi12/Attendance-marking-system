"""
Test suite for the Computer Vision Attendance System.

Covers:
  * Model behavior (Student, Attendance, the per-day uniqueness constraint)
  * Form validation
  * Views (dashboard, students, attendance, URL routing)
  * The parts of the CV pipeline that do not require a physical webcam
    (preprocessing, blur detection, and face detection on synthetic images)

Tests that would require an actual camera device (VideoCamera.open/read
against real hardware) are intentionally not included - see the README's
"Testing" section for why, and how the standalone scripts fill that gap
for manual verification.
"""

from __future__ import annotations

import base64
import json
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
from django.db import IntegrityError, transaction
from django.test import Client, TestCase
from django.urls import reverse
from django.utils import timezone

from .forms import StudentForm
from .models import Attendance, Student
from .services import attendance_service
from .services.face_detection import FaceDetector
from .services.face_recognition import (
    FaceRecognitionService,
    ModelNotTrainedError,
    blur_variance,
    preprocess_face,
    save_face_sample,
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class StudentModelTests(TestCase):
    def test_create_student(self):
        student = Student.objects.create(
            student_id="CS001", name="Asha Rao", email="asha@example.com"
        )
        self.assertEqual(str(student), "Asha Rao (CS001)")
        self.assertTrue(student.is_active)

    def test_student_id_must_be_unique(self):
        Student.objects.create(student_id="CS001", name="Asha Rao")
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                Student.objects.create(student_id="CS001", name="Someone Else")

    def test_sample_count_zero_when_no_directory(self):
        student = Student.objects.create(student_id="CS002", name="Bilal Khan")
        self.assertEqual(student.sample_count, 0)
        self.assertFalse(student.is_enrolled)


class AttendanceModelTests(TestCase):
    def setUp(self):
        self.student = Student.objects.create(student_id="CS003", name="Chen Wei")

    def test_create_attendance(self):
        record = Attendance.objects.create(student=self.student, confidence=87.5)
        self.assertEqual(record.status, Attendance.Status.PRESENT)
        self.assertEqual(record.date, timezone.localdate())

    def test_duplicate_attendance_same_day_rejected_at_db_level(self):
        Attendance.objects.create(student=self.student)
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                Attendance.objects.create(student=self.student)

    def test_attendance_allowed_on_different_days(self):
        yesterday = timezone.localdate() - timezone.timedelta(days=1)
        Attendance.objects.create(student=self.student, date=yesterday)
        Attendance.objects.create(student=self.student)  # today
        self.assertEqual(Attendance.objects.filter(student=self.student).count(), 2)


# ---------------------------------------------------------------------------
# Services
# ---------------------------------------------------------------------------


class AttendanceServiceTests(TestCase):
    def setUp(self):
        self.student = Student.objects.create(student_id="CS004", name="Dana Ibrahim")

    def test_mark_attendance_creates_record(self):
        result = attendance_service.mark_attendance(self.student, confidence=91.2)
        self.assertTrue(result.created)
        self.assertEqual(Attendance.objects.count(), 1)
        self.assertEqual(result.attendance.confidence, 91.2)

    def test_mark_attendance_is_idempotent_for_same_day(self):
        first = attendance_service.mark_attendance(self.student, confidence=91.2)
        second = attendance_service.mark_attendance(self.student, confidence=95.0)

        self.assertTrue(first.created)
        self.assertFalse(second.created)
        self.assertEqual(Attendance.objects.count(), 1)
        # The original confidence is preserved; a later "detection" of the
        # same student on the same day does not overwrite the record.
        self.assertEqual(Attendance.objects.first().confidence, 91.2)

    def test_dashboard_stats_percentage(self):
        Student.objects.create(student_id="CS005", name="Second Student")
        attendance_service.mark_attendance(self.student)

        stats = attendance_service.get_dashboard_stats()
        self.assertEqual(stats["total_students"], 2)
        self.assertEqual(stats["today_attendance"], 1)
        self.assertEqual(stats["attendance_percentage"], 50.0)

    def test_dashboard_stats_no_students_no_division_error(self):
        Student.objects.all().delete()
        stats = attendance_service.get_dashboard_stats()
        self.assertEqual(stats["attendance_percentage"], 0.0)


# ---------------------------------------------------------------------------
# Forms
# ---------------------------------------------------------------------------


class StudentFormTests(TestCase):
    def test_valid_form(self):
        form = StudentForm(data={"student_id": "CS006", "name": "Elena Petrova", "email": ""})
        self.assertTrue(form.is_valid())

    def test_blank_name_rejected(self):
        form = StudentForm(data={"student_id": "CS007", "name": "  ", "email": ""})
        self.assertFalse(form.is_valid())
        self.assertIn("name", form.errors)

    def test_missing_student_id_rejected(self):
        form = StudentForm(data={"student_id": "", "name": "Farah Noor", "email": ""})
        self.assertFalse(form.is_valid())
        self.assertIn("student_id", form.errors)

    def test_invalid_email_rejected(self):
        form = StudentForm(
            data={"student_id": "CS008", "name": "Gabriel Silva", "email": "not-an-email"}
        )
        self.assertFalse(form.is_valid())
        self.assertIn("email", form.errors)


# ---------------------------------------------------------------------------
# Views / URL routing
# ---------------------------------------------------------------------------


class ViewTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.student = Student.objects.create(student_id="CS009", name="Hana Kim")

    def test_dashboard_response(self):
        response = self.client.get(reverse("attendance:dashboard"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "attendance/dashboard.html")
        self.assertContains(response, "Dashboard")

    def test_student_list_response(self):
        response = self.client.get(reverse("attendance:student_list"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Hana Kim")

    def test_student_create_get(self):
        response = self.client.get(reverse("attendance:student_create"))
        self.assertEqual(response.status_code, 200)

    def test_student_create_post_valid(self):
        response = self.client.post(
            reverse("attendance:student_create"),
            {"student_id": "CS010", "name": "Ivan Petrov", "email": ""},
        )
        self.assertEqual(response.status_code, 302)
        self.assertTrue(Student.objects.filter(student_id="CS010").exists())

    def test_student_create_post_invalid_shows_errors(self):
        response = self.client.post(
            reverse("attendance:student_create"), {"student_id": "", "name": "", "email": ""}
        )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(Student.objects.filter(name="").exists())

    def test_student_capture_page(self):
        response = self.client.get(
            reverse("attendance:student_capture", args=[self.student.pk])
        )
        self.assertEqual(response.status_code, 200)

    def test_student_capture_page_404_for_missing_student(self):
        response = self.client.get(reverse("attendance:student_capture", args=[999999]))
        self.assertEqual(response.status_code, 404)

    def test_attendance_list_response(self):
        response = self.client.get(reverse("attendance:attendance_list"))
        self.assertEqual(response.status_code, 200)

    def test_attendance_list_filter_by_student(self):
        attendance_service.mark_attendance(self.student)
        response = self.client.get(
            reverse("attendance:attendance_list"), {"student": self.student.pk}
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Hana Kim")

    def test_live_attendance_page(self):
        response = self.client.get(reverse("attendance:live_attendance"))
        self.assertEqual(response.status_code, 200)

    def test_train_recognizer_without_samples_returns_error(self):
        response = self.client.post(reverse("attendance:train_recognizer"))
        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.json()["success"])

    def test_capture_sample_requires_post(self):
        response = self.client.get(
            reverse("attendance:capture_face_sample", args=[self.student.pk])
        )
        self.assertEqual(response.status_code, 405)

    def test_process_live_frame_without_model_reports_untrained(self):
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        ok, buffer = cv2.imencode(".jpg", blank)
        self.assertTrue(ok)
        data_url = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("ascii")

        response = self.client.post(
            reverse("attendance:process_live_frame"),
            data=json.dumps({"image": data_url}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body["success"])
        self.assertIn("no trained recognition model", body["message"].lower())


# ---------------------------------------------------------------------------
# Computer vision pipeline (no webcam required)
# ---------------------------------------------------------------------------


class FaceDetectionTests(TestCase):
    def test_detector_loads_a_backend(self):
        detector = FaceDetector()
        self.assertIn(detector.backend, {"dnn", "haar"})

    def test_detect_faces_on_blank_frame_returns_empty(self):
        detector = FaceDetector()
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        self.assertEqual(detector.detect_faces(blank), [])

    def test_detect_faces_handles_none_gracefully(self):
        detector = FaceDetector()
        self.assertEqual(detector.detect_faces(None), [])

    def test_min_face_size_filters_small_detections(self):
        detector = FaceDetector(min_face_size=500)
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        # Nothing this small could possibly pass a 500px minimum.
        self.assertEqual(detector.detect_faces(blank), [])


class FaceRecognitionPreprocessingTests(TestCase):
    def test_preprocess_face_output_shape(self):
        face = np.random.randint(0, 255, (120, 100, 3), dtype=np.uint8)
        processed = preprocess_face(face)
        self.assertEqual(processed.shape, (200, 200))

    def test_preprocess_face_rejects_empty_crop(self):
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            preprocess_face(empty)

    def test_blur_variance_is_higher_for_sharp_image(self):
        sharp = np.zeros((100, 100), dtype=np.uint8)
        sharp[::2, :] = 255  # high-frequency stripe pattern
        flat = np.full((100, 100), 128, dtype=np.uint8)
        self.assertGreater(blur_variance(sharp), blur_variance(flat))


class FaceRecognitionServiceTests(TestCase):
    """
    Exercises the full train -> save -> load -> recognize cycle against a
    temporary directory of synthetic (non-photographic) "face" images, so
    the LBPH wiring is verified without needing real photos or a webcam.
    """

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.faces_dir = self.tmp_dir / "faces"
        self.service = FaceRecognitionService(
            model_path=self.tmp_dir / "trained_model.yml",
            labels_path=self.tmp_dir / "labels.json",
            faces_dir=self.faces_dir,
            distance_threshold=90.0,
            blur_threshold=0.0,  # synthetic patterns are sharp enough already
        )
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)

    @staticmethod
    def _synthetic_face_bgr(seed: int) -> np.ndarray:
        """A deterministic, textured 200x200 BGR pattern - distinct per
        seed, standing in for a real face crop straight off the camera."""
        rng = np.random.default_rng(seed)
        gray = rng.integers(0, 255, size=(200, 200), dtype=np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def _write_samples(self, student_id: str, seed: int, count: int = 8) -> None:
        student_dir = self.faces_dir / student_id
        base = self._synthetic_face_bgr(seed)
        for i in range(count):
            # Small per-sample jitter, like slightly different frames of
            # the same person, while staying clearly distinct per student.
            # Goes through the exact same save_face_sample() helper the
            # real enrollment endpoint uses, so training data is
            # preprocessed identically to production.
            noise = np.random.default_rng(seed * 100 + i).integers(-10, 10, size=base.shape)
            sample = np.clip(base.astype(int) + noise, 0, 255).astype(np.uint8)
            save_face_sample(sample, student_dir, index=i + 1, blur_threshold=0.0)

    def test_recognize_before_training_raises(self):
        with self.assertRaises(ModelNotTrainedError):
            self.service.recognize(np.zeros((200, 200, 3), dtype=np.uint8))

    def test_train_with_no_images_raises(self):
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        with self.assertRaises(ValueError):
            self.service.train_from_directory()

    def test_train_and_recognize_round_trip(self):
        self._write_samples("STU-A", seed=1)
        self._write_samples("STU-B", seed=2)

        stats = self.service.train_from_directory()
        self.assertEqual(stats["student_count"], 2)
        self.assertTrue(self.service.is_trained)

        # Feed back a frame close to STU-A's pattern and expect it to be
        # recognized as STU-A.
        probe_bgr = self._synthetic_face_bgr(1)
        result = self.service.recognize(probe_bgr)

        self.assertTrue(result.is_match)
        self.assertEqual(result.student_id, "STU-A")
        self.assertGreater(result.confidence, 0)
