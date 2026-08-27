"""
Database models for the Computer Vision Attendance System.

Two models cover the whole domain:

* ``Student``   - a person who can be enrolled and recognized.
* ``Attendance`` - one row per student per calendar day, written by the
  recognition pipeline (see attendance/services/attendance_service.py).

A unique constraint on (student, date) is the mechanism that guarantees a
student cannot accidentally receive two attendance records for the same
day, even if the recognizer fires on several frames in a row.
"""

from django.db import models
from django.utils import timezone


class Student(models.Model):
    """A student who can be enrolled for face recognition."""

    student_id = models.CharField(
        max_length=32,
        unique=True,
        help_text="Roll number / unique student identifier used for enrollment.",
    )
    name = models.CharField(max_length=150)
    email = models.EmailField(blank=True, null=True)
    is_active = models.BooleanField(
        default=True,
        help_text="Inactive students are excluded from recognition and reporting.",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return f"{self.name} ({self.student_id})"

    @property
    def face_samples_dir(self):
        """Filesystem path where this student's enrollment images live."""
        from django.conf import settings

        return settings.CV_FACES_DIR / self.student_id

    @property
    def sample_count(self):
        """Number of captured face-sample images on disk for this student."""
        directory = self.face_samples_dir
        if not directory.exists():
            return 0
        return len([f for f in directory.iterdir() if f.is_file()])

    @property
    def is_enrolled(self):
        """True once the student has at least one usable face sample."""
        return self.sample_count > 0


class Attendance(models.Model):
    """A single attendance record for one student on one day."""

    class Status(models.TextChoices):
        PRESENT = "PRESENT", "Present"
        LATE = "LATE", "Late"

    student = models.ForeignKey(
        Student, on_delete=models.CASCADE, related_name="attendance_records"
    )
    date = models.DateField(default=timezone.localdate)
    marked_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(
        max_length=10, choices=Status.choices, default=Status.PRESENT
    )
    confidence = models.FloatField(
        null=True,
        blank=True,
        help_text="Recognition confidence percentage (0-100) at the moment attendance was marked.",
    )

    class Meta:
        ordering = ["-date", "-marked_at"]
        constraints = [
            models.UniqueConstraint(
                fields=["student", "date"], name="unique_attendance_per_student_per_day"
            )
        ]
        indexes = [
            models.Index(fields=["date"]),
            models.Index(fields=["student", "date"]),
        ]

    def __str__(self):
        return f"{self.student.name} - {self.date} ({self.status})"
