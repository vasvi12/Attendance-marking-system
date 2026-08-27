"""
Business logic for turning a recognition result into an attendance record,
and for the aggregate numbers the dashboard shows.

Kept separate from views.py on purpose: views handle HTTP, this module
handles rules ("what counts as marking someone present", "how do we compute
today's percentage"). That split is what STEP 18 of the project brief asks
for ("business logic is separated from UI logic").
"""

from __future__ import annotations

from dataclasses import dataclass

from django.db import IntegrityError, transaction
from django.utils import timezone

from attendance.models import Attendance, Student


@dataclass
class MarkAttendanceResult:
    attendance: Attendance | None
    created: bool
    reason: str = ""


def mark_attendance(student: Student, confidence: float | None = None) -> MarkAttendanceResult:
    """
    Record ``student`` as present for today, unless they already have a
    record for today.

    The (student, date) unique constraint on ``Attendance`` is the actual
    source of truth for "no duplicates" - this function just wraps a
    get-or-create around it so a duplicate call (e.g. the recognizer firing
    on the student in five consecutive frames) is a harmless no-op instead
    of an error.
    """
    today = timezone.localdate()
    try:
        with transaction.atomic():
            attendance, created = Attendance.objects.get_or_create(
                student=student,
                date=today,
                defaults={"confidence": confidence},
            )
    except IntegrityError:
        # Extremely unlikely race (two near-simultaneous requests), but the
        # unique constraint guarantees correctness either way - just fetch
        # the row that won.
        attendance = Attendance.objects.get(student=student, date=today)
        created = False

    reason = "marked" if created else "already marked today"
    return MarkAttendanceResult(attendance=attendance, created=created, reason=reason)


def get_dashboard_stats() -> dict:
    """Aggregate numbers shown on the dashboard's stat cards."""
    today = timezone.localdate()
    total_students = Student.objects.filter(is_active=True).count()
    today_attendance = Attendance.objects.filter(date=today).count()
    percentage = round((today_attendance / total_students) * 100, 1) if total_students else 0.0

    recent = (
        Attendance.objects.select_related("student")
        .order_by("-marked_at")[:10]
    )

    return {
        "total_students": total_students,
        "today_attendance": today_attendance,
        "attendance_percentage": percentage,
        "recent_attendance": recent,
        "today": today,
    }
