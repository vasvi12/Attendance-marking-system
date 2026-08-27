#!/usr/bin/env python3
"""
capture_faces.py - standalone command-line face enrollment.

The primary way to enroll a student is the web UI (Add Student -> capture
page), which uses the browser's camera. This script is an alternative for
enrolling directly from the machine that owns the webcam - handy when
setting up many students in a row at a classroom PC, without a browser tab
open.

It saves samples to the exact same location the web app reads from
(``settings.CV_FACES_DIR/<student_id>/``), so images captured here are
picked up by the "Retrain Recognizer" action with no extra steps.

Usage:
    python scripts/capture_faces.py --student-id CS2024-041 --name "Asha Rao"
    python scripts/capture_faces.py --student-id CS2024-041 --samples 25 --camera 1
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django  # noqa: E402

django.setup()

from django.conf import settings  # noqa: E402

from attendance.models import Student  # noqa: E402
from attendance.services.face_detection import CameraError, FaceDetector, VideoCamera  # noqa: E402
from attendance.services.face_recognition import FrameQualityError, save_face_sample  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enroll a student's face samples via webcam.")
    parser.add_argument("--student-id", required=True, help="Student ID / roll number.")
    parser.add_argument(
        "--name", default=None, help="Full name (used to create the Student record if it "
        "doesn't already exist)."
    )
    parser.add_argument("--camera", type=int, default=None, help="Camera index (default from settings).")
    parser.add_argument(
        "--samples", type=int, default=None, help="Number of samples to capture (default from settings)."
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Seconds to wait between automatic captures (default: 1.0).",
    )
    return parser.parse_args()


def get_or_create_student(student_id: str, name: str | None) -> Student:
    student = Student.objects.filter(student_id=student_id).first()
    if student:
        return student
    if not name:
        print(
            f"No existing student with ID '{student_id}'. Pass --name to create one.",
            file=sys.stderr,
        )
        sys.exit(1)
    student = Student.objects.create(student_id=student_id, name=name)
    print(f"Created new student record: {student}")
    return student


def main() -> int:
    args = parse_args()
    student = get_or_create_student(args.student_id, args.name)

    camera_index = args.camera if args.camera is not None else settings.CV_CAMERA_INDEX
    target_samples = args.samples if args.samples is not None else settings.CV_SAMPLES_PER_STUDENT

    detector = FaceDetector(min_face_size=settings.CV_MIN_FACE_SIZE)
    print(f"Face detector backend: {detector.backend}")

    try:
        camera = VideoCamera(camera_index=camera_index)
        camera.open()
    except CameraError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    dest_dir = student.face_samples_dir
    existing = student.sample_count
    print(f"Capturing samples for {student.name} ({student.student_id}) -> {dest_dir}")
    print(f"Already have {existing} sample(s). Target: {target_samples}.")
    print("Press Ctrl+C at any time to stop early.")

    captured = 0
    index = existing
    try:
        while existing + captured < target_samples:
            try:
                frame = camera.read()
            except CameraError as exc:
                print(f"ERROR reading frame: {exc}", file=sys.stderr)
                return 1

            faces = detector.detect_faces(frame)
            if len(faces) != 1:
                print(f"  ... waiting for exactly one face (currently see {len(faces)})")
                time.sleep(args.interval)
                continue

            index += 1
            try:
                path = save_face_sample(faces[0].crop(frame), dest_dir, index)
            except FrameQualityError as exc:
                print(f"  skipped: {exc}")
                index -= 1
                time.sleep(args.interval)
                continue

            captured += 1
            print(f"  saved {path.name} ({existing + captured}/{target_samples})")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped early by user.")
    finally:
        camera.release()

    print(f"Done. {existing + captured} total sample(s) saved for {student.student_id}.")
    print("Run the 'Retrain Recognizer' action on the Students page (or /train/) to use them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
