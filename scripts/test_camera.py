#!/usr/bin/env python3
"""
test_camera.py - quick sanity check that OpenCV can see your webcam.

Run this before trying the Django app's Live Attendance page. It opens the
requested camera, shows the live feed with any detected faces boxed in
green, and prints a clear message if the camera can't be reached at all.

Usage:
    python scripts/test_camera.py
    python scripts/test_camera.py --camera 1

Press 'q' (with the preview window focused) to exit, or Ctrl+C in the
terminal if no window is available.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the project importable when this script is run directly, e.g.
# `python scripts/test_camera.py` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from attendance.services.face_detection import CameraError, FaceDetector, VideoCamera  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify webcam access with OpenCV.")
    parser.add_argument(
        "--camera", type=int, default=0, help="Camera index to open (default: 0)."
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Don't try to open a preview window (useful on headless machines); "
        "just confirms frames can be read and prints detection counts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    detector = FaceDetector()
    print(f"Face detector backend: {detector.backend}")

    try:
        camera = VideoCamera(camera_index=args.camera)
        camera.open()
    except CameraError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Camera {args.camera} opened successfully. Press 'q' to quit.")

    try:
        frame_count = 0
        while True:
            try:
                frame = camera.read()
            except CameraError as exc:
                print(f"ERROR reading frame: {exc}", file=sys.stderr)
                return 1

            frame_count += 1
            faces = detector.detect_faces(frame)

            if args.no_window:
                if frame_count % 15 == 0:
                    print(f"frame {frame_count}: {len(faces)} face(s) detected")
                if frame_count >= 150:
                    print("Reached 150 frames in --no-window mode, exiting.")
                    return 0
                continue

            annotated = detector.draw_detections(frame, faces)
            cv2.putText(
                annotated,
                f"Faces: {len(faces)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            try:
                cv2.imshow("test_camera.py - press q to quit", annotated)
            except cv2.error as exc:
                print(
                    "ERROR: could not open a display window "
                    f"({exc}). Re-run with --no-window on headless systems.",
                    file=sys.stderr,
                )
                return 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
