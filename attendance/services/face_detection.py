"""
Face detection.

This module is deliberately isolated from Django: it only depends on
OpenCV/NumPy, so it can be imported by the web app, by the standalone
``scripts/`` utilities, or by a test file, without ever starting Django.

Two detection backends are supported:

1. A DNN-based detector (OpenCV's "res10 300x300 SSD" Caffe model). This is
   noticeably more robust to head pose, camera angle and partial occlusion
   than a Haar cascade, which is exactly the kind of robustness this project
   cares about. It is NOT bundled with the repository (see the README's
   "Face Detection Model" section for exact download instructions) because
   committing binary model files to git is bad practice.
2. OpenCV's built-in Haar cascade frontal-face detector, which ships inside
   every opencv-python install (``cv2.data.haarcascades``) and requires no
   download at all.

``FaceDetector`` picks the DNN backend automatically when the model files
are present on disk, and falls back to the Haar cascade otherwise, logging
which backend is active. This means the system runs out of the box, and
gets more accurate if you take the extra step of downloading the DNN model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger("attendance")


@dataclass
class FaceBox:
    """A detected face's bounding box in pixel coordinates, plus a score."""

    x: int
    y: int
    w: int
    h: int
    confidence: float = 1.0

    def as_tuple(self):
        return (self.x, self.y, self.w, self.h)

    def crop(self, frame: np.ndarray) -> np.ndarray:
        """Return the pixel region of ``frame`` covered by this box."""
        height, width = frame.shape[:2]
        x1, y1 = max(0, self.x), max(0, self.y)
        x2, y2 = min(width, self.x + self.w), min(height, self.y + self.h)
        return frame[y1:y2, x1:x2]


class CameraError(RuntimeError):
    """Raised when the webcam cannot be opened or read from."""


class FaceDetector:
    """Detects faces in a BGR image using the best available backend."""

    DNN_INPUT_SIZE = (300, 300)
    DNN_SCORE_THRESHOLD = 0.6

    def __init__(self, min_face_size: int = 60):
        self.min_face_size = min_face_size
        self._net = None
        self._cascade = None
        self.backend = self._load_backend()

    # -- backend setup -----------------------------------------------------

    def _load_backend(self) -> str:
        """Try the DNN model first, fall back to Haar cascade. Never raises."""
        try:
            from django.conf import settings

            prototxt = settings.CV_DNN_PROTOTXT
            weights = settings.CV_DNN_WEIGHTS
            if prototxt.exists() and weights.exists():
                self._net = cv2.dnn.readNetFromCaffe(str(prototxt), str(weights))
                logger.info("FaceDetector: using DNN backend (%s)", weights.name)
                return "dnn"
        except Exception as exc:  # pragma: no cover - defensive, e.g. Django not configured
            logger.debug("FaceDetector: DNN backend unavailable (%s)", exc)

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():  # pragma: no cover - would indicate a broken OpenCV install
            raise RuntimeError(
                "Could not load the bundled Haar cascade classifier. "
                "Your OpenCV installation may be corrupted."
            )
        self._cascade = cascade
        logger.info("FaceDetector: using Haar cascade backend (fallback)")
        return "haar"

    # -- detection -----------------------------------------------------------

    def detect_faces(self, frame: np.ndarray) -> list[FaceBox]:
        """Detect faces in a BGR frame. Returns an empty list if none found."""
        if frame is None or frame.size == 0:
            return []

        if self.backend == "dnn":
            boxes = self._detect_dnn(frame)
        else:
            boxes = self._detect_haar(frame)

        return [b for b in boxes if b.w >= self.min_face_size and b.h >= self.min_face_size]

    def _detect_dnn(self, frame: np.ndarray) -> list[FaceBox]:
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, self.DNN_INPUT_SIZE, (104.0, 177.0, 123.0)
        )
        self._net.setInput(blob)
        detections = self._net.forward()

        boxes = []
        for i in range(detections.shape[2]):
            score = float(detections[0, 0, i, 2])
            if score < self.DNN_SCORE_THRESHOLD:
                continue
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append(FaceBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1, confidence=score))
        return boxes

    def _detect_haar(self, frame: np.ndarray) -> list[FaceBox]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        detections = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size),
        )
        return [FaceBox(x=int(x), y=int(y), w=int(w), h=int(h)) for (x, y, w, h) in detections]

    # -- visual feedback -----------------------------------------------------

    @staticmethod
    def draw_detections(frame: np.ndarray, boxes, labels: list[str] | None = None) -> np.ndarray:
        """Draw bounding boxes (and optional labels) onto a copy of ``frame``."""
        annotated = frame.copy()
        labels = labels or [None] * len(boxes)
        for box, label in zip(boxes, labels):
            x, y, w, h = box.as_tuple() if isinstance(box, FaceBox) else box
            color = (0, 200, 0) if label else (0, 165, 255)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            if label:
                cv2.putText(
                    annotated,
                    label,
                    (x, max(0, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )
        return annotated


class VideoCamera:
    """
    Thin, error-handling wrapper around ``cv2.VideoCapture``.

    Used by the standalone command-line utilities (``scripts/test_camera.py``
    and ``scripts/capture_faces.py``) that access the OS webcam directly. The
    Django web app does NOT use this class - it receives frames captured by
    the browser via JavaScript instead, so the server does not need direct
    camera access. See docs/architecture.md for why.
    """

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self._capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        capture = cv2.VideoCapture(self.camera_index)
        if not capture.isOpened():
            capture.release()
            raise CameraError(
                f"Could not open camera at index {self.camera_index}. "
                "Check that a webcam is connected, that no other application "
                "is using it, and that this process has camera permission."
            )
        self._capture = capture

    def read(self) -> np.ndarray:
        if self._capture is None:
            raise CameraError("Camera is not open. Call open() first.")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise CameraError("Failed to read a frame from the camera.")
        return frame

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "VideoCamera":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
