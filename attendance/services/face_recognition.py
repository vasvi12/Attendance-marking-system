"""
Face preprocessing + recognition.

Recognition uses OpenCV's LBPH (Local Binary Patterns Histograms) algorithm
via ``cv2.face.LBPHFaceRecognizer``. LBPH was chosen deliberately over a
deep-learning embedding model:

* It trains directly from the enrolled students' images, locally, in
  seconds, with no GPU and no pretrained weights to download - there is no
  model file to fabricate or fake.
* It ships inside ``opencv-contrib-python``, so `pip install -r
  requirements.txt` is genuinely all a developer needs to run recognition.
* Its accuracy ceiling is lower than a modern embedding network (see the
  README's "Limitations" section) - that trade-off is explicit and
  documented, not hidden.

The pipeline this module implements is:

    face crop -> grayscale -> resize -> CLAHE -> LBPH predict -> distance

A lower LBPH distance means a closer match. ``settings.CV_RECOGNITION_DISTANCE_THRESHOLD``
is the cutoff beyond which a prediction is reported as "Unknown" rather than
trusted.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("attendance")

FACE_SIZE = (200, 200)


@dataclass
class RecognitionResult:
    """Outcome of running one face crop through the recognizer."""

    student_id: str | None
    confidence: float  # 0-100, higher is better
    distance: float | None  # raw LBPH distance, lower is better; None if unrecognized

    @property
    def is_match(self) -> bool:
        return self.student_id is not None


class ModelNotTrainedError(RuntimeError):
    """Raised when recognition is attempted before any model has been trained."""


class FrameQualityError(ValueError):
    """Raised when a captured frame/crop is unusable (too blurry, too small, etc.)."""


def preprocess_face(face_bgr: np.ndarray) -> np.ndarray:
    """
    Turn a raw face crop into the normalized grayscale image the recognizer
    expects: grayscale -> resize to a fixed size -> CLAHE contrast
    normalization. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    is what gives the system most of its tolerance to classroom lighting
    that is uneven, too dim, or backlit.
    """
    if face_bgr is None or face_bgr.size == 0:
        raise FrameQualityError("Empty face crop.")

    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, FACE_SIZE, interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(resized)


def blur_variance(gray_image: np.ndarray) -> float:
    """
    Laplacian variance is a standard, cheap "is this image sharp?" measure:
    a blurry image has few sharp edges, so its Laplacian has low variance.
    Used to reject motion-blurred or out-of-focus frames before they reach
    the recognizer, where they would otherwise just produce noisy guesses.
    """
    return float(cv2.Laplacian(gray_image, cv2.CV_64F).var())


def save_face_sample(
    face_bgr: np.ndarray, dest_dir: Path, index: int, blur_threshold: float = 40.0
) -> Path:
    """
    Preprocess a face crop captured during enrollment and save it to disk as
    the next numbered sample for a student.

    Samples are stored *already preprocessed* (grayscale, resized, CLAHE-
    normalized) so that ``FaceRecognitionService.train_from_directory`` and
    ``FaceRecognitionService.recognize`` apply the exact same transform -
    training on raw images while recognizing on CLAHE-normalized ones would
    silently hurt accuracy.
    """
    processed = preprocess_face(face_bgr)
    quality = blur_variance(processed)
    if quality < blur_threshold:
        raise FrameQualityError(
            f"Sample too blurry to keep (sharpness={quality:.1f}). Hold still and retry."
        )

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"sample_{index:03d}.jpg"
    cv2.imwrite(str(dest_path), processed)
    return dest_path


class FaceRecognitionService:
    """Loads/trains an LBPH model and recognizes preprocessed face crops."""

    def __init__(
        self,
        model_path: Path,
        labels_path: Path,
        faces_dir: Path,
        distance_threshold: float = 75.0,
        blur_threshold: float = 40.0,
    ):
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        self.faces_dir = Path(faces_dir)
        self.distance_threshold = distance_threshold
        self.blur_threshold = blur_threshold

        self._recognizer = None
        self._label_to_student: dict[int, str] = {}

    # -- persistence -----------------------------------------------------

    @property
    def is_trained(self) -> bool:
        return self.model_path.exists() and self.labels_path.exists()

    def _load(self) -> None:
        if self._recognizer is not None:
            return
        if not self.is_trained:
            raise ModelNotTrainedError(
                "No trained recognition model was found. Enroll at least one "
                "student with face samples and train the recognizer first "
                "(see the 'Retrain Recognizer' action on the Students page)."
            )
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(str(self.model_path))
        with open(self.labels_path, encoding="utf-8") as fh:
            raw = json.load(fh)
        self._label_to_student = {int(k): v for k, v in raw.items()}
        self._recognizer = recognizer

    def reload(self) -> None:
        """Force the next recognition call to re-read the model from disk."""
        self._recognizer = None
        self._label_to_student = {}

    # -- training -----------------------------------------------------

    def train_from_directory(self) -> dict:
        """
        Rebuild the LBPH model from every image under
        ``<faces_dir>/<student_id>/*.jpg``. Safe to call repeatedly (e.g.
        after each new enrollment) - it always retrains from scratch, which
        for LBPH on a classroom-sized dataset takes well under a second per
        hundred images.

        Returns a small summary dict; raises ``FrameQualityError`` if there
        are no usable images to train on at all.
        """
        if not self.faces_dir.exists():
            raise FrameQualityError(
                f"No enrollment images found at {self.faces_dir}. "
                "Enroll at least one student before training."
            )

        images: list[np.ndarray] = []
        labels: list[int] = []
        label_to_student: dict[int, str] = {}
        student_to_label: dict[str, int] = {}
        per_student_counts: dict[str, int] = {}

        student_dirs = sorted(p for p in self.faces_dir.iterdir() if p.is_dir())
        for student_dir in student_dirs:
            student_id = student_dir.name
            image_paths = sorted(student_dir.glob("*.jpg"))
            if not image_paths:
                continue

            label = student_to_label.setdefault(student_id, len(student_to_label))
            label_to_student[label] = student_id

            for image_path in image_paths:
                raw = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if raw is None:
                    logger.warning("Skipping unreadable enrollment image: %s", image_path)
                    continue
                if raw.shape[:2] != FACE_SIZE[::-1]:
                    raw = cv2.resize(raw, FACE_SIZE, interpolation=cv2.INTER_AREA)
                images.append(raw)
                labels.append(label)
                per_student_counts[student_id] = per_student_counts.get(student_id, 0) + 1

        if not images:
            raise FrameQualityError(
                "Found student folders but no readable .jpg samples inside them. "
                "Re-run enrollment/capture before training."
            )

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(images, np.array(labels))

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        recognizer.write(str(self.model_path))
        with open(self.labels_path, "w", encoding="utf-8") as fh:
            json.dump({str(k): v for k, v in label_to_student.items()}, fh, indent=2)

        self.reload()
        logger.info(
            "Trained LBPH recognizer on %d images across %d students",
            len(images),
            len(student_to_label),
        )
        return {
            "total_images": len(images),
            "student_count": len(student_to_label),
            "per_student_counts": per_student_counts,
        }

    # -- recognition -----------------------------------------------------

    def recognize(self, face_bgr: np.ndarray) -> RecognitionResult:
        """
        Run one face crop through preprocessing + LBPH prediction.

        Raises ``ModelNotTrainedError`` if no model exists yet, and
        ``FrameQualityError`` if the crop is too blurry to trust. Both are
        expected, recoverable conditions the caller (the live-attendance
        view) is expected to catch and surface to the user rather than
        letting the request crash.
        """
        self._load()

        processed = preprocess_face(face_bgr)
        quality = blur_variance(processed)
        if quality < self.blur_threshold:
            raise FrameQualityError(
                f"Frame too blurry to recognize reliably (sharpness={quality:.1f})."
            )

        label, distance = self._recognizer.predict(processed)
        confidence = max(0.0, 100.0 - distance)

        if distance > self.distance_threshold:
            return RecognitionResult(student_id=None, confidence=confidence, distance=distance)

        student_id = self._label_to_student.get(label)
        return RecognitionResult(student_id=student_id, confidence=confidence, distance=distance)
