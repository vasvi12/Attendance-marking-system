# Computer Vision Attendance System

A real-time classroom attendance system that uses computer vision to detect
and recognize students from a webcam feed and automatically record their
attendance, replacing manual roll calls.

## Overview

A teacher opens the **Live Attendance** page, points a webcam at the room,
and the system continuously detects faces in the video feed, recognizes
enrolled students, and marks them present in the database the moment it is
confident about a match - no manual data entry. Students are enrolled once
through a guided face-capture flow, and everything after that (detection,
recognition, attendance logging, dashboards, admin tooling) runs on top of
Django + OpenCV.

## Why I built it

Manual classroom attendance is slow, interruptible, and easy to game (one
student answering for another). It doesn't have to be: a webcam and a
laptop are usually already in the room. The interesting engineering problem
here was never "can OpenCV find a face in a picture" - that's a few lines
of code. The real problem is making recognition **hold up outside a
perfectly lit demo**: a classroom has mixed lighting, students at different
distances and angles from the camera, several faces in frame at once, and
people who simply forgot they were being enrolled and are mid-turn or
mid-blink. This project is built around that problem, not around it.

## Features

- Student registration with a guided, browser-based face-capture flow (no
  extra software - it uses `getUserMedia()`).
- Local, from-scratch face recognition training (OpenCV LBPH) - no
  pretrained model download required to get started.
- Optional higher-accuracy DNN face detector, auto-detected if you add the
  model files, with automatic fallback to OpenCV's bundled Haar cascade if
  you don't.
- Real-time multi-face detection and recognition against a live webcam feed,
  with on-screen bounding boxes, names, and confidence.
- Automatic attendance logging with a hard database constraint preventing
  duplicate attendance for the same student on the same day.
- Consecutive-frame confirmation before marking attendance, to avoid a
  single noisy frame producing a false mark.
- Preprocessing aimed at real-world robustness: CLAHE contrast
  normalization, blur/quality rejection, minimum face size filtering.
- Dashboard with live stats (total students, today's attendance, attendance
  percentage, recent activity).
- Filterable attendance log (by date and/or student).
- Configured Django admin for managing students and attendance directly.
- Standalone CLI utilities for camera testing and face enrollment without
  the browser.
- A real Django test suite (37 tests) covering models, forms, views, URL
  routing, and the parts of the CV pipeline that don't require a physical
  camera - including a full synthetic train -> recognize round trip.

## Tech Stack

Python, OpenCV (`opencv-contrib-python`), Django, SQLite (via Django ORM),
HTML, CSS, and vanilla JavaScript (no frontend framework/build step).

## Architecture

```text
Camera (browser getUserMedia)
   |
   v
OpenCV (server-side, via a base64 JPEG frame POST)
   |
   v
Face Detection        (DNN if available, else Haar cascade)
   |
   v
Face Preprocessing    (grayscale -> resize -> CLAHE)
   |
   v
Face Recognition      (OpenCV LBPH)
   |
   v
Student Identification (distance threshold -> confidence)
   |
   v
Attendance Service    (consecutive-frame check -> get_or_create)
   |
   v
Django ORM
   |
   v
SQL Database (SQLite)
   |
   v
Web Dashboard
```

See [`docs/architecture.md`](docs/architecture.md) for the full breakdown,
including sequence diagrams for enrollment and live recognition, the data
model, and how errors are handled at each stage.

## Installation

```bash
git clone <repository-url>
cd Attendance-marking-system
python -m venv venv
```

Activate the virtual environment:

**macOS/Linux**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

Then install dependencies and set up the database:

```bash
pip install -r requirements.txt
cp .env.example .env
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

Open `http://127.0.0.1:8000/` for the app, or `http://127.0.0.1:8000/admin/`
for the admin panel.

### Face Detection Model (optional, recommended)

Out of the box, face detection uses OpenCV's bundled Haar cascade - no
download needed. For meaningfully better robustness to head pose and camera
angle, you can add OpenCV's standard DNN face detector (SSD ResNet10,
Caffe). Download these two files into `data/models/` (already gitignored):

```bash
mkdir -p data/models
curl -L -o data/models/deploy.prototxt \
  https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
curl -L -o data/models/res10_300x300_ssd_iter_140000.caffemodel \
  https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
```

These are OpenCV's own publicly distributed sample model files (~10 MB
total), not something this project trained. If they're absent, the system
logs that it's falling back to the Haar cascade and works normally with no
other changes needed.

## Usage

1. Start the server: `python manage.py runserver`.
2. Open the **Dashboard** to see current stats.
3. Go to **Add Student**, fill in the student's details, and submit - you're
   taken straight to the face-capture page.
4. On the capture page, allow camera access and click **Capture Sample**
   (or **Auto-Capture**) until you reach the target sample count, varying
   angle/lighting slightly between captures.
5. Go to **Students** and click **Retrain Recognizer** once you've enrolled
   everyone you want recognized.
6. Go to **Live Attendance**, click **Start Attendance**, and point the
   camera at the room. Recognized students are marked present automatically
   after a few confirming frames; unrecognized faces are labeled "Unknown".
7. Review records any time on the **Attendance** page, filterable by date
   or student, or in the Django admin.

### Command-line utilities

```bash
# Verify OpenCV can see your webcam before trying the web app
python scripts/test_camera.py

# Enroll a student directly from the machine with the webcam attached,
# without going through the browser
python scripts/capture_faces.py --student-id CS2024-041 --name "Asha Rao"
```

## Project Structure

```text
config/                Django project settings, root URLconf, WSGI/ASGI
attendance/
  models.py            Student, Attendance
  forms.py              StudentForm, AttendanceFilterForm
  views.py              Thin HTTP views; no OpenCV calls here
  urls.py
  admin.py               Django admin configuration
  tests.py               Full test suite
  services/
    face_detection.py    FaceDetector, VideoCamera (OpenCV, Django-independent)
    face_recognition.py  Preprocessing + LBPH training/recognition
    attendance_service.py Attendance business rules + dashboard stats
  management/commands/   Reserved for future custom manage.py commands
templates/                Base layout + all pages (dashboard, students, capture, attendance, live)
static/css/style.css      Hand-written design system (no framework)
static/js/app.js          Camera capture + fetch helpers shared by pages
scripts/
  test_camera.py          Standalone webcam sanity check
  capture_faces.py        Standalone CLI enrollment tool
data/                     Enrollment images, trained model, labels (gitignored)
docs/architecture.md      Full architecture writeup with diagrams
```

## Engineering Challenge

The hard part here was never "detect a
face in an image" - it's **staying reasonably reliable across lighting,
camera angle, face distance/pose, and multiple people in frame at once**,
which is what an actual classroom looks like. Concretely, this
implementation addresses that with:

- **Lighting** - CLAHE (adaptive histogram equalization) is applied to every
  face before both training and recognition, so local contrast is
  normalized instead of relying on the raw pixel values a single global
  light source produced.
- **Camera angle / head pose** - the optional DNN detector is meaningfully
  more tolerant of off-angle faces than a Haar cascade; either way, LBPH
  itself is a texture-pattern method that degrades more gracefully with
  moderate pose change than raw pixel matching would.
- **Face distance** - `CV_MIN_FACE_SIZE` filters out detections too small to
  recognize reliably (i.e. someone far from the camera), instead of forcing
  a low-confidence guess.
- **Multiple people simultaneously** - detection returns every face in the
  frame, and each is preprocessed, recognized, and attendance-checked
  independently in the same request.
- **Noisy single frames** - a blur/sharpness check (Laplacian variance)
  rejects unusable frames outright, and a consecutive-frame requirement
  (`CV_CONSECUTIVE_MATCHES_REQUIRED`) means one lucky or unlucky frame can't
  flip an attendance decision by itself.
- **Unknown/unenrolled faces** - a configurable distance threshold
  (`CV_RECOGNITION_DISTANCE_THRESHOLD`) means low-confidence matches are
  reported as "Unknown" rather than mapped to the nearest enrolled student.

## Limitations

Being direct about this matters more than sounding impressive:

- Camera quality has a real, direct effect on recognition accuracy - a low
  resolution or poorly-focused webcam will underperform a good one.
- Extreme lighting (very dark rooms, strong backlighting) still degrades
  accuracy meaningfully; CLAHE helps, it doesn't solve lighting.
- Partial occlusion (masks, hands, hair covering the face) can cause missed
  detections or misrecognition.
- Recognition accuracy drops with steep head pose or very close/far
  distance from the camera, even with the DNN detector.
- LBPH is a classical (non deep-learning) method. It is fast and needs no
  pretrained weights, but its accuracy ceiling is lower than a modern face
  embedding network - see "Future Improvements" below.
- This system has no anti-spoofing: a printed photo or a phone screen
  showing a student's face can, in principle, fool it. It is not suitable
  as a security/access-control system.
- There is no built-in login/role separation for the main app (the Django
  admin has its own auth). Anyone who can reach the app can view records
  and enroll students.
- **This is a personal project, not a production biometric system.** Deploying anything like this for real would require explicit
  consent flows, a data retention policy, stronger access control, and
  likely legal review around biometric data handling, none of which is in
  scope here.

## Future Improvements

- Deep-learning face embeddings (e.g. a FaceNet/ArcFace-style model) for a
  meaningfully higher accuracy ceiling than LBPH.
- Better low-light handling (e.g. gamma correction or exposure-aware
  preprocessing ahead of CLAHE).
- Basic anti-spoofing (liveness checks such as blink detection).
- PostgreSQL for multi-classroom deployments at larger scale.
- Role-based authentication for teachers vs. administrators.

## Testing

```bash
python manage.py test attendance
```

37 tests cover model behavior (including the duplicate-attendance
constraint), form validation, every major view and URL route, and the parts
of the CV pipeline that don't require physical camera hardware - including
a full synthetic enroll -> train -> recognize round trip against the real
`FaceRecognitionService`. Tests that would require an actual webcam
(`VideoCamera.open()`/`.read()` against real hardware) are intentionally
excluded from the automated suite; use `scripts/test_camera.py` to verify
camera access by hand on a machine that has one.

This project was developed and verified in an environment with no physical
webcam attached - `python manage.py check`, `makemigrations`/`migrate`, the
full test suite, and the dev server were all run and passed there. The one
thing that genuinely cannot be verified without physical hardware is a live
webcam feed end-to-end; the code path for it (`VideoCamera`,
`getUserMedia`-based capture, `/live/process-frame/`) is exercised by the
synthetic tests and should be checked once against a real camera after
installation using `scripts/test_camera.py` followed by the Live Attendance
page.

## License

MIT - see [LICENSE](LICENSE).
