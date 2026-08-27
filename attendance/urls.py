from django.urls import path

from . import views

app_name = "attendance"

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("students/", views.student_list, name="student_list"),
    path("students/add/", views.student_create, name="student_create"),
    path("students/<int:student_id>/capture/", views.student_capture, name="student_capture"),
    path(
        "students/<int:student_id>/capture-sample/",
        views.capture_face_sample,
        name="capture_face_sample",
    ),
    path("train/", views.train_recognizer, name="train_recognizer"),
    path("attendance/", views.attendance_list, name="attendance_list"),
    path("live/", views.live_attendance, name="live_attendance"),
    path("live/process-frame/", views.process_live_frame, name="process_live_frame"),
]
