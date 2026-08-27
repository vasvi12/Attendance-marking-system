from django.contrib import admin

from .models import Attendance, Student


@admin.register(Student)
class StudentAdmin(admin.ModelAdmin):
    list_display = ("student_id", "name", "email", "enrollment_status", "is_active", "created_at")
    list_filter = ("is_active", "created_at")
    search_fields = ("student_id", "name", "email")
    ordering = ("name",)
    readonly_fields = ("created_at", "updated_at")

    @admin.display(description="Enrollment")
    def enrollment_status(self, obj):
        count = obj.sample_count
        return f"Enrolled ({count} samples)" if count else "Not enrolled"


@admin.register(Attendance)
class AttendanceAdmin(admin.ModelAdmin):
    list_display = ("student", "date", "status", "confidence", "marked_at")
    list_filter = ("status", "date")
    search_fields = ("student__name", "student__student_id")
    date_hierarchy = "date"
    ordering = ("-date", "-marked_at")
    autocomplete_fields = ("student",)
