from django import forms

from .models import Student


class StudentForm(forms.ModelForm):
    """Registration form for STEP 6 (student enrollment)."""

    class Meta:
        model = Student
        fields = ["student_id", "name", "email"]
        widgets = {
            "student_id": forms.TextInput(
                attrs={"placeholder": "e.g. CS2024-041", "autofocus": True}
            ),
            "name": forms.TextInput(attrs={"placeholder": "Full name"}),
            "email": forms.EmailInput(attrs={"placeholder": "optional@example.com"}),
        }

    def clean_student_id(self):
        student_id = self.cleaned_data["student_id"].strip()
        if not student_id:
            raise forms.ValidationError("Student ID cannot be blank.")
        return student_id

    def clean_name(self):
        name = self.cleaned_data["name"].strip()
        if len(name) < 2:
            raise forms.ValidationError("Enter the student's full name.")
        return name


class AttendanceFilterForm(forms.Form):
    """Filters for the attendance list page (STEP 8: 'Attendance')."""

    date = forms.DateField(required=False, widget=forms.DateInput(attrs={"type": "date"}))
    student = forms.ModelChoiceField(
        queryset=Student.objects.all(), required=False, empty_label="All students"
    )
