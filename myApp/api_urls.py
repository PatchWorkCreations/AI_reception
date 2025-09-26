from django.urls import path
from .api import faq, book_appointment

urlpatterns = [
    path("faq", faq, name="faq"),
    path("book", book_appointment, name="book"),  # returns not_enabled note in voice-only mode
]
