# project/urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),

    # Twilio webhooks + helpers (voice, status, call-me, debug)
    # Assumes myApp/urls.py defines: app_name = "twilio"
    path("twilio/", include(("myApp.urls", "twilio"), namespace="twilio")),

    # Your API routes
    path("api/", include("myApp.api_urls")),
]
