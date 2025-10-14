# project/urls.py
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("admin/", admin.site.urls),

    # Twilio webhooks + helpers (voice, status, call-me, debug)
    # Assumes myApp/urls.py defines: app_name = "twilio"
    path("twilio/", include(("myApp.urls", "twilio"), namespace="twilio")),

    # Your API routes
    path("api/", include("myApp.api_urls")),
]

# Serve static files in development and production
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    # Also serve files from STATICFILES_DIRS
    from django.contrib.staticfiles.urls import staticfiles_urlpatterns
    urlpatterns += staticfiles_urlpatterns()
