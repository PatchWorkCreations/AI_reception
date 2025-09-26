from django.urls import path
from .views import voice_answer, status_cb, call_me, ws_url_debug

urlpatterns = [
    path("voice", voice_answer, name="twilio_voice"),
    path("status", status_cb, name="twilio_status"),
    path("call-me", call_me, name="twilio_call_me"),
    path("debug/ws-url", ws_url_debug, name="twilio_debug_ws_url"),
]
