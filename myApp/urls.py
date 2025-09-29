# myApp/urls.py
from django.urls import path
from .views import voice_answer, status_cb, call_me, ws_url_debug
from . import views

app_name = "twilio"

urlpatterns = [
    path("voice",  voice_answer, name="voice"),
    path("voice/", voice_answer),
    path("status",  status_cb, name="status"),
    path("status/", status_cb),
    path("call-me",  call_me, name="call_me"),
    path("call-me/", call_me),
    path("debug/ws-url",  ws_url_debug, name="ws_url_debug"),
    path("debug/ws-url/", ws_url_debug),


    path('dev/', views.dev_client, name='dev_client'),
]
