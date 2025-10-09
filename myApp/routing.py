from django.urls import re_path
from .consumers import VoiceAIConsumer

websocket_urlpatterns = [
    re_path(r"ws/voice-ai/$", VoiceAIConsumer.as_asgi()),
]

