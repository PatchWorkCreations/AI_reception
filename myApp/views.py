# views.py
from django.conf import settings
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
# views.py
from django.conf import settings
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

from django.conf import settings
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def voice_answer(request):
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream
      url="{settings.PUBLIC_WS_URL}"
      statusCallback="{settings.PUBLIC_HTTP_ORIGIN}/twilio/status"
      statusCallbackMethod="POST" />
  </Connect>
</Response>""".strip()
    return HttpResponse(twiml, content_type="text/xml")




@csrf_exempt
def status_cb(request):
    import json
    payload = request.POST.dict() or json.loads(request.body or "{}")
    print("TWILIO STATUS ▶", payload)   # look for Event=start/media/stop and Reason
    return JsonResponse({"ok": True})



# views.py (or twilio_utils.py if you keep helpers there)
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from twilio.rest import Client

@csrf_exempt
def call_me(request):
    client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
    call = client.calls.create(
        to=settings.MY_PHONE_NUMBER,
        from_=settings.TWILIO_FROM_NUMBER,
        url=f"{settings.PUBLIC_HTTP_ORIGIN}/twilio/voice",
        status_callback=f"{settings.PUBLIC_HTTP_ORIGIN}/twilio/status",
        status_callback_event=["initiated","ringing","answered","completed"],
        status_callback_method="POST",
    )
    return JsonResponse({"sid": call.sid})



from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def ws_url_debug(request):
    """Return the WS URL Django thinks it should use."""
    return JsonResponse({
        "PUBLIC_WS_URL": settings.PUBLIC_WS_URL,
        "PUBLIC_HTTP_ORIGIN": settings.PUBLIC_HTTP_ORIGIN,
    })



# myApp/views.py
from django.shortcuts import render

def dev_client(request):
    # Optional: pass WS url from settings; or rely on the default in the HTML.
    return render(request, "dev_client.html", {
        "WS_URL": getattr(settings, "DEV_PUBLIC_WS_URL", "ws://localhost:8080/ws/twilio")
    })

def voice_ai_speaking(request):
    """Serve the simplified voice AI speaking interface."""
    return render(request, "talking_ai/speaking.html")


@csrf_exempt
def serve_preroll_wav(request):
    """Serve the pre-roll audio file for the voice bot."""
    import os
    from django.http import FileResponse
    
    # Path to the WAV file
    wav_path = os.path.join(settings.BASE_DIR, 'myApp', 'static', 'neuromed_intro_G711.org_.wav')
    
    if not os.path.exists(wav_path):
        return HttpResponse("WAV file not found", status=404)
    
    # Serve the file with appropriate content type
    response = FileResponse(open(wav_path, 'rb'), content_type='audio/wav')
    response['Content-Disposition'] = 'inline; filename="neuromed_intro_G711.org_.wav"'
    return response