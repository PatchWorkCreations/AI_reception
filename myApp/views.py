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
    # Pull helpful context from Twilio's webhook request
    call_sid = request.POST.get("CallSid", "") or request.GET.get("CallSid", "")
    from_num = request.POST.get("From", "") or request.GET.get("From", "")
    to_num   = request.POST.get("To", "") or request.GET.get("To", "")

    # Make sure the WS URL has the right path
    base_ws = (getattr(settings, "PUBLIC_WS_URL", "") or "").strip()
    if base_ws and not base_ws.endswith("/ws/twilio"):
        # allow both ".../ws" and bare host to resolve correctly
        if base_ws.endswith("/ws"):
            base_ws = f"{base_ws}/twilio"
        else:
            base_ws = f"{base_ws.rstrip('/')}/ws/twilio"

    public_http = (getattr(settings, "PUBLIC_HTTP_ORIGIN", "") or "").strip()
    status_cb   = f"{public_http.rstrip('/')}/twilio/status" if public_http else ""

    # Build TwiML
    # - track="inbound_audio" ensures a single stream of caller audio only
    # - name is optional, but handy for logs
    # - custom <Parameter> values show up in the WebSocket `start` event as customParameters.*
    # - statusCallbackEvent controls which events Twilio POSTs back (start/mark/stop are the useful ones)
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream
      url="{base_ws}"
      track="inbound_audio"
      name="neuromed-stream"
      statusCallback="{status_cb}"
      statusCallbackMethod="POST"
      statusCallbackEvent="start mark stop">
      <Parameter name="call_sid" value="{call_sid}"/>
      <Parameter name="from" value="{from_num}"/>
      <Parameter name="to" value="{to_num}"/>
    </Stream>
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
