# views.py
from django.conf import settings
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse

@csrf_exempt
def voice_answer(request):
    twiml = f"""
<Response>
  <Start>
    <Stream
      url="{settings.PUBLIC_WS_URL}"
      track="both_tracks"
      statusCallback="{settings.PUBLIC_HTTP_ORIGIN}/twilio/status"
      statusCallbackMethod="POST"
      statusCallbackEvent="start media stop">
      <Parameter name="context" value="neuromed-reception"/>
    </Stream>
  </Start>
</Response>
""".strip()
    return HttpResponse(twiml, content_type="text/xml")


@csrf_exempt
def status_cb(request):
    payload = request.POST.dict() or json.loads(request.body or "{}")
    print("TWILIO STATUS ▶", payload)   # shows event: start/media/stop and reasons
    return JsonResponse({"ok": True})


# views.py (or twilio_utils.py if you keep helpers there)
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from twilio.rest import Client

@csrf_exempt
def call_me(request):
    """
    Twilio dials your PH number and connects the call to the /twilio/voice webhook,
    which in turn opens a Media Stream to the bot.
    """
    client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)

    try:
        call = client.calls.create(
            to=settings.MY_PHONE_NUMBER,              # e.g. +639xxxxxxx
            from_=settings.TWILIO_FROM_NUMBER,        # your Twilio US number
            url=f"{settings.PUBLIC_HTTP_ORIGIN}/twilio/voice",   # webhook for TwiML
            status_callback=f"{settings.PUBLIC_HTTP_ORIGIN}/twilio/status", # optional
            status_callback_event=["initiated","ringing","answered","completed"],
            status_callback_method="POST"
        )
        return JsonResponse({"sid": call.sid})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


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
