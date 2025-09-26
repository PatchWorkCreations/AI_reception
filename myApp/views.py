from django.conf import settings
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from twilio.rest import Client

@csrf_exempt
def voice_answer(request):
    twiml = f"""
<Response>
  <Connect>
    <Stream url="{settings.PUBLIC_WS_URL}" track="both_tracks">
      <Parameter name="context" value="neuromed-reception"/>
    </Stream>
  </Connect>
</Response>
""".strip()
    return HttpResponse(twiml, content_type="text/xml")


@csrf_exempt
def status_cb(request):
    return JsonResponse({"ok": True})


@csrf_exempt
def call_me(request):
    """Twilio dials your PH number and connects to the receptionist bot"""
    client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)

    call = client.calls.create(
        to=settings.MY_PHONE_NUMBER,
        from_=settings.TWILIO_FROM_NUMBER,
        url=f"{settings.PUBLIC_HTTP_ORIGIN}/twilio/voice"
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
