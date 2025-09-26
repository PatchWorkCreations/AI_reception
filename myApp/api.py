import json, datetime as dt
from django.http import JsonResponse
from django.views.decorators.http import require_GET, require_POST
from django.views.decorators.csrf import csrf_exempt

# Branded FAQ content (matches your script)
_FAQ = {
    "hipaa": (
        "Yes. We take privacy very seriously. NeuroMed AI is designed with HIPAA-level "
        "security—health information is protected and never shared without consent."
    ),
    "cost": (
        "We offer flexible pilot programs. Pricing depends on organization size and usage. "
        "We usually start small, then scale after the pilot."
    ),
    "pilot": (
        "Yes. We typically begin with a 3–6 month pilot in one department so staff and "
        "families can experience the benefits before a wider rollout."
    ),
    "faith": (
        "Faith-aware mode is optional. Summaries can include gentle spiritual support—like "
        "Bible verses or uplifting quotes—alongside the medical explanation."
    ),
    "what_is_neuromed": (
        "NeuroMed AI turns medical files—discharge notes, labs, and reports—into clear, "
        "compassionate summaries. Modes include plain/simple, caregiver-friendly, faith + "
        "encouragement, clinical, and bilingual. Families can ask follow-ups and get voice updates."
    ),
    "hours": "We’re available for demos by appointment.",
    "address": "We operate remotely and support providers and families wherever they are.",
    "pricing": "See: cost.",
}

@require_GET
def faq(request):
    q = (request.GET.get("q") or "").lower()
    if "hipaa" in q or "privacy" in q:                 return JsonResponse({"answer": _FAQ["hipaa"]})
    if "cost" in q or "price" in q or "pricing" in q:  return JsonResponse({"answer": _FAQ["cost"]})
    if "pilot" in q or "test" in q or "trial" in q:    return JsonResponse({"answer": _FAQ["pilot"]})
    if "faith" in q:                                   return JsonResponse({"answer": _FAQ["faith"]})
    if ("what" in q and "neuro" in q) or "what is neuromed" in q:
                                                       return JsonResponse({"answer": _FAQ["what_is_neuromed"]})
    if "hour" in q or "open" in q:                     return JsonResponse({"answer": _FAQ["hours"]})
    if "address" in q or "where" in q:                 return JsonResponse({"answer": _FAQ["address"]})
    return JsonResponse({"answer": "I can help with HIPAA, pricing, pilots, and faith-aware mode—what would you like to know?"})

@csrf_exempt
@require_POST
def book_appointment(request):
    # Voice-only: keep endpoint present but disabled for now
    data = json.loads(request.body or "{}")
    return JsonResponse({
        "status": "not_enabled",
        "note": "Voice-only mode right now; phone booking is disabled.",
        "name": data.get("name", "Caller"),
        "phone": data.get("phone", "unknown"),
        "slot": data.get("slot", (dt.datetime.now() + dt.timedelta(days=1)).isoformat()),
    })
