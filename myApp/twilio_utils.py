import os
from twilio.rest import Client

def place_outbound_test_call():
    """
    Dials your PH mobile from your Twilio number and points the call to /twilio/voice,
    which streams into the AI receptionist. Returns the Twilio Call SID.
    """
    to_number   = os.getenv("MY_PHONE_NUMBER")        # e.g. +63917XXXXXXX
    from_number = os.getenv("TWILIO_FROM_NUMBER")     # e.g. +17407213718 (your Twilio US number)
    base_url    = os.getenv("PUBLIC_HTTP_ORIGIN")     # e.g. https://k5177qxt-8000.asse.devtunnels.ms
    sid         = os.getenv("TWILIO_ACCOUNT_SID")
    token       = os.getenv("TWILIO_AUTH_TOKEN")

    if not all([to_number, from_number, base_url, sid, token]):
        raise RuntimeError("Missing one or more env vars: MY_PHONE_NUMBER, TWILIO_FROM_NUMBER, PUBLIC_HTTP_ORIGIN, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN")

    client = Client(sid, token)
    call = client.calls.create(
        to=to_number,
        from_=from_number,
        url=f"{base_url}/twilio/voice"   # when you answer, Twilio fetches your TwiML here
    )
    return call.sid
