import os, json, base64, asyncio, websockets, httpx, re, time, random
from dotenv import load_dotenv

load_dotenv()

# --------- ENV ---------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # not used in simple mode
DEEPGRAM_KEY   = os.getenv("DEEPGRAM_API_KEY")
ELEVEN_KEY     = os.getenv("ELEVENLABS_API_KEY")
HTTP_ORIGIN    = os.getenv("PUBLIC_HTTP_ORIGIN", "http://localhost:8000")
PORT           = int(os.getenv("PORT", "8080"))

ECHO_BACK      = os.getenv("ECHO_BACK", "0") == "1"
ELEVEN_VOICE   = os.getenv("ELEVENLABS_VOICE_ID", "Bella")

# --------- Short, deterministic content ---------
WELCOME_MENU = (
    "Hey there! Welcome to NeuroMed AI — your kind and clever health assistant. "
    "I can help you with our overview, privacy, pricing, pilot programs, hours, or how to get started. "
    "Which one sounds good to you today?"
)

RESPONSES = {
    "overview": (
        "NeuroMed AI takes medical files — like discharge notes, lab results, and imaging reports — "
        "and translates them into clear, human-friendly summaries. "
        "You can even choose your tone: plain and simple, caregiver-gentle, faith-filled and encouraging, or crisp and clinical. "
        "Because healthcare should speak your language."
    ),
    "privacy": (
        "Your privacy is sacred to us. "
        "All files are securely stored and never shared — ever. "
        "We’re HIPAA-conscious and heart-conscious."
    ),
    "pricing": (
        "We have flexible plans for families, caregivers, and care facilities. "
        "I can send a quick pricing overview — what’s the best email to reach you?"
    ),
    "pilot": (
        "We partner with nursing homes, clinics, and community groups for pilot programs that make care smarter and more personal. "
        "Would you like me to email you the details? If so, what’s the best address to send it to?"
    ),
    "hours": (
        "We’re usually available on weekdays for demos and support. "
        "If you share your time zone, I’ll find a time that works best for you."
    ),
    "start": (
        "Awesome! Let’s get you started. "
        "Can I grab your name and best email so we can send your next steps and a short demo link?"
    ),
    "fallback": (
        "I can tell you about NeuroMed AI’s overview, privacy, pricing, pilot programs, hours, or how to get started. "
        "Which one would you like to explore first?"
    ),
}


# --------- Minimal, readable intent matching ---------
RX = lambda p: re.compile(p, re.I)
INTENTS = [
    ("overview", RX(r"\b(over\s*view|what\s+is|tell\s+me\s+more|how\s+does\s+it\s+work)\b")),
    ("privacy",  RX(r"\b(hipaa|privacy|private|secure|security|gdpr|phi|compliance|compliant)\b")),
    ("pricing",  RX(r"\b(price|pricing|cost|how\s+much|rate|fees?|plans?|tiers?)\b")),
    ("pilot",    RX(r"\b(pilot|trial|poc|demo|evaluate|evaluation|test\s*drive)\b")),
    ("hours",    RX(r"\b(hours?|availability|available|open|close|schedule|book|appointment|appt)\b")),
    ("start",    RX(r"\b(get\s*started|start|begin|sign\s*up|setup|set\s*up)\b")),
]

FILLERS_RE = RX(r"^(?:\s*(?:yeah|yep|uh|um|hmm|hello|hi|hey|okay|ok)[,.\s]*)+")
def _strip_fillers(s: str) -> str:
    return FILLERS_RE.sub("", (s or "").strip()).strip()

def classify_intent(text: str) -> str:
    t = _strip_fillers(text or "")
    for name, rx in INTENTS:
        if rx.search(t):
            return name
    return "fallback"

# --------- helpers ---------
def b64_to_bytes(s: str) -> bytes:
    return base64.b64decode(s)

def bytes_to_b64(b: bytes) -> str:
    return base64.b64encode(b).decode()

def _ulaw_silence(ms: int) -> bytes:
    return b"\xff" * int(8 * ms)  # 8kHz μ-law, 8 samples/ms

async def send_silence(ws_send_pcm, ms: int):
    if ms <= 0:
        return
    try:
        await ws_send_pcm(_ulaw_silence(ms))
    except (websockets.exceptions.ConnectionClosedOK, websockets.exceptions.ConnectionClosedError):
        pass

# --------- ElevenLabs TTS (ulaw_8000) with simple cache ---------
from functools import lru_cache
@lru_cache(maxsize=256)
def _tts_key(text: str) -> str:
    return (text or "").strip().lower()

async def eleven_tts_stream(text: str):
    if not ELEVEN_KEY:
        raise RuntimeError("ELEVENLABS_API_KEY missing.")

    url = (
        f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE}/stream"
        "?optimize_streaming_latency=3&output_format=ulaw_8000"
    )
    headers = { "xi-api-key": ELEVEN_KEY, "Content-Type": "application/json", "Accept": "*/*" }
    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
    }

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as r:
            if r.is_error:
                body = await r.aread()
                print("ELEVENLABS HTTP ERROR ▶", r.status_code, body[:200])
                return
            async for chunk in r.aiter_bytes():
                if chunk:
                    yield chunk

async def eleven_tts_stream_cached(text: str):
    key = _tts_key(text)
    if hasattr(eleven_tts_stream_cached, "_mem") and key in eleven_tts_stream_cached._mem:
        for c in eleven_tts_stream_cached._mem[key]:
            yield c
        return

    chunks = []
    async for c in eleven_tts_stream(text):
        chunks.append(c)
        yield c

    if len(b"".join(chunks)) < 2_000_000:
        eleven_tts_stream_cached._mem = getattr(eleven_tts_stream_cached, "_mem", {})
        eleven_tts_stream_cached._mem[key] = chunks

# --------- Deepgram ASR ---------
async def deepgram_stream(pcm_iter):
    if not DEEPGRAM_KEY:
        print("ASR WARN ▶ DEEPGRAM_API_KEY missing; ASR disabled.")
        return

    url = (
        "wss://api.deepgram.com/v1/listen"
        "?model=nova-2"
        "&encoding=mulaw"
        "&sample_rate=8000"
        "&language=en-US"
        "&punctuate=true"
        "&interim_results=true"
        "&vad_events=true"
        "&endpointing=true"
        "&vad_turnoff=300"
    )
    headers = [("Authorization", f"Token {DEEPGRAM_KEY}")]

    async with websockets.connect(url, extra_headers=headers, max_size=2**20) as dg:
        # Light keyword biasing
        try:
            await dg.send(json.dumps({
                "type": "Configure",
                "keywords": [
                    "NeuroMed", "HIPAA", "privacy", "pricing", "pilot", "demo",
                    "hours", "appointment", "schedule", "caregiver", "faith"
                ]
            }))
        except Exception as e:
            print("DG CONFIG WARN ▶", repr(e))

        async def feeder():
            try:
                async for chunk in pcm_iter():
                    if chunk is None:
                        break
                    try:
                        await dg.send(chunk)
                    except (websockets.exceptions.ConnectionClosedOK,
                            websockets.exceptions.ConnectionClosedError):
                        break
            finally:
                try:
                    await dg.send(json.dumps({"type": "CloseStream"}))
                except Exception:
                    pass

        feed_task = asyncio.create_task(feeder())

        last_sent = ""
        last_emit = 0.0
        try:
            async for msg in dg:
                try:
                    obj = json.loads(msg)
                except Exception:
                    continue

                if isinstance(obj, dict) and obj.get("type") in {
                    "Metadata","Warning","Error","Close","UtteranceEnd","SpeechStarted","SpeechFinished"
                }:
                    continue

                alts, is_final = [], bool(obj.get("is_final"))
                chan = obj.get("channel")
                if isinstance(chan, dict):
                    alts = chan.get("alternatives") or []
                elif isinstance(chan, list) and chan:
                    first_chan = chan[0] or {}
                    if isinstance(first_chan, dict):
                        alts = first_chan.get("alternatives") or []
                elif isinstance(obj, dict) and "alternatives" in obj:
                    alts = obj.get("alternatives") or []

                if not alts: continue
                txt = (alts[0].get("transcript") or "").strip()
                if not txt: continue

                now = time.time()
                if not (is_final or (now - last_emit) > 0.2):  # throttle partials
                    continue
                if not is_final and txt == last_sent:
                    continue

                print(f"ASR{'(final)' if is_final else ''} ▶ {txt}")
                yield txt, is_final
                last_emit = now
                last_sent = txt
        finally:
            await feed_task

# --------- Twilio WS Server (deterministic loop) ---------
async def handle_twilio(ws):
    if ws.path not in ("/ws/twilio",):
        await ws.close(code=1008, reason="Unexpected path")
        return

    print("WS ▶ Twilio connected.")
    inbound_q: asyncio.Queue[bytes] = asyncio.Queue()
    conn_open = True
    stream_sid = None
    speak_task: asyncio.Task | None = None
    first_media = asyncio.Event()

    async def pcm_iter():
        while True:
            b = await inbound_q.get()
            if b is None:
                break
            yield b

    async def send_pcm(pcm: bytes) -> bool:
        nonlocal conn_open, stream_sid
        if not conn_open or not stream_sid:
            return False
        try:
            await ws.send(json.dumps({
                "event": "media",
                "streamSid": stream_sid,
                "media": {"payload": bytes_to_b64(pcm)}
            }))
            return True
        except (websockets.exceptions.ConnectionClosedOK, websockets.exceptions.ConnectionClosedError):
            conn_open = False
            return False

    async def speak(text: str):
        nonlocal speak_task
        text = (text or "").strip()
        if not text: return

        if speak_task and not speak_task.done():
            speak_task.cancel()
            try: await speak_task
            except asyncio.CancelledError: pass

        async def _run():
            # tiny pre-breath for natural pacing
            pre = 50 + int(random.uniform(-10, 15))
            await send_silence(send_pcm, max(pre, 20))
            async for pcm in eleven_tts_stream_cached(text):
                ok = await send_pcm(pcm)
                if not ok: return
            post = 50 + int(random.uniform(-10, 15))
            await send_silence(send_pcm, max(post, 20))

        speak_task = asyncio.create_task(_run())

    async def brain():
        if not DEEPGRAM_KEY:
            return
        try:
            await asyncio.wait_for(first_media.wait(), timeout=20)
        except asyncio.TimeoutError:
            print("ASR ▶ no media within 20s"); return

        pending_final = None
        pending_ts = 0.0
        FINAL_DEBOUNCE = 0.3

        async def handle_user(text: str):
            # core: classify → fixed response
            intent = classify_intent(text)
            print("INTENT ▶", intent)
            reply = RESPONSES.get(intent, RESPONSES["fallback"])
            await speak(reply)

        async for utter, is_final in deepgram_stream(pcm_iter):
            if not utter: continue
            now = time.time()

            if is_final:
                if pending_final and (now - pending_ts) <= FINAL_DEBOUNCE and not re.search(r"[.!?…]\s*$", pending_final):
                    pending_final = (pending_final + " " + utter).strip()
                    pending_ts = now
                else:
                    pending_final = utter.strip()
                    pending_ts = now

                if re.search(r"[.!?…]\s*$", pending_final) or len(pending_final.split()) >= 6:
                    await handle_user(pending_final)
                    pending_final = None

        if pending_final:
            await handle_user(pending_final)

    brain_task = asyncio.create_task(brain())

    try:
        async for raw in ws:
            data = json.loads(raw)
            ev = data.get("event")

            if ev == "start":
                start_info = data.get("start", {}) or {}
                stream_sid = start_info.get("streamSid")
                print(f"WS ▶ start streamSid={stream_sid}")
                # greet immediately with the short menu
                asyncio.create_task(speak(WELCOME_MENU))

            elif ev == "media":
                payload_b64 = data["media"]["payload"]
                buf = b64_to_bytes(payload_b64)
                if not first_media.is_set():
                    first_media.set()
                    print("WS ▶ first media frame")
                await inbound_q.put(buf)
                if ECHO_BACK and stream_sid and conn_open:
                    await send_pcm(buf)

            elif ev == "stop":
                print("WS ▶ stop")
                conn_open = False
                if speak_task and not speak_task.done():
                    speak_task.cancel()
                    try: await speak_task
                    except asyncio.CancelledError: pass
                break

    except Exception as e:
        print("WS ERR ▶", e)
    finally:
        conn_open = False
        await inbound_q.put(None)
        if speak_task and not speak_task.done():
            speak_task.cancel()
            try: await speak_task
            except asyncio.CancelledError: pass
        await brain_task
        print("WS ▶ closed")

async def main():
    print("ENV ▶ HTTP_ORIGIN =", os.getenv("PUBLIC_HTTP_ORIGIN"))
    print("ENV ▶ ELEVEN key? ", "yes" if ELEVEN_KEY else "missing")
    print("ENV ▶ DEEPGRAM?   ", "yes" if DEEPGRAM_KEY else "missing")
    print("ENV ▶ ECHO_BACK   =", ECHO_BACK)
    print("ENV ▶ ELEVEN_VOICE=", ELEVEN_VOICE)
    print("ENV ▶ PORT        =", PORT)

    async with websockets.serve(
        handle_twilio, "0.0.0.0", PORT,
        max_size=2**20, ping_interval=20, ping_timeout=60,
        subprotocols=["audio.twilio.com"],
    ):
        print(f"WS bot listening on ws://0.0.0.0:{PORT}")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
