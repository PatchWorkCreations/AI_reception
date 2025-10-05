import os, json, base64, asyncio, websockets, httpx, re, time, random
from dotenv import load_dotenv
from functools import lru_cache
from difflib import SequenceMatcher

load_dotenv()

# --------- ENV ---------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # not used in simple mode
DEEPGRAM_KEY   = os.getenv("DEEPGRAM_API_KEY")
ELEVEN_KEY     = os.getenv("ELEVENLABS_API_KEY")
HTTP_ORIGIN    = os.getenv("PUBLIC_HTTP_ORIGIN", "http://localhost:8000")
PORT           = int(os.getenv("PORT", "8080"))

ECHO_BACK      = os.getenv("ECHO_BACK", "0") == "1"
ELEVEN_VOICE   = os.getenv("ELEVENLABS_VOICE_ID", "Bella")

# --------- Menu content ---------
def menu_text():
    return (
        "I can help with: overview, privacy, pricing, pilot programs, hours, or how to get started. "
        "Which one would you like? You can also say 'none' to finish."
    )

WELCOME = "Hey there! Welcome to NeuroMed AI — your kind and clever health assistant. "
WELCOME_MENU = WELCOME + menu_text()
GOODBYE = "Thank you! Don’t forget to visit neuromedai.org."

RESPONSES = {
    "overview": (
        "NeuroMed AI turns medical files like discharge notes, lab results, and imaging reports "
        "into clear, human-friendly summaries. You can pick the tone: plain and simple, caregiver-gentle, "
        "faith-filled and encouraging, or crisp and clinical."
    ),
    "privacy": (
        "Your privacy is sacred to us. All files are securely stored and never shared — ever. "
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
        "Awesome! Just head over to neuromedai.org and sign up as one of our beta testers — see you there!"
    ),
    "fallback": menu_text(),
}

# --------- Regex helpers ---------
RX = lambda p: re.compile(p, re.I)
HELLO_RE   = RX(r"\b(hello|hi|hey|good\s*(morning|afternoon|evening)|yo)\b")
THANKS_RE  = RX(r"\b(thanks?|thank\s*you|that'?s\s*all|nothing\s*else|i'?m\s*good|we'?re\s*good)\b")
INTENT_NONE = RX(r"\b(none|finish|end|no(?:\s+thanks|\s+thank\s*you)?)\b")

# Primary patterns
INTENTS = [
    ("overview", RX(
        r"\b(over\s*view|overview|what\s+is|tell\s+me\s+more|how\s+does\s+it\s+work|"
        r"highlights?|highlight\s+program|key\s+features?|capabilit(?:y|ies)|"
        r"what\s+can\s+you\s+do)\b"
    )),
    ("privacy",  RX(r"\b(hipaa|privacy|private|secure|security|gdpr|phi|compliance|compliant)\b")),
    ("pricing",  RX(r"\b(price|pricing|cost|how\s*much|rate|fees?|plans?|tiers?)\b")),
    ("pilot",    RX(r"\b(pilot(\s+program)?|trial|poc|demo|evaluate|evaluation|test\s*drive)\b")),
    ("hours",    RX(r"\b(hours?|availability|available|open|close|schedule|book|appointment|appt)\b")),
    ("start",    RX(r"\b(get\s*started|start|begin|sign\s*up|setup|set\s*up|how\s*to\s*get\s*started)\b")),
]

FILLERS_RE = RX(r"^(?:\s*(?:yeah|yep|uh|um|hmm|hello|hi|hey|okay|ok)[,.\s]*)+")
def _strip_fillers(s: str) -> str:
    return FILLERS_RE.sub("", (s or "").strip()).strip()

# --------- Soft/fuzzy intent layer (handles mishears) ---------
CHOICE_SYNONYMS = {
    "overview": {"overview","over view","what is","how it works","highlights","highlight","key features","capabilities","what can you do"},
    "privacy": {"privacy","private","security","secure","hipaa","gdpr","compliance","compliant","phi"},
    "pricing": {"pricing","price","prices","cost","how much","rates","fees","plans","tiers"},
    "pilot":   {"pilot","pilot program","trial","poc","demo","evaluation","test drive","program"},
    "hours":   {"hours","availability","available","open","close","schedule","appointment","appt","book"},
    "start":   {"get started","start","begin","sign up","setup","set up","how to get started"},
}

SOUNDA_LIKE = {
    # pilot-ish
    "violet": "pilot",
    "violent": "pilot",
    "silo": "pilot",
    "pylet": "pilot",
    "pilate": "pilot",
    "pilot": "pilot",
    "kylas": "pilot",
    "kyla": "pilot",
    "kaila": "pilot",
    "kyle": "pilot",
    "pylot": "pilot",
    "bogdan": "pilot",   # surgical mapping based on your logs
    # program
    "program": "program",
    "doggroom": "program",
    "doggroomer": "program",
    "dawgroom": "program",
    # pricing
    "prizing": "pricing",
    "prise": "price",
    "prices": "pricing",
    # hours
    "ours": "hours",
    "hour": "hours",
    # privacy
    "privacy": "privacy",
    "private": "privacy",
    "secure": "privacy",
}

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", (s or "").lower()).strip()

def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def _soft_map_token(tok: str) -> str:
    t = _norm(tok)
    return SOUNDA_LIKE.get(t, t)

def _soft_match_intent(text: str) -> str | None:
    t = _norm(text)
    if not t:
        return None
    if INTENT_NONE.search(t) or THANKS_RE.search(t):
        return "none"
    if HELLO_RE.search(t):
        return "fallback"

    tokens = [_soft_map_token(w) for w in t.split()]
    joined = " ".join(tokens)

    # short/lone "program" → pilot
    if "program" in tokens and len(tokens) <= 3:
        return "pilot"
    # "program" near pilot-ish
    if "program" in tokens:
        for w in tokens:
            if _similar(w, "pilot") >= 0.60:
                return "pilot"

    best_intent, best_score = None, 0.0
    for intent, syns in CHOICE_SYNONYMS.items():
        for syn in syns:
            score = max(_similar(joined, syn), max((_similar(tok, syn) for tok in tokens), default=0.0))
            if score > best_score:
                best_intent, best_score = intent, score
    if best_score >= 0.75:
        return best_intent

    for intent, syns in CHOICE_SYNONYMS.items():
        if any(tok in syns for tok in tokens):
            return intent
    if "program" in tokens:
        return "pilot"
    return None

def classify_intent_or_none(text: str) -> str:
    t = _strip_fillers(text or "")
    if not t:
        return "fallback"
    soft = _soft_match_intent(t)
    if soft:
        return soft
    for name, rx in INTENTS:
        if rx.search(t):
            return name
    bare = _norm(t)
    if bare in {"overview","privacy","pricing","pilot","hours","start"}:
        return bare
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
    payload = {"text": text, "model_id": "eleven_multilingual_v2",
               "voice_settings": {"stability": 0.5, "similarity_boost": 0.8}}
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as r:
            if r.is_error:
                body = await r.aread()
                print("ELEVENLABS HTTP ERROR ▶", r.status_code, body[:200]); return
            async for chunk in r.aiter_bytes():
                if chunk: yield chunk

async def eleven_tts_stream_cached(text: str):
    key = _tts_key(text)
    if hasattr(eleven_tts_stream_cached, "_mem") and key in eleven_tts_stream_cached._mem:
        for c in eleven_tts_stream_cached._mem[key]:
            yield c
        return
    chunks = []
    async for c in eleven_tts_stream(text):
        chunks.append(c); yield c
    if len(b"".join(chunks)) < 2_000_000:
        eleven_tts_stream_cached._mem = getattr(eleven_tts_stream_cached, "_mem", {})
        eleven_tts_stream_cached._mem[key] = chunks

# --------- Deepgram ASR ---------
async def deepgram_stream(pcm_iter):
    """
    Yields:
      - ("__EVENT__", "SpeechStarted") on speech start
      - ("__EVENT__", "SpeechFinished") on speech end
      - (text, is_final) for hypotheses
    """
    if not DEEPGRAM_KEY:
        print("ASR WARN ▶ DEEPGRAM_API_KEY missing; ASR disabled.")
        return
    url = (
        "wss://api.deepgram.com/v1/listen"
        "?model=nova-2&encoding=mulaw&sample_rate=8000&language=en-US"
        "&punctuate=true&interim_results=true&vad_events=true&endpointing=true&vad_turnoff=300"
    )
    headers = [("Authorization", f"Token {DEEPGRAM_KEY}")]

    async with websockets.connect(url, extra_headers=headers, max_size=2**20) as dg:
        try:
            await dg.send(json.dumps({
                "type": "Configure",
                "keywords": [
                    "NeuroMed","overview","privacy","pricing","pilot","pilot program","hours","start",
                    "trial","demo","evaluation","schedule","appointment",
                    "violet","violent","silo","pilate","pylot","kyla","kylas","kaila","kyle",
                    "program","highlights","features","email","address"
                ]
            }))
        except Exception as e:
            print("DG CONFIG WARN ▶", repr(e))

        async def feeder():
            try:
                async for chunk in pcm_iter():
                    if chunk is None: break
                    try: await dg.send(chunk)
                    except (websockets.exceptions.ConnectionClosedOK,
                            websockets.exceptions.ConnectionClosedError):
                        break
            finally:
                try: await dg.send(json.dumps({"type": "CloseStream"}))
                except Exception: pass

        feed_task = asyncio.create_task(feeder())

        last_sent = ""; last_emit = 0.0
        try:
            async for msg in dg:
                try: obj = json.loads(msg)
                except Exception: continue

                t = obj.get("type")
                if t in {"SpeechStarted", "SpeechFinished"}:
                    yield ("__EVENT__", t); continue
                if isinstance(obj, dict) and t in {"Metadata","Warning","Error","Close","UtteranceEnd"}:
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
                if not (is_final or (now - last_emit) > 0.2): continue
                if not is_final and txt == last_sent: continue

                print(f"ASR{'(final)' if is_final else ''} ▶ {txt}")
                yield (txt, is_final)
                last_emit = now; last_sent = txt
        finally:
            await feed_task

# --------- Twilio WS Server with email capture state ---------
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

async def handle_twilio(ws):
    if ws.path not in ("/ws/twilio",):
        await ws.close(code=1008, reason="Unexpected path"); return

    print("WS ▶ Twilio connected.")
    inbound_q: asyncio.Queue[bytes] = asyncio.Queue()
    conn_open = True
    stream_sid = None
    speak_task: asyncio.Task | None = None
    first_media = asyncio.Event()

    done_flag = False
    stopped_flag = False
    menu_task: asyncio.Task | None = None
    followup_task: asyncio.Task | None = None
    email_timeout_task: asyncio.Task | None = None

    current_tts_label = ""  # "welcome_menu" | "answer:*" | "menu" | "goodbye" | "fallback" | ""

    # simple state
    awaiting_email = False
    email_context = None  # "pricing" or "pilot"

    async def pcm_iter():
        while True:
            b = await inbound_q.get()
            if b is None: break
            yield b

    async def send_pcm(pcm: bytes) -> bool:
        nonlocal conn_open, stream_sid
        if not conn_open or not stream_sid: return False
        try:
            await ws.send(json.dumps({
                "event": "media",
                "streamSid": stream_sid,
                "media": {"payload": bytes_to_b64(pcm)}
            }))
            return True
        except (websockets.exceptions.ConnectionClosedOK, websockets.exceptions.ConnectionClosedError):
            conn_open = False; return False

    async def speak(text: str, label: str = "tts") -> asyncio.Task:
        nonlocal speak_task, current_tts_label
        text = (text or "").strip()
        if not text or stopped_flag:
            return asyncio.create_task(asyncio.sleep(0))  # dummy task

        if speak_task and not speak_task.done():
            speak_task.cancel()
            try: await speak_task
            except asyncio.CancelledError: pass

        current_tts_label = label
        print(f"SAY ▶ {label}")

        async def _run():
            nonlocal current_tts_label
            try:
                pre = 50 + int(random.uniform(-10, 15))
                await send_silence(send_pcm, max(pre, 20))
                async for pcm in eleven_tts_stream_cached(text):
                    ok = await send_pcm(pcm)
                    if not ok: return
                post = 50 + int(random.uniform(-10, 15))
                await send_silence(send_pcm, max(post, 20))
            finally:
                if current_tts_label == label:
                    current_tts_label = ""

        speak_task = asyncio.create_task(_run())
        return speak_task

    def cancel_task(t: asyncio.Task | None):
        if t and not t.done():
            t.cancel()

    def schedule_menu_after_answer(answer_task: asyncio.Task, delay_ms: int = 1200):
        nonlocal followup_task
        cancel_task(followup_task)
        async def _watch():
            try:
                try:
                    await answer_task
                except asyncio.CancelledError:
                    return
                await asyncio.sleep(delay_ms / 1000.0)
                if not (stopped_flag or done_flag) and current_tts_label == "" and not awaiting_email:
                    await speak(menu_text(), label="menu")
            except asyncio.CancelledError:
                return
        followup_task = asyncio.create_task(_watch())

    def start_email_timeout(seconds: float = 12.0):
        nonlocal email_timeout_task
        cancel_task(email_timeout_task)
        async def _wait():
            try:
                await asyncio.sleep(seconds)
                if not (stopped_flag or done_flag) and awaiting_email and current_tts_label == "":
                    await speak("What’s the best email to reach you?", label="email_reprompt")
            except asyncio.CancelledError:
                return
        email_timeout_task = asyncio.create_task(_wait())

    async def confirm_and_menu(captured_email: str):
        nonlocal awaiting_email, email_context
        awaiting_email = False
        email_context = None
        cancel_task(email_timeout_task)
        await speak(f"Got it. We’ll email you at {captured_email}.", label="email_confirm")
        # after confirm, follow normal menu loop
        await asyncio.sleep(1.0)
        if not (stopped_flag or done_flag):
            await speak(menu_text(), label="menu")

    async def handle_user(text: str):
        nonlocal done_flag, awaiting_email, email_context
        if done_flag or stopped_flag: return

        # If we're waiting for an email, try to capture it first.
        if awaiting_email:
            m = EMAIL_RE.search(text)
            if m:
                await confirm_and_menu(m.group(0))
                return
            # Also allow the user to bail out
            if classify_intent_or_none(text) == "none":
                done_flag = True
                cancel_task(email_timeout_task)
                await speak(GOODBYE, label="goodbye")
                return
            # Otherwise ignore non-emails here (no menu); the timeout will reprompt.
            print("EMAIL ▶ not found in utterance")
            return

        intent = classify_intent_or_none(text)
        print("INTENT ▶", intent)

        if intent == "none":
            done_flag = True
            cancel_task(followup_task)
            await speak(GOODBYE, label="goodbye")
            return

        if intent in RESPONSES and intent != "fallback":
            cancel_task(followup_task)
            # Speak the answer
            ans_task = await speak(RESPONSES[intent], label=f"answer:{intent}")

            # If this path requests an email, go into capture mode; do NOT schedule menu yet
            if intent in ("pricing", "pilot"):
                awaiting_email = True
                email_context = intent
                # when the answer finishes, start the email timeout
                async def _after_answer():
                    try:
                        await ans_task
                        start_email_timeout(12.0)
                    except asyncio.CancelledError:
                        pass
                asyncio.create_task(_after_answer())
            else:
                # normal intents → menu after the answer
                schedule_menu_after_answer(ans_task, 1200)
            return

        # fallback: offer choices (acts like menu)
        await speak(RESPONSES["fallback"], label="fallback")

    async def brain():
        if not DEEPGRAM_KEY: return
        try:
            await asyncio.wait_for(first_media.wait(), timeout=20)
        except asyncio.TimeoutError:
            print("ASR ▶ no media within 20s"); return

        pending_final = None
        pending_ts = 0.0
        FINAL_DEBOUNCE = 0.3

        async for item in deepgram_stream(pcm_iter):
            if done_flag or stopped_flag: break

            if isinstance(item, tuple) and len(item) == 2 and item[0] == "__EVENT__":
                ev = item[1]
                if ev == "SpeechStarted":
                    cancel_task(menu_task)
                    cancel_task(followup_task)
                    # don't interrupt the initial welcome+menu
                    if current_tts_label not in ("welcome_menu",):
                        if speak_task and not speak_task.done():
                            speak_task.cancel()
                            try: await speak_task
                            except asyncio.CancelledError: pass
                continue

            utter, is_final = item
            if not utter: continue
            now = time.time()

            if is_final:
                if pending_final and (now - pending_ts) <= FINAL_DEBOUNCE and not re.search(r"[.!?…]\s*$", pending_final):
                    pending_final = (pending_final + " " + utter).strip()
                    pending_ts = now
                else:
                    pending_final = utter.strip()
                    pending_ts = now

                if re.search(r"[.!?…]\s*$", pending_final) or len(pending_final.split()) >= 2:
                    await handle_user(pending_final)
                    pending_final = None

        if not (done_flag or stopped_flag) and pending_final:
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

                # Say full welcome + menu immediately so callers know the options
                await speak(WELCOME_MENU, label="welcome_menu")

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
                stopped_flag = True
                conn_open = False
                for t in (menu_task, followup_task, email_timeout_task, speak_task, brain_task):
                    cancel_task(t)
                break

    except Exception as e:
        print("WS ERR ▶", e)
    finally:
        conn_open = False
        await inbound_q.put(None)
        for t in (menu_task, followup_task, email_timeout_task, speak_task):
            cancel_task(t)
        try:
            await asyncio.wait_for(brain_task, timeout=2)
        except Exception:
            pass
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
