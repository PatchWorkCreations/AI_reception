import os, json, base64, asyncio, websockets, httpx, re, time, random, struct
from dotenv import load_dotenv
from functools import lru_cache

load_dotenv()

# --------- ENV ---------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # optional for micro NLU (email decoding)
DEEPGRAM_KEY   = os.getenv("DEEPGRAM_API_KEY")
ELEVEN_KEY     = os.getenv("ELEVENLABS_API_KEY")
HTTP_ORIGIN    = os.getenv("PUBLIC_HTTP_ORIGIN", "https://neuromed-django-production.up.railway.app")
PORT           = int(os.getenv("PORT", "8080"))

ECHO_BACK      = os.getenv("ECHO_BACK", "0") == "1"
ELEVEN_VOICE   = os.getenv("ELEVENLABS_VOICE_ID", "Bella")

OPENAI_MODEL         = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TIMEOUT_S     = float(os.getenv("OPENAI_TIMEOUT_S", "0.6"))
OPENAI_ENABLE        = bool(OPENAI_API_KEY)

# --------- Preroll audio (your WAV μ-law 8k) ---------
PREROLL_WAV_URL = f"{HTTP_ORIGIN.rstrip('/')}/twilio/preroll-audio/"
PREROLL_CHUNK_SIZE = 160  # 20ms at 8kHz μ-law (1 byte per sample)

# --------- Shared HTTP client ---------
HTTP2_OK = False
try:
    import h2  # noqa
    HTTP2_OK = True
except Exception:
    HTTP2_OK = False

SHARED_CLIENT = httpx.AsyncClient(
    timeout=httpx.Timeout(60.0),
    http2=HTTP2_OK,
    limits=httpx.Limits(max_connections=20, max_keepalive_connections=10, keepalive_expiry=30),
    headers={"User-Agent": "NeuroMedAI-VoiceBot/1.0"}
)

async def _post_json(url: str, headers: dict, payload: dict, timeout_s: float | None = None):
    timeout = httpx.Timeout(timeout_s or 30.0)
    for attempt in range(5):
        try:
            r = await SHARED_CLIENT.post(url, headers=headers, json=payload, timeout=timeout)
            if r.status_code in (408, 409, 429, 500, 502, 503, 504):
                await asyncio.sleep(min(2 ** attempt + random.random(), 10))
                continue
            r.raise_for_status()
            return r
        except (httpx.HTTPStatusError, httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError) as e:
            if attempt >= 4:
                print("HTTP POST FAIL ▶", url, repr(e))
                raise
            await asyncio.sleep(min(2 ** attempt + random.random(), 8))

# --------- Script content ---------
DOMAIN_PLAIN  = "NeuroMedAI.org"
DOMAIN_SPOKEN = "Neuro Med A I dot org"

ASK_EMAIL = "What’s the best email to reach you?"
CONFIRM_EMAIL_READBACK = "I heard {email_spelled}. Is that correct? Please say yes or no."
CONFIRM_EMAIL_OK = "Great, we’ll email you at {email}."
CONFIRM_EMAIL_NO = "No problem. Let’s try again. What’s the best email to reach you?"
GOODBYE = (
    f"Thank you! You can upload a medical file anytime at {DOMAIN_SPOKEN}. "
    "Choose your preferred tone and get your summary within minutes."
)

# --------- Helpers (audio + utils) ---------
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

def spell_for_tts(email: str) -> str:
    mapping = {".": " dot ", "@": " at ", "+": " plus ", "_": " underscore ", "-": " dash "}
    out = []
    for ch in email:
        if ch.lower() in "abcdefghijklmnopqrstuvwxyz0123456789":
            out.append(ch)
        else:
            out.append(mapping.get(ch, f" {ch} "))
    s = "".join(out)
    return re.sub(r"\s+", " ", s).strip()

# ---- Parse WAV and return μ-law payload (data chunk only) ----
def extract_wav_ulaw_payload(wav_bytes: bytes) -> bytes:
    # Very small WAV reader for PCM μ-law: look for RIFF/WAVE and 'data' chunk.
    if len(wav_bytes) < 44 or wav_bytes[0:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
        raise ValueError("Not a valid WAV file.")
    i = 12
    data_payload = None
    while i + 8 <= len(wav_bytes):
        chunk_id = wav_bytes[i:i+4]
        chunk_sz = struct.unpack("<I", wav_bytes[i+4:i+8])[0]
        i += 8
        if chunk_id == b"data":
            if i + chunk_sz > len(wav_bytes):
                chunk_sz = len(wav_bytes) - i
            data_payload = wav_bytes[i:i+chunk_sz]
            break
        i += chunk_sz
    if not data_payload:
        raise ValueError("WAV has no data chunk.")
    return data_payload

async def play_preroll_wav(ws_send_pcm, url: str = PREROLL_WAV_URL):
    try:
        r = await SHARED_CLIENT.get(url, timeout=httpx.Timeout(30.0))
        r.raise_for_status()
        ulaw_payload = extract_wav_ulaw_payload(r.content)
        # stream in ~20ms frames
        for j in range(0, len(ulaw_payload), PREROLL_CHUNK_SIZE):
            piece = ulaw_payload[j:j+PREROLL_CHUNK_SIZE]
            ok = await ws_send_pcm(piece)
            if not ok:
                return
            await asyncio.sleep(0)  # yield
    except Exception as e:
        print("PREROLL ERR ▶", repr(e))

# --------- ElevenLabs TTS (ulaw_8000) with cache ---------
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
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.8}
    }
    for attempt in range(4):
        try:
            async with SHARED_CLIENT.stream("POST", url, headers=headers, json=payload,
                                            timeout=httpx.Timeout(60.0)) as r:
                if r.is_error:
                    body = await r.aread()
                    print("ELEVENLABS HTTP ERROR ▶", r.status_code, body[:200])
                    if attempt >= 3:
                        return
                    await asyncio.sleep(min(2 ** attempt + random.random(), 6))
                    continue
                async for chunk in r.aiter_bytes():
                    if chunk:
                        yield chunk
                return
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError) as e:
            print("ELEVENLABS EXC ▶", repr(e))
            if attempt >= 3:
                return
            await asyncio.sleep(min(2 ** attempt + random.random(), 6))

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
                    "email","at","dot","underscore","dash","hyphen","plus",
                    "NeuroMed","pilot","proposal","onboarding"
                ]
            }))
        except Exception as e:
            print("DG CONFIG WARN ▶", repr(e))

        async def feeder():
            try:
                async for chunk in pcm_iter():
                    if chunk is None: break
                    try:
                        await dg.send(chunk)
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
                yield (txt, is_final)
                last_emit = now; last_sent = txt
        finally:
            await feed_task

# --------- Email parsing ---------
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

_NUMBER_WORDS = {
    "zero":"0","oh":"0","one":"1","two":"2","three":"3","four":"4","for":"4","five":"5",
    "six":"6","seven":"7","eight":"8","nine":"9","ten":"10","eleven":"11","twelve":"12",
    "thirteen":"13","fourteen":"14","fifteen":"15","sixteen":"16","seventeen":"17",
    "eighteen":"18","nineteen":"19","twenty":"20","thirty":"30","forty":"40","fifty":"50",
    "sixty":"60","seventy":"70","eighty":"80","ninety":"90",
}
_SP_OK = {"at":"@", "dot":".", "period":".", "underscore":"_", "dash":"-", "hyphen":"-", "plus":"+"}
_EMAIL_FILLER = {
    "my","email","mail","address","is","it's","its","this","the","to","send","reach","me","at:",
    "and","please","you","can","use","on","for"
}
_PROVIDER_FIXES = [
    (r"\bg\s*mail\b", "gmail"),
    (r"\bgee\s*mail\b", "gmail"),
    (r"\bg\s*male\b", "gmail"),
    (r"\bgee\s*male\b", "gmail"),
    (r"\bout\s*look\b", "outlook"),
    (r"\bproton\s*mail\b", "protonmail"),
    (r"\bhot\s*mail\b", "hotmail"),
    (r"\bicloud\b", "icloud"),
    (r"\byahoo\b", "yahoo"),
]

def _word_numbers_to_digits(tokens):
    out = []
    for w in tokens:
        lw = w.lower()
        if lw in _NUMBER_WORDS: out.append(_NUMBER_WORDS[lw])
        else: out.append(w)
    return out

def _clean_tokens_for_email(toks):
    return [t for t in toks if t not in _EMAIL_FILLER]

def normalize_spoken_email(text: str) -> str | None:
    if not text: return None
    t = re.sub(r"[^\w\s@.+-]", " ", text.lower())
    for pat, rep in _PROVIDER_FIXES:
        t = re.sub(pat, rep, t)
    toks = [w for w in t.split() if w]
    toks = _word_numbers_to_digits(toks)
    toks = [_SP_OK.get(w, w) for w in toks]
    toks = _clean_tokens_for_email(toks)
    s = " ".join(toks)
    s = re.sub(r"\s*@\s*", "@", s)
    s = re.sub(r"\s*\.\s*", ".", s)
    s = s.replace("-dot-", ".").replace("-@", "@").replace(".@", "@").replace("@.", "@")
    s = re.sub(r"(@|\.)\s+", r"\1", s)
    if "@" in s:
        local, _, domain = s.partition("@")
        local = re.sub(r"\s+", "", local)
        domain = re.sub(r"\s+", "", domain)
        s = f"{local}@{domain}"
    s = re.sub(r"\.come\b", ".com", s)
    s = re.sub(r"\.calm\b", ".com", s)
    s = re.sub(r"\.orgg\b", ".org", s)
    s = s.strip()
    if EMAIL_RE.fullmatch(s or ""): return s
    m = EMAIL_RE.search(s)
    if m: return m.group(0)
    if "@" in s:
        m2 = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", s)
        if m2: return m2.group(0)
    return None

# ===== OpenAI micro NLU (email only; optional) =====
_openai_cache: dict[str, dict] = {}
OPENAI_CACHE_MAX = 200
def _cache_get(k: str): return _openai_cache.get(k)
def _cache_set(k: str, v: dict):
    if len(_openai_cache) > OPENAI_CACHE_MAX: _openai_cache.clear()
    _openai_cache[k] = v
def _intent_system_prompt():
    return (
        "Return strict JSON with key email. "
        "email is a string if a valid email is spoken (e.g., 'julia sixteen at gmail dot com' → julia16@gmail.com), else null. "
        "Do not include extra keys. Consider homophones and ASR noise."
    )
async def call_openai_email(text: str) -> str | None:
    if not OPENAI_ENABLE or not text.strip():
        return None
    key = f"email::{text.strip().lower()}"
    cached = _cache_get(key)
    if cached: return cached.get("email")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": OPENAI_MODEL,
        "temperature": 0.0,
        "max_tokens": 30,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": _intent_system_prompt()},
            {"role": "user", "content": text.strip()},
        ],
    }
    try:
        r = await _post_json(url, headers, payload, timeout_s=OPENAI_TIMEOUT_S or 1.0)
        data = r.json()
        content = (data.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()
        obj = json.loads(content or "{}")
        email = obj.get("email")
        if isinstance(email, str) and EMAIL_RE.fullmatch(email):
            _cache_set(key, {"email": email})
            return email
        return None
    except Exception as e:
        print("OPENAI EXC ▶", repr(e))
        return None

# --------- States ---------
STATE_IDLE            = "IDLE"
STATE_AWAITING_EMAIL  = "AWAITING_EMAIL"
STATE_CONFIRM_EMAIL   = "CONFIRM_EMAIL"
STATE_ENDED           = "ENDED"

async def handle_twilio(ws):
    if ws.path not in ("/ws/twilio",):
        await ws.close(code=1008, reason="Unexpected path"); return

    print("WS ▶ Twilio connected.")
    inbound_q: asyncio.Queue[bytes] = asyncio.Queue()
    conn_open = True
    stream_sid = None

    state = STATE_IDLE
    speak_task: asyncio.Task | None = None
    speak_lock = asyncio.Lock()
    barge_grace_until = 0.0

    first_media = asyncio.Event()
    reprompt_task: asyncio.Task | None = None
    last_heard_ts = time.time()
    last_tts_end_ts = time.time()
    REPROMPT_INTERVALS = [15.0, 30.0, 45.0]
    reprompt_idx = 0

    email_buffer = ""
    done_flag = False
    stopped_flag = False

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

    def cancel_task(t: asyncio.Task | None):
        if t and not t.done():
            t.cancel()

    async def speak(text: str, label: str = "tts"):
        nonlocal barge_grace_until, last_tts_end_ts, last_heard_ts
        t = (text or "").strip()
        if not t or stopped_flag: return
        async with speak_lock:
            cancel_task(speak_task)
            if label in {"confirm","goodbye","ask"}:
                barge_grace_until = time.time() + 0.9
            else:
                barge_grace_until = 0.0

            print(f"SAY ▶ {label}")
            try:
                pre = 50 + int(random.uniform(-10, 15))
                await send_silence(send_pcm, max(pre, 20))
                async for pcm in eleven_tts_stream_cached(t):
                    ok = await send_pcm(pcm)
                    if not ok: break
                post = 50 + int(random.uniform(-10, 15))
                await send_silence(send_pcm, max(post, 20))
            finally:
                last_tts_end_ts = time.time()
                if label in {"ask","ask-reprompt"}:
                    last_heard_ts = last_tts_end_ts

    def start_reprompt_loop():
        nonlocal reprompt_task, reprompt_idx
        reprompt_idx = 0
        cancel_task(reprompt_task)
        async def _loop():
            nonlocal reprompt_idx
            try:
                while not stopped_flag and not done_flag and state in (STATE_AWAITING_EMAIL, STATE_CONFIRM_EMAIL) and reprompt_idx < len(REPROMPT_INTERVALS):
                    await asyncio.sleep(1.0)
                    now = time.time()
                    target = REPROMPT_INTERVALS[reprompt_idx]
                    quiet_since_caller = (now - last_heard_ts) >= target
                    quiet_since_us     = (now - last_tts_end_ts) >= target
                    not_speaking       = (not speak_lock.locked())
                    if quiet_since_caller and quiet_since_us and not_speaking:
                        if state == STATE_AWAITING_EMAIL:
                            await speak("What’s the best email to reach you?", label="ask-reprompt")
                        else:
                            await speak("Please say yes or no.", label="ask-reprompt")
                        reprompt_idx += 1
                        await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                return
        reprompt_task = asyncio.create_task(_loop())

    async def confirm_and_end(captured_email: str):
        nonlocal state, email_buffer, done_flag
        email_buffer = captured_email
        cancel_task(reprompt_task)
        await speak(CONFIRM_EMAIL_OK.format(email=captured_email), label="confirm")
        await speak(GOODBYE, label="goodbye")
        state = STATE_ENDED
        done_flag = True

    YES_PAT = re.compile(r"\b(yes|yeah|yep|correct|that'?s right|affirmative|ok|okay|sure)\b", re.I)
    NO_PAT  = re.compile(r"\b(no|nope|nah|incorrect|that'?s wrong|negative)\b", re.I)

    async def ask_confirm_email(candidate: str):
        nonlocal state, email_buffer
        email_buffer = candidate
        spelled = spell_for_tts(candidate)
        await speak(CONFIRM_EMAIL_READBACK.format(email_spelled=spelled), label="confirm")
        state = STATE_CONFIRM_EMAIL

    async def handle_user(text: str):
        nonlocal state, email_buffer
        if state == STATE_CONFIRM_EMAIL:
            t = (text or "").strip()
            if YES_PAT.search(t):
                em = normalize_spoken_email(email_buffer) or (EMAIL_RE.search(email_buffer or "") and EMAIL_RE.search(email_buffer or "").group(0))
                if isinstance(em, str) and EMAIL_RE.fullmatch(em):
                    await confirm_and_end(em); return
                await speak(CONFIRM_EMAIL_NO, label="ask")
                state = STATE_AWAITING_EMAIL
                email_buffer = ""
                return
            if NO_PAT.search(t):
                email_buffer = ""
                await speak(CONFIRM_EMAIL_NO, label="ask")
                state = STATE_AWAITING_EMAIL
                return
            await speak("Please say yes or no.", label="ask-reprompt")
            return

        if state == STATE_AWAITING_EMAIL:
            if text:
                email_buffer = (email_buffer + " " + text).strip()[-400:]

            m = EMAIL_RE.search(text or "")
            if m:
                await ask_confirm_email(m.group(0)); return

            spoken = normalize_spoken_email(email_buffer)
            if spoken and EMAIL_RE.fullmatch(spoken):
                await ask_confirm_email(spoken); return

            if OPENAI_ENABLE:
                try:
                    em = await asyncio.wait_for(call_openai_email(email_buffer), timeout=OPENAI_TIMEOUT_S)
                    if em and EMAIL_RE.fullmatch(em):
                        await ask_confirm_email(em); return
                except asyncio.TimeoutError:
                    pass
            return

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
            if stopped_flag: break
            if isinstance(item, tuple) and item[0] == "__EVENT__":
                ev = item[1]
                if ev == "SpeechStarted":
                    nonlocal last_heard_ts
                    last_heard_ts = time.time()
                    if time.time() >= barge_grace_until:
                        cancel_task(speak_task)
                continue

            utter, is_final = item
            if not utter: continue
            now = time.time()
            last_heard_ts = now

            if is_final:
                if pending_final and (now - pending_ts) <= FINAL_DEBOUNCE and not re.search(r"[.!?…]\s*$", pending_final):
                    pending_final = (pending_final + " " + utter).strip()
                    pending_ts = now
                else:
                    pending_final = utter.strip()
                    pending_ts = now
                if pending_final:
                    words = pending_final.split()
                    has_punct = bool(re.search(r"[.!?…]\s*$", pending_final))
                    if has_punct or len(words) >= 2:
                        await handle_user(pending_final)
                        pending_final = None
        if not stopped_flag and pending_final:
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

                # ▶▶ Play your μ-law WAV payload instead of TTS intro
                await play_preroll_wav(send_pcm)

                # Ask for email
                await speak(ASK_EMAIL, label="ask")
                state = STATE_AWAITING_EMAIL
                now = time.time()
                last_tts_end_ts = now
                last_heard_ts   = now
                start_reprompt_loop()

            elif ev == "media":
                payload_b64 = data["media"]["payload"]
                buf = b64_to_bytes(payload_b64)
                if not first_media.is_set():
                    first_media.set()
                    print("WS ▶ first media frame")
                if inbound_q.qsize() > 50:
                    try: inbound_q.get_nowait()
                    except Exception: pass
                await inbound_q.put(buf)
                if ECHO_BACK and stream_sid and conn_open:
                    await send_pcm(buf)

            elif ev == "stop":
                print("WS ▶ stop")
                stopped_flag = True
                conn_open = False
                cancel_task(reprompt_task)
                cancel_task(speak_task)
                break

    except Exception as e:
        print("WS ERR ▶", e)
    finally:
        await inbound_q.put(None)
        try:
            await asyncio.wait_for(brain_task, timeout=2)
        except Exception:
            pass
        print("WS ▶ closed")

# --------- Server ---------
async def main():
    print("ENV ▶ HTTP_ORIGIN =", os.getenv("PUBLIC_HTTP_ORIGIN"))
    print("ENV ▶ ELEVEN key? ", "yes" if ELEVEN_KEY else "missing")
    print("ENV ▶ DEEPGRAM?   ", "yes" if DEEPGRAM_KEY else "missing")
    print("ENV ▶ ECHO_BACK   =", ECHO_BACK)
    print("ENV ▶ ELEVEN_VOICE=", ELEVEN_VOICE)
    print("ENV ▶ OPENAI      =", "yes" if OPENAI_ENABLE else "disabled")
    print("ENV ▶ PORT        =", PORT)
    print("ENV ▶ HTTP2_OK    =", HTTP2_OK)

    async with websockets.serve(
        handle_twilio, "0.0.0.0", PORT,
        max_size=2**20, ping_interval=20, ping_timeout=60,
        subprotocols=["audio.twilio.com"],
    ):
        print(f"WS bot listening on ws://0.0.0.0:{PORT}")
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        try:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(SHARED_CLIENT.aclose())
            loop.close()
        except Exception:
            pass
