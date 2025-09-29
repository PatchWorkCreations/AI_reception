import os, json, base64, asyncio, websockets, httpx, re, time, random
from dotenv import load_dotenv

# Load envs next to this file (same folder as manage.py)
load_dotenv()

# --------- ENV ---------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEEPGRAM_KEY   = os.getenv("DEEPGRAM_API_KEY")
ELEVEN_KEY     = os.getenv("ELEVENLABS_API_KEY")
HTTP_ORIGIN    = os.getenv("PUBLIC_HTTP_ORIGIN", "http://localhost:8000")
PORT           = int(os.getenv("DEV_WS_PORT", "8080"))

# Toggle to debug the audio path: 1 = echo caller audio back immediately
ECHO_BACK      = os.getenv("ECHO_BACK", "0") == "1"
# Use a real ElevenLabs voice **ID** from your dashboard (Voices -> copy ID)
ELEVEN_VOICE   = os.getenv("ELEVENLABS_VOICE_ID", "Bella")

SYSTEM_PROMPT = """
You are the NeuroMed AI receptionist. Be warm, clear, concise, and human-sounding.
Mission:
- NeuroMed AI helps families, caregivers, and clinics by turning medical files (discharge notes, labs, imaging) into plain-language summaries, with optional styles: plain, caregiver-friendly, faith + encouragement, or clinical.

Guardrails:
- Do not give medical advice or diagnose. Never promise unlisted features. Avoid speculative claims.
- Privacy: reassure that data is kept private; do not discuss implementation details unless asked.
- If asked for pricing/partners/pilots, acknowledge tiers/pilots and offer email follow-up rather than quoting numbers.
- If the user wants next steps, politely collect name + email.

Style:
- Sound human and warm. 1–2 sentences by default; longer only for the “what is NeuroMed AI” explanation (~30–60s).
- Use light natural pacing (brief pauses are added by the audio layer). Avoid filler endings like “feel free to ask more questions.”
- If the user greets or backchannels (“okay”, “yeah”), acknowledge briefly and continue.
- If the user is unclear, ask a short clarifying question.

Output Rules:
- Stay on purpose: clarity, comfort, compassion. Use American English only (no code-switching).
- Never end with generic closers (“feel free to ask”, “how can I assist further?”) unless explicitly requested.
"""

# Few-shots to teach tone (not scripts)
FEWSHOTS = [
    {"role": "user", "content": "Can you tell me more about miramad ai?"},
    {"role": "assistant", "content": "NeuroMed AI turns medical files—like discharge notes, labs, or images—into clear summaries you can choose the tone for (plain, caregiver-friendly, faith + encouragement, or clinical). It’s built to support families, caregivers, and clinics with clarity and compassion."},

    {"role": "user", "content": "is this hipaa compliant? how private is it"},
    {"role": "assistant", "content": "Your files are kept private and secure, and we don’t share your data with anyone. If you’d like, I can send a brief summary of our privacy approach to your email."},

    {"role": "user", "content": "how much does it cost"},
    {"role": "assistant", "content": "We offer options for families, caregivers, and facilities. I can email you a quick overview so you can pick what fits best—what’s the best email to use?"},

    {"role": "user", "content": "do you have a faith based option?"},
    {"role": "assistant", "content": "Yes—there’s a mode with encouragement and Scripture for callers who want hope alongside clarity."},

    {"role": "user", "content": "ok, how do we start?"},
    {"role": "assistant", "content": "Great—can I have your name and best email? We’ll send next steps and a short demo."},
]

# --------- helpers ---------
def b64_to_bytes(s: str) -> bytes:
    return base64.b64decode(s)

def bytes_to_b64(b: bytes) -> str:
    return base64.b64encode(b).decode()

# ---- pacing helpers (μ-law silence @ 8 kHz) ----
def _ulaw_silence(ms: int) -> bytes:
    # μ-law "silence" byte is 0xFF for zero amplitude; 8 samples/ms at 8kHz
    return b"\xff" * int(8 * ms)

async def send_silence(ws_send_pcm, ms: int):
    if ms > 0:
        await ws_send_pcm(_ulaw_silence(ms))

# --------- domain correction for common mishears (PATCH 1) ---------
def domain_corrections(text: str) -> str:
    t = (text or "").strip()

    # Common brand/name mishears
    t = re.sub(r"\bmiu?ra?med\b", "NeuroMed", t, flags=re.I)
    t = re.sub(r"\bmira\s*med\b", "NeuroMed", t, flags=re.I)
    t = re.sub(r"\bneuro\s*med\b", "NeuroMed", t, flags=re.I)

    # “… tell me more / more about … video/media” → “NeuroMed AI”
    if re.search(r"\b(tell me more|more about)\b", t, flags=re.I) and re.search(r"\b(video|media)\b\.?\??$", t, flags=re.I):
        t = re.sub(r"\b(video|media)\b\.?\??$", "NeuroMed AI", t, flags=re.I)

    # “… EA?” / “… AI?” (end of sentence) → “NeuroMed AI?”
    t = re.sub(r"\bE\.?A\.?\??$", "NeuroMed AI?", t, flags=re.I)
    t = re.sub(r"\bA\.?I\.?\??$", "NeuroMed AI?", t, flags=re.I)
    t = re.sub(r"\bEA\??$",      "NeuroMed AI?", t, flags=re.I)

    # “your …” trailing → normalize to brand
    if re.search(r"\byour\s+(?:ai|app|system)?\s*\??$", t, flags=re.I):
        t = re.sub(r"\byour\s+(?:ai|app|system)?\s*\??$", "NeuroMed AI?", t, flags=re.I)

    return t

# --------- Deepgram ASR (ws) ---------
async def deepgram_stream(pcm_iter):
    """
    Stream μ-law 8k audio to Deepgram and yield (text, is_final).
    - Throttles partials (~200ms), always emits finals.
    - Handles dict/list channel payload shapes.
    """
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
        # Optional: bias the recognizer (best-effort)
        try:
            await dg.send(json.dumps({
                "type": "Configure",
                "keywords": ["NeuroMed", "NeuroMed AI", "HIPAA", "pilot", "pricing", "faith"]
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

                # Ignore non-transcript events
                if isinstance(obj, dict) and obj.get("type") in {
                    "Metadata", "Warning", "Error", "Close", "UtteranceEnd",
                    "SpeechStarted", "SpeechFinished"
                }:
                    continue

                # Normalize shapes
                alts = []
                is_final = bool(obj.get("is_final"))
                chan = obj.get("channel")
                if isinstance(chan, dict):
                    alts = chan.get("alternatives") or []
                elif isinstance(chan, list) and chan:
                    first_chan = chan[0] or {}
                    if isinstance(first_chan, dict):
                        alts = first_chan.get("alternatives") or []
                elif "alternatives" in obj and isinstance(obj["alternatives"], list):
                    alts = obj["alternatives"]

                if not alts:
                    continue

                txt = (alts[0].get("transcript") or "").strip()
                if not txt:
                    continue

                # Throttle partials; always emit finals; skip unchanged partials
                now = time.time()
                if not (is_final or (now - last_emit) > 0.2):
                    continue
                if not is_final and txt == last_sent:
                    continue

                print(f"ASR{'(final)' if is_final else ''} ▶ {txt}")
                yield txt, is_final
                last_emit = now
                last_sent = txt
        finally:
            await feed_task

# --------- ElevenLabs TTS (ulaw_8000 stream) ---------
async def eleven_tts_stream(text: str):
    if not ELEVEN_KEY:
        raise RuntimeError("ELEVENLABS_API_KEY missing.")

    url = (
        f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE}/stream"
        "?optimize_streaming_latency=3&output_format=ulaw_8000"
    )
    headers = {
        "xi-api-key": ELEVEN_KEY,
        "Content-Type": "application/json",
        "Accept": "*/*",
    }
    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
    }

    async with httpx.AsyncClient(timeout=None) as client:
        try:
            async with client.stream("POST", url, headers=headers, json=payload) as r:
                if r.is_error:
                    body = await r.aread()
                    print("ELEVENLABS HTTP ERROR ▶", r.status_code, body[:200])
                    return
                async for chunk in r.aiter_bytes():
                    if chunk:
                        yield chunk
        except httpx.HTTPStatusError as e:
            body = b""
            try:
                body = await e.response.aread()
            except Exception:
                pass
            print("ELEVENLABS EXC ▶", getattr(e.response, "status_code", "?"), body[:200])
            return
        except Exception as e:
            print("ELEVENLABS ERROR ▶", repr(e))
            return

# --------- Simple tool call to your Django FAQ ---------
async def call_faq(q: str) -> str:
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{HTTP_ORIGIN}/api/faq", params={"q": q})
        r.raise_for_status()
        return r.json().get("answer", "")

def looks_like_what_is_neuromed(s: str) -> bool:
    q = re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower())
    if ("what is" in q) or ("what s" in q) or ("what's" in q):
        keywords = ["neuromed", "neuro med", "miura", "miramad", "mira med", "your med ai", "neuro ai", "med ai"]
        return any(k in q for k in keywords)
    return False

# --------- Streaming LLM sentences via SSE ---------
SENTENCE_END = re.compile(r'([.!?…]+)(\s+|$)')

async def stream_llm_sentences(messages: list[dict]):
    """
    Async generator: yields sentences as the LLM streams tokens (SSE).
    """
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    payload = {
        "model": OPENAI_MODEL,
        "temperature": 0.2,
        "stream": True,
        "messages": messages,
    }

    buf = ""
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except Exception:
                    continue
                delta = ((obj.get("choices") or [{}])[0].get("delta") or {}).get("content") or ""
                if not delta:
                    continue
                buf += delta

                # Emit complete sentences as soon as they’re ready
                while True:
                    m = SENTENCE_END.search(buf)
                    if not m:
                        break
                    end_idx = m.end()
                    sent = buf[:end_idx].strip()
                    buf = buf[end_idx:]
                    if sent:
                        yield sent
    # Flush any tail text (short last sentence)
    tail = buf.strip()
    if tail:
        yield tail

# --------- Intent & slot extraction ---------
INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": [
            "greeting","what_is","hipaa","pricing","faith_mode","pilot","hours",
            "address","schedule_contact","smalltalk","other"
        ]},
        "name":   {"type":"string"},
        "email":  {"type":"string"},
        "org":    {"type":"string"},
        "notes":  {"type":"string"}
    },
    "required": ["intent"]
}

async def extract_intent_slots(utterance: str) -> dict:
    """
    Use Chat Completions tool-calling to force a structured JSON result.
    Falls back gracefully if anything goes wrong.
    """
    tool_schema = {
        "type": "function",
        "function": {
            "name": "set_intent",
            "description": "Set the user's intent and any available contact info.",
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {"type": "string", "enum": [
                        "greeting","what_is","hipaa","pricing","faith_mode","pilot","hours",
                        "address","schedule_contact","smalltalk","other"
                    ]},
                    "name":  {"type":"string"},
                    "email": {"type":"string"},
                    "org":   {"type":"string"},
                    "notes": {"type":"string"}
                },
                "required": ["intent"],
                "additionalProperties": False
            }
        }
    }

    sys = {"role":"system","content":"Given the last user utterance, call the set_intent function with extracted fields. If uncertain, use intent='other'."}
    usr = {"role":"user","content": utterance}

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json={
                    "model": OPENAI_MODEL,
                    "temperature": 0.0,
                    "messages": [sys, usr],
                    "tools": [tool_schema],
                    "tool_choice": {"type":"function","function":{"name":"set_intent"}},
                }
            )
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            tool_calls = msg.get("tool_calls") or []
            if tool_calls and tool_calls[0]["function"]["name"] == "set_intent":
                args_raw = tool_calls[0]["function"].get("arguments") or "{}"
                try:
                    return json.loads(args_raw)
                except Exception:
                    return {"intent":"other"}
    except Exception as e:
        print("INTENT ERROR ▶", repr(e))

    return {"intent":"other"}

# --------- Planner (what to say) ---------
WEAK_CLOSERS = {
  "if you have more questions, feel free to ask.",
  "if you have any questions, feel free to ask.",
  "if you have any specific questions or need assistance, feel free to ask.",
  "let me know if you have other questions.",
  "how can i assist you further?",
  "feel free to ask!",
  "feel free to ask more questions.",
}

def should_skip_sentence(s: str) -> bool:
    return (s or "").strip().lower() in WEAK_CLOSERS

GREETING_RE = re.compile(r"^(hi|hello|hey)[,!.\s]*(?:i'?m|this is)?\b", re.I)
ASSIST_RE   = re.compile(r"\b(how can i (?:help|assist)|let me know|feel free to ask)\b", re.I)

def should_drop_assistant_line(s: str, *, greeted_already: bool) -> bool:
    s0 = (s or "").strip()
    if not s0:
        return True
    low = s0.lower()
    if low in WEAK_CLOSERS:
        return True
    # After greeting, block “Hello/Hi … how can I assist…” repetitions
    if greeted_already and (GREETING_RE.search(s0) or ASSIST_RE.search(s0)):
        return True
    return False

def should_block_contact_line(s: str, *, allow_contact_request: bool) -> bool:
    """
    Suppress lines that prematurely ask for email/contact unless we've observed intent/opening.
    """
    if allow_contact_request:
        return False
    return bool(re.search(r"\b(email|best email|contact|reach you)\b", (s or ""), re.I))

def extract_contact_inline(txt: str) -> dict:
    """
    Very light inline extraction to capture an email or a likely name token.
    """
    out = {}
    # email
    m = re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", txt, re.I)
    if m:
        out["email"] = m.group(0)
    # naive name (first + last) if caller states "My name is ..."
    m2 = re.search(r"\bmy name is ([A-Za-z][A-Za-z\-']+\s+[A-Za-z][A-Za-z\-']+)\b", txt, re.I)
    if m2:
        out["name"] = m2.group(1).strip()
    return out

async def plan_reply(history: list[dict], state: dict | None = None) -> list[str]:
    """Return a list of sentences to speak. Empty list => use generative streaming."""
    last = (history[-1]["content"] if history else "").strip()
    slots = await extract_intent_slots(last)
    intent = slots.get("intent","other")

    if intent == "greeting":
        return ["Hi! What would you like help with today—overview, pricing, or getting started?"]
    if intent == "hipaa":
        return ["Your files are kept private and secure, and we don’t share your data with anyone."]
    if intent == "pricing":
        return ["We offer options for families, caregivers, and facilities. I can send a quick overview—what’s the best email to use?"]
    if intent == "faith_mode":
        return ["Yes—there’s a mode with encouragement and Scripture for those who want hope alongside clarity."]
    if intent == "pilot":
        return ["We do pilot programs with nursing homes, clinics, and community groups. I can email details—what email should we use?"]
    if intent == "hours":
        return ["We’re available weekdays; if you share your time zone, I can suggest a slot."]
    if intent == "address":
        return ["We operate online and partner with facilities; if you share your city, I can route you."]

    if intent == "what_is":
        text = (
          "NeuroMed AI turns medical files like discharge notes, labs, and imaging into plain-language summaries. "
          "You can choose the tone—plain, caregiver-friendly, faith + encouragement, or clinical—so it meets the moment. "
          "It’s built to help families, caregivers, and clinics find clarity quickly without medical advice."
        )
        return [s for s in re.split(r'(?<=[.!?])\s+', text) if s and not should_skip_sentence(s)]

    if intent == "schedule_contact":
        name = slots.get("name")
        email = slots.get("email")
        if name and email:
            return [f"Thanks, {name}. We’ll email next steps to {email} shortly."]
        return ["Great—could I have your name and best email so we can send next steps?"]

    # Fallback to generative (use FEWSHOTS + SYSTEM_PROMPT)
    return []

# --------- Twilio Media Streams WS server ---------
async def handle_twilio(ws):
    # Accept only the expected path
    if ws.path not in ("/ws/twilio",):
        await ws.close(code=1008, reason="Unexpected path")
        return

    print("WS ▶ Twilio connected, path:", ws.path)
    print("WS ▶ protocol negotiated:", ws.subprotocol)

    inbound_q: asyncio.Queue[bytes] = asyncio.Queue()
    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Conversation/runtime state
    state = {
        "greeted": False,
        "speaking": False,
        "last_reply": "",
        "last_reply_ts": 0.0,
        "barge_in_enabled": False,   # armed after greeting grace
        "long_answer_until": 0.0,    # grace window while bot is explaining
        "last_ack_ts": 0.0,          # rate limit quick acks
        "local_no_barge_until": 0.0, # per-utterance grace while starting TTS
        "allow_contact_request": False,
        "contact": {"name": None, "email": None},
    }

    async def arm_barge_in_after(seconds: float):
        await asyncio.sleep(seconds)
        state["barge_in_enabled"] = True
        print(f"BARGe-IN ▶ enabled after {seconds:.1f}s")

    media_stats = {"frames": 0, "bytes": 0}
    first_media = asyncio.Event()

    speak_task: asyncio.Task | None = None
    stream_sid = None

    QUICK_ACKS = ["Okay.", "Got it.", "Sure.", "Alright.", "Understood.", "One moment."]

    async def quick_ack():
        now = time.time()
        if state["speaking"]:
            return
        if now - state["last_ack_ts"] < 1.2:
            return
        state["last_ack_ts"] = now
        await speak(random.choice(QUICK_ACKS))

    async def pcm_iter():
        while True:
            b = await inbound_q.get()
            if b is None:
                break
            yield b

    async def send_pcm(pcm: bytes):
        await ws.send(json.dumps({
            "event": "media",
            "streamSid": stream_sid,
            "media": {"payload": bytes_to_b64(pcm)}
        }))

    async def speak(text: str):
        """
        Cancelable TTS via ElevenLabs, streamed back to Twilio.
        - Cancels any in-flight TTS (barge-in)
        - Suppresses duplicate replies within 2.5s
        - Adds small randomized pre/post silences for natural pacing
        - Per-utterance no-barge grace (~1.2s)
        """
        nonlocal stream_sid, speak_task

        text = (text or "").strip()
        if not text:
            return

        # Wait for Twilio 'start' so we have a streamSid
        if not stream_sid:
            for _ in range(100):
                await asyncio.sleep(0.01)
                if stream_sid:
                    break

        # De-dupe
        now = time.time()
        if text == state["last_reply"].strip() and (now - state["last_reply_ts"]) < 2.5:
            print("TTS ▶ duplicate suppressed")
            return

        # If previous TTS is still speaking, cancel it (barge-in)
        if speak_task and not speak_task.done():
            speak_task.cancel()
            try:
                await speak_task
            except asyncio.CancelledError:
                pass

        async def _run():
            state["speaking"] = True
            any_chunk = False
            cancelled = False
            # Per-utterance no-barge grace
            prev_barge = state.get("barge_in_enabled", False)
            state["local_no_barge_until"] = time.time() + 1.2

            try:
                # Pre-breath (randomized)
                pre = 120 if len(text.split()) > 1 else 40
                pre += int(random.uniform(-15, 20))
                await send_silence(send_pcm, max(pre, 20))

                async for pcm in eleven_tts_stream(text):
                    any_chunk = True
                    await send_pcm(pcm)

                # Post-pause (randomized)
                post = 60 + int(random.uniform(-15, 25))
                await send_silence(send_pcm, max(post, 20))

            except asyncio.CancelledError:
                cancelled = True
                print("TTS ▶ canceled (barge-in)")
                raise
            finally:
                state["barge_in_enabled"] = prev_barge
                state["speaking"] = False
                if any_chunk:
                    state["last_reply"] = text
                    state["last_reply_ts"] = time.time()
                    print(f"TTS ▶ {text[:60]}...")
                else:
                    if not cancelled:
                        print("TTS ▶ no audio produced (check TTS provider)")

        speak_task = asyncio.create_task(_run())

    # --------- brain loop with final debouncing (PATCH 2) ---------
    async def brain():
        if not DEEPGRAM_KEY:
            return

        try:
            await asyncio.wait_for(first_media.wait(), timeout=20.0)
        except asyncio.TimeoutError:
            print("ASR ▶ no media within 20s; skipping ASR session")
            return

        last_user_sent = ""
        FINAL_DEBOUNCE_SECS = 0.9
        pending_final = None
        pending_ts = 0.0

        async def process_final_utterance(text: str):
            fixed = domain_corrections(text or "")
            if not fixed:
                return

            # Opportunistic contact extraction
            info = extract_contact_inline(fixed)
            if "email" in info:
                state["contact"]["email"] = info["email"]
            if "name" in info and not state["contact"]["name"]:
                state["contact"]["name"] = info["name"]

            # If caller explicitly asked for follow-up, allow contact request
            if re.search(r"\b(email|send|share)\b.*\b(info|details|pricing|pilot|follow\s*up)\b", fixed, re.I):
                state["allow_contact_request"] = True

            # Barge if substantial and currently speaking
            words = fixed.split()
            nowt = time.time()
            in_long_window = nowt < state["long_answer_until"]
            should_barge = len(words) >= (5 if in_long_window else 3)
            if should_barge and state.get("barge_in_enabled") and state["speaking"] and speak_task and not speak_task.done():
                speak_task.cancel()
                try:
                    await speak_task
                except asyncio.CancelledError:
                    pass

            # Route “what is NeuroMed …” directly to FAQ
            if looks_like_what_is_neuromed(fixed):
                if not state["speaking"]:
                    asyncio.create_task(speak("Sure—"))
                answer = await call_faq("what_is_neuromed")
                state["long_answer_until"] = time.time() + 2.0
                for sentence in re.split(r'(?<=[.!?])\s+', (answer or "").strip()):
                    if sentence and not should_skip_sentence(sentence):
                        if not should_drop_assistant_line(sentence, greeted_already=state["greeted"]):
                            await speak(sentence)
                return

            # Normal: add to history
            history.append({"role": "user", "content": fixed})

            # Quick ack for short turns
            if len(words) <= 6:
                asyncio.create_task(quick_ack())

            # Plan or generative
            state["long_answer_until"] = time.time() + 2.0
            try:
                plan = await plan_reply(history, state=state)
            except Exception as e:
                print("PLAN ERROR ▶", repr(e))
                plan = []

            if plan:
                for sentence in plan:
                    if should_skip_sentence(sentence):
                        continue
                    if should_block_contact_line(sentence, allow_contact_request=state["allow_contact_request"]):
                        continue
                    if should_drop_assistant_line(sentence, greeted_already=state["greeted"]):
                        continue
                    await speak(sentence)
            else:
                gen_messages = [{"role":"system","content": SYSTEM_PROMPT}, *FEWSHOTS, *history]
                async for sentence in stream_llm_sentences(gen_messages):
                    if should_skip_sentence(sentence):
                        continue
                    if should_block_contact_line(sentence, allow_contact_request=state["allow_contact_request"]):
                        continue
                    if should_drop_assistant_line(sentence, greeted_already=state["greeted"]):
                        continue
                    await speak(sentence)

        async for utter, is_final in deepgram_stream(pcm_iter):
            if not utter:
                continue

            # De-dupe identical emissions
            if utter == last_user_sent:
                continue
            last_user_sent = utter

            # Ignore tiny backchannels while bot is speaking
            if state["speaking"] and len(utter.split()) <= 1:
                continue

            # Flush stale pending final if it sat too long
            now = time.time()
            if pending_final and (now - pending_ts) > FINAL_DEBOUNCE_SECS:
                await process_final_utterance(pending_final)
                pending_final = None

            if is_final:
                # Merge fragmented finals if they arrive quickly and previous didn't end with punctuation
                if pending_final and (now - pending_ts) <= FINAL_DEBOUNCE_SECS and not re.search(r"[.!?…]\s*$", pending_final):
                    pending_final = (pending_final + " " + utter).strip()
                    pending_ts = now
                else:
                    pending_final = utter.strip()
                    pending_ts = now

                # If complete-looking, process immediately
                if re.search(r"[.!?…]\s*$", pending_final) or len(pending_final.split()) >= 8:
                    await process_final_utterance(pending_final)
                    pending_final = None
            else:
                # For partials, just wait for subsequent finals
                pass

        # Session ended: flush any leftover pending final
        if pending_final:
            await process_final_utterance(pending_final)

    brain_task = asyncio.create_task(brain())

    try:
        async for raw in ws:
            data = json.loads(raw)
            ev = data.get("event")

            if ev == "start":
                start_info = data.get("start", {}) or {}
                stream_sid = start_info.get("streamSid")
                call_sid = start_info.get("callSid")
                print(f"WS ▶ start streamSid={stream_sid} callSid={call_sid}")
                state["barge_in_enabled"] = False  # ensure off at start
                if not state["greeted"]:
                    state["greeted"] = True
                    asyncio.create_task(
                        speak("Hi! I’m the NeuroMed assistant. What do you need today—pricing, hours, or a quick overview?")
                    )
                    # fallback arming in case first media is late
                    asyncio.create_task(arm_barge_in_after(1.5))

            elif ev == "media":
                payload_b64 = data["media"]["payload"]
                buf = b64_to_bytes(payload_b64)

                media_stats["frames"] += 1
                media_stats["bytes"]  += len(buf)
                if media_stats["frames"] == 1:
                    first_media.set()
                    print("WS ▶ first media frame received")
                    # Let greeting start for ~1.2s before allowing interruptions
                    asyncio.create_task(arm_barge_in_after(1.2))

                if media_stats["frames"] % 25 == 0:
                    print(f"WS ▶ media frames={media_stats['frames']} bytes={media_stats['bytes']}")

                await inbound_q.put(buf)

                # Optional loopback: only for debugging when ECHO_BACK=1
                if ECHO_BACK and stream_sid:
                    await send_pcm(buf)

            elif ev == "stop":
                print("WS ▶ stop")
                break

    except Exception as e:
        print("WS ERR ▶", e)
    finally:
        await inbound_q.put(None)
        await brain_task
        print("WS ▶ closed")

async def main():
    # Quick env prints
    print("ENV ▶ HTTP_ORIGIN =", os.getenv("PUBLIC_HTTP_ORIGIN"))
    print("ENV ▶ WS URL hint =", os.getenv("PUBLIC_WS_URL"))
    print("ENV ▶ ELEVEN key? ", "yes" if ELEVEN_KEY else "missing")
    print("ENV ▶ DEEPGRAM?   ", "yes" if DEEPGRAM_KEY else "missing")
    print("ENV ▶ ECHO_BACK   =", ECHO_BACK)
    print("ENV ▶ ELEVEN_VOICE=", ELEVEN_VOICE)
    print("ENV ▶ PORT        =", PORT)

    async with websockets.serve(
        handle_twilio,
        "0.0.0.0",
        PORT,                       # ← bind to Railway-assigned port
        max_size=2**20,
        ping_interval=20,
        ping_timeout=60,
        subprotocols=["audio.twilio.com"],  # ← REQUIRED by Twilio
    ):
        print(f"WS bot listening on ws://0.0.0.0:{PORT}")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
