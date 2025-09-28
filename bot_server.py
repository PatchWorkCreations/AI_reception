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
PORT           = int(os.getenv("PORT", "8080"))

# Toggle to debug the audio path: 1 = echo caller audio back immediately
ECHO_BACK      = os.getenv("ECHO_BACK", "0") == "1"
# Use a real ElevenLabs voice **ID** from your dashboard (Voices -> copy ID)
ELEVEN_VOICE   = os.getenv("ELEVENLABS_VOICE_ID", "Bella")

SYSTEM_PROMPT = """
You are the NeuroMed AI receptionist. Be warm, clear, concise, and human-sounding.
Goals:
- Greet and assist. If asked “what is NeuroMed AI”, give a succinct ~30–60s explanation.
- Answer FAQs: HIPAA/privacy, cost (pilot-based), pilots (3–6 months), faith-aware mode, hours/address.
Rules:
- Keep replies to 1–2 sentences unless doing the explanation.
- Never give medical advice or promise unlisted features.
- Speak in American English.
"""

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

# --------- Deepgram ASR (ws) ---------
async def deepgram_stream(pcm_iter):
    if not DEEPGRAM_KEY:
        print("ASR WARN ▶ DEEPGRAM_API_KEY missing; ASR disabled.")
        return

    url = (
        "wss://api.deepgram.com/v1/listen"
        "?model=nova-2"
        "&encoding=mulaw"
        "&sample_rate=8000"
        "&punctuate=true"
        "&interim_results=true"
        "&vad_events=true"
        "&endpointing=true"
        "&vad_turnoff=300"
    )
    headers = [("Authorization", f"Token {DEEPGRAM_KEY}")]

    async with websockets.connect(url, extra_headers=headers, max_size=2**20) as dg:
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
                # Politely close DG stream, ignore if already closed
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
                    "Metadata","Warning","Error","Close","UtteranceEnd",
                    "SpeechStarted","SpeechFinished"
                }:
                    continue

                # Normalize shapes: dict or list channel, or top-level alternatives
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
                should_emit = is_final or (now - last_emit) > 0.2
                if not should_emit:
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
        "?optimize_streaming_latency=4&output_format=ulaw_8000"
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
                    body = await r.aread()  # must read body on streaming responses
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

def is_greeting(s: str) -> bool:
    s = s.strip().lower().replace("?", "")
    short = {"hi", "hello", "hey", "yo", "good morning", "good afternoon", "good evening"}
    return s in short

def looks_like_what_is_neuromed(s: str) -> bool:
    q = re.sub(r"[^a-z0-9 ]+", " ", s.lower())
    if ("what is" in q) or ("what s" in q) or ("what's" in q):
        # catch misspellings / variants
        keywords = ["neuromed", "neuro med", "miura", "miramad", "mira med", "your med ai", "neuro ai", "med ai"]
        return any(k in q for k in keywords)
    return False

# --------- Streaming LLM sentences via SSE ---------
SENTENCE_END = re.compile(r'([.!?…]+)(\s+|$)')

async def stream_llm_sentences(history: list[dict]):
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
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *history],
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
        "barge_in_enabled": False,   # armed after greeting
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
        if not state["speaking"]:
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
        - Adds small pre/post silences for natural pacing
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

        # De-dupe: avoid repeating the same line within 2.5s
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
            try:
                # Pre-breath (120 ms)
                await send_silence(send_pcm, 120)

                async for pcm in eleven_tts_stream(text):
                    any_chunk = True
                    await send_pcm(pcm)

                # Post-pause (80 ms)
                await send_silence(send_pcm, 80)

            except asyncio.CancelledError:
                cancelled = True
                print("TTS ▶ canceled (barge-in)")
                raise
            finally:
                state["speaking"] = False
                if any_chunk:
                    state["last_reply"] = text
                    state["last_reply_ts"] = time.time()
                    print(f"TTS ▶ {text[:60]}...")
                else:
                    if not cancelled:
                        print("TTS ▶ no audio produced (check TTS provider)")

        speak_task = asyncio.create_task(_run())

    async def brain():
        if not DEEPGRAM_KEY:
            return

        # Wait until Twilio actually sends audio so Deepgram won't time out
        try:
            await asyncio.wait_for(first_media.wait(), timeout=20.0)
        except asyncio.TimeoutError:
            print("ASR ▶ no media within 20s; skipping ASR session")
            return

        last_user_sent = ""

        async for utter, is_final in deepgram_stream(pcm_iter):
            utter = (utter or "").strip()
            if not utter:
                continue

            # De-dupe identical emissions
            if utter == last_user_sent:
                continue
            last_user_sent = utter

            # Ignore tiny backchannels while bot is speaking
            if state["speaking"] and len(utter.split()) <= 1:
                continue

            # Barge-in only on finals or longer partials (≥4 words) and only if armed
            should_barge = is_final or (len(utter.split()) >= 4)
            if should_barge and state.get("barge_in_enabled") and state["speaking"] and speak_task and not speak_task.done():
                speak_task.cancel()
                try:
                    await speak_task
                except asyncio.CancelledError:
                    pass

            # Route "what is neuromed" style queries (with typos) to FAQ, sentence-by-sentence
            if is_final and looks_like_what_is_neuromed(utter):
                if not state["speaking"]:
                    asyncio.create_task(speak("Sure—"))
                answer = await call_faq("what_is_neuromed")
                for sentence in re.split(r'(?<=[.!?])\s+', (answer or "").strip()):
                    if sentence:
                        await speak(sentence)
                continue

            # Friendly quick-ack on finals (if not currently speaking)
            if is_final and not state["speaking"]:
                asyncio.create_task(speak(random.choice(QUICK_ACKS)))

            # Normal streaming: update history and stream LLM sentences (only on finals)
            history.append({"role": "user", "content": utter})
            if is_final:
                async for sentence in stream_llm_sentences(history):
                    await speak(sentence)
            # If partial: do nothing (let it accumulate)

    brain_task = asyncio.create_task(brain())

    try:
        async for raw in ws:
            data = json.loads(raw)
            ev = data.get("event")

            if ev == "start":
                stream_sid = data.get("start", {}).get("streamSid")
                print(f"WS ▶ start streamSid={stream_sid}")
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
