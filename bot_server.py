import os, json, base64, asyncio, websockets, httpx
from dotenv import load_dotenv

# Load envs next to this file (same folder as manage.py)
load_dotenv()

# --------- ENV ---------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEEPGRAM_KEY   = os.getenv("DEEPGRAM_API_KEY")
ELEVEN_KEY     = os.getenv("ELEVENLABS_API_KEY")
HTTP_ORIGIN    = os.getenv("PUBLIC_HTTP_ORIGIN", "http://localhost:8000")
PORT = int(os.getenv("PORT", "8080"))

# Toggle to debug the audio path: 1 = echo caller audio back immediately
ECHO_BACK      = os.getenv("ECHO_BACK", "0") == "1"
# Use a real ElevenLabs voice **ID** from your dashboard (Voices -> copy ID)
ELEVEN_VOICE   = os.getenv("ELEVENLABS_VOICE_ID", "Bella")

SYSTEM_PROMPT = """
You are the NeuroMed AI receptionist. Be warm, clear, concise, and compassionate (Taglish allowed).
Goals:
- Greet and assist. If asked “what is NeuroMed AI”, give a succinct ~30-60s explanation.
- Answer FAQs: HIPAA/privacy, cost (pilot-based), pilots (3–6 months), faith-aware mode, hours/address.
Rules:
- Keep replies to 1–2 sentences unless doing the explanation.
- Never give medical advice or promise unlisted features.
"""

# --------- helpers ---------
def b64_to_bytes(s: str) -> bytes: return base64.b64decode(s)
def bytes_to_b64(b: bytes) -> str: return base64.b64encode(b).decode()

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
    "&interim_results=true"    # get partials earlier
    "&vad_events=true"         # get turn-taking faster
    "&endpointing=true"
    "&vad_turnoff=300"         # 300ms silence to finalize
)

    headers = [("Authorization", f"Token {DEEPGRAM_KEY}")]
    async with websockets.connect(url, extra_headers=headers, max_size=2**20) as dg:
        async def feeder():
            async for chunk in pcm_iter():
                await dg.send(chunk)
            await dg.send(json.dumps({"type": "CloseStream"}))
        feed_task = asyncio.create_task(feeder())

        buffer = ""
        last_emit = 0.0

        try:
            async for msg in dg:
                try:
                    obj = json.loads(msg)
                except Exception:
                    continue

                # Handle interim/final transcripts
                alt = (obj.get("channel", {}).get("alternatives") or [{}])[0]
                txt = (alt.get("transcript") or "").strip()
                is_final = bool(obj.get("is_final"))

                if not txt:
                    continue

                # Accumulate partials, throttle emits to ~200ms
                import time
                buffer = txt
                now = time.time()
                if is_final or (now - last_emit) > 0.2:
                    yield buffer
                    last_emit = now
                    if is_final:
                        buffer = ""
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
    headers = {"xi-api-key": ELEVEN_KEY, "Content-Type": "application/json"}
    payload = {"text": text, "voice_settings": {"stability": 0.5, "similarity_boost": 0.8}}
    async with httpx.AsyncClient(timeout=None) as client:
        try:
            async with client.stream("POST", url, headers=headers, json=payload) as r:
                r.raise_for_status()
                async for chunk in r.aiter_bytes():
                    if chunk:
                        yield chunk
        except httpx.HTTPStatusError as e:
            print("ELEVENLABS HTTP ERROR ▶", e.response.status_code, e.response.text[:200])
            raise
        except Exception as e:
            print("ELEVENLABS ERROR ▶", e)
            raise

# --------- Simple tool call to your Django FAQ ---------
async def call_faq(q: str) -> str:
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{HTTP_ORIGIN}/api/faq", params={"q": q})
        r.raise_for_status()
        return r.json().get("answer", "")
    
def is_greeting(s: str) -> bool:
    s = s.strip().lower()
    short = {"hi", "hello", "hey", "yo", "good morning", "good afternoon", "good evening"}
    s = s.replace("?", "")
    return s in short


async def llm_reply(history: list[dict], *, greeted: bool) -> str:
    last = history[-1]["content"].strip()
    low  = last.lower()

    if is_greeting(low):
        return "Hi! What would you like help with—appointments, pricing, or a quick overview of NeuroMed?"

    if any(k in low for k in ["hipaa","privacy","secure","security","phi","compliant","compliance","confidential"]):
        return await call_faq("hipaa")
    if any(k in low for k in ["price","pricing","cost","how much"]):
        return await call_faq("pricing")
    if "pilot" in low or "test" in low or "trial" in low:
        return await call_faq("pilot")
    if "faith" in low:
        return await call_faq("faith")
    if "what is neuromed" in low or ("what" in low and "neuromed" in low):
        return await call_faq("what_is_neuromed")
    if any(k in low for k in ["hour","open","schedule availability","hours"]):
        return await call_faq("hours")
    if any(k in low for k in ["address","location","where"]):
        return await call_faq("address")

    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": OPENAI_MODEL,
                "temperature": 0.2,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *history],
            },
        )
        r.raise_for_status()
        out = r.json()["choices"][0]["message"]["content"].strip()
        if not out.endswith("?") and len(out.split()) < 36:
            out += " How can I help further?"
        print(f"LLM ▶ {out[:80]}...")
        return out



# --------- Twilio Media Streams WS server ---------
import time

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
    }

    media_stats = {"frames": 0, "bytes": 0}
    first_media = asyncio.Event()


    speak_task: asyncio.Task | None = None

    stream_sid = None

    QUICK_ACKS = [
    "Sige.", "Got it.", "Okay.", "Noted.", "Sure.", "Sige po."
    ]
    import random

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
        - Updates state.last_reply / timestamps on success
        """
        nonlocal stream_sid, speak_task
        import time

        # Wait for Twilio to send 'start' so we have a streamSid
        if not stream_sid:
            for _ in range(100):
                await asyncio.sleep(0.01)
                if stream_sid:
                    break

        # De-dupe: avoid repeating the same line within 2.5s
        now = time.time()
        if text.strip() == state["last_reply"].strip() and (now - state["last_reply_ts"]) < 2.5:
            print("TTS ▶ duplicate suppressed")
            return

        # If a previous TTS is still speaking, cancel it (barge-in)
        if speak_task and not speak_task.done():
            speak_task.cancel()
            try:
                await speak_task
            except asyncio.CancelledError:
                pass

        async def _run():
            state["speaking"] = True
            any_chunk = False
            try:
                async for pcm in eleven_tts_stream(text):
                    any_chunk = True
                    await send_pcm(pcm)
            except asyncio.CancelledError:
                print("TTS ▶ canceled (barge-in)")
                raise
            finally:
                state["speaking"] = False
                if any_chunk:
                    state["last_reply"] = text
                    state["last_reply_ts"] = time.time()
                    print(f"TTS ▶ {text[:60]}...")
                else:
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

        # Consume ASR results and respond
        async for utter in deepgram_stream(pcm_iter):
            # Ignore tiny backchannels while bot is speaking
            if state["speaking"] and len(utter.split()) <= 1:
                continue
            history.append({"role": "user", "content": utter})
            reply = await llm_reply(history, greeted=state["greeted"])
            await speak(reply)



    brain_task = asyncio.create_task(brain())

    try:
        async for raw in ws:
            data = json.loads(raw)
            ev = data.get("event")

            if ev == "start":
                stream_sid = data.get("start", {}).get("streamSid")
                print(f"WS ▶ start streamSid={stream_sid}")
                if not state["greeted"]:
                    state["greeted"] = True
                    asyncio.create_task(
                        speak("Hi! I’m the NeuroMed assistant. What would you like help with today?")
                    )

            elif ev == "media":
                payload_b64 = data["media"]["payload"]
                buf = b64_to_bytes(payload_b64)

                # Stats & first-media signal
                media_stats["frames"] += 1
                media_stats["bytes"]  += len(buf)
                if media_stats["frames"] == 1:
                    first_media.set()
                    print("WS ▶ first media frame received")
                if media_stats["frames"] % 25 == 0:
                    print(f"WS ▶ media frames={media_stats['frames']} bytes={media_stats['bytes']}")

                # Feed ASR
                await inbound_q.put(buf)

                # Barge-in: cancel any speaking TTS when user talks
                if state["speaking"] and speak_task and not speak_task.done():
                    speak_task.cancel()
                    try:
                        await speak_task
                    except asyncio.CancelledError:
                        pass

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
