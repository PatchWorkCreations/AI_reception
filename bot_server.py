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
        "?model=nova-2&punctuate=true&interim_results=false&encoding=mulaw&sample_rate=8000"
    )
    headers = [("Authorization", f"Token {DEEPGRAM_KEY}")]
    async with websockets.connect(url, extra_headers=headers, max_size=2**20) as dg:
        async def feeder():
            async for chunk in pcm_iter():
                # Twilio sends 8k μ-law bytes; Deepgram accepts raw binary frames here
                await dg.send(chunk)
            await dg.send(json.dumps({"type": "CloseStream"}))

        feed_task = asyncio.create_task(feeder())
        try:
            async for msg in dg:
                try:
                    obj = json.loads(msg)
                    alts = obj.get("channel", {}).get("alternatives") or []
                    if alts:
                        txt = (alts[0].get("transcript") or "").strip()
                        if txt:
                            print(f"ASR ▶ {txt}")
                            yield txt
                except Exception:
                    continue
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

# --------- LLM router ---------
async def llm_reply(history: list[dict]) -> str:
    last = history[-1]["content"].lower()
    if any(k in last for k in ["hipaa","privacy","secure","security","phi","compliant","compliance","confidential"]):
        return await call_faq("hipaa")
    if any(k in last for k in ["price","pricing","cost","how much"]):
        return await call_faq("pricing")
    if "pilot" in last or "test" in last or "trial" in last:
        return await call_faq("pilot")
    if "faith" in last:
        return await call_faq("faith")
    if "what is neuromed" in last or ("what" in last and "neuromed" in last):
        return await call_faq("what_is_neuromed")
    if any(k in last for k in ["hour","open","schedule availability"]):
        return await call_faq("hours")
    if any(k in last for k in ["address","location","where"]):
        return await call_faq("address")

    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": OPENAI_MODEL,
                "temperature": 0.3,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *history],
            },
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        print(f"LLM ▶ {text[:80]}...")
        return text

# --------- Twilio Media Streams WS server ---------
# --------- Twilio Media Streams WS server ---------
async def handle_twilio(ws):
    # Accept only the expected path (Railway public URL ends with /ws/twilio)
    if ws.path not in ("/ws/twilio",):
        await ws.close(code=1008, reason="Unexpected path")
        return

    print("WS ▶ Twilio connected, path:", ws.path)
    print("WS ▶ protocol negotiated:", ws.subprotocol)

    inbound_q: asyncio.Queue[bytes] = asyncio.Queue()
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    stream_sid = None

    async def pcm_iter():
        while True:
            b = await inbound_q.get()
            if b is None:
                break
            yield b

    async def send_pcm(pcm: bytes):
        """Send one ulaw_8000 audio chunk back to Twilio."""
        await ws.send(json.dumps({
            "event": "media",
            "streamSid": stream_sid,
            "media": {"payload": bytes_to_b64(pcm)}
        }))

    async def speak(text: str):
        """TTS via ElevenLabs, streamed back to Twilio."""
        nonlocal stream_sid
        if not stream_sid:
            for _ in range(100):
                await asyncio.sleep(0.01)
                if stream_sid:
                    break
        async for pcm in eleven_tts_stream(text):
            await send_pcm(pcm)
        print(f"TTS ▶ {text[:60]}...")

    async def brain():
        if DEEPGRAM_KEY:
            async for text in deepgram_stream(pcm_iter):
                history.append({"role": "user", "content": text})
                reply = await llm_reply(history)
                await speak(reply)

    brain_task = asyncio.create_task(brain())

    try:
        async for raw in ws:
            data = json.loads(raw)
            ev = data.get("event")
            if ev == "start":
                stream_sid = data.get("start", {}).get("streamSid")
                print(f"WS ▶ start streamSid={stream_sid}")
                asyncio.create_task(
                    speak("Hi, I’m your NeuroMed assistant. How can I help you today?")
                )
            elif ev == "media":
                buf = b64_to_bytes(data["media"]["payload"])
                await inbound_q.put(buf)
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
