import os
import json
import asyncio
import websockets
import httpx
from channels.generic.websocket import AsyncWebsocketConsumer
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPGRAM_KEY   = os.getenv("DEEPGRAM_API_KEY")
ELEVEN_KEY     = os.getenv("ELEVENLABS_API_KEY")
ELEVEN_VOICE   = os.getenv("ELEVENLABS_VOICE_ID", "Bella")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Startup diagnostics
print("=" * 60)
print("VoiceAI Consumer - Environment Check")
print("=" * 60)
if DEEPGRAM_KEY:
    print(f"DEEPGRAM_KEY: ✅ SET (starts: {DEEPGRAM_KEY[:16]}... length: {len(DEEPGRAM_KEY)})")
    if len(DEEPGRAM_KEY) < 20:
        print("  ⚠️  WARNING: Key looks too short - may be invalid")
else:
    print("DEEPGRAM_KEY: ❌ NOT SET")

if ELEVEN_KEY:
    print(f"ELEVEN_KEY:   ✅ SET (starts: {ELEVEN_KEY[:16]}... length: {len(ELEVEN_KEY)})")
else:
    print("ELEVEN_KEY:   ❌ NOT SET")

print(f"ELEVEN_VOICE: {ELEVEN_VOICE}")
if ELEVEN_VOICE == "Bella" or len(ELEVEN_VOICE) < 15:
    print("  ⚠️  WARNING: Use a Voice ID (UUID), not a name. Get from: https://elevenlabs.io/app/voice-library")

if OPENAI_API_KEY:
    print(f"OPENAI_KEY:   ✅ SET (starts: {OPENAI_API_KEY[:16]}... length: {len(OPENAI_API_KEY)})")
else:
    print("OPENAI_KEY:   ❌ NOT SET")
print("=" * 60)

DEEPGRAM_URL = (
    "wss://api.deepgram.com/v1/listen"
    "?model=nova-2"
    "&container=webm"
    "&encoding=opus"
    "&sample_rate=48000"
    "&channels=1"
    "&interim_results=true"
    "&punctuate=true"
    "&endpointing=true"
    "&vad_events=true"
    "&no_speech_timeout=20000"
)

RECEPTIONIST_MENU = (
    "I can help with: overview, privacy, pricing, pilot programs, hours, or getting started. "
    "What would you like to know?"
)

def route_intent(user_text: str) -> str:
    """
    Simple intent router for receptionist-style replies.
    Returns short, helpful responses based on user intent.
    """
    t = (user_text or "").lower()
    if any(k in t for k in ["overview", "what is", "what does", "how it works"]):
        return ("NeuroMed AI turns medical files like discharge notes, labs, and imaging into clear summaries. "
                "You can choose the tone—plain, caregiver-friendly, faith + encouragement, or clinical. " + RECEPTIONIST_MENU)
    if "privacy" in t or "secure" in t or "security" in t:
        return "Your files stay private and secure. We don't sell data. We can sign a BAA if needed. " + RECEPTIONIST_MENU
    if "price" in t or "pricing" in t or "cost" in t or "plans" in t:
        return "We have options for families and facilities. I can email a one-pager—what's the best email to use?"
    if "pilot" in t or "trial" in t or "demo" in t:
        return "We run pilots with nursing homes, clinics, and community groups. I can email the pilot details—what email should I use?"
    if "hours" in t or "available" in t or "when" in t or "time" in t:
        return "We're available on weekdays. Share your time zone and I can suggest a slot."
    if "start" in t or "getting started" in t or "onboard" in t:
        return "Great! Can I have your name and best email so I can set up your account?"
    # default: quick, kind reply
    return "Got it. " + RECEPTIONIST_MENU

class VoiceAIConsumer(AsyncWebsocketConsumer):
    """
    One WebSocket:
      - Browser sends mic chunks (audio/webm; codecs=opus)
      - We relay to Deepgram realtime
      - On final transcript, call OpenAI for a SHORT reply (1–2 sentences)
      - Stream ElevenLabs TTS audio bytes back to the browser (binary frames)
    """

    async def connect(self):
        await self.accept()
        self._dg_ws = None
        self._relay_task = None
        self._dg_keepalive = None
        self._closing = False
        self._first_audio = True  # Track first audio chunk

        # Connect to Deepgram
        await self._ensure_deepgram()

        # Start background tasks
        self._relay_task = asyncio.create_task(self._deepgram_listener())
        self._dg_keepalive = asyncio.create_task(self._deepgram_heartbeat())
        
        await self.send_json({"type": "ready", "message": "voice socket ready"})
        
        # Send spoken greeting to confirm audio path
        print("[VoiceAI] Sending greeting via ElevenLabs...")
        await self._speak("Hi! Welcome to NeuroMed AI. " + RECEPTIONIST_MENU)
        print("[VoiceAI] Greeting sent successfully")

    async def _ensure_deepgram(self):
        """Connect to Deepgram and send settings"""
        try:
            if not DEEPGRAM_KEY:
                raise ValueError("DEEPGRAM_API_KEY not set in environment")
            
            # Validate API key format (should be a long alphanumeric string)
            if len(DEEPGRAM_KEY) < 20:
                print(f"[VoiceAI Warning] Deepgram API key looks too short (len={len(DEEPGRAM_KEY)})")
            
            print(f"[VoiceAI] Connecting to Deepgram... (key starts with: {DEEPGRAM_KEY[:8]}...)")
            
            # Test the API key first with a simple HTTP request
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    test_response = await client.get(
                        "https://api.deepgram.com/v1/projects",
                        headers={"Authorization": f"Token {DEEPGRAM_KEY}"},
                        timeout=5.0
                    )
                    if test_response.status_code == 200:
                        print("[VoiceAI] ✅ Deepgram API key validated successfully")
                    else:
                        print(f"[VoiceAI] ❌ Deepgram API key test failed: {test_response.status_code} {test_response.text}")
                        raise Exception(f"API key invalid: HTTP {test_response.status_code}")
            except Exception as e:
                print(f"[VoiceAI] ❌ Deepgram API key validation failed: {e}")
                raise
            
            self._dg_ws = await websockets.connect(
                DEEPGRAM_URL,
                extra_headers={"Authorization": f"Token {DEEPGRAM_KEY}"}
            )
            print("[VoiceAI] ✅ Deepgram WebSocket connection successful")
            
            # Send settings immediately after connecting
            await self._dg_ws.send(json.dumps({
                "type": "Settings",
                "container": "webm",
                "encoding": "opus",
                "sample_rate": 48000,
                "channels": 1,
                "interim_results": True,
                "endpointing": True,
                "vad_events": True,
                "punctuate": True,
                "no_speech_timeout": 20000
            }))
            print("[VoiceAI] ✅ Deepgram settings sent")
            
        except Exception as e:
            error_msg = f"Deepgram connection failed: {type(e).__name__} - {str(e)}"
            print(f"[VoiceAI Error] {error_msg}")  # Server-side logging
            
            # Add helpful context
            if "401" in str(e) or "unauthorized" in str(e).lower():
                error_msg += "\n\n💡 API key is invalid. Get a valid key from https://console.deepgram.com/"
            
            await self.send_json({"type": "error", "message": error_msg, "critical": True})
            await self.close()
            raise

    async def disconnect(self, code):
        self._closing = True
        for t in (getattr(self, "_dg_keepalive", None), getattr(self, "_relay_task", None)):
            if t: t.cancel()
        try:
            if self._dg_ws:
                await self._dg_ws.close()
        except:
            pass

    async def receive(self, text_data=None, bytes_data=None):
        # control messages (JSON) vs audio chunks (binary)
        if text_data:
            try:
                msg = json.loads(text_data)
            except:
                return

            # Handle test-say control message
            if msg.get("type") == "say" and msg.get("text"):
                await self._speak(str(msg["text"]))
                return

            if msg.get("type") == "end":
                # tell deepgram stream that input finished
                if self._dg_ws:
                    try:
                        await self._dg_ws.send(json.dumps({"type": "CloseStream"}))
                    except:
                        pass
                return

            # optional: handle UI commands (mute, unmute, etc.)
            return

        if bytes_data:
            if not self._dg_ws or self._dg_ws.closed:
                await self._ensure_deepgram()
            if len(bytes_data) < 32:
                return
            try:
                # Log first audio chunk received
                if self._first_audio:
                    print(f"[VoiceAI] Receiving audio from browser ({len(bytes_data)} bytes)")
                    print(f"[VoiceAI] First 32 bytes: {bytes_data[:32].hex()}")
                    self._first_audio = False
                
                await self._dg_ws.send(bytes_data)
            except websockets.ConnectionClosedError:
                await self._ensure_deepgram()
                try:
                    await self._dg_ws.send(bytes_data)
                except Exception as e:
                    print(f"[VoiceAI] Relay error after reconnect: {e}")
            except Exception as e:
                print(f"[VoiceAI] Relay error: {e}")

    async def _deepgram_heartbeat(self):
        """Send keepalive messages to prevent Deepgram timeout"""
        try:
            while not self._closing and self._dg_ws and not self._dg_ws.closed:
                await asyncio.sleep(8)
                try:
                    await self._dg_ws.send(json.dumps({"type": "KeepAlive"}))
                except:
                    break
        except asyncio.CancelledError:
            pass

    async def _deepgram_listener(self):
        """
        Listens to Deepgram messages; on final transcript -> get reply -> TTS -> stream audio to client
        """
        print("[VoiceAI] Deepgram listener started")
        try:
            async for raw in self._dg_ws:
                try:
                    data = json.loads(raw)
                except:
                    continue

                # Deepgram sends channel.alternatives[0].transcript
                if data.get("type") == "Results":
                    results = data.get("channel", {}).get("alternatives", [])
                    if not results:
                        continue
                    alt = results[0]
                    transcript = alt.get("transcript", "").strip()
                    is_final = data.get("is_final", False)

                    # forward interim transcript to UI (optional)
                    if transcript and not is_final:
                        await self.send_json({"type": "partial", "text": transcript})

                    if transcript and is_final:
                        print(f"[VoiceAI] Final transcript: {transcript}")
                        await self.send_json({"type": "transcript", "text": transcript})
                        # Use receptionist-style intent routing for replies
                        reply = route_intent(transcript)
                        print(f"[VoiceAI] Reply: {reply[:100]}...")
                        # Optional: polish with LLM if desired
                        # reply = await self._llm_short_reply(reply)
                        await self.send_json({"type": "assistant_text", "text": reply})
                        await self._speak(reply)
        except asyncio.CancelledError:
            return
        except Exception as e:
            if not self._closing:
                error_msg = f"Deepgram listener error: {type(e).__name__} - {str(e)}"
                print(f"[VoiceAI Error] {error_msg}")
                
                # Add helpful context for common Deepgram errors
                if "1011" in str(e) or "net0000" in str(e).lower():
                    error_msg += "\n\n💡 NET0000 after successful connection usually means:\n" \
                                 "• Audio format issue (browser sending wrong codec)\n" \
                                 "• WebSocket stream interrupted\n" \
                                 "• Try speaking again or refresh the page\n" \
                                 "• This is often temporary - not an API key issue"
                
                await self.send_json({"type": "error", "message": error_msg, "critical": True})

    async def _llm_short_reply(self, user_text: str) -> str:
        """
        Keep replies short and friendly (1–2 sentences).
        """
        prompt = (
            "You are a kind, efficient medical receptionist for NeuroMed AI. "
            "Reply in 1–2 short sentences. If user asks about overview, privacy, pricing, pilots, hours, or getting started, answer briefly and offer to email details."
            f"\nUser: {user_text}\nAssistant:"
        )
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                r = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                    json={
                        "model": OPENAI_MODEL,
                        "messages": [
                            {"role": "system", "content": "Be concise, warm, and helpful."},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.4,
                        "max_tokens": 90,
                    },
                )
            r.raise_for_status()
            choice = r.json()["choices"][0]["message"]["content"]
            return choice.strip()
        except Exception as e:
            error_msg = f"OpenAI error: {type(e).__name__} - {str(e)}"
            print(f"[VoiceAI Error] {error_msg}")
            await self.send_json({"type": "error", "message": error_msg, "critical": True})
            return "Sorry—can you say that one more time?"

    async def _speak(self, text: str):
        """
        Speak text via ElevenLabs TTS and stream audio chunks to client.
        """
        chunk_count = 0
        async for chunk in self._eleven_stream(text):
            await self.send(bytes_data=chunk)
            chunk_count += 1
        print(f"[VoiceAI] Sent {chunk_count} audio chunks to client")

    async def _eleven_stream(self, text: str):
        """
        Yields small audio chunks (MP3 stream) from ElevenLabs.
        """
        if not ELEVEN_KEY:
            error_msg = "ELEVENLABS_API_KEY not set in environment"
            print(f"[VoiceAI Error] {error_msg}")
            await self.send_json({"type": "error", "message": error_msg, "critical": True})
            return
            
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE}/stream"
        params = {"optimize_streaming_latency": 3, "output_format": "mp3_22050_32"}
        payload = {
            "text": text,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        }
        headers = {
            "xi-api-key": ELEVEN_KEY,
            "Content-Type": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", url, headers=headers, params=params, json=payload) as resp:
                    resp.raise_for_status()
                    async for chunk in resp.aiter_bytes():
                        if chunk:
                            yield chunk
        except Exception as e:
            error_msg = f"ElevenLabs TTS error: {type(e).__name__} - {str(e)}"
            print(f"[VoiceAI Error] {error_msg}")
            await self.send_json({"type": "error", "message": error_msg, "critical": True})
            return

    async def send_json(self, obj: dict):
        try:
            await super().send(text_data=json.dumps(obj))
        except:
            pass

