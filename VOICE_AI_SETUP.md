# NeuroMed AI - Simplified Voice Receptionist Setup

## Overview
This is a simplified voice AI receptionist that uses a single WebSocket route to:
- Stream browser microphone audio to Deepgram for real-time speech-to-text
- Send transcripts to OpenAI (gpt-4o-mini) for short, conversational replies
- Stream ElevenLabs TTS audio back to the browser

## Architecture
- **WebSocket Route**: `/ws/voice-ai/`
- **Frontend Page**: `/twilio/voice-ai/`
- **Consumer**: `VoiceAIConsumer` in `myApp/consumers.py`

## Required Environment Variables
Make sure these are set in your `.env` file or environment:

```bash
DEEPGRAM_API_KEY=your_deepgram_key
ELEVENLABS_API_KEY=your_elevenlabs_key
OPENAI_API_KEY=your_openai_key
ELEVENLABS_VOICE_ID=Bella  # optional, defaults to "Bella"
OPENAI_MODEL=gpt-4o-mini   # optional, defaults to "gpt-4o-mini"
```

## Installation
1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Run Django with ASGI support (using Daphne or Uvicorn):
   ```bash
   # Using Daphne (recommended for production)
   pip install daphne
   daphne -b 0.0.0.0 -p 8000 myProject.asgi:application
   
   # Or using Uvicorn (for development)
   pip install uvicorn
   uvicorn myProject.asgi:application --host 0.0.0.0 --port 8000
   ```

## Testing
1. Navigate to: `http://localhost:8000/twilio/voice-ai/`
2. Click "🎙️ Start Mic" to begin
3. Speak into your microphone
4. You'll see:
   - Interim transcripts (prefixed with "…")
   - Final transcripts (prefixed with "You:")
   - Bot responses (prefixed with "Bot:")
   - Audio will play back automatically

## Pipeline Flow
1. Browser captures mic audio (Opus @ 48kHz)
2. Audio streams to Django via WebSocket
3. Django relays to Deepgram for STT
4. On final transcript, Django calls OpenAI for a reply (1-2 sentences)
5. Django streams reply to ElevenLabs for TTS
6. MP3 audio chunks stream back to browser
7. Browser plays audio chunks sequentially

## Files Created/Modified
- `myApp/routing.py` - WebSocket URL configuration
- `myApp/consumers.py` - Main VoiceAIConsumer class
- `myApp/templates/talking_ai/speaking.html` - Frontend interface
- `myApp/views.py` - Added `voice_ai_speaking` view
- `myApp/urls.py` - Added `/voice-ai/` route
- `myProject/asgi.py` - Updated for Channels support
- `myProject/settings.py` - Added 'channels' to INSTALLED_APPS and ASGI_APPLICATION
- `requirements.txt` - Added channels==4.0.0

## Notes
- This implementation does NOT use Twilio or `bot_server.py`
- Audio format: Browser sends Opus, receives MP3
- Replies are intentionally short (1-2 sentences) for conversational feel
- All errors are sent to the browser as JSON messages

