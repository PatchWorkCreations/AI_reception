"""
Quick diagnostic to check environment variables
"""
import os
from dotenv import load_dotenv

# Load .env if it exists
load_dotenv()

print("\n" + "="*60)
print("ENVIRONMENT VARIABLE DIAGNOSTIC")
print("="*60)

# Check all required API keys
keys_to_check = [
    "DEEPGRAM_API_KEY",
    "ELEVENLABS_API_KEY",
    "OPENAI_API_KEY",
    "ELEVENLABS_VOICE_ID",
]

for key in keys_to_check:
    value = os.getenv(key)
    if value:
        # Show first 10 chars and length for security
        masked = f"{value[:10]}...{value[-4:]}" if len(value) > 14 else f"{value[:8]}..."
        print(f"✅ {key:25} = {masked} (length: {len(value)})")
    else:
        print(f"❌ {key:25} = NOT SET")

print("\n" + "="*60)
print("RECOMMENDATIONS:")
print("="*60)

deepgram = os.getenv("DEEPGRAM_API_KEY")
if not deepgram:
    print("❌ DEEPGRAM_API_KEY is not set!")
    print("   Create a .env file with: DEEPGRAM_API_KEY=your_key_here")
elif len(deepgram) < 20:
    print("⚠️  DEEPGRAM_API_KEY looks too short (invalid key?)")
else:
    print("✅ DEEPGRAM_API_KEY looks valid")

elevenlabs = os.getenv("ELEVENLABS_API_KEY")
if not elevenlabs:
    print("❌ ELEVENLABS_API_KEY is not set!")
elif len(elevenlabs) < 20:
    print("⚠️  ELEVENLABS_API_KEY looks too short")
else:
    print("✅ ELEVENLABS_API_KEY looks valid")

voice_id = os.getenv("ELEVENLABS_VOICE_ID", "Bella")
if voice_id == "Bella" or len(voice_id) < 10:
    print("⚠️  ELEVENLABS_VOICE_ID should be a UUID (like: 21m00Tcm4TlvDq8ikWAM)")
    print("   Get voice IDs from: https://elevenlabs.io/app/voice-library")
else:
    print(f"✅ ELEVENLABS_VOICE_ID looks valid: {voice_id[:16]}...")

print("\n" + "="*60)
print("If keys are NOT SET, create a .env file in the myProject folder:")
print("="*60)
print("""
# .env file example
DEEPGRAM_API_KEY=your_deepgram_key_here
ELEVENLABS_API_KEY=your_elevenlabs_key_here
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM
OPENAI_API_KEY=your_openai_key_here
""")
print("="*60 + "\n")

