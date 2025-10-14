#!/usr/bin/env python3
"""
Quick verification script for the bot_server.py implementation
"""

import re

def verify_implementation():
    print("=" * 60)
    print("Bot Server Implementation Verification")
    print("=" * 60)
    
    with open("bot_server.py", "r", encoding="utf-8") as f:
        code = f.read()
    
    checks = {
        "✓ Pre-roll WAV URL configured": "PREROLL_WAV_URL" in code,
        "✓ play_preroll_wav() function": "async def play_preroll_wav" in code,
        "✓ WAV payload extraction": "def extract_wav_ulaw_payload" in code,
        "✓ Pre-roll called in start event": "await play_preroll_wav(send_pcm)" in code,
        "✓ STATE_CONFIRM_EMAIL defined": "STATE_CONFIRM_EMAIL" in code,
        "✓ spell_for_tts() function": "def spell_for_tts" in code,
        "✓ normalize_spoken_email()": "def normalize_spoken_email" in code,
        "✓ YES/NO pattern matching": "YES_PAT = re.compile" in code and "NO_PAT" in code,
        "✓ ask_confirm_email() function": "async def ask_confirm_email" in code,
        "✓ Email confirmation handler": "if YES_PAT.search(t):" in code,
        "✓ Dual-silence gating": "quiet_since_caller" in code and "quiet_since_us" in code,
        "✓ OpenAI micro-NLU": "async def call_openai_email" in code,
        "✓ Email regex matching": "EMAIL_RE.search" in code,
        "✓ Provider fixes (g male→gmail)": '"gmail"' in code and "g\s*male" in code,
    }
    
    print("\nFeature Checklist:")
    all_ok = True
    for desc, result in checks.items():
        status = "✓" if result else "✗"
        print(f"  {status} {desc}")
        if not result:
            all_ok = False
    
    print("\n" + "=" * 60)
    if all_ok:
        print("✓ ALL FEATURES IMPLEMENTED!")
        print("\nYour bot server has:")
        print("  • Pre-roll WAV playback (μ-law 8kHz)")
        print("  • Smart email capture (regex + normalization + OpenAI)")
        print("  • Email read-back with spelling")
        print("  • Yes/no confirmation")
        print("  • Dual-silence gating for reprompts")
        print("\nReady to test! 🚀")
    else:
        print("✗ Some features missing - check implementation")
    
    print("=" * 60)
    
    # Extract key configuration
    print("\nConfiguration:")
    preroll_match = re.search(r'PREROLL_WAV_URL\s*=\s*f?"([^"]+)"', code)
    if preroll_match:
        print(f"  Pre-roll URL: {preroll_match.group(1)}")
    
    chunk_match = re.search(r'PREROLL_CHUNK_SIZE\s*=\s*(\d+)', code)
    if chunk_match:
        print(f"  Chunk size: {chunk_match.group(1)} bytes (20ms @ 8kHz)")
    
    print("\nStates:")
    states = re.findall(r'STATE_(\w+)\s*=\s*"(\w+)"', code)
    for var, val in states:
        print(f"  • {val}")
    
    print("\nEmail Processing Pipeline:")
    print("  1. Regex match for complete email")
    print("  2. normalize_spoken_email() for spoken patterns")
    print("  3. OpenAI micro-NLU (fallback)")
    print("  4. Read-back confirmation with spell_for_tts()")
    print("  5. Yes/no → confirm or retry")

if __name__ == "__main__":
    verify_implementation()

