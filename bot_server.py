import os, json, base64, asyncio, websockets, httpx, re, time, random
from dotenv import load_dotenv
import difflib
from functools import lru_cache

# =============================================================================
#  NeuroMed AI – Voice Receptionist (Twilio Media Streams <> Deepgram <> Eleven)
#  Enhanced: scripts/FAQs, richer intent routing, follow‑up templates, caching
# =============================================================================

load_dotenv()

# --------- ENV ---------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEEPGRAM_KEY   = os.getenv("DEEPGRAM_API_KEY")
ELEVEN_KEY     = os.getenv("ELEVENLABS_API_KEY")
HTTP_ORIGIN    = os.getenv("PUBLIC_HTTP_ORIGIN", "http://localhost:8000")
PORT           = int(os.getenv("PORT", "8080"))
CALENDLY_URL   = os.getenv("CALENDLY_URL", "https://calendly.com/neuromed/demo15")
VOICEMAIL_EMAIL= os.getenv("VOICEMAIL_EMAIL", "hello@neuromedai.org")

# Debug helpers
ECHO_BACK      = os.getenv("ECHO_BACK", "0") == "1"
ELEVEN_VOICE   = os.getenv("ELEVENLABS_VOICE_ID", "Bella")

# -----------------------------------------------------------------------------
#  Knowledge base (scripts, FAQs, follow‑ups). These are used BEFORE any HTTP
#  FAQ call, so you can ship a fully‑offline receptionist for common flows.
# -----------------------------------------------------------------------------
KB = {
  "greetings": {
    "phone": (
      "Hello, you’ve reached NeuroMed AI. We help families and care teams by "
      "turning complex medical records into clear, compassionate summaries—"
      "with an optional faith‑aware tone if requested. How may I help you today?"
    ),
    "voicemail": (
      "Hello, this is NeuroMed AI. We’re sorry we missed your call. Please leave "
      "your name, organization, and best contact, and we’ll reach out within 24 "
      "hours. You can also email us at hello@neuromedai.org. Thank you."
    ),
    "receptionist_lead": (
      "Hi, thank you for calling NeuroMed AI. This is Reception. May I have your name?"
    )
  },
  "explain": {
    "what_is": (
      "NeuroMed AI is a simple tool that helps patients, families, and caregivers "
      "understand medical notes in plain language. It saves staff time, reduces "
      "confusion, and improves trust. Families can also choose a faith‑aware mode "
      "for encouragement."
    ),
    "who_uses": (
      "We currently work with nursing homes, clinics, and universities who want to "
      "improve family communication and reduce staff burnout."
    ),
    "how_try": (
      "We usually start with a 30‑day pilot using 10–20 de‑identified cases—"
      "safe, low‑lift, and shows time saved and family response."
    ),
    "how_it_works_steps": (
      "It’s simple: 1) Upload medical documents. 2) Choose a tone—Plain, Caregiver, or "
      "Faith‑aware. 3) Generate a summary instantly. 4) Share it as print, email, or voice note. "
      "5) Optionally add encouragement—faith‑based or secular."
    )
  },
  "faq": {
    "hipaa": (
      "Yes. We use de‑identified files for pilots, and BAAs are available when working with PHI. "
      "Data is never shared externally."
    ),
    "replace_staff": (
      "No. It supports staff by reducing repeated explanations and boosting family clarity."
    ),
    "faith_only": (
      "NeuroMed AI is for everyone. Tone modes include plain, caregiver‑friendly, faith‑aware, and multilingual. "
      "Encouragement libraries include Christian, Muslim, Jewish, Hindu, Buddhist, and secular options."
    ),
    "cost": (
      "We’re in beta—no charge to try while we collect feedback. Long‑term pricing depends on whether you’re a family, "
      "nursing home, or hospital partner. Best next step is a quick demo to recommend the right package."
    ),
    "founder": (
      "NeuroMed AI was founded by Maria Gregory, inspired by her family’s experience with confusing medical updates "
      "since 2009. She built NeuroMed to bring clarity and comfort to families."
    ),
    "security": (
      "Privacy and compliance are core. We handle PHI with HIPAA, GDPR, and SOC 2 practices. We never sell your data."
    ),
    "partnership": (
      "We have several partnership tracks—60‑day hospital pilots, discharge planning pilots for nursing homes, and multi‑faith "
      "chaplaincy support. We’ll email the pilot brochure and book a call."
    )
  },
  "followups": {
    "sms": (
      "Thank you for reaching out to NeuroMed AI • We help families understand medical notes in plain language. "
      f"Would you like to book a 15‑minute demo? Here’s a link: {CALENDLY_URL}"
    ),
    "email_subject": "Thank you for your interest in NeuroMed AI",
    "email_body": (
      "Hi {name},\n\nIt was wonderful connecting today. As shared, NeuroMed AI helps care teams and families with "
      "clear, compassionate medical summaries. I’d love to schedule a 15‑minute demo to show how it works in practice.\n\n"
      f"You can pick a time here: {CALENDLY_URL}\n\n— NeuroMed AI"
    )
  },
  "contact": {
    "founder": "Maria Gregory",
    "email": "hello@neuromedai.org",
    "site":  "https://www.neuromedai.org"
  }
}

WHAT_IS_FALLBACK = (
  KB["explain"]["what_is"] + " " +
  "You can choose the tone—plain, caregiver‑friendly, faith‑aware, or clinical—so it meets the moment. "
  "It’s built to help families, caregivers, and clinics find clarity quickly without medical advice."
)

SYSTEM_PROMPT = (
  "You are the NeuroMed AI receptionist. Be warm, clear, concise, and human‑sounding.\n\n"
  "Mission:\n"
  "- NeuroMed AI helps families, caregivers, and clinics by turning medical files (discharge notes, labs, imaging) into plain‑language summaries, with optional styles: plain, caregiver‑friendly, faith‑aware (multi‑faith) or clinical.\n\n"
  "CRITICAL SCOPE LIMIT:\n"
  "- Callers are contacting about NeuroMed only. If asked anything unrelated to NeuroMed’s product, privacy, pricing, pilots, demo, faith‑aware mode, or support, politely say you can only answer questions about NeuroMed and offer to email more info.\n\n"
  "Guardrails:\n"
  "- Do not give medical advice or diagnose. Never promise unlisted features. Avoid speculative claims.\n"
  "- Privacy: reassure that data is kept private; do not discuss implementation details unless asked.\n"
  "- If asked for pricing/partners/pilots, acknowledge tiers/pilots and offer email follow‑up rather than quoting numbers.\n"
  "- If the user wants next steps, politely collect name + email.\n\n"
  "Style:\n"
  "- 1–2 sentences by default; longer only for the ‘what is NeuroMed AI’ explanation (~30–60s).\n"
  "- Use light natural pacing (brief pauses are added by the audio layer). Avoid filler endings.\n"
  "- If the user greets or backchannels, acknowledge briefly and continue.\n"
  "- If the user is unclear, ask a short clarifying question.\n\n"
  "Output Rules:\n"
  "- Stay on purpose: clarity, comfort, compassion. American English only.\n"
  "- Never end with generic closers unless explicitly requested."
)

TTS_WARM_LINES = [
    "Hello—", "Okay—", "Sure—",
    "Thanks, one moment—",
    "Hi! What would you like help with today—overview, or getting started?"
]

# =========================================================================
#  Utilities & audio helpers
# =========================================================================
@lru_cache(maxsize=256)
def _tts_cache_key(text: str) -> str:
    return text.strip().lower()

async def eleven_tts_stream_cached(text: str):
    key = _tts_cache_key(text)
    if hasattr(eleven_tts_stream_cached, "_mem") and key in eleven_tts_stream_cached._mem:
        for chunk in eleven_tts_stream_cached._mem[key]:
            yield chunk
        return

    chunks = []
    async for c in eleven_tts_stream(text):
        chunks.append(c)
        yield c

    # Cap per‑utterance cache size (~2MB)
    if len(b"".join(chunks)) < 2_000_000:
        eleven_tts_stream_cached._mem = getattr(eleven_tts_stream_cached, "_mem", {})
        eleven_tts_stream_cached._mem[key] = chunks

async def warm_eleven():
    try:
        for line in TTS_WARM_LINES:
            async for _ in eleven_tts_stream(line):
                break
    except Exception as e:
        print("ELEVEN WARM WARN ▶", repr(e))

# =========================================================================
#  Brand detection / text normalization
# =========================================================================
NEURO_CORE = "neuromed"
NEURO_AI   = "neuromed ai"

FILLERS_RE = re.compile(r"^(?:\s*(?:yeah|yep|uh|um|hmm|hello|hi|hey|okay|ok)[,.\s]*)+", re.I)
YES_RE = re.compile(r"^(?:yes|yeah|yep|sure|correct|right|ok(?:ay)?)\.?$", re.I)
NO_RE  = re.compile(r"^(?:no|nope|nah|negative|not\s+really)\.?$", re.I)

WEAK_CLOSERS = {
  "if you have more questions, feel free to ask.",
  "if you have any questions, feel free to ask.",
  "if you have any specific questions or need assistance, feel free to ask.",
  "let me know if you have other questions.",
  "how can i assist you further?",
  "feel free to ask!",
  "feel free to ask more questions.",
}

SENTENCE_END = re.compile(r'([.!?…]+)(\s+|$)')

# --- light normalizers ------------------------------------------------------
def _soft_norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ").strip()
    return s

def _collapse_ai_runs(s: str) -> str:
    s = re.sub(r"\b(a|e)\s*i\b", "ai", s)
    s = re.sub(r"\b(ay|eye)\b", "ai", s)
    return s

# --- token confusion / scoring ---------------------------------------------
def _confuse_token(tok: str) -> str:
    tok = tok.replace("nero", "neuro").replace("neero", "neuro").replace("nuro", "neuro")
    tok = tok.replace("neural", "neuro")
    for a,b in ("mediate","media"),("media","med"),("maid","med"),("met","med"),("mad","med"),("mid","med"),("medd","med"),("mayd","med"):
        tok = tok.replace(a,b)
    tok = tok.replace("miura", "mira")
    return tok

def _token_score(tok: str, targets: list[str]) -> int:
    t = _confuse_token(tok)
    for target in targets:
        if t == target: return 2
        r = difflib.SequenceMatcher(None, t, target).ratio()
        if r >= 0.8: return 2
        if r >= 0.6: return 1
    return 0

def _looks_like_brand(s: str) -> dict:
    s0 = _collapse_ai_runs(_soft_norm(s))
    toks = s0.split()
    neuro_like = {"neuro","nero","neero","nuro","neuromed","neuro-med","neural"}
    med_like   = {"med","met","mid","mad","mira","miura","miramed","mira-med","miuramed"}
    ai_like    = {"ai","a.i","a.i.","ay","eye"}

    idxs_neuro, idxs_med, idxs_ai = [], [], []
    for i, tok in enumerate(toks):
        if _token_score(tok, list(neuro_like)) >= 1: idxs_neuro.append(i)
        if _token_score(tok, list(med_like))   >= 1: idxs_med.append(i)
        if tok in ai_like or _token_score(tok, ["ai"]) >= 1: idxs_ai.append(i)

    has_brand = any(abs(i - j) <= 2 for i in idxs_neuro for j in idxs_med)
    has_ai    = has_brand and (bool(idxs_ai) or s0.endswith(" ai"))

    if not has_brand:
        sim_nm  = difflib.SequenceMatcher(None, s0, NEURO_CORE).ratio()
        sim_nma = difflib.SequenceMatcher(None, s0, NEURO_AI).ratio()
        if sim_nm >= 0.62 or sim_nma >= 0.58:
            has_brand = True
            has_ai = has_ai or (sim_nma > sim_nm)

    return {"is_brand": has_brand, "has_ai": has_ai}

_NEURO_ALIASES = [
    "neuro med", "neuromed", "neuro med ai", "neuro ai", "neuro media", "neuro mediate",
    "miura med", "mira med", "miramad", "miuramed", "miu rammed", "neuro mid", "neuro maid",
    "mid ai", "med ai", "my ai", "your ai", "media i", "mid a i"
]

FILL_FIXES = [
    (r"\bare you on security\b", "is it secure"),
    (r"\bis it secured\b", "is it secure"),
    (r"\bis (?:your|the)\s+data\s+(?:safe|secured?)\b", "is it secure"),
    (r"\bis it (?:hipa|hippa)\b", "is it hipaa"),
]

def _strip_fillers(s: str) -> str:
    return FILLERS_RE.sub("", (s or "").strip()).strip()

def _brandify_tail(s: str) -> str:
    tokens = s.lower().split()
    tail = " ".join(tokens[-3:]) if tokens else ""
    for alias in _NEURO_ALIASES:
        if difflib.SequenceMatcher(None, tail, alias).ratio() >= 0.7:
            return re.sub(r"(\b" + re.escape(tail) + r"\b)$", "NeuroMed AI", s, flags=re.I)
    return s

def domain_corrections(text: str) -> str:
    t = _strip_fillers(text or "")
    subs = [
        (r"\bmiu?ra?med\b", "NeuroMed"),
        (r"\bmira\s*med\b", "NeuroMed"),
        (r"\bmiura\s*med\b", "NeuroMed"),
        (r"\bneuro\s*med\b", "NeuroMed"),
        (r"\bneuro\s*mid\b", "NeuroMed"),
        (r"\bneuro\s*mediate\b", "NeuroMed AI"),
        (r"\bneuro\s*media(te)?\b", "NeuroMed AI"),
        (r"\bneuro\s*med\s*(?:ai|a\s*i|a[i1])\b", "NeuroMed AI"),
        (r"\b(?:mid|med)\s*a\s*i\b", "NeuroMed AI"),
        (r"\bmedia\s*i\b", "NeuroMed AI"),
        (r"\bmy\s*ai\b", "NeuroMed AI"),
        (r"\byour\s+(?:ai|app|system)\b", "NeuroMed AI"),
    ]
    for pat, repl in subs + FILL_FIXES:
        t = re.sub(pat, repl, t, flags=re.I)

    if re.search(r"\b(tell me more|more about)\b", t, flags=re.I) and re.search(r"\b(video|media)\b\.??$", t, flags=re.I):
        t = re.sub(r"\b(video|media)\b\.??$", "NeuroMed AI", t, flags=re.I)

    t = re.sub(r"\bE\.?A\.?\??$", "NeuroMed AI?", t, flags=re.I)
    t = re.sub(r"\bA\.?I\.?\??$", "NeuroMed AI?", t, flags=re.I)
    t = re.sub(r"\bEA\??$",      "NeuroMed AI?", t, flags=re.I)

    if re.search(r"\byour\s+(?:ai|app|system)?\s*\??$", t, flags=re.I):
        t = re.sub(r"\byour\s+(?:ai|app|system)?\s*\??$", "NeuroMed AI?", t, flags=re.I)

    t = _brandify_tail(t)
    return t

# =========================================================================
#  Intent classification
# =========================================================================
RX = lambda p: re.compile(p, re.I)
HIPAA_RE = RX(r"\b(hipaa|hippa|phi|privacy|private|confidential|secure(?:d|ly)?|security|gdpr|soc\s*2|compliance|compliant|safeguard(?:s)?)\b")
PRICING_RE = RX(r"\b(price|pricing|cost|how\s+(?:much|many)|rate|rates|quote|fees?|charges?|billing|estimate|plan(?:s)?|tier(?:s)?)\b")
PILOT_RE   = RX(r"\b(pilot|trial|evaluate|evaluation|proof\s+of\s+concept|poc|demo|walkthrough|sandbox|kick\s*the\s+tires)\b")
FAITH_RE   = RX(r"\b(faith|bible|scripture|psalm|verse|christ|jesus|prayer|chaplain|spiritual|religio(?:us|n)|encourag(?:e|ement))\b")
LANG_RE    = RX(r"\b(multilingual|multi[-\s]?language|bilingual|translate|translation|spanish|tagalog|filipino|language\s+support|foreign\s+language)\b")
SCHED_RE   = RX(r"\b(schedule|book|set\s*(?:up)?\s*(?:a\s*)?(?:call|meeting|appt|appointment)|calendar|availability|reschedul(?:e|ing))\b")
HUMAN_RE   = RX(r"\b(human|agent|representative|somebody|someone)\b|\b(speak|talk)\s+to\s+(a|the)\s+(human|person|agent)\b")
INTEG_RE   = RX(r"\b(ehr|emr|hl7|fhir|api|integrat(?:e|ion)|sso|okta|azure\s+ad|oauth|epic|cerner|meditech|athenahealth|allscripts)\b")
SUPPORT_RE = RX(r"\b(help|support|issue|error|bug|broken|can'?t\s+(?:login|log\s*in|sign\s*in)|password\s+reset|timeout|down|glitch)\b")
ADDRESS_RE = RX(r"\b(address|where\s+are\s+you|location|office|hq|headquarters)\b")
HOURS_RE   = RX(r"\b(hours?|open|close|availability|available\s+when)\b")
FOUNDER_RE = RX(r"\b(founder|who\s+is\s+behind|who\s+made\s+this|who\s+created)\b")
SECURITY_RE= RX(r"\b(security|secure|safety|safe|hipaa|gdpr|soc\s*2|baa|phi|compliance)\b")
INTERFAITH_RE = RX(r"\b(christian|muslim|jewish|hindu|buddhist|secular|multi\s*faith)\b")
HOWWORKS_RE= RX(r"\b(how\s+does\s+it\s+work|how\s+it\s+works|steps?|process)\b")
PARTNER_RE = RX(r"\b(partner(ship)?|collaborat(e|ion)|beta\s*test|university|hospital|nursing\s*home|chaplaincy)\b")
VOICEMAIL_RE= RX(r"\b(voicemail|leave\s+a\s+message)\b")

WHATIS_Q = RX(r"\b(what\s+is|what'?s|tell\s+me\s+more|how\s+does\s+it\s+work)\b")

INTENTS = [
  "greeting","what_is","hipaa","pricing","faith_mode","pilot","hours","address",
  "schedule_contact","founder","security","interfaith","how_it_works","partnership",
  "multilingual","scheduling","human","integration","support","voicemail","other"
]

def classify_intent_by_keywords(s: str) -> str:
    txt = (s or "")
    if HIPAA_RE.search(txt):   return "hipaa"
    if PRICING_RE.search(txt): return "pricing"
    if FAITH_RE.search(txt):   return "faith_mode"
    if PILOT_RE.search(txt):   return "pilot"
    if SCHED_RE.search(txt):   return "scheduling"
    if LANG_RE.search(txt):    return "multilingual"
    if HUMAN_RE.search(txt):   return "human"
    if INTEG_RE.search(txt):   return "integration"
    if SUPPORT_RE.search(txt): return "support"
    if ADDRESS_RE.search(txt): return "address"
    if HOURS_RE.search(txt):   return "hours"
    if FOUNDER_RE.search(txt): return "founder"
    if SECURITY_RE.search(txt):return "security"
    if INTERFAITH_RE.search(txt):return "interfaith"
    if HOWWORKS_RE.search(txt):return "how_it_works"
    if PARTNER_RE.search(txt): return "partnership"
    if VOICEMAIL_RE.search(txt):return "voicemail"
    if WHATIS_Q.search(txt) and _looks_like_brand(txt)["is_brand"]: return "what_is"
    return "other"

# =========================================================================
#  Byte helpers & μ‑law silence
# =========================================================================

def b64_to_bytes(s: str) -> bytes: return base64.b64decode(s)

def bytes_to_b64(b: bytes) -> str: return base64.b64encode(b).decode()

def _ulaw_silence(ms: int) -> bytes: return b"\xff" * int(8 * ms)  # 8kHz, 8 samples/ms

async def send_silence(ws_send_pcm, ms: int):
    if ms <= 0: return
    try:
        await ws_send_pcm(_ulaw_silence(ms))
    except (websockets.exceptions.ConnectionClosedOK, websockets.exceptions.ConnectionClosedError):
        pass

# =========================================================================
#  Local KB lookup and HTTP FAQ
# =========================================================================

def kb_lookup(query: str) -> str:
    q = (query or "").lower()
    # Simple switches
    if "hipaa" in q or "privacy" in q or "secure" in q: return KB["faq"]["hipaa"]
    if "founder" in q or "who is" in q: return KB["faq"]["founder"]
    if "cost" in q or "price" in q: return KB["faq"]["cost"]
    if "partnership" in q or "partner" in q: return KB["faq"]["partnership"]
    if "how it works" in q or "steps" in q: return KB["explain"]["how_it_works_steps"]
    return ""

async def call_faq(q: str) -> str:
    # 1) Try local KB
    ans = kb_lookup(q)
    if ans: return ans

    # 2) Try your Django FAQ endpoint (optional)
    try:
        timeout = httpx.Timeout(connect=1.0, read=2.5, write=2.5, pool=2.5)
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(f"{HTTP_ORIGIN}/api/faq", params={"q": q})
            r.raise_for_status()
            return r.json().get("answer", "")
    except Exception as e:
        print("FAQ WARN ▶", repr(e))
        return ""

# =========================================================================
#  OpenAI streaming (sentences)
# =========================================================================
async def stream_llm_sentences(messages: list[dict]):
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
                if not line or not line.startswith("data:"): continue
                data = line[5:].strip()
                if data == "[DONE]": break
                try:
                    obj = json.loads(data)
                except Exception:
                    continue
                delta = ((obj.get("choices") or [{}])[0].get("delta") or {}).get("content") or ""
                if not delta: continue
                buf += delta
                while True:
                    m = SENTENCE_END.search(buf)
                    if not m: break
                    end_idx = m.end()
                    sent = buf[:end_idx].strip()
                    buf = buf[end_idx:]
                    if sent: yield sent
    tail = buf.strip()
    if tail: yield tail

# =========================================================================
#  Light slot extraction (OpenAI tools)
# =========================================================================
async def extract_intent_slots(utterance: str) -> dict:
    tool_schema = {
        "type": "function",
        "function": {
            "name": "set_intent",
            "description": "Set the user's intent and any available contact info.",
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {"type": "string", "enum": INTENTS},
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

# =========================================================================
#  Planner (rules first, then generative)
# =========================================================================
GREETING_RE = re.compile(r"^(hi|hello|hey)[,!\.\s]*(?:i'?m|this is)?\b", re.I)
ASSIST_RE   = re.compile(r"\b(how can i (?:help|assist)|let me know|feel free to ask)\b", re.I)

def should_skip_sentence(s: str) -> bool:
    return (s or "").strip().lower() in WEAK_CLOSERS

def should_drop_assistant_line(s: str, *, greeted_already: bool) -> bool:
    s0 = (s or "").strip()
    if not s0: return True
    low = s0.lower()
    if low in WEAK_CLOSERS: return True
    has_substance = bool(re.search(r"\b(neuromed|medical|files?|summary|pricing|pilot|demo|email|upload|church|caregivers?)\b", low))
    is_greetingish = bool(GREETING_RE.search(s0))
    is_assistish   = bool(ASSIST_RE.search(s0))
    if greeted_already and (is_greetingish or is_assistish) and not has_substance:
        return True
    return False

def should_block_contact_line(s: str, *, allow_contact_request: bool) -> bool:
    if allow_contact_request: return False
    return bool(re.search(r"\b(email|best email|contact|reach you)\b", (s or ""), re.I))

def extract_contact_inline(txt: str) -> dict:
    out = {}
    m = re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", txt, re.I)
    if m: out["email"] = m.group(0)
    m2 = re.search(r"\bmy name is ([A-Za-z][A-Za-z\-']+\s+[A-Za-z][A-Za-z\-']+)\b", txt, re.I)
    if m2: out["name"] = m2.group(1).strip()
    return out

FEWSHOTS = [
    {"role": "user", "content": "Can you tell me more about miramad ai?"},
    {"role": "assistant", "content": KB["explain"]["what_is"]},
    {"role": "user", "content": "is this hipaa compliant? how private is it"},
    {"role": "assistant", "content": KB["faq"]["hipaa"] + " I can email a 1‑pager if you’d like."},
    {"role": "user", "content": "how much does it cost"},
    {"role": "assistant", "content": KB["faq"]["cost"] + " What’s the best email to use?"},
    {"role": "user", "content": "do you have a faith based option?"},
    {"role": "assistant", "content": KB["faq"]["faith_only"]},
    {"role": "user", "content": "ok, how do we start?"},
    {"role": "assistant", "content": "Great—can I have your name and best email? We’ll send next steps and a short demo."},
]

async def plan_reply(history: list[dict], state: dict | None = None) -> list[str]:
    last = (history[-1]["content"] if history else "").strip()
    slots = await extract_intent_slots(last)
    intent = slots.get("intent","other")

    if intent == "greeting":
        return ["Hi! Would you like an overview, pricing, or to try a quick demo?"]
    if intent == "hipaa":
        return [KB["faq"]["hipaa"]]
    if intent == "pricing":
        return [KB["faq"]["cost"] + " I can send options—what’s the best email to use?"]
    if intent == "faith_mode":
        return [KB["faq"]["faith_only"]]
    if intent == "pilot":
        return [KB["explain"]["how_try"] + " I can email details—what email should we use?"]
    if intent == "hours":
        return ["We’re available weekdays; if you share your time zone, I can suggest a slot."]
    if intent == "address":
        return ["We operate online and partner with facilities; if you share your city, I can route you."]
    if intent == "founder":
        return [KB["faq"]["founder"]]
    if intent == "security":
        return [KB["faq"]["security"] + " I can send a one‑page security overview—should I email it?"]
    if intent == "interfaith":
        return [KB["faq"]["faith_only"]]
    if intent == "how_it_works":
        return [KB["explain"]["how_it_works_steps"]]
    if intent == "partnership":
        return [KB["faq"]["partnership"] + " May I take your email to send the brochure?"]
    if intent == "what_is":
        return [s for s in re.split(r'(?<=[.!?])\s+', WHAT_IS_FALLBACK) if s and not should_skip_sentence(s)]
    if intent == "schedule_contact":
        name = slots.get("name"); email = slots.get("email")
        if name and email: return [f"Thanks, {name}. We’ll email next steps to {email} shortly."]
        return ["Great—could I have your name and best email so we can send next steps?"]
    return ["I can help with NeuroMed only—overview, privacy, pricing, faith‑aware mode, or getting started. Which would you like?"]

# =========================================================================
#  Deepgram ASR
# =========================================================================
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
                    "NeuroMed", "NeuroMed AI", "HIPAA", "privacy", "secure", "security", "GDPR", "SOC 2",
                    "pricing", "pilot", "demo", "faith", "schedule", "Spanish", "Tagalog",
                    "founder", "partnership", "chaplaincy", "university",
                    "neuro med", "neuro mid", "neuro mediate", "mira med", "miuramed",
                    "med ai", "mid ai", "a i", "ay", "eye", "media i"
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
                    except (websockets.exceptions.ConnectionClosedOK, websockets.exceptions.ConnectionClosedError):
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
                if isinstance(obj, dict) and obj.get("type") in {"Metadata","Warning","Error","Close","UtteranceEnd","SpeechStarted","SpeechFinished"}:
                    continue
                alts = []
                is_final = bool(obj.get("is_final"))
                chan = obj.get("channel")
                if isinstance(chan, dict): alts = chan.get("alternatives") or []
                elif isinstance(chan, list) and chan:
                    first_chan = chan[0] or {}
                    if isinstance(first_chan, dict): alts = first_chan.get("alternatives") or []
                elif "alternatives" in obj and isinstance(obj["alternatives"], list):
                    alts = obj["alternatives"]
                if not alts: continue
                txt = (alts[0].get("transcript") or "").strip()
                if not txt: continue

                now = time.time()
                if not (is_final or (now - last_emit) > 0.2): continue
                if not is_final and txt == last_sent: continue
                print(f"ASR{'(final)' if is_final else ''} ▶ {txt}")
                yield txt, is_final
                last_emit = now; last_sent = txt
        finally:
            await feed_task

# =========================================================================
#  ElevenLabs (μ‑law 8k streaming)
# =========================================================================
async def eleven_tts_stream(text: str):
    if not ELEVEN_KEY:
        raise RuntimeError("ELEVENLABS_API_KEY missing.")

    async def _do():
        url = (
            f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE}/stream"
            "?optimize_streaming_latency=3&output_format=ulaw_8000"
        )
        headers = {"xi-api-key": ELEVEN_KEY, "Content-Type": "application/json", "Accept": "*/*"}
        payload = {"text": text, "model_id": "eleven_multilingual_v2", "voice_settings": {"stability": 0.5, "similarity_boost": 0.8}}
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as r:
                if r.is_error:
                    body = await r.aread(); print("ELEVENLABS HTTP ERROR ▶", r.status_code, body[:200]); return
                async for chunk in r.aiter_bytes():
                    if chunk: yield chunk

    got_audio = False
    async for c in _do():
        got_audio = True; yield c
    if not got_audio:
        print("ELEVENLABS WARN ▶ no audio; quick retry")
        async for c in _do(): yield c

# =========================================================================
#  Twilio Media Streams WebSocket server
# =========================================================================
async def handle_twilio(ws):
    if ws.path not in ("/ws/twilio",):
        await ws.close(code=1008, reason="Unexpected path"); return

    print("WS ▶ Twilio connected, path:", ws.path)
    print("WS ▶ protocol negotiated:", ws.subprotocol)

    inbound_q: asyncio.Queue[bytes] = asyncio.Queue()
    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    state = {
        "greeted": False,
        "speaking": False,
        "last_reply": "",
        "last_reply_ts": 0.0,
        "barge_in_enabled": False,
        "long_answer_until": 0.0,
        "last_ack_ts": 0.0,
        "local_no_barge_until": 0.0,
        "allow_contact_request": False,
        "contact": {"name": None, "email": None},
        "last_intent": "other",
        "last_intent_ts": 0.0,
        "last_question": None,
        "TOPIC_STICKY_SECS": 10.0,
    }

    conn_open = True
    stream_sid = None
    speak_task: asyncio.Task | None = None

    async def arm_barge_in_after(seconds: float):
        await asyncio.sleep(seconds)
        state["barge_in_enabled"] = True
        print(f"BARGE‑IN ▶ enabled after {seconds:.1f}s")

    media_stats = {"frames": 0, "bytes": 0}
    first_media = asyncio.Event()

    async def pcm_iter():
        while True:
            b = await inbound_q.get()
            if b is None: break
            yield b

    async def send_pcm(pcm: bytes) -> bool:
        nonlocal conn_open, stream_sid
        if not conn_open or not stream_sid: return False
        try:
            await ws.send(json.dumps({"event": "media", "streamSid": stream_sid, "media": {"payload": bytes_to_b64(pcm)}}))
            return True
        except (websockets.exceptions.ConnectionClosedOK, websockets.exceptions.ConnectionClosedError):
            conn_open = False; return False

    async def speak(text: str):
        nonlocal stream_sid, speak_task, conn_open
        text = (text or "").strip()
        if not text: return

        if not stream_sid:
            for _ in range(200):
                if stream_sid or not conn_open: break
                await asyncio.sleep(0.005)
            if not stream_sid or not conn_open: return

        now = time.time()
        if text == state["last_reply"].strip() and (now - state["last_reply_ts"]) < 2.5:
            print("TTS ▶ duplicate suppressed"); return

        if speak_task and not speak_task.done():
            speak_task.cancel()
            try: await speak_task
            except asyncio.CancelledError: pass

        async def _run():
            state["speaking"] = True
            any_chunk = False; cancelled = False
            prev_barge = state.get("barge_in_enabled", False)
            state["local_no_barge_until"] = time.time() + 1.2
            try:
                if not conn_open or not stream_sid: return
                pre = 60 if len(text.split()) > 1 else 30
                pre += int(random.uniform(-15, 20))
                await send_silence(send_pcm, max(pre, 20))
                async for pcm in eleven_tts_stream_cached(text):
                    any_chunk = True
                    ok = await send_pcm(pcm)
                    if not ok: return
                if not conn_open: return
                post = 60 + int(random.uniform(-15, 25))
                await send_silence(send_pcm, max(post, 20))
            except asyncio.CancelledError:
                cancelled = True; print("TTS ▶ canceled (barge‑in)"); raise
            finally:
                state["barge_in_enabled"] = prev_barge
                state["speaking"] = False
                if any_chunk:
                    state["last_reply"] = text; state["last_reply_ts"] = time.time()
                    print(f"TTS ▶ {text[:60]}...")
                else:
                    if not cancelled: print("TTS ▶ no audio produced (check TTS provider)")

        speak_task = asyncio.create_task(_run())

    async def speak_lines(lines: list[str], *, greeted_already: bool, allow_contact_request: bool) -> int:
        spoken = 0
        for sentence in lines:
            if should_skip_sentence(sentence): continue
            if should_block_contact_line(sentence, allow_contact_request=allow_contact_request): continue
            if should_drop_assistant_line(sentence, greeted_already=greeted_already): continue
            await speak(sentence); spoken += 1
        if spoken == 0:
            await speak("Okay—would you like an overview, pricing, or getting started?"); spoken = 1
        return spoken

    # ------------------------------- BRAIN ----------------------------------
    async def brain():
        if not DEEPGRAM_KEY: return
        try:
            await asyncio.wait_for(first_media.wait(), timeout=20.0)
        except asyncio.TimeoutError:
            print("ASR ▶ no media within 20s; skipping ASR session"); return

        last_user_sent = ""; FINAL_DEBOUNCE_SECS = 0.35
        pending_final = None; pending_ts = 0.0

        async def process_final_utterance(text: str):
            try:
                fixed = domain_corrections(text or "")
                if not fixed: return
                print("ROUTER ▶", fixed)

                nowt = time.time()
                kw_intent = classify_intent_by_keywords(fixed)

                if kw_intent == "other" and (state.get("last_intent") not in (None, "other")):
                    if len(fixed.split()) <= 3 and (nowt - state.get("last_intent_ts", 0)) < state.get("TOPIC_STICKY_SECS", 10.0):
                        kw_intent = state["last_intent"]

                yn = "yes" if YES_RE.match(fixed.strip()) else ("no" if NO_RE.match(fixed.strip()) else None)
                if yn and state.get("last_question") == "yn" and state.get("last_intent") not in (None, "other"):
                    kw_intent = state["last_intent"]

                if kw_intent != "other":
                    state["last_intent"] = kw_intent; state["last_intent_ts"] = nowt

                    if kw_intent == "hipaa":
                        state["last_question"] = "open"
                        await speak_lines([KB["faq"]["hipaa"]], greeted_already=state["greeted"], allow_contact_request=state["allow_contact_request"]); return
                    if kw_intent == "pricing":
                        state["last_question"] = "yn"
                        await speak_lines([KB["faq"]["cost"] + " I can send a quick overview—what’s the best email to use?"], greeted_already=state["greeted"], allow_contact_request=True); return
                    if kw_intent == "faith_mode":
                        state["last_question"] = "open"
                        await speak_lines([KB["faq"]["faith_only"]], greeted_already=state["greeted"], allow_contact_request=state["allow_contact_request"]); return
                    if kw_intent == "pilot":
                        state["last_question"] = "yn"
                        await speak_lines([KB["explain"]["how_try"] + " I can email details—what email should we use?"], greeted_already=state["greeted"], allow_contact_request=True); return
                    if kw_intent == "multilingual":
                        state["last_question"] = "open"
                        await speak_lines(["We can provide summaries in multiple languages—for families who need clarity in their heart language."], greeted_already=state["greeted"], allow_contact_request=state["allow_contact_request"]); return
                    if kw_intent == "scheduling":
                        state["last_question"] = "open"
                        await speak_lines([f"We’re available on weekdays. You can also pick a time here: {CALENDLY_URL}"], greeted_already=state["greeted"], allow_contact_request=True); return
                    if kw_intent == "human":
                        state["last_question"] = "open"
                        await speak_lines(["I can collect your name and email so a teammate follows up shortly."], greeted_already=state["greeted"], allow_contact_request=True); return
                    if kw_intent == "integration":
                        state["last_question"] = "open"
                        await speak_lines(["We integrate with clinical systems via standards like HL7 and FHIR."], greeted_already=state["greeted"], allow_contact_request=state["allow_contact_request"]); return
                    if kw_intent == "support":
                        state["last_question"] = "open"
                        await speak_lines(["I can help get you support—can I have your best email so we can follow up?"], greeted_already=state["greeted"], allow_contact_request=True); return
                    if kw_intent == "address":
                        state["last_question"] = "open"
                        await speak_lines(["We operate online and partner with facilities; if you share your city, I can route you."], greeted_already=state["greeted"], allow_contact_request=state["allow_contact_request"]); return
                    if kw_intent == "hours":
                        state["last_question"] = "open"
                        await speak_lines(["We’re available weekdays."], greeted_already=state["greeted"], allow_contact_request=state["allow_contact_request"]); return
                    if kw_intent == "founder":
                        state["last_question"] = "open"
                        await speak_lines([KB["faq"]["founder"]], greeted_already=state["greeted"], allow_contact_request=state["allow_contact_request"]); return
                    if kw_intent == "security":
                        state["last_question"] = "open"
                        await speak_lines([KB["faq"]["security"] + " I can send a 1‑page overview by email."], greeted_already=state["greeted"], allow_contact_request=True); return
                    if kw_intent == "interfaith":
                        state["last_question"] = "open"
                        await speak_lines([KB["faq"]["faith_only"]], greeted_already=state["greeted"], allow_contact_request=state["allow_contact_request"]); return
                    if kw_intent == "how_it_works":
                        state["last_question"] = "open"
                        await speak_lines([KB["explain"]["how_it_works_steps"]], greeted_already=state["greeted"], allow_contact_request=state["allow_contact_request"]); return
                    if kw_intent == "partnership":
                        state["last_question"] = "yn"
                        await speak_lines([KB["faq"]["partnership"] + " May I take your email to send the brochure?"], greeted_already=state["greeted"], allow_contact_request=True); return
                    if kw_intent == "voicemail":
                        state["last_question"] = None
                        await speak_lines([KB["greetings"]["voicemail"]], greeted_already=True, allow_contact_request=True); return

                brand = _looks_like_brand(fixed)
                if WHATIS_Q.search(fixed) and brand["is_brand"]:
                    asyncio.create_task(speak("Sure—"))
                    answer = await call_faq("what_is_neuromed") or WHAT_IS_FALLBACK
                    state["long_answer_until"] = time.time() + 2.0
                    sentences = [s for s in re.split(r'(?<=[.!?])\s+', answer.strip()) if s]
                    await speak_lines(sentences, greeted_already=state["greeted"], allow_contact_request=state["allow_contact_request"]); return

                if WHATIS_Q.search(fixed):
                    asyncio.create_task(speak("Sure—"))
                    answer = await call_faq("what_is_neuromed") or WHAT_IS_FALLBACK
                    state["long_answer_until"] = time.time() + 2.0
                    sentences = [s for s in re.split(r'(?<=[.!?])\s+', answer.strip()) if s]
                    await speak_lines(sentences, greeted_already=state["greeted"], allow_contact_request=state["allow_contact_request"]); return

                info = extract_contact_inline(fixed)
                if "email" in info: state["contact"]["email"] = info["email"]
                if "name" in info and not state["contact"]["name"]: state["contact"]["name"] = info["name"]
                if re.search(r"\b(email|send|share)\b.*\b(info|details|pricing|pilot|follow\s*up)\b", fixed, re.I):
                    state["allow_contact_request"] = True

                words = fixed.split()
                in_long_window = nowt < state["long_answer_until"]
                should_barge = len(words) >= (5 if in_long_window else 3)
                if should_barge and state.get("barge_in_enabled") and state["speaking"] and speak_task and not speak_task.done():
                    speak_task.cancel();
                    try: await speak_task
                    except asyncio.CancelledError: pass

                history.append({"role": "user", "content": fixed})
                if len(words) <= 6:
                    asyncio.create_task(asyncio.sleep(0))
                    asyncio.create_task(speak("Okay—"))

                state["long_answer_until"] = time.time() + 2.0
                try:
                    plan = await plan_reply(history, state=state)
                except Exception as e:
                    print("PLAN ERROR ▶", repr(e)); plan = []

                if plan:
                    await speak_lines(plan, greeted_already=state["greeted"], allow_contact_request=state["allow_contact_request"])
                else:
                    gen_messages = [{"role": "system", "content": SYSTEM_PROMPT}, *FEWSHOTS, *history]
                    spoke_count = 0
                    async def nudge():
                        await asyncio.sleep(1.2)
                        if spoke_count == 0: await speak("Okay—")
                    nudge_task = asyncio.create_task(nudge())
                    try:
                        async for sentence in stream_llm_sentences(gen_messages):
                            said = await speak_lines([sentence], greeted_already=state["greeted"], allow_contact_request=state["allow_contact_request"])
                            if said > 0:
                                spoke_count += said
                                if not nudge_task.done(): nudge_task.cancel()
                    except Exception as e:
                        print("GEN ERROR ▶", repr(e))
                    if spoke_count == 0:
                        await speak_lines(["Okay—would you like an overview, pricing, or to get started with a quick demo?"], greeted_already=state["greeted"], allow_contact_request=state["allow_contact_request"])
            except Exception as e:
                print("PROCESS ERROR ▶", repr(e))
                await speak("Okay—would you like an overview, pricing, or a quick demo?")

        async for utter, is_final in deepgram_stream(pcm_iter):
            if not utter: continue
            utter = domain_corrections(utter)
            if utter == last_user_sent: continue
            last_user_sent = utter

            now = time.time()
            if pending_final and (now - pending_ts) > FINAL_DEBOUNCE_SECS:
                await process_final_utterance(pending_final); pending_final = None

            if is_final:
                if pending_final and (now - pending_ts) <= FINAL_DEBOUNCE_SECS and not re.search(r"[.!?…]\s*$", pending_final):
                    pending_final = (pending_final + " " + utter).strip(); pending_ts = now
                else:
                    pending_final = utter.strip(); pending_ts = now
                if re.search(r"[.!?…]\s*$", pending_final) or len(pending_final.split()) >= 8:
                    await process_final_utterance(pending_final); pending_final = None

        if pending_final: await process_final_utterance(pending_final)

    brain_task = asyncio.create_task(brain())

    try:
        async for raw in ws:
            data = json.loads(raw); ev = data.get("event")
            if ev == "start":
                start_info = data.get("start", {}) or {}
                stream_sid = start_info.get("streamSid"); call_sid = start_info.get("callSid")
                print(f"WS ▶ start streamSid={stream_sid} callSid={call_sid}")
                state["barge_in_enabled"] = False
                if not state["greeted"]:
                    state["greeted"] = True
                    asyncio.create_task(speak(KB["greetings"]["phone"]))
                    asyncio.create_task(arm_barge_in_after(1.5))
            elif ev == "media":
                payload_b64 = data["media"]["payload"]; buf = b64_to_bytes(payload_b64)
                media_stats["frames"] += 1; media_stats["bytes"]  += len(buf)
                if media_stats["frames"] == 1:
                    first_media.set(); print("WS ▶ first media frame received")
                    asyncio.create_task(arm_barge_in_after(0.6))
                if media_stats["frames"] % 25 == 0:
                    print(f"WS ▶ media frames={media_stats['frames']} bytes={media_stats['bytes']}")
                await inbound_q.put(buf)
                if ECHO_BACK and stream_sid and conn_open: await send_pcm(buf)
            elif ev == "stop":
                print("WS ▶ stop"); conn_open = False
                if speak_task and not speak_task.done():
                    speak_task.cancel()
                    try: await speak_task
                    except asyncio.CancelledError: pass
                break
    except Exception as e:
        print("WS ERR ▶", e)
    finally:
        conn_open = False
        await inbound_q.put(None)
        if speak_task and not speak_task.done():
            speak_task.cancel()
            try: await speak_task
            except asyncio.CancelledError: pass
        await brain_task
        print("WS ▶ closed")

# =========================================================================
#  Server main
# =========================================================================
async def main():
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
        PORT,
        max_size=2**20,
        ping_interval=20,
        ping_timeout=60,
        subprotocols=["audio.twilio.com"],
    ):
        print(f"WS bot listening on ws://0.0.0.0:{PORT}")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(warm_eleven())
    asyncio.run(main())
