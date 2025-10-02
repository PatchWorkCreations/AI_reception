import os, json, base64, asyncio, websockets, httpx, re, time, random, difflib
from dotenv import load_dotenv
from functools import lru_cache

# ---------------------------- ENV & SETUP ----------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEEPGRAM_KEY   = os.getenv("DEEPGRAM_API_KEY")
ELEVEN_KEY     = os.getenv("ELEVENLABS_API_KEY")
HTTP_ORIGIN    = os.getenv("PUBLIC_HTTP_ORIGIN", "http://localhost:8000")
PORT           = int(os.getenv("PORT", "8080"))
ECHO_BACK      = os.getenv("ECHO_BACK", "0") == "1"
ELEVEN_VOICE   = os.getenv("ELEVENLABS_VOICE_ID", "Bella")

# Reuse a single HTTP/2 client across providers (saves 100–300ms handshakes)
SHARED_CLIENT = httpx.AsyncClient(timeout=None, http2=True)

SYSTEM_PROMPT = """
You are the NeuroMed AI receptionist. Be warm, clear, concise, and human-sounding.

Mission:
- NeuroMed AI helps families, caregivers, and clinics by turning medical files (discharge notes, labs, imaging) into plain-language summaries, with optional styles: plain, caregiver-friendly, faith + encouragement, or clinical.

CRITICAL SCOPE LIMIT:
- Callers are contacting about NeuroMed only. If asked anything unrelated to NeuroMed's product, privacy, pricing, pilots, faith mode, demo, or support, politely say you can only answer questions about NeuroMed and offer to email more info.

Guardrails:
- Do not give medical advice or diagnose. Never promise unlisted features. Avoid speculative claims.
- Privacy: reassure that data is kept private; do not discuss implementation details unless asked.
- If asked for pricing/partners/pilots, acknowledge tiers/pilots and offer email follow-up rather than quoting numbers.
- If the user wants next steps, politely collect name + email.

Style:
- 1–2 sentences by default; longer only for the “what is NeuroMed AI” explanation (~30–60s).
- Use light natural pacing (brief pauses are added by the audio layer). Avoid filler endings like “feel free to ask more questions.”
- If the user greets or backchannels (“okay”, “yeah”), acknowledge briefly and continue.
- If the user is unclear, ask a short clarifying question.

Output Rules:
- Stay on purpose: clarity, comfort, compassion. American English only.
- Never end with generic closers unless explicitly requested.
"""

# ---------------------------- TTS WARMUP & CACHE ----------------------------
TTS_WARM_LINES = [
    "Hello—", "Okay—", "Sure—", "Thanks, one moment—",
    "Hi! What would you like help with today—overview, pricing, or getting started?"
]

@lru_cache(maxsize=256)
def _tts_cache_key(text: str) -> str:
    return text.strip().lower()

async def eleven_tts_stream(text: str):
    if not ELEVEN_KEY:
        raise RuntimeError("ELEVENLABS_API_KEY missing.")
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE}/stream"
    params = {"optimize_streaming_latency": "3", "output_format": "ulaw_8000"}
    headers = {"xi-api-key": ELEVEN_KEY, "Content-Type": "application/json", "Accept": "*/*"}
    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
    }

    async def _do():
        async with SHARED_CLIENT.stream("POST", url, headers=headers, params=params, json=payload) as r:
            if r.is_error:
                body = await r.aread()
                print("ELEVENLABS HTTP ERROR ▶", r.status_code, body[:200])
                return
            async for chunk in r.aiter_bytes():
                if chunk:
                    yield chunk

    got_audio = False
    async for c in _do():
        got_audio = True
        yield c
    if not got_audio:
        print("ELEVENLABS WARN ▶ no audio; quick retry")
        async for c in _do():
            yield c

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

    total = b"".join(chunks)
    if len(total) < 2_000_000:  # ~2MB cap
        eleven_tts_stream_cached._mem = getattr(eleven_tts_stream_cached, "_mem", {})
        eleven_tts_stream_cached._mem[key] = [total]  # compact single blob for fewer writes

async def warm_eleven():
    try:
        for line in TTS_WARM_LINES:
            async for _ in eleven_tts_stream(line):
                break
    except Exception as e:
        print("ELEVEN WARM WARN ▶", repr(e))

# ---------------------------- FEWSHOTS & FALLBACK ----------------------------
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

WHAT_IS_FALLBACK = (
  "NeuroMed AI turns medical files like discharge notes, labs, and imaging into plain-language summaries. "
  "You can choose the tone—plain, caregiver-friendly, faith + encouragement, or clinical—so it meets the moment. "
  "It’s built to help families, caregivers, and clinics find clarity quickly without medical advice."
)

# ---------------------------- BRAND / NORMALIZATION ----------------------------
NEURO_CORE = "neuromed"
NEURO_AI   = "neuromed ai"

def _soft_norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _collapse_ai_runs(s: str) -> str:
    s = re.sub(r"\b(a|e)\s*i\b", "ai", s)
    s = re.sub(r"\b(ay|eye)\b", "ai", s)
    return s

def _confuse_token(tok: str) -> str:
    tok = tok.replace("nero", "neuro").replace("neero", "neuro").replace("nuro", "neuro").replace("neural", "neuro")
    tok = tok.replace("miura", "mira")
    tok = tok.replace("mediate", "media").replace("media", "med")
    tok = tok.replace("maid", "med").replace("met", "med").replace("mad", "med").replace("mid", "med")
    tok = tok.replace("medd", "med").replace("mayd", "med")
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

FILLERS_RE = re.compile(r"^(?:\s*(?:yeah|yep|uh|um|hmm|hello|hi|hey|okay|ok)[,.\s]*)+", re.I)

def _strip_fillers(s: str) -> str:
    return FILLERS_RE.sub("", (s or "").strip()).strip()

_NEURO_ALIASES = [
    "neuro med", "neuromed", "neuro med ai", "neuro ai", "neuro media", "neuro mediate",
    "miura med", "mira med", "miramad", "miuramed", "miu rammed", "neuro mid", "neuro maid",
    "mid ai", "med ai", "my ai", "your ai", "media i", "mid a i", "neura med"
]

def _brandify_tail(s: str) -> str:
    tokens = s.lower().split()
    tail = " ".join(tokens[-3:]) if tokens else ""
    for alias in _NEURO_ALIASES:
        if difflib.SequenceMatcher(None, tail, alias).ratio() >= 0.7:
            return re.sub(r"(\b" + re.escape(tail) + r"\b)$", "NeuroMed AI", s, flags=re.I)
    return s

def domain_corrections(text: str) -> str:
    t = (text or "").strip()
    t = _strip_fillers(t)
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
        (r"\bneura\s*med\b", "NeuroMed"),
    ]
    for pat, repl in subs:
        t = re.sub(pat, repl, t, flags=re.I)

    if re.search(r"\b(tell me more|more about)\b", t, flags=re.I) and re.search(r"\b(video|media)\b\.?\??$", t, flags=re.I):
        t = re.sub(r"\b(video|media)\b\.?\??$", "NeuroMed AI", t, flags=re.I)

    t = re.sub(r"\bE\.?A\.?\??$", "NeuroMed AI?", t, flags=re.I)
    t = re.sub(r"\bA\.?I\.?\??$", "NeuroMed AI?", t, flags=re.I)
    t = re.sub(r"\bEA\??$",      "NeuroMed AI?", t, flags=re.I)
    t = re.sub(r"\bare you on security\b", "is it secure", t, flags=re.I)
    t = re.sub(r"\bis it secured\b", "is it secure", t, flags=re.I)
    t = re.sub(r"\bis (?:your|the)\s+data\s+(?:safe|secured?)\b", "is it secure", t, flags=re.I)
    t = re.sub(r"\bis it (?:hipa|hippa)\b", "is it hipaa", t, flags=re.I)

    if re.search(r"\byour\s+(?:ai|app|system)?\s*\??$", t, flags=re.I):
        t = re.sub(r"\byour\s+(?:ai|app|system)?\s*\??$", "NeuroMed AI?", t, flags=re.I)

    t = _brandify_tail(t)
    return t

def looks_like_what_is_neuromed(s: str) -> bool:
    q = (s or "")
    if re.search(r"\b(what\s+is|what'?s|tell\s+me\s+more|how\s+does\s+it\s+work)\b", q, re.I):
        return _looks_like_brand(q)["is_brand"]
    return False

# ---------------------------- ASR (Deepgram) ----------------------------
async def deepgram_stream(pcm_iter):
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
        "&vad_turnoff=220"
        "&utterance_end_ms=200"
    )
    headers = [("Authorization", f"Token {DEEPGRAM_KEY}")]

    async with websockets.connect(url, extra_headers=headers, max_size=2**20) as dg:
        try:
            await dg.send(json.dumps({
                "type": "Configure",
                "keywords": [
                    "NeuroMed","NeuroMed AI","HIPAA","privacy","private","secure","secured",
                    "security","encrypted","encryption","compliance","compliant","GDPR","PHI",
                    "pricing","pilot","demo","faith","schedule","appointment","Spanish","Tagalog",
                    "neuro med","neuro mid","neuro media","neuro mediate","mira med","miura med","miuramed",
                    "med ai","mid ai","a i","ay","eye","media i"
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
                try:
                    await dg.send(json.dumps({"type": "CloseStream"}))
                except Exception:
                    pass

        feed_task = asyncio.create_task(feeder())
        last_sent, last_emit = "", 0.0

        try:
            async for msg in dg:
                try:
                    obj = json.loads(msg)
                except Exception:
                    continue

                if isinstance(obj, dict) and obj.get("type") in {
                    "Metadata","Warning","Error","Close","UtteranceEnd","SpeechStarted","SpeechFinished"
                }:
                    continue

                alts, is_final = [], bool(obj.get("is_final"))
                chan = obj.get("channel")
                if isinstance(chan, dict):
                    alts = chan.get("alternatives") or []
                elif isinstance(chan, list) and chan:
                    first_chan = chan[0] or {}
                    if isinstance(first_chan, dict):
                        alts = first_chan.get("alternatives") or []
                elif "alternatives" in obj and isinstance(obj["alternatives"], list):
                    alts = obj["alternatives"]
                if not alts: continue

                txt = (alts[0].get("transcript") or "").strip()
                if not txt: continue

                now = time.time()
                if not (is_final or (now - last_emit) > 0.2):  # throttle partials
                    continue
                if not is_final and txt == last_sent:
                    continue

                print(f"ASR{'(final)' if is_final else ''} ▶ {txt}")
                yield txt, is_final
                last_emit, last_sent = now, txt
        finally:
            await feed_task

# ---------------------------- FAQ fetch ----------------------------
async def call_faq(q: str) -> str:
    try:
        timeout = httpx.Timeout(connect=1.0, read=2.5, write=2.5, pool=2.5)
        async with httpx.AsyncClient(timeout=timeout, http2=True) as client:
            r = await client.get(f"{HTTP_ORIGIN}/api/faq", params={"q": q})
            r.raise_for_status()
            return r.json().get("answer", "")
    except Exception as e:
        print("FAQ WARN ▶", repr(e))
        return ""

# ---------------------------- LLM Streaming (clauses too) ----------------------------
SENTENCE_END = re.compile(r'([.!?…]+)(\s+|$)')
CLAUSE_END   = re.compile(r'([,;:])(\s+|$)')
MIN_CLAUSE_CHARS = 45

async def stream_llm_sentences(messages: list[dict]):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json", "Accept": "text/event-stream"}
    payload = {"model": OPENAI_MODEL, "temperature": 0.2, "stream": True, "messages": messages}

    buf = ""
    async with SHARED_CLIENT.stream("POST", url, headers=headers, json=payload) as r:
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
            if not delta: continue
            buf += delta

            # Emit full sentences first
            while True:
                m = SENTENCE_END.search(buf)
                if not m: break
                end_idx = m.end()
                yield buf[:end_idx].strip()
                buf = buf[end_idx:]

            # Opportunistic clause emission
            cm = CLAUSE_END.search(buf)
            if cm and len(buf[:cm.end()].strip()) >= MIN_CLAUSE_CHARS:
                end_idx = cm.end()
                yield buf[:end_idx].strip()
                buf = buf[end_idx:]

    tail = buf.strip()
    if tail:
        yield tail

# ---------------------------- Intent / Slots ----------------------------
def RX(p): return re.compile(p, re.I)

HIPAA_RE     = RX(r"\b(hipaa|hippa|hipa|phi|privacy|private|confidential|secure(?:d|ly)?|security|data\s+safe|data\s+protect(?:ed|ion)|encrypt(?:ed|ion|ing)?|compliance|compliant|gdpr|safeguard(?:s)?)\b")
PRICING_RE   = RX(r"\b(price|pricing|cost|how\s+(?:much|many)|rate|rates|quote|quotation|fees?|charges?|billing|ballpark|estimate|plan(?:s)?|tier(?:s)?)\b")
PILOT_RE     = RX(r"\b(pilot|trial|try|evaluate|evaluation|proof\s+of\s+concept|poc|demo|walkthrough|test\s*drive|sandbox|kick\s*the\s+tires)\b")
FAITH_RE     = RX(r"\b(faith|bible|scripture|psalm|verse|christ|jesus|prayer|chaplain|spiritual|encourag(?:e|ement)|devotional|religio(?:us|n))\b")
LANG_RE      = RX(r"\b(multilingual|multi[-\s]?language|bilingual|translate|translation|spanish|tagalog|filipino|language\s+support|foreign\s+language)\b")
SCHED_RE     = RX(r"\b(schedule|book|set\s*(?:up)?\s*(?:a\s*)?(?:call|meeting|appt|appointment)|calendar|availability|reschedul(?:e|ing)|confirm\s+(?:time|meeting)|follow[-\s]?up\s+call)\b")
HUMAN_RE     = RX(r"\b(human|agent|representative|somebody|someone)\b|\b(speak|talk)\s+to\s+(a|the)\s+(human|person|agent)\b")
INTEG_RE     = RX(r"\b(ehr|emr|hl7|fhir|api|integrat(?:e|ion)|sso|okta|azure\s+ad|oauth|epic|cerner|meditech|athenahealth|allscripts)\b")
SUPPORT_RE   = RX(r"\b(help|support|troubleshoot|issue|error|bug|broken|not\s+working|can'?t\s+(?:login|log\s*in|sign\s*in)|password\s+reset|freeze|lag|timeout|down|glitch)\b")
ADDRESS_RE   = RX(r"\b(address|where\s+are\s+you|location|office|hq|headquarters)\b")
HOURS_RE     = RX(r"\b(hours?|open|close|availability|available\s+when)\b")
FOUNDER_RE   = RX(r"\b(founder|who'?s?\s+behind|who\s+made|who\s+built|who\s+is\s+(?:the\s+)?ceo|maria\s+gregory)\b")
PARTNER_RE   = RX(r"\b(partner|partnership|collaborat(?:e|ion)|reseller|chaplain(?:cy)?)\b")
HOW_RE       = RX(r"\b(how\s+does\s+it\s+work|how\s+it\s+works|what'?s\s+the\s+process)\b")
BETA_RE      = RX(r"\b(beta|no\s*charge|free\s*trial)\b")
MULTIFAITH_RE= RX(r"\b(muslim|islam|jewish|hindu|buddhist|multi-?faith|all\s+faiths?)\b")

YES_RE = RX(r"^(?:yes|yeah|yep|sure|correct|right|ok(?:ay)?)\.?$")
NO_RE  = RX(r"^(?:no|nope|nah|negative|not\s+really)\.?$")

def classify_intent_by_keywords(s: str) -> str:
    txt = (s or "")
    if HIPAA_RE.search(txt):      return "hipaa"
    if PRICING_RE.search(txt):    return "pricing"
    if FAITH_RE.search(txt):      return "faith_mode"
    if PARTNER_RE.search(txt):    return "partnership"
    if FOUNDER_RE.search(txt):    return "founder"
    if HOW_RE.search(txt):        return "how_it_works"
    if BETA_RE.search(txt):       return "cost_beta"
    if MULTIFAITH_RE.search(txt): return "multifaith"
    if PILOT_RE.search(txt):      return "pilot"
    if SCHED_RE.search(txt):      return "scheduling"
    if LANG_RE.search(txt):       return "multilingual"
    if HUMAN_RE.search(txt):      return "human"
    if INTEG_RE.search(txt):      return "integration"
    if SUPPORT_RE.search(txt):    return "support"
    if ADDRESS_RE.search(txt):    return "address"
    if HOURS_RE.search(txt):      return "hours"
    return "other"

# ---------------------------- Planner helpers ----------------------------
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

async def plan_reply(history: list[dict], state: dict | None = None) -> list[str]:
    last = (history[-1]["content"] if history else "").strip()

    # keyword fast-path (cheap + consistent)
    intent = classify_intent_by_keywords(last)

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

    # new fast-paths from your scripts
    if intent == "founder":
        return ["NeuroMed AI was founded by Maria Gregory after years of confusing hospital updates with her husband—our goal is clarity and compassion for families."]
    if intent == "partnership":
        return ["We offer 60-day pilots for hospitals and nursing homes, plus chaplaincy and multifaith options. I can email the pilot brochure—what’s the best email?"]
    if intent == "how_it_works":
        return ["Upload medical documents, pick a tone—Plain, Caregiver, or Faith-aware—then get a clear summary to share as print, email, or voice note."]
    if intent == "cost_beta":
        return ["We’re in beta—there’s no charge to try it while we collect feedback. After pilots, pricing is tailored for families or facilities."]
    if intent == "multifaith":
        return ["We’re for everyone—modes include Plain, Caregiver, Faith-aware, and multilingual, with encouragement across several faiths or a secular option."]

    # “what is” fallback (longer)
    if looks_like_what_is_neuromed(last):
        text = WHAT_IS_FALLBACK
        return [s for s in re.split(r'(?<=[.!?])\s+', text) if s and not should_skip_sentence(s)]

    # collect contact
    if intent == "scheduling":
        return ["Great—could I have your name and best email so we can send next steps?"]

    # Strict scope limit by design
    return ["I can help with NeuroMed only—overview, privacy, pricing, faith mode, or getting started. Which would you like?"]

# ---------------------------- Twilio WS Server ----------------------------
def b64_to_bytes(s: str) -> bytes: return base64.b64decode(s)
def bytes_to_b64(b: bytes) -> str: return base64.b64encode(b).decode()

def _ulaw_silence(ms: int) -> bytes:
    return b"\xff" * int(8 * ms)

async def send_silence(ws_send_pcm, ms: int):
    if ms <= 0: return
    try:
        await ws_send_pcm(_ulaw_silence(ms))
    except (websockets.exceptions.ConnectionClosedOK, websockets.exceptions.ConnectionClosedError):
        pass

async def handle_twilio(ws):
    if ws.path not in ("/ws/twilio",):
        await ws.close(code=1008, reason="Unexpected path")
        return

    print("WS ▶ Twilio connected, path:", ws.path)
    print("WS ▶ protocol negotiated:", ws.subprotocol)
    if ws.subprotocol not in (None, "audio.twilio.com"):
        print("WS WARN ▶ unexpected subprotocol:", ws.subprotocol)

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
        "last_question": None,        # "yn" | "open" | None
        "TOPIC_STICKY_SECS": 10.0,
    }

    conn_open = True
    stream_sid = None
    speak_task: asyncio.Task | None = None

    async def arm_barge_in_after(seconds: float):
        await asyncio.sleep(seconds)
        state["barge_in_enabled"] = True
        print(f"BARGE-IN ▶ enabled after {seconds:.1f}s")

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
            await ws.send(json.dumps({
                "event": "media",
                "streamSid": stream_sid,
                "media": {"payload": bytes_to_b64(pcm)}
            }))
            return True
        except (websockets.exceptions.ConnectionClosedOK, websockets.exceptions.ConnectionClosedError):
            conn_open = False
            return False

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
            print("TTS ▶ duplicate suppressed")
            return

        if speak_task and not speak_task.done():
            speak_task.cancel()
            try:
                await speak_task
            except asyncio.CancelledError:
                pass

        async def _run():
            state["speaking"] = True
            any_chunk, cancelled = False, False
            prev_barge = state.get("barge_in_enabled", False)
            state["local_no_barge_until"] = time.time() + 0.6  # shorter grace

            try:
                if not conn_open or not stream_sid: return

                # shorter pre/post silences
                pre = 25 if len(text.split()) > 1 else 15
                pre += int(random.uniform(-10, 10))
                await send_silence(send_pcm, max(pre, 10))

                async for pcm in eleven_tts_stream_cached(text):
                    any_chunk = True
                    ok = await send_pcm(pcm)
                    if not ok: return

                if not conn_open: return

                post = 35 + int(random.uniform(-10, 10))
                await send_silence(send_pcm, max(post, 10))

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

    async def speak_lines(lines: list[str], *, greeted_already: bool, allow_contact_request: bool) -> int:
        spoken = 0
        for sentence in lines:
            if should_skip_sentence(sentence): continue
            if should_block_contact_line(sentence, allow_contact_request=allow_contact_request): continue
            if should_drop_assistant_line(sentence, greeted_already=greeted_already): continue
            await speak(sentence)
            spoken += 1
        if spoken == 0:
            # ultra-short fallback—keeps the line live
            await speak("Okay—")
            spoken = 1
        return spoken

    async def brain():
        if not DEEPGRAM_KEY:
            return

        try:
            await asyncio.wait_for(first_media.wait(), timeout=20.0)
        except asyncio.TimeoutError:
            print("ASR ▶ no media within 20s; skipping ASR session")
            return

        last_user_sent = ""
        FINAL_DEBOUNCE_SECS = 0.35
        pending_final, pending_ts = None, 0.0
        last_partial_ack = 0.0
        PARTIAL_ACK_DEBOUNCE = 0.45

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

                yn = None
                if YES_RE.match(fixed.strip()): yn = "yes"
                elif NO_RE.match(fixed.strip()): yn = "no"
                if yn and state.get("last_question") == "yn" and state.get("last_intent") not in (None, "other"):
                    kw_intent = state["last_intent"]

                if kw_intent != "other":
                    state["last_intent"] = kw_intent
                    state["last_intent_ts"] = nowt

                    quicks = {
                        "hipaa":         ["Your files are kept private and secure, and we don’t share your data with anyone."],
                        "pricing":       ["We offer options for families, caregivers, and facilities. I can send a quick overview—what’s the best email to use?"],
                        "faith_mode":    ["Yes—there’s a mode with encouragement and Scripture for those who want hope alongside clarity."],
                        "pilot":         ["We do pilot programs with nursing homes, clinics, and community groups. I can email details—what email should we use?"],
                        "multilingual":  ["We can provide summaries in multiple languages—for families who need clarity in their heart language."],
                        "scheduling":    ["We’re available on weekdays. If you share your time zone, I can suggest a slot."],
                        "human":         ["I can collect your name and email so a teammate follows up shortly."],
                        "integration":   ["We integrate with clinical systems like EHRs via standards like HL7 and FHIR."],
                        "support":       ["I can help get you support—can I have your best email so we can follow up?"],
                        "address":       ["We operate online and partner with facilities; if you share your city, I can route you."],
                        "hours":         ["We’re available weekdays."],
                        "founder":       ["NeuroMed AI was founded by Maria Gregory after years of confusing hospital updates with her husband—our goal is clarity and compassion for families."],
                        "partnership":   ["We offer 60-day pilots for hospitals and nursing homes, plus chaplaincy and multifaith options. I can email the pilot brochure—what’s the best email?"],
                        "how_it_works":  ["Upload medical documents, pick a tone—Plain, Caregiver, or Faith-aware—then get a clear summary to share as print, email, or voice note."],
                        "cost_beta":     ["We’re in beta—there’s no charge to try it while we collect feedback. After pilots, pricing is tailored for families or facilities."],
                        "multifaith":    ["We’re for everyone—modes include Plain, Caregiver, Faith-aware, and multilingual, with encouragement across several faiths or a secular option."],
                    }

                    if kw_intent in quicks:
                        state["last_question"] = "yn" if kw_intent in {"pricing","pilot","support"} else "open"
                        await speak_lines(quicks[kw_intent], greeted_already=state["greeted"], allow_contact_request=(kw_intent in {"pricing","pilot","support"}))
                        return

                brand = _looks_like_brand(fixed)
                if re.search(r"\b(what\s+is|what'?s|tell\s+me\s+more|how\s+does\s+it\s+work)\b", fixed, re.I) and brand["is_brand"]:
                    asyncio.create_task(speak("Sure—"))
                    answer = await call_faq("what_is_neuromed") or WHAT_IS_FALLBACK
                    state["long_answer_until"] = time.time() + 2.0
                    sentences = [s for s in re.split(r'(?<=[.!?])\s+', answer.strip()) if s]
                    await speak_lines(sentences, greeted_already=state["greeted"], allow_contact_request=state["allow_contact_request"])
                    return

                if re.search(r"\b(what\s+is|what'?s|tell\s+me\s+more|how\s+does\s+it\s+work)\b", fixed, re.I):
                    asyncio.create_task(speak("Sure—"))
                    answer = await call_faq("what_is_neuromed") or WHAT_IS_FALLBACK
                    state["long_answer_until"] = time.time() + 2.0
                    sentences = [s for s in re.split(r'(?<=[.!?])\s+', answer.strip()) if s]
                    await speak_lines(sentences, greeted_already=state["greeted"], allow_contact_request=state["allow_contact_request"])
                    return

                info = extract_contact_inline(fixed)
                if "email" in info: state["contact"]["email"] = info["email"]
                if "name"  in info and not state["contact"]["name"]: state["contact"]["name"] = info["name"]

                if re.search(r"\b(email|send|share)\b.*\b(info|details|pricing|pilot|follow\s*up)\b", fixed, re.I):
                    state["allow_contact_request"] = True

                words = fixed.split()
                in_long_window = nowt < state["long_answer_until"]
                should_barge = len(words) >= (5 if in_long_window else 3)
                if should_barge and state.get("barge_in_enabled") and state["speaking"] and speak_task and not speak_task.done():
                    speak_task.cancel()
                    try:
                        await speak_task
                    except asyncio.CancelledError:
                        pass

                history.append({"role": "user", "content": fixed})

                if len(words) <= 6:
                    asyncio.create_task(asyncio.sleep(0))
                    asyncio.create_task(speak("Okay—"))

                state["long_answer_until"] = time.time() + 2.0
                try:
                    plan = await plan_reply(history, state=state)
                except Exception as e:
                    print("PLAN ERROR ▶", repr(e))
                    plan = []

                if plan:
                    await speak_lines(plan, greeted_already=state["greeted"], allow_contact_request=state["allow_contact_request"])
                else:
                    gen_messages = [{"role": "system", "content": SYSTEM_PROMPT}, *FEWSHOTS, *history]
                    spoke_count = 0

                    async def nudge():
                        await asyncio.sleep(1.2)
                        if spoke_count == 0:
                            await speak("Okay—")

                    nudge_task = asyncio.create_task(nudge())
                    try:
                        async for sentence in stream_llm_sentences(gen_messages):
                            said = await speak_lines([sentence], greeted_already=state["greeted"], allow_contact_request=state["allow_contact_request"])
                            if said > 0:
                                spoke_count += said
                                if not nudge_task.done():
                                    nudge_task.cancel()
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

            # fast partial ack so the line never feels dead
            if not is_final:
                nowp = time.time()
                if len(utter.split()) >= 4 and (nowp - last_partial_ack) > PARTIAL_ACK_DEBOUNCE:
                    last_partial_ack = nowp
                    asyncio.create_task(speak("Okay—"))

            if utter == last_user_sent:
                continue
            last_user_sent = utter

            now = time.time()
            if pending_final and (now - pending_ts) > FINAL_DEBOUNCE_SECS:
                await process_final_utterance(pending_final)
                pending_final = None

            if is_final:
                if pending_final and (now - pending_ts) <= FINAL_DEBOUNCE_SECS and not re.search(r"[.!?…]\s*$", pending_final):
                    pending_final = (pending_final + " " + utter).strip()
                    pending_ts = now
                else:
                    pending_final = utter.strip()
                    pending_ts = now

                if re.search(r"[.!?…]\s*$", pending_final) or len(pending_final.split()) >= 8:
                    await process_final_utterance(pending_final)
                    pending_final = None

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

                # arm barge-in immediately so callers can interrupt the greeting
                state["barge_in_enabled"] = True
                if not state["greeted"]:
                    state["greeted"] = True
                    asyncio.create_task(
                        speak("Hello, you’ve reached NeuroMed AI — where we turn complex medical notes into clear, compassionate summaries for families and caregivers. How may I help you today?")
                    )

            elif ev == "media":
                payload_b64 = data["media"]["payload"]
                buf = b64_to_bytes(payload_b64)

                media_stats["frames"] += 1
                media_stats["bytes"]  += len(buf)
                if media_stats["frames"] == 1:
                    first_media.set()
                    print("WS ▶ first media frame received")

                if media_stats["frames"] % 25 == 0:
                    print(f"WS ▶ media frames={media_stats['frames']} bytes={media_stats['bytes']}")

                await inbound_q.put(buf)

                if ECHO_BACK and stream_sid and conn_open:
                    await send_pcm(buf)

            elif ev == "stop":
                print("WS ▶ stop")
                conn_open = False
                if speak_task and not speak_task.done():
                    speak_task.cancel()
                    try:
                        await speak_task
                    except asyncio.CancelledError:
                        pass
                break

    except Exception as e:
        print("WS ERR ▶", e)
    finally:
        conn_open = False
        await inbound_q.put(None)
        if speak_task and not speak_task.done():
            speak_task.cancel()
            try:
                await speak_task
            except asyncio.CancelledError:
                pass
        await brain_task
        print("WS ▶ closed")

# ---------------------------- Main ----------------------------
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
    try:
        asyncio.run(main())
    finally:
        try:
            asyncio.get_event_loop().run_until_complete(SHARED_CLIENT.aclose())
        except Exception:
            pass
