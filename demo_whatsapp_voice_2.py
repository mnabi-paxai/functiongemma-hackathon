#!/usr/bin/env python3
"""
Voice-to-WhatsApp demo (laptop)

What it does:
- Continuously records short audio chunks from your microphone
- Transcribes locally using Cactus Whisper (cactus_transcribe)
- When it hears a WhatsApp command, it:
  1) extracts (recipient, message) via your existing generate_hybrid tool-caller
  2) opens WhatsApp Web and sends the message (Playwright automation)

Important notes:
- This is a demo script; it is intentionally simple and "chunked" (e.g., 4s audio windows).
- WhatsApp Web requires you to scan a QR code on first run. The script uses a persistent
  browser profile so you only need to scan once.

Dependencies:
- pip install sounddevice soundfile playwright
- python -m playwright install

Run:
- python demo_whatsapp_voice.py

Stop:
- Ctrl+C

This file does NOT change your benchmark/leaderboard code.
"""

import os
import re
import time
import json
import tempfile
import difflib
from pathlib import Path

# Audio capture
import sounddevice as sd
import soundfile as sf

# Cactus (local)
import sys
sys.path.insert(0, "cactus/python/src")
# Disable Cactus cloud telemetry noise for this demo
os.environ.setdefault("CACTUS_NO_CLOUD_TELE", "1")

from cactus import cactus_init, cactus_transcribe, cactus_destroy
from main import generate_hybrid  # re-use your existing routing + tool-call output

# Browser automation (WhatsApp Web)
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


# -----------------------------
# Config
# -----------------------------
WHISPER_WEIGHTS = os.environ.get("WHISPER_WEIGHTS", "weights/whisper-small")
AUDIO_SR = int(os.environ.get("AUDIO_SR", "16000"))
CHUNK_SECONDS = float(os.environ.get("CHUNK_SECONDS", "10.0"))

# "Command gate" phrases to reduce accidental sends.
# You can expand these.
COMMAND_TRIGGERS = (
    "whatsapp",
    "what's app",
    "what app",


)

# Wake word(s). The assistant will only act on commands after hearing a wake word.
# Examples the transcriber might output: "hey cactus", "hi cactus".
WAKE_WORDS = tuple(w.strip().lower() for w in os.environ.get("WAKE_WORDS", "hey cactus,hi cactus").split(",") if w.strip())

# How long (in seconds) the assistant stays "armed" after hearing the wake word.
ARMED_SECONDS = float(os.environ.get("ARMED_SECONDS", "12"))


# Persistent browser profile directory so WhatsApp login persists.
PROFILE_DIR = Path(os.environ.get("WHATSAPP_PROFILE_DIR", ".whatsapp_profile")).resolve()

# WhatsApp Web
WHATSAPP_WEB_URL = "https://web.whatsapp.com/"


# -----------------------------
# Tool schema for parsing intent
# -----------------------------
TOOL_WHATSAPP_SEND = {
    "name": "whatsapp_send",
    "description": (
        "Send a WhatsApp message to a contact via WhatsApp Web. "
        "Use this when the user asks to message someone on WhatsApp/WhatsApp."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "recipient": {"type": "string", "description": "Contact name exactly as it appears in WhatsApp"},
            "message": {"type": "string", "description": "The message content to send"},
        },
        "required": ["recipient", "message"],
    },
}


SYSTEM_PROMPT = (
    "You are a voice assistant. "
    "When the user asks to message someone on WhatsApp, call whatsapp_send with recipient and message. "
    "If recipient is ambiguous, use the name the user said verbatim. "
    "Return ONLY tool calls; do not chat."
)




def validate_whisper_weights(path_str: str) -> str:
    """
    Returns a usable whisper weights directory path, or raises with a helpful message.
    Tries a couple common locations used in this repo (./weights and ./cactus/weights).
    """
    candidates = [path_str, "weights/whisper-small", "cactus/weights/whisper-small"]
    for c in candidates:
        cfg = Path(c) / "config.txt"
        if cfg.exists():
            return str(Path(c))
    raise FileNotFoundError(
        "Whisper weights not found. Expected a config.txt under one of:\n"
        f"  - {candidates[0]}\n  - weights/whisper-small\n  - cactus/weights/whisper-small\n\n"
        "Fix:\n"
        "  1) Download the model into ./weights:\n"
        "       cactus download openai/whisper-small\n"
        "  2) Or set WHISPER_WEIGHTS to the correct folder:\n"
        "       export WHISPER_WEIGHTS=/path/to/whisper-small\n"
    )

# -----------------------------
# Audio + Transcription
# -----------------------------
def record_chunk(seconds: float, sr: int) -> str:
    """Record an audio chunk and return a path to a temporary WAV."""
    frames = int(seconds * sr)
    audio = sd.rec(frames, samplerate=sr, channels=1, dtype="float32")
    sd.wait()

    # Write to a temp WAV file
    fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="cactus_chunk_")
    os.close(fd)
    sf.write(wav_path, audio, sr)
    return wav_path


def transcribe_chunk(whisper, wav_path: str) -> str:
    """Transcribe a WAV file with Cactus Whisper."""
    # Prompt from README example for English transcription
    prompt = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
    raw = cactus_transcribe(whisper, wav_path, prompt=prompt)
    try:
        data = json.loads(raw)
        return (data.get("response") or "").strip()
    except json.JSONDecodeError:
        return ""


def is_non_speech(text: str) -> bool:
    """
    Whisper often outputs non-speech in parentheses, e.g. '(upbeat music)'.
    Ignore those chunks to reduce false triggers.
    """
    t = text.strip()
    return (t.startswith("(") and t.endswith(")")) or t in {"", "[BLANK_AUDIO]"}

def detect_wake_word(text: str) -> bool:
    """
    Wake word detection is intentionally fuzzy because Whisper can mis-hear "cactus"
    (e.g., "practice", "cactus", "cactus.", etc.).
    Strategy:
      - If any configured WAKE_WORDS appears as a substring -> wake.
      - Else if text starts with hey/hi/hello and the next token is *similar* to "cactus" -> wake.
    """
    t = text.lower().strip()
    if not t:
        return False

    # Direct substring match for explicit wake phrases
    if any(w in t for w in WAKE_WORDS):
        return True

    # Fuzzy: "hey <something like cactus>"
    tokens = re.findall(r"[a-z']+", t)
    if len(tokens) >= 2 and tokens[0] in {"hey", "hi", "hello", "okay", "ok"}:
        cand = tokens[1]
        # Similarity against "cactus"
        sim = difflib.SequenceMatcher(None, cand, "cactus").ratio()
        # Also allow cases like "hey, practice" where practice ~ cactus (often ~0.57-0.66)
        if sim >= 0.58:
            return True

    return False


def strip_wake_word_prefix(text: str) -> str:
    """
    If the user said "hey cactus, ..." remove the wake phrase so parsing is cleaner.
    This is intentionally conservative: we only strip when the wake word appears near the start.
    """
    t = text.strip()
    low = t.lower()
    for w in WAKE_WORDS:
        # wake word near the start (+ optional punctuation)
        if low.startswith(w):
            return t[len(w):].lstrip(" ,.:;!-")
        # allow "hey cactus," with minor leading filler
        m = re.match(rf"^\s*(?:hey|hi|hello)\s+cactus\b[ ,.:;!-]*", low, re.IGNORECASE)
        if m:
            return t[m.end():].lstrip(" ,.:;!-")
    return t


def looks_like_whatsapp_command(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in COMMAND_TRIGGERS) and ("message" in t or "text" in t or "send" in t)

# -----------------------------
# Intent extraction
# -----------------------------
def extract_with_hybrid_llm(text: str) -> dict | None:
    """
    Use your existing generate_hybrid() to extract recipient+message as a tool call.

    Returns: {"recipient": ..., "message": ...} or None
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    result = generate_hybrid(messages, [TOOL_WHATSAPP_SEND])
    calls = result.get("function_calls", []) or []
    for c in calls:
        if c.get("name") == "whatsapp_send":
            args = c.get("arguments", {}) or {}
            r = (args.get("recipient") or "").strip()
            m = (args.get("message") or "").strip()
            if r and m:
                return {"recipient": r, "message": m, "parse_source": result.get("source", "unknown")}
    return None


def fallback_extract(text: str) -> dict | None:
    """
    Heuristic fallback if the model doesn't return a tool call.
    Tries patterns like:
      - "message Alice on whatsapp saying hello"
      - "send a whatsapp to Bob: I'm late"
    """
    t = text.strip()

    # Common "to <name> ... say/saying <msg>" pattern
    m = re.search(r"\b(?:message|text|send)\b\s+(?:a\s+)?(?:whatsapp\s+)?(?:to\s+)?([A-Z][\w\- ]{1,40}?)\s+(?:on\s+)?(?:whatsapp|what'?s\s*app)\b.*?\b(?:say|saying|that)\b\s+(.+)$", t, re.IGNORECASE)
    if m:
        return {"recipient": m.group(1).strip(), "message": m.group(2).strip(), "parse_source": "regex"}

    # "whatsapp <name>: <msg>"
    m = re.search(r"\b(?:whatsapp|what'?s\s*app)\b\s+([A-Z][\w\- ]{1,40}?)\s*[:\-]\s*(.+)$", t, re.IGNORECASE)
    if m:
        return {"recipient": m.group(1).strip(), "message": m.group(2).strip(), "parse_source": "regex"}

    return None


# -----------------------------
# WhatsApp Web automation
# -----------------------------
def _try_fill(page, selectors, text, press_enter=False, timeout_ms=8000):
    last_err = None
    for sel in selectors:
        try:
            page.wait_for_selector(sel, timeout=timeout_ms)
            page.click(sel)
            page.fill(sel, text)
            if press_enter:
                page.keyboard.press("Enter")
            return True
        except Exception as e:
            last_err = e
    if last_err:
        raise last_err
    return False


def ensure_whatsapp_logged_in(page, timeout_s: int = 120):
    """
    If not logged in, WhatsApp shows a QR code. We give the user time to scan it.
    We consider "logged in" when the chat search box is visible.
    """
    # WhatsApp Web UI changes, so we try multiple selectors for the left search box.
    search_selectors = [
        'div[contenteditable="true"][data-tab="3"]',
        'div[contenteditable="true"][data-tab="2"]',
        'div[role="textbox"][contenteditable="true"]',
    ]

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        for sel in search_selectors:
            try:
                if page.query_selector(sel):
                    return
            except Exception:
                pass
        time.sleep(1)

    raise RuntimeError("Timed out waiting for WhatsApp Web login (scan the QR code).")


def whatsapp_send_via_web(recipient: str, message: str):
    """
    Send a message via WhatsApp Web using a persistent Playwright profile.
    """
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            headless=False,
        )
        page = browser.pages[0] if browser.pages else browser.new_page()
        page.goto(WHATSAPP_WEB_URL, wait_until="domcontentloaded")

        ensure_whatsapp_logged_in(page)

        # 1) Search for recipient
        # Try a few candidate selectors for the search field.
        search_selectors = [
            'div[contenteditable="true"][data-tab="3"]',
            'div[contenteditable="true"][data-tab="2"]',
            'div[role="textbox"][contenteditable="true"]',
        ]

        # Clear any prior text (Ctrl/Cmd+A then Backspace)
        page.keyboard.press("Control+A")
        page.keyboard.press("Backspace")

        _try_fill(page, search_selectors, recipient, press_enter=False)

        # Wait briefly for results to populate
        time.sleep(1.0)

        # 2) Click the first matching chat in the results list
        # WhatsApp Web often uses span[title="Name"] in the results.
        candidate_chat_selectors = [
            f'span[title="{recipient}"]',
            f'text="{recipient}"',
        ]
        clicked = False
        for sel in candidate_chat_selectors:
            try:
                el = page.query_selector(sel)
                if el:
                    el.click()
                    clicked = True
                    break
            except Exception:
                pass
        if not clicked:
            # As a fallback, press Enter to open the top result.
            page.keyboard.press("Enter")

        # 3) Type message in composer and send
        composer_selectors = [
            'div[contenteditable="true"][data-tab="10"]',
            'div[contenteditable="true"][data-tab="9"]',
            'footer div[contenteditable="true"][role="textbox"]',
            'div[role="textbox"][contenteditable="true"]',
        ]

        _try_fill(page, composer_selectors, message, press_enter=False)
        page.keyboard.press("Enter")

        # Give a moment for send to complete
        time.sleep(1.0)

        browser.close()


# -----------------------------
# Main loop
# -----------------------------
def main():
    print("\n🎙️  Voice-to-WhatsApp demo started.")
    print("Listening continuously in short chunks (wake-word required).")
    print(f"Chunk length: {CHUNK_SECONDS}s | Sample rate: {AUDIO_SR}Hz")
    print("Say something like: 'Hey Cactus, send a WhatsApp message to Alice saying I'm running late.'")
    print("Stop with Ctrl+C.\n")

    try:
        whisper = cactus_init(validate_whisper_weights(WHISPER_WEIGHTS))
    except Exception as e:
        print(f"\n❌ {e}\n")
        return

    armed_until = 0.0  # epoch seconds until which commands are accepted

    try:
        while True:
            wav_path = record_chunk(CHUNK_SECONDS, AUDIO_SR)
            try:
                text = transcribe_chunk(whisper, wav_path)
            finally:
                try:
                    os.remove(wav_path)
                except OSError:
                    pass

            if not text:
                continue
            print(f"Heard: {text}")

            # Wake-word gating:
            # - If wake word is heard in this chunk, arm the assistant for ARMED_SECONDS.
            # - If already armed (armed_until in the future), accept commands even without wake word.
            now = time.time()
            heard_wake = detect_wake_word(text)
            if heard_wake:
                armed_until = now + ARMED_SECONDS

            if not heard_wake and now > armed_until:
                # Ignore everything until wake word is spoken.
                continue

            # If wake word is present, strip it for cleaner parsing.
            cleaned = strip_wake_word_prefix(text)

            if not looks_like_whatsapp_command(cleaned):
                continue

            parsed = extract_with_hybrid_llm(cleaned) or fallback_extract(cleaned)
            if not parsed:
                print("Could not extract recipient/message yet. Try rephrasing.")
                continue

            recipient = parsed["recipient"]
            message = parsed["message"]
            print(f"→ Sending WhatsApp to '{recipient}': {message}  (parsed via {parsed.get('parse_source')})")

            try:
                whatsapp_send_via_web(recipient, message)
                print("✅ Sent.\n")
            except Exception as e:
                print(f"❌ Failed to send via WhatsApp Web: {e}\n")

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        cactus_destroy(whisper)


if __name__ == "__main__":
    main()
