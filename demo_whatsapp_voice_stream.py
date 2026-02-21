#!/usr/bin/env python3
"""
Always-listening Voice-to-WhatsApp demo (streaming mic)

Key difference vs chunked version:
- Uses a continuous microphone stream (sounddevice.InputStream) so the mic NEVER stops
  while the program runs.
- Every STEP_SECONDS, it snapshots the most recent WINDOW_SECONDS of audio and submits
  a transcription job in a background thread. While transcribing, the mic keeps streaming.
- Wake word gating remains: you must say "Hey Cactus" (fuzzy) before WhatsApp commands run.

Install:
  pip install sounddevice soundfile playwright numpy
  python -m playwright install
  (macOS) brew install portaudio   # if sounddevice needs it

Run:
  python demo_whatsapp_voice_stream.py

Env knobs:
  WHISPER_WEIGHTS=weights/whisper-small
  AUDIO_SR=16000
  WINDOW_SECONDS=6
  STEP_SECONDS=1.5
  WAKE_WORDS="hey cactus,hi cactus"
  ARMED_SECONDS=12
"""

import os
import re
import time
import json
import tempfile
import threading
from pathlib import Path
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import difflib

import numpy as np
import sounddevice as sd
import soundfile as sf

import sys
sys.path.insert(0, "cactus/python/src")
os.environ.setdefault("CACTUS_NO_CLOUD_TELE", "1")  # reduce telemetry noise

from cactus import cactus_init, cactus_transcribe, cactus_destroy
from main import generate_hybrid

from playwright.sync_api import sync_playwright


# -----------------------------
# Config
# -----------------------------
WHISPER_WEIGHTS = os.environ.get("WHISPER_WEIGHTS", "weights/whisper-small")
AUDIO_SR = int(os.environ.get("AUDIO_SR", "16000"))

# Streaming windowing:
WINDOW_SECONDS = float(os.environ.get("WINDOW_SECONDS", "6"))
STEP_SECONDS = float(os.environ.get("STEP_SECONDS", "1.5"))

# WhatsApp command gating
COMMAND_TRIGGERS = (
    "whatsapp",
    "what's app",
    "what app",
)

WAKE_WORDS = tuple(
    w.strip().lower()
    for w in os.environ.get("WAKE_WORDS", "hey cactus,hi cactus").split(",")
    if w.strip()
)
ARMED_SECONDS = float(os.environ.get("ARMED_SECONDS", "12"))

PROFILE_DIR = Path(os.environ.get("WHATSAPP_PROFILE_DIR", ".whatsapp_profile")).resolve()
WHATSAPP_WEB_URL = "https://web.whatsapp.com/"


TOOL_WHATSAPP_SEND = {
    "name": "whatsapp_send",
    "description": (
        "Send a WhatsApp message to a contact via WhatsApp Web. "
        "Use this when the user asks to message someone on WhatsApp."
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
    "Return ONLY tool calls; do not chat."
)


# -----------------------------
# Whisper weights validation
# -----------------------------
def validate_whisper_weights(path_str: str) -> str:
    candidates = [path_str, "weights/whisper-small", "cactus/weights/whisper-small"]
    for c in candidates:
        if (Path(c) / "config.txt").exists():
            return str(Path(c))
    raise FileNotFoundError(
        "Whisper weights not found. Expected a config.txt under one of:\n"
        f"  - {candidates[0]}\n  - weights/whisper-small\n  - cactus/weights/whisper-small\n\n"
        "Fix:\n"
        "  cactus download openai/whisper-small\n"
        "or set:\n"
        "  export WHISPER_WEIGHTS=/path/to/whisper-small\n"
    )


# -----------------------------
# Helpers
# -----------------------------
def is_non_speech(text: str) -> bool:
    t = text.strip()
    return (t.startswith("(") and t.endswith(")")) or t in {"", "[BLANK_AUDIO]"}


def detect_wake_word(text: str) -> bool:
    t = text.lower().strip()
    if not t:
        return False
    if any(w in t for w in WAKE_WORDS):
        return True
    tokens = re.findall(r"[a-z']+", t)
    if len(tokens) >= 2 and tokens[0] in {"hey", "hi", "hello", "okay", "ok"}:
        cand = tokens[1]
        sim = difflib.SequenceMatcher(None, cand, "cactus").ratio()
        if sim >= 0.58:
            return True
    return False


def strip_wake_word_prefix(text: str) -> str:
    t = text.strip()
    low = t.lower()
    for w in WAKE_WORDS:
        if low.startswith(w):
            return t[len(w):].lstrip(" ,.:;!-")
    m = re.match(r"^\s*(?:hey|hi|hello|ok|okay)\s+cactus\b[ ,.:;!-]*", low, re.IGNORECASE)
    if m:
        return t[m.end():].lstrip(" ,.:;!-")
    return t


def looks_like_whatsapp_command(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in COMMAND_TRIGGERS) and ("message" in t or "text" in t or "send" in t)


# -----------------------------
# Transcription
# -----------------------------
def transcribe_audio_array(whisper, audio: np.ndarray, sr: int) -> str:
    """
    audio: shape (n,) float32 [-1,1]
    """
    fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="cactus_stream_")
    os.close(fd)
    try:
        sf.write(wav_path, audio, sr)
        prompt = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
        raw = cactus_transcribe(whisper, wav_path, prompt=prompt)
        try:
            data = json.loads(raw)
            return (data.get("response") or "").strip()
        except json.JSONDecodeError:
            return ""
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass


# -----------------------------
# Intent extraction
# -----------------------------
def extract_with_hybrid_llm(text: str) -> dict | None:
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
                return {"recipient": r, "message": m}
    return None


def fallback_extract(text: str) -> dict | None:
    t = text.strip()
    m = re.search(
        r"\b(?:message|text|send)\b\s+(?:a\s+)?(?:whatsapp\s+)?(?:to\s+)?([A-Z][\w\- ]{1,40}?)\s+"
        r"(?:on\s+)?(?:whatsapp|what'?s\s*app)\b.*?\b(?:say|saying|that)\b\s+(.+)$",
        t,
        re.IGNORECASE,
    )
    if m:
        return {"recipient": m.group(1).strip(), "message": m.group(2).strip()}
    m = re.search(r"\b(?:whatsapp|what'?s\s*app)\b\s+([A-Z][\w\- ]{1,40}?)\s*[:\-]\s*(.+)$", t, re.IGNORECASE)
    if m:
        return {"recipient": m.group(1).strip(), "message": m.group(2).strip()}
    return None


# -----------------------------
# WhatsApp Web automation
# -----------------------------
def ensure_whatsapp_logged_in(page, timeout_s: int = 120):
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
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch_persistent_context(user_data_dir=str(PROFILE_DIR), headless=False)
        page = browser.pages[0] if browser.pages else browser.new_page()
        page.goto(WHATSAPP_WEB_URL, wait_until="domcontentloaded")
        ensure_whatsapp_logged_in(page)

        # Search box
        search_selectors = [
            'div[contenteditable="true"][data-tab="3"]',
            'div[contenteditable="true"][data-tab="2"]',
            'div[role="textbox"][contenteditable="true"]',
        ]
        # Focus search and clear
        page.keyboard.press("Control+A")
        page.keyboard.press("Backspace")
        for sel in search_selectors:
            try:
                page.wait_for_selector(sel, timeout=8000)
                page.click(sel)
                page.fill(sel, recipient)
                break
            except Exception:
                continue

        time.sleep(1.0)

        # Click match or open top result
        clicked = False
        for sel in [f'span[title="{recipient}"]', f'text="{recipient}"']:
            try:
                el = page.query_selector(sel)
                if el:
                    el.click()
                    clicked = True
                    break
            except Exception:
                pass
        if not clicked:
            page.keyboard.press("Enter")

        # Composer
        composer_selectors = [
            'div[contenteditable="true"][data-tab="10"]',
            'div[contenteditable="true"][data-tab="9"]',
            'footer div[contenteditable="true"][role="textbox"]',
            'div[role="textbox"][contenteditable="true"]',
        ]
        for sel in composer_selectors:
            try:
                page.wait_for_selector(sel, timeout=8000)
                page.click(sel)
                page.fill(sel, message)
                break
            except Exception:
                continue
        page.keyboard.press("Enter")
        time.sleep(1.0)
        browser.close()


# -----------------------------
# Streaming capture
# -----------------------------
class AudioRingBuffer:
    def __init__(self, max_seconds: float, sr: int):
        self.sr = sr
        self.max_frames = int(max_seconds * sr)
        self.buf = deque()  # chunks of np.ndarray (n, )
        self.n_frames = 0
        self.lock = threading.Lock()

    def push(self, frames: np.ndarray):
        # frames: (n, 1) float32
        mono = frames.reshape(-1).astype(np.float32, copy=False)
        with self.lock:
            self.buf.append(mono)
            self.n_frames += mono.shape[0]
            # trim oldest
            while self.n_frames > self.max_frames and self.buf:
                old = self.buf.popleft()
                self.n_frames -= old.shape[0]

    def snapshot_last(self, seconds: float) -> np.ndarray:
        need = int(seconds * self.sr)
        with self.lock:
            if self.n_frames == 0:
                return np.zeros((0,), dtype=np.float32)
            parts = list(self.buf)
        data = np.concatenate(parts) if parts else np.zeros((0,), dtype=np.float32)
        if data.shape[0] <= need:
            return data
        return data[-need:]


def main():
    print("\n🎙️  Always-listening Voice-to-WhatsApp demo started.")
    print("Mic stream stays on continuously while code runs (transcription happens in background).")
    print(f"Window: {WINDOW_SECONDS}s | Step: {STEP_SECONDS}s | Sample rate: {AUDIO_SR}Hz")
    print("Say: 'Hey Cactus, send a WhatsApp message to Alice saying I'm running late.'")
    print("Stop with Ctrl+C.\n")

    try:
        whisper = cactus_init(validate_whisper_weights(WHISPER_WEIGHTS))
    except Exception as e:
        print(f"\n❌ {e}\n")
        return

    ring = AudioRingBuffer(max_seconds=max(WINDOW_SECONDS, 20), sr=AUDIO_SR)

    # Wake-word armed state
    armed_until = 0.0

    # Only allow one transcription at a time; skip if still running
    executor = ThreadPoolExecutor(max_workers=1)
    pending = None

    # Track last time we triggered a snapshot
    next_tick = time.time()

    def callback(indata, frames, time_info, status):
        # indata: (frames, channels)
        if status:
            # Drop status noise to console
            pass
        ring.push(indata.copy())

    try:
        with sd.InputStream(samplerate=AUDIO_SR, channels=1, dtype="float32", callback=callback):
            while True:
                now = time.time()
                if now < next_tick:
                    time.sleep(0.02)
                    continue
                next_tick = now + STEP_SECONDS

                # If a transcription job is still running, skip this tick to avoid backlog
                if pending is not None and not pending.done():
                    continue

                audio = ring.snapshot_last(WINDOW_SECONDS)
                if audio.shape[0] < int(0.5 * AUDIO_SR):
                    continue  # not enough audio yet

                # Submit transcription
                pending = executor.submit(transcribe_audio_array, whisper, audio, AUDIO_SR)

                # If completed quickly, handle immediately; else we will handle on next tick
                if pending.done():
                    text = pending.result()
                else:
                    # Handle results next loop iteration (non-blocking)
                    continue

                if not text or is_non_speech(text):
                    continue

                # Print only useful chunks (wake word or whatsapp)
                if detect_wake_word(text) or "whatsapp" in text.lower():
                    print(f"Heard: {text}")

                heard_wake = detect_wake_word(text)
                if heard_wake:
                    armed_until = time.time() + ARMED_SECONDS

                if not heard_wake and time.time() > armed_until:
                    continue

                cleaned = strip_wake_word_prefix(text)
                if not looks_like_whatsapp_command(cleaned):
                    continue

                parsed = extract_with_hybrid_llm(cleaned) or fallback_extract(cleaned)
                if not parsed:
                    print("Could not extract recipient/message yet. Try rephrasing.")
                    continue

                recipient = parsed["recipient"]
                message = parsed["message"]
                print(f"→ Sending WhatsApp to '{recipient}': {message}")

                try:
                    whatsapp_send_via_web(recipient, message)
                    print("✅ Sent.\n")
                except Exception as e:
                    print(f"❌ Failed to send via WhatsApp Web: {e}\n")

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        cactus_destroy(whisper)


if __name__ == "__main__":
    main()
