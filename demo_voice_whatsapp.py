#!/usr/bin/env python3
"""
Low-latency Voice-to-WhatsApp demo (on-device)
===============================================
Pipeline:
  1. Press ENTER to start recording your voice command
  2. Press ENTER again to stop
  3. cactus_transcribe  → on-device Whisper speech-to-text
  4. generate_hybrid    → on-device FunctionGemma extracts (recipient, message)
  5. Playwright         → sends the message via WhatsApp Web

Leverages Cactus on-device inference for both transcription and intent extraction,
keeping the full pipeline local and low-latency.

Run:
  ./cactus/venv/bin/python3 demo_voice_whatsapp.py

Dependencies (already in cactus venv):
  pip install sounddevice soundfile playwright
  python -m playwright install chromium
"""

import os
import sys
import json
import time
import tempfile
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

sys.path.insert(0, "cactus/python/src")
os.environ.setdefault("CACTUS_NO_CLOUD_TELE", "1")

from cactus import cactus_init, cactus_transcribe, cactus_destroy
from main import generate_hybrid

from playwright.sync_api import sync_playwright


# ── Config ────────────────────────────────────────────────────────────────────

WHISPER_WEIGHTS  = os.environ.get("WHISPER_WEIGHTS", "weights/whisper-small")
AUDIO_SR         = int(os.environ.get("AUDIO_SR", "16000"))
PROFILE_DIR      = Path(os.environ.get("WHATSAPP_PROFILE_DIR", ".whatsapp_profile")).resolve()
WHATSAPP_WEB_URL = "https://web.whatsapp.com/"
WHISPER_PROMPT   = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"

# Tool schema for intent extraction
TOOL_WHATSAPP_SEND = {
    "name": "whatsapp_send",
    "description": (
        "Send a WhatsApp message to a contact. "
        "Call this when the user asks to send or message someone on WhatsApp."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "recipient": {
                "type": "string",
                "description": "The contact's name exactly as spoken by the user",
            },
            "message": {
                "type": "string",
                "description": "The text message to send",
            },
        },
        "required": ["recipient", "message"],
    },
}

SYSTEM_PROMPT = (
    "You are a voice assistant. "
    "When the user asks to send a WhatsApp message, call whatsapp_send "
    "with the recipient name and the message text. "
    "Output ONLY the tool call, no extra text."
)


# ── Audio recording (push-to-talk) ────────────────────────────────────────────

def record_until_enter(sr: int) -> np.ndarray:
    """Stream mic audio until the user presses ENTER. Returns float32 mono array."""
    chunks = []
    stop_event = threading.Event()

    def _callback(indata, frames, time_info, status):
        chunks.append(indata.copy())

    stream = sd.InputStream(samplerate=sr, channels=1, dtype="float32", callback=_callback)
    stream.start()

    # Wait for ENTER in a background thread so the mic keeps going
    def _wait():
        input()
        stop_event.set()

    t = threading.Thread(target=_wait, daemon=True)
    t.start()
    stop_event.wait()
    stream.stop()
    stream.close()

    if not chunks:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate([c.reshape(-1) for c in chunks]).astype(np.float32)


# ── Transcription ─────────────────────────────────────────────────────────────

def transcribe(whisper, audio: np.ndarray, sr: int) -> str:
    """Write audio to a temp WAV and run cactus_transcribe. Returns transcript string."""
    fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="cactus_")
    os.close(fd)
    try:
        sf.write(wav_path, audio, sr)
        t0 = time.perf_counter()
        raw = cactus_transcribe(whisper, wav_path, prompt=WHISPER_PROMPT)
        elapsed = (time.perf_counter() - t0) * 1000
        try:
            text = (json.loads(raw).get("response") or "").strip()
        except json.JSONDecodeError:
            text = ""
        print(f"  [transcribe]  {elapsed:.0f}ms  →  \"{text}\"")
        return text
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass


# ── Intent extraction ─────────────────────────────────────────────────────────

def extract_intent(text: str) -> dict | None:
    """
    Use generate_hybrid (on-device FunctionGemma) to extract recipient + message.
    Returns {"recipient": ..., "message": ..., "source": ...} or None.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": text},
    ]
    t0 = time.perf_counter()
    result = generate_hybrid(messages, [TOOL_WHATSAPP_SEND])
    elapsed = (time.perf_counter() - t0) * 1000
    source = result.get("source", "unknown")

    calls = result.get("function_calls") or []
    for call in calls:
        if call.get("name") == "whatsapp_send":
            args = call.get("arguments", {})
            recipient = (args.get("recipient") or "").strip()
            message   = (args.get("message")   or "").strip()
            if recipient and message:
                print(f"  [intent]      {elapsed:.0f}ms  →  recipient=\"{recipient}\"  message=\"{message}\"  ({source})")
                return {"recipient": recipient, "message": message, "source": source}

    print(f"  [intent]      {elapsed:.0f}ms  →  no valid tool call returned ({source})")
    return None


# ── WhatsApp Web automation ───────────────────────────────────────────────────

def whatsapp_send(recipient: str, message: str):
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR), headless=False
        )
        page = browser.pages[0] if browser.pages else browser.new_page()
        page.goto(WHATSAPP_WEB_URL, wait_until="domcontentloaded")

        # Wait for login (QR scan on first run; subsequent runs use saved session)
        _wait_for_login(page)

        # Search for the contact
        _fill_first(page, [
            'div[contenteditable="true"][data-tab="3"]',
            'div[contenteditable="true"][data-tab="2"]',
            'div[role="textbox"][contenteditable="true"]',
        ], recipient)
        time.sleep(1.0)

        # Open the chat
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

        # Type and send the message
        _fill_first(page, [
            'div[contenteditable="true"][data-tab="10"]',
            'div[contenteditable="true"][data-tab="9"]',
            'footer div[contenteditable="true"][role="textbox"]',
            'div[role="textbox"][contenteditable="true"]',
        ], message)
        page.keyboard.press("Enter")
        time.sleep(1.0)
        browser.close()


def _wait_for_login(page, timeout_s: int = 120):
    selectors = [
        'div[contenteditable="true"][data-tab="3"]',
        'div[contenteditable="true"][data-tab="2"]',
        'div[role="textbox"][contenteditable="true"]',
    ]
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        for sel in selectors:
            try:
                if page.query_selector(sel):
                    return
            except Exception:
                pass
        time.sleep(1)
    raise RuntimeError("Timed out waiting for WhatsApp Web login. Please scan the QR code.")


def _fill_first(page, selectors: list[str], text: str, timeout_ms: int = 8000):
    for sel in selectors:
        try:
            page.wait_for_selector(sel, timeout=timeout_ms)
            page.click(sel)
            page.fill(sel, text)
            return
        except Exception:
            continue
    raise RuntimeError(f"Could not find any selector to fill: {selectors}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n=== Voice-to-WhatsApp  (on-device / low-latency) ===\n")
    print(f"Loading Whisper from: {WHISPER_WEIGHTS}")
    whisper = cactus_init(WHISPER_WEIGHTS)
    print("Whisper ready.\n")

    try:
        while True:
            print("─" * 52)
            print("Press ENTER to start recording your command...")
            input()
            print("Recording... (press ENTER to stop)")

            t_record_start = time.perf_counter()
            audio = record_until_enter(AUDIO_SR)
            duration = (time.perf_counter() - t_record_start) * 1000
            print(f"  [record]      {duration:.0f}ms  ({audio.shape[0] / AUDIO_SR:.1f}s of audio)")

            if audio.shape[0] < int(0.3 * AUDIO_SR):
                print("  Too short — try again.\n")
                continue

            # Step 1: Transcribe (on-device Whisper)
            text = transcribe(whisper, audio, AUDIO_SR)
            if not text:
                print("  Nothing heard — try again.\n")
                continue

            # Step 2: Extract intent (on-device FunctionGemma)
            intent = extract_intent(text)
            if not intent:
                print("  Could not parse a WhatsApp command. Try:\n"
                      "  \"Send a WhatsApp message to Parsa saying hi, how are you?\"\n")
                continue

            # Step 3: Send via WhatsApp Web
            recipient = intent["recipient"]
            message   = intent["message"]
            print(f"\nSending WhatsApp message to '{recipient}': \"{message}\"")
            t0 = time.perf_counter()
            try:
                whatsapp_send(recipient, message)
                elapsed = (time.perf_counter() - t0) * 1000
                print(f"  [whatsapp]    {elapsed:.0f}ms  → Sent!\n")
            except Exception as e:
                print(f"  [whatsapp]    ERROR: {e}\n")

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cactus_destroy(whisper)


if __name__ == "__main__":
    main()
