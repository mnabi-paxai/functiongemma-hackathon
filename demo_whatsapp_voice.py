#!/usr/bin/env python3
"""
Voice-to-WhatsApp Demo  (on-device, low-latency)
=================================================
Run:
    ./cactus/venv/bin/python3 demo_whatsapp_voice.py

How it works:
    1. Laptop waits silently, listening for your voice.
    2. When it detects speech it starts recording.
    3. When you stop talking it automatically captures and processes.
    4. On-device Whisper (cactus_transcribe) converts speech to text.
    5. On-device FunctionGemma (generate_hybrid) extracts recipient + message.
    6. Playwright opens WhatsApp Web, finds the contact and sends the message.

Example command to say:
    "Send a WhatsApp message to Parsa saying hey, what's up?"

Stop: Ctrl+C
"""

import os
import sys
import json
import time
import tempfile
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

sys.path.insert(0, "cactus/python/src")
os.environ.setdefault("CACTUS_NO_CLOUD_TELE", "1")

from cactus import cactus_init, cactus_transcribe, cactus_destroy
from main import generate_hybrid, generate_cloud
from playwright.sync_api import sync_playwright


# ── Config ────────────────────────────────────────────────────────────────────

WHISPER_WEIGHTS  = os.environ.get("WHISPER_WEIGHTS", "weights/whisper-small")
AUDIO_SR         = 16000
WHISPER_PROMPT   = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"

# Voice activity detection
VAD_CHUNK_MS       = 100    # analyse energy every 100 ms
VAD_SPEECH_THRESH  = 0.015  # RMS threshold to consider "speech"
VAD_SILENCE_SEC    = 1.2    # seconds of silence before stopping
VAD_MAX_SEC        = 10.0   # hard cap per recording
VAD_WAIT_SEC       = 8.0    # seconds to wait for speech before demo fallback

# Demo fallback — used when no voice is detected (processed via gemini-2.5-flash)
FALLBACK_COMMAND   = "Send a WhatsApp to Parsa saying Hi"

PROFILE_DIR      = Path(os.environ.get("WHATSAPP_PROFILE_DIR", ".whatsapp_profile")).resolve()
WHATSAPP_WEB_URL = "https://web.whatsapp.com/"

TOOL_WHATSAPP_SEND = {
    "name": "whatsapp_send",
    "description": (
        "Send a WhatsApp message to a contact. "
        "Call this whenever the user asks to send or message someone on WhatsApp."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "recipient": {"type": "string", "description": "The contact name as spoken by the user"},
            "message":   {"type": "string", "description": "The message text to send"},
        },
        "required": ["recipient", "message"],
    },
}

SYSTEM_PROMPT = (
    "You are a voice assistant. "
    "When the user asks to send a WhatsApp message, call whatsapp_send "
    "with the recipient name and the exact message text. "
    "Output ONLY the tool call, no extra text."
)


# ── Voice activity detection + recording ─────────────────────────────────────

def record_on_voice(sr: int) -> np.ndarray | None:
    """
    Block until speech is detected, then record until silence.
    Returns a float32 mono array ready for Whisper,
    or None if no speech is detected within VAD_WAIT_SEC (triggers demo fallback).
    """
    chunk_frames   = int(sr * VAD_CHUNK_MS / 1000)
    silence_chunks = int(VAD_SILENCE_SEC * 1000 / VAD_CHUNK_MS)
    max_chunks     = int(VAD_MAX_SEC * 1000 / VAD_CHUNK_MS)
    wait_chunks    = int(VAD_WAIT_SEC * 1000 / VAD_CHUNK_MS)

    print("Listening...", end="", flush=True)

    frames_captured = []
    speech_started  = False
    silent_count    = 0
    waited_chunks   = 0

    with sd.InputStream(samplerate=sr, channels=1, dtype="float32") as stream:
        while True:
            chunk, _ = stream.read(chunk_frames)
            mono      = chunk.reshape(-1)
            rms       = float(np.sqrt(np.mean(mono ** 2)))

            if not speech_started:
                waited_chunks += 1
                if rms >= VAD_SPEECH_THRESH:
                    speech_started = True
                    print(" Recording...", end="", flush=True)
                    frames_captured.append(mono.copy())
                    silent_count = 0
                elif waited_chunks >= wait_chunks:
                    print(" (no voice detected — using demo fallback)")
                    return None          # triggers Gemini fallback
            else:
                frames_captured.append(mono.copy())
                if rms < VAD_SPEECH_THRESH:
                    silent_count += 1
                    if silent_count >= silence_chunks:
                        break
                else:
                    silent_count = 0
                if len(frames_captured) >= max_chunks:
                    break

    print()
    if not frames_captured:
        return None
    return np.concatenate(frames_captured)


# ── Transcription ─────────────────────────────────────────────────────────────

def transcribe(whisper, audio: np.ndarray) -> tuple[str, float]:
    """Run cactus_transcribe on a float32 array. Returns (text, elapsed_ms)."""
    fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="cactus_")
    os.close(fd)
    try:
        sf.write(wav_path, audio, AUDIO_SR)
        t0  = time.perf_counter()
        raw = cactus_transcribe(whisper, wav_path, prompt=WHISPER_PROMPT)
        ms  = (time.perf_counter() - t0) * 1000
        try:
            text = (json.loads(raw).get("response") or "").strip()
        except json.JSONDecodeError:
            text = ""
        return text, ms
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass


# ── Intent extraction ─────────────────────────────────────────────────────────

def extract_intent_cloud(text: str) -> tuple[dict | None, float]:
    """
    Fallback: use gemini-2.5-flash to extract recipient + message.
    Returns (intent_dict_or_None, elapsed_ms).
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": text},
    ]
    t0     = time.perf_counter()
    result = generate_cloud(messages, [TOOL_WHATSAPP_SEND])
    ms     = (time.perf_counter() - t0) * 1000

    for call in (result.get("function_calls") or []):
        if call.get("name") == "whatsapp_send":
            args      = call.get("arguments", {})
            recipient = (args.get("recipient") or "").strip()
            message   = (args.get("message")   or "").strip()
            if recipient and message:
                return {"recipient": recipient, "message": message}, ms

    return None, ms


def extract_intent(text: str) -> tuple[dict | None, float, str]:
    """
    Use generate_hybrid (on-device FunctionGemma) to parse recipient + message.
    Returns (intent_dict_or_None, elapsed_ms, source).
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": text},
    ]
    t0     = time.perf_counter()
    result = generate_hybrid(messages, [TOOL_WHATSAPP_SEND])
    ms     = (time.perf_counter() - t0) * 1000
    source = result.get("source", "unknown")

    for call in (result.get("function_calls") or []):
        if call.get("name") == "whatsapp_send":
            args      = call.get("arguments", {})
            recipient = (args.get("recipient") or "").strip()
            message   = (args.get("message")   or "").strip()
            if recipient and message:
                return {"recipient": recipient, "message": message}, ms, source

    return None, ms, source


# ── WhatsApp Web automation ───────────────────────────────────────────────────

def whatsapp_send(recipient: str, message: str):
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR), headless=False
        )
        page = browser.pages[0] if browser.pages else browser.new_page()
        page.goto(WHATSAPP_WEB_URL, wait_until="domcontentloaded")
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

        # Type and send
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
    raise RuntimeError("Timed out waiting for WhatsApp Web login. Scan the QR code.")


def _fill_first(page, selectors: list, text: str, timeout_ms: int = 8000):
    for sel in selectors:
        try:
            page.wait_for_selector(sel, timeout=timeout_ms)
            page.click(sel)
            page.fill(sel, text)
            return
        except Exception:
            continue
    raise RuntimeError("Could not find input field on WhatsApp Web.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n========================================")
    print("  Voice-to-WhatsApp  (Powered by Cactus)")
    print("========================================")
    print(f"Loading Whisper weights: {WHISPER_WEIGHTS}")
    whisper = cactus_init(WHISPER_WEIGHTS)
    print("Ready.\n")
    print('Speak your command, e.g.:')
    print('  "Send a WhatsApp message to Parsa saying hey, what\'s up?"\n')
    print("Press Ctrl+C to quit.\n")

    try:
        while True:
            # ── Step 1: Record ──────────────────────────────────────
            audio = record_on_voice(AUDIO_SR)

            # No voice detected → demo fallback via gemini-2.5-flash
            if audio is None:
                print(f"  Fallback command: \"{FALLBACK_COMMAND}\"")
                intent, t_intent = extract_intent_cloud(FALLBACK_COMMAND)
                source = "gemini-2.5-flash (fallback)"
                print(f"  Intent ({t_intent:.0f}ms, {source}): ", end="")
                if not intent:
                    print("failed to parse fallback.\n")
                    continue
                recipient = intent["recipient"]
                message   = intent["message"]
                print(f"recipient=\"{recipient}\"  message=\"{message}\"")
                print(f"\n  Sending WhatsApp to '{recipient}'...")
                t0 = time.perf_counter()
                try:
                    whatsapp_send(recipient, message)
                    print(f"  Sent! ({(time.perf_counter()-t0)*1000:.0f}ms)\n")
                except Exception as e:
                    print(f"  ERROR: {e}\n")
                continue

            duration_s = audio.shape[0] / AUDIO_SR
            if duration_s < 0.5:
                print("(too short, ignored)\n")
                continue

            # ── Step 2: Transcribe (on-device Whisper) ──────────────
            text, t_transcribe = transcribe(whisper, audio)
            print(f"  Transcribed ({t_transcribe:.0f}ms): \"{text}\"")

            if not text:
                print("  Nothing recognised — try again.\n")
                continue

            # ── Step 3: Extract intent (on-device FunctionGemma) ────
            intent, t_intent, source = extract_intent(text)
            print(f"  Intent ({t_intent:.0f}ms, {source}): ", end="")

            if not intent:
                print("no WhatsApp command detected.")
                print('  Try: "Send a WhatsApp to [name] saying [message]"\n')
                continue

            recipient = intent["recipient"]
            message   = intent["message"]
            print(f"recipient=\"{recipient}\"  message=\"{message}\"")

            # ── Step 4: Send via WhatsApp Web ────────────────────────
            print(f"\n  Sending WhatsApp to '{recipient}'...")
            t0 = time.perf_counter()
            try:
                whatsapp_send(recipient, message)
                t_send = (time.perf_counter() - t0) * 1000
                print(f"  Sent! ({t_send:.0f}ms)\n")
            except Exception as e:
                print(f"  ERROR sending: {e}\n")

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cactus_destroy(whisper)


if __name__ == "__main__":
    main()
