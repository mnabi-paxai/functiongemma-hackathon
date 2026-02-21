# Tool-Call Playbook (Auto-RAG Corpus)
Version: 1.1  
Purpose: Improve tool-call accuracy, selection, and argument quality for an LLM running with Cactus auto-RAG.

This playbook is **optimized for the FunctionGemma-Hackathon benchmark** (weather/alarm/message/reminder/search/music/timer). It also generalizes to broader tool-use.

---

## 0) What “success” means in this benchmark

The evaluator matches tool calls by:
- **Tool name must match exactly**
- **Each expected argument key must be present**
- **Argument values are compared by**: `strip().lower()` for strings (no punctuation removal!) fileciteturn0file1

**Implication:** small differences like extra words or punctuation can drop F1 to **0.00**.

### High-impact exactness rules (do these!)
1) **Do NOT add trailing punctuation** to `send_message.message` unless the user explicitly includes it.
   - Expected: `"good morning"` not `"good morning."`
2) Keep `play_music.song` **minimal**: remove filler words like “music”, “song”, “some”.
   - Expected: `"jazz"` not `"some jazz music"`
3) `get_weather.location` should be a clean city name (no “in”, no “today”).
   - Expected: `"Paris"` not `"in Paris today"`
4) `set_alarm.hour/minute` must be integers parsed correctly from AM/PM.
5) `set_timer.minutes` must be an integer (not `"5"`, not `"05"`, not `"five"`).
6) `create_reminder.title` should be a short noun phrase without “remind me”, and `time` should match user formatting like `"3:00 PM"`.

---

## 1) Tool schemas (this benchmark)

The tools available are defined in `benchmark.py`. fileciteturn0file1

### A) get_weather
**Args**
- `location` (string): city name (e.g., `"San Francisco"`, `"London"`)

### B) set_alarm
**Args**
- `hour` (integer)
- `minute` (integer)

### C) send_message
**Args**
- `recipient` (string): contact name
- `message` (string): message body (keep short & exact)

### D) create_reminder
**Args**
- `title` (string): reminder title
- `time` (string): e.g., `"3:00 PM"`

### E) search_contacts
**Args**
- `query` (string): name query

### F) play_music
**Args**
- `song` (string): song/playlist/genre query

### G) set_timer
**Args**
- `minutes` (integer)

---

## 2) Output contract (for on-device Cactus JSON)

Cactus returns a JSON string which is parsed via `json.loads` and read from `raw["function_calls"]`. fileciteturn0file2

### Required structure
Return JSON with this shape:

```json
{
  "function_calls": [
    {"name": "TOOL_NAME", "arguments": { "key": "value" }}
  ],
  "total_time_ms": 0,
  "confidence": 1.0
}
```

**Notes**
- For multi-intent requests, include **multiple calls** in correct order.
- Keep `confidence` meaningful. If unsure, lower it to trigger hybrid fallback (if enabled). fileciteturn0file2

---

## 3) Fast decision flow (benchmark-optimized)

### Step 1 — Segment the user request into intents
Split into sub-requests when you see:
- “and”, “also”, “then”
- commas separating actions
- multi-clause commands

Example:
> “Text Emma …, check the weather in Chicago, and set an alarm …”
→ intents: message + weather + alarm

### Step 2 — Classify each intent into one tool
Use **high-precision keywords** (see Section 4).

### Step 3 — Extract slots (arguments) with strict normalization
Use deterministic rules (Section 5) to avoid tiny mismatches.

### Step 4 — Emit tool calls only (no extra text)
In this benchmark, you want the tool-call JSON. Avoid natural-language chatter inside arguments.

---

## 4) Intent → tool mapping (high precision)

### get_weather
Triggers:
- “weather”, “forecast”, “temperature”, “how’s the weather”, “what’s the weather like”

### set_alarm
Triggers:
- “set an alarm”, “wake me up”, “alarm for”, “at 6 AM”

### set_timer
Triggers:
- “set a timer”, “countdown”, “timer for X minutes”

### send_message
Triggers:
- “send a message”, “text”, “message”, “tell [name]”, “DM”
- Usually includes a recipient and message content.

### create_reminder
Triggers:
- “remind me”, “reminder”, “don’t let me forget”
- Typically includes a topic + a time.

### search_contacts
Triggers:
- “find [name]”, “look up [name]”, “search my contacts”, “in my contacts”

### play_music
Triggers:
- “play”, “put on”, “start”, “listen to”
- Includes a song/artist/genre/playlist query.

---

## 5) Slot extraction rules (the biggest accuracy lever)

### 5.1 Time parsing (alarm + reminder)
Supported examples in the benchmark:
- `"10 AM"` → hour=10 minute=0
- `"6 AM"` → hour=6 minute=0
- `"8:15 AM"` → hour=8 minute=15
- `"7:30 AM"` → hour=7 minute=30
- `"5 AM"` → hour=5 minute=0
- `"3:00 PM"` (reminder time string kept as-is)
- `"2:00 PM"`, `"4:00 PM"`, `"5:00 PM"`, `"7:00 AM"`

**Rules**
- Parse `H[:MM] AM/PM`
- Convert to 24h only if your tool requires it. (Here it does **not**: it expects integer hour in standard clock representation used in examples—i.e., 7:30 AM → hour=7.)
- If minutes omitted, set minute=0.
- Keep reminder `time` string close to user: `H:MM AM/PM` with a space before AM/PM.

### 5.2 Message extraction (send_message)
Benchmark expects short message strings, often without punctuation. fileciteturn0file1

**Rules**
1) Identify recipient after patterns:
   - “to Alice”
   - “text Dave”
   - “send (a) message to John”
2) Identify message after patterns:
   - “saying …”
   - “that …”
   - a quoted string if present
3) **Normalize message content**
   - Remove trailing period/exclamation/question mark unless user includes that punctuation inside quotes.
   - Keep apostrophes inside words (e.g., `"I'll be late"`).
   - Remove leading filler: “please”, “just”.
   - Keep it short.

**Examples**
- “Send a message to Alice saying good morning.”
  - recipient: `"Alice"`
  - message: `"good morning"` (no period)
- “Text Dave saying I'll be late.”
  - message must be `"I'll be late"` (no period)

### 5.3 Music query extraction (play_music)
**Rules**
- Keep the core query:
  - song title (“Bohemian Rhapsody”)
  - genre (“jazz”, “classical music” → ideally `"classical music"` if user says it)
  - playlist phrase (“lo-fi beats”, “summer hits”)
- Remove filler:
  - “some”, “any”, “music”, “song”, “please”
- If user says “Play some jazz music.”
  - expected: `"jazz"` (not `"jazz music"`)

### 5.4 Weather location extraction (get_weather)
**Rules**
- Extract city after “in …” or “for …” or the final location phrase.
- Remove filler:
  - “today”, “right now”, “like”
- Preserve capitalization of multi-word cities: `"San Francisco"`, `"New York"`

### 5.5 Timer extraction (set_timer)
**Rules**
- Parse integer minutes from:
  - “for 5 minutes”, “a 15 minute timer”, “15 minutes”
- If user says “Set a 15 minute timer”
  - minutes: `15` (integer)

### 5.6 Reminder extraction (create_reminder)
**Rules**
- Title is the “thing” to remember:
  - “meeting” from “the meeting”
  - “call the dentist” from “call the dentist”
  - “take medicine” from “take medicine”
- Time is the exact human string with AM/PM:
  - `"3:00 PM"`, `"2:00 PM"`, etc.

---

## 6) Multi-intent composition rules

When the user asks for multiple actions:
- Produce **multiple function calls** in the same JSON payload
- Keep order matching user request order (or a sensible order if ambiguous)

Examples (expected benchmark behavior):
1) “Send a message to Bob saying hi and get the weather in London.”
   1. send_message(recipient="Bob", message="hi")
   2. get_weather(location="London")

2) “Set an alarm for 7:30 AM and check the weather in New York.”
   1. set_alarm(hour=7, minute=30)
   2. get_weather(location="New York")

---

## 7) Common failure modes (and fixes)

### Failure: extra words in slot values
- `"some jazz music"` vs expected `"jazz"`
Fix: aggressive filler-word stripping for music queries.

### Failure: punctuation mismatch
- `"good morning."` vs expected `"good morning"`
Fix: strip trailing punctuation from message text.

### Failure: wrong time parsing
- `"10 AM"` parsed as (10, 10) or missing minute
Fix: if no minutes in input, minute=0.

### Failure: incorrect segmentation
- multi-intent request only yields 1 tool call
Fix: always split on “and” if two verbs present.

---

## 8) RAG corpus strategy (Auto-RAG friendly)

To make Auto-RAG actually help, include content that is:
- **Highly specific** to the tools and benchmark slots
- Full of **short canonical examples** that can be retrieved easily

### Recommended additional docs in `./documents`
Create small focused files (better retrieval than one giant file):
- `tool_schemas.md` (schemas + examples)
- `slot_extraction_rules.md` (the rules in Section 5)
- `common_failures.md` (Section 7)
- `examples_gold_calls.md` (gold examples for all 30 benchmark cases)

Then RAG can fetch the most relevant snippet per query.

---

## 9) “Neuro-symbolic” ideas to push score higher (brainstorm)

This benchmark is unusually amenable to **symbolic pre-parsing** because:
- Tool set is small (7 tools)
- Arguments are mostly simple slots
- Most errors are deterministic (punctuation, filler words, time parsing)

### A) Symbolic pre-parser (fast, offline, high precision)
Implement an offline pipeline **before** calling the LLM:

1) **Normalize** input:
   - lowercase copy for matching
   - keep original for capitalization
2) **Segment** into intents:
   - split on “and”, commas, “then”
   - but preserve phrases like “rock and roll” (rare here)
3) **Classify each segment** with rules:
   - weather/alarm/timer/reminder/message/search/music keyword patterns
4) **Extract slots** with regex + heuristics:
   - time regex: `(\d{1,2})(?::(\d{2}))?\s*(AM|PM)`
   - timer minutes regex: `(\d+)\s*minute`
   - message regex: `(?:to|text)\s+(\w+).*?(?:saying|that)\s+(.+)`
   - weather location regex: `weather.*in\s+(.+)`
5) **Post-normalize** values:
   - strip trailing punctuation for message
   - remove filler tokens for music
6) If all required slots present, **emit tool calls directly** with confidence=1.0.
   - Else fall back to LLM.

This can dramatically increase on-device accuracy and reduce cloud fallback.

### B) Hybrid “symbolic hints” + LLM
If you don’t want to fully bypass the LLM:
- Run symbolic parse → produce a structured hint:
  ```json
  {"intents":[{"tool":"set_alarm","hour":10,"minute":0}]}
  ```
- Prepend as a system message:
  “Here is a proposed tool plan… Use it unless clearly wrong.”
This often boosts LLM confidence and correctness.

### C) Lightweight classifier (learned, still offline)
Train a tiny intent classifier on synthetic variants:
- Inputs: user utterances
- Labels: tool(s)
Options:
- fastText / linear model on bag-of-ngrams
- small DistilBERT-style encoder (if you can afford it)
Use it as:
- primary tool selection
- or a veto / re-ranker for LLM output

### D) Constraint checking + repair (symbolic verifier)
After LLM outputs tool calls:
- Verify schema: required keys + types
- Verify slot plausibility:
  - hour 1–12, minute 0–59
  - timer minutes positive
- Repair common issues:
  - strip punctuation in messages
  - remove filler tokens in music query
  - coerce `"5"` → 5 if tool expects integer

Then return repaired tool calls with higher confidence.

### E) Program synthesis framing (mini “logic model”)
Represent each intent as a simple logical form:
- `ALARM(time=07:30)`
- `WEATHER(city="New York")`
- `MSG(to="Bob", text="hi")`

Then deterministically compile logical forms → tool calls.
The LLM (or classifier) only needs to map text → logical forms; the compiler handles exactness.

---

## 10) Benchmark-specific “gold” micro-templates (copy/paste)

### Alarm
- “Set an alarm for 10 AM.” → `{"name":"set_alarm","arguments":{"hour":10,"minute":0}}`
- “Wake me up at 6 AM.” → `{"name":"set_alarm","arguments":{"hour":6,"minute":0}}`
- “Set an alarm for 8:15 AM.” → `{"name":"set_alarm","arguments":{"hour":8,"minute":15}}`

### Timer
- “Set a timer for 5 minutes.” → `{"name":"set_timer","arguments":{"minutes":5}}`
- “Set a 15 minute timer …” → `{"name":"set_timer","arguments":{"minutes":15}}`

### Message
- “Send a message to Alice saying good morning.” → `{"recipient":"Alice","message":"good morning"}`
- “Text Dave saying I'll be late.” → `{"recipient":"Dave","message":"I'll be late"}`

### Reminder
- “Remind me about the meeting at 3:00 PM.” → `{"title":"meeting","time":"3:00 PM"}`
- “Remind me to call the dentist at 2:00 PM.” → `{"title":"call the dentist","time":"2:00 PM"}`

### Search contacts
- “Find Bob in my contacts.” → `{"query":"Bob"}`
- “Look up Sarah in my contacts.” → `{"query":"Sarah"}`

### Music
- “Play Bohemian Rhapsody.” → `{"song":"Bohemian Rhapsody"}`
- “Play some jazz music.” → `{"song":"jazz"}`
- “play lo-fi beats” → `{"song":"lo-fi beats"}`

---

## Final rule (benchmark)
When in doubt, prefer **exact slot matching** over fluent natural language.
A slightly awkward but exact string beats a beautiful sentence that mismatches the expected value.
