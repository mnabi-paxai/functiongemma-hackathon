# Tool-Call Playbook (Auto-RAG Corpus)
Version: 2.0
Purpose: Teach robust, schema-driven tool-calling reasoning to a small on-device model.
This corpus is retrieved at inference time to improve tool selection, argument quality, and multi-call accuracy.

---

## 1. Detecting compound (multi-call) requests

A single user message may contain multiple independent actions. Each action maps to a separate tool call. Always produce one call per distinct action — do not merge them into one.

### Signals that multiple tool calls are needed

**Conjunctions connecting distinct actions:**
- "… and …" when each side contains its own verb and object
  - "Set an alarm for 7 AM **and** check the weather in Tokyo" → two calls
  - "Play jazz **and** set a timer for 30 minutes" → two calls
- Comma-separated imperative clauses:
  - "Send a message to Alice, set a timer for 10 minutes, and play some music" → three calls
- Sequential imperative verbs in one sentence:
  - "Find Bob in my contacts, then send him a message" → two calls

**How to distinguish "two actions" from "two arguments":**
- Two actions: each side has its own verb phrase targeting a different tool
  - "Set an alarm **and** play music" → verb 1 = set alarm, verb 2 = play music → two calls
- Two arguments (same call): the conjunction connects parts of the same argument
  - "Send a message to Alice **and** Bob" → one call per recipient, or one call if tool supports multi-recipient
  - "Play jazz **and** blues" → one play_music call with combined song value

**Rule:** If both sides of "and" require different tools, always emit separate calls. If both sides feed the same tool, emit one call.

---

## 2. Argument extraction from natural language

User messages rarely use schema field names directly. Extract arguments from natural phrasing.

### Time expressions → integer fields (hour, minute)

Users express time in many ways. Always convert to numeric fields:

| User says | hour | minute |
|-----------|------|--------|
| "10 AM" | 10 | 0 |
| "10:30 AM" | 10 | 30 |
| "2 PM" | 14 | 0 (if 24h) or 2 (if 12h schema) |
| "half past 3" | 3 | 30 |
| "quarter to 8" | 7 | 45 |
| "noon" | 12 | 0 |
| "midnight" | 0 | 0 |
| "6 in the morning" | 6 | 0 |
| "wake me up at 7" | 7 | 0 (assume minute=0 when not stated) |

**When minute is not mentioned, default to 0.**
**Always output hour and minute as integers, not strings.**

### Duration expressions → integer fields (minutes, seconds, hours)

| User says | minutes |
|-----------|---------|
| "5 minutes" | 5 |
| "half an hour" | 30 |
| "an hour" | 60 |
| "90 seconds" | 1 (or 0 if field is minutes) |
| "a minute and a half" | 1 |

**Always output durations as integers matching the schema field type.**

### Names → string fields (recipient, contact, person)

Extract the proper noun immediately following "to", "for", "with", or at the start of a possessive:
- "Send a message **to Alice**" → recipient = "Alice"
- "Find **Bob** in my contacts" → query = "Bob"
- "Text **Dave** saying…" → recipient = "Dave"

Strip honorifics and whitespace. Preserve capitalization.

### Locations → string fields (location, city, place)

Extract the place name following "in", "at", "for", "near":
- "weather **in San Francisco**" → location = "San Francisco"
- "forecast **for London**" → location = "London"
- "How's the weather **in New York**?" → location = "New York"

Use the name as stated by the user. Do not expand abbreviations or add country names unless the user included them.

### Message content → string fields (message, content, body, text)

Extract the content after "saying", "that says", "with the message", or after quoted speech:
- "Send a message to Bob **saying hi**" → message = "hi"
- "Text Lisa **'see you tonight'**" → message = "see you tonight"
- "Tell him **I'll be late**" → message = "I'll be late"

Strip surrounding quotes. Do not paraphrase or expand the user's words.

### Reminder and task titles → string fields (title, task, note)

Extract the subject of the reminder, stripped of filler words:
- "Remind me **about the meeting**" → title = "meeting" (strip "about the")
- "Remind me **to call the dentist**" → title = "call the dentist"
- "Don't let me forget **my medication**" → title = "medication"

Strip leading articles ("the", "a", "an") and prepositions ("about", "to") when they are filler, not part of the semantic content.

### Song, playlist, or media → string fields (song, track, query)

Extract the song or genre name directly:
- "Play **Bohemian Rhapsody**" → song = "Bohemian Rhapsody"
- "Put on **some jazz**" → song = "jazz"
- "Play **lo-fi beats**" → song = "lo-fi beats"

Preserve the user's phrasing. Do not normalize to a canonical title.

---

## 3. Schema compliance rules

Every tool call must conform to its schema. Violations cause silent failures.

### Required fields

Every field listed under `required` in the schema must be present. If you cannot infer a value from the user message:
- For numeric fields with an obvious default (e.g., minute when not mentioned): use 0
- For string fields: use the closest extractable value
- Never omit a required field and never use null for a required field unless the schema allows it

### Type rules

| Schema type | Valid values | Invalid (must convert) |
|-------------|-------------|------------------------|
| integer | 10, 0, -1 | "10", "ten", 10.5 |
| number | 1.5, 10, 0 | "1.5", "one point five" |
| string | "London", "hi" | 123, true |
| boolean | true, false | "true", "yes", 1 |
| array | ["a", "b"] | "a,b", "a" (single string) |

**Always output the type the schema specifies, not the type the user's words suggest.**

### Unknown fields

Do not include fields that are not in the schema's `properties`. Extraneous fields can cause tool call rejection.

---

## 4. Tool disambiguation

When multiple tools could apply to a user request, use these heuristics:

### Time-based tools: alarm vs. timer vs. reminder

| Tool | When to use | Key signal |
|------|-------------|------------|
| set_alarm | User wants to wake up or be alerted at a specific clock time | "wake me up", "alarm for 7 AM", specific time of day |
| set_timer | User wants a countdown from now | "timer for 5 minutes", "count down", "in 20 minutes" |
| create_reminder | User wants a notification with associated content at a time | "remind me to…", "don't forget to…", content + time |

**Alarm vs. timer:** alarm = absolute clock time; timer = relative duration.
**Timer vs. reminder:** timer is pure countdown; reminder has content (a task, note, or item to remember).

### Communication tools: message vs. search

| Tool | When to use |
|------|-------------|
| send_message | User wants to send content to a person |
| search_contacts | User wants to find/look up a person without sending anything |

"Find Bob and send him a message" → use both: search_contacts first, then send_message.

### General disambiguation principle

Choose the tool whose description most precisely matches the user's stated goal. When uncertain, prefer the tool that is more specific (narrower scope) over a general-purpose one.

---

## 5. Validation checklist before emitting a tool call

Before finalizing each tool call, verify:

1. **Tool name exists** in the provided tools list — never invent tool names
2. **All required fields present** — check the schema's `required` array
3. **Types are correct** — integers are integers, strings are strings
4. **No extra fields** — only include fields defined in `properties`
5. **Values are non-empty** — required string fields must not be empty strings
6. **Numeric values are plausible** — hours 0-23, minutes 0-59, durations > 0
7. **One call per distinct action** — do not merge two actions into one call

---

## 6. Common failure modes

### Missing minute field on alarm calls
User says "Set an alarm for 10 AM" → include both hour=10 AND minute=0.
Never emit only one of the two required fields.

### String instead of integer
User says "Set a timer for 5 minutes" → minutes must be the integer 5, not the string "5".

### Merging two actions into one call
"Set an alarm and play music" → two separate calls, not one call with both arguments.

### Over-extracting message content
"Send Bob a message saying good morning" → message = "good morning", not "saying good morning".

### Adding filler to location
User says "weather in Paris" → location = "Paris", not "Paris, France" or "Paris, Île-de-France, France".

### Using the wrong time-based tool
"Set a 10 minute timer" → set_timer(minutes=10), not set_alarm(hour=10, minute=0).

### Calling a tool that is not in the provided list
Only call tools that appear in the tools list given to you. Never hallucinate tool names.

---

## 7. Multi-call generation pattern

When you detect a compound request:

1. Identify each distinct action (each imperative verb phrase with its own arguments)
2. For each action, identify the best matching tool
3. Extract arguments for each call independently
4. Validate each call against its schema
5. Emit all calls in the order they appear in the user message

**Example:**
User: "Text Emma saying good night, check the weather in Chicago, and set an alarm for 5 AM"

Actions:
1. "Text Emma saying good night" → send_message(recipient="Emma", message="good night")
2. "check the weather in Chicago" → get_weather(location="Chicago")
3. "set an alarm for 5 AM" → set_alarm(hour=5, minute=0)

Result: three tool calls, in order.

---

## 8. Confidence and uncertainty

If you are uncertain which tool to call or what value to use for an argument:
- Prefer the interpretation that produces a schema-valid call over one that does not
- Prefer the tool whose description most closely matches the user's words
- When a value is genuinely ambiguous, use the most common default (e.g., minute=0 for whole-hour alarms)
- Never emit a call with a hallucinated tool name or fabricated argument values
