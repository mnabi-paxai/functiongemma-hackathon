import os

def main():
    # Set your key as an env var first:
    #   export GEMINI_API_KEY="YOUR_KEY"
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Missing env var GEMINI_API_KEY")

    try:
        from google import genai
    except ImportError:
        raise SystemExit(
            "Missing dependency. Install with:\n"
            "  pip install -U google-genai"
        )

    client = genai.Client(api_key=api_key)

    try:
        resp = client.models.generate_content(
            model="gemini-1.5-flash",
            contents="Say 'API key works' in 3 words."
        )
        print("✅ SUCCESS: API key works.")
        print("Response:", resp.text)
    except Exception as e:
        print("❌ FAILED: API key does NOT work (or is restricted).")
        print("Error:", repr(e))

if __name__ == "__main__":
    main()