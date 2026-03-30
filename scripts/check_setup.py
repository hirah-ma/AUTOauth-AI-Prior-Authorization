"""
check_setup.py
==============
Run this FIRST before anything else.
Verifies your API key, checks available models, and tests embedding + generation.

Usage:
    python check_setup.py
"""

import os

import load_gemini_env

load_gemini_env.load()

import google.generativeai as genai

GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
genai.configure(api_key=GEMINI_API_KEY)

EMBED_MODEL    = "models/gemini-embedding-2-preview" 
# 'gemini-3-flash-preview' is the high-efficiency decision model
DECISION_MODEL = "models/gemini-3-flash-preview"

# Get API key from environment
#GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# CRITICAL FIX: In the Google SDK, genai.configure() sets a global config.
# It does not return a client object. We call methods directly from genai.
#genai.configure(api_key=GEMINI_API_KEY)

print("=" * 55)
print("   Prior Authorization - Setup Checker")
print("=" * 55)

# 1. API Key check
print("\n[1/3] Checking API key...")
if not GEMINI_API_KEY:
    print("  [FAIL] API key not set!")
    print("     Create FILES/.env with GEMINI_API_KEY=... (copy from .env.example)")
    print("     OR run:  set GEMINI_API_KEY=your_key   (Windows)")
    print("              export GEMINI_API_KEY=your_key (Mac/Linux)")
    exit(1)
else:
    # Masking the key for security while showing enough to confirm it's loaded
    print(f"  [OK] API key found: {GEMINI_API_KEY[:8]}...")

# 2. Embedding test
print(f"\n[2/3] Testing embedding model: {EMBED_MODEL}")
embed_ok = False
try:
    # Use genai.embed_content for the modern SDK
    result = genai.embed_content(
        model=EMBED_MODEL,
        content="test patient with diabetes",
        task_type="retrieval_query"
    )
    # The result contains a list of embeddings
    dim = len(result['embedding'])
    print(f"  [OK] Embedding works! Vector dimension: {dim}")
    embed_ok = True
except Exception as e:
    print(f"  [FAIL] Embedding failed: {e}")
    print("  -> Check if 'gemini-embedding-2-preview' is enabled in your region.")

# 3. Generation test
print(f"\n[3/3] Testing generation model: {DECISION_MODEL}")
gen_ok = False
try:
    # Use GenerativeModel class for chat/text generation
    model = genai.GenerativeModel(DECISION_MODEL)
    resp = model.generate_content("Reply with just the word: OK")
    
    # Access response text safely
    response_text = resp.text.strip()
    print(f"  [OK] Generation works! Response: {response_text}")
    gen_ok = True
except Exception as e:
    print(f"  [FAIL] Generation failed: {e}")
    print("  -> Try: 'models/gemini-2.5-flash' as a legacy fallback")

print("\n" + "=" * 55)
if embed_ok and gen_ok:
    print("  [OK] All checks passed - you are ready to run:")
    print("     python ingest_policies.py")
    print("=" * 55)
else:
    print("  [FAIL] Fix the errors above before ingesting policies.")
    print("=" * 55)
    exit(1)