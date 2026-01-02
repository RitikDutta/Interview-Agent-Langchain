from __future__ import annotations

import os
import sys
from dotenv import load_dotenv

# Load env early so modules can rely on it.
load_dotenv()

# --- 🔍 DEBUG SECTION ---
test_val = os.getenv("TEST_SECRET")

print("-" * 30)
if test_val == "test_done":
    print("✅ SUCCESS: .env file is being read correctly!")
    print(f"   Value found: {test_val}")
else:
    print("❌ FAILURE: .env file is NOT being read.")
    print(f"   Value found: {test_val} (None means file wasn't found)")
print("-" * 30)
# ------------------------

# --- API KEYS ---
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- MODEL CONFIG ---
GENAI_MODEL = os.getenv("GENAI_MODEL") or os.getenv("GOOGLE_GENAI_MODEL") or "gemini-flash-lite-latest"
GENAI_MODEL_LOW_TEMP = os.getenv("GENAI_MODEL_LOW_TEMP") or GENAI_MODEL
QUESTIONS_NAMESPACE = os.getenv("QUESTIONS_NAMESPACE") or "questions_v4"